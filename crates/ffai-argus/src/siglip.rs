//! Our own `SigLIP` vision encoder — same maths, fewer passes over memory.
//!
//! # Why reimplement something candle already has
//!
//! Not for the architecture. `candle_transformers::models::siglip` is correct,
//! and steps 3-6 were built on it. The reason is that half of a `SigLIP` layer
//! is spent in kernels candle runs on **one core**.
//!
//! `examples/vision_ops_probe` prices one encoder layer at the real shapes
//! (seq 1024, hidden 768, 12 heads, inter 3072):
//!
//! | op | x | ms/layer | share | rate |
//! |---|---:|---:|---:|---|
//! | GELU on `(1,1024,3072)` | 1 | 12.17 | **19.7 %** | 2.1 GB/s |
//! | `* scale` on `(1,12,1024,1024)` | 1 | 9.62 | **15.6 %** | 10.5 GB/s |
//! | matmul q·kᵀ | 1 | 7.15 | 11.5 % | 225 GF/s |
//! | linear up | 1 | 7.35 | 11.9 % | 657 GF/s |
//! | linear down | 1 | 6.66 | 10.8 % | 726 GF/s |
//! | softmax | 1 | 4.35 | 7.0 % | 23.1 GB/s |
//! | linear q/k/v | 3 | 4.73 | 7.6 % | 767 GF/s |
//! | reshape+transpose+contiguous | 3 | 1.72 | 2.8 % | 11.0 GB/s |
//! | *(total)* | | **61.9** | | |
//!
//! **50 % matmul, 50 % everything else.** The matmuls are not the problem —
//! 657-798 GF/s is parity with PyTorch on this box (§16). The other half runs
//! at 2-23 GB/s because candle's CPU backend uses rayon for `conv2d` and
//! nothing else.
//!
//! Two of those rows are pure waste rather than slow kernels, and this module
//! exists to delete them:
//!
//! * **`* scale` (15.6 %)** scales a `(1,12,1024,1024)` tensor — 12.6 M
//!   elements — when the same result comes from scaling **q**, which is
//!   786 K elements. Better still: scale the *weight* at load and pay nothing
//!   at all.
//! * **GELU (19.7 %)** is `f32::tanh` per element, 3.1 M scalar libm calls,
//!   single-threaded.
//!
//! # What is NOT changed
//!
//! Every matmul still goes through candle, and so do `Conv2d`, `Linear` and
//! `LayerNorm`. This is not a tensor library; it is the same graph with the
//! memory-bound steps rewritten.
//!
//! # How it is gated
//!
//! `tests/siglip_parity.rs` runs this tower and candle's over the same weights
//! and the same input and compares tensors. It is deliberately **not** a
//! bit-identity gate: folding the scale into the weights and replacing `tanh`
//! both change the last bits. The bar is the one §16 established as
//! meaningful — far below the 2.06e-4 that provably flips no token — and the
//! real gate remains what it has always been: 32/32 tokens and a caption
//! identical to the reference's.

use candle_core::{IndexOp, Module, Result as CandleResult, Tensor, D};
use candle_nn::{Conv2dConfig, LayerNorm, Linear, VarBuilder};
use candle_transformers::models::siglip::VisionConfig;
use crate::par::prelude::*;

/// Should the elementwise kernels use the thread pool?
///
/// **They must not when the caller is already threading tiles.** The engine
/// runs up to six towers concurrently (`engine::run_tower`), which fills the
/// machine on its own; a rayon fan-out *inside* each of those six then
/// oversubscribes, and measurement says the nesting costs more than it buys:
///
/// | | sequential tiles | 6 tile-workers |
/// |---|---:|---:|
/// | candle's tower | 24635 ms | 10807 ms |
/// | ours, kernels parallel | 17748 ms | 8810 ms |
///
/// A 1.39x win on one tile shrinks to 1.23x once six run at once, and the
/// difference is contention rather than anything about the kernels.
///
/// So the parallelism lives at exactly ONE level: whichever level has work to
/// spread. One tile (every video frame is one tile) parallelises inside;
/// seventeen tiles parallelise across.
static KERNELS_PARALLEL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Tell the kernels whether they own the machine. Returns the previous value.
pub fn set_kernels_parallel(on: bool) -> bool {
    KERNELS_PARALLEL.swap(on, std::sync::atomic::Ordering::Relaxed)
}

fn kernels_parallel() -> bool {
    KERNELS_PARALLEL.load(std::sync::atomic::Ordering::Relaxed)
}

/// Patch + position embeddings.
struct Embeddings {
    /// The patch conv as a flat `(patch*patch*C, hidden)` matrix — see
    /// [`Embeddings::new`]. The `Conv2d` it is derived from is NOT retained:
    /// it was briefly kept as a "fallback", which the compiler correctly
    /// reported as a never-read field. A fallback nothing can reach is not a
    /// fallback — the equality this rests on is gated by a test instead.
    w_flat: Tensor,
    bias: Option<Tensor>,
    channels: usize,
    position: Tensor,
    patch_size: usize,
}

impl Embeddings {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> CandleResult<Self> {
        let patch = candle_nn::conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            vb.pp("patch_embedding"),
        )?;
        let side = cfg.image_size / cfg.patch_size;
        let position =
            candle_nn::embedding(side * side, cfg.hidden_size, vb.pp("position_embedding"))?
                .embeddings()
                .clone();
        // The patch conv, ALSO kept as a flat `(patch*patch*C, hidden)` matrix.
        //
        // Stride equals the kernel, so this convolution is non-overlapping and
        // its im2col is a pure PERMUTATION — no element is duplicated, which is
        // what makes the matmul form exact in work as well as in result.
        // Measured (`examples/embed_probe`): candle's `conv2d` **10.925 ms**
        // at 113 GF/s against **4.146 ms** at 291 GF/s for the matmul, and the
        // matmul form also emits `(seq, hidden)` directly, deleting the
        // `flatten+transpose` that followed. 18.1 ms -> ~5 ms per tile.
        let w_flat = patch
            .weight()
            .reshape((cfg.hidden_size, cfg.num_channels * cfg.patch_size * cfg.patch_size))?
            .t()?
            .contiguous()?;
        let bias = patch.bias().cloned();
        Ok(Self {
            w_flat,
            bias,
            position,
            patch_size: cfg.patch_size,
            channels: cfg.num_channels,
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> CandleResult<Tensor> {
        let (_b, _c, h, w) = pixel_values.dims4()?;
        if h % self.patch_size != 0 || w % self.patch_size != 0 {
            candle_core::bail!(
                "image {h}x{w} is not a multiple of patch size {}",
                self.patch_size
            );
        }
        // The convolution AS a matmul — see [`Embeddings::new`] for why that is
        // the same operation and what it is worth. im2col here is one
        // permutation; the product then lands in `(seq, hidden)` directly, so
        // the `flatten_from(2).transpose(1,2)` this used to need is gone too.
        let (b, c, gh, gw) = (
            pixel_values.dim(0)?,
            self.channels,
            h / self.patch_size,
            w / self.patch_size,
        );
        let p = self.patch_size;
        let seq = gh * gw;
        let k = c * p * p;
        let cols = pixel_values
            .reshape((b, c, gh, p, gw, p))?
            .permute((0, 2, 4, 1, 3, 5))?
            .contiguous()?
            .reshape((b * seq, k))?;
        crate::cost::copy((b * seq * k) as u64);
        let xs = cols.matmul(&self.w_flat)?;
        let c_out = self.w_flat.dim(1)?;
        crate::cost::matmul(1, (b * seq) as u64, k as u64, c_out as u64);
        // Both adds in ONE parallel pass — see [`EmbedAddOp`]. candle's two
        // chained `broadcast_add`s are single-threaded and each traverses the
        // whole activation; this is the same defect the projections had.
        let out = match self.bias.as_ref() {
            Some(bias) => xs.apply_op3_no_bwd(bias, &self.position, &EmbedAddOp)?,
            None => xs.reshape((b, seq, c_out))?.broadcast_add(&self.position)?,
        };
        crate::cost::elementwise((c_out * seq) as u64, 2, 1);
        Ok(out)
    }
}

/// Per-op wall clock inside the REAL tower, behind `FFAI_VIS_PROFILE=1`.
///
/// Every vision figure this crate has is from `vision_ops_now`, which prices
/// ops in ISOLATION. That is not the same measurement: for the text tower the
/// isolated sum came to 940 ms against a real 1283 ms forward, and the missing
/// 18 % was allocation churn nobody had priced. The vision tower has never had
/// the equivalent check, so its 77 %-matmul decomposition is an assumption.
pub mod prof {
    use std::sync::Mutex;
    use crate::clock::Instant;

    static ACC: Mutex<Vec<(&'static str, f64)>> = Mutex::new(Vec::new());

    pub(crate) fn on() -> bool {
        use std::sync::atomic::{AtomicU8, Ordering};
        static C: AtomicU8 = AtomicU8::new(u8::MAX);
        match C.load(Ordering::Relaxed) {
            u8::MAX => {
                let v = std::env::var("FFAI_VIS_PROFILE").is_ok_and(|x| x == "1");
                C.store(u8::from(v), Ordering::Relaxed);
                v
            }
            v => v == 1,
        }
    }

    pub(crate) fn add(name: &'static str, t: Instant) {
        if !on() {
            return;
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if let Ok(mut v) = ACC.lock() {
            match v.iter_mut().find(|(n, _)| *n == name) {
                Some(e) => e.1 += ms,
                None => v.push((name, ms)),
            }
        }
    }

    /// Drain accumulated per-op totals, largest first.
    #[must_use]
    pub fn take() -> Vec<(&'static str, f64)> {
        let mut v = ACC.lock().map(|mut g| std::mem::take(&mut *g)).unwrap_or_default();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }
}

/// One encoder layer, with the attention projections fused.
struct Layer {
    ln1: LayerNorm,
    /// **q, k and v in ONE matrix** — `(hidden, 3*hidden)`.
    ///
    /// Same arithmetic, a third of the calls, and one pass over `xs` instead of
    /// three. The q third additionally carries the attention scale (see
    /// [`Layer::load`]), so the `(1,12,1024,1024)` scaling pass disappears.
    /// **Bias-free.** Its bias lives in `qkv_bias`, folded into the packed
    /// permute-copy that follows — see [`PackedQkvOp`].
    qkv: Linear,
    qkv_bias: Option<Tensor>,
    /// **Bias-free.** Its bias lives in `out_bias`, folded into the residual
    /// add that follows — see [`AddBiasOp`].
    out_proj: Linear,
    out_bias: Option<Tensor>,
    ln2: LayerNorm,
    /// **Bias-free.** Its bias lives in `fc1_bias` and is folded into the
    /// GELU that follows — see [`GeluBiasOp`].
    fc1: Linear,
    fc1_bias: Option<Tensor>,
    /// **Bias-free.** Its bias lives in `fc2_bias`, folded into the residual
    /// add that follows — see [`AddBiasOp`].
    fc2: Linear,
    fc2_bias: Option<Tensor>,
    heads: usize,
    head_dim: usize,
}

impl Layer {
    fn load(cfg: &VisionConfig, vb: &VarBuilder) -> CandleResult<Self> {
        let hidden = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let head_dim = hidden / heads;
        let attn = vb.pp("self_attn");

        // ---- fuse q/k/v, and fold the attention scale into q ---------------
        //
        // `attn_weights = (q·kᵀ) * scale` is algebraically `(q*scale)·kᵀ`, and
        // `q = x·Wqᵀ + bq`, so `q*scale = x·(Wq*scale)ᵀ + bq*scale`. Scaling
        // the WEIGHT therefore costs nothing at inference — it is done once,
        // here, on 768x768 numbers instead of 12.6 M per layer per tile.
        //
        // It is not bit-identical to scaling the product, and could not be:
        // the multiply happens before the sum rather than after. It is one
        // rounding of a value already rounded, and the parity test bounds what
        // that costs.
        let scale = (head_dim as f64).powf(-0.5);
        let wq = (attn.get((hidden, hidden), "q_proj.weight")? * scale)?;
        let wk = attn.get((hidden, hidden), "k_proj.weight")?;
        let wv = attn.get((hidden, hidden), "v_proj.weight")?;
        let bq = (attn.get(hidden, "q_proj.bias")? * scale)?;
        let bk = attn.get(hidden, "k_proj.bias")?;
        let bv = attn.get(hidden, "v_proj.bias")?;
        let qkv = Linear::new(
            Tensor::cat(&[&wq, &wk, &wv], 0)?.contiguous()?,
            Some(Tensor::cat(&[&bq, &bk, &bv], 0)?.contiguous()?),
        );

        // Load each projection ONCE, then split the bias off to the kernel
        // downstream that already reads and writes every element.
        let (qkv, qkv_bias) = split_bias(qkv);
        let out = split_bias(candle_nn::linear(hidden, hidden, attn.pp("out_proj"))?);
        let fc1 = split_bias(candle_nn::linear(hidden, cfg.intermediate_size, vb.pp("mlp.fc1"))?);
        let fc2 = split_bias(candle_nn::linear(cfg.intermediate_size, hidden, vb.pp("mlp.fc2"))?);

        Ok(Self {
            ln1: candle_nn::layer_norm(hidden, cfg.layer_norm_eps, vb.pp("layer_norm1"))?,
            qkv,
            qkv_bias,
            out_proj: out.0,
            out_bias: out.1,
            ln2: candle_nn::layer_norm(hidden, cfg.layer_norm_eps, vb.pp("layer_norm2"))?,
            fc1: fc1.0,
            fc1_bias: fc1.1,
            fc2: fc2.0,
            fc2_bias: fc2.1,
            heads,
            head_dim,
        })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (b, seq, hidden) = xs.dims3()?;
        let residual = xs;
        let (bu, sq, hd) = (b as u64, seq as u64, hidden as u64);
        let heads = self.heads as u64;
        let hdim = self.head_dim as u64;

        // ---- attention -----------------------------------------------------
        let _t = crate::clock::Instant::now();
        let normed = layer_norm(&self.ln1, xs)?;
        prof::add("ln1+ln2", _t);
        // layer_norm: one pass, plus the mean/var reduction over the same data.
        crate::cost::elementwise(bu * sq * hd, 2, 1);
        let _t = crate::clock::Instant::now();
        let qkv = self.qkv.forward(&normed)?;
        prof::add("qkv linear", _t);
        crate::cost::matmul(1, bu * sq, hd, 3 * hd);
        // ONE copy, not three.
        //
        // The obvious form narrows q, k and v out of the fused matmul and gives
        // each its own `reshape -> transpose -> contiguous`. That is three
        // copies of 3 MB. Reshaping the whole thing to `(b, seq, 3, heads, hd)`
        // and permuting once produces the same layout in **one** copy of 9 MB,
        // after which the three narrows are free views.
        //
        // Deterministically the two are equal in VOLUME — 2 359 296 elements
        // either way — so no counter distinguishes them and the win is pure
        // locality. Judged by a sign test rather than a ratio, because this box
        // swings +-12 % and the ordering is trustworthy long before the
        // magnitude is: **one-copy won 20/20 interleaved rounds**
        // (`examples/kernel_ab`), and the result is bit-identical.
        // qkv goes to the OUTERMOST axis, not the second.
        //
        // `permute((0, 2, 3, 1, 4))` gives `(b, 3, heads, seq, hd)`, and then
        // `packed.i((.., 0))` is contiguous ONLY at `b == 1` — at any larger
        // batch the three q/k/v groups interleave with the batch and the
        // narrow is strided, which candle's matmul rejects outright
        // (`MatMulUnexpectedStriding: non-contiguous lhs`). This module
        // documents itself as batch-aware and the tower is called batch-aware
        // by `tile_batching_ab`, but every caller so far passed one tile, so
        // nothing exercised it.
        //
        // `permute((2, 0, 3, 1, 4))` gives `(3, b, heads, seq, hd)` instead, so
        // `packed.i(0)` selects a whole contiguous block for any `b`. Same
        // single copy, same volume, and identical at `b == 1` — which is what
        // the existing oracle gates confirm.
        let _t = crate::clock::Instant::now();
        // One pass: the permute-copy the layout forces anyway, with `qkv`'s
        // bias added on the way through instead of in a pass of its own.
        let packed = match (self.qkv_bias.as_ref(), fuse_bias()) {
            (Some(bias), true) => qkv.apply_op2_no_bwd(
                bias,
                &PackedQkvOp { heads: self.heads, head_dim: self.head_dim },
            )?,
            (bias, _) => {
                let qkv = match bias {
                    Some(bias) => qkv.broadcast_add(bias)?,
                    None => qkv,
                };
                qkv.reshape((b, seq, 3, self.heads, self.head_dim))?
                    .permute((2, 0, 3, 1, 4))?
                    .contiguous()?
            }
        };
        crate::cost::copy(3 * bu * sq * hd);
        prof::add("packed permute+copy", _t);
        let q = packed.i(0)?;
        let k = packed.i(1)?;
        let v = packed.i(2)?;

        // No `* scale` here: it is already in q's weights.
        //
        // What the fold DELETES, exactly: an elementwise multiply over
        // `b*heads*seq*seq` elements, reading and writing the 50 MB score
        // matrix — per layer, per tile. That is the deterministic statement of
        // the win; no stopwatch involved.
        if head_attn() {
            // One head at a time: 4.2 MB of scores instead of 50.3 MB, so
            // softmax and attn.v read what q.k^T just wrote, still in cache.
            let mut per_head = Vec::with_capacity(self.heads);
            for h in 0..self.heads {
                let _t = crate::clock::Instant::now();
                let qh = q.i((.., h))?;
                let kh = k.i((.., h))?;
                let scores = qh.matmul(&kh.t()?)?;
                prof::add("q.k^T", _t);
                crate::cost::matmul(bu, sq, hdim, sq);
                let _t = crate::clock::Instant::now();
                // Ours, in place — not `candle_nn::ops::softmax_last_dim`.
                //
                // This branch was the one the 17-tile path takes, and it was
                // the reason vectorising the other two softmax sites moved the
                // 1-tile caption 1.49x and left the 17-tile one flat. candle's
                // version is scalar AND allocates a second (seq, seq) buffer
                // per head; `SoftmaxInplace` reuses the scores tensor and goes
                // through the same lane-split kernels.
                scores.inplace_op1(&SoftmaxInplace)?;
                let probs = scores;
                prof::add("softmax", _t);
                crate::cost::elementwise(bu * sq * sq, 2, 1);
                crate::cost::transcendental_vector(bu * sq * sq);
                let _t = crate::clock::Instant::now();
                per_head.push(probs.matmul(&v.i((.., h))?)?);
                crate::cost::matmul(bu, sq, sq, hdim);
                prof::add("attn.v", _t);
            }
            let _t = crate::clock::Instant::now();
            let attn = Tensor::stack(&per_head, 1)?
                .transpose(1, 2)?
                .reshape((b, seq, hidden))?;
            crate::cost::copy(bu * sq * hd);
            prof::add("attn transpose", _t);
            return self.finish_attention(residual, &attn, bu, sq, hd);
        }

        // ---- blocked attention ---------------------------------------------
        let blk = attn_block();
        if late_normalize() && blk > 0 && blk < seq {
            let kt = k.t()?;
            let mut outs: Vec<Tensor> = Vec::with_capacity(seq.div_ceil(blk));
            let mut sums_parts: Vec<Tensor> = Vec::with_capacity(outs.capacity());
            let mut start = 0usize;
            while start < seq {
                let len = blk.min(seq - start);
                // `narrow` on the seq axis leaves each head's slice strided
                // relative to the whole tensor, and candle's matmul wants a
                // contiguous lhs. The copy is `heads * len * head_dim` — 786 KB
                // at BLOCK 256 — against the 12.6 MB of scores it makes
                // cache-resident.
                let _t = crate::clock::Instant::now();
                let q_blk = q.narrow(2, start, len)?.contiguous()?;
                let scores = q_blk.matmul(&kt)?;
                crate::cost::matmul(bu * heads, len as u64, hdim, sq);
                prof::add("q.k^T", _t);

                let _t = crate::clock::Instant::now();
                let rows = (bu * heads) as usize * len;
                let mut sums = vec![0f32; rows];
                scores.inplace_op1(&SoftmaxExpInplace {
                    sums: RowSums(sums.as_mut_ptr()),
                    rows,
                })?;
                crate::cost::transcendental_vector(bu * heads * len as u64 * sq);
                prof::add("softmax", _t);
                sums_parts.push(Tensor::from_vec(sums, (b, self.heads, len), scores.device())?);

                let _t = crate::clock::Instant::now();
                outs.push(scores.matmul(&v)?);
                crate::cost::matmul(bu * heads, len as u64, sq, hdim);
                prof::add("attn.v", _t);
                start += len;
            }
            let _t = crate::clock::Instant::now();
            let attn = Tensor::cat(&outs, 2)?;
            let sums = Tensor::cat(&sums_parts, 2)?;
            let attn = attn.apply_op2_no_bwd(
                &sums,
                &AttnMergeOp { heads: self.heads, head_dim: self.head_dim },
            )?;
            crate::cost::copy(bu * sq * hd);
            prof::add("attn transpose", _t);
            return self.finish_attention(residual, &attn, bu, sq, hd);
        }

        let _t = crate::clock::Instant::now();
        let scores = q.matmul(&k.t()?)?;
        prof::add("q.k^T", _t);
        crate::cost::matmul(bu * heads, sq, hdim, sq);
        // candle's, measured 11x faster than ours here — see the note above.
        let _t = crate::clock::Instant::now();
        // Deferred normalisation: exp in place, keep the row totals, and let
        // `AttnMergeOp` divide the (seq, head_dim) OUTPUT instead of the
        // (seq, seq) scores. One 100 MB round trip per layer deleted.
        if late_normalize() {
            let rows = (bu * heads * sq) as usize;
            let mut sums = vec![0f32; rows];
            scores.inplace_op1(&SoftmaxExpInplace {
                sums: RowSums(sums.as_mut_ptr()),
                rows,
            })?;
            prof::add("softmax", _t);
            let sums = Tensor::from_vec(sums, (b, self.heads, seq), scores.device())?;
            let _t = crate::clock::Instant::now();
            let attn = scores.matmul(&v)?;
            crate::cost::matmul(bu * heads, sq, sq, hdim);
            prof::add("attn.v", _t);
            let _t = crate::clock::Instant::now();
            let attn = attn.apply_op2_no_bwd(
                &sums,
                &AttnMergeOp { heads: self.heads, head_dim: self.head_dim },
            )?;
            crate::cost::copy(bu * sq * hd);
            prof::add("attn transpose", _t);
            return self.finish_attention(residual, &attn, bu, sq, hd);
        }

        let probs = if inplace_softmax() {
            scores.inplace_op1(&SoftmaxInplace)?;
            scores
        } else {
            candle_nn::ops::softmax_last_dim(&scores)?
        };
        prof::add("softmax", _t);
        // softmax: read the row for the max, read again for exp+sum, write.
        crate::cost::elementwise(bu * heads * sq * sq, 2, 1);
        // candle's softmax uses a vectorized exp (~2.7 G/s measured), NOT the
        // scalar libm path GELU was on (~75 M/s). Counting them under one
        // weight predicted 32 s of transcendentals against a ~16 s tower —
        // the arithmetic failing to close is what exposed the conflation.
        crate::cost::transcendental_vector(bu * heads * sq * sq);
        let _t = crate::clock::Instant::now();
        let attn = probs.matmul(&v)?;
        crate::cost::matmul(bu * heads, sq, sq, hdim);
        prof::add("attn.v", _t);
        let _t = crate::clock::Instant::now();
        let attn = attn.transpose(1, 2)?.reshape((b, seq, hidden))?;
        crate::cost::copy(bu * sq * hd);
        prof::add("attn transpose", _t);
        self.finish_attention(residual, &attn, bu, sq, hd)
    }

    /// Everything after attention: out_proj, the residual, and the whole MLP.
    ///
    /// Shared verbatim by both attention arms ([`head_attn`]) so the toggle can
    /// only change HOW the scores are computed, never what happens to them
    /// afterwards. An A/B whose two arms have separately-maintained tails is
    /// measuring the tails too.
    fn finish_attention(
        &self,
        residual: &Tensor,
        attn: &Tensor,
        bu: u64,
        sq: u64,
        hd: u64,
    ) -> CandleResult<Tensor> {
        let (b, seq, hidden) = attn.dims3()?;
        let _ = (b, seq, hidden);
        let _t = crate::clock::Instant::now();
        let out = self.out_proj.forward(attn)?;
        prof::add("out_proj", _t);
        crate::cost::matmul(1, bu * sq, hd, hd);
        let _t = crate::clock::Instant::now();
        // `out_proj`'s bias rides along here rather than costing its own pass.
        let xs = match (self.out_bias.as_ref(), fuse_bias()) {
            (Some(bias), true) => residual.apply_op3_no_bwd(&out, bias, &AddBiasOp)?,
            (Some(bias), false) => (residual + out.broadcast_add(bias)?)?,
            (None, _) => (residual + out)?,
        };
        prof::add("residual+bias", _t);
        crate::cost::elementwise(bu * sq * hd, 2, 1);

        // ---- mlp -----------------------------------------------------------
        let residual = &xs;
        let _t = crate::clock::Instant::now();
        let normed = layer_norm(&self.ln2, &xs)?;
        prof::add("ln1+ln2", _t);
        crate::cost::elementwise(bu * sq * hd, 2, 1);
        let inter = self.fc1.weight().dims()[0] as u64;
        let _t = crate::clock::Instant::now();
        let h = self.fc1.forward(&normed)?;
        prof::add("fc1", _t);
        crate::cost::matmul(1, bu * sq, hd, inter);
        let _t = crate::clock::Instant::now();
        // The bias `fc1` no longer applies is folded in here, where the data
        // is already being read and written.
        let h = match (self.fc1_bias.as_ref(), fuse_bias()) {
            (Some(b), true) => h.apply_op2_no_bwd(b, &GeluBiasOp)?,
            (Some(b), false) => gelu_tanh_par(&h.broadcast_add(b)?)?,
            (None, _) => gelu_tanh_par(&h)?,
        };
        prof::add("gelu+bias", _t);
        let _t = crate::clock::Instant::now();
        let down = self.fc2.forward(&h)?;
        prof::add("fc2", _t);
        crate::cost::matmul(1, bu * sq, inter, hd);
        let _t = crate::clock::Instant::now();
        // Likewise `fc2`'s.
        let out = match (self.fc2_bias.as_ref(), fuse_bias()) {
            (Some(bias), true) => residual.apply_op3_no_bwd(&down, bias, &AddBiasOp)?,
            (Some(bias), false) => (residual + down.broadcast_add(bias)?)?,
            (None, _) => (residual + down)?,
        };
        prof::add("residual+bias", _t);
        crate::cost::elementwise(bu * sq * hd, 2, 1);
        Ok(out)
    }
}

/// One GELU element, from the shared kernel.
///
/// # Why this is no longer a local polynomial
///
/// Argus had its own `exp_poly`. So did `ffai-diana` (`exp_fast`) and
/// `ffai-mercury` (`fast_exp`) — three implementations of one idea, none aware
/// of the others, and they had **drifted in the detail that decides whether the
/// win happens**: the round-to-integer step.
///
/// Argus used `f32::round`, which is ties-away-from-zero, which no x86
/// instruction implements — so it lowered to a libm call unless the whole
/// kernel was behind `#[target_feature]`. Diana had already found and fixed
/// that with the `+1.5*2^23` trick, measured at **4.71x on the rounding step
/// alone and bit-identical**, and needing no `target_feature` at all.
///
/// `ffai_core::fastmath` is that fix, shared. The AVX2 twin below is kept
/// because it still helps, but it is no longer load-bearing for correctness of
/// the *scalar* path's speed — which is what the non-x86 targets run.
#[inline(always)]
fn gelu_one(v: f32) -> f32 {
    ffai_core::fastmath::gelu_tanh(v)
}

/// The scalar kernel. Written as a plain loop over a slice with no branches
/// and no calls, which is the form LLVM will widen on its own once the target
/// permits it.
#[inline]
fn gelu_chunk_scalar(chunk: &mut [f32]) {
    for x in chunk {
        *x = gelu_one(*x);
    }
}

/// The same kernel compiled for **AVX2 + FMA**, selected at runtime.
///
/// # ⚠ Its justification SHRANK when the shared kernel landed
///
/// This twin was written when `gelu_one` used `f32::round`, which lowers to a
/// libm call outside `target_feature` — so the AVX2 path was the only fast
/// path and measured **2.30x** over scalar.
///
/// `ffai_core::fastmath` replaced that rounding with the `+1.5*2^23` trick
/// (`ffai-diana` had already found it), which removes the call entirely and
/// lets the SCALAR loop auto-vectorise. Re-measured on the same box, same
/// binary, 3.1 M elements:
///
/// | | before | after |
/// |---|---:|---:|
/// | scalar | 13.77 ms | **5.68 ms** |
/// | avx2+fma | 5.98 ms | 4.76 ms |
/// | AVX2 advantage | **2.30x** | **1.19x** |
///
/// The scalar path got 2.42x faster and the hand-written twin went from
/// load-bearing to marginal. It is kept because 1.19x is real (20/20 sign
/// test, bit-identical) and it costs nothing at runtime — but by
/// `codec-vectorize-kernel`'s Step 0 ("already vectorised -> STOP") a NEW
/// kernel would not be written for 1.19x, and if this one ever needs
/// maintenance that is the number to weigh against the `unsafe`.
///
/// The baseline `x86_64` target is SSE2, so the scalar loop above is compiled
/// four-wide at best and usually one-wide. This is the same arithmetic with
/// `target_feature` applied, which lets LLVM use 8-wide registers and fused
/// multiply-add for the Horner chains.
///
/// Runtime-detected rather than build-time: `-C target-cpu=native` would make
/// a binary that crashes on any older machine, and `FFai` ships published
/// crates.
///
/// # Safety
///
/// The caller must have verified AVX2 and FMA are present. The only entry
/// point is [`gelu_chunk`], which checks once via `is_x86_feature_detected!`
/// and caches the answer, so no caller can reach this without the check.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_chunk_avx2(chunk: &mut [f32]) {
    for x in chunk {
        *x = gelu_one(*x);
    }
}

/// One softmax row, scalar. The oracle for the AVX2 twin below.
#[inline]
fn softmax_row_scalar(dst: &mut [f32], row: &[f32]) {
    let mut max = f32::NEG_INFINITY;
    for &x in row {
        if x > max {
            max = x;
        }
    }
    // EIGHT independent accumulators, not one.
    //
    // `sum += e` is a loop-carried dependency, and f32 addition is not
    // associative — so LLVM refuses to reorder it into lanes and the whole
    // loop stays scalar **even inside a `target_feature` function**. That is
    // what the first two softmax attempts measured: 12.6 M x ~15 ops at ~1
    // op/cycle is ~63 ms, which is exactly what they took.
    //
    // Splitting the chain is the documented fix. It is a reassociation, so the
    // sum differs from a sequential fold in the last bits — gated by tolerance
    // against candle's, not by bit-identity.
    let mut acc = [0.0f32; 8];
    let n = row.len();
    let tail = n % 8;
    let mut i = 0;
    while i + 8 <= n {
        for l in 0..8 {
            let e = ffai_core::fastmath::exp(row[i + l] - max);
            dst[i + l] = e;
            acc[l] += e;
        }
        i += 8;
    }
    let mut sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for k in 0..tail {
        let e = ffai_core::fastmath::exp(row[i + k] - max);
        dst[i + k] = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for d in dst.iter_mut() {
        *d *= inv;
    }
}

/// The same row kernel compiled for AVX2+FMA.
///
/// # Why softmax needed this and the first attempt did not have it
///
/// The first zero-copy softmax called `exp_poly` from a plain closure and
/// measured **0.05x** — twenty times *slower* than candle. The cause was not the
/// algorithm: without `target_feature`, `exp_poly`'s `round()` has no baseline
/// instruction (`roundps` is SSE4.1) and lowers to a **call to `roundf`**. So
/// the "no libm calls" kernel was making 12.6 M libm calls per invocation —
/// precisely the cost it existed to remove.
///
/// GELU never showed this because its kernel was behind `target_feature` from
/// the start. Two call sites, one polynomial, and only one of them was compiled
/// somewhere the polynomial could actually be arithmetic.
///
/// # Safety
///
/// The caller must have verified AVX2 and FMA. The only entry point is the
/// dispatch in `SoftmaxLastDimOp::cpu_fwd`, which checks once.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_row_avx2(dst: &mut [f32], row: &[f32]) {
    softmax_row_scalar(dst, row);
}

/// Is AVX2+FMA available? Probed once; `is_x86_feature_detected!` reads a
/// cached value after the first call, but this makes that explicit.
#[cfg(target_arch = "x86_64")]
fn have_avx2_cached() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0); // 0 unknown, 1 no, 2 yes
    match CACHE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let yes = std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma");
            CACHE.store(u8::from(yes) + 1, Ordering::Relaxed);
            yes
        }
    }
}

/// Which GELU kernel this CPU will actually run — for probes and reports.
///
/// Exposed because "we added AVX2" is a claim, and a claim about a runtime
/// dispatch that nobody checked is how a fallback path ships unnoticed.
#[must_use]
pub fn gelu_kernel_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if have_avx2_cached() {
            return "avx2+fma";
        }
        return "scalar (no avx2)";
    }
    #[cfg(not(target_arch = "x86_64"))]
    "scalar (non-x86_64)"
}

/// Force the scalar kernel — for `examples/kernel_ab` only.
///
/// Exposed so the two paths can be raced **inside one binary**. Confirming
/// that dispatch *selects* AVX2 is not the same as confirming AVX2 is faster,
/// and the second claim is the one worth making.
pub fn gelu_scalar_for_probe(chunk: &mut [f32]) {
    gelu_chunk_scalar(chunk);
}

/// Force the AVX2 kernel — for `examples/kernel_ab` only.
///
/// # Panics
/// If AVX2+FMA is absent. A probe that silently fell back would report the
/// scalar path's time under the AVX2 label, which is the exact confusion this
/// function exists to remove.
#[cfg(target_arch = "x86_64")]
pub fn gelu_avx2_for_probe(chunk: &mut [f32]) {
    assert!(have_avx2_cached(), "AVX2+FMA not available on this CPU");
    // SAFETY: the assert above is the documented precondition.
    unsafe { gelu_chunk_avx2(chunk) };
}

/// Non-x86 stub, so the `cfg!(...) && have_avx2_cached()` above type-checks on
/// every target rather than only the one being developed on.
///
/// # Why there is no NEON mirror
///
/// The skill's rule is per-arch parity or a written note. This is the note.
///
/// **NEON is BASELINE on `aarch64`**, so `gelu_chunk_scalar` is already
/// compiled to vector instructions there without any `target_feature`, any
/// runtime detection, or a second code path that production may never execute.
/// The reason an x86 twin is needed at all is that the portable x86-64 baseline
/// is SSE2, and this kernel needs `roundps` (SSE4.1) and FMA — both above it.
/// That asymmetry is about the two baselines, not about the kernel.
///
/// The polynomial was written branch-free and call-free precisely so a compiler
/// can widen it unaided wherever the baseline allows.
#[cfg(not(target_arch = "x86_64"))]
const fn have_avx2_cached() -> bool {
    false
}


/// LayerNorm as ONE fused parallel pass. **REFUTED — off by default.**
///
/// # What candle actually runs
///
/// `candle_nn::LayerNorm::forward` is built from whole-tensor ops:
///
/// ```text
/// mean = x.sum_keepdim(-1) / n
/// x    = x.broadcast_sub(mean)
/// var  = x.sqr().sum_keepdim(-1) / n
/// x    = x.broadcast_div((var + eps).sqrt())
/// out  = x.broadcast_mul(weight).broadcast_add(bias)
/// ```
///
/// Every line there is a **separate full traversal** of the activation, and
/// every `broadcast_*` is one of candle's **single-threaded** binary ops. For
/// `(1, 1024, 768)` that is six passes over 3.1 MB, twice per layer, twenty-four
/// times per tile — around **450 MB of traffic per tile** to apply two numbers
/// per column.
///
/// # What this does instead
///
/// Two traversals of each row, in one parallel kernel: one to accumulate the
/// sum and sum-of-squares together, one to write
/// `(x - mean) * inv_std * weight + bias`. The row is 768 floats — 3 KB — so
/// the second traversal reads it straight back out of L1.
///
/// # Numerics
///
/// Variance comes from `E[x^2] - E[x]^2` rather than candle's two-pass
/// `E[(x - mean)^2]`. That is the standard fused formulation and is what makes
/// the single accumulation possible; it is float-close, not bit-identical, so
/// it is gated on the reference caption. `eps` is added before the square root
/// exactly as candle does, and the variance is clamped at zero so cancellation
/// in the subtraction can never reach `sqrt` with a negative argument.
///
/// # The refutation
///
/// It moves a THIRD of candle's bytes and is still not faster. Measured at the
/// op level (`examples/kernel_micro_ab`, best-of-30 ABBA) it read **1.09x on one
/// run and 0.89x on the next** — a verdict that changes sign between runs is the
/// instrument talking, not the code, and the honest reading is parity.
///
/// The premise was wrong: the activation is 3.1 MB and fits L3, so candle's
/// "six passes" are cache hits, not six trips to DRAM. Pass-counting only
/// predicts cost when the data does not fit — which is exactly why the same
/// reasoning DID pay on the 12.6 MB bias adds and the 50 MB score matrix.
///
/// Off by default: it is a wash on speed and it changes the variance
/// formulation, so it is risk without return. Kept as a measurable arm.
struct LayerNormOp {
    eps: f64,
}

impl candle_core::CustomOp3 for LayerNormOp {
    fn name(&self) -> &'static str {
        "ffai-layer-norm"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
        s3: &candle_core::CpuStorage,
        l3: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        let (x, w, b) = match (s1, s2, s3) {
            (
                candle_core::CpuStorage::F32(x),
                candle_core::CpuStorage::F32(w),
                candle_core::CpuStorage::F32(b),
            ) => (x, w, b),
            _ => candle_core::bail!("ffai-layer-norm expects f32"),
        };
        let (Some((xo, xe)), Some((wo, we)), Some((bo, be))) = (
            l1.contiguous_offsets(),
            l2.contiguous_offsets(),
            l3.contiguous_offsets(),
        ) else {
            candle_core::bail!("ffai-layer-norm expects contiguous inputs")
        };
        let (x, w, b) = (&x[xo..xe], &w[wo..we], &b[bo..be]);
        let width = w.len();
        if width == 0 || b.len() != width || x.len() % width != 0 {
            candle_core::bail!("ffai-layer-norm: {} not divisible by {width}", x.len());
        }
        let n = x.len();
        let inv_n = 1.0f64 / width as f64;
        let eps = self.eps;
        let mut out: Vec<f32> = Vec::with_capacity(n);
        {
            let spare = out.spare_capacity_mut();
            // SAFETY: exactly `n` contiguous `MaybeUninit<f32>`; the row chunks
            // partition `[0, n)` and each writes every element it owns before
            // `set_len` publishes them. `f32` has no invalid bit patterns.
            #[allow(unsafe_code)]
            let dst: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n) };
            // One reduction pass and one write pass over each row.
            crate::cost::elementwise(n as u64, 2, 1);
            let row = |(o, r): (&mut [f32], &[f32])| {
                // EIGHT independent accumulator pairs, not one.
                //
                // f32 addition is not associative, so a single running total is
                // a serial dependency chain LLVM may not reorder into lanes —
                // it emitted scalar code and the kernel ran at ~6 GB/s despite
                // moving a third of candle's bytes. Eight partial sums break
                // the chain into eight independent ones, which vectorise, and
                // they are also MORE accurate than the serial fold: error grows
                // with the depth of the chain, and this cuts it eightfold.
                const L: usize = 8;
                let (mut sum, mut sq) = ([0.0f32; L], [0.0f32; L]);
                let mut it = r.chunks_exact(L);
                for c in &mut it {
                    for i in 0..L {
                        sum[i] += c[i];
                        sq[i] += c[i] * c[i];
                    }
                }
                for &v in it.remainder() {
                    sum[0] += v;
                    sq[0] += v * v;
                }
                let (sum, sq) = (
                    f64::from(sum.iter().sum::<f32>()),
                    f64::from(sq.iter().sum::<f32>()),
                );
                let mean = sum * inv_n;
                // Clamped: catastrophic cancellation in E[x^2] - E[x]^2 can
                // produce a tiny negative, and sqrt of that is NaN.
                let var = (sq * inv_n - mean * mean).max(0.0);
                let inv_std = 1.0 / (var + eps).sqrt();
                let (mean, inv_std) = (mean as f32, inv_std as f32);
                for (((o, &v), &w), &b) in o.iter_mut().zip(r).zip(w).zip(b) {
                    *o = (v - mean) * inv_std * w + b;
                }
            };
            if kernels_parallel() {
                dst.par_chunks_mut(width).zip(x.par_chunks(width)).for_each(row);
            } else {
                dst.chunks_mut(width).zip(x.chunks(width)).for_each(row);
            }
        }
        // SAFETY: every element written above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(n);
        }
        Ok((candle_core::CpuStorage::F32(out), l1.shape().clone()))
    }
}

/// Use the fused LayerNorm? `FFAI_ARGUS_FUSED_LN=0` uses candle's chain.
fn fused_ln() -> bool {
    FUSED_LN.load(std::sync::atomic::Ordering::Relaxed)
}

/// Toggle the fused LayerNorm, returning the previous setting.
pub fn set_fused_ln(on: bool) -> bool {
    FUSED_LN.swap(on, std::sync::atomic::Ordering::Relaxed)
}

static FUSED_LN: std::sync::LazyLock<std::sync::atomic::AtomicBool> =
    std::sync::LazyLock::new(|| {
        std::sync::atomic::AtomicBool::new(arm_flag("FFAI_ARGUS_FUSED_LN", false))
    });

/// Apply `ln` to `x`, through [`LayerNormOp`] when it is enabled.
fn layer_norm(ln: &LayerNorm, x: &Tensor) -> CandleResult<Tensor> {
    match (fused_ln(), ln.bias()) {
        (true, Some(bias)) => x.apply_op3_no_bwd(ln.weight(), bias, &LayerNormOp { eps: ln.eps() }),
        _ => ln.forward(x),
    }
}

/// [`layer_norm`], exposed so `examples/kernel_micro_ab` can price it against
/// candle's at the op level — where the effect is 100 % of what the clock sees.
pub fn layer_norm_for_probe(ln: &LayerNorm, x: &Tensor) -> CandleResult<Tensor> {
    layer_norm(ln, x)
}

/// Read a `0`/`1` override for one of the vision arms, defaulting to `default`.
///
/// The arm statics were originally plain `AtomicBool`s with hardcoded defaults,
/// while their doc comments advertised environment overrides. That is a
/// documentation bug of the worst kind — an operator reading the docs would set
/// the variable, see no change, and conclude the arm does not matter. Seeding
/// from the environment on first touch makes the documented control real, and
/// `LazyLock` keeps the setters working for interleaved A/Bs.
fn arm_flag(key: &str, default: bool) -> bool {
    match std::env::var(key).ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => default,
    }
}

/// Query-row block size for attention. **REFUTED — 0 (unblocked) by default.**
///
/// # The 50 MB that fits nowhere
///
/// Batched over 12 heads the score matrix is `(1, 12, 1024, 1024)` = **50.3 MB**,
/// against roughly 32 MB of L3. It does not fit, so `q.k^T` writes it to DRAM,
/// softmax reads and rewrites it from DRAM, and `attn.v` reads it from DRAM
/// again: **200 MB of round trip per layer, 2.4 GB per tile**. Softmax measures
/// ~16 GB/s across that traffic, which is main memory, so most of the attention
/// block's cost is the distance rather than the arithmetic.
///
/// Blocking the QUERY rows shrinks the live scores to `(1, 12, BLOCK, 1024)` —
/// 12.6 MB at `BLOCK = 256`, comfortably L3-resident. Softmax and `attn.v` then
/// read what `q.k^T` just wrote, while it is still in cache.
///
/// # Why this is not the per-head attempt again
///
/// Looping the HEADS was refuted at 0.888x: candle's batched matmul
/// parallelises across the batch dimension, and the heads *are* that dimension,
/// so looping them starved the gemm of threads. This splits the query rows and
/// keeps all 12 heads batched in every gemm — a `(1, 12, 256, 64)` lhs still
/// offers 12 independent gemms to divide. It buys the locality the per-head
/// version wanted without paying what the per-head version paid.
///
/// The FLOP count is identical either way; only the distance the bytes travel
/// changes. `FFAI_ARGUS_ATTN_BLOCK` tunes it; 0 restores the unblocked path.
///
/// # The refutation, and the law it completes
///
/// Swept against the unblocked path on an interleaved A/B whose two arms are
/// **bit-identical** (`max |on - off| = 0`):
///
/// | BLOCK | speedup |
/// |---:|---:|
/// | 128 | **0.732x** |
/// | 256 | **0.824x** |
/// | 512 | **0.926x** |
/// | 0 (unblocked) | 1.000x — the winner |
///
/// Monotonic: the closer the block gets to "no blocking", the better it does.
/// The locality is real — 12.6 MB is L3-resident where 50.3 MB is not — but
/// candle re-packs the k panel once per block, and the gemm's efficiency falls
/// with `M`. Both costs exceed what the cache buys.
///
/// This is the THIRD independent refutation of the same idea, and together they
/// are a law worth stating: **candle's gemm rewards large batched calls, and
/// restructuring attention to improve locality has lost every time.** Looping
/// heads lost 0.888x, blocking query rows loses here, and even dropping a
/// vestigial leading batch-of-1 to rank 3 lost 0.74x. The 200 MB per layer is
/// real traffic, and it is still cheaper than any arrangement that avoids it.
fn attn_block() -> usize {
    ATTN_BLOCK.load(std::sync::atomic::Ordering::Relaxed)
}

/// Set the attention query-block size, returning the previous value.
///
/// An atomic rather than a one-shot read because the A/B has to flip it
/// BETWEEN forwards inside a single process — two processes measured minutes
/// apart on this box differ by more than the effect being tested.
pub fn set_attn_block(n: usize) -> usize {
    ATTN_BLOCK.swap(n, std::sync::atomic::Ordering::Relaxed)
}

static ATTN_BLOCK: std::sync::LazyLock<std::sync::atomic::AtomicUsize> =
    std::sync::LazyLock::new(|| {
        std::sync::atomic::AtomicUsize::new(
            std::env::var("FFAI_ARGUS_ATTN_BLOCK")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0),
        )
    });

/// [`kernels_parallel`] for sibling modules with their own kernels.
pub(crate) fn kernels_parallel_for_probe() -> bool {
    kernels_parallel()
}

/// `(x + bias) + position` — the patch embedding's two adds as ONE pass.
///
/// # The same trap, at the one site nobody looked at
///
/// The embedding finishes with two chained `broadcast_add`s:
///
/// ```text
/// xs  = xs.broadcast_add(bias)          // (1024, 768), 3.1 MB
/// out = xs.reshape(..).broadcast_add(position)   // again, 3.1 MB
/// ```
///
/// Both are candle binary ops, so both are **single-threaded**, and each reads
/// and writes the whole activation. That is the identical defect this round
/// found in `candle_nn::Linear` — and it was hiding here too, in a stage the
/// per-op profile reports as a flat "patch+pos embed" with no interior.
///
/// One pass instead of two: **3.1 MB of read+write deleted per tile, 53 MB per
/// caption**, and the surviving pass is parallel where both originals were not.
///
/// # Bit-identical, deliberately
///
/// The additions happen in the same ORDER the two chained ops used —
/// `(x + bias) + position`, never `x + (bias + position)`. Folding
/// `bias + position` into one tensor at load time would be tempting (it is what
/// the attention-scale fold does, and it would delete the bias entirely), but
/// float addition is not associative, so that changes rounding. Here the order
/// is preserved, so this kernel is **exactly** equal to the pair it replaces
/// and its test asserts equality, not a tolerance.
struct EmbedAddOp;

impl candle_core::CustomOp3 for EmbedAddOp {
    fn name(&self) -> &'static str {
        "ffai-embed-add"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
        s3: &candle_core::CpuStorage,
        l3: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        let (x, bias, pos) = match (s1, s2, s3) {
            (
                candle_core::CpuStorage::F32(x),
                candle_core::CpuStorage::F32(b),
                candle_core::CpuStorage::F32(p),
            ) => (x, b, p),
            _ => candle_core::bail!("ffai-embed-add expects f32"),
        };
        let (Some((xo, xe)), Some((bo, be)), Some((po, pe))) = (
            l1.contiguous_offsets(),
            l2.contiguous_offsets(),
            l3.contiguous_offsets(),
        ) else {
            candle_core::bail!("ffai-embed-add expects contiguous inputs")
        };
        let (x, bias, pos) = (&x[xo..xe], &bias[bo..be], &pos[po..pe]);
        let hidden = bias.len();
        if hidden == 0 || x.len() % hidden != 0 || pos.len() % hidden != 0 {
            candle_core::bail!("ffai-embed-add: {} elems, hidden {hidden}", x.len());
        }
        let seq = pos.len() / hidden;
        let rows = x.len() / hidden;
        if seq == 0 || rows % seq != 0 {
            candle_core::bail!("ffai-embed-add: {rows} rows not a multiple of seq {seq}");
        }
        let b = rows / seq;
        let n = x.len();
        let mut out: Vec<f32> = Vec::with_capacity(n);
        {
            let spare = out.spare_capacity_mut();
            // SAFETY: exactly `n` contiguous `MaybeUninit<f32>`; the row chunks
            // partition `[0, n)` and each writes every element it owns before
            // `set_len` publishes them. `f32` has no invalid bit patterns.
            #[allow(unsafe_code)]
            let dst: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n) };
            crate::cost::elementwise(n as u64, 2, 1);
            let row = |(i, (o, xr)): (usize, (&mut [f32], &[f32]))| {
                // `pos` repeats every `seq` rows, once per batch element.
                let pr = &pos[(i % seq) * hidden..(i % seq + 1) * hidden];
                for (((o, &v), &bz), &pz) in o.iter_mut().zip(xr).zip(bias).zip(pr) {
                    // Same association as the two chained broadcast_adds.
                    *o = (v + bz) + pz;
                }
            };
            if kernels_parallel() {
                dst.par_chunks_mut(hidden).zip(x.par_chunks(hidden)).enumerate().for_each(row);
            } else {
                dst.chunks_mut(hidden).zip(x.chunks(hidden)).enumerate().for_each(row);
            }
        }
        // SAFETY: every element written above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(n);
        }
        Ok((candle_core::CpuStorage::F32(out), (b, seq, hidden).into()))
    }
}

/// A row-sum side channel for [`SoftmaxExpInplace`].
///
/// # Safety
///
/// The pointer addresses `rows` initialised `f32`s owned by the caller for the
/// whole call. Row `r` of the score matrix is written by exactly one
/// `par_chunks_mut` chunk — the chunks partition the buffer, so no two threads
/// ever address the same `r`, and each `r` is written exactly once. That is the
/// entire aliasing argument; nothing reads the buffer until `inplace_op1`
/// returns and the rayon join has completed.
struct RowSums(*mut f32);

// SAFETY: writes are partitioned one row per chunk, as argued on `RowSums`.
#[allow(unsafe_code)]
unsafe impl Sync for RowSums {}
// SAFETY: as above.
#[allow(unsafe_code)]
unsafe impl Send for RowSums {}

/// `exp(x - rowmax)` in place, WITHOUT the normalising divide, handing the row
/// sums back through `sums`.
///
/// # The pass this deletes
///
/// A softmax is three passes over its input: find the row max, write
/// `exp(x - max)` and total it, then divide every element by that total. On
/// vision attention the input is **50.3 MB**, so that third pass reads and
/// writes 100 MB per layer to perform 12.6 M divides.
///
/// But softmax feeds straight into `attn.v`, and the divide commutes with it:
///
/// ```text
/// out[i,:] = SUM_j (p[i,j] / S_i) * v[j,:]  ==  (SUM_j p[i,j] * v[j,:]) / S_i
/// ```
///
/// So the normalisation can be applied to the OUTPUT instead — `(1024, 64)` per
/// head rather than `(1024, 1024)`. That is **786 K divides instead of 12.6 M**,
/// and one whole 100 MB round trip per layer deleted: **20 GB per caption**.
/// [`AttnMergeOp`] performs it inside the transpose the layer already pays for,
/// so it costs nothing at all.
///
/// # Numerics
///
/// `exp(x - max) <= 1` and the row sum is at least 1 (the max element
/// contributes `exp(0) = 1`), so no term can overflow and the divisor can never
/// be zero or subnormal. Deferring the divide changes the rounding — it is the
/// same rescaling flash-attention performs — so this is float-close, not
/// bit-identical, and is gated on the reference caption.
struct SoftmaxExpInplace {
    sums: RowSums,
    rows: usize,
}

impl candle_core::InplaceOp1 for SoftmaxExpInplace {
    fn name(&self) -> &'static str {
        "ffai-softmax-exp-inplace"
    }

    fn cpu_fwd(
        &self,
        storage: &mut candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> CandleResult<()> {
        let candle_core::CpuStorage::F32(x) = storage else {
            candle_core::bail!("ffai-softmax-exp-inplace expects f32")
        };
        let Some((o, e)) = layout.contiguous_offsets() else {
            candle_core::bail!("ffai-softmax-exp-inplace expects a contiguous input")
        };
        let width = *layout.shape().dims().last().expect("rank >= 1");
        if width == 0 || (e - o) != self.rows * width {
            candle_core::bail!("ffai-softmax-exp-inplace: {} vs {}x{width}", e - o, self.rows);
        }
        crate::cost::elementwise((e - o) as u64, 1, 1);
        crate::cost::transcendental_vector((e - o) as u64);
        let sums = &self.sums;
        x[o..e].par_chunks_mut(width).enumerate().for_each(|(r, row)| {
            // Vectorised too — this is a full pass over the same 50 MB-per-layer
            // score tensor, and `max = max.max(v)` is a loop-carried reduction
            // on a NaN-aware function, so LLVM lanes it no better than the sum.
            // Exact (max is associative on non-NaN floats), so its twin is
            // gated by `assert_eq!` rather than a tolerance.
            let max = ffai_core::fastmath::max_f32(row);
            // Explicit lanes, not a scalar rewrite. This loop was 22.7 % of
            // the vision tower at 228 M elem/s — 2.7x the `q.k^T` matmul that
            // produces its input, which is the impossible ratio that pointed
            // here. `(*v - max).exp()` is a libm CALL per element and
            // `sum += ex` is a loop-carried dependency on a non-associative
            // add, so LLVM will not lane it.
            //
            // A scalar polynomial exp with eight split accumulators was tried
            // first and measured SLOWER than libm (2410 ms vs 2090 ms): it
            // removes the call and still does not vectorise. Only explicit
            // intrinsics move it. See `ffai_core::fastmath::exp_sub_sum_inplace`
            // — AVX2 twin, wasm SIMD128 twin, scalar oracle, tolerance-gated.
            let sum = ffai_core::fastmath::exp_sub_sum_inplace(row, max);
            // SAFETY: `r < self.rows` because the chunks partition a buffer of
            // exactly `rows * width`, and this chunk is the only writer of `r`.
            #[allow(unsafe_code)]
            unsafe {
                *sums.0.add(r) = sum;
            }
        });
        Ok(())
    }
}

/// `(b, heads, seq, hd) -> (b, seq, heads*hd)`, dividing each row by its
/// softmax total on the way through.
///
/// The layer already pays for this transpose — it is the copy that turns the
/// per-head attention output back into a hidden vector. Folding
/// [`SoftmaxExpInplace`]'s deferred normalisation into it makes the divide
/// free: the reciprocal is computed once per `(b, h, s)` and applied to that
/// row's `head_dim` contiguous outputs.
struct AttnMergeOp {
    heads: usize,
    head_dim: usize,
}

impl candle_core::CustomOp2 for AttnMergeOp {
    fn name(&self) -> &'static str {
        "ffai-attn-merge"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        let (a, sums) = match (s1, s2) {
            (candle_core::CpuStorage::F32(a), candle_core::CpuStorage::F32(b)) => (a, b),
            _ => candle_core::bail!("ffai-attn-merge expects f32"),
        };
        let (Some((ao, ae)), Some((so, se))) = (l1.contiguous_offsets(), l2.contiguous_offsets())
        else {
            candle_core::bail!("ffai-attn-merge expects contiguous inputs")
        };
        let (a, sums) = (&a[ao..ae], &sums[so..se]);
        let (heads, hd) = (self.heads, self.head_dim);
        let hidden = heads * hd;
        let n = ae - ao;
        if hidden == 0 || n % hidden != 0 || sums.len() * hd != n {
            candle_core::bail!("ffai-attn-merge: {n} elems, {} sums", sums.len());
        }
        let dims = l1.shape().dims4()?;
        let (b, seq) = (dims.0, dims.2);
        let mut out: Vec<f32> = Vec::with_capacity(n);
        {
            let spare = out.spare_capacity_mut();
            // SAFETY: exactly `n` contiguous `MaybeUninit<f32>`. The loop writes
            // one output row of `hidden` elements per `(b, seq)` pair and those
            // tile `[0, n)` exactly, so all are initialised before `set_len`.
            #[allow(unsafe_code)]
            let dst: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n) };
            crate::cost::elementwise(n as u64, 1, 1);
            let row = |(i, o): (usize, &mut [f32])| {
                let (bb, s) = (i / seq, i % seq);
                for h in 0..heads {
                    let inv = 1.0 / sums[(bb * heads + h) * seq + s];
                    let src = (bb * heads + h) * seq * hd + s * hd;
                    for (o, &v) in o[h * hd..(h + 1) * hd].iter_mut().zip(&a[src..src + hd]) {
                        *o = v * inv;
                    }
                }
            };
            if kernels_parallel() {
                dst.par_chunks_mut(hidden).enumerate().for_each(row);
            } else {
                dst.chunks_mut(hidden).enumerate().for_each(row);
            }
        }
        // SAFETY: every element written above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(n);
        }
        Ok((candle_core::CpuStorage::F32(out), (b, seq, hidden).into()))
    }
}

/// Defer softmax's divide past `attn.v`? `FFAI_ARGUS_LATE_NORM=0` divides early.
fn late_normalize() -> bool {
    LATE_NORM.load(std::sync::atomic::Ordering::Relaxed)
}

/// Toggle deferred normalisation, returning the previous setting.
pub fn set_late_normalize(on: bool) -> bool {
    LATE_NORM.swap(on, std::sync::atomic::Ordering::Relaxed)
}

static LATE_NORM: std::sync::LazyLock<std::sync::atomic::AtomicBool> =
    std::sync::LazyLock::new(|| {
        std::sync::atomic::AtomicBool::new(arm_flag("FFAI_ARGUS_LATE_NORM", true))
    });

/// Softmax over the last axis, written back into the SAME buffer.
///
/// # What this deletes
///
/// `candle_nn::ops::softmax_last_dim` allocates its output. For vision
/// attention that output is `(1, 12, 1024, 1024)` — **50.3 MB** — so each layer
/// asks the allocator for 50 MB, touches 12 800 fresh pages, and frees the
/// 50 MB it just read. Across 12 layers and 17 tiles that is **10.3 GB** of
/// allocate-fault-free doing no arithmetic.
///
/// In place, softmax reads and writes the buffer `q.k^T` has just written,
/// which is still as warm as it will ever be.
///
/// # Why the earlier attempt lost, and what changed
///
/// A previous in-place softmax measured 0.92x and was removed. It was not a
/// fair fight: that kernel respected [`kernels_parallel`] and ran SERIAL, while
/// candle's `softmax_last_dim` is unconditionally parallel
/// (`par_chunks`/`par_chunks_mut`). This one is parallel over rows on the same
/// terms.
///
/// # Numerics
///
/// Identical to candle's, op for op and in the same order: row max, then
/// `exp(x - max)` into the destination, then a sum of those, then a divide. The
/// only difference is that the destination and the source are the same memory.
struct SoftmaxInplace;

impl candle_core::InplaceOp1 for SoftmaxInplace {
    fn name(&self) -> &'static str {
        "ffai-softmax-inplace"
    }

    fn cpu_fwd(
        &self,
        storage: &mut candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> CandleResult<()> {
        let candle_core::CpuStorage::F32(x) = storage else {
            candle_core::bail!("ffai-softmax-inplace expects f32")
        };
        let Some((o, e)) = layout.contiguous_offsets() else {
            candle_core::bail!("ffai-softmax-inplace expects a contiguous input")
        };
        let width = *layout.shape().dims().last().expect("rank >= 1");
        if width == 0 {
            return Ok(());
        }
        let rows = &mut x[o..e];
        crate::cost::elementwise((e - o) as u64, 2, 1);
        crate::cost::transcendental_vector((e - o) as u64);
        // The same vectorised primitives as the deferred-normalise kernel this
        // is the fallback for (`SoftmaxExpInplace`). Reachable when
        // `FFAI_ARGUS_LATE_NORMALIZE=0`, and there is no reason for the arm a
        // measurement selects to be slower than the arm that ships.
        let row = |r: &mut [f32]| {
            let max = ffai_core::fastmath::max_f32(r);
            let sum = ffai_core::fastmath::exp_sub_sum_inplace(r, max);
            let inv = 1.0 / sum;
            for v in r.iter_mut() {
                *v *= inv;
            }
        };
        // Unconditionally parallel — candle's is, and an arm that is serial
        // where its reference is parallel measures the threading, not the idea.
        rows.par_chunks_mut(width).for_each(row);
        Ok(())
    }
}

/// Is the in-place softmax on? `FFAI_ARGUS_INPLACE_SOFTMAX=0` uses candle's.
fn inplace_softmax() -> bool {
    INPLACE_SOFTMAX.load(std::sync::atomic::Ordering::Relaxed)
}

/// Toggle the in-place softmax, returning the previous setting.
pub fn set_inplace_softmax(on: bool) -> bool {
    INPLACE_SOFTMAX.swap(on, std::sync::atomic::Ordering::Relaxed)
}

static INPLACE_SOFTMAX: std::sync::LazyLock<std::sync::atomic::AtomicBool> =
    std::sync::LazyLock::new(|| {
        std::sync::atomic::AtomicBool::new(arm_flag("FFAI_ARGUS_INPLACE_SOFTMAX", true))
    });

/// Run attention ONE HEAD AT A TIME? **REFUTED — off by default.**
///
/// # The 50 MB that never needed to exist
///
/// Batched over all heads, `q.k^T` produces `(1, 12, 1024, 1024)` — **50.3 MB**
/// — and softmax allocates a second one. Nothing in cache survives that: the
/// scores are written to DRAM, read back by softmax, written again, and read a
/// third time by `attn.v`. It shows up in the profile as the only two matmuls
/// below peak (`q.k^T` 210 GF/s, `attn.v` 329 GF/s, against 500-563 for the
/// four projections).
///
/// One head's scores are `(1024, 1024)` — **4.2 MB**, which fits L3 comfortably.
/// Looping the heads keeps the same FLOPs and the same output, but softmax and
/// `attn.v` then read what `q.k^T` just wrote, while it is still hot.
///
/// The FLOP count is IDENTICAL either way — this moves no arithmetic, only the
/// distance the bytes travel.
///
/// # The refutation
///
/// It measured **0.888x min / 0.925x median** — decisively SLOWER — on an
/// interleaved 12-sample ABBA A/B whose two arms agreed to 1e-6
/// (`examples/vision_arm_ab head_attn`).
///
/// The premise was right and the conclusion inverted. candle's batched matmul
/// **parallelises across the batch dimension**, and for `(1, 12, 1024, 1024)`
/// that dimension *is* the 12 heads. Looping them hands the gemm a single
/// `(1024, 64) @ (64, 1024)` at a time, whose `k = 64` reduction gives the
/// threads far less to divide. The cache locality is real; the parallelism it
/// costs is worth more.
///
/// Kept as a toggle rather than deleted so the refutation stays measurable —
/// and because it is the null arm for anything else that touches this path.
fn head_attn() -> bool {
    HEAD_ATTN.load(std::sync::atomic::Ordering::Relaxed)
}

/// Toggle per-head attention, returning the previous setting. See [`head_attn`].
pub fn set_head_attn(on: bool) -> bool {
    HEAD_ATTN.swap(on, std::sync::atomic::Ordering::Relaxed)
}

static HEAD_ATTN: std::sync::LazyLock<std::sync::atomic::AtomicBool> =
    std::sync::LazyLock::new(|| {
        std::sync::atomic::AtomicBool::new(arm_flag("FFAI_ARGUS_HEAD_ATTN", false))
    });

/// Is bias fusion on? `FFAI_ARGUS_FUSE_BIAS=0` takes the unfused path.
///
/// This exists so the win can be MEASURED. This box has shown a 1.76x spread
/// for identical code, so a before/after stopwatch across two builds decides
/// nothing. With both arms in one binary the A/B can be interleaved in a single
/// process, where drift hits both arms equally.
///
/// The `false` arm is not a stub: it applies every bias the ordinary way
/// (`broadcast_add`), so the two arms compute the same thing and the only
/// difference is how many passes over memory it takes.
fn fuse_bias() -> bool {
    FUSE_BIAS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Toggle bias fusion, returning the previous setting.
///
/// An atomic rather than a `OnceLock` for one reason: an A/B that runs the two
/// arms in SEPARATE PROCESSES measures the box's drift as much as the change.
/// Flipping this between calls puts both arms in one process, interleaved, so
/// a thermal or contention swing lands on both.
pub fn set_fuse_bias(on: bool) -> bool {
    FUSE_BIAS.swap(on, std::sync::atomic::Ordering::Relaxed)
}

static FUSE_BIAS: std::sync::LazyLock<std::sync::atomic::AtomicBool> =
    std::sync::LazyLock::new(|| {
        std::sync::atomic::AtomicBool::new(arm_flag("FFAI_ARGUS_FUSE_BIAS", true))
    });

/// The packed q/k/v copy, with `qkv`'s bias folded in.
///
/// `(b, seq, 3*hidden) -> (3, b, heads, seq, head_dim)`, adding the bias on the
/// way through.
///
/// # Why this is free
///
/// The permute already forces a full copy — candle's `.contiguous()` reads and
/// writes every one of `3*b*seq*hidden` elements (9.4 MB per layer). The bias
/// add was a SECOND pass over the same 9.4 MB, and at the tower's shapes it is
/// the most expensive of the four: **1.81 ms per layer, 369 ms per caption**.
/// Doing the add inside the copy costs nothing — for any fixed `(g, b, h)` the
/// bias slice is 64 contiguous floats reused for all 1024 rows, so it is read
/// from L1 every time.
///
/// # Layout
///
/// The output is written strictly in order, so the destination stream is
/// sequential. The source is read in `head_dim`-wide runs strided by
/// `3*hidden` — the same access pattern the permute performed anyway.
struct PackedQkvOp {
    heads: usize,
    head_dim: usize,
}

impl candle_core::CustomOp2 for PackedQkvOp {
    fn name(&self) -> &'static str {
        "ffai-packed-qkv"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        let (x, bias) = match (s1, s2) {
            (candle_core::CpuStorage::F32(a), candle_core::CpuStorage::F32(b)) => (a, b),
            _ => candle_core::bail!("ffai-packed-qkv expects f32"),
        };
        let (Some((xo, xe)), Some((bo, be))) = (l1.contiguous_offsets(), l2.contiguous_offsets())
        else {
            candle_core::bail!("ffai-packed-qkv expects contiguous inputs")
        };
        let (x, bias) = (&x[xo..xe], &bias[bo..be]);
        let (b, seq, wide) = l1.shape().dims3()?;
        let (heads, hd) = (self.heads, self.head_dim);
        let hidden = heads * hd;
        if wide != 3 * hidden || bias.len() != wide {
            candle_core::bail!("ffai-packed-qkv: {wide} != 3*{hidden}, bias {}", bias.len());
        }
        let n = 3 * b * hidden * seq;
        let mut out: Vec<f32> = Vec::with_capacity(n);
        {
            let spare = out.spare_capacity_mut();
            // SAFETY: exactly `n` contiguous `MaybeUninit<f32>`. The loop below
            // writes each of the `3*b*heads` chunks of `seq*hd` exactly once and
            // together they tile `[0, n)`, so every element is initialized
            // before `set_len`. `f32` has no invalid bit patterns, no drop glue.
            #[allow(unsafe_code)]
            let dst: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n) };
            crate::cost::elementwise(n as u64, 1, 1);
            // One chunk per (group, batch, head): its bias slice is 64 floats,
            // read once into L1 and reused for all `seq` rows.
            let chunk = |(c, o): (usize, &mut [f32])| {
                let (g, rem) = (c / (b * heads), c % (b * heads));
                let (bb, h) = (rem / heads, rem % heads);
                let col = g * hidden + h * hd;
                let bias = &bias[col..col + hd];
                let row0 = bb * seq * wide + col;
                for (s, o) in o.chunks_mut(hd).enumerate() {
                    let src = &x[row0 + s * wide..row0 + s * wide + hd];
                    for ((o, &v), &c) in o.iter_mut().zip(src).zip(bias) {
                        *o = v + c;
                    }
                }
            };
            if kernels_parallel() {
                dst.par_chunks_mut(seq * hd).enumerate().for_each(chunk);
            } else {
                dst.chunks_mut(seq * hd).enumerate().for_each(chunk);
            }
        }
        // SAFETY: every element written above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(n);
        }
        Ok((candle_core::CpuStorage::F32(out), (3, b, heads, seq, hd).into()))
    }
}

/// Split a loaded `Linear` into a bias-free one and its bias.
///
/// The bias is not dropped — it is handed to whichever kernel already touches
/// every element downstream ([`GeluBiasOp`], [`AddBiasOp`], [`PackedQkvOp`]),
/// which absorbs it for free. Loading the projection once and splitting is what
/// keeps this from doubling the weight read at construction.
fn split_bias(l: Linear) -> (Linear, Option<Tensor>) {
    let bias = l.bias().cloned();
    (Linear::new(l.weight().clone(), None), bias)
}

/// `residual + projection + bias` — three tensors, ONE pass.
///
/// # Why this is free
///
/// `out_proj` and `fc2` are each followed immediately by a residual add. Left
/// as `Linear::forward` + `+`, that is **two** full passes over the activation:
/// one to broadcast the bias in, one to add the residual. Both read and write
/// every element of a 3.1 MB tensor, and candle's binary ops are single-thread.
///
/// Folding the bias into the residual makes it one pass. The bias vector is 768
/// floats — it stays in L1 for the whole kernel, so the extra add is free
/// against the memory traffic already being paid.
///
/// Measured at the tower's shapes, the two biases this deletes cost
/// `1.13 + 0.23 = 1.36 ms` per layer, or **277 ms per caption** across 12
/// layers and 17 tiles.
///
/// Arithmetically it is `(r + p) + b` reassociated to `r + (p + b)`. Float
/// addition is not associative, so this is float-close, not bit-identical —
/// gated on the reference caption, not on a byte compare.
struct AddBiasOp;

impl candle_core::CustomOp3 for AddBiasOp {
    fn name(&self) -> &'static str {
        "ffai-add-bias"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
        s3: &candle_core::CpuStorage,
        l3: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        let (a, b, bias) = match (s1, s2, s3) {
            (
                candle_core::CpuStorage::F32(a),
                candle_core::CpuStorage::F32(b),
                candle_core::CpuStorage::F32(c),
            ) => (a, b, c),
            _ => candle_core::bail!("ffai-add-bias expects f32"),
        };
        let (Some((ao, ae)), Some((bo, be)), Some((co, ce))) = (
            l1.contiguous_offsets(),
            l2.contiguous_offsets(),
            l3.contiguous_offsets(),
        ) else {
            candle_core::bail!("ffai-add-bias expects contiguous inputs")
        };
        let (a, b, bias) = (&a[ao..ae], &b[bo..be], &bias[co..ce]);
        let width = bias.len();
        if width == 0 || a.len() != b.len() || a.len() % width != 0 {
            candle_core::bail!("ffai-add-bias: {} vs {} width {width}", a.len(), b.len());
        }
        let n = a.len();
        let mut out: Vec<f32> = Vec::with_capacity(n);
        {
            let spare = out.spare_capacity_mut();
            // SAFETY: exactly `n` contiguous `MaybeUninit<f32>`; every element is
            // written below before `set_len` publishes them, and `f32` has no
            // invalid bit patterns and no drop glue.
            #[allow(unsafe_code)]
            let dst: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n) };
            crate::cost::elementwise(n as u64, 2, 1);
            let row = |((o, x), y): ((&mut [f32], &[f32]), &[f32])| {
                for (((o, &x), &y), &c) in o.iter_mut().zip(x).zip(y).zip(bias) {
                    *o = x + (y + c);
                }
            };
            if kernels_parallel() {
                dst.par_chunks_mut(width)
                    .zip(a.par_chunks(width))
                    .zip(b.par_chunks(width))
                    .for_each(row);
            } else {
                dst.chunks_mut(width).zip(a.chunks(width)).zip(b.chunks(width)).for_each(row);
            }
        }
        // SAFETY: every element written above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(n);
        }
        Ok((candle_core::CpuStorage::F32(out), l1.shape().clone()))
    }
}

/// `gelu(x + bias)` — the bias folded into the activation, for free.
///
/// # Why this is free
///
/// `candle_nn::Linear::forward` is a matmul followed by
/// `broadcast_add(bias)`, and candle's binary ops are not parallel. Measured
/// at the tower's own shapes, that add costs:
///
/// | projection | bias add | x12 layers x17 tiles |
/// |---|---:|---:|
/// | qkv `768->2304` | 1.81 ms | 369 ms |
/// | **fc1 `768->3072`** | **1.75 ms** | **356 ms** |
/// | fc2 `3072->768` | 1.13 ms | 230 ms |
/// | out `768->768` | 0.23 ms | 47 ms |
///
/// — about **1.0 s of an ~18.3 s tower, 5.5 %**, spent on a pass that reads and
/// writes 12.6 MB to add one number per column. GELU *already* reads and writes
/// every one of those elements, so doing the add inside it costs nothing: the
/// bias vector is 3072 floats and stays in L1 for the whole kernel.
///
/// It is also exactly as accurate: `gelu(x + b)` is what the two-op form
/// computes, in the same order, with the same rounding.
struct GeluBiasOp;

impl candle_core::CustomOp2 for GeluBiasOp {
    fn name(&self) -> &'static str {
        "ffai-gelu-bias"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        let (x, bias) = match (s1, s2) {
            (candle_core::CpuStorage::F32(a), candle_core::CpuStorage::F32(b)) => (a, b),
            _ => candle_core::bail!("ffai-gelu-bias expects f32"),
        };
        let (Some((xo, xe)), Some((bo, be))) = (l1.contiguous_offsets(), l2.contiguous_offsets())
        else {
            candle_core::bail!("ffai-gelu-bias expects contiguous inputs")
        };
        let (x, bias) = (&x[xo..xe], &bias[bo..be]);
        let width = bias.len();
        if width == 0 || x.len() % width != 0 {
            candle_core::bail!("ffai-gelu-bias: {} not divisible by {width}", x.len());
        }
        let n = x.len();
        let mut out: Vec<f32> = Vec::with_capacity(n);
        {
            let spare = out.spare_capacity_mut();
            // SAFETY: exactly `n` contiguous `MaybeUninit<f32>`; every element
            // is written by the loop below before `set_len` publishes them, and
            // `f32` has no invalid bit patterns and no drop glue.
            #[allow(unsafe_code)]
            let dst: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n) };
            crate::cost::elementwise(n as u64, 1, 1);
            let row = |(o, i): (&mut [f32], &[f32])| {
                for ((o, &v), &b) in o.iter_mut().zip(i).zip(bias) {
                    *o = v + b;
                }
                // `fastmath` dispatches AVX2 / wasm SIMD128 / scalar. The
                // block this replaces had an x86_64 arm and NOTHING for wasm,
                // so the browser took the scalar loop for an activation that
                // runs over `seq * 4 * hidden` elements per layer — the same
                // ISA asymmetry the softmax had.
                ffai_core::fastmath::gelu_tanh_inplace(o);
            };
            if kernels_parallel() {
                dst.par_chunks_mut(width).zip(x.par_chunks(width)).for_each(row);
            } else {
                dst.chunks_mut(width).zip(x.chunks(width)).for_each(row);
            }
        }
        // SAFETY: every element written above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(n);
        }
        Ok((candle_core::CpuStorage::F32(out), l1.shape().clone()))
    }
}

/// `gelu_pytorch_tanh` as a **zero-copy** candle op.
///
/// # Why `CustomOp1` and not a `Vec` round trip
///
/// The previous version did `xs.flatten_all()?.to_vec1::<f32>()?`, computed,
/// and rebuilt with `Tensor::from_vec`. That extraction is a real copy — 12.6 MB
/// per call at `(1,1024,3072)`, and the deterministic cost model counted it:
/// **12 copies and 302 MB per tile**, purely to hand candle's own bytes back to
/// candle.
///
/// `CustomOp1::cpu_fwd` receives the `CpuStorage` directly, so the kernel reads
/// the tensor's memory in place and returns new storage once. The marshalling
/// disappears rather than getting faster.
///
/// This is the `codec-vectorize-kernel` lesson stated as a law: *"if you
/// benchmark a hand kernel through copy-in/copy-out glue, you are benchmarking
/// the glue"* — one f16 GEMV there measured 0.946x through marshalling and
/// 1.144x through the zero-copy hook, **the same arithmetic**.
struct GeluOp;

impl candle_core::CustomOp1 for GeluOp {
    fn name(&self) -> &'static str {
        "ffai-gelu-tanh"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        let src = match storage {
            candle_core::CpuStorage::F32(v) => v,
            _ => candle_core::bail!("ffai-gelu expects f32"),
        };
        // `contiguous_offsets` is Some only when the layout really is a flat
        // run; a strided view would otherwise be read as if it were dense,
        // which is silent corruption rather than an error.
        let Some((o, end)) = layout.contiguous_offsets() else {
            candle_core::bail!("ffai-gelu expects a contiguous input")
        };
        let src = &src[o..end];
        // `vec![0.0; n]` then fill, NOT `src.to_vec()` then modify in place.
        // The copy-first form reads and writes the whole tensor before the
        // kernel has done anything — two extra passes. Measured on softmax,
        // where the tensor is 50 MB, it cost 25x.
        // `Vec::with_capacity` + `set_len`, NOT `vec![0.0; n]`.
        //
        // Every element is written by the kernel below, so zero-initialising
        // first is a whole extra pass over the buffer that is then discarded —
        // 12.6 MB per call, 12 calls per tile, **2.6 GB of pure memset per
        // image**. This is `rusty-unsafe-optimizations` pattern 2 (uninitialised
        // output buffers) applied to the one place in this file that qualifies.
        let n = src.len();
        let mut out: Vec<f32> = Vec::with_capacity(n);
        {
            // Write through `spare_capacity_mut` rather than `set_len`-then-fill.
            //
            // Both avoid the discarded zero-init pass, but only this one is
            // sound to hold: `set_len` immediately after `with_capacity` makes
            // a `&mut [f32]` over uninitialised memory, which is UB the moment
            // anything reads it — clippy's `uninit_vec` says so, correctly.
            // Here the `&mut [f32]` is created from the SPARE region and every
            // element is written by `copy_from_slice` before `set_len` publishes
            // it, so no reader ever sees an uninitialised value.
            let spare = out.spare_capacity_mut();
            // SAFETY: `spare` is exactly `n` contiguous `MaybeUninit<f32>`.
            // `f32` has no invalid bit patterns and no drop glue, and every one
            // of the n elements is written by the partitioned `copy_from_slice`
            // below before `set_len` is called — so nothing observes an
            // uninitialised value, and nothing is dropped on unwind.
            #[allow(unsafe_code)]
            let dst: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n) };
            crate::cost::elementwise(n as u64, 1, 1);
            fill_gelu(dst, src, kernels_parallel());
        }
        // SAFETY: `fill_gelu` wrote all n elements above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(n);
        }
        Ok((candle_core::CpuStorage::F32(out), layout.shape().clone()))
    }
}

/// Fill `dst` with `gelu(src)`. `dst` and `src` must be the same length.
fn fill_gelu(dst: &mut [f32], src: &[f32], parallel: bool) {
    debug_assert_eq!(dst.len(), src.len());
    // `fastmath::gelu_tanh_inplace` carries the AVX2 kernel this used to call
    // AND a wasm SIMD128 twin. The block it replaces dispatched on
    // `have_avx2_cached()`, which is `false` by construction off x86_64, so
    // every browser ran `gelu_chunk_scalar` over `seq * 4 * hidden` elements
    // per layer. Same ISA asymmetry the softmax had.
    let kernel = |(o, i): (&mut [f32], &[f32])| {
        o.copy_from_slice(i);
        ffai_core::fastmath::gelu_tanh_inplace(o);
    };
    if parallel {
        dst.par_chunks_mut(8192).zip(src.par_chunks(8192)).for_each(kernel);
    } else {
        dst.chunks_mut(8192).zip(src.chunks(8192)).for_each(kernel);
    }
}

/// GELU over a tensor, zero-copy.
///
/// # Errors
/// If the input is not contiguous f32.
pub fn gelu_tanh_par(xs: &Tensor) -> CandleResult<Tensor> {
    xs.apply_op1_no_bwd(&GeluOp)
}

/// ⚠ **REFUTED THREE TIMES — candle's `softmax_last_dim` wins. Kept only as
/// the oracle the parity test compares against.**
///
/// | attempt | serial | parallel |
/// |---|---:|---:|
/// | 1. `Vec` round trip + scalar `expf` | 0.09x | 0.40x |
/// | 2. zero-copy `CustomOp1` + polynomial `exp` | 0.05x | 0.25x |
/// | 3. + `target_feature` dispatch | 0.06x | 0.71x |
/// | 4. + split accumulator to break the reduction chain | **0.04x** | 0.54x |
///
/// Each attempt fixed a REAL defect — the marshalling, the libm call, the
/// missing `target_feature`, the loop-carried `sum` — and the op still loses by
/// more than 10x. That is the three-probe refutation rule doing its job: the
/// idea is dead, and the reason is that candle's softmax is genuinely good
/// (3.8 ms for a 50 MB tensor is ~40 GB/s, L3-bandwidth territory).
///
/// The contrast with GELU is the lesson. Same polynomial, same dispatch, same
/// zero-copy hook — **39x on GELU, 0.5x here** — because candle's `tanh` GELU
/// was a scalar libm call per element and its softmax was not. *Replace
/// expensive arithmetic; do not re-implement a good kernel.*
///
/// Original doc follows.
///
/// Softmax over the last dimension, zero-copy and AVX2.
///
/// # Why revisit this — round 2 measured ours as 0.09x and reverted
///
/// That refutation was real but incomplete. The kernel lost for two reasons and
/// only one of them was the kernel: it round-tripped a **50 MB** tensor through
/// a `Vec`, and it computed `exp` with a scalar libm call. Fixing the glue and
/// the transcendental separately is what the earlier note could not distinguish
/// — exactly the correction `codec-vectorize-kernel` records against itself
/// (*"my delivery was too expensive and the ratio is too small are different
/// claims, and the first masquerades as the second"*).
///
/// So this is the same idea delivered properly: storage in place, and
/// [`exp_poly`] instead of `expf`. Softmax is the largest elementwise term
/// left — **151 M element-visits per tile**, against GELU's 37.7 M.
///
/// The per-row arithmetic is unchanged (max, subtract, exp, sum, scale), so the
/// reduction order within a row is identical to a serial implementation.
struct SoftmaxLastDimOp {
    last: usize,
}

impl candle_core::CustomOp1 for SoftmaxLastDimOp {
    fn name(&self) -> &'static str {
        "ffai-softmax-last-dim"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        let src = match storage {
            candle_core::CpuStorage::F32(v) => v,
            _ => candle_core::bail!("ffai-softmax expects f32"),
        };
        let Some((o, end)) = layout.contiguous_offsets() else {
            candle_core::bail!("ffai-softmax expects a contiguous input")
        };
        let src = &src[o..end];
        let mut out = vec![0.0f32; src.len()];
        let n = self.last;
        crate::cost::elementwise(out.len() as u64, 2, 1);
        // src -> out, one read and one write; the row never round-trips.
        // Dispatched exactly like GELU: without `target_feature`, `exp_poly`'s
        // `round()` becomes a libm CALL and the kernel loses to candle by 20x.
        let avx2 = cfg!(target_arch = "x86_64") && have_avx2_cached();
        let kernel = move |(dst, row): (&mut [f32], &[f32])| {
            #[cfg(target_arch = "x86_64")]
            if avx2 {
                // SAFETY: `have_avx2_cached()` returned true above.
                unsafe { softmax_row_avx2(dst, row) };
                return;
            }
            softmax_row_scalar(dst, row);
        };
        if kernels_parallel() {
            out.par_chunks_mut(n).zip(src.par_chunks(n)).for_each(kernel);
        } else {
            out.chunks_mut(n).zip(src.chunks(n)).for_each(kernel);
        }
        Ok((candle_core::CpuStorage::F32(out), layout.shape().clone()))
    }
}

/// Softmax over the last dim, zero-copy.
///
/// # Errors
/// If the input is not contiguous f32.
pub fn softmax_last_dim_ours(xs: &Tensor) -> CandleResult<Tensor> {
    let last = xs.dims()[xs.rank() - 1];
    xs.apply_op1_no_bwd(&SoftmaxLastDimOp { last })
}

/// The vision tower: embeddings, encoder layers, post-layernorm.
pub struct VisionTower {
    embeddings: Embeddings,
    layers: Vec<Layer>,
    post_ln: LayerNorm,
}

impl VisionTower {
    /// Load from the same prefix candle's `VisionModel` uses.
    ///
    /// # Errors
    /// Propagates candle's load errors; a missing tensor names itself.
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> CandleResult<Self> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::load(cfg, &vb.pp(format!("encoder.layers.{i}"))))
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Self {
            embeddings: Embeddings::new(cfg, vb.pp("embeddings"))?,
            layers,
            post_ln: candle_nn::layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("post_layernorm"),
            )?,
        })
    }

    /// `(b, 3, H, W)` -> `(b, patches, hidden)`, the reference's
    /// `last_hidden_state`.
    ///
    /// # Errors
    /// Propagates candle's errors.
    pub fn forward(&self, pixel_values: &Tensor) -> CandleResult<Tensor> {
        let _t = crate::clock::Instant::now();
        let mut xs = self.embeddings.forward(pixel_values)?;
        prof::add("patch+pos embed", _t);
        for layer in &self.layers {
            let _t = crate::clock::Instant::now();
            let next = layer.forward(&xs)?;
            // Timed OUTSIDE the layer's own probes, so the difference between
            // this and their sum is the layer's untimed remainder — chiefly the
            // drop of ~165 MB of intermediates per layer. That remainder was
            // 12 % of the tower and invisible until it was given a name.
            prof::add("LAYER TOTAL", _t);
            xs = next;
        }
        let _t = crate::clock::Instant::now();
        let out = layer_norm(&self.post_ln, &xs);
        prof::add("post_ln", _t);
        out
    }
}

#[cfg(test)]
mod tests {
    /// A stride-N, kernel-N convolution IS a matmul over permuted patches.
    ///
    /// `Embeddings::forward` relies on that identity to replace candle's
    /// `conv2d` (113 GF/s) with a product (291 GF/s). If it ever stops holding
    /// — a padded or dilated config, a stride that is not the kernel — the
    /// embedding would silently produce a plausible wrong picture, which is
    /// the failure mode this crate has already paid for once at the resampler
    /// (§16). The `Conv2d` that used to sit beside it as a "fallback" was
    /// never reachable; this test is what replaces it.
    #[test]
    fn a_non_overlapping_conv_is_a_matmul_over_permuted_patches() {
        use candle_core::{Device, Module, Tensor};
        let d = Device::Cpu;
        // Deliberately not square-and-round: 2 batch, 3 channels, 6x8 grid.
        let (b, c, p, gh, gw, out) = (2usize, 3usize, 4usize, 6usize, 8usize, 5usize);
        let px = Tensor::rand(-1.0f32, 1.0, (b, c, gh * p, gw * p), &d).expect("px");
        let w = Tensor::rand(-1.0f32, 1.0, (out, c, p, p), &d).expect("w");
        let bias = Tensor::rand(-1.0f32, 1.0, out, &d).expect("bias");

        let conv = candle_nn::Conv2d::new(
            w.clone(),
            Some(bias.clone()),
            candle_nn::Conv2dConfig { stride: p, ..Default::default() },
        );
        let want = conv
            .forward(&px)
            .expect("conv")
            .flatten_from(2)
            .expect("flat")
            .transpose(1, 2)
            .expect("t")
            .contiguous()
            .expect("c");

        let w_flat = w
            .reshape((out, c * p * p)).expect("wr")
            .t().expect("wt")
            .contiguous().expect("wc");
        let got = px
            .reshape((b, c, gh, p, gw, p)).expect("r")
            .permute((0, 2, 4, 1, 3, 5)).expect("perm")
            .contiguous().expect("c")
            .reshape((b * gh * gw, c * p * p)).expect("r2")
            .matmul(&w_flat).expect("mm")
            .broadcast_add(&bias).expect("bias")
            .reshape((b, gh * gw, out)).expect("r3");

        assert_eq!(want.dims(), got.dims(), "shape");
        let (a, e) = (
            want.flatten_all().expect("f").to_vec1::<f32>().expect("v"),
            got.flatten_all().expect("f").to_vec1::<f32>().expect("v"),
        );
        let worst = a.iter().zip(&e).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        // Float reassociation only — a WRONG permutation differs by O(1), not
        // by an ulp, so this tolerance still catches a transposed patch grid.
        assert!(worst < 1e-4, "conv and matmul disagree by {worst:.3e}");
    }

    use super::*;

    /// The four fused-bias kernels against the two-op form they replace.
    ///
    /// Each fusion is only legitimate if `f(x + bias)` computed inside the
    /// kernel equals `f(broadcast_add(x, bias))` computed the ordinary way. The
    /// tolerance is float-close rather than exact because folding reassociates
    /// the additions — that is the whole point of the change — but the two must
    /// agree far more tightly than any tolerance the caption gate cares about.
    mod fused_bias {
        use super::*;

        fn rnd(shape: (usize, usize, usize)) -> Tensor {
            Tensor::rand(-2.0f32, 2.0, shape, &Device::Cpu).expect("rand")
        }

        fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
            (a - b)
                .expect("sub")
                .abs()
                .expect("abs")
                .max_all()
                .expect("max")
                .to_scalar::<f32>()
                .expect("scalar")
        }


        /// The fused embedding add against the two chained `broadcast_add`s.
        ///
        /// `assert_eq!` on every element, not a tolerance. The kernel adds in
        /// the SAME order the two ops did — `(x + bias) + position` — so there
        /// is no reassociation to excuse a difference. Folding
        /// `bias + position` at load time would have been faster still and is
        /// exactly what this test would have caught: it changes the rounding.
        #[test]
        fn embed_add_matches_the_two_broadcast_adds() {
            let d = Device::Cpu;
            for (b, seq, hidden) in [(1usize, 16usize, 8usize), (2, 9, 5), (1, 4, 3)] {
                let xs = Tensor::rand(-2.0f32, 2.0, (b * seq, hidden), &d).expect("rand");
                let bias = Tensor::rand(-1.0f32, 1.0, hidden, &d).expect("rand");
                let pos = Tensor::rand(-1.0f32, 1.0, (seq, hidden), &d).expect("rand");

                let fused = xs.apply_op3_no_bwd(&bias, &pos, &EmbedAddOp).expect("fused");
                let want = xs
                    .broadcast_add(&bias)
                    .expect("bias")
                    .reshape((b, seq, hidden))
                    .expect("reshape")
                    .broadcast_add(&pos)
                    .expect("pos");

                assert_eq!(fused.dims(), want.dims(), "shape at b{b} seq{seq}");
                let got = fused.flatten_all().expect("f").to_vec1::<f32>().expect("v");
                let want = want.flatten_all().expect("f").to_vec1::<f32>().expect("v");
                assert_eq!(got, want, "embed add at b{b} seq{seq} hidden{hidden}");
            }
        }

        #[test]
        fn gelu_bias_matches_gelu_of_broadcast_add() {
            let (x, bias) = (rnd((1, 64, 128)), rnd((1, 1, 128)).flatten_all().expect("flat"));
            let fused = x.apply_op2_no_bwd(&bias, &GeluBiasOp).expect("fused");
            let plain =
                gelu_tanh_par(&x.broadcast_add(&bias).expect("add")).expect("gelu");
            let d = max_abs_diff(&fused, &plain);
            assert!(d < 1e-6, "gelu(x+b) fused vs two-op differ by {d:e}");
        }

        #[test]
        fn add_bias_matches_residual_plus_broadcast_add() {
            let (r, y) = (rnd((1, 64, 128)), rnd((1, 64, 128)));
            let bias = rnd((1, 1, 128)).flatten_all().expect("flat");
            let fused = r.apply_op3_no_bwd(&y, &bias, &AddBiasOp).expect("fused");
            let plain = (&r + y.broadcast_add(&bias).expect("add")).expect("sum");
            let d = max_abs_diff(&fused, &plain);
            assert!(d < 1e-5, "r+(y+b) fused vs two-op differ by {d:e}");
        }

        /// The packed copy must reproduce candle's permute EXACTLY — it is a
        /// pure data movement, so unlike the arithmetic fusions there is no
        /// reassociation to excuse a difference. Anything but zero here is a
        /// layout bug, and a layout bug in this kernel silently scrambles which
        /// head reads which weights.
        #[test]
        fn packed_qkv_matches_the_permute_it_replaces() {
            let (heads, hd, seq, b) = (4usize, 8usize, 16usize, 2usize);
            let hidden = heads * hd;
            let x = Tensor::rand(-2.0f32, 2.0, (b, seq, 3 * hidden), &Device::Cpu).expect("rand");
            let bias = Tensor::rand(-1.0f32, 1.0, 3 * hidden, &Device::Cpu).expect("rand");

            let fused = x
                .apply_op2_no_bwd(&bias, &PackedQkvOp { heads, head_dim: hd })
                .expect("fused");
            let plain = x
                .broadcast_add(&bias)
                .expect("add")
                .reshape((b, seq, 3, heads, hd))
                .expect("reshape")
                .permute((2, 0, 3, 1, 4))
                .expect("permute")
                .contiguous()
                .expect("contig");

            assert_eq!(fused.dims(), plain.dims(), "packed shape");
            let d = max_abs_diff(&fused, &plain);
            assert_eq!(d, 0.0, "packed copy is pure movement; differs by {d:e}");
        }

        /// Deferring softmax's divide past `attn.v` must land in the same place
        /// as dividing first. This is the algebra the optimisation rests on:
        /// `SUM_j (p_ij / S_i) v_j  ==  (SUM_j p_ij v_j) / S_i`.
        #[test]
        fn deferred_normalisation_matches_softmax_then_matmul() {
            let (heads, hd, seq, b) = (3usize, 8usize, 32usize, 2usize);
            let d = Device::Cpu;
            let scores = Tensor::rand(-4.0f32, 4.0, (b, heads, seq, seq), &d).expect("rand");
            let v = Tensor::rand(-1.0f32, 1.0, (b, heads, seq, hd), &d).expect("rand");

            // The ordinary way: normalise, matmul, transpose.
            let want = candle_nn::ops::softmax_last_dim(&scores)
                .expect("softmax")
                .matmul(&v)
                .expect("av")
                .transpose(1, 2)
                .expect("t")
                .reshape((b, seq, heads * hd))
                .expect("reshape");

            // Ours: exp in place, matmul, then normalise inside the transpose.
            let exps = scores.copy().expect("copy");
            let rows = b * heads * seq;
            let mut sums = vec![0f32; rows];
            exps.inplace_op1(&SoftmaxExpInplace { sums: RowSums(sums.as_mut_ptr()), rows })
                .expect("exp");
            let sums = Tensor::from_vec(sums, (b, heads, seq), &d).expect("sums");
            let got = exps
                .matmul(&v)
                .expect("av")
                .apply_op2_no_bwd(&sums, &AttnMergeOp { heads, head_dim: hd })
                .expect("merge");

            assert_eq!(got.dims(), want.dims(), "merged shape");
            let e = max_abs_diff(&got, &want);
            assert!(e < 1e-6, "deferred normalisation differs by {e:e}");
        }

        /// Every row total must be at least 1: the row max contributes
        /// `exp(0) = 1`. That is what guarantees `AttnMergeOp` can never divide
        /// by zero or by a subnormal, which is the safety argument the deferred
        /// normalisation rests on.
        #[test]
        fn row_sums_are_never_below_one() {
            let d = Device::Cpu;
            // Deliberately extreme: a row of large negatives would underflow to
            // zero if the max were not subtracted first.
            let scores = Tensor::rand(-90.0f32, -80.0, (1, 2, 8, 8), &d).expect("rand");
            let rows = 1 * 2 * 8;
            let mut sums = vec![0f32; rows];
            scores
                .inplace_op1(&SoftmaxExpInplace { sums: RowSums(sums.as_mut_ptr()), rows })
                .expect("exp");
            for (i, &s) in sums.iter().enumerate() {
                assert!(s >= 1.0, "row {i} total {s} < 1 — max was not subtracted");
                assert!(s.is_finite(), "row {i} total {s} is not finite");
            }
        }
    }
    use candle_core::{DType, Device};

    fn dev() -> Device {
        Device::Cpu
    }

    #[test]
    /// The refuted softmax is still CORRECT — it was reverted for speed, not
    /// for wrongness. Kept as an independent check that candle's softmax does
    /// what we think: if the two ever disagree, one of them changed.
    fn parallel_softmax_matches_candles() {
        let d = dev();
        // A shape with the awkward properties: several rows, values spread far
        // enough that a missing max-subtract would overflow.
        let xs = Tensor::rand(-30.0f32, 30.0, (2, 3, 7, 65), &d).expect("xs");
        let ours = softmax_last_dim_ours(&xs).expect("ours");
        let theirs = candle_nn::ops::softmax_last_dim(&xs).expect("candle");
        let worst = (&ours - &theirs)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max")
            .to_scalar::<f32>()
            .expect("scalar");
        assert!(worst < 1e-6, "parallel softmax differs by {worst:.3e}");

        // And it is a softmax: every row sums to one.
        let sums = ours.sum(D::Minus1).expect("sum").flatten_all().expect("f");
        for s in sums.to_vec1::<f32>().expect("v") {
            assert!((s - 1.0).abs() < 1e-5, "row sums to {s}");
        }
    }

    #[test]
    fn parallel_gelu_matches_candles() {
        let d = dev();
        let xs = Tensor::rand(-8.0f32, 8.0, (4, 1000), &d).expect("xs");
        let ours = gelu_tanh_par(&xs).expect("ours");
        let theirs = xs.gelu().expect("candle");
        let worst = (&ours - &theirs)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max")
            .to_scalar::<f32>()
            .expect("scalar");
        // The identity is exact; the gap is f32 rounding of one extra
        // operation, and it must stay orders below the 2.06e-4 that §16
        // measured flips no token.
        assert!(worst < 1e-5, "parallel gelu differs by {worst:.3e}");
    }

    /// Shape properties, checked independently of candle.
    ///
    /// The first version of this test asserted GELU was monotone and **failed**
    /// — correctly. GELU is *not* monotone: it dips below zero on the negative
    /// side, bottoming out near `x = -0.75` at about `-0.17`, before rising
    /// back to ~0. `gelu(-10)` is therefore GREATER than `gelu(-2)`.
    ///
    /// Kept as the non-obvious property rather than deleted, because "it goes
    /// negative and comes back" is exactly the part of the curve a bad
    /// approximation gets wrong, and the parity test against candle would not
    /// notice if both were wrong the same way.
    /// The kernel's non-negotiable: **the SIMD twin is gated against the
    /// scalar oracle, in a test, not only in a probe.**
    ///
    /// A benchmark that compares them proves they are both fast. Only a test
    /// keeps them equal as the code changes, and the scalar path is the one
    /// that runs on every CPU without AVX2 — so a divergence would be a
    /// machine-dependent wrong answer, which is the worst kind.
    ///
    /// Bit-identity is asserted rather than a tolerance, because it held: LLVM
    /// did not contract the Horner chains differently between the two, so the
    /// AVX2 twin returns exactly the scalar result. If a future compiler fuses
    /// one and not the other this will fail loudly, which is the correct
    /// outcome — it is a real change in what the engine computes.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn gelu_avx2_matches_scalar() {
        if !super::have_avx2_cached() {
            eprintln!("SKIP: no AVX2 on this CPU — the scalar path is the only one");
            return;
        }
        // Values that matter: the near-zero region, the negative dip, the
        // saturation shoulders, and past the clamp on both sides. Plus a tail
        // length that is NOT a multiple of 8, so the vector remainder is
        // exercised rather than assumed.
        let mut src: Vec<f32> = Vec::new();
        for i in 0..4099 {
            src.push((i as f32 - 2049.0) / 97.0); // spans about -21 .. 21
        }
        src.extend_from_slice(&[0.0, -0.0, 1e-30, -1e-30, 88.0, -88.0, 120.0, -120.0]);
        assert_ne!(src.len() % 8, 0, "the tail must not be a whole vector");

        let (mut a, mut b) = (src.clone(), src.clone());
        super::gelu_chunk_scalar(&mut a);
        // SAFETY: guarded by the `have_avx2_cached()` check above.
        unsafe { super::gelu_chunk_avx2(&mut b) };

        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "lane {i} (input {}) diverged: scalar {x} vs avx2 {y}",
                src[i]
            );
        }
    }

    #[test]
    fn gelu_is_still_gelu_at_the_landmarks() {
        let d = dev();
        let xs = Tensor::from_vec(vec![-10.0f32, -2.0, -0.75, 0.0, 0.5, 2.0, 10.0], 7, &d)
            .expect("xs");
        let g = gelu_tanh_par(&xs).expect("g").to_vec1::<f32>().expect("v");
        assert!(g[0].abs() < 1e-6, "gelu(-10) should vanish, got {}", g[0]);
        assert!(g[3].abs() < 1e-9, "gelu(0) must be exactly 0, got {}", g[3]);
        assert!((g[6] - 10.0).abs() < 1e-4, "gelu(10) ~= 10, got {}", g[6]);
        // The dip: negative, and near the known minimum.
        assert!(
            (-0.18..-0.15).contains(&g[2]),
            "gelu(-0.75) should sit near the -0.17 minimum, got {}",
            g[2]
        );
        assert!(g[1] < 0.0, "gelu(-2) should be negative, got {}", g[1]);
        // Monotone where it IS monotone: from the minimum upward.
        for w in g[2..].windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "not monotone above the dip: {w:?}");
        }
    }

    #[test]
    fn a_zero_length_tensor_does_not_panic() {
        let d = dev();
        let xs = Tensor::zeros((0, 8), DType::F32, &d).expect("xs");
        assert_eq!(gelu_tanh_par(&xs).expect("gelu").elem_count(), 0);
        assert_eq!(
            softmax_last_dim_ours(&xs).expect("sm").elem_count(),
            0
        );
    }
}
