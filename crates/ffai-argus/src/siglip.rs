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
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};
use candle_transformers::models::siglip::VisionConfig;
use rayon::prelude::*;

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
    patch: Conv2d,
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
        Ok(Self {
            patch,
            position,
            patch_size: cfg.patch_size,
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
        // conv2d IS one of the two things candle parallelises, so it stays.
        let xs = self.patch.forward(pixel_values)?;
        // conv2d as a matmul: each output patch is a (patch*patch*channels)
        // dot product. Counted so the embedding does not vanish from the
        // budget just because candle spells it `conv2d`.
        let (_b2, c_out, oh, ow) = xs.dims4()?;
        let k = (self.patch_size * self.patch_size * 3) as u64;
        crate::cost::matmul(1, (oh * ow) as u64, k, c_out as u64);
        let xs = xs.flatten_from(2)?.transpose(1, 2)?;
        let out = xs.broadcast_add(&self.position)?;
        crate::cost::elementwise((c_out * oh * ow) as u64, 2, 1);
        Ok(out)
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
    qkv: Linear,
    out_proj: Linear,
    ln2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
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

        Ok(Self {
            ln1: candle_nn::layer_norm(hidden, cfg.layer_norm_eps, vb.pp("layer_norm1"))?,
            qkv,
            out_proj: candle_nn::linear(hidden, hidden, attn.pp("out_proj"))?,
            ln2: candle_nn::layer_norm(hidden, cfg.layer_norm_eps, vb.pp("layer_norm2"))?,
            fc1: candle_nn::linear(hidden, cfg.intermediate_size, vb.pp("mlp.fc1"))?,
            fc2: candle_nn::linear(cfg.intermediate_size, hidden, vb.pp("mlp.fc2"))?,
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
        let normed = self.ln1.forward(xs)?;
        // layer_norm: one pass, plus the mean/var reduction over the same data.
        crate::cost::elementwise(bu * sq * hd, 2, 1);
        let qkv = self.qkv.forward(&normed)?;
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
        let packed = qkv
            .reshape((b, seq, 3, self.heads, self.head_dim))?
            .permute((0, 2, 3, 1, 4))?
            .contiguous()?;
        crate::cost::copy(3 * bu * sq * hd);
        let q = packed.i((.., 0))?;
        let k = packed.i((.., 1))?;
        let v = packed.i((.., 2))?;

        // No `* scale` here: it is already in q's weights.
        //
        // What the fold DELETES, exactly: an elementwise multiply over
        // `b*heads*seq*seq` elements, reading and writing the 50 MB score
        // matrix — per layer, per tile. That is the deterministic statement of
        // the win; no stopwatch involved.
        let scores = q.matmul(&k.t()?)?;
        crate::cost::matmul(bu * heads, sq, hdim, sq);
        // candle's, measured 11x faster than ours here — see the note above.
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        // softmax: read the row for the max, read again for exp+sum, write.
        crate::cost::elementwise(bu * heads * sq * sq, 2, 1);
        // candle's softmax uses a vectorized exp (~2.7 G/s measured), NOT the
        // scalar libm path GELU was on (~75 M/s). Counting them under one
        // weight predicted 32 s of transcendentals against a ~16 s tower —
        // the arithmetic failing to close is what exposed the conflation.
        crate::cost::transcendental_vector(bu * heads * sq * sq);
        let attn = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, seq, hidden))?;
        crate::cost::matmul(bu * heads, sq, sq, hdim);
        crate::cost::copy(bu * sq * hd);
        let out = self.out_proj.forward(&attn)?;
        crate::cost::matmul(1, bu * sq, hd, hd);
        let xs = (residual + out)?;
        crate::cost::elementwise(bu * sq * hd, 2, 1);

        // ---- mlp -----------------------------------------------------------
        let residual = &xs;
        let normed = self.ln2.forward(&xs)?;
        crate::cost::elementwise(bu * sq * hd, 2, 1);
        let inter = self.fc1.weight().dims()[0] as u64;
        let h = self.fc1.forward(&normed)?;
        crate::cost::matmul(1, bu * sq, hd, inter);
        let h = gelu_tanh_par(&h)?;
        let down = self.fc2.forward(&h)?;
        crate::cost::matmul(1, bu * sq, inter, hd);
        let out = (residual + down)?;
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

/// Dispatch to the widest GELU this CPU can run. The **only** safe entry point.
#[inline]
fn gelu_chunk(chunk: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if have_avx2_cached() {
            // SAFETY: the check above is the documented precondition.
            unsafe { gelu_chunk_avx2(chunk) };
            return;
        }
    }
    gelu_chunk_scalar(chunk);
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
    let avx2 = cfg!(target_arch = "x86_64") && have_avx2_cached();
    let kernel = move |(o, i): (&mut [f32], &[f32])| {
        o.copy_from_slice(i);
        #[cfg(target_arch = "x86_64")]
        if avx2 {
            // SAFETY: `have_avx2_cached()` returned true above, which is
            // `gelu_chunk_avx2`'s documented precondition.
            unsafe { gelu_chunk_avx2(o) };
            return;
        }
        gelu_chunk_scalar(o);
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
        let mut xs = self.embeddings.forward(pixel_values)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        self.post_ln.forward(&xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
