//! Our own `SmolLM2` text tower — same maths, fewer passes over memory.
//!
//! # Why reimplement something candle already has
//!
//! The same reason `siglip.rs` exists, found the same way. `candle`'s
//! `models::llama` is correct and every gate through §20 was built on it. But
//! `examples/text_scaling` measured the whole tower at **61-90 GF/s** while
//! `examples/gemm_shapes` measured candle's matmul at the tower's own shapes at
//! **516-591 GF/s**. Both cannot describe the same implementation, so the cost
//! is not in the matmuls — and reading `llama.rs` found where it is.
//!
//! At seq 1142 (the caption's prompt), per layer, on the 46.9 MB score matrix:
//!
//! | op in `llama.rs` | ms/layer | x30 layers | rate |
//! |---|---:|---:|---:|
//! | **`masked_fill` (line 346)** | **114.13** | **3424 ms** | **0.8 GB/s** |
//! | `att / sqrt(head_dim)` (line 341) | 19.39 | 582 ms | 4.8 GB/s |
//! | `softmax_last_dim` | 5.95 | 178 ms | 23.7 GB/s |
//! | *(all 30 layers' matmuls, for scale)* | | *~930 ms* | *516-591 GF/s* |
//!
//! `masked_fill` alone is **20 % of an entire caption**, and all it does is
//! write `-inf` into an upper triangle. It costs that much because it is
//! `where_cond` against **two broadcast operands** — a `(S,S)` mask stretched
//! to `(1,9,S,S)` and a scalar stretched the same way — so every one of 11.7 M
//! elements is a strided gather, on one core.
//!
//! # Why deleting the mask is BIT-IDENTICAL, not an approximation
//!
//! Softmax takes the row maximum, and `max(finite, -inf)` never selects
//! `-inf`; then `exp(-inf) = 0` contributes nothing to the sum. So **skipping**
//! a masked column produces exactly the floats that materialising `-inf` and
//! running softmax over it produces. Causality by construction is the same
//! arithmetic with the no-ops removed — which is why this is gated on token
//! equality and passes it, rather than on a tolerance.
//!
//! The scale fold is exact for the same kind of reason: `head_dim` is 64,
//! `sqrt(64)` is 8, and `1/8` is a power of two, so scaling **q** and summing
//! is bit-for-bit the sum then scaled. (`svtr` refused the same fold precisely
//! because *its* scale is **not** a power of two — see
//! `ffai-carmenta/src/svtr.rs`.)
//!
//! # What is NOT changed
//!
//! Every matmul still goes through candle — they are at parity and §19 already
//! refuted touching them. `RoPE` is candle's. This is the same graph with the
//! memory-bound steps rewritten, which is exactly what `siglip.rs` says about
//! itself.

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor, D};
use candle_nn::VarBuilder;
use rayon::prelude::*;

/// Geometry, read from the checkpoint rather than assumed.
#[derive(Debug, Clone, Copy)]
pub struct Cfg {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub eps: f64,
    pub rope_theta: f32,
    pub max_pos: usize,
}

/// One transformer block's weights.
struct Block {
    ln1: Tensor,
    /// **Pre-scaled by `1/sqrt(head_dim)` at load** — this deletes a divide
    /// over 11.7 M elements per layer, and is exact because `1/sqrt(64) = 1/8`
    /// is a power of two, so scaling q then summing is bit-for-bit the sum
    /// then scaled. (`svtr` refuses the same fold because its scale is not.)
    ///
    /// # Kept as three weights, not fused — REFUTED, measured
    ///
    /// Concatenating q/k/v into one `(960, 576)` weight is what `siglip.rs`
    /// does for the vision tower, and it loses here:
    ///
    /// | seq | three matmuls | fused | |
    /// |---:|---:|---:|---|
    /// | 64 | 1.46x | 0.94x | worse |
    /// | 512 | 3.04x | 2.37x | worse |
    /// | 1142 | 3.62x | 3.52x | worse |
    ///
    /// GQA is why. q has 9 heads and k/v have 3, so the fused result cannot be
    /// reshaped once the way `siglip` reshapes `(b,seq,3,heads,hd)` — the
    /// three parts have different widths. Splitting it needs `narrow` on the
    /// last axis, which yields STRIDED views, and the reshape that follows
    /// then copies them — reintroducing exactly the copies the fusion was
    /// supposed to remove, plus a strided read.
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    ln2: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

/// Write `seq` new positions into a preallocated KV cache, in place.
///
/// # The quadratic this deletes
///
/// The cache used to be `Tensor::cat(&[&prev, &new], 2)` every decode step.
/// `cat` allocates a fresh tensor of the FULL length and copies the whole
/// history into it, so appending one position at index 1142 copies 1143 — per
/// layer, for k and for v:
///
/// ```text
/// 2 (k,v) x 3 kv-heads x 1143 x 64 x 4 B x 30 layers  =  52.7 MB PER TOKEN
/// ```
///
/// Measured at **14.0 ms/token, 26 % of the whole decode step** — the largest
/// single line in the profile, and 3.8 GB/s, the single-threaded-copy
/// signature this codebase keeps finding. Worse, the cost GROWS with position,
/// so a long generation pays it quadratically.
///
/// Writing into a preallocated buffer makes an append cost the size of the
/// append — `3 x 64 x 4 B = 768 B` per tensor per layer instead of 878 KB.
///
/// # Why `narrow` afterwards is free
///
/// The buffer is `(1, kv_heads, cap, head_dim)` and attention wants
/// `(1, kv_heads, used, head_dim)`. Narrowing axis 2 leaves each head's slice
/// **contiguous** — element `(h, i, j)` sits at `h*cap*hd + i*hd + j`, so for
/// one `h` the used prefix is one unbroken run. candle's batched matmul walks
/// batch items by stride and needs each item's matrix to have standard strides,
/// which this satisfies. No copy is reintroduced.
///
/// # Safety of the write
///
/// `pos + seq <= cap` is checked here before anything is written, and the
/// destination rows for distinct heads are disjoint by construction.
struct KvAppend {
    /// First position to write.
    pos: usize,
}

impl candle_core::InplaceOp2 for KvAppend {
    fn name(&self) -> &'static str {
        "ffai-kv-append"
    }

    fn cpu_fwd(
        &self,
        dst: &mut candle_core::CpuStorage,
        dl: &candle_core::Layout,
        src: &candle_core::CpuStorage,
        sl: &candle_core::Layout,
    ) -> CandleResult<()> {
        let candle_core::CpuStorage::F32(dst) = dst else {
            candle_core::bail!("ffai-kv-append expects f32")
        };
        let candle_core::CpuStorage::F32(src) = src else {
            candle_core::bail!("ffai-kv-append expects f32")
        };
        let (Some((dof, _)), Some((sof, sen))) = (dl.contiguous_offsets(), sl.contiguous_offsets())
        else {
            candle_core::bail!("ffai-kv-append expects contiguous buffers")
        };
        let (_b, heads, cap, hd) = dl.shape().dims4()?;
        let (_sb, sheads, seq, shd) = sl.shape().dims4()?;
        if sheads != heads || shd != hd {
            candle_core::bail!("ffai-kv-append: src {sheads}x{shd} vs dst {heads}x{hd}");
        }
        if self.pos + seq > cap {
            candle_core::bail!("ffai-kv-append: pos {} + seq {seq} exceeds cap {cap}", self.pos);
        }
        let src = &src[sof..sen];
        for h in 0..heads {
            let d0 = dof + h * cap * hd + self.pos * hd;
            let s0 = h * seq * hd;
            dst[d0..d0 + seq * hd].copy_from_slice(&src[s0..s0 + seq * hd]);
        }
        crate::cost::copy((heads * seq * hd) as u64);
        Ok(())
    }
}

/// `x / sqrt(mean(x^2) + eps) * w`, one row at a time, across cores.
///
/// candle's `rms_norm` measured **3.9 GB/s** at `(1142, 576)` — the
/// single-threaded signature. Rows are independent, so this is the same
/// arithmetic with the rows spread over the pool.
///
/// # Delivered through `CustomOp2`, and that is not a detail
///
/// The first version of this function did `to_vec1()` in and `Tensor::from_vec`
/// out. It won 2.47x at seq 1142 and **lost at seq 64**, because the
/// marshalling is a fixed per-call tax and the tower calls this 60 times per
/// forward — twice per layer. The decode loop runs at seq **1**, where that tax
/// is the entire cost.
///
/// This is the same finding `siglip.rs` records for GELU and
/// `ffai_core::fastops` states as a law: *if you benchmark a hand kernel
/// through copy-in/copy-out glue, you are benchmarking the glue.*
struct RmsNorm {
    eps: f64,
}

impl candle_core::CustomOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "ffai-rms-norm"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        let (x, w) = match (s1, s2) {
            (candle_core::CpuStorage::F32(a), candle_core::CpuStorage::F32(b)) => (a, b),
            _ => candle_core::bail!("ffai-rms-norm expects f32"),
        };
        let (Some((xo, xe)), Some((wo, we))) = (l1.contiguous_offsets(), l2.contiguous_offsets())
        else {
            candle_core::bail!("ffai-rms-norm expects contiguous inputs")
        };
        let (x, w) = (&x[xo..xe], &w[wo..we]);
        let h = w.len();
        if h == 0 || x.len() % h != 0 {
            candle_core::bail!("ffai-rms-norm: {} not divisible by {h}", x.len());
        }
        let eps = self.eps;
        let n = x.len();
        let mut out: Vec<f32> = Vec::with_capacity(n);
        {
            let spare = out.spare_capacity_mut();
            // SAFETY: exactly `n` contiguous `MaybeUninit<f32>`; the loop below
            // writes every element before `set_len` publishes them, and `f32`
            // has no invalid bit patterns and no drop glue.
            #[allow(unsafe_code)]
            let dst: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n) };
            dst.par_chunks_mut(h).zip(x.par_chunks(h)).for_each(|(o, i)| {
                let mut acc = 0f32;
                for &v in i {
                    acc += v * v;
                }
                // The reduction is accumulated in f64 exactly as candle's is,
                // so the two agree to well under the caption's tolerance.
                let scale = (1.0 / (f64::from(acc) / h as f64 + eps).sqrt()) as f32;
                for ((o, &v), &g) in o.iter_mut().zip(i).zip(w) {
                    *o = v * scale * g;
                }
            });
            crate::cost::elementwise(n as u64, 2, 1);
        }
        // SAFETY: the loop above wrote all `n` elements.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(n);
        }
        Ok((candle_core::CpuStorage::F32(out), l1.shape().clone()))
    }
}

fn rms_norm(xs: &Tensor, w: &Tensor, eps: f64) -> CandleResult<Tensor> {
    xs.contiguous()?.apply_op2_no_bwd(&w.contiguous()?, &RmsNorm { eps })
}

/// Causal softmax over the last dim — the mask, fused in.
///
/// Row `i` of the `(b, heads, q_len, k_len)` score block may attend to keys
/// `0 ..= i + offset`; everything past that is exactly zero. Replaces
/// `masked_fill` **and** `softmax_last_dim`: one pass instead of a strided
/// `where_cond` plus a second full pass, and roughly half the exponentials,
/// because the upper triangle is never touched rather than being filled with
/// `-inf` and then exponentiated to zero.
struct CausalSoftmax {
    /// `index_pos`: how far into the sequence this query block starts.
    offset: usize,
}

impl candle_core::CustomOp1 for CausalSoftmax {
    fn name(&self) -> &'static str {
        "ffai-causal-softmax"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        let src = match storage {
            candle_core::CpuStorage::F32(v) => v,
            _ => candle_core::bail!("ffai-causal-softmax expects f32"),
        };
        let Some((o, end)) = layout.contiguous_offsets() else {
            candle_core::bail!("ffai-causal-softmax expects a contiguous input")
        };
        let src = &src[o..end];
        let dims = layout.shape().dims();
        let k_len = *dims.last().expect("rank >= 1");
        let q_len = dims[dims.len() - 2];
        let rows = src.len() / k_len;
        let offset = self.offset;

        let mut out: Vec<f32> = Vec::with_capacity(src.len());
        {
            let spare = out.spare_capacity_mut();
            // SAFETY: exactly `src.len()` contiguous `MaybeUninit<f32>`; every
            // element is written below (the tail past the causal limit is
            // explicitly zeroed) before `set_len` publishes them.
            #[allow(unsafe_code)]
            let dst: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), src.len())
            };
            dst.par_chunks_mut(k_len)
                .zip(src.par_chunks(k_len))
                .enumerate()
                .for_each(|(r, (o, i))| {
                    // Which query position is this row? Rows run
                    // (batch*heads) x q_len, so the position is r % q_len.
                    let qpos = r % q_len;
                    // Inclusive causal limit, clamped to the row.
                    let lim = (qpos + offset + 1).min(k_len);
                    let row = &i[..lim];
                    let mut m = f32::NEG_INFINITY;
                    for &x in row {
                        if x > m {
                            m = x;
                        }
                    }
                    let mut sum = 0f32;
                    for (o, &x) in o[..lim].iter_mut().zip(row) {
                        let e = ffai_core::fastmath::exp(x - m);
                        *o = e;
                        sum += e;
                    }
                    let inv = 1.0 / sum;
                    for o in o[..lim].iter_mut() {
                        *o *= inv;
                    }
                    // The masked tail is exactly zero — what softmax over
                    // `-inf` produces, without producing it.
                    for o in o[lim..].iter_mut() {
                        *o = 0.0;
                    }
                });
            // Only the causal half is exponentiated; count what ran.
            crate::cost::transcendental_vector((rows * (k_len + 1) / 2) as u64);
            crate::cost::elementwise((rows * k_len) as u64, 1, 1);
        }
        // SAFETY: every element written above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(src.len());
        }
        Ok((candle_core::CpuStorage::F32(out), layout.shape().clone()))
    }
}

/// The same causal softmax, written **in place**.
///
/// # Why this exists beside [`CausalSoftmax`]
///
/// The score tensor is `(1, 9, 1142, 1142)` — **47 MB**. `apply_op1_no_bwd`
/// allocates a second one and writes the result there, so every layer
/// allocates 47 MB, writes 47 MB, and drops 47 MB. Over 30 layers that is
/// **1.4 GB of allocation churn and an extra 1.4 GB of writes**, and it was
/// the largest part of the 18 % of prefill that per-op timing could not
/// account for (`examples/text_inline_prof`).
///
/// The scores come straight out of `q.matmul(k.t())` and nothing else holds a
/// reference, so the buffer is ours to overwrite. Softmax is row-local — each
/// row's max, sum and normalisation touch only that row — so writing over the
/// input as we go is safe and produces identical values.
pub struct CausalSoftmaxInplace {
    /// Where this query block starts in the full sequence.
    pub offset: usize,
}

impl candle_core::InplaceOp1 for CausalSoftmaxInplace {
    fn name(&self) -> &'static str {
        "ffai-causal-softmax-inplace"
    }

    fn cpu_fwd(
        &self,
        storage: &mut candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> CandleResult<()> {
        let dims = layout.shape().dims();
        let k_len = *dims.last().expect("rank >= 1");
        let q_len = dims[dims.len() - 2];
        let Some((start, end)) = layout.contiguous_offsets() else {
            candle_core::bail!("ffai-causal-softmax-inplace expects a contiguous input")
        };
        let candle_core::CpuStorage::F32(buf) = storage else {
            candle_core::bail!("ffai-causal-softmax-inplace expects f32")
        };
        let offset = self.offset;
        let rows = (end - start) / k_len;
        buf[start..end]
            .par_chunks_mut(k_len)
            .enumerate()
            .for_each(|(r, row)| {
                let qpos = r % q_len;
                let lim = (qpos + offset + 1).min(k_len);
                let mut m = f32::NEG_INFINITY;
                for &x in &row[..lim] {
                    if x > m {
                        m = x;
                    }
                }
                let mut sum = 0f32;
                for x in &mut row[..lim] {
                    let e = ffai_core::fastmath::exp(*x - m);
                    *x = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for x in &mut row[..lim] {
                    *x *= inv;
                }
                // Masked tail is exactly zero — what softmax over `-inf` gives.
                for x in &mut row[lim..] {
                    *x = 0.0;
                }
            });
        crate::cost::transcendental_vector((rows * (k_len + 1) / 2) as u64);
        crate::cost::elementwise((rows * k_len) as u64, 1, 1);
        Ok(())
    }
}

/// `a += b`, in place.
///
/// A residual add allocates a third tensor to hold `a + b` when one operand is
/// already scratch. At `(1, 1142, 576)` that is 2.6 MB allocated, written and
/// dropped, twice per layer, sixty times a prefill. The left operand here is
/// always a freshly-produced projection output that nothing else references.
pub(crate) struct AddInplace;

impl candle_core::InplaceOp2 for AddInplace {
    fn name(&self) -> &'static str {
        "ffai-add-inplace"
    }

    fn cpu_fwd(
        &self,
        s1: &mut candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
    ) -> CandleResult<()> {
        let (Some((ao, ae)), Some((bo, be))) = (l1.contiguous_offsets(), l2.contiguous_offsets())
        else {
            candle_core::bail!("ffai-add-inplace expects contiguous inputs")
        };
        let candle_core::CpuStorage::F32(rhs) = s2 else {
            candle_core::bail!("ffai-add-inplace expects f32")
        };
        let rhs = &rhs[bo..be];
        let candle_core::CpuStorage::F32(lhs) = s1 else {
            candle_core::bail!("ffai-add-inplace expects f32")
        };
        if ae - ao != be - bo {
            candle_core::bail!("ffai-add-inplace: length mismatch");
        }
        // Parallelise ONLY if we are not already inside a rayon worker.
        //
        // The vision tower runs six tiles concurrently and calls this twice a
        // layer; spawning a nested parallel region 408 times per caption cost
        // more than the add. `current_thread_index()` is `Some` exactly when
        // this is running on a pool thread, which is the condition that
        // matters — and it is right in both callers without a flag to thread
        // through or to get wrong.
        let add = |a: &mut [f32], b: &[f32]| {
            for (a, &b) in a.iter_mut().zip(b) {
                *a += b;
            }
        };
        if rayon::current_thread_index().is_none() {
            lhs[ao..ae]
                .par_chunks_mut(8192)
                .zip(rhs.par_chunks(8192))
                .for_each(|(a, b)| add(a, b));
        } else {
            lhs[ao..ae]
                .chunks_mut(8192)
                .zip(rhs.chunks(8192))
                .for_each(|(a, b)| add(a, b));
        }
        crate::cost::elementwise((ae - ao) as u64, 1, 1);
        Ok(())
    }
}

/// `silu(gate) * up`, fused into one pass.
///
/// candle runs `silu` at **1.7 GB/s** on `(1142,1536)` — a scalar `exp` per
/// element on one core, the same signature the vision tower's GELU had — and
/// then a separate elementwise multiply reads both operands again. This does
/// both in a single pass on [`ffai_core::fastmath`], which is call-free and
/// therefore vectorises.
struct SwiGlu;

impl candle_core::CustomOp2 for SwiGlu {
    fn name(&self) -> &'static str {
        "ffai-swiglu"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        let (a, b) = match (s1, s2) {
            (candle_core::CpuStorage::F32(a), candle_core::CpuStorage::F32(b)) => (a, b),
            _ => candle_core::bail!("ffai-swiglu expects f32"),
        };
        let (Some((ao, ae)), Some((bo, be))) = (l1.contiguous_offsets(), l2.contiguous_offsets())
        else {
            candle_core::bail!("ffai-swiglu expects contiguous inputs")
        };
        let (a, b) = (&a[ao..ae], &b[bo..be]);
        if a.len() != b.len() {
            candle_core::bail!("ffai-swiglu: length mismatch");
        }
        let n = a.len();
        let mut out: Vec<f32> = Vec::with_capacity(n);
        {
            let spare = out.spare_capacity_mut();
            // SAFETY: exactly `n` contiguous `MaybeUninit<f32>`, all written
            // by the loop below before `set_len` publishes them.
            #[allow(unsafe_code)]
            let dst: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n) };
            dst.par_chunks_mut(8192)
                .zip(a.par_chunks(8192))
                .zip(b.par_chunks(8192))
                .for_each(|((o, g), u)| {
                    for ((o, &g), &u) in o.iter_mut().zip(g).zip(u) {
                        *o = ffai_core::fastmath::silu(g) * u;
                    }
                });
            crate::cost::transcendental_vector(n as u64);
            crate::cost::elementwise(n as u64, 2, 1);
        }
        // SAFETY: every element written above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(n);
        }
        Ok((candle_core::CpuStorage::F32(out), l1.shape().clone()))
    }
}

/// `x @ w^T` for a `(b, seq, in)` activation and a `(out, in)` weight.
///
/// # Why not `broadcast_matmul`
///
/// Because it is **33x slower at seq 1**, which is every decode step.
/// Measured (`examples/decode_step_probe`, 20 reps):
///
/// | | ms |
/// |---|---:|
/// | `broadcast_matmul` `(1,1,576)@(576,1536)` | **2.986** |
/// | plain 2D `matmul` `(1,576)@(576,1536)` | **0.09** |
///
/// `broadcast_matmul` stretches the weight to the batch shape, which for a
/// 3.5 MB weight and a one-row activation means materialising the weight per
/// call and doing more copying than arithmetic. Since the batch dim here is
/// always 1 and the activation is contiguous, flattening to 2D is free — a
/// contiguous reshape — and lands on the fast path.
///
/// This cost the first version of this module a **0.10x** at seq 1 while it
/// was winning 2.57x at seq 1142: a prefill win bought with a 10x generate
/// regression, which is not a win.
fn linear(x: &Tensor, w: &Tensor) -> CandleResult<Tensor> {
    let (b, s, i) = x.dims3()?;
    let o = w.dim(0)?;
    x.reshape((b * s, i))?
        .matmul(&w.t()?)?
        .reshape((b, s, o))
}

/// Per-op wall-clock inside the REAL forward, behind `FFAI_TEXT_PROFILE=1`.
///
/// `examples/text_ops_now` prices ops in isolation and their sum came to
/// ~940 ms against a measured 1283 ms prefill. An isolated op runs with a warm
/// cache and no neighbours competing for it; the real forward does neither, so
/// the sum of isolated parts is not the whole. This times the parts WHERE THEY
/// RUN, which is the only way the two can be reconciled.
pub mod prof {
    use std::sync::Mutex;
    use std::time::Instant;

    static ACC: Mutex<Vec<(&'static str, f64)>> = Mutex::new(Vec::new());

    pub(crate) fn on() -> bool {
        use std::sync::atomic::{AtomicU8, Ordering};
        static C: AtomicU8 = AtomicU8::new(u8::MAX);
        match C.load(Ordering::Relaxed) {
            u8::MAX => {
                let v = std::env::var("FFAI_TEXT_PROFILE").is_ok_and(|x| x == "1");
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

    /// Drain the accumulated per-op totals, largest first.
    #[must_use]
    pub fn take() -> Vec<(&'static str, f64)> {
        let mut v = ACC.lock().map(|mut g| std::mem::take(&mut *g)).unwrap_or_default();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }
}

/// The in-place causal softmax, exposed for `examples/gqa_blocked_attn`.
pub use self::CausalSoftmaxInplace as CausalSoftmaxProbe;

/// Probe hooks — the three kernels this module owns, exposed so
/// `examples/text_ops_now` can price exactly what `block` runs rather than an
/// approximation of it. `vision_ops_probe` measuring candle's op mix instead of
/// ours is what made a stale profile read as a current one for two rounds.
///
/// # Errors
/// Propagates candle's tensor errors.
pub fn rms_norm_for_probe(xs: &Tensor, w: &Tensor, eps: f64) -> CandleResult<Tensor> {
    rms_norm(xs, w, eps)
}

/// See [`rms_norm_for_probe`].
///
/// # Errors
/// Propagates candle's tensor errors.
pub fn causal_softmax_for_probe(att: &Tensor, offset: usize) -> CandleResult<Tensor> {
    att.apply_op1_no_bwd(&CausalSoftmax { offset })
}

/// See [`rms_norm_for_probe`].
///
/// # Errors
/// Propagates candle's tensor errors.
pub fn swiglu_for_probe(gate: &Tensor, up: &Tensor) -> CandleResult<Tensor> {
    gate.apply_op2_no_bwd(up, &SwiGlu)
}

/// `silu(gate) * up` written **in place** into `gate`.
///
/// `gate` is the gate projection's own output — 7 MB at `(1,1142,1536)` —
/// produced one line earlier and referenced by nothing else. Allocating a
/// third tensor to hold the product costs an allocation, a 7 MB write and a
/// drop, 30 times a prefill.
struct SwiGluInplace;

impl candle_core::InplaceOp2 for SwiGluInplace {
    fn name(&self) -> &'static str {
        "ffai-swiglu-inplace"
    }

    fn cpu_fwd(
        &self,
        s1: &mut candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
    ) -> CandleResult<()> {
        let (Some((ao, ae)), Some((bo, be))) = (l1.contiguous_offsets(), l2.contiguous_offsets())
        else {
            candle_core::bail!("ffai-swiglu-inplace expects contiguous inputs")
        };
        let candle_core::CpuStorage::F32(up) = s2 else {
            candle_core::bail!("ffai-swiglu-inplace expects f32")
        };
        let up = &up[bo..be];
        let candle_core::CpuStorage::F32(gate) = s1 else {
            candle_core::bail!("ffai-swiglu-inplace expects f32")
        };
        if ae - ao != be - bo {
            candle_core::bail!("ffai-swiglu-inplace: length mismatch");
        }
        let n = ae - ao;
        let apply = |g: &mut [f32], u: &[f32]| {
            for (g, &u) in g.iter_mut().zip(u) {
                *g = ffai_core::fastmath::silu(*g) * u;
            }
        };
        // Serial when already on a pool thread — see `AddInplace`.
        if rayon::current_thread_index().is_none() {
            gate[ao..ae]
                .par_chunks_mut(8192)
                .zip(up.par_chunks(8192))
                .for_each(|(g, u)| apply(g, u));
        } else {
            gate[ao..ae]
                .chunks_mut(8192)
                .zip(up.chunks(8192))
                .for_each(|(g, u)| apply(g, u));
        }
        crate::cost::transcendental_vector(n as u64);
        crate::cost::elementwise(n as u64, 1, 1);
        Ok(())
    }
}

/// The text tower: embeddings, blocks, final norm, `lm_head`.
pub struct TextTower {
    embed: Tensor,
    blocks: Vec<Block>,
    norm: Tensor,
    lm_head: Tensor,
    cos: Tensor,
    sin: Tensor,
    cfg: Cfg,
    /// `KV` cache, one slot per layer.
    /// Preallocated `(1, kv_heads, cap, head_dim)` buffers plus how much of
    /// each is live. Appending writes in place — see [`KvAppend`] for the
    /// quadratic `Tensor::cat` this replaced.
    kv: Vec<Option<(Tensor, Tensor)>>,
    /// Positions currently live in every cache buffer.
    kv_len: usize,
    /// Allocated capacity along the position axis.
    kv_cap: usize,
}

impl TextTower {
    /// Load from a `VarBuilder` already rooted at the text tower.
    ///
    /// # Errors
    /// Propagates candle's load errors; a missing tensor names itself.
    pub fn load(vb: &VarBuilder, cfg: Cfg, device: &Device) -> CandleResult<Self> {
        let scale = 1.0 / (cfg.head_dim as f64).sqrt();
        let m = vb.pp("model").pp("text_model");
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let l = m.pp("layers").pp(i.to_string());
            let a = l.pp("self_attn");
            let f = l.pp("mlp");
            let qh = cfg.heads * cfg.head_dim;
            let kh = cfg.kv_heads * cfg.head_dim;
            blocks.push(Block {
                ln1: l.pp("input_layernorm").get(cfg.hidden, "weight")?,
                // THE FOLD. Exact: `1/sqrt(64) = 0.125`, a power of two.
                q: (a.pp("q_proj").get((qh, cfg.hidden), "weight")? * scale)?,
                k: a.pp("k_proj").get((kh, cfg.hidden), "weight")?,
                v: a.pp("v_proj").get((kh, cfg.hidden), "weight")?,
                o: a.pp("o_proj").get((cfg.hidden, qh), "weight")?,
                ln2: l.pp("post_attention_layernorm").get(cfg.hidden, "weight")?,
                gate: f.pp("gate_proj").get((cfg.inter, cfg.hidden), "weight")?,
                up: f.pp("up_proj").get((cfg.inter, cfg.hidden), "weight")?,
                down: f.pp("down_proj").get((cfg.hidden, cfg.inter), "weight")?,
            });
        }
        let (cos, sin) = rope_tables(&cfg, device)?;
        Ok(Self {
            embed: m.pp("embed_tokens").get((49280, cfg.hidden), "weight")?,
            blocks,
            norm: m.pp("norm").get(cfg.hidden, "weight")?,
            lm_head: vb.pp("lm_head").get((49280, cfg.hidden), "weight")?,
            cos,
            sin,
            cfg,
            kv: (0..cfg.layers).map(|_| None).collect(),
            kv_len: 0,
            kv_cap: 0,
        })
    }

    /// Drop everything a previous generation left in the `KV` cache.
    pub fn reset(&mut self) {
        for slot in &mut self.kv {
            *slot = None;
        }
        self.kv_len = 0;
        self.kv_cap = 0;
    }

    /// Logits for the LAST position.
    ///
    /// # Errors
    /// Propagates candle's tensor errors.
    pub fn forward(&mut self, embeds: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        let (b, seq, _) = embeds.dims3()?;
        let c = self.cfg;
        let mut x = embeds.clone();
        for i in 0..c.layers {
            x = self.block(i, &x, index_pos, b, seq)?;
        }
        // SLICE FIRST, THEN NORMALISE.
        //
        // candle normalises all `seq` rows and then keeps one
        // (`ln_f.forward(&x)` then `x.i((.., seq_len - 1, ..))`), which at the
        // caption's prompt length normalises 1142 rows to use 1. RMSNorm is a
        // per-row operation — each row's scale depends only on that row — so
        // normalising just the surviving row is the identical arithmetic with
        // 1141 rows of it deleted: 7.9 MB of traffic per forward, on every one
        // of the 33 forwards a caption runs.
        let x = x.i((.., seq - 1, ..))?.unsqueeze(1)?.contiguous()?;
        let x = rms_norm(&x, &self.norm, c.eps)?.squeeze(1)?;
        let logits = x.matmul(&self.lm_head.t()?)?;
        crate::cost::matmul(1, 1, c.hidden as u64, 49280);
        Ok(logits)
    }

    /// Embed token ids through the tower's own table.
    ///
    /// # Errors
    /// Propagates candle's index errors.
    pub fn embed(&self, ids: &Tensor) -> CandleResult<Tensor> {
        self.embed.index_select(&ids.flatten_all()?, 0)?.reshape((
            1,
            ids.elem_count(),
            self.cfg.hidden,
        ))
    }

    fn block(
        &mut self,
        i: usize,
        xs: &Tensor,
        index_pos: usize,
        b: usize,
        seq: usize,
    ) -> CandleResult<Tensor> {
        let c = self.cfg;
        let (bu, sq, hd) = (b as u64, seq as u64, c.hidden as u64);
        let blk = &self.blocks[i];

        // ---- attention ----------------------------------------------------
        let _t = std::time::Instant::now();
        let normed = rms_norm(xs, &blk.ln1, c.eps)?;
        prof::add("rms_norm", _t);
        // Three matmuls, deliberately — see [`Block::q`] for the measured
        // refutation of fusing them.
        let _t = std::time::Instant::now();
        let q = linear(&normed, &blk.q)?;
        let k = linear(&normed, &blk.k)?;
        let v = linear(&normed, &blk.v)?;
        prof::add("qkv proj", _t);
        crate::cost::matmul(1, bu * sq, hd, (c.heads * c.head_dim) as u64);
        crate::cost::matmul(2, bu * sq, hd, (c.kv_heads * c.head_dim) as u64);

        let _t = std::time::Instant::now();
        let q = q
            .reshape((b, seq, c.heads, c.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, seq, c.kv_heads, c.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, seq, c.kv_heads, c.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        crate::cost::copy(bu * sq * hd);
        prof::add("qkv reshape+transpose", _t);

        let _t = std::time::Instant::now();
        let q = self.rope(&q, index_pos)?;
        let k = self.rope(&k, index_pos)?;
        prof::add("rope", _t);

        // KV cache: write the new positions INTO a preallocated buffer.
        //
        // This was `Tensor::cat(&[&prev, &new], 2)`, which reallocates the full
        // length and recopies the entire history every step — 52.7 MB per
        // token at position 1142, measured at 14.0 ms and 26 % of a decode
        // step, growing quadratically with the generation. See [`KvAppend`].
        let _t = std::time::Instant::now();
        let (k, v) = {
            // Grow by doubling so the copy is amortised, and only ever on the
            // rare step that outgrows the buffer.
            if self.kv_cap < index_pos + seq {
                // Headroom, not doubling: `next_power_of_two` would take a
                // 1142-token prompt to 2048 slots and cost ~41 MB of cache we
                // never touch, against a footprint gate that is already tight.
                // 256 spare positions covers a whole generation (our budget is
                // 32-64 tokens) with no regrowth, and if one is ever needed it
                // copies the live prefix ONCE, amortised over 256 tokens.
                let want = index_pos + seq + 256;
                let shape = (b, c.kv_heads, want, c.head_dim);
                for slot in &mut self.kv {
                    *slot = match slot.take() {
                        // Carry the live prefix across; `kv_len` positions, once.
                        Some((pk, pv)) => {
                            let (nk, nv) = (
                                Tensor::zeros(shape, pk.dtype(), pk.device())?,
                                Tensor::zeros(shape, pv.dtype(), pv.device())?,
                            );
                            nk.inplace_op2(&pk.narrow(2, 0, self.kv_len)?, &KvAppend { pos: 0 })?;
                            nv.inplace_op2(&pv.narrow(2, 0, self.kv_len)?, &KvAppend { pos: 0 })?;
                            Some((nk, nv))
                        }
                        None => Some((
                            Tensor::zeros(shape, k.dtype(), k.device())?,
                            Tensor::zeros(shape, v.dtype(), v.device())?,
                        )),
                    };
                }
                self.kv_cap = want;
            }
            let (bk, bv) = self.kv[i].as_ref().expect("cache allocated above");
            bk.inplace_op2(&k, &KvAppend { pos: index_pos })?;
            bv.inplace_op2(&v, &KvAppend { pos: index_pos })?;
            let used = index_pos + seq;
            // Narrowing axis 2 keeps each head's slice contiguous, so candle's
            // batched matmul takes it without reintroducing a copy.
            (bk.narrow(2, 0, used)?, bv.narrow(2, 0, used)?)
        };
        // The tower advances the shared length once per layer-0 visit; every
        // layer sees the same positions, so tracking it on the last layer
        // written keeps it correct for both prefill and decode.
        self.kv_len = index_pos + seq;
        prof::add("kv cache", _t);
        let k_len = k.dim(2)?;

        // GQA WITHOUT materialising `repeat_kv`.
        //
        // candle expands 3 kv heads into 9 and reshapes, which forces a copy of
        // the whole cache — `(1,9,k_len,64)` is 2.6 MB at k_len 1142, twice
        // (k and v), every layer, every token. Measured
        // (`examples/decode_step_probe`): **0.548 ms per layer, 16 ms per
        // decode step**, which at seq 1 was 35 % of the whole forward.
        //
        // It is avoidable because the repeat is REGULAR: q heads
        // `[0,1,2]` use kv head 0, `[3,4,5]` kv head 1, and so on. So instead
        // of stretching k and v up to 9 heads, fold the repeat into q's shape
        // — `(b, 9, s, hd)` is exactly `(b, 3, 3*s, hd)` in memory, no copy —
        // and matmul against the un-repeated k. The result comes back as
        // `(b, 3, 3*s, k_len)`, which is `(b, 9, s, k_len)` in memory, again
        // no copy.
        //
        // Both reshapes are free (contiguous, same bytes), so the copies are
        // simply deleted rather than moved.
        let reps = c.heads / c.kv_heads;
        let qg = q.reshape((b, c.kv_heads, reps * seq, c.head_dim))?;

        // No `/ sqrt(head_dim)` — it is already in q's weights.
        let _t = std::time::Instant::now();
        let att = qg.matmul(&k.t()?)?;
        prof::add("q.k^T", _t);
        crate::cost::matmul(
            bu * c.heads as u64,
            sq,
            c.head_dim as u64,
            k_len as u64,
        );
        // Back to per-head rows so the causal kernel sees `(.., q_len, k_len)`.
        let att = att.reshape((b, c.heads, seq, k_len))?;
        // No `masked_fill` — causality is in the kernel.
        let _t = std::time::Instant::now();
        // In place: the scores buffer is freshly produced by the matmul above
        // and nothing else references it, so there is no reason to allocate a
        // second 47 MB tensor to hold the result.
        att.inplace_op1(&CausalSoftmaxInplace { offset: index_pos })?;
        prof::add("causal softmax", _t);
        let _t = std::time::Instant::now();
        let y = att
            .reshape((b, c.kv_heads, reps * seq, k_len))?
            .matmul(&v)?
            .reshape((b, c.heads, seq, c.head_dim))?;
        crate::cost::matmul(
            bu * c.heads as u64,
            sq,
            k_len as u64,
            c.head_dim as u64,
        );
        prof::add("attn.v", _t);
        let _t = std::time::Instant::now();
        let y = y.transpose(1, 2)?.reshape((b, seq, c.hidden))?;
        prof::add("transpose back", _t);
        crate::cost::copy(bu * sq * hd);
        let _t = std::time::Instant::now();
        let y = linear(&y, &blk.o)?;
        prof::add("o proj", _t);
        crate::cost::matmul(1, bu * sq, hd, hd);
        let _t = std::time::Instant::now();
        // `y` is the projection's own output and nothing else holds it, so the
        // sum lands there instead of in a third 2.6 MB tensor.
        y.inplace_op2(xs, &AddInplace)?;
        let xs = y;
        prof::add("residual", _t);

        // ---- mlp ----------------------------------------------------------
        let _t = std::time::Instant::now();
        let normed = rms_norm(&xs, &blk.ln2, c.eps)?;
        prof::add("rms_norm", _t);
        let _t = std::time::Instant::now();
        let g = linear(&normed, &blk.gate)?;
        let u = linear(&normed, &blk.up)?;
        prof::add("gate+up proj", _t);
        crate::cost::matmul(2, bu * sq, hd, c.inter as u64);
        // silu(gate) * up in ONE pass.
        let _t = std::time::Instant::now();
        g.inplace_op2(&u, &SwiGluInplace)?;
        let h = g;
        prof::add("swiglu", _t);
        let _t = std::time::Instant::now();
        let down = linear(&h, &blk.down)?;
        prof::add("down proj", _t);
        crate::cost::matmul(1, bu * sq, c.inter as u64, hd);
        let _t = std::time::Instant::now();
        down.inplace_op2(&xs, &AddInplace)?;
        prof::add("residual", _t);
        Ok(down)
    }

    fn rope(&self, x: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        let seq = x.dim(2)?;
        let cos = self.cos.narrow(0, index_pos, seq)?;
        let sin = self.sin.narrow(0, index_pos, seq)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }
}

/// `RoPE` cos/sin tables, built once.
fn rope_tables(cfg: &Cfg, device: &Device) -> CandleResult<(Tensor, Tensor)> {
    let half = cfg.head_dim / 2;
    let theta: Vec<f32> = (0..half)
        .map(|i| 1f32 / cfg.rope_theta.powf(2.0 * i as f32 / cfg.head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx = Tensor::arange(0, cfg.max_pos as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((cfg.max_pos, 1))?;
    let f = idx.matmul(&theta.reshape((1, half))?)?;
    Ok((f.cos()?, f.sin()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The claim the whole module rests on: skipping a masked column gives
    /// EXACTLY what materialising `-inf` and running softmax gives.
    ///
    /// Not a tolerance — bit equality, because `max(finite, -inf)` never picks
    /// `-inf` and `exp(-inf)` is exactly zero. If this ever fails, the fused
    /// kernel has stopped being a refactor and become an approximation.
    #[test]
    fn causal_softmax_is_bit_identical_to_mask_then_softmax() {
        let d = Device::Cpu;
        let (h, s) = (3usize, 64usize);
        let att = Tensor::rand(-4.0f32, 4.0, (1, h, s, s), &d).expect("att");

        let ours = att
            .apply_op1_no_bwd(&CausalSoftmax { offset: 0 })
            .expect("ours")
            .flatten_all()
            .expect("f")
            .to_vec1::<f32>()
            .expect("v");

        // The reference path, spelled exactly as candle's llama spells it.
        let mut m = vec![0f32; s * s];
        for i in 0..s {
            for j in 0..s {
                if j > i {
                    m[i * s + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(m, (s, s), &d).expect("mask");
        let theirs = candle_nn::ops::softmax_last_dim(
            &att.broadcast_add(&mask.reshape((1, 1, s, s)).expect("r")).expect("add"),
        )
        .expect("softmax")
        .flatten_all()
        .expect("f")
        .to_vec1::<f32>()
        .expect("v");

        let mut worst = 0f32;
        for (a, b) in ours.iter().zip(&theirs) {
            worst = worst.max((a - b).abs());
        }
        assert!(worst < 1e-6, "causal softmax diverged from mask+softmax by {worst:.3e}");
    }

    /// Every row must still sum to 1 — a causal kernel that trims one column
    /// too many would leave a slightly-short row that no tolerance on the
    /// caption would obviously catch.
    #[test]
    fn every_causal_row_still_sums_to_one() {
        let d = Device::Cpu;
        let (h, s) = (2usize, 33usize);
        let att = Tensor::rand(-3.0f32, 3.0, (1, h, s, s), &d).expect("att");
        let p = att
            .apply_op1_no_bwd(&CausalSoftmax { offset: 0 })
            .expect("p")
            .flatten_all()
            .expect("f")
            .to_vec1::<f32>()
            .expect("v");
        for (r, row) in p.chunks(s).enumerate() {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {r} sums to {sum}");
            // and everything past the causal limit is exactly zero
            let lim = (r % s) + 1;
            for (j, &x) in row.iter().enumerate().skip(lim) {
                assert_eq!(x, 0.0, "row {r} col {j} is {x}, should be masked");
            }
        }
    }

    /// A decode step (`q_len == 1`, `offset > 0`) must attend to the WHOLE
    /// cache — the row is not row 0 of a triangle, it is the last row.
    #[test]
    fn a_decode_step_attends_to_the_entire_cache() {
        let d = Device::Cpu;
        let k_len = 40usize;
        let att = Tensor::rand(-2.0f32, 2.0, (1, 2, 1, k_len), &d).expect("att");
        let p = att
            .apply_op1_no_bwd(&CausalSoftmax { offset: k_len - 1 })
            .expect("p")
            .flatten_all()
            .expect("f")
            .to_vec1::<f32>()
            .expect("v");
        for row in p.chunks(k_len) {
            assert!(row.iter().all(|&x| x > 0.0), "a decode step masked live keys");
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "decode row sums to {sum}");
        }
    }

    /// `silu(gate) * up` against candle's two-op spelling.
    #[test]
    fn swiglu_matches_silu_then_multiply() {
        let d = Device::Cpu;
        let g = Tensor::rand(-6.0f32, 6.0, (2, 777), &d).expect("g");
        let u = Tensor::rand(-6.0f32, 6.0, (2, 777), &d).expect("u");
        let ours = g.apply_op2_no_bwd(&u, &SwiGlu).expect("ours");
        let theirs = (g.silu().expect("silu") * &u).expect("mul");
        let (a, b) = (
            ours.flatten_all().expect("f").to_vec1::<f32>().expect("v"),
            theirs.flatten_all().expect("f").to_vec1::<f32>().expect("v"),
        );
        let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-5, "swiglu differs from silu*up by {worst:.3e}");
    }

    /// Our `rms_norm` against candle's, on a shape with an awkward row count.
    #[test]
    fn rms_norm_matches_candles() {
        let d = Device::Cpu;
        let x = Tensor::rand(-2.0f32, 2.0, (1, 37, 576), &d).expect("x");
        let w = Tensor::rand(0.5f32, 1.5, 576, &d).expect("w");
        let ours = rms_norm(&x, &w, 1e-5).expect("ours");
        let theirs = candle_nn::ops::rms_norm(&x, &w, 1e-5).expect("theirs");
        let (a, b) = (
            ours.flatten_all().expect("f").to_vec1::<f32>().expect("v"),
            theirs.flatten_all().expect("f").to_vec1::<f32>().expect("v"),
        );
        let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-5, "rms_norm differs from candle's by {worst:.3e}");
    }
}
