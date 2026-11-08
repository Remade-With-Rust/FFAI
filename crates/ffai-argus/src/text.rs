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

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use crate::par::prelude::*;

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
    /// `ln2`'s weight as a host slice, for the fused residual + RMSNorm.
    ln2_host: std::sync::Arc<[f32]>,
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
    /// q (pre-scaled), k and v stacked into one `((heads + 2*kv_heads) * hd,
    /// hidden)` weight. Their consumers ([`RopeHeads`], [`HeadsIntoCache`])
    /// read each part at its offset within the fused row, so — unlike the
    /// refuted fusion above, which split with `narrow` + `reshape` — nothing is
    /// copied to separate them.
    qkv: Tensor,
    o: Tensor,
    /// `gate_proj` and `up_proj` stacked into one `(2*inter, hidden)` weight:
    /// one GEMM, one pass over the normalised activation, half the calls.
    ///
    /// This is NOT the refuted q/k/v fusion again. That lost because candle's
    /// `reshape` copied the strided views a fused output must be split into;
    /// here the consumers are our own — the SwiGLU op works on
    /// each row's two halves, and the down projection reads the gate half at
    /// its row stride through `mm::matmul` — so nothing is copied.
    gate_up: Tensor,
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

/// Below this many elements a row op runs serially: a decode step's RMSNorm or
/// RoPE is ONE row of 576, and rayon's split/join around it costs more than
/// the row. Prefill (1142 rows) stays parallel.
const SERIAL_BELOW: usize = 16 * 1024;

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
            let avx2 = crate::siglip::avx2_only();
            // Consumed only by the x86_64 arm below.
            #[cfg(not(target_arch = "x86_64"))]
            let _ = avx2;
            if n < SERIAL_BELOW {
                dst.chunks_mut(h).zip(x.chunks(h)).for_each(|(o, i)| rms_row_dispatch(avx2, o, i, w, eps));
            } else {
                dst.par_chunks_mut(h).zip(x.par_chunks(h)).for_each(|(o, i)| rms_row_dispatch(avx2, o, i, w, eps));
            }
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

/// One RMSNorm row: `o = i * (1 / sqrt(mean(i^2) + eps)) * w`.
///
/// The sum of squares runs in EIGHT independent lanes. The single running
/// `acc += v * v` it replaces was a loop-carried non-associative reduction
/// LLVM will not reorder, so it stayed scalar; eight partial sums vectorise,
/// and they are also more accurate — rounding error grows with the length of
/// the dependency chain, which this cuts eightfold. (Candle's `rms_norm`
/// accumulates in a single `f32` too — `candle-nn/src/ops.rs:480`; the f64
/// here is only the final scale, as before.) Float-close, token-gated.
#[inline(always)]
fn rms_row(o: &mut [f32], i: &[f32], w: &[f32], eps: f64) {
    const L: usize = 8;
    let mut acc = [0f32; L];
    let mut it = i.chunks_exact(L);
    for c in &mut it {
        for l in 0..L {
            acc[l] += c[l] * c[l];
        }
    }
    let mut ss = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for &v in it.remainder() {
        ss += v * v;
    }
    let scale = (1.0 / (f64::from(ss) / i.len() as f64 + eps).sqrt()) as f32;
    for ((o, &v), &g) in o.iter_mut().zip(i).zip(w) {
        *o = v * scale * g;
    }
}

/// [`rms_row`] compiled for AVX2 (no `fma`): byte-identical, eight lanes.
///
/// # Safety
/// The caller must have verified AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn rms_row_avx2(o: &mut [f32], i: &[f32], w: &[f32], eps: f64) {
    rms_row(o, i, w, eps);
}

/// [`rms_row`] through its AVX2 twin when `avx2` (from `avx2_only`) says so.
#[inline(always)]
fn rms_row_dispatch(avx2: bool, o: &mut [f32], i: &[f32], w: &[f32], eps: f64) {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        // SAFETY: `avx2` came from `crate::siglip::avx2_only()`.
        #[allow(unsafe_code)]
        unsafe {
            rms_row_avx2(o, i, w, eps);
        }
        return;
    }
    let _ = avx2;
    rms_row(o, i, w, eps);
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
                    for o in &mut o[..lim] {
                        *o *= inv;
                    }
                    // The masked tail is exactly zero — what softmax over
                    // `-inf` produces, without producing it.
                    for o in &mut o[lim..] {
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
        let row_fn = |(r, row): (usize, &mut [f32])| {
                let qpos = r % q_len;
                let lim = (qpos + offset + 1).min(k_len);
                // Vectorised max (AVX2 / NEON / simd128). Exact: `max` is
                // associative on non-NaN floats, so this is bit-identical to
                // the scalar `if x > m` walk it replaces, which LLVM would not
                // lane (a loop-carried reduction).
                let m = ffai_core::fastmath::max_f32(&row[..lim]);
                // `exp(x - m)` and the row total in one vectorised pass. The
                // scalar loop it replaces carried `sum += e` from iteration to
                // iteration, a non-associative reduction LLVM will not lane, so
                // the polynomial `exp` ran one element at a time: ~176 M of
                // them per 1142-token prefill. The lane split reassociates the
                // sum, so this is float-close (token-gated), not bit-identical.
                let sum = ffai_core::fastmath::exp_sub_sum_inplace(&mut row[..lim], m);
                let inv = 1.0 / sum;
                for x in &mut row[..lim] {
                    *x *= inv;
                }
                // Masked tail is exactly zero — what softmax over `-inf` gives.
                for x in &mut row[lim..] {
                    *x = 0.0;
                }
            };
        // A decode step calls this on ONE row per head (`decode_gemv`): 270
        // single-row calls per token, each paying a rayon split/join that costs
        // more than the row. Serial below the same threshold the other row ops
        // use; prefill-sized blocks stay parallel.
        if end - start < SERIAL_BELOW {
            buf[start..end].chunks_mut(k_len).enumerate().for_each(row_fn);
        } else {
            buf[start..end].par_chunks_mut(k_len).enumerate().for_each(row_fn);
        }
        crate::cost::transcendental_vector((rows * (k_len + 1) / 2) as u64);
        crate::cost::elementwise((rows * k_len) as u64, 1, 1);
        Ok(())
    }
}

/// [`CausalSoftmaxInplace`] WITHOUT the normalising divide: `exp(x - max)` in
/// place, the row totals handed back through `sums`.
///
/// # The pass this deletes
///
/// The divide pass reads and writes every live score — `rows * k_len / 2`
/// elements, 5.9 M per layer at seq 1142 — to apply one reciprocal per row.
/// Softmax feeds straight into `attn.v`, and the divide commutes with it
/// (`SUM_j (p_j / S) v_j == (SUM_j p_j v_j) / S`), so it is applied to the
/// `(seq, head_dim)` OUTPUT instead, inside the transpose-back copy the layer
/// already pays for ([`crate::siglip::AttnMergeOp`], the vision tower's
/// version of the same fold). 64 multiplies per row instead of up to 1142.
///
/// Float-close, not bit-identical: the divide moves past a sum. Each row's
/// total is at least 1 (the max contributes `exp(0)`), so the reciprocal can
/// never overflow.
struct CausalExpInplace {
    offset: usize,
    sums: crate::siglip::RowSums,
    rows: usize,
}

impl candle_core::InplaceOp1 for CausalExpInplace {
    fn name(&self) -> &'static str {
        "ffai-causal-exp-inplace"
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
            candle_core::bail!("ffai-causal-exp-inplace expects a contiguous input")
        };
        let candle_core::CpuStorage::F32(buf) = storage else {
            candle_core::bail!("ffai-causal-exp-inplace expects f32")
        };
        if k_len == 0 || (end - start) != self.rows * k_len {
            candle_core::bail!("ffai-causal-exp-inplace: {} vs {}x{k_len}", end - start, self.rows);
        }
        let offset = self.offset;
        let sums = &self.sums;
        // Resolved once per layer, not probed twice per row.
        let rk = ffai_core::fastmath::row_kernels();
        let row_fn = |(r, row): (usize, &mut [f32])| {
                let qpos = r % q_len;
                let lim = (qpos + offset + 1).min(k_len);
                let m = (rk.max)(&row[..lim]);
                // `m` is this row's max, so the exp's upper clamp is dead.
                let sum = (rk.exp_sub_rowmax_sum)(&mut row[..lim], m);
                // Masked tail is exactly zero — what softmax over `-inf` gives.
                // An explicit store loop, NOT `row[lim..].fill(0.0)`: the fill
                // lowered to a `memset` call per row and cost 1.4 G Ir more per
                // prefill than these inline vector stores (Ir campaign, 1c).
                for x in &mut row[lim..] {
                    *x = 0.0;
                }
                // SAFETY: `r < self.rows` (the chunks partition exactly
                // `rows * k_len`), and this chunk is the only writer of `r`.
                #[allow(unsafe_code)]
                unsafe {
                    *sums.0.add(r) = sum;
                }
            };
        // A decode step is 9 rows of one layer's cache: serial below the
        // shared threshold — rayon's split/join cost more than the rows.
        if end - start < SERIAL_BELOW {
            buf[start..end].chunks_mut(k_len).enumerate().for_each(row_fn);
        } else {
            buf[start..end].par_chunks_mut(k_len).enumerate().for_each(row_fn);
        }
        crate::cost::transcendental_vector((self.rows * (k_len + 1) / 2) as u64);
        crate::cost::elementwise((self.rows * k_len) as u64, 1, 1);
        Ok(())
    }
}

/// `xs = y + x` AND `rms_norm(xs)` in one pass over the rows.
///
/// The residual add used to land in `y` (an in-place add op) and the RMSNorm that
/// follows read the whole of it straight back. Here each row of the sum is
/// formed and normalised while it is still in L1. Output is one `2 n` buffer
/// shaped `(2, ..shape)`: index 0 the sum, index 1 its normalised rows.
/// Byte-identical: the sum is the in-place add's `y + x`, the norm is [`rms_row`].
struct AddRmsOp {
    w: std::sync::Arc<[f32]>,
    eps: f64,
}

impl candle_core::CustomOp2 for AddRmsOp {
    fn name(&self) -> &'static str {
        "ffai-add-rms"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        let (candle_core::CpuStorage::F32(a), candle_core::CpuStorage::F32(b)) = (s1, s2) else {
            candle_core::bail!("ffai-add-rms expects f32")
        };
        let (Some((ao, ae)), Some((bo, be))) = (l1.contiguous_offsets(), l2.contiguous_offsets())
        else {
            candle_core::bail!("ffai-add-rms expects contiguous inputs")
        };
        let (a, b) = (&a[ao..ae], &b[bo..be]);
        let h = self.w.len();
        if h == 0 || a.len() != b.len() || a.len() % h != 0 {
            candle_core::bail!("ffai-add-rms: {} vs {} width {h}", a.len(), b.len());
        }
        let n = a.len();
        let (w, eps) = (&self.w[..], self.eps);
        let avx2 = crate::siglip::avx2_only();
        // Consumed only by the x86_64 arm below.
        #[cfg(not(target_arch = "x86_64"))]
        let _ = avx2;
        let mut out: Vec<f32> = Vec::with_capacity(2 * n);
        {
            let spare = out.spare_capacity_mut();
            // SAFETY: exactly `2n` contiguous `MaybeUninit<f32>`; both halves
            // are partitioned into rows of `h` and every row writes all of its
            // elements in both halves before `set_len` publishes them.
            #[allow(unsafe_code)]
            let dst: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), 2 * n) };
            let (sum, normed) = dst.split_at_mut(n);
            let row = |(((s, o), x), y): crate::siglip::RowPair<'_>| {
                #[cfg(target_arch = "x86_64")]
                if avx2 {
                    // SAFETY: `avx2` came from `crate::siglip::avx2_only()`.
                    #[allow(unsafe_code)]
                    unsafe {
                        add_rms_row_avx2(s, o, x, y, w, eps);
                    }
                    return;
                }
                add_rms_row(s, o, x, y, w, eps);
            };
            if n < SERIAL_BELOW {
                sum.chunks_mut(h).zip(normed.chunks_mut(h)).zip(a.chunks(h)).zip(b.chunks(h)).for_each(row);
            } else {
                sum.par_chunks_mut(h)
                    .zip(normed.par_chunks_mut(h))
                    .zip(a.par_chunks(h))
                    .zip(b.par_chunks(h))
                    .for_each(row);
            }
            crate::cost::elementwise(n as u64, 2, 2);
        }
        // SAFETY: every element written above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(2 * n);
        }
        let mut dims = vec![2];
        dims.extend_from_slice(l1.dims());
        Ok((candle_core::CpuStorage::F32(out), dims.into()))
    }
}

/// `(b, seq, heads*hd)` projection -> `(b, heads, seq, hd)` with `RoPE` applied
/// on the way: the head transpose and candle's rotary embedding as ONE pass.
///
/// # The passes this deletes
///
/// The tower used to `reshape -> transpose -> contiguous` (a strided copy of
/// the whole projection) and then hand the result to candle's `rope`, which
/// allocates a zero-filled output and writes it again — two passes and two
/// allocations per tensor per layer, where one does.
///
/// # Bit-identical
///
/// The rotation is candle's non-interleaved kernel verbatim
/// (`candle-nn/src/rotary_emb.rs`): `x1*c - x2*s` and `x1*s + x2*c`, the same
/// operands in the same order, with `cos`/`sin` read from host copies of the
/// SAME tables (`rope_tables`). Rust does not contract a multiply-add, so the
/// floats match bit for bit.
struct RopeHeads {
    cos: std::sync::Arc<[f32]>,
    sin: std::sync::Arc<[f32]>,
    /// Absolute position of the first row.
    pos: usize,
    heads: usize,
    hd: usize,
    /// Column where this tensor's `heads * hd` values start in each input row
    /// (0 unless the input is the fused q|k|v projection).
    col: usize,
}

/// Rotate one `hd`-float row at absolute position `p` from `src` into `o`.
#[inline(always)]
fn rope_row(o: &mut [f32], src: &[f32], cos: &[f32], sin: &[f32], p: usize, hd: usize) {
    let half = hd / 2;
    let (cs, sn) = (&cos[p * half..(p + 1) * half], &sin[p * half..(p + 1) * half]);
    let (x1, x2) = src.split_at(half);
    let (o1, o2) = o.split_at_mut(half);
    for i in 0..half {
        o1[i] = x1[i] * cs[i] - x2[i] * sn[i];
        o2[i] = x1[i] * sn[i] + x2[i] * cs[i];
    }
}

impl candle_core::CustomOp1 for RopeHeads {
    fn name(&self) -> &'static str {
        "ffai-rope-heads"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        let candle_core::CpuStorage::F32(x) = storage else {
            candle_core::bail!("ffai-rope-heads expects f32")
        };
        let Some((o0, o1)) = layout.contiguous_offsets() else {
            candle_core::bail!("ffai-rope-heads expects a contiguous input")
        };
        let x = &x[o0..o1];
        let (b, seq, wide) = layout.shape().dims3()?;
        let (heads, hd, col) = (self.heads, self.hd, self.col);
        if col + heads * hd > wide || (self.pos + seq) * (hd / 2) > self.cos.len() {
            candle_core::bail!("ffai-rope-heads: width {wide} vs {heads}x{hd} at {col}, pos {}", self.pos);
        }
        let n = b * seq * heads * hd;
        let mut out: Vec<f32> = Vec::with_capacity(n);
        {
            let spare = out.spare_capacity_mut();
            // SAFETY: exactly `n` contiguous `MaybeUninit<f32>`; the chunks
            // below are the `b * heads` blocks of `seq * hd`, and every row of
            // every block is fully written by `rope_row` before `set_len`.
            #[allow(unsafe_code)]
            let dst: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n) };
            let (cos, sin) = (&self.cos[..], &self.sin[..]);
            let block = |(bh, blk): (usize, &mut [f32])| {
                let (bb, h) = (bh / heads, bh % heads);
                for (t, o) in blk.chunks_mut(hd).enumerate() {
                    let s0 = (bb * seq + t) * wide + col + h * hd;
                    rope_row(o, &x[s0..s0 + hd], cos, sin, self.pos + t, hd);
                }
            };
            if n < SERIAL_BELOW {
                dst.chunks_mut(seq * hd).enumerate().for_each(block);
            } else {
                dst.par_chunks_mut(seq * hd).enumerate().for_each(block);
            }
        }
        // SAFETY: every element written above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(n);
        }
        crate::cost::copy(n as u64);
        Ok((candle_core::CpuStorage::F32(out), (b, heads, seq, hd).into()))
    }
}

/// A `(b, seq, heads*hd)` projection written into rows `[pos, pos + seq)` of a
/// `(b, heads, cap, hd)` KV cache, transposed to heads on the way and — when
/// `rope` is set — rotated by [`rope_row`].
///
/// Replaces `reshape -> transpose -> contiguous` (copy 1), candle's `rope`
/// (copy 2, zero-filled allocation) and [`KvAppend`] (copy 3) with a single
/// pass from the projection into the cache. Same arithmetic as [`RopeHeads`]:
/// byte-identical.
struct HeadsIntoCache {
    pos: usize,
    rope: Option<crate::siglip::LnParams>,
    /// Column where this tensor's values start in each source row.
    col: usize,
}

impl candle_core::InplaceOp2 for HeadsIntoCache {
    fn name(&self) -> &'static str {
        "ffai-heads-into-cache"
    }

    fn cpu_fwd(
        &self,
        dst: &mut candle_core::CpuStorage,
        dl: &candle_core::Layout,
        src: &candle_core::CpuStorage,
        sl: &candle_core::Layout,
    ) -> CandleResult<()> {
        let candle_core::CpuStorage::F32(dst) = dst else {
            candle_core::bail!("ffai-heads-into-cache expects f32")
        };
        let candle_core::CpuStorage::F32(src) = src else {
            candle_core::bail!("ffai-heads-into-cache expects f32")
        };
        let (Some((dof, _)), Some((sof, sen))) = (dl.contiguous_offsets(), sl.contiguous_offsets())
        else {
            candle_core::bail!("ffai-heads-into-cache expects contiguous buffers")
        };
        let (b, heads, cap, hd) = dl.shape().dims4()?;
        let (sb, seq, wide) = sl.shape().dims3()?;
        if sb != b || self.col + heads * hd > wide || self.pos + seq > cap {
            candle_core::bail!("ffai-heads-into-cache: src {sb}x{seq}x{wide} vs cache {b}x{heads}x{cap}x{hd} at {}", self.pos);
        }
        let src = &src[sof..sen];
        for bb in 0..b {
            for h in 0..heads {
                for t in 0..seq {
                    let d0 = dof + ((bb * heads + h) * cap + self.pos + t) * hd;
                    let s0 = (bb * seq + t) * wide + self.col + h * hd;
                    let (o, x) = (&mut dst[d0..d0 + hd], &src[s0..s0 + hd]);
                    match &self.rope {
                        Some((cos, sin)) => rope_row(o, x, cos, sin, self.pos + t, hd),
                        None => o.copy_from_slice(x),
                    }
                }
            }
        }
        crate::cost::copy((b * heads * seq * hd) as u64);
        Ok(())
    }
}

/// Attention for ONE query row per head — the decode step — streamed over the
/// cache.
///
/// # What gemm was doing here
///
/// After the GQA fold a decode step's products are `(3, 64) @ (64, k_len)` and
/// `(3, k_len) @ (k_len, 64)` per kv head. Three rows is too many for gemm's
/// gemv path and far too few to amortise its packing, so every step PACKED the
/// whole K and V cache: ~105 M floats over eight steps at position 1142, 173 M
/// instructions of `pack_operands` — more than the attention arithmetic.
///
/// This touches K once and V once, straight from the cache layout: a dot
/// product per key, the causal softmax (the whole cache is visible to the last
/// row), and an AXPY per value row, normalised at the end.
///
/// Float-close to the gemm path (the sums associate differently), token-gated.
struct DecodeAttn {
    heads: usize,
    kv_heads: usize,
}

/// One head of [`DecodeAttn`]: `o = softmax(q . K^T) . V`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn decode_head(
    o: &mut [f32],
    q: &[f32],
    k: &[f32],
    k0: usize,
    ksj: usize,
    v: &[f32],
    v0: usize,
    vsj: usize,
    scores: &mut [f32],
    rk: ffai_core::fastmath::RowKernels,
) {
    let hd = q.len();
    for (j, s) in scores.iter_mut().enumerate() {
        let kr = &k[k0 + j * ksj..k0 + j * ksj + hd];
        let mut acc = [0f32; 8];
        for (qc, kc) in q.chunks_exact(8).zip(kr.chunks_exact(8)) {
            for l in 0..8 {
                acc[l] += qc[l] * kc[l];
            }
        }
        let mut d = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (a, b) in q.chunks_exact(8).remainder().iter().zip(kr.chunks_exact(8).remainder()) {
            d += a * b;
        }
        *s = d;
    }
    let m = (rk.max)(scores);
    let sum = (rk.exp_sub_rowmax_sum)(scores, m);
    for x in o.iter_mut() {
        *x = 0.0;
    }
    for (j, &p) in scores.iter().enumerate() {
        let vr = &v[v0 + j * vsj..v0 + j * vsj + hd];
        for (x, &vv) in o.iter_mut().zip(vr) {
            *x += p * vv;
        }
    }
    let inv = 1.0 / sum;
    for x in o.iter_mut() {
        *x *= inv;
    }
}

/// [`decode_head`] compiled for AVX2 (no `fma`): byte-identical, eight lanes
/// for the dot products and the value AXPY.
///
/// # Safety
/// The caller must have verified AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn decode_head_avx2(
    o: &mut [f32],
    q: &[f32],
    k: &[f32],
    k0: usize,
    ksj: usize,
    v: &[f32],
    v0: usize,
    vsj: usize,
    scores: &mut [f32],
    rk: ffai_core::fastmath::RowKernels,
) {
    decode_head(o, q, k, k0, ksj, v, v0, vsj, scores, rk);
}

impl candle_core::CustomOp3 for DecodeAttn {
    fn name(&self) -> &'static str {
        "ffai-decode-attn"
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
        let (
            candle_core::CpuStorage::F32(q),
            candle_core::CpuStorage::F32(k),
            candle_core::CpuStorage::F32(v),
        ) = (s1, s2, s3)
        else {
            candle_core::bail!("ffai-decode-attn expects f32")
        };
        let Some((qo, qe)) = l1.contiguous_offsets() else {
            candle_core::bail!("ffai-decode-attn expects a contiguous q")
        };
        let q = &q[qo..qe];
        let (b, heads, one, hd) = l1.shape().dims4()?;
        let (kb, kvh, k_len, khd) = l2.shape().dims4()?;
        let (ks, vs) = (l2.stride(), l3.stride());
        if one != 1 || heads != self.heads || kvh != self.kv_heads || kb != b || khd != hd
            || l3.dims() != l2.dims() || ks[3] != 1 || vs[3] != 1
        {
            candle_core::bail!("ffai-decode-attn: q {:?} k {:?} v {:?}", l1.dims(), l2.dims(), l3.dims());
        }
        let reps = heads / kvh;
        let (ko, vo) = (l2.start_offset(), l3.start_offset());
        let mut out = vec![0f32; b * heads * hd];
        let mut scores = vec![0f32; k_len];
        let avx2 = crate::siglip::avx2_only();
        // Consumed only by the x86_64 arm below.
        #[cfg(not(target_arch = "x86_64"))]
        let _ = avx2;
        let rk = ffai_core::fastmath::row_kernels();
        for bb in 0..b {
            for h in 0..heads {
                let g = h / reps;
                let qr = &q[(bb * heads + h) * hd..(bb * heads + h + 1) * hd];
                let o = &mut out[(bb * heads + h) * hd..(bb * heads + h + 1) * hd];
                #[cfg(target_arch = "x86_64")]
                if avx2 {
                    // SAFETY: `avx2` came from `crate::siglip::avx2_only()`.
                    #[allow(unsafe_code)]
                    unsafe {
                        decode_head_avx2(
                            o,
                            qr,
                            k,
                            ko + bb * ks[0] + g * ks[1],
                            ks[2],
                            v,
                            vo + bb * vs[0] + g * vs[1],
                            vs[2],
                            &mut scores,
                            rk,
                        );
                    }
                    continue;
                }
                let _ = avx2;
                decode_head(
                    o,
                    qr,
                    k,
                    ko + bb * ks[0] + g * ks[1],
                    ks[2],
                    v,
                    vo + bb * vs[0] + g * vs[1],
                    vs[2],
                    &mut scores,
                    rk,
                );
            }
        }
        crate::cost::matmul((b * heads) as u64, 1, hd as u64, k_len as u64);
        crate::cost::matmul((b * heads) as u64, 1, k_len as u64, hd as u64);
        Ok((candle_core::CpuStorage::F32(out), (b, 1, heads * hd).into()))
    }
}

/// One row of [`AddRmsOp`]: `s = x + y`, then `o = rms_norm(s)`.
#[inline(always)]
fn add_rms_row(s: &mut [f32], o: &mut [f32], x: &[f32], y: &[f32], w: &[f32], eps: f64) {
    for ((s, &x), &y) in s.iter_mut().zip(x).zip(y) {
        *s = x + y;
    }
    rms_row(o, s, w, eps);
}

/// [`add_rms_row`] compiled for AVX2 (no `fma`): byte-identical. Counted
/// (vein 8, brick 8c): with only the norm half behind the twin, the fused op
/// cost 0.35 M per call MORE than the in-place add + `rms_norm`; the add has to be
/// inside the twin too.
///
/// # Safety
/// The caller must have verified AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn add_rms_row_avx2(s: &mut [f32], o: &mut [f32], x: &[f32], y: &[f32], w: &[f32], eps: f64) {
    add_rms_row(s, o, x, y, w, eps);
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
    // `mm::matmul`: candle's gemm call minus the zero fill of the output, and
    // straight on the 3-D activation — no reshape to 2-D and back. Those two
    // reshapes were two tensor operations (and their allocations) per
    // projection, 7 projections x 30 layers per decoded token, for the same
    // gemm call (vein 15, the decode step's fixed overhead).
    let _ = (b, s, i, o);
    crate::mm::matmul(x, &w.t()?)
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
    use crate::clock::Instant;

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

    /// A start time for [`add`], or `None` — WITHOUT reading the clock —
    /// when profiling is off. The forwards time every op, and an unread
    /// `Instant::now()` is still a `clock_gettime` per op: ~2,400 of them in
    /// an eight-token decode (Ir campaign, vein 14).
    #[inline]
    pub(crate) fn now() -> Option<Instant> {
        on().then(Instant::now)
    }

    pub(crate) fn add(name: &'static str, t: Option<Instant>) {
        let Some(t) = t else {
            return;
        };
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
        // Eight lanes with FMA (`fastmath::silu_mul_inplace`); the scalar loop
        // it replaces did not vectorise on the SSE2 baseline.
        let apply = |g: &mut [f32], u: &[f32]| ffai_core::fastmath::silu_mul_inplace(g, u);
        // Serial when already on a pool thread: a nested rayon region inside a
        // tile worker costs more than the work.
        if crate::par::current_thread_index().is_none() {
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

/// `(.., 2*inter)` rows of `[gate | up]` -> a DENSE `(b*seq, inter)` product.
///
/// The fused gate|up GEMM halves the calls and the passes over the normalised
/// activation; writing the product out of place keeps the down projection's
/// left operand contiguous. Writing it in place on the gate half (12a) handed
/// gemm a row-strided operand, which it packs through a slower generic path:
/// +131.8 M per prefill, more than the fusion saved.
struct SwiGluOut {
    inter: usize,
}

impl candle_core::CustomOp1 for SwiGluOut {
    fn name(&self) -> &'static str {
        "ffai-swiglu-out"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        let candle_core::CpuStorage::F32(x) = storage else {
            candle_core::bail!("ffai-swiglu-out expects f32")
        };
        let Some((o0, o1)) = layout.contiguous_offsets() else {
            candle_core::bail!("ffai-swiglu-out expects a contiguous input")
        };
        let x = &x[o0..o1];
        let (inter, w) = (self.inter, 2 * self.inter);
        if w == 0 || x.len() % w != 0 {
            candle_core::bail!("ffai-swiglu-out: {} not rows of {w}", x.len());
        }
        let rows = x.len() / w;
        let n = rows * inter;
        let mut out: Vec<f32> = Vec::with_capacity(n);
        {
            let spare = out.spare_capacity_mut();
            // SAFETY: exactly `n` contiguous `MaybeUninit<f32>`; each of the
            // `rows` chunks of `inter` is fully written by `silu_mul_into`
            // before `set_len`.
            #[allow(unsafe_code)]
            let dst: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n) };
            let row = |(o, r): (&mut [f32], &[f32])| {
                let (g, u) = r.split_at(inter);
                ffai_core::fastmath::silu_mul_into(o, g, u);
            };
            // Serial when already on a pool thread: a nested rayon region inside a
        // tile worker costs more than the work.
            if crate::par::current_thread_index().is_none() {
                dst.par_chunks_mut(inter).zip(x.par_chunks(w)).for_each(row);
            } else {
                dst.chunks_mut(inter).zip(x.chunks(w)).for_each(row);
            }
        }
        // SAFETY: every element written above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(n);
        }
        crate::cost::transcendental_vector(n as u64);
        crate::cost::elementwise(n as u64, 2, 1);
        Ok((candle_core::CpuStorage::F32(out), (rows, inter).into()))
    }
}

/// The text tower: embeddings, blocks, final norm, `lm_head`.
pub struct TextTower {
    embed: Tensor,
    blocks: Vec<Block>,
    norm: Tensor,
    lm_head: Tensor,
    /// Host copies of the `RoPE` cos / sin tables, `(max_pos, head_dim / 2)` row-major, for
    /// the fused transpose+`RoPE` ops ([`RopeHeads`]).
    cos_host: std::sync::Arc<[f32]>,
    sin_host: std::sync::Arc<[f32]>,
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
            let ln2: Tensor = l.pp("post_attention_layernorm").get(cfg.hidden, "weight")?;
            let ln2_host: Vec<f32> = ln2.to_vec1()?;
            blocks.push(Block {
                ln1: l.pp("input_layernorm").get(cfg.hidden, "weight")?,
                ln2_host: ln2_host.into(),
                // THE FOLD. Exact: `1/sqrt(64) = 0.125`, a power of two.
                qkv: Tensor::cat(
                    &[
                        &(a.pp("q_proj").get((qh, cfg.hidden), "weight")? * scale)?,
                        &a.pp("k_proj").get((kh, cfg.hidden), "weight")?,
                        &a.pp("v_proj").get((kh, cfg.hidden), "weight")?,
                    ],
                    0,
                )?
                .contiguous()?,
                o: a.pp("o_proj").get((cfg.hidden, qh), "weight")?,
                gate_up: Tensor::cat(
                    &[
                        &f.pp("gate_proj").get((cfg.inter, cfg.hidden), "weight")?,
                        &f.pp("up_proj").get((cfg.inter, cfg.hidden), "weight")?,
                    ],
                    0,
                )?
                .contiguous()?,
                down: f.pp("down_proj").get((cfg.hidden, cfg.inter), "weight")?,
            });
        }
        let (cos, sin) = rope_tables(&cfg, device)?;
        let cos_host: Vec<f32> = cos.flatten_all()?.to_vec1()?;
        let sin_host: Vec<f32> = sin.flatten_all()?.to_vec1()?;
        Ok(Self {
            cos_host: cos_host.into(),
            sin_host: sin_host.into(),
            embed: m.pp("embed_tokens").get((49280, cfg.hidden), "weight")?,
            blocks,
            norm: m.pp("norm").get(cfg.hidden, "weight")?,
            lm_head: vb.pp("lm_head").get((49280, cfg.hidden), "weight")?,
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
        let logits = crate::mm::matmul(&x, &self.lm_head.t()?)?;
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
        let t = prof::now();
        let normed = rms_norm(xs, &blk.ln1, c.eps)?;
        prof::add("rms_norm", t);
        // Three matmuls, deliberately — see [`Block::q`] for the measured
        // refutation of fusing them.
        let t = prof::now();
        // Prefill: ONE GEMM for q, k and v, each consumer below reading its
        // own columns of the fused output. Decode (one row): three m = 1
        // products over row views of the same weight — the fused m = 1
        // product measured +2.02 M over eight steps (12b), the fused prefill
        // GEMM wins. Same views either way: no weight is duplicated.
        let (qw, kw) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
        let (q, k, v, kcol, vcol) = if seq > 1 {
            let qkv = linear(&normed, &blk.qkv)?;
            (qkv.clone(), qkv.clone(), qkv, qw, qw + kw)
        } else {
            (
                linear(&normed, &blk.qkv.narrow(0, 0, qw)?)?,
                linear(&normed, &blk.qkv.narrow(0, qw, kw)?)?,
                linear(&normed, &blk.qkv.narrow(0, qw + kw, kw)?)?,
                0,
                0,
            )
        };
        prof::add("qkv proj", t);
        crate::cost::matmul(1, bu * sq, hd, (c.heads * c.head_dim) as u64);
        crate::cost::matmul(2, bu * sq, hd, (c.kv_heads * c.head_dim) as u64);

        let t = prof::now();
        // q: head transpose and RoPE in one pass ([`RopeHeads`]).
        let q = q.apply_op1_no_bwd(&RopeHeads {
            cos: self.cos_host.clone(),
            sin: self.sin_host.clone(),
            pos: index_pos,
            heads: c.heads,
            hd: c.head_dim,
            col: 0,
        })?;
        // k and v are NOT transposed here: each goes straight from its
        // projection into its cache slot, k rotated on the way
        // ([`HeadsIntoCache`]).
        prof::add("qkv reshape+transpose", t);



        // KV cache: write the new positions INTO a preallocated buffer.
        //
        // This was `Tensor::cat(&[&prev, &new], 2)`, which reallocates the full
        // length and recopies the entire history every step — 52.7 MB per
        // token at position 1142, measured at 14.0 ms and 26 % of a decode
        // step, growing quadratically with the generation. See [`KvAppend`].
        let t = prof::now();
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
            bk.inplace_op2(
                &k,
                &HeadsIntoCache {
                    pos: index_pos,
                    rope: Some((self.cos_host.clone(), self.sin_host.clone())),
                    col: kcol,
                },
            )?;
            bv.inplace_op2(&v, &HeadsIntoCache { pos: index_pos, rope: None, col: vcol })?;
            let used = index_pos + seq;
            // Narrowing axis 2 keeps each head's slice contiguous, so candle's
            // batched matmul takes it without reintroducing a copy.
            (bk.narrow(2, 0, used)?, bv.narrow(2, 0, used)?)
        };
        // The tower advances the shared length once per layer-0 visit; every
        // layer sees the same positions, so tracking it on the last layer
        // written keeps it correct for both prefill and decode.
        self.kv_len = index_pos + seq;
        prof::add("kv cache", t);

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
        let t = prof::now();
        let y = if seq > CAUSAL_CHUNK {
            attend_chunked(&q, &k, &v, index_pos, &c)?
        } else {
            attend(&q, &k, &v, index_pos, &c)?
        };
        prof::add("attention", t);
        let t = prof::now();
        let y = linear(&y, &blk.o)?;
        prof::add("o proj", t);
        crate::cost::matmul(1, bu * sq, hd, hd);
        let t = prof::now();
        // The residual sum and ln2's RMSNorm in one row pass ([`AddRmsOp`]).
        let both = y.apply_op2_no_bwd(xs, &AddRmsOp { w: blk.ln2_host.clone(), eps: c.eps })?;
        let (xs, normed) = (both.i(0)?, both.i(1)?);
        prof::add("residual+rms_norm", t);

        // ---- mlp ----------------------------------------------------------
        let t = prof::now();
        // gate and up: ONE GEMM at decode, two at prefill. Counted both ways
        // (vein 12): the fused product won -1.23 M per eight decode steps (one
        // m = 1 gemv instead of two) and lost +175.4 M per prefill (an
        // N = 3072 GEMM plus the dense copy costs more than two N = 1536 ones),
        // so the fusion is split by call site. `FFAI_ARGUS_GATEUP` =
        // `decode` (default) | `never` | `always` keeps both measurable.
        let fused = match gateup_arm() {
            GateUp::Always => true,
            GateUp::Never => false,
            GateUp::Decode => seq == 1,
        };
        let h = if fused {
            let gu = linear(&normed, &blk.gate_up)?;
            prof::add("gate+up proj", t);
            crate::cost::matmul(2, bu * sq, hd, c.inter as u64);
            gu.apply_op1_no_bwd(&SwiGluOut { inter: c.inter })?
        } else {
            let g = linear(&normed, &blk.gate_up.narrow(0, 0, c.inter)?)?;
            let u = linear(&normed, &blk.gate_up.narrow(0, c.inter, c.inter)?)?;
            prof::add("gate+up proj", t);
            crate::cost::matmul(2, bu * sq, hd, c.inter as u64);
            g.inplace_op2(&u, &SwiGluInplace)?;
            g.reshape((b * seq, c.inter))?
        };
        let t = prof::now();
        prof::add("swiglu", t);
        let t = prof::now();
        // `xs` is this block's own residual buffer (from `AddRmsOp`), so the
        // down projection accumulates straight into it: `xs += h @ down^T`.
        crate::mm::matmul_acc_into(&xs, &h, &blk.down.t()?)?;
        prof::add("down proj + residual", t);
        crate::cost::matmul(1, bu * sq, c.inter as u64, hd);
        Ok(xs)
    }
}

/// Where gate|up run as one GEMM (`FFAI_ARGUS_GATEUP`, read once).
#[derive(Clone, Copy, PartialEq, Eq)]
enum GateUp {
    /// Fused at decode (`seq == 1`), separate at prefill — the default.
    Decode,
    Never,
    Always,
}

fn gateup_arm() -> GateUp {
    use std::sync::atomic::{AtomicU8, Ordering};
    static C: AtomicU8 = AtomicU8::new(u8::MAX);
    let v = match C.load(Ordering::Relaxed) {
        u8::MAX => {
            let v = match std::env::var("FFAI_ARGUS_GATEUP").ok().as_deref() {
                Some("never") => 1,
                Some("always") => 2,
                _ => 0,
            };
            C.store(v, Ordering::Relaxed);
            v
        }
        v => v,
    };
    match v {
        1 => GateUp::Never,
        2 => GateUp::Always,
        _ => GateUp::Decode,
    }
}

/// Which decode-step attention runs (`FFAI_ARGUS_DECODE_ATTN`, read once).
#[derive(Clone, Copy, PartialEq, Eq)]
enum DecodeArm {
    /// [`DecodeDirect`] — the default.
    Direct,
    /// Per-head m = 1 products as candle tensors ([`decode_gemv`]).
    Gemv,
    /// The batched GQA products, `(3, 64) @ (64, k_len)` per kv head.
    Gemm,
    /// [`DecodeAttn`]'s hand-written streaming kernel.
    Stream,
}

fn decode_attn_arm() -> DecodeArm {
    use std::sync::atomic::{AtomicU8, Ordering};
    static C: AtomicU8 = AtomicU8::new(u8::MAX);
    let v = match C.load(Ordering::Relaxed) {
        u8::MAX => {
            let v = match std::env::var("FFAI_ARGUS_DECODE_ATTN").ok().as_deref() {
                Some("stream") => 2,
                Some("gemv") => 3,
                Some("direct") => 4,
                _ => 0,
            };
            C.store(v, Ordering::Relaxed);
            v
        }
        v => v,
    };
    // The batched GQA products are the default: three alternatives that
    // avoid gemm's per-step repacking of the cache were each counted and each
    // LOST (stream +955.6 M, per-head tensors +93.2 M, direct m = 1 gemm
    // +62.5 M over eight steps) — gemm's gemv path computes these strided
    // products less efficiently than the packed microkernel.
    match v {
        2 => DecodeArm::Stream,
        3 => DecodeArm::Gemv,
        4 => DecodeArm::Direct,
        _ => DecodeArm::Gemm,
    }
}

/// Decode-step attention: per head, two DIRECT `m = 1` gemm calls.
///
/// The batched GQA products pack the whole K and V cache every step (three
/// query rows per kv head is too many for gemm's gemv path); 173 M of an
/// 805 M eight-step decode was `pack_operands`. One query row per call takes
/// the gemv path and packs nothing — but doing that through candle tensors
/// ([`decode_gemv`]) added ~95 K tensor operations and measured +93 M. Here
/// the same two products per head are plain `gemm::gemm` calls on the cache's
/// own memory, inside one op: `scores = q_h . K_g^T`, the softmax in place,
/// `o_h = p . V_g`, with the divide deferred to the 64 outputs.
///
/// Float-close to the batched path (summation order), token-gated.
struct DecodeDirect {
    heads: usize,
    kv_heads: usize,
}

impl candle_core::CustomOp3 for DecodeDirect {
    fn name(&self) -> &'static str {
        "ffai-decode-direct"
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
        let (
            candle_core::CpuStorage::F32(q),
            candle_core::CpuStorage::F32(k),
            candle_core::CpuStorage::F32(v),
        ) = (s1, s2, s3)
        else {
            candle_core::bail!("ffai-decode-direct expects f32")
        };
        let Some((qo, qe)) = l1.contiguous_offsets() else {
            candle_core::bail!("ffai-decode-direct expects a contiguous q")
        };
        let q = &q[qo..qe];
        let (b, heads, one, hd) = l1.shape().dims4()?;
        let (kb, kvh, k_len, khd) = l2.shape().dims4()?;
        let (ks, vs) = (l2.stride(), l3.stride());
        if one != 1 || heads != self.heads || kvh != self.kv_heads || kb != b || khd != hd
            || l3.dims() != l2.dims() || ks[3] != 1 || vs[3] != 1 || k_len == 0
        {
            candle_core::bail!("ffai-decode-direct: q {:?} k {:?} v {:?}", l1.dims(), l2.dims(), l3.dims());
        }
        let reps = heads / kvh;
        let (ko, vo) = (l2.start_offset(), l3.start_offset());
        let num_threads = candle_core::utils::get_num_threads();
        let par = if num_threads > 1 {
            gemm::Parallelism::Rayon(num_threads)
        } else {
            gemm::Parallelism::None
        };
        let rk = ffai_core::fastmath::row_kernels();
        let mut out = vec![0f32; b * heads * hd];
        let mut scores = vec![0f32; k_len];
        for bb in 0..b {
            for h in 0..heads {
                let g = h / reps;
                let qr = (bb * heads + h) * hd;
                let k0 = ko + bb * ks[0] + g * ks[1];
                let v0 = vo + bb * vs[0] + g * vs[1];
                // SAFETY: `q` holds `hd` floats at `qr`; key `j`, dim `d` of
                // group `g` sits at `k0 + j*ks[2] + d` (strides from the layout
                // candle validated against `k`), likewise for `v`; `scores`
                // holds `k_len` and the output row `hd` floats. gemm with
                // `read_dst = false` writes every destination element.
                #[allow(unsafe_code)]
                unsafe {
                    gemm::gemm(
                        1,
                        k_len,
                        hd,
                        scores.as_mut_ptr(),
                        1,
                        k_len as isize,
                        false,
                        q.as_ptr().add(qr),
                        1,
                        hd as isize,
                        k.as_ptr().add(k0),
                        ks[2] as isize,
                        1,
                        0.0f32,
                        1.0f32,
                        false,
                        false,
                        false,
                        par,
                    );
                }
                let m = (rk.max)(&scores);
                let sum = (rk.exp_sub_rowmax_sum)(&mut scores, m);
                let o = &mut out[qr..qr + hd];
                // SAFETY: as above, `p . V_g` into the `hd`-float output row.
                #[allow(unsafe_code)]
                unsafe {
                    gemm::gemm(
                        1,
                        hd,
                        k_len,
                        o.as_mut_ptr(),
                        1,
                        hd as isize,
                        false,
                        scores.as_ptr(),
                        1,
                        k_len as isize,
                        v.as_ptr().add(v0),
                        1,
                        vs[2] as isize,
                        0.0f32,
                        1.0f32,
                        false,
                        false,
                        false,
                        par,
                    );
                }
                let inv = 1.0 / sum;
                for x in o.iter_mut() {
                    *x *= inv;
                }
            }
        }
        crate::cost::matmul((b * heads) as u64, 1, hd as u64, k_len as u64);
        crate::cost::matmul((b * heads) as u64, 1, k_len as u64, hd as u64);
        Ok((candle_core::CpuStorage::F32(out), (b, 1, heads * hd).into()))
    }
}

/// Decode-step attention as one `m = 1` product per query head.
///
/// # Why not the batched GQA products
///
/// After the GQA fold a decode step multiplies `(3, 64) @ (64, k_len)` and
/// `(3, k_len) @ (k_len, 64)` per kv head. Three rows is too many for gemm's
/// gemv path and too few to amortise its packing, so every step packed the
/// whole K and V cache — 173 M instructions of `pack_operands` over eight steps
/// at position 1142, more than the products themselves. A single query row
/// (`m = 1`) takes the gemv path, which reads K and V where they lie.
///
/// A hand-written streaming kernel for the same job was tried first
/// ([`DecodeAttn`]) and LOST by 955 M Ir: gemm's FMA microkernels compute the
/// products far more cheaply than a lane-array loop at the SSE2 baseline.
/// This keeps gemm's arithmetic and drops only the packing.
fn decode_gemv(q: &Tensor, k: &Tensor, v: &Tensor, c: &Cfg) -> CandleResult<Tensor> {
    let heads = c.heads;
    let reps = heads / c.kv_heads;
    let k_len = k.dim(2)?;
    let mut outs = Vec::with_capacity(heads);
    for h in 0..heads {
        let g = h / reps;
        let qh = q.i((0, h))?; // (1, hd)
        let kg = k.i((0, g))?; // (k_len, hd), a view into the cache
        let scores = crate::mm::matmul(&qh, &kg.t()?)?; // (1, k_len)
        // The last row of a causal block sees the whole cache.
        let scores = scores.reshape((1, 1, 1, k_len))?;
        scores.inplace_op1(&CausalSoftmaxInplace { offset: k_len - 1 })?;
        let vg = v.i((0, g))?; // (k_len, hd)
        outs.push(crate::mm::matmul(&scores.reshape((1, k_len))?, &vg)?);
    }
    let bu = 1u64;
    crate::cost::matmul(bu * heads as u64, 1, c.head_dim as u64, k_len as u64);
    crate::cost::matmul(bu * heads as u64, 1, k_len as u64, c.head_dim as u64);
    Tensor::cat(&outs, 1)?.reshape((1, 1, c.hidden))
}

/// Defer the softmax divide past `attn.v` in the text tower?
/// `FFAI_ARGUS_TEXT_LATE_NORM=0` divides early — the measurement arm. Read once.
fn text_late_norm() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static C: AtomicU8 = AtomicU8::new(u8::MAX);
    match C.load(Ordering::Relaxed) {
        u8::MAX => {
            let v = std::env::var("FFAI_ARGUS_TEXT_LATE_NORM").ok().as_deref() != Some("0");
            C.store(u8::from(v), Ordering::Relaxed);
            v
        }
        v => v == 1,
    }
}

/// Query rows per causal chunk in prefill. See [`attend_chunked`].
const CAUSAL_CHUNK: usize = 64;

/// Causal attention for one block of query rows: `q (b, heads, seq, hd)`
/// against `k`, `v (b, kv_heads, k_len, hd)`, row `i` attending to keys
/// `0 ..= offset + i`. Returns `(b, seq, hidden)`.
///
/// GQA WITHOUT materialising `repeat_kv`: q heads `[0,1,2]` use kv head 0,
/// `[3,4,5]` kv head 1, and so on, so `(b, 9, s, hd)` is folded to
/// `(b, 3, 3*s, hd)` — the same bytes — and multiplied against the
/// un-repeated k. No `/ sqrt(head_dim)` (it is in q's weights) and no
/// `masked_fill` (causality is in [`CausalExpInplace`]).
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, offset: usize, c: &Cfg) -> CandleResult<Tensor> {
    attend_into(q, k, v, offset, c, None)
}

/// [`attend`], optionally writing its `(b, seq, hidden)` rows into rows
/// `[row0, row0 + seq)` of `dst` instead of a fresh tensor.
fn attend_into(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    offset: usize,
    c: &Cfg,
    dst: Option<(&Tensor, usize)>,
) -> CandleResult<Tensor> {
    let (b, _heads, seq, _hd) = q.dims4()?;
    if seq == 1 && dst.is_none() {
        match decode_attn_arm() {
            // Two direct m = 1 gemm calls per head inside one op ([`DecodeDirect`]).
            DecodeArm::Direct => {
                return q.apply_op3_no_bwd(k, v, &DecodeDirect { heads: c.heads, kv_heads: c.kv_heads });
            }
            // One m = 1 product per query head as candle tensors (refuted).
            DecodeArm::Gemv if b == 1 => return decode_gemv(q, k, v, c),
            // The hand-written streaming kernel (refuted, kept measurable).
            DecodeArm::Stream => {
                return q.apply_op3_no_bwd(k, v, &DecodeAttn { heads: c.heads, kv_heads: c.kv_heads });
            }
            // The batched GQA path below.
            _ => {}
        }
    }
    let k_len = k.dim(2)?;
    let reps = c.heads / c.kv_heads;
    let qg = q.reshape((b, c.kv_heads, reps * seq, c.head_dim))?;
    let att = crate::mm::matmul(&qg, &k.t()?)?;
    let (bu, sq) = (b as u64, seq as u64);
    crate::cost::matmul(bu * c.heads as u64, sq, c.head_dim as u64, k_len as u64);
    let att = att.reshape((b, c.heads, seq, k_len))?;
    if !text_late_norm() {
        // The EARLY-divide arm (`FFAI_ARGUS_TEXT_LATE_NORM=0`): normalise the
        // scores, then a plain transposing copy. Kept so the deferred divide
        // stays measurable against exactly what it replaced.
        att.inplace_op1(&CausalSoftmaxInplace { offset })?;
        let y = crate::mm::matmul(&att.reshape((b, c.kv_heads, reps * seq, k_len))?, v)?
            .reshape((b, c.heads, seq, c.head_dim))?;
        crate::cost::matmul(bu * c.heads as u64, sq, k_len as u64, c.head_dim as u64);
        crate::cost::copy(bu * sq * c.hidden as u64);
        if let Some((dst, row0)) = dst {
            let ones = Tensor::zeros((0,), y.dtype(), y.device())?;
            dst.inplace_op3(&y, &ones, &AttnMergeInto { heads: c.heads, head_dim: c.head_dim, row0, divide: false })?;
            return Ok(y);
        }
        return y.transpose(1, 2)?.reshape((b, seq, c.hidden));
    }
    let rows = b * c.heads * seq;
    let mut sums = vec![0f32; rows];
    att.inplace_op1(&CausalExpInplace {
        offset,
        sums: crate::siglip::RowSums(sums.as_mut_ptr()),
        rows,
    })?;
    let y = crate::mm::matmul(&att.reshape((b, c.kv_heads, reps * seq, k_len))?, v)?
        .reshape((b, c.heads, seq, c.head_dim))?;
    crate::cost::matmul(bu * c.heads as u64, sq, k_len as u64, c.head_dim as u64);
    // The transpose-back copy, each row divided by its softmax total.
    let sums = Tensor::from_vec(sums, (b, c.heads, seq), y.device())?;
    crate::cost::copy(bu * sq * c.hidden as u64);
    if let Some((dst, row0)) = dst {
        dst.inplace_op3(&y, &sums, &AttnMergeInto { heads: c.heads, head_dim: c.head_dim, row0, divide: true })?;
        return Ok(y);
    }
    y.apply_op2_no_bwd(
        &sums,
        &crate::siglip::AttnMergeOp { heads: c.heads, head_dim: c.head_dim },
    )
}

/// [`crate::siglip::AttnMergeOp`] writing into rows `[row0, row0 + len)` of an
/// existing `(b, seq_total, hidden)` buffer: `(b, heads, len, hd)` attention
/// output in, each row divided by its softmax total on the way.
///
/// This is what lets [`attend_chunked`] assemble its chunks without a
/// `Tensor::cat`, which copied the whole `(seq, hidden)` output a second time
/// and allocated once per chunk.
struct AttnMergeInto {
    heads: usize,
    head_dim: usize,
    row0: usize,
    /// `false` only for the early-divide measurement arm: a plain copy.
    divide: bool,
}

impl candle_core::InplaceOp3 for AttnMergeInto {
    fn name(&self) -> &'static str {
        "ffai-attn-merge-into"
    }

    fn cpu_fwd(
        &self,
        s1: &mut candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
        s3: &candle_core::CpuStorage,
        l3: &candle_core::Layout,
    ) -> CandleResult<()> {
        let (Some((d0, d1)), Some((ao, ae)), Some((so, se))) =
            (l1.contiguous_offsets(), l2.contiguous_offsets(), l3.contiguous_offsets())
        else {
            candle_core::bail!("ffai-attn-merge-into expects contiguous inputs")
        };
        let (candle_core::CpuStorage::F32(a), candle_core::CpuStorage::F32(sums)) = (s2, s3) else {
            candle_core::bail!("ffai-attn-merge-into expects f32")
        };
        let candle_core::CpuStorage::F32(dst) = s1 else {
            candle_core::bail!("ffai-attn-merge-into expects f32")
        };
        let (a, sums, dst) = (&a[ao..ae], &sums[so..se], &mut dst[d0..d1]);
        let (heads, hd) = (self.heads, self.head_dim);
        let hidden = heads * hd;
        let (b, _h, len, _) = l2.shape().dims4()?;
        let (db, seq_total, dh) = l1.shape().dims3()?;
        if db != b || dh != hidden || self.row0 + len > seq_total || (self.divide && sums.len() != b * heads * len) {
            candle_core::bail!("ffai-attn-merge-into: shapes disagree");
        }
        let avx2 = crate::siglip::avx2_only();
        // Consumed only by the x86_64 arm below.
        #[cfg(not(target_arch = "x86_64"))]
        let _ = avx2;
        for bb in 0..b {
            for s in 0..len {
                let o0 = (bb * seq_total + self.row0 + s) * hidden;
                let o = &mut dst[o0..o0 + hidden];
                if self.divide {
                    crate::siglip::merge_row_dispatch(avx2, o, a, sums, bb, s, len, heads, hd);
                    continue;
                }
                for h in 0..heads {
                    let src = (bb * heads + h) * len * hd + s * hd;
                    o[h * hd..(h + 1) * hd].copy_from_slice(&a[src..src + hd]);
                }
            }
        }
        Ok(())
    }
}

/// Causal attention for a long query block, a chunk of rows at a time.
///
/// # The half of the square that was computed to be thrown away
///
/// Unchunked, `q.k^T` and `attn.v` run over the whole `seq x k_len` square:
/// `2 x 9 x 1142^2 x 64 = 1.5 GFLOP` each per layer at the caption's prompt,
/// and then the causal kernel zeroes the upper triangle and `attn.v` multiplies
/// those zeros. Chunk `[s0, s1)` can only ever see keys `[0, offset + s1)`, so
/// giving each chunk exactly that prefix of k and v skips the triangle
/// outright: at chunk 128 the computed area is `N^2/2 + N*128/2` of `N^2`, a
/// 44 % cut of both products — ~45 GFLOP per prefill.
///
/// The `q.k^T` entries that ARE computed are the same dot products, so the
/// scores are exact; `attn.v` sums over a shorter K and gemm's blocking then
/// associates differently, so the output is float-close (token-gated), not
/// bit-identical. Unlike the three refuted vision-attention restructures, this
/// one deletes arithmetic rather than only moving bytes.
fn attend_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    offset: usize,
    c: &Cfg,
) -> CandleResult<Tensor> {
    let (b, _heads, seq, _hd) = q.dims4()?;
    let k_len = k.dim(2)?;
    // The chunks below tile rows `[0, seq)` exactly; each chunk's
    // `AttnMergeInto` writes every element of its rows.
    let out = crate::mm::zeroed_f32(&[b, seq, c.hidden], q.device())?;
    let mut s0 = 0;
    while s0 < seq {
        let len = CAUSAL_CHUNK.min(seq - s0);
        let lim = (offset + s0 + len).min(k_len);
        // Per-head rows `[s0, s0+len)` are strided across heads; the GQA fold
        // needs them contiguous. 9 x 128 x 64 floats, against the ~1.5 M
        // scores per chunk the limit removes.
        let qc = q.narrow(2, s0, len)?.contiguous()?;
        crate::cost::copy((qc.elem_count()) as u64);
        attend_into(
            &qc,
            &k.narrow(2, 0, lim)?,
            &v.narrow(2, 0, lim)?,
            offset + s0,
            c,
            Some((&out, s0)),
        )?;
        s0 += len;
    }
    Ok(out)
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
