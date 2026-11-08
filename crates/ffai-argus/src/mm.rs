//! `matmul` without the zero fill.
//!
//! candle's CPU matmul (`candle-core/src/cpu_backend/mod.rs`, `impl Map2 for
//! MatMul`) allocates its output with `vec![T::zero(); b * m * n]` and then
//! calls `gemm` with `read_dst = false`, which overwrites every element: the
//! fill is a full write of the output for nothing. At the tower's shapes that
//! is **81.6 MB per layer, 979 MB per tile** (qkv 9.4 + out 3.1 + fc1 12.6 +
//! fc2 3.1 + scores 50.3 + attn.v 3.1), and ~2.2 GB per 1142-token prefill.
//!
//! [`matmul`] makes the identical `gemm` call — same strides, same
//! `alpha = 0`, `beta = 1`, same parallelism rule, same per-batch loop — into
//! uninitialised capacity. Same arithmetic in the same order, so the product is
//! **bit-identical** to candle's; `ffai-diana`'s `cpuop::matmul2` is the same
//! technique, counted there at -20 M Ir and gated by an fnv anchor.
//!
//! It accepts exactly the layouts candle's matmul accepts at Argus's call sites
//! — each batch matrix with unit stride on one of its two axes, and batch dims
//! that collapse to one skip (a broadcast skip of 0 included) — and falls back
//! to candle's `matmul` for anything else, so a new call shape is slower rather
//! than wrong.

use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor};

/// Is this a layout whose batch dims collapse to ONE skip?
///
/// Returns `(batch, skip, offset, row_stride, col_stride)` for the last two
/// axes. A dim of size 1 never constrains the skip (its stride is
/// meaningless), which is what lets a narrowed `(1, 3, used, 64)` KV-cache view
/// through.
fn batch_view(l: &Layout) -> Option<(usize, usize, usize, isize, isize)> {
    let dims = l.dims();
    let strides = l.stride();
    let r = dims.len();
    if r < 2 {
        return None;
    }
    let (rs, cs) = (strides[r - 2], strides[r - 1]);
    let batch: usize = dims[..r - 2].iter().product();
    // Walk the batch dims innermost-first: each non-unit dim must step by the
    // previous dim's size times its skip.
    let mut skip: Option<usize> = None;
    let mut span = 1usize; // elements covered by the inner batch dims
    for i in (0..r - 2).rev() {
        if dims[i] == 1 {
            continue;
        }
        match skip {
            None => {
                skip = Some(strides[i]);
                span = dims[i];
            }
            Some(s) => {
                if strides[i] != s * span {
                    return None;
                }
                span *= dims[i];
            }
        }
    }
    Some((batch, skip.unwrap_or(0), l.start_offset(), rs as isize, cs as isize))
}

struct MatMulUninit;

// `a_*` / `b_*` name the two operands' mirrored layout fields on purpose.
#[allow(clippy::similar_names)]
impl CustomOp2 for MatMulUninit {
    fn name(&self) -> &'static str {
        "ffai-argus-matmul-uninit"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(a), CpuStorage::F32(b)) = (s1, s2) else {
            candle_core::bail!("ffai-argus-matmul-uninit expects f32")
        };
        let (Some((ba, a_skip, a_off, a_rs, a_cs)), Some((bb, b_skip, b_off, b_rs, b_cs))) =
            (batch_view(l1), batch_view(l2))
        else {
            candle_core::bail!("ffai-argus-matmul-uninit: uncollapsible batch layout")
        };
        let (ad, bd) = (l1.dims(), l2.dims());
        let (m, k) = (ad[ad.len() - 2], ad[ad.len() - 1]);
        let (k2, n) = (bd[bd.len() - 2], bd[bd.len() - 1]);
        if k != k2 || (ba != bb && bb != 1) {
            candle_core::bail!("ffai-argus-matmul-uninit: {ad:?} @ {bd:?}");
        }
        let b_skip = if bb == 1 { 0 } else { b_skip };
        let total = ba * m * n;
        let mut out: Vec<f32> = Vec::with_capacity(total);
        let num_threads = candle_core::utils::get_num_threads();
        let parallelism = if num_threads > 1 {
            gemm::Parallelism::Rayon(num_threads)
        } else {
            gemm::Parallelism::None
        };
        for step in 0..ba {
            // SAFETY: `batch_view` read the offsets and strides from the
            // layouts candle validated against these storages, so every
            // `(row, col)` gemm addresses at `off + step*skip + row*rs +
            // col*cs` lies inside `a`/`b`. The destination is the `m x n`
            // block at `step * m * n` of a buffer with capacity `ba * m * n`,
            // and `read_dst = false` makes gemm write every element of it
            // before `set_len` below publishes them.
            #[allow(unsafe_code)]
            unsafe {
                gemm::gemm(
                    m,
                    n,
                    k,
                    out.as_mut_ptr().add(step * m * n),
                    1,
                    n as isize,
                    false,
                    a.as_ptr().add(a_off + step * a_skip),
                    a_cs,
                    a_rs,
                    b.as_ptr().add(b_off + step * b_skip),
                    b_cs,
                    b_rs,
                    0.0f32,
                    1.0f32,
                    false,
                    false,
                    false,
                    parallelism,
                );
            }
        }
        // SAFETY: every batch block written above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(total);
        }
        let mut shape = ad[..ad.len() - 2].to_vec();
        shape.push(m);
        shape.push(n);
        Ok((CpuStorage::F32(out), Shape::from(shape)))
    }
}

/// A zeroed `(dims)` f32 tensor, for a buffer the caller then fills through an
/// in-place op.
///
/// It was uninitialised (`with_capacity` + `set_len`) while the campaign
/// measured 3b; clippy's `uninit_vec` is right that a `Vec<f32>` of
/// uninitialised values is not sound to hold, so it is zeroed. The fill is
/// one memset of the attention output per prefill layer.
pub(crate) fn zeroed_f32(dims: &[usize], device: &candle_core::Device) -> Result<Tensor> {
    Tensor::zeros(dims, candle_core::DType::F32, device)
}

/// `dst += a @ b` for 2-D operands, inside gemm (`read_dst = true`).
///
/// A residual block computes `x + h @ W^T` and used to do it in two passes:
/// the product into a fresh buffer, then an add of the residual. With the
/// residual as gemm's destination the add happens as the product is written —
/// one buffer, one pass fewer.
///
/// Float-close to the two-pass form: gemm adds the destination once per
/// K-block rather than once at the end, so the association can differ.
struct MatMulAccInto;

// `a_*` / `b_*` name the two operands' mirrored layout fields on purpose.
#[allow(clippy::similar_names)]
impl candle_core::InplaceOp3 for MatMulAccInto {
    fn name(&self) -> &'static str {
        "ffai-argus-matmul-acc-into"
    }

    fn cpu_fwd(
        &self,
        s1: &mut CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<()> {
        let (CpuStorage::F32(a), CpuStorage::F32(b)) = (s2, s3) else {
            candle_core::bail!("ffai-argus-matmul-acc-into expects f32")
        };
        let CpuStorage::F32(dst) = s1 else {
            candle_core::bail!("ffai-argus-matmul-acc-into expects f32")
        };
        let (Some((_, _, a_off, a_rs, a_cs)), Some((_, _, b_off, b_rs, b_cs))) = (batch_view(l2), batch_view(l3))
        else {
            candle_core::bail!("ffai-argus-matmul-acc-into: layout")
        };
        let Some((d0, d1)) = l1.contiguous_offsets() else {
            candle_core::bail!("ffai-argus-matmul-acc-into expects a contiguous destination")
        };
        let (m, k) = l2.shape().dims2()?;
        let (k2, n) = l3.shape().dims2()?;
        if k != k2 || d1 - d0 != m * n {
            candle_core::bail!("ffai-argus-matmul-acc-into: {m}x{k} @ {k2}x{n} into {}", d1 - d0);
        }
        let num_threads = candle_core::utils::get_num_threads();
        let parallelism = if num_threads > 1 {
            gemm::Parallelism::Rayon(num_threads)
        } else {
            gemm::Parallelism::None
        };
        // SAFETY: as in `MatMulUninit`: the offsets and strides come from the
        // layouts candle validated against these storages; the destination is
        // `m * n` contiguous initialised floats (checked above), read and
        // written by gemm with `read_dst = true`.
        #[allow(unsafe_code)]
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                dst.as_mut_ptr().add(d0),
                1,
                n as isize,
                true,
                a.as_ptr().add(a_off),
                a_cs,
                a_rs,
                b.as_ptr().add(b_off),
                b_cs,
                b_rs,
                1.0f32,
                1.0f32,
                false,
                false,
                false,
                parallelism,
            );
        }
        Ok(())
    }
}

/// `dst += a @ b` for 2-D `a`, `b` and a contiguous `(m, n)`-element `dst`.
///
/// # Errors
/// Shape mismatch.
pub fn matmul_acc_into(dst: &Tensor, a: &Tensor, b: &Tensor) -> Result<()> {
    dst.inplace_op3(a, b, &MatMulAccInto)
}

/// `a @ b`, bit-identical to candle's `matmul`, without zero-filling the output.
///
/// # Errors
/// Shape mismatch, as candle's.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if off()
        || a.dtype() != candle_core::DType::F32
        || b.dtype() != candle_core::DType::F32
        // Same rank, or a rank-3 activation against a rank-2 weight (the
        // weight broadcast over the batch: `MatMulUninit` takes a skip of 0).
        || !(a.rank() == b.rank() || (a.rank() == 3 && b.rank() == 2))
        || batch_view(a.layout()).is_none()
        || batch_view(b.layout()).is_none()
    {
        return a.matmul(b);
    }
    a.apply_op2_no_bwd(b, &MatMulUninit)
}

/// `FFAI_ARGUS_MATMUL_CANDLE=1` routes every product back through candle's
/// zero-filling `matmul` — the oracle arm, read once.
fn off() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static C: AtomicU8 = AtomicU8::new(u8::MAX);
    match C.load(Ordering::Relaxed) {
        u8::MAX => {
            let v = std::env::var("FFAI_ARGUS_MATMUL_CANDLE").is_ok_and(|x| x == "1");
            C.store(u8::from(v), Ordering::Relaxed);
            v
        }
        v => v == 1,
    }
}
