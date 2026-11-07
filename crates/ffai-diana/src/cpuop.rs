//! The zero-copy seam between our kernels and candle's tensors.
//!
//! # Why this module exists
//!
//! Every kernel used to start with `x.flatten_all()?.to_vec1::<f32>()?` —
//! a full copy of the input — and end with `Tensor::from_vec(..)`. Splitting
//! the 3x3 kernels showed the cost: **im2col 12.5% of detect, the GEMM 9.1%,
//! and ~14.4% in the marshalling around them.** The framework glue was
//! larger than the im2col and dwarfed the arithmetic, so each intermediate
//! round-tripped Tensor -> Vec -> Tensor between consecutive layers.
//!
//! candle's `CustomOp1` hands the CPU storage over directly, so the input is
//! read in place and the output `Vec` *becomes* the tensor's storage rather
//! than being copied into one. This is the same hook that recovered
//! 90 -> 78 ms for Mercury with no change to the arithmetic — "delivery is
//! not a detail".
//!
//! Byte-identical by construction: only the plumbing changes, never an
//! arithmetic operation or its order.

use candle_core::{CpuStorage, Layout, Result, Shape, Tensor};

/// Borrow a contiguous `f32` view of a tensor's storage.
///
/// The `CustomOp1` contract warns that storage "can use arbitrary strides,
/// offsets etc", so this refuses a non-contiguous layout rather than
/// silently reading the wrong elements — a class of bug that produces
/// plausible numbers instead of an error.
pub fn contiguous_f32<'a>(storage: &'a CpuStorage, layout: &Layout) -> Result<&'a [f32]> {
    let all = storage.as_slice::<f32>()?;
    match layout.contiguous_offsets() {
        Some((start, end)) => Ok(&all[start..end]),
        None => candle_core::bail!(
            "ffai-diana kernels require a contiguous input; call .contiguous() first"
        ),
    }
}

/// [`contiguous_f32`] for a `Storage` guard from `Tensor::storage_and_layout`,
/// so a reader outside a custom op can borrow a tensor's buffer in place.
pub fn storage_f32<'a>(storage: &'a candle_core::Storage, layout: &Layout) -> Result<&'a [f32]> {
    match storage {
        candle_core::Storage::Cpu(cpu) => contiguous_f32(cpu, layout),
        _ => candle_core::bail!("ffai-diana: expected a CPU tensor"),
    }
}

/// A bias (or any small contiguous f32 tensor) as a `Vec<f32>`, read straight
/// from its storage. `flatten_all()?.to_vec1()?` built a reshaped `Tensor`
/// (shape, layout and op allocations) to copy `c_out` floats; this is one
/// bounds-checked slice and one `memcpy`. Same bytes.
pub fn bias_vec(b: &Tensor) -> Result<Vec<f32>> {
    let (store, layout) = b.storage_and_layout();
    Ok(storage_f32(&store, layout)?.to_vec())
}

/// `a (m, k) @ b (k, n)` for two contiguous f32 matrices, through the same
/// `gemm` crate and call candle's `matmul` makes — minus the zero fill.
///
/// candle allocates the output with `vec![0.0; m * n]` and then calls `gemm`
/// with `read_dst = false`, which overwrites every element: the fill is a
/// full write of the output for nothing (21.8 M instructions of `memset` per
/// image on the census). This writes into uninitialised capacity. Every other
/// argument — strides, `alpha = 0`, `beta = 1`, the parallelism rule — is
/// candle's, so the product is bit-identical; it falls back to candle for any
/// shape it does not handle.
pub fn matmul2(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if copying_fallback() || !a.is_contiguous() || !b.is_contiguous() || a.rank() != 2 || b.rank() != 2 {
        return a.matmul(b);
    }
    let (m, k) = a.dims2()?;
    let (k2, n) = b.dims2()?;
    if k != k2 {
        candle_core::bail!("matmul2: {m}x{k} @ {k2}x{n}");
    }
    SliceOp2::new("ffai-matmul-uninit", move |xa, _, xb, _| {
        let mut out: Vec<f32> = Vec::with_capacity(m * n);
        let num_threads = candle_core::utils::get_num_threads();
        let parallelism = if num_threads > 1 {
            gemm::Parallelism::Rayon(num_threads)
        } else {
            gemm::Parallelism::None
        };
        // SAFETY: `xa` holds `m * k` and `xb` `k * n` contiguous floats (the
        // layouts were checked contiguous and the dims read above); `out` has
        // capacity for `m * n`, and `gemm` with `read_dst = false` writes every
        // element of the `m x n` destination before `set_len`.
        #[allow(unsafe_code)]
        unsafe {
            gemm::gemm(
                /* m */ m,
                /* n */ n,
                /* k */ k,
                /* dst */ out.as_mut_ptr(),
                /* dst_cs */ 1,
                /* dst_rs */ n as isize,
                /* read_dst */ false,
                /* lhs */ xa.as_ptr(),
                /* lhs_cs */ 1,
                /* lhs_rs */ k as isize,
                /* rhs */ xb.as_ptr(),
                /* rhs_cs */ 1,
                /* rhs_rs */ n as isize,
                /* alpha */ 0.0f32,
                /* beta */ 1.0f32,
                false,
                false,
                false,
                parallelism,
            );
            out.set_len(m * n);
        }
        Ok((out, (m, n).into()))
    })
    .run(a, b)
}

/// Batched [`matmul2`]: `a (B, m, k) @ b (B, k, n) -> (B, m, n)`, one `gemm`
/// per batch into one uninitialised output. candle's batched matmul does the
/// same loop over a zero-filled output (16 x Co x T floats of `memset` per
/// Winograd convolution).
pub fn matmul2_batched(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if copying_fallback() || !a.is_contiguous() || !b.is_contiguous() || a.rank() != 3 || b.rank() != 3 {
        return a.matmul(b);
    }
    let (ba, m, k) = a.dims3()?;
    let (bb, k2, n) = b.dims3()?;
    if k != k2 || ba != bb {
        candle_core::bail!("matmul2_batched: {ba}x{m}x{k} @ {bb}x{k2}x{n}");
    }
    SliceOp2::new("ffai-matmul-batched-uninit", move |xa, _, xb, _| {
        let mut out: Vec<f32> = Vec::with_capacity(ba * m * n);
        let num_threads = candle_core::utils::get_num_threads();
        let parallelism = if num_threads > 1 {
            gemm::Parallelism::Rayon(num_threads)
        } else {
            gemm::Parallelism::None
        };
        for step in 0..ba {
            // SAFETY: as in `matmul2`, per batch: `m*k` and `k*n` contiguous
            // floats at the batch offsets, an `m x n` destination fully written
            // by `gemm` with `read_dst = false`.
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
                    xa.as_ptr().add(step * m * k),
                    1,
                    k as isize,
                    xb.as_ptr().add(step * k * n),
                    1,
                    n as isize,
                    0.0f32,
                    1.0f32,
                    false,
                    false,
                    false,
                    parallelism,
                );
            }
        }
        // SAFETY: every batch's `m x n` block written above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(ba * m * n);
        }
        Ok((out, (ba, m, n).into()))
    })
    .run(a, b)
}

/// A `Vec<f32>` of length `n` left UNINITIALISED, for a buffer the caller
/// writes in full before reading. `f32` has no invalid bit patterns.
///
/// # Safety
///
/// Every element must be written before it is read.
#[allow(unsafe_code)]
pub unsafe fn uninit_f32(n: usize) -> Vec<f32> {
    let mut v: Vec<f32> = Vec::with_capacity(n);
    // SAFETY: the caller's contract above.
    unsafe {
        v.set_len(n);
    }
    v
}

/// `Tensor::cat(parts, 1)` for contiguous `(1, c_i, h, w)` parts: each source
/// borrowed from its storage, one `memcpy` each into an uninitialised buffer
/// that becomes the result. candle's `cat` goes through `copy2d` with per-call
/// layout work and a zeroed destination; the copies themselves are the same.
pub fn cat_channels(parts: &[&Tensor]) -> Result<Tensor> {
    if copying_fallback() || parts.is_empty() || parts.iter().any(|t| t.rank() != 4 || t.dim(0).ok() != Some(1)) {
        return Tensor::cat(parts, 1);
    }
    let (_, _, h, w) = parts[0].dims4()?;
    let mut c_total = 0;
    for t in parts {
        let (_, c, hh, ww) = t.dims4()?;
        if hh != h || ww != w {
            return Tensor::cat(parts, 1);
        }
        c_total += c;
    }
    let plane = h * w;
    let n = c_total * plane;
    let mut out: Vec<f32> = Vec::with_capacity(n);
    for t in parts {
        let (store, layout) = t.storage_and_layout();
        let Ok(src) = storage_f32(&store, layout) else {
            return Tensor::cat(parts, 1);
        };
        out.extend_from_slice(src);
    }
    debug_assert_eq!(out.len(), n);
    Tensor::from_vec(out, (1, c_total, h, w), parts[0].device())
}

/// `Tensor::cat(parts, 2)` for contiguous `(1, r, a_i)` parts: row `k` of the
/// result is row `k` of every part in order. One `memcpy` per (row, part).
pub fn cat_last(parts: &[&Tensor]) -> Result<Tensor> {
    if copying_fallback() || parts.is_empty() || parts.iter().any(|t| t.rank() != 3 || t.dim(0).ok() != Some(1)) {
        return Tensor::cat(parts, 2);
    }
    let (_, r, _) = parts[0].dims3()?;
    let mut a_total = 0;
    for t in parts {
        let (_, rr, a) = t.dims3()?;
        if rr != r {
            return Tensor::cat(parts, 2);
        }
        a_total += a;
    }
    let mut out: Vec<f32> = Vec::with_capacity(r * a_total);
    let guards: Vec<_> = parts.iter().map(|t| t.storage_and_layout()).collect();
    let mut srcs = Vec::with_capacity(parts.len());
    for (store, layout) in &guards {
        let Ok(src) = storage_f32(store, layout) else {
            return Tensor::cat(parts, 2);
        };
        srcs.push(src);
    }
    for k in 0..r {
        for (src, t) in srcs.iter().zip(parts) {
            let a = t.dim(2)?;
            out.extend_from_slice(&src[k * a..(k + 1) * a]);
        }
    }
    Tensor::from_vec(out, (1, r, a_total), parts[0].device())
}

/// A `(1, c, h, w)` tensor over UNINITIALISED storage, for a concatenation
/// buffer whose every channel block is written (by `matmul2_into` or
/// `write_rows`) before anything reads it.
///
/// # Safety
///
/// Every element must be written before the tensor is read.
#[allow(unsafe_code)]
pub unsafe fn uninit_chw(c: usize, h: usize, w: usize, device: &candle_core::Device) -> Result<Tensor> {
    // SAFETY: the caller's contract above.
    let v = unsafe { uninit_f32(c * h * w) };
    Tensor::from_vec(v, (1, c, h, w), device)
}

/// `dst[row0..row0 + m, ..] = a (m, k) @ b (k, n)` where `dst` is a contiguous
/// `(R, n)` (or `(1, R, h, w)` with `h*w == n`) tensor — the GEMM written
/// straight into its slot of a concatenation buffer, no intermediate, no copy.
/// Same `gemm` call as [`matmul2`].
pub fn matmul2_into(dst: &Tensor, row0: usize, a: &Tensor, b: &Tensor) -> Result<()> {
    let (m, k) = a.dims2()?;
    let (k2, n) = b.dims2()?;
    let total = dst.elem_count();
    if k != k2 || !a.is_contiguous() || !b.is_contiguous() || !dst.is_contiguous() || (row0 + m) * n > total {
        candle_core::bail!("matmul2_into: {m}x{k} @ {k2}x{n} into row {row0} of {total} elements");
    }
    dst.inplace_op3(a, b, &MatmulInto { row0, m, n, k })
}

struct MatmulInto {
    row0: usize,
    m: usize,
    n: usize,
    k: usize,
}

impl candle_core::InplaceOp3 for MatmulInto {
    fn name(&self) -> &'static str {
        "ffai-matmul-into"
    }

    fn cpu_fwd(&self, s1: &mut CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout, s3: &CpuStorage, l3: &Layout) -> Result<()> {
        let xa = contiguous_f32(s2, l2)?;
        let xb = contiguous_f32(s3, l3)?;
        let CpuStorage::F32(buf) = s1 else {
            candle_core::bail!("matmul_into: expected f32 storage");
        };
        let Some((lo, hi)) = l1.contiguous_offsets() else {
            candle_core::bail!("matmul_into: non-contiguous destination");
        };
        let (m, n, k) = (self.m, self.n, self.k);
        let dst = &mut buf[lo..hi];
        let slot = &mut dst[self.row0 * n..(self.row0 + m) * n];
        if xa.len() < m * k || xb.len() < k * n {
            candle_core::bail!("matmul_into: operands too short");
        }
        let num_threads = candle_core::utils::get_num_threads();
        let parallelism = if num_threads > 1 {
            gemm::Parallelism::Rayon(num_threads)
        } else {
            gemm::Parallelism::None
        };
        // SAFETY: `slot` holds `m * n` floats, `xa` `m * k`, `xb` `k * n`;
        // row-major strides; `read_dst = false` writes every element.
        #[allow(unsafe_code)]
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                slot.as_mut_ptr(),
                1,
                n as isize,
                false,
                xa.as_ptr(),
                1,
                k as isize,
                xb.as_ptr(),
                1,
                n as isize,
                0.0f32,
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

/// `dst[row0.., ..] = src`, one `memcpy` of a contiguous `src` into a
/// contiguous `dst` at element offset `row0 * stride`.
pub fn write_block(dst: &Tensor, elem0: usize, src: &Tensor) -> Result<()> {
    if !dst.is_contiguous() || !src.is_contiguous() || elem0 + src.elem_count() > dst.elem_count() {
        candle_core::bail!("write_block: {} elements at {elem0} into {}", src.elem_count(), dst.elem_count());
    }
    dst.inplace_op2(src, &WriteBlock { elem0 })
}

struct WriteBlock {
    elem0: usize,
}

impl candle_core::InplaceOp2 for WriteBlock {
    fn name(&self) -> &'static str {
        "ffai-write-block"
    }

    fn cpu_fwd(&self, s1: &mut CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<()> {
        let src = contiguous_f32(s2, l2)?;
        let CpuStorage::F32(buf) = s1 else {
            candle_core::bail!("write_block: expected f32 storage");
        };
        let Some((lo, hi)) = l1.contiguous_offsets() else {
            candle_core::bail!("write_block: non-contiguous destination");
        };
        let dst = &mut buf[lo..hi];
        dst[self.elem0..self.elem0 + src.len()].copy_from_slice(src);
        Ok(())
    }
}

/// Split a fused `(1, heads, 2*kd + hd, n)` qkv tensor into contiguous
/// `(1, heads, kd, n)` scaled-query (`q * mul + 0.0`, exactly candle's
/// `affine`), `(1, heads, kd, n)` key and `(1, heads, hd, n)` value, reading
/// the source once. candle did this as one strided `affine` and two strided
/// `contiguous()` copies.
pub fn split_qkv(qkv: &Tensor, kd: usize, hd: usize, mul: f32) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, heads, rows, n) = qkv.dims4()?;
    if b != 1 || rows != 2 * kd + hd || !qkv.is_contiguous() {
        let q = qkv.narrow(2, 0, kd)?;
        let k = qkv.narrow(2, kd, kd)?;
        let v = qkv.narrow(2, 2 * kd, hd)?;
        return Ok(((q * f64::from(mul))?, k.contiguous()?, v.contiguous()?));
    }
    let (store, layout) = qkv.storage_and_layout();
    let src = storage_f32(&store, layout)?;
    let mut qs: Vec<f32> = Vec::with_capacity(heads * kd * n);
    let mut ks: Vec<f32> = Vec::with_capacity(heads * kd * n);
    let mut vs: Vec<f32> = Vec::with_capacity(heads * hd * n);
    for hh in 0..heads {
        let base = hh * rows * n;
        qs.extend(src[base..base + kd * n].iter().map(|&x| x * mul + 0.0));
        ks.extend_from_slice(&src[base + kd * n..base + 2 * kd * n]);
        vs.extend_from_slice(&src[base + 2 * kd * n..base + rows * n]);
    }
    let dev = qkv.device();
    Ok((
        Tensor::from_vec(qs, (1, heads, kd, n), dev)?,
        Tensor::from_vec(ks, (1, heads, kd, n), dev)?,
        Tensor::from_vec(vs, (1, heads, hd, n), dev)?,
    ))
}

/// A `CustomOp1` built from a closure over the borrowed input slice.
///
/// The closure returns the output `Vec` and its shape; that `Vec` is moved
/// into the result tensor's storage, so neither side is copied.
pub struct SliceOp<F> {
    name: &'static str,
    f: F,
}

impl<F> SliceOp<F>
where
    F: Fn(&[f32], &Layout) -> Result<(Vec<f32>, Shape)>,
{
    pub fn new(name: &'static str, f: F) -> Self {
        Self { name, f }
    }

    /// Apply to `x` without building a backward graph — inference only.
    ///
    /// `FFAI_DIANA_NO_ZEROCOPY=1` routes through the old copy-in/copy-out
    /// path instead. The two arms run **identical arithmetic** and differ
    /// only in delivery, which is exactly what makes them a clean A/B — and
    /// on a drifting machine one binary with both arms behind a knob is the
    /// only trustworthy way to measure a change this size. The knob doubles
    /// as the fallback if a future candle version changes the custom-op
    /// contract.
    pub fn run(&self, x: &Tensor) -> Result<Tensor> {
        if copying_fallback() {
            let xs = x.flatten_all()?.to_vec1::<f32>()?;
            let (out, shape) = (self.f)(&xs, x.layout())?;
            return Tensor::from_vec(out, shape, x.device());
        }
        // Timed as its own bucket: this wrapper is entered ~2745 times per
        // image (one per convolution and one per activation), so if it costs
        // anything it costs it 2745 times, and it was previously folded into
        // whichever parent happened to call it.
        if crate::profile::roofline_enabled() {
            let t0 = std::time::Instant::now();
            let r = x.apply_op1_no_bwd(self);
            crate::profile::record_sliceop(
                self.name,
                t0.elapsed().as_nanos() as u64,
                crate::profile::take_last_work_ns(),
            );
            return r;
        }
        crate::profile::timed(|p| &p.sliceop, || x.apply_op1_no_bwd(self))
    }
}

/// The two-input twin of [`SliceOp`]: both inputs borrowed in place, one
/// output `Vec` that becomes the result's storage.
pub struct SliceOp2<F> {
    name: &'static str,
    f: F,
}

impl<F> SliceOp2<F>
where
    F: Fn(&[f32], &Layout, &[f32], &Layout) -> Result<(Vec<f32>, Shape)>,
{
    pub fn new(name: &'static str, f: F) -> Self {
        Self { name, f }
    }

    /// Apply to `(a, b)` without building a backward graph — inference only.
    pub fn run(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if copying_fallback() {
            let xa = a.flatten_all()?.to_vec1::<f32>()?;
            let xb = b.flatten_all()?.to_vec1::<f32>()?;
            let (out, shape) = (self.f)(&xa, a.layout(), &xb, b.layout())?;
            return Tensor::from_vec(out, shape, a.device());
        }
        crate::profile::timed(|p| &p.sliceop, || a.apply_op2_no_bwd(b, self))
    }
}

impl<F> candle_core::CustomOp2 for SliceOp2<F>
where
    F: Fn(&[f32], &Layout, &[f32], &Layout) -> Result<(Vec<f32>, Shape)>,
{
    fn name(&self) -> &'static str {
        self.name
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let a = contiguous_f32(s1, l1)?;
        let b = contiguous_f32(s2, l2)?;
        let (out, shape) = (self.f)(a, l1, b, l2)?;
        Ok((CpuStorage::F32(out), shape))
    }
}

/// `y += x`, in place on `y`'s own storage.
///
/// The residual connections wrote `x + y` as a third tensor: an allocation
/// and a traversal that read two buffers and wrote a cold third, for a sum
/// whose left operand is a conv output nothing else will read. IEEE addition
/// commutes exactly, so `y[i] + x[i]` is `x[i] + y[i]` to the bit.
pub fn add_assign(y: Tensor, x: &Tensor) -> Result<Tensor> {
    if copying_fallback() || !y.is_contiguous() || !x.is_contiguous() {
        return &y + x;
    }
    y.inplace_op2(x, &AddAssign)?;
    Ok(y)
}

struct AddAssign;

impl candle_core::InplaceOp2 for AddAssign {
    fn name(&self) -> &'static str {
        "ffai-add-assign"
    }

    fn cpu_fwd(&self, s1: &mut CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<()> {
        let xs = contiguous_f32(s2, l2)?;
        let CpuStorage::F32(buf) = s1 else {
            candle_core::bail!("ffai add_assign: expected f32 storage");
        };
        let Some((a, b)) = l1.contiguous_offsets() else {
            candle_core::bail!("ffai add_assign: non-contiguous destination");
        };
        let ys = &mut buf[a..b];
        if ys.len() != xs.len() {
            candle_core::bail!("ffai add_assign: {} vs {} elements", ys.len(), xs.len());
        }
        for (y, &x) in ys.iter_mut().zip(xs) {
            *y += x;
        }
        Ok(())
    }
}

fn copying_fallback() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FFAI_DIANA_NO_ZEROCOPY").is_some())
}

impl<F> candle_core::CustomOp1 for SliceOp<F>
where
    F: Fn(&[f32], &Layout) -> Result<(Vec<f32>, Shape)>,
{
    fn name(&self) -> &'static str {
        self.name
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let xs = contiguous_f32(storage, layout)?;
        let t0 = crate::profile::roofline_enabled().then(std::time::Instant::now);
        let (out, shape) = (self.f)(xs, layout)?;
        if let Some(t0) = t0 {
            crate::profile::set_last_work_ns(t0.elapsed().as_nanos() as u64);
        }
        Ok((CpuStorage::F32(out), shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn round_trips_a_tensor_without_changing_it() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 3, 4), &dev).unwrap();
        let op = SliceOp::new("double", |xs, l| {
            Ok((xs.iter().map(|v| v * 2.0).collect(), l.shape().clone()))
        });
        let y = op.run(&x).unwrap();
        assert_eq!(y.dims(), x.dims());
        let (a, b) = (
            x.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        for (i, (p, q)) in a.iter().zip(&b).enumerate() {
            assert!((p * 2.0 - q).abs() < 1e-6, "element {i}");
        }
    }

    #[test]
    fn cats_match_candle_bit_for_bit() {
        let dev = Device::Cpu;
        let a = Tensor::randn(0f32, 1.0, (1, 3, 4, 5), &dev).unwrap();
        let b = Tensor::randn(0f32, 1.0, (1, 2, 4, 5), &dev).unwrap();
        let half = a.narrow(1, 1, 2).unwrap();
        for parts in [vec![&a, &b], vec![&half, &b, &a]] {
            let want = Tensor::cat(&parts, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let got = cat_channels(&parts).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        let x = Tensor::randn(0f32, 1.0, (1, 4, 7), &dev).unwrap();
        let y = Tensor::randn(0f32, 1.0, (1, 4, 3), &dev).unwrap();
        let z = Tensor::randn(0f32, 1.0, (1, 4, 1), &dev).unwrap();
        let want = Tensor::cat(&[&x, &y, &z], 2).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let got = cat_last(&[&x, &y, &z]).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }

    #[test]
    fn rejects_a_non_contiguous_input_rather_than_misreading_it() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (4, 6), &dev).unwrap();
        let t = x.t().unwrap(); // transposed view: not contiguous
        let op = SliceOp::new("identity", |xs, l| Ok((xs.to_vec(), l.shape().clone())));
        assert!(op.run(&t).is_err(), "a non-contiguous layout must be refused");
    }
}
