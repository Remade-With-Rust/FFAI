//! Depthwise convolution, because candle's CPU backend has none.
//!
//! # Why this exists
//!
//! `Tensor::conv2d` with `groups > 1` does `chunk(groups, 1)`, one
//! `conv2d_single_group` per group, then `cat` (candle-core 0.11
//! `conv.rs:262`). A depthwise layer has `groups == channels`, so SVTR's
//! 480-channel depthwise runs **480 separate single-channel convolutions**,
//! each through the full im2col machinery, plus 480 allocations and a 480-way
//! concatenation. ffai-diana measured the same mechanism at 72.9x slower per
//! FLOP than the dense convolution beside it (`ffai-diana/src/dwconv.rs`).
//!
//! SVTR has 14 depthwise layers per forward and runs one forward per text
//! line; PdfCraft measured recognition at ~300 ms a line, 97 % of the page.
//!
//! # What this does instead
//!
//! A direct stencil: one pass per output row, one AXPY per tap over the row,
//! parallel over `(batch, channel)` planes. Any kernel size, independent
//! strides and paddings per axis (SVTR uses `(2, 1)` and `(1, 2)`), so the
//! callers no longer compute a stride-1 output and throw half of it away.
//!
//! # Correctness
//!
//! On AVX2+FMA hardware the result is BIT-IDENTICAL to candle's grouped path:
//! see [`plane_fma`]. Elsewhere it is the same products summed in the same
//! order with a separate multiply and add, so it agrees to a tolerance.
//! `tests::matches_candle_grouped` holds both against candle on every shape
//! the two models use.

use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor};

use crate::par::prelude::*;

struct DwOp {
    stride: (usize, usize),
    pad: (usize, usize),
}

/// Output length of a strided, padded, valid convolution along one axis.
fn out_len(n: usize, k: usize, s: usize, p: usize) -> usize {
    (n + 2 * p).saturating_sub(k) / s + 1
}

/// One output plane. `inp` is one `h x w` input plane, `wt` its `kh x kw`
/// kernel, `out` the `ho x wo` result (overwritten). `madd(w, x, acc)` is the
/// accumulation step: see [`plane_fma`] for why it is a parameter.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn plane_with(
    inp: &[f32],
    h: usize,
    w: usize,
    wt: &[f32],
    kh: usize,
    kw: usize,
    (sh, sw): (usize, usize),
    (ph, pw): (usize, usize),
    out: &mut [f32],
    wo: usize,
    madd: impl Fn(f32, f32, f32) -> f32,
) {
    out.fill(0.0);
    for (oy, orow) in out.chunks_exact_mut(wo).enumerate() {
        for ky in 0..kh {
            // iy = oy*sh + ky - ph, skipped when it falls in the zero padding.
            let Some(iy) = (oy * sh + ky).checked_sub(ph) else { continue };
            if iy >= h {
                continue;
            }
            let irow = &inp[iy * w..(iy + 1) * w];
            for kx in 0..kw {
                let wv = wt[ky * kw + kx];
                // Valid ox: 0 <= ox*sw + kx - pw < w.
                let lo = pw.saturating_sub(kx).div_ceil(sw);
                let hi = if w + pw > kx { (w + pw - kx - 1) / sw + 1 } else { 0 };
                let hi = hi.min(wo);
                if lo >= hi {
                    continue;
                }
                if sw == 1 {
                    // Contiguous on both sides: the loop the compiler vectorises.
                    let src = &irow[lo + kx - pw..hi + kx - pw];
                    for (o, &v) in orow[lo..hi].iter_mut().zip(src) {
                        *o = madd(wv, v, *o);
                    }
                } else {
                    for ox in lo..hi {
                        orow[ox] = madd(wv, irow[ox * sw + kx - pw], orow[ox]);
                    }
                }
            }
        }
    }
}

/// The portable plane: a separate multiply and add per tap.
#[allow(clippy::too_many_arguments)]
fn plane(
    inp: &[f32],
    h: usize,
    w: usize,
    wt: &[f32],
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    pad: (usize, usize),
    out: &mut [f32],
    wo: usize,
) {
    plane_with(inp, h, w, wt, kh, kw, stride, pad, out, wo, |w, x, acc| acc + w * x);
}

/// The same plane accumulated with FUSED multiply-adds, which is what makes it
/// agree with candle bit for bit rather than to a tolerance.
///
/// Candle's grouped path im2cols each channel and hands the `kh * kw` taps to
/// gemm, whose microkernel accumulates `acc = fma(a, b, acc)` over them in
/// `(ky, kx)` order from zero. This walks the taps in that same order with the
/// same fused step, and a tap that lands in the padding is skipped where gemm
/// adds `w * 0` — which leaves the accumulator unchanged. `mul_add` is a libm
/// call unless the function is compiled with FMA, hence the target feature.
///
/// # Safety
/// The CPU must support AVX2 and FMA (checked by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_code)]
unsafe fn plane_fma(
    inp: &[f32],
    h: usize,
    w: usize,
    wt: &[f32],
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    pad: (usize, usize),
    out: &mut [f32],
    wo: usize,
) {
    plane_with(inp, h, w, wt, kh, kw, stride, pad, out, wo, |w, x, acc| w.mul_add(x, acc));
}

fn has_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| {
            std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

impl CustomOp2 for DwOp {
    fn name(&self) -> &'static str {
        "ffai-carmenta-dwconv"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(inp), CpuStorage::F32(wt)) = (s1, s2) else {
            candle_core::bail!("ffai-carmenta-dwconv: f32 only")
        };
        let (n, c, h, w) = l1.shape().dims4()?;
        let (wc, one, kh, kw) = l2.shape().dims4()?;
        if wc != c || one != 1 {
            candle_core::bail!(
                "ffai-carmenta-dwconv: weight {:?} is not depthwise for {c} channels",
                l2.shape()
            );
        }
        let inp = match l1.contiguous_offsets() {
            Some((o, e)) => &inp[o..e],
            None => candle_core::bail!("ffai-carmenta-dwconv: non-contiguous input"),
        };
        let wt = match l2.contiguous_offsets() {
            Some((o, e)) => &wt[o..e],
            None => candle_core::bail!("ffai-carmenta-dwconv: non-contiguous weight"),
        };
        let (sh, sw) = self.stride;
        let (ph, pw) = self.pad;
        if h + 2 * ph < kh || w + 2 * pw < kw || sh == 0 || sw == 0 {
            candle_core::bail!("ffai-carmenta-dwconv: kernel larger than padded input");
        }
        let (ho, wo) = (out_len(h, kh, sh, ph), out_len(w, kw, sw, pw));
        let mut out = vec![0f32; n * c * ho * wo];
        let fma = has_fma();
        out.par_chunks_mut(ho * wo).enumerate().for_each(|(i, o)| {
            let ch = i % c;
            let (x, k) = (&inp[i * h * w..(i + 1) * h * w], &wt[ch * kh * kw..(ch + 1) * kh * kw]);
            #[cfg(target_arch = "x86_64")]
            if fma {
                // SAFETY: `has_fma` confirmed AVX2 and FMA at run time.
                #[allow(unsafe_code)]
                unsafe { plane_fma(x, h, w, k, kh, kw, self.stride, self.pad, o, wo) };
                return;
            }
            let _ = fma;
            plane(x, h, w, k, kh, kw, self.stride, self.pad, o, wo);
        });
        Ok((CpuStorage::F32(out), (n, c, ho, wo).into()))
    }
}

/// Depthwise convolution of `x` `(n, c, h, w)` with `weight` `(c, 1, kh, kw)`,
/// zero padding `pad = (ph, pw)` on both sides of each axis and per-axis
/// `stride`. No bias; callers add their own.
pub fn depthwise(
    x: &Tensor,
    weight: &Tensor,
    stride: (usize, usize),
    pad: (usize, usize),
) -> Result<Tensor> {
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    x.apply_op2_no_bwd(&weight, &DwOp { stride, pad })
}

/// Whether a conv with this weight and group count is one [`depthwise`]
/// handles: one group per channel, one input channel per group, f32, CPU.
/// `FFAI_DWCONV=0` sends every call back to candle's grouped path, so the
/// two arms stay one environment variable apart for an A/B.
#[must_use]
pub fn applies(x: &Tensor, weight: &Tensor, groups: usize) -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let off = *OFF.get_or_init(|| std::env::var("FFAI_DWCONV").as_deref() == Ok("0"));
    !off && groups > 1
        && weight.dims4().is_ok_and(|(oc, icg, _, _)| oc == groups && icg == 1)
        && x.dims4().is_ok_and(|(_, c, _, _)| c == groups)
        && x.dtype() == candle_core::DType::F32
        && x.device().is_cpu()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Candle's grouped conv with SVTR's emulation of asymmetric stride and
    /// padding: pad, stride-1 conv, keep every `stride`-th row and column.
    fn reference(x: &Tensor, w: &Tensor, stride: (usize, usize), pad: (usize, usize)) -> Tensor {
        let c = x.dim(1).unwrap();
        let xp = x.pad_with_zeros(2, pad.0, pad.0).unwrap().pad_with_zeros(3, pad.1, pad.1).unwrap();
        let y = xp.conv2d(w, 0, 1, 1, c).unwrap();
        let pick = |y: Tensor, dim: usize, s: usize| {
            if s == 1 {
                return y;
            }
            let n = y.dim(dim).unwrap().div_ceil(s);
            let idx: Vec<u32> = (0..n).map(|i| (i * s) as u32).collect();
            y.index_select(&Tensor::from_vec(idx, n, y.device()).unwrap(), dim).unwrap()
        };
        pick(pick(y, 2, stride.0), 3, stride.1)
    }

    #[test]
    fn matches_candle_grouped() {
        let d = Device::Cpu;
        // (n, c, h, w, k, stride, pad): every depthwise shape in SVTR and
        // MobileDet, plus odd sizes and a batch.
        let cases = [
            (1, 16, 24, 161, 3, (1, 1), (1, 1)),
            (1, 64, 24, 161, 3, (2, 1), (1, 1)),
            (1, 128, 12, 161, 3, (1, 2), (1, 1)),
            (1, 240, 12, 81, 5, (1, 1), (2, 2)),
            (1, 240, 12, 81, 5, (2, 1), (2, 2)),
            (1, 480, 6, 81, 5, (2, 1), (2, 2)),
            (1, 32, 37, 53, 3, (2, 2), (1, 1)),
            (1, 192, 19, 23, 5, (1, 1), (2, 2)),
            (3, 48, 9, 17, 3, (1, 1), (1, 1)),
            (2, 8, 5, 4, 5, (1, 1), (2, 2)),
        ];
        for (n, c, h, w, k, stride, pad) in cases {
            let x = Tensor::randn(0f32, 1.0, (n, c, h, w), &d).unwrap();
            let wt = Tensor::randn(0f32, 1.0, (c, 1, k, k), &d).unwrap();
            let got = depthwise(&x, &wt, stride, pad).unwrap();
            let want = reference(&x, &wt, stride, pad);
            assert_eq!(got.dims(), want.dims(), "shape for {:?}", (n, c, h, w, k, stride, pad));
            let diff = (got.clone() - &want).unwrap().abs().unwrap().max_all().unwrap();
            let diff = diff.to_scalar::<f32>().unwrap();
            assert!(diff < 1e-4, "max abs diff {diff} for {:?}", (n, c, h, w, k, stride, pad));
            if has_fma() {
                let (gv, wv) = (got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                              want.flatten_all().unwrap().to_vec1::<f32>().unwrap());
                assert!(gv == wv, "FMA arm not bit-identical (max {diff}) for {:?}",
                        (n, c, h, w, k, stride, pad));
            }
        }
    }
}
