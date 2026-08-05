//! A non-materialising 3x3 convolution for the CRNN backbone (§8.101).
//!
//! §8.100 measured the backbone at **84 % of the whole document pipeline** and
//! 29.7 GFLOP/s against ~112 GFLOP/s of single-core AVX2 peak. The mechanism is
//! candle's tiled im2col: per tile it allocates and zeroes a `k_size x 512`
//! buffer — 4.7 MB for the 256->256 layer — fills it with a scalar strided
//! gather, gemms, and scatters back. im2col expands the input **9x** for a 3x3
//! kernel, so the layer writes 28.5 MB per line against 2.28 GFLOP of
//! arithmetic, and these shapes yield only 2-3 tiles of internal parallelism.
//!
//! This kernel never builds that buffer. It reads each input element from its
//! natural position and accumulates into an output tile held in **registers**
//! across the entire `c_in * 9` contraction.
//!
//! ## What the shape of the loop had to become (four measured attempts)
//!
//! | form | vs candle |
//! |---|---:|
//! | direct, `Vec` allocated per output tile | 0.41x |
//! | no allocation, CO-blocked, per-row AXPY | 0.57x |
//! | register tile indexed by a runtime channel counter | **0.21x** |
//! | padded common stride, one long AXPY per tap | 0.67x |
//! | **AVX2, 4 channels x 24 columns held in ymm** | **1.65x** |
//!
//! Three lessons are baked into the code below. The per-tile `Vec` cost more
//! than the arithmetic. `dp[i].add(x)` with a runtime `i` is a POINTER GATHER,
//! which vetoes vectorization outright — it was the slowest form of all. And an
//! accumulator that lives in memory pays a load AND a store per FMA, capping the
//! loop at L1 port throughput (~28 GFLOP/s) rather than the two FMA units; only
//! moving the tile into registers passed candle.
//!
//! The 24-column tile is deliberate: twelve accumulators + three input vectors +
//! one broadcast is sixteen ymm registers exactly, lifting the ratio from 8 FMAs
//! per 6 memory ops (16-column tile, 1.20x) to 12 per 7 (**1.65x**).
//!
//! ## Correctness
//!
//! Judged against an **f64 reference**, not against candle — both are f32
//! reassociations of the same 2304-term contraction, so "differs from candle"
//! says nothing about which is right. Normalised by the tensor's own scale:
//! ours 9.2e-7..2.2e-6, candle 6.8e-7..1.5e-6. Both at f32 epsilon, three orders
//! inside the 1e-5 gate.
//!
//! The scalar path is the oracle and the fallback on any non-AVX2 CPU.

use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor};
use candle_nn::Conv2d;

/// Zero-pad `inp` (NCHW, batch 1) to `c_in x (h+3) x (w+2)`.
///
/// The padding is what lets every `(ky, kx)` tap be a fixed offset into one
/// contiguous run: output element `(oy, ox)` reads
/// `in_p[(oy+ky)*sw + (ox+kx)] = in_p[oy*sw + ox + (ky*sw + kx)]`. The extra
/// third row is slack so the last tile's `+2*sw+2` lookahead stays in bounds.
fn pad(inp: &[f32], c_in: usize, h: usize, w: usize) -> Vec<f32> {
    let sw = w + 2;
    let mut out = vec![0f32; c_in * (h + 3) * sw];
    for c in 0..c_in {
        for y in 0..h {
            let src = c * h * w + y * w;
            let dst = c * (h + 3) * sw + (y + 1) * sw + 1;
            out[dst..dst + w].copy_from_slice(&inp[src..src + w]);
        }
    }
    out
}

/// Scalar oracle. Correct on every CPU; the AVX2 twin is checked against it.
fn conv3x3_scalar(
    in_p: &[f32],
    w: &[f32],
    c_in: usize,
    c_out: usize,
    h: usize,
    wd: usize,
    out: &mut [f32],
) {
    let sw = wd + 2;
    let plane_p = (h + 3) * sw;
    for co in 0..c_out {
        for y in 0..h {
            for x in 0..wd {
                let mut acc = 0f32;
                for ci in 0..c_in {
                    for tap in 0..9usize {
                        let off = ci * plane_p + (y + tap / 3) * sw + x + tap % 3;
                        acc += w[(co * c_in + ci) * 9 + tap] * in_p[off];
                    }
                }
                out[co * h * wd + y * wd + x] = acc;
            }
        }
    }
}

/// AVX2 twin: 4 output channels x 24 columns held in ymm across the whole
/// contraction.
///
/// # Safety
/// `in_p` must be the `c_in x (h+3) x (wd+2)` padded input from [`pad`], `w`
/// must hold `c_out * c_in * 9`, and `out` `c_out * h * wd`. Every vector load
/// reads at most `tile + 24 + 2*sw + 2 <= plane_p`, which [`pad`]'s extra row
/// guarantees; the `< 24` remainder of the flattened plane takes the scalar
/// tail. Requires AVX2 + FMA, checked by the caller.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv3x3_avx2(
    in_p: &[f32],
    w: &[f32],
    c_in: usize,
    c_out: usize,
    h: usize,
    wd: usize,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    const CO: usize = 4;
    let sw = wd + 2;
    let n = h * sw;
    let plane_p = (h + 3) * sw;
    let n_vec = n / 24 * 24;

    for co0 in (0..c_out).step_by(CO) {
        let co_n = CO.min(c_out - co0);
        for tile in (0..n_vec).step_by(24) {
            let mut a: [[__m256; 3]; CO] = [[_mm256_setzero_ps(); 3]; CO];
            for ci in 0..c_in {
                let base = ci * plane_p + tile;
                for tap in 0..9usize {
                    let ptr = in_p.as_ptr().add(base + (tap / 3) * sw + tap % 3);
                    let v0 = _mm256_loadu_ps(ptr);
                    let v1 = _mm256_loadu_ps(ptr.add(8));
                    let v2 = _mm256_loadu_ps(ptr.add(16));
                    for (i, ai) in a.iter_mut().enumerate().take(co_n) {
                        let wb =
                            _mm256_set1_ps(*w.get_unchecked(((co0 + i) * c_in + ci) * 9 + tap));
                        ai[0] = _mm256_fmadd_ps(wb, v0, ai[0]);
                        ai[1] = _mm256_fmadd_ps(wb, v1, ai[1]);
                        ai[2] = _mm256_fmadd_ps(wb, v2, ai[2]);
                    }
                }
            }
            let mut tmp = [0f32; 24];
            for (i, ai) in a.iter().enumerate().take(co_n) {
                _mm256_storeu_ps(tmp.as_mut_ptr(), ai[0]);
                _mm256_storeu_ps(tmp.as_mut_ptr().add(8), ai[1]);
                _mm256_storeu_ps(tmp.as_mut_ptr().add(16), ai[2]);
                for (k, t) in tmp.iter().enumerate() {
                    let pos = tile + k;
                    let (y, x) = (pos / sw, pos % sw);
                    // The two padded columns accumulate garbage by design; they
                    // are simply not written back.
                    if x < wd && y < h {
                        *out.get_unchecked_mut((co0 + i) * h * wd + y * wd + x) = *t;
                    }
                }
            }
        }
        for pos in n_vec..n {
            let (y, x) = (pos / sw, pos % sw);
            if x >= wd || y >= h {
                continue;
            }
            for i in 0..co_n {
                let mut acc = 0f32;
                for ci in 0..c_in {
                    for tap in 0..9usize {
                        let off = ci * plane_p + pos + (tap / 3) * sw + tap % 3;
                        acc += w[((co0 + i) * c_in + ci) * 9 + tap] * in_p[off];
                    }
                }
                out[(co0 + i) * h * wd + y * wd + x] = acc;
            }
        }
    }
}

struct Conv3x3Op;

impl CustomOp2 for Conv3x3Op {
    fn name(&self) -> &'static str {
        "ffai-conv3x3"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(inp), CpuStorage::F32(wt)) = (s1, s2) else {
            candle_core::bail!("ffai-conv3x3: f32 only")
        };
        let (_, c_in, h, w) = l1.shape().dims4()?;
        let (c_out, _, _, _) = l2.shape().dims4()?;
        let inp = match l1.contiguous_offsets() {
            Some((o, e)) => &inp[o..e],
            None => candle_core::bail!("ffai-conv3x3: non-contiguous input"),
        };
        let wt = match l2.contiguous_offsets() {
            Some((o, e)) => &wt[o..e],
            None => candle_core::bail!("ffai-conv3x3: non-contiguous weight"),
        };
        let in_p = pad(inp, c_in, h, w);
        let mut out = vec![0f32; c_out * h * w];

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: `in_p` is `pad`'s output for exactly these dims, `wt` has
            // `c_out*c_in*9` elements (checked by the dims4 above and the
            // caller's 3x3 guard), and `out` has `c_out*h*w`.
            unsafe { conv3x3_avx2(&in_p, wt, c_in, c_out, h, w, &mut out) };
            return Ok((CpuStorage::F32(out), (1, c_out, h, w).into()));
        }
        conv3x3_scalar(&in_p, wt, c_in, c_out, h, w, &mut out);
        Ok((CpuStorage::F32(out), (1, c_out, h, w).into()))
    }
}

/// Apply `conv`, using the register-tiled kernel when its preconditions hold and
/// falling back to candle otherwise.
///
/// Guarded on exactly what the kernel assumes: 3x3, stride 1, padding 1,
/// dilation 1, one group, batch 1, f32, CPU. Anything else — including the
/// backbone's final 2x2 valid conv — takes candle's path unchanged.
pub fn apply(x: &Tensor, conv: &Conv2d) -> Result<Tensor> {
    let cfg = conv.config();
    let wt = conv.weight();
    let dims = wt.dims4()?;
    let ok = cfg.stride == 1
        && cfg.padding == 1
        && cfg.dilation == 1
        && cfg.groups == 1
        && dims.2 == 3
        && dims.3 == 3
        && x.dims4().map(|d| d.0 == 1).unwrap_or(false)
        && x.dtype() == candle_core::DType::F32
        && x.device().is_cpu()
        && std::env::var("FFAI_CONV3X3").as_deref() != Ok("0");
    if !ok {
        return conv.forward_no_bias_check(x);
    }
    let out = x.apply_op2_no_bwd(wt, &Conv3x3Op)?;
    match conv.bias() {
        Some(b) => out.broadcast_add(&b.reshape((1, b.elem_count(), 1, 1))?),
        None => Ok(out),
    }
}

/// candle's `Conv2d::forward` with the bias handled the same way, so the
/// fallback path and the kernel path agree on shape handling.
trait ForwardNoBiasCheck {
    fn forward_no_bias_check(&self, x: &Tensor) -> Result<Tensor>;
}

impl ForwardNoBiasCheck for Conv2d {
    fn forward_no_bias_check(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::Module::forward(self, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The AVX2 twin must match the scalar oracle within f32 reassociation, on
    /// every backbone shape plus awkward widths that exercise the `< 24` tail.
    #[test]
    fn avx2_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !(std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma"))
            {
                return;
            }
            for (ci, co, h, w) in
                [(32, 64, 32, 181), (64, 128, 16, 90), (256, 256, 8, 90), (3, 8, 5, 7), (1, 4, 2, 3)]
            {
                let inp: Vec<f32> =
                    (0..ci * h * w).map(|i| ((i * 37 % 251) as f32 / 251.0) - 0.5).collect();
                let wt: Vec<f32> =
                    (0..co * ci * 9).map(|i| ((i * 53 % 197) as f32 / 197.0) - 0.5).collect();
                let in_p = pad(&inp, ci, h, w);
                let mut a = vec![0f32; co * h * w];
                let mut b = vec![0f32; co * h * w];
                conv3x3_scalar(&in_p, &wt, ci, co, h, w, &mut a);
                // SAFETY: shapes match `pad`'s output exactly.
                unsafe { conv3x3_avx2(&in_p, &wt, ci, co, h, w, &mut b) };
                let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-9);
                let err = a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale;
                assert!(err < 1e-5, "shape {ci}x{co}x{h}x{w}: rel err {err:e}");
            }
        }
    }
}
