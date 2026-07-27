//! # MEASURED AND REVERTED — this kernel LOSES
//!
//! **3/21 paired rounds, z = -3.3, ratio 0.946x.** Decoder 0.175 -> 0.200 s.
//! It is slower than candle's f32 path, and slower still than the candle f16
//! matmul it was written to beat (which measured 1.056x on the same harness).
//!
//! The reasoning above is sound and the kernel is correct; the arithmetic just
//! does not clear the overhead. Per call it marshals the activation out with
//! `to_vec1()` and builds a result `Tensor::from_vec` — the same glue that
//! cost ~12 ms/layer in the attention kernel until `CustomOp3` removed it —
//! and it pays rayon fork-join on as few as 384 output rows. At **2x** traffic
//! saving that overhead is not covered.
//!
//! The int8 kernel in [`super::vocab_int8`] has the SAME structure and DOES
//! win (1.093x) because it saves **4x**, which is enough to cover the same
//! glue. So the dividing line is not "hand kernel vs framework" — it is
//! whether the traffic saved exceeds the marshalling paid, and 2x does not.
//!
//! Retained, not deleted: routed through `CustomOp1` to read candle's storage
//! directly rather than copying, this becomes worth re-testing. That is the
//! change that took the attention kernel from losing to winning.

//! AVX2 + F16C GEMV: **f16 bytes, f32 arithmetic.**
//!
//! Storing the decoder's single-row weights as f16 (`QLinear::Half`) matched
//! the reference's precision and won 1.056x — but that was only about half the
//! traffic saving the precision makes available. Two costs ate the rest:
//!
//!   1. candle's f16 matmul runs below its f32 FLOP rate (421 vs 615 GFLOP/s
//!      measured on square shapes), so halving the bytes bought less than it
//!      should on a path that is not purely bandwidth-bound;
//!   2. the activation had to be converted to f16 on every call, and the
//!      result converted back.
//!
//! This kernel avoids both. `_mm256_cvtph_ps` widens 8 f16 weights to f32 in
//! one instruction as they stream, so the multiply-accumulate happens in f32
//! against an f32 activation that is never converted. We pay f16's memory
//! traffic and keep f32's arithmetic — strictly better than either arm above,
//! and the reason ggml stores f16 and computes f32 rather than doing f16 math.
//!
//! Contrast with the int8 kernel in [`super::vocab_int8`]: int8 halves the
//! traffic again (8.3 vs 16.5 MB/token) and measured faster still (1.093x),
//! but its error compounds through the residual stream and it failed the
//! corpus quality gate at 8.39 % WER. f16 keeps ~3 decimal digits.

use ffai_core::candle::{DType, Device, Result as CandleResult, Tensor};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// f16 weights in (out, in) row-major — the layout `QLinear` already stores,
/// so no transpose is needed.
pub struct F16Gemv {
    w: Vec<u16>,
    out_dim: usize,
    in_dim: usize,
}

#[cfg(target_arch = "x86_64")]
fn have_f16c() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
    })
}

#[cfg(not(target_arch = "x86_64"))]
fn have_f16c() -> bool {
    false
}

impl F16Gemv {
    /// `weight` is (out, in). Returns `None` when this machine or shape is not
    /// one the kernel serves.
    pub fn new(weight: &Tensor) -> CandleResult<Option<Self>> {
        if !have_f16c() || !matches!(weight.device(), Device::Cpu) {
            return Ok(None);
        }
        let (out_dim, in_dim) = match weight.dims2() {
            Ok(d) => d,
            Err(_) => return Ok(None),
        };
        // The kernel steps 8 lanes; every Whisper dimension is a multiple of
        // 8, but do not assume it.
        if in_dim % 8 != 0 {
            return Ok(None);
        }
        let half = weight.to_dtype(DType::F16)?.flatten_all()?.to_vec1::<half::f16>()?;
        let w = half.into_iter().map(|h| h.to_bits()).collect();
        Ok(Some(F16Gemv { w, out_dim, in_dim }))
    }

    /// `x` is (1, in) f32 → (1, out) f32. The activation is never converted.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let xf: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let mut out = vec![0f32; self.out_dim];
        let (d, w) = (self.in_dim, &self.w);
        out.par_chunks_mut(256).enumerate().for_each(|(ci, o)| {
            let v0 = ci * 256;
            for (i, slot) in o.iter_mut().enumerate() {
                let row = (v0 + i) * d;
                *slot = unsafe { dot_f16(w.as_ptr().add(row), &xf, d) };
            }
        });
        Tensor::from_vec(out, (1, self.out_dim), x.device())
    }
}

/// Widen f16 weights to f32 as they stream and accumulate in f32.
///
/// Two accumulator chains so the dependent adds overlap; the horizontal
/// reduction happens once per output row, amortized over `d` contractions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_f16(w: *const u16, x: &[f32], d: usize) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let xp = x.as_ptr();
    let mut k = 0;
    while k + 16 <= d {
        let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w.add(k) as *const __m128i));
        acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp.add(k)), acc0);
        let w1 = _mm256_cvtph_ps(_mm_loadu_si128(w.add(k + 8) as *const __m128i));
        acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp.add(k + 8)), acc1);
        k += 16;
    }
    while k + 8 <= d {
        let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w.add(k) as *const __m128i));
        acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp.add(k)), acc0);
        k += 8;
    }
    let acc = _mm256_add_ps(acc0, acc1);
    let s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn dot_f16(_w: *const u16, _x: &[f32], _d: usize) -> f32 {
    unreachable!("guarded by have_f16c()")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Must agree with an f32 matmul to f16's own precision.
    #[test]
    fn matches_f32_matmul() -> CandleResult<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (1536usize, 384usize);
        let w = Tensor::randn(0f32, 1., (out_dim, in_dim), &dev)?;
        let x = Tensor::randn(0f32, 1., (1, in_dim), &dev)?;

        let Some(k) = F16Gemv::new(&w)? else {
            return Ok(());
        };
        let ours: Vec<f32> = k.forward(&x)?.flatten_all()?.to_vec1()?;
        let want: Vec<f32> = x.matmul(&w.t()?)?.flatten_all()?.to_vec1()?;

        let scale = want.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let err = ours
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            / scale;
        // f16 has ~11 bits of mantissa; over a 384-term sum this is the
        // expected band.
        assert!(err < 5e-3, "relative error {err:e} exceeds f16's band");
        Ok(())
    }

    #[test]
    fn declines_shapes_it_cannot_serve() -> CandleResult<()> {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1., (16usize, 12usize), &dev)?;
        assert!(F16Gemv::new(&w)?.is_none());
        Ok(())
    }
}
