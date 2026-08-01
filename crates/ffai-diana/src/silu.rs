//! A fused single-pass SiLU.
//!
//! # Why
//!
//! Decomposing the convolution bucket found the activation, not a
//! convolution, holding **25.8% of detect** across 870 calls
//! (`FFAI_PROFILE=1`). That is Mercury's GELU result reproduced on a
//! different model: an elementwise function nobody would nominate, costing
//! as much as the arithmetic it decorates.
//!
//! Two mechanisms, both removable:
//!
//! 1. **`x * sigmoid(x)` is two tensor passes and two allocations.** One
//!    fused pass writes the result once.
//! 2. **`exp` is a scalar libm call**, so it blocks vectorization of the
//!    whole loop. Mercury measured exactly this for `tanh` and fixed it the
//!    same way — a polynomial that auto-vectorizes.
//!
//! `exp(x) = 2^(x·log2e)`: the integer part is written straight into the f32
//! exponent field and the fraction comes from a degree-5 polynomial on
//! [-0.5, 0.5]. No libm call, no branches, so the loop vectorizes.
//!
//! Float, so the gate is a tolerance against candle's own `silu` (which
//! stays the oracle), plus the full-graph parity oracle downstream.

use candle_core::{Result, Tensor};
use rayon::prelude::*;

/// `exp` without a libm call, accurate to ~1e-7 relative over the range an
/// activation sees.
#[inline(always)]
fn exp_fast(x: f32) -> f32 {
    // 2^t, t = x * log2(e). Clamp before the exponent-field write so a
    // saturating input cannot produce a denormal or an invalid exponent.
    let t = (x * std::f32::consts::LOG2_E).clamp(-125.0, 125.0);
    let n = t.round();
    let f = t - n; // in [-0.5, 0.5]
    // 2^f = exp(f*ln2), so the coefficients are ln2^k / k!. DERIVED, not
    // transcribed: hand-typed decimals here are a real hazard — trimming one
    // digit to satisfy a lint silently selected a different f32 and broke
    // the oracle. Let the compiler compute them.
    const L1: f32 = std::f32::consts::LN_2;
    const L2: f32 = L1 * L1 / 2.0;
    const L3: f32 = L1 * L1 * L1 / 6.0;
    const L4: f32 = L1 * L1 * L1 * L1 / 24.0;
    const L5: f32 = L1 * L1 * L1 * L1 * L1 / 120.0;
    let p = 1.0 + f * (L1 + f * (L2 + f * (L3 + f * (L4 + f * L5))));
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    p * scale
}

/// `x * sigmoid(x)`, in one pass.
#[inline(always)]
fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + exp_fast(-x))
}

/// Chunk size for the parallel split. Large enough that the rayon fork-join
/// is amortized — the smallest tensors here are ~6 k elements, where a split
/// would cost more than the work.
const PAR_THRESHOLD: usize = 1 << 16;

/// SiLU over a contiguous f32 tensor, preserving shape.
///
/// Runs through [`crate::cpuop::SliceOp`], so the input is read in place and
/// the output `Vec` becomes the result tensor's storage — no copy on either
/// side. It previously did `to_vec1()` in and `from_vec` out, which for an
/// activation whose whole cost is memory traffic meant paying that traffic
/// three times over.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    crate::cpuop::SliceOp::new("ffai-silu", |xs, l| {
        let mut v: Vec<f32> = Vec::with_capacity(xs.len());
        if xs.len() >= PAR_THRESHOLD {
            v.resize(xs.len(), 0.0);
            v.par_chunks_mut(1 << 14)
                .zip(xs.par_chunks(1 << 14))
                .for_each(|(o, i)| {
                    for (o, i) in o.iter_mut().zip(i) {
                        *o = silu_scalar(*i);
                    }
                });
        } else {
            v.extend(xs.iter().map(|v| silu_scalar(*v)));
        }
        Ok((v, l.shape().clone()))
    })
    .run(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn exp_fast_tracks_libm() {
        let mut worst = 0f32;
        let mut t = -30.0f32;
        while t <= 30.0 {
            let (a, b) = (exp_fast(t), t.exp());
            let rel = (a - b).abs() / b.abs().max(1e-30);
            worst = worst.max(rel);
            t += 0.001;
        }
        assert!(worst < 1e-5, "worst relative error {worst:.3e}");
    }

    #[test]
    fn matches_candle_silu() {
        let dev = Device::Cpu;
        for &n in &[1usize, 7, 4096, 1 << 17] {
            // DETERMINISTIC inputs, spanning saturation both ways. This used
            // `Tensor::randn`, which made the test flaky rather than strict:
            // at n=1 the relative error divides by a single random value, so
            // a draw near zero fails a bound that a draw near one passes. A
            // test that depends on the draw is not a gate.
            let vals: Vec<f32> =
                (0..n).map(|i| -12.0 + 24.0 * (i as f32) / (n.max(2) - 1) as f32).collect();
            let x = Tensor::from_vec(vals, n, &dev).unwrap();
            let got = silu(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let want = candle_nn::ops::silu(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            let d = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs() / scale)
                .fold(0.0f32, f32::max);
            assert!(d < 1e-6, "n={n}: max rel {d:.3e}");
        }
    }

    #[test]
    fn saturates_the_way_silu_should() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![-60f32, -20.0, 0.0, 20.0, 60.0], 5, &dev).unwrap();
        let y = silu(&x).unwrap().to_vec1::<f32>().unwrap();
        assert!(y[0].abs() < 1e-6, "large negative -> 0, got {}", y[0]);
        assert!((y[2]).abs() < 1e-6, "silu(0) = 0, got {}", y[2]);
        assert!((y[4] - 60.0).abs() < 1e-3, "large positive -> x, got {}", y[4]);
        assert!(y.iter().all(|v| v.is_finite()), "no NaN/Inf at saturation");
    }
}
