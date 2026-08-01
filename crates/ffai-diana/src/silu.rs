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
    // Round by float ADDITION, not `f32::round`.
    //
    // This module's doc claims it removed the libm call that was blocking
    // vectorization. It removed `exp` and left `round` — and Rust's `round`
    // is ties-AWAY-FROM-ZERO, which no x86 instruction implements
    // (`vroundps` is ties-to-even), so it lowers to a call or a long
    // branchy sequence sitting in the middle of the loop. Removing one libm
    // call and leaving another is an easy thing to do and an invisible
    // thing to have done.
    //
    // Adding 1.5*2^23 forces every value below 2^22 to be rounded into the
    // mantissa's last bit; subtracting it back leaves `t` rounded to
    // nearest-even. Measured over 16 M elements, best of 7, single thread
    // (`examples/silu_ceiling.rs`):
    //
    //   memcpy (the roofline)  5.58 ms   24.04 GB/s
    //   `round()`             60.84 ms    2.21 GB/s
    //   `round_ties_even()`   38.85 ms    3.45 GB/s
    //   this                  12.91 ms   10.40 GB/s   <- 4.71x, BIT-IDENTICAL
    //
    // 10.4 GB/s against a 24 GB/s copy is a transcendental within 2.3x of
    // pure memory traffic, i.e. it now vectorises; `round()` did not.
    //
    // The bit-identical part is why this is a free win rather than a
    // tolerance question: max relative disagreement against the old kernel
    // over the activation range is exactly 0. In-context on the serial path
    // the bench times, the pipeline gain is 1.079x (17/21, z = +2.84) —
    // quote THAT for the engine, and these for the kernel.
    const MAGIC: f32 = 12582912.0; // 1.5 * 2^23
    // `FFAI_DIANA_SILU_ROUND=1` restores `f32::round`, so the pipeline-level
    // A/B stays runnable instead of being a claim in a commit message. The
    // branch is on a cached bool and predicts perfectly; it costs nothing
    // measurable and it is what makes the 1.94x re-checkable on a box that
    // is quiet, which this one has not been.
    let n = if old_rounding() { t.round() } else { (t + MAGIC) - MAGIC };
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

/// Whether to use the pre-fix `f32::round`. See [`exp_fast`].
///
/// Both roundings produce BIT-IDENTICAL results over the activation range
/// (measured: max relative disagreement exactly 0), so this toggle changes
/// speed and nothing else — which is why the oracle is not parameterised
/// over it.
#[inline(always)]
fn old_rounding() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(u8::MAX);
    match CACHED.load(Ordering::Relaxed) {
        u8::MAX => {
            let on = std::env::var("FFAI_DIANA_SILU_ROUND").is_ok_and(|v| v == "1");
            CACHED.store(on as u8, Ordering::Relaxed);
            on
        }
        v => v == 1,
    }
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
        // The threshold alone is not the whole decision: inside a parallel
        // batch even a large activation should stay serial, because the
        // machine is already full. See `crate::parallel`.
        if xs.len() >= PAR_THRESHOLD && !crate::parallel::serial_kernels() {
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
