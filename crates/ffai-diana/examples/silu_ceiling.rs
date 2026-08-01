//! D5: why is SiLU 30.9 % of a detection, and what is fixing it worth?
//!
//! The serial stage profile (`RAYON_NUM_THREADS=1 FFAI_PROFILE=1`) puts the
//! activation at **30.9 % of detect** — 1305 calls, ~89 ms per image — ahead
//! of every individual convolution shape. That is this repo's Whisper result
//! reproduced a third time: an elementwise function nobody would nominate,
//! costing more than the arithmetic it decorates.
//!
//! SiLU is pure streaming: read 4 bytes, write 4 bytes. If it were running
//! at memory speed it would cost what a copy costs. **The ceiling below is
//! measured with a copy of the same buffers on the same machine**, not
//! quoted from a spec sheet — the mission plan already has one campaign
//! where a roofline built with the system under test was 2.5x too low.
//!
//! The suspect is `f32::round`. Rust's `round` is ties-AWAY-FROM-ZERO, which
//! has no single AVX2 instruction; `vroundps` implements ties-to-even. If
//! LLVM cannot lower it inline, the libm call sits in the middle of the loop
//! and blocks vectorization of everything around it — which is exactly the
//! mechanism the module doc claims to have already removed by replacing
//! `exp`. Removing a libm call and leaving another one behind is an easy
//! thing to do and an invisible thing to have done.
//!
//! Variants, all of which must agree with the current kernel to 1e-6:
//!   v0  current: `t.round()`
//!   v1  `t.round_ties_even()` — should lower to `vroundps`
//!   v2  magic-number rounding: `(t + 2^23*1.5) - 2^23*1.5`, no intrinsic
//!
//! ```sh
//! cargo run --release -p ffai-diana --example silu_ceiling
//! ```

use std::time::Instant;

const LOG2_E: f32 = std::f32::consts::LOG2_E;
const L1: f32 = std::f32::consts::LN_2;
const L2: f32 = L1 * L1 / 2.0;
const L3: f32 = L1 * L1 * L1 / 6.0;
const L4: f32 = L1 * L1 * L1 * L1 / 24.0;
const L5: f32 = L1 * L1 * L1 * L1 * L1 / 120.0;

#[inline(always)]
fn poly(f: f32) -> f32 {
    1.0 + f * (L1 + f * (L2 + f * (L3 + f * (L4 + f * L5))))
}

/// v0 — what ships today.
#[inline(always)]
fn silu_v0(x: f32) -> f32 {
    let t = (-x * LOG2_E).clamp(-125.0, 125.0);
    let n = t.round();
    let f = t - n;
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    x / (1.0 + poly(f) * scale)
}

/// v1 — ties-to-even, which is what the hardware instruction does.
#[inline(always)]
fn silu_v1(x: f32) -> f32 {
    let t = (-x * LOG2_E).clamp(-125.0, 125.0);
    let n = t.round_ties_even();
    let f = t - n;
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    x / (1.0 + poly(f) * scale)
}

/// v2 — rounding by float addition. No intrinsic, no call, and the rounded
/// integer falls out of the mantissa so the exponent write is free.
#[inline(always)]
fn silu_v2(x: f32) -> f32 {
    const MAGIC: f32 = 12582912.0; // 1.5 * 2^23
    let t = (-x * LOG2_E).clamp(-125.0, 125.0);
    let z = t + MAGIC;
    let n = z - MAGIC;
    let f = t - n;
    // z's low mantissa bits ARE n's two's-complement integer value.
    let scale = f32::from_bits((z.to_bits() << 23).wrapping_add(0x3f80_0000));
    x / (1.0 + poly(f) * scale)
}

fn bench(name: &str, n: usize, iters: usize, src: &[f32], dst: &mut [f32], f: fn(f32) -> f32) -> f64 {
    // Warm, then best-of: upward noise is what a loaded box adds, so the
    // minimum is the honest estimate of the code's own cost.
    for (o, i) in dst.iter_mut().zip(src) {
        *o = f(*i);
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        for (o, i) in dst.iter_mut().zip(src) {
            *o = f(*i);
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    let gbs = (n as f64 * 8.0) / best / 1e9; // 4 B in + 4 B out
    println!("{name:<28} {:>8.2} ms  {:>7.2} GB/s  {:>8.2} Gelem/s", best * 1e3, gbs, n as f64 / best / 1e9);
    best
}

fn main() {
    let n = 1 << 24; // 16 M elements, 64 MiB in + 64 MiB out — well past L3
    let iters = 7;
    // Values spanning what an activation actually sees, deterministic.
    let src: Vec<f32> = (0..n).map(|i| ((i % 2003) as f32 - 1000.0) * 0.01).collect();
    let mut dst = vec![0f32; n];

    println!("SiLU ceiling · {} M elements · best of {iters}\n", n >> 20);

    // The ceiling, measured with the same buffers on this machine.
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        dst.copy_from_slice(&src);
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!(
        "{:<28} {:>8.2} ms  {:>7.2} GB/s  {:>8.2} Gelem/s   <- ceiling",
        "memcpy (roofline)",
        best * 1e3,
        (n as f64 * 8.0) / best / 1e9,
        n as f64 / best / 1e9
    );
    let roof = best;

    println!();
    let t0 = bench("v0 round() [ships today]", n, iters, &src, &mut dst, silu_v0);
    let t1 = bench("v1 round_ties_even()", n, iters, &src, &mut dst, silu_v1);
    let t2 = bench("v2 magic-number rounding", n, iters, &src, &mut dst, silu_v2);

    // Agreement first: a faster kernel that computes something else is not a
    // faster kernel. Checked over the same distribution, not a happy sample.
    let (mut d1, mut d2) = (0f32, 0f32);
    for &x in src.iter().step_by(97) {
        let a = silu_v0(x);
        let s = a.abs().max(1e-6);
        d1 = d1.max((silu_v1(x) - a).abs() / s);
        d2 = d2.max((silu_v2(x) - a).abs() / s);
    }
    println!();
    println!("max relative disagreement vs v0:  v1 {d1:.3e}   v2 {d2:.3e}");
    println!();
    println!("v0 is {:.2}x off the copy ceiling", t0 / roof);
    println!("prize if SiLU hit the ceiling: it is 30.9% of detect, so");
    println!("  v1 would cut detect by {:.1}%  (pipeline {:.3}x)", 30.9 * (1.0 - t1 / t0), 1.0 / (1.0 - 0.309 * (1.0 - t1 / t0)));
    println!("  v2 would cut detect by {:.1}%  (pipeline {:.3}x)", 30.9 * (1.0 - t2 / t0), 1.0 / (1.0 - 0.309 * (1.0 - t2 / t0)));
    println!("  a PERFECT silu would cut it by {:.1}%  (pipeline {:.3}x) <- the hard ceiling", 30.9 * (1.0 - roof / t0), 1.0 / (1.0 - 0.309 * (1.0 - roof / t0)));
}
