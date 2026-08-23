//! Does Diana's hand-written AVX2 `silu` still earn its `unsafe`?
//!
//! # Why ask now
//!
//! The twin was written when `exp_fast` lived here and the scalar path was what
//! it was. Since then `ffai_core::fastmath` centralised the `+1.5*2^23`
//! rounding, and the same centralisation demoted `ffai-argus`'s AVX2 GELU
//! twin sharply — its scalar path got **2.42x faster** and the AVX2 advantage
//! fell from **2.30x to 1.19x**, because a call-free branch-free loop is one
//! the compiler widens on its own.
//!
//! Diana's twin predates that being shared. It may be earning much less than
//! when it was written, or it may not — which is the point: the Argus campaign
//! got the answer wrong four times by reasoning instead of measuring.
//!
//! # What it measured (2026-08-22, 4 M elements, best of 5)
//!
//! **The AVX2 twin KEEPS its `unsafe`** — 2.15-2.22x over a scalar path with
//! nothing left in it, 20/20 on the sign test. That is a different answer from
//! Argus's twin (1.19x, marginal), and the reason is the kernel: this one is
//! eight lanes of a polynomial with no gather and no cross-lane step, which is
//! the shape that keeps scaling with width.
//!
//! **But asking the question found a bigger defect than the answer.** The
//! first run read 4.23x, which did not fit — Argus had just demonstrated that a
//! call-free branch-free scalar loop closes most of that gap on its own. The
//! arithmetic not closing indicted the DECOMPOSITION, and one layer down
//! `exp_fast` was reading `old_rounding()` — a relaxed atomic load, which LLVM
//! will not hoist out of a loop — once per element:
//!
//! | | scalar | avx2 | advantage |
//! |---|---:|---:|---:|
//! | toggle read per element | 14.30 ms | 3.40 ms | 4.21x |
//! | toggle hoisted per call | **7.05 ms** | 3.18 ms | **2.22x** |
//!
//! **1.92x on the scalar path, bit-identical** (`max_abs` exactly 0), because
//! the toggle never changed the arithmetic — it only decided whether to look.
//! That path is not a rarely-taken fallback: `silu_scalar_pub` is called
//! per-element from seven loops in `epilogue.rs` and `conv3x3.rs` and from the
//! AVX2 kernel's own scalar tail, so the barrier was in the shipping path on
//! **every** CPU, this one included.
//!
//! The general form is the trap `rusty-fast-transcendentals` §4 already names
//! — replacing the libm call and leaving something else un-hoistable in the
//! loop removes nothing — with a new spelling: **the guard around the fix can
//! be the barrier the fix removed.**
//!
//! Both arms run in ONE process, interleaved, and the verdict is a **sign
//! test** rather than a ratio: this box swings ±12 %, so the ordering is
//! trustworthy long before the magnitude is.
//!
//! ```sh
//! cargo run --release -p ffai-diana --example silu_avx2_worth
//! ```
use candle_core::{Device, Tensor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    // A real activation shape, and one big enough that the fork-join is noise.
    let n = 1 << 22; // 4 M elements
    let xs = Tensor::rand(-8.0f32, 8.0, n, &d)?;

    let time = |f: &dyn Fn() -> candle_core::Result<Tensor>| -> f64 {
        let _ = f().expect("warm");
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            let o = f().expect("silu");
            std::hint::black_box(o.dims());
            best = best.min(t.elapsed().as_secs_f64());
        }
        best * 1e3
    };

    // The toggle already exists — `FFAI_DIANA_NO_AVX2=1` — but it is read once
    // and cached, so it cannot be flipped inside one process. Drive the two
    // paths directly instead.
    let avx2 = ffai_diana::silu_avx2::available();
    println!("silu, {} M elements, avx2 available: {avx2}\n", n as f64 / 1e6);
    if !avx2 {
        println!("no AVX2 on this CPU — nothing to compare");
        return Ok(());
    }

    let src: Vec<f32> = xs.to_vec1()?;
    let scalar = || -> candle_core::Result<Tensor> {
        let out: Vec<f32> = src.iter().map(|&v| ffai_diana::silu::silu_scalar_pub(v)).collect();
        Tensor::from_vec(out, n, &d)
    };
    let vector = || -> candle_core::Result<Tensor> {
        let mut out = vec![0.0f32; n];
        // SAFETY: `available()` returned true above.
        unsafe { ffai_diana::silu_avx2::silu_into(&src, &mut out) };
        Tensor::from_vec(out, n, &d)
    };

    // Equality first: a faster arm computing something else is not an arm.
    let (a, b) = (scalar()?, vector()?);
    let worst = (&a - &b)?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("  scalar vs avx2 max_abs: {worst:.3e}");

    // ARM 3 — the shared kernel with NOTHING between it and the loop.
    //
    // `silu_scalar` reaches `ffai_core::fastmath::exp` through `exp_fast`,
    // which first evaluates `old_rounding()` — a relaxed ATOMIC LOAD AND A
    // BRANCH, per element. LLVM does not hoist atomic loads out of loops, so
    // that toggle is a vectorization barrier sitting in the hot path of every
    // SiLU in the pipeline. This arm is the same arithmetic with the barrier
    // removed, which is the single variable separating it from `scalar`.
    let shared = || -> candle_core::Result<Tensor> {
        let out: Vec<f32> = src.iter().map(|&v| ffai_core::fastmath::silu(v)).collect();
        Tensor::from_vec(out, n, &d)
    };
    let c = shared()?;
    let d3 = (&a - &c)?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("  scalar vs shared max_abs: {d3:.3e}   (same math, toggle removed)");

    let (s_ms, v_ms, sh_ms) = (time(&scalar), time(&vector), time(&shared));

    // The sign test runs SINGLE shots, alternating, not best-of-5 each: the
    // point is to ask "did the vector arm win THIS round" 20 independent times.
    // Nesting a best-of inside would smooth away the very variation being
    // counted.
    let once = |f: &dyn Fn() -> candle_core::Result<Tensor>| -> f64 {
        let t = Instant::now();
        let o = f().expect("silu");
        std::hint::black_box(o.dims());
        t.elapsed().as_secs_f64()
    };
    let mut wins = 0;
    for i in 0..20 {
        // Alternate which arm goes first, so a warm-cache advantage cannot
        // accrue to one side.
        let (a, b) = if i % 2 == 0 {
            let s = once(&scalar);
            (s, once(&vector))
        } else {
            let v = once(&vector);
            (once(&scalar), v)
        };
        if b < a {
            wins += 1;
        }
    }
    println!("  scalar    {s_ms:>8.2} ms");
    println!("  avx2+fma  {v_ms:>8.2} ms   {:>5.2}x", s_ms / v_ms);
    println!("  sign test: AVX2 faster in {wins}/20 interleaved rounds");
    println!("  shared, no toggle  {sh_ms:>8.2} ms   {:>5.2}x vs scalar", s_ms / sh_ms);
    println!("  AVX2 over the toggle-free scalar: {:>5.2}x\n", sh_ms / v_ms);

    let ratio = s_ms / v_ms;
    if wins >= 19 && ratio >= 1.5 {
        println!("  VERDICT: keep. {ratio:.2}x is well clear of what a compiler gets");
        println!("  from the shared kernel on its own, so the `unsafe` is earning.");
    } else if wins >= 19 {
        println!("  VERDICT: MARGINAL — {ratio:.2}x, consistent but small. Same position");
        println!("  Argus's twin ended in (1.19x). Real, but a NEW kernel would not be");
        println!("  written for it; weigh it against the `unsafe` at the next change.");
    } else {
        println!("  VERDICT: NOT clear of this box's noise ({wins}/20). Refuse to claim");
        println!("  a win rather than average one out of a swing.");
    }
    Ok(())
}
