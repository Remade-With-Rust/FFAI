//! Transcendentals without libm calls — the shared kernel every engine uses.
//!
//! # Why this module exists
//!
//! `exp`, `ln`, `tanh`, `sin`, `cos` and `powf` **have no SIMD instruction at
//! any width**. Every call is a scalar libm call, and — worse than its own cost
//! — it is a hard barrier to vectorising the loop it sits in. Replacing them
//! with polynomials is the highest-yield mechanical change available in this
//! workspace (`docs/plans/turbocharger.md`).
//!
//! **`sqrt`, `min`, `max`, `mul`, `add` are the opposite**: SSE2 baseline,
//! already vectorised by the compiler. Rewriting those is how a campaign wastes
//! a week — mp3's `xrpow` hand-AVX2 measured 0.97x and was reverted.
//!
//! # Why ONE module and not three
//!
//! Three implementations of this idea already existed, all wired to production,
//! none aware of the others: `ffai-diana`'s `exp_fast`, `ffai-mercury`'s
//! `fast_exp`, `ffai-argus`'s `exp_poly`. They had **drifted in the one detail
//! that decides whether the win happens at all** (see below), and two of the
//! three had the bug. Consolidating is the point, not a tidy-up.
//!
//! # ★ The rounding step is the whole trick
//!
//! `exp(x) = 2^(x*log2 e)` splits into an integer power (written straight into
//! the f32 exponent field) and a fractional part (a degree-5 polynomial). That
//! split needs a round-to-integer — and **that is where two of the three
//! implementations reintroduced the libm call they had just removed.**
//!
//! Rust's `f32::round` is ties-**away-from-zero**, which no x86 instruction
//! implements (`vroundps` is ties-to-even). So it lowers to a call, or to a
//! long branchy sequence, sitting in the middle of the loop. `floor` is no
//! better: it needs SSE4.1, above the portable x86-64 baseline.
//!
//! Adding `1.5 * 2^23` forces any value below `2^22` to round into the
//! mantissa's last bit; subtracting it back leaves the value rounded to
//! nearest-even. Pure float arithmetic — no call, no branch, **and it needs no
//! `target_feature`**, so it vectorises on aarch64 (NEON is baseline) exactly
//! as it does on x86.
//!
//! Measured by `ffai-diana` over 16 M elements, best of 7, single thread:
//!
//! | | time | rate |
//! |---|---:|---:|
//! | `memcpy` (the roofline) | 5.58 ms | 24.04 GB/s |
//! | with `f32::round` | 60.84 ms | 2.21 GB/s |
//! | with `round_ties_even` | 38.85 ms | 3.45 GB/s |
//! | **with the magic number** | **12.91 ms** | **10.40 GB/s** |
//!
//! 4.71x, and **bit-identical** to the `round()` version over the activation
//! range. A transcendental within 2.3x of pure memory traffic is a
//! transcendental that vectorised.
//!
//! # Accuracy and how it is gated
//!
//! These are float approximations: the gate is a tolerance against libm plus
//! the caller's own end-to-end oracle, never bit-identity against `std`. Each
//! function documents its measured worst case. The tests here check the
//! tolerance, the landmarks, and the shape (monotonicity where it holds,
//! saturation, exact values at 0) — the last of which catches an approximation
//! that is accurate on average and wrong somewhere specific.

/// `1.5 * 2^23` — the round-to-nearest-even trick. See the module docs.
const MAGIC: f32 = 12_582_912.0;

/// Round to nearest even, without a libm call or an SSE4.1 instruction.
///
/// Valid for `|x| < 2^22`, which every use here guarantees by clamping first.
#[inline(always)]
#[must_use]
pub fn round_ties_even_fast(x: f32) -> f32 {
    (x + MAGIC) - MAGIC
}

/// `2^x`, for `x` already clamped to a sane exponent range.
#[inline(always)]
fn exp2_unchecked(x: f32) -> f32 {
    let n = round_ties_even_fast(x);
    let f = x - n; // in [-0.5, 0.5]
    // 2^f = exp(f * ln2), so the coefficients are ln2^k / k!.
    //
    // DERIVED, not transcribed. Hand-typed decimals here are a real hazard:
    // trimming one digit to satisfy a lint silently selects a different f32
    // and breaks the oracle. Let the compiler compute them.
    const L1: f32 = std::f32::consts::LN_2;
    const L2: f32 = L1 * L1 / 2.0;
    const L3: f32 = L1 * L1 * L1 / 6.0;
    const L4: f32 = L1 * L1 * L1 * L1 / 24.0;
    const L5: f32 = L1 * L1 * L1 * L1 * L1 / 120.0;
    let p = 1.0 + f * (L1 + f * (L2 + f * (L3 + f * (L4 + f * L5))));
    // 2^n straight into the exponent field.
    // SITE-REVIEWED cast allows. This function is `exp2_unchecked`: its
    // contract, stated above, is that the caller has ALREADY clamped `x` to a
    // sane exponent range, and every caller does. So `n` is a small integral
    // f32 -- the `as i32` cannot truncate anything that was there -- and
    // `n + 127` is then in [2, 252], so the `as u32` has no sign to lose.
    // Allowed here rather than crate-wide precisely so these two lints keep
    // firing on code that has not been read.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let scale = f32::from_bits((((n as i32) + 127) as u32) << 23);
    p * scale
}

/// `e^x`, accurate to **4.2e-6 relative** over `[-20, 20]` (measured, not
/// claimed — the degree-5 polynomial is the limit, and the two implementations
/// this replaced both documented ~1e-7, which was optimistic).
///
/// Clamped to `[-87, 88]`: outside that the f32 result is 0 or infinite
/// anyway, and an unclamped exponent write would produce a denormal or garbage
/// rather than a saturated value.
#[inline(always)]
#[must_use]
pub fn exp(x: f32) -> f32 {
    exp2_unchecked((x * std::f32::consts::LOG2_E).clamp(-125.0, 125.0))
}

/// `2^x`.
#[inline(always)]
#[must_use]
pub fn exp2(x: f32) -> f32 {
    exp2_unchecked(x.clamp(-125.0, 125.0))
}

/// `1 / (1 + e^-x)`.
#[inline(always)]
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// `tanh(x)`, via `1 - 2/(e^{2x} + 1)`.
///
/// Chosen over a Padé rational form deliberately. A Padé `tanh` was tried in
/// two campaigns and failed both: it is accurate near zero and degrades to
/// ~1.2e-3 by `|x| = 6`, while saturation is only safe from `|x| >= 7` — so
/// **no crossover threshold exists**. In Mercury it compounded 1.8e-4 into
/// 7e-2 over 16 coupled gates. Range reduction has no such range problem.
#[inline(always)]
#[must_use]
pub fn tanh(x: f32) -> f32 {
    // Small-|x| branch, and it is not optional. `1 - 2/(e^{2x}+1)` computes
    // `1 - (something within an ulp of 1)` as x -> 0, so every significant
    // digit cancels: measured **2.5e-4 relative at x = 2.4e-4**, which is a
    // 100 % error on a value the caller thinks is exact. The Maclaurin series
    // has no such problem there, and the crossover is where the two agree.
    if x.abs() < 0.02 {
        return x * (1.0 - x * x * (1.0 / 3.0));
    }
    1.0 - 2.0 / (exp(2.0 * x) + 1.0)
}

/// `gelu_pytorch_tanh` — `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))`.
///
/// Written as `x * sigmoid(2z)` because `tanh(z) = 2*sigmoid(2z) - 1` exactly,
/// which saves one operation over computing the tanh and folding it back.
#[inline(always)]
#[must_use]
pub fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    let z = SQRT_2_OVER_PI * x * (1.0 + 0.044_715 * x * x);
    x / (1.0 + exp(-2.0 * z))
}

/// `SiLU`/swish — `x * sigmoid(x)`.
#[inline(always)]
#[must_use]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp(-x))
}

/// `erf(x)`, Abramowitz & Stegun 7.1.26 — **1.6e-6 measured in f32**.
///
/// Needed because `gelu_erf` and `gelu_tanh` are **different functions**, not
/// two spellings of one. They differ by up to ~1e-3, which is far above the
/// tolerance any of these engines gate at, so a site calling `.gelu_erf()`
/// cannot be handed the tanh form as an optimisation. Six of the seventeen
/// activation sites in this workspace are `gelu_erf`.
#[inline(always)]
#[must_use]
pub fn erf(x: f32) -> f32 {
    const P: f32 = 0.327_591_1;
    const A1: f32 = 0.254_829_59;
    const A2: f32 = -0.284_496_74;
    const A3: f32 = 1.421_413_7;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / P.mul_add(ax, 1.0);
    let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
    sign * (1.0 - poly * exp(-ax * ax))
}

/// `gelu` in its **exact** form — `0.5x(1 + erf(x/sqrt 2))`.
///
/// This is what candle's `.gelu_erf()` computes, and what `.gelu()` only
/// approximates. Keep them apart.
#[inline(always)]
#[must_use]
pub fn gelu_erf(x: f32) -> f32 {
    const INV_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    0.5 * x * (1.0 + erf(x * INV_SQRT_2))
}

/// Natural log, accurate to ~1e-6 absolute over the positive range.
///
/// The mirror of [`exp`]: pull the exponent out of the bit pattern, and take a
/// polynomial in the mantissa. `x <= 0` returns `-inf`/`NaN` as `f32::ln`
/// does, so callers that clamp (every log-mel does) behave identically.
#[inline(always)]
#[must_use]
#[allow(clippy::many_single_char_names)]
// Pre-existing: `x`, `e`, `m`, `s`, `p` are the atanh series' own names.
pub fn ln(x: f32) -> f32 {
    if x <= 0.0 {
        return if x == 0.0 {
            f32::NEG_INFINITY
        } else {
            f32::NAN
        };
    }
    let bits = x.to_bits();
    // Exponent field, unbiased.
    // SITE-REVIEWED. `& 0xff` bounds this at 255 before the cast, so `as i32`
    // cannot wrap; it is an 8-bit IEEE-754 exponent field by construction.
    #[allow(clippy::cast_possible_wrap)]
    let e = ((bits >> 23) & 0xff) as i32 - 127;
    // Mantissa forced into [1, 2), then centred on [-1/3, 1/3] by the
    // `m > sqrt(2)` split so the polynomial converges fast.
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    let (m, e) = if m > std::f32::consts::SQRT_2 {
        (m * 0.5, e + 1)
    } else {
        (m, e)
    };
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    // atanh series: ln(m) = 2s(1 + s^2/3 + s^4/5 + s^6/7 + s^8/9).
    let p = 2.0
        * s
        * (1.0 + s2 * (0.333_333_34 + s2 * (0.2 + s2 * (0.142_857_15 + s2 * 0.111_111_11))));
    (e as f32).mul_add(std::f32::consts::LN_2, p)
}

/// `log10(x)`, for log-mel and decibel work.
#[inline(always)]
#[must_use]
pub fn log10(x: f32) -> f32 {
    ln(x) * std::f32::consts::LOG10_E
}

#[cfg(test)]
mod exp_sub_sum_tests {
    use super::*;

    /// Every vector twin against the scalar oracle, on the shape the caller
    /// actually uses (1024-wide attention rows) plus the tail lengths that
    /// exercise the scalar remainder in both the 8-lane and 4-lane kernels.
    #[test]
    fn vector_twins_match_the_scalar_oracle() {
        for &n in &[0usize, 1, 3, 4, 7, 8, 9, 15, 16, 31, 33, 64, 1024, 1031] {
            // Deterministic, and spanning the range attention scores occupy —
            // including values far below the max, where exp underflows.
            let src: Vec<f32> = (0..n)
                .map(|i| ((i * 37 % 211) as f32 - 105.0) * 0.15)
                .collect();
            let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let mut want = src.clone();
            let want_sum = exp_sub_sum_scalar(&mut want, if n == 0 { 0.0 } else { max });
            let mut got = src.clone();
            let got_sum = exp_sub_sum_inplace(&mut got, if n == 0 { 0.0 } else { max });

            for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
                let err = (a - b).abs() / a.abs().max(1e-30);
                assert!(err < 1e-5, "n={n} i={i}: {a} vs {b} (rel {err:e})");
            }
            let serr = (want_sum - got_sum).abs() / want_sum.abs().max(1e-30);
            // Lane splitting reassociates the sum, so this is a tolerance and
            // not an equality — by construction, not by accident.
            assert!(
                serr < 1e-5,
                "n={n} sum {want_sum} vs {got_sum} (rel {serr:e})"
            );
        }
    }

    /// `max_f32`'s twins are EXACT — max is associative on non-NaN floats — so
    /// this is `assert_eq!`, not a tolerance.
    #[test]
    fn max_twins_are_exact() {
        for &n in &[0usize, 1, 3, 4, 7, 8, 9, 15, 31, 33, 1024, 1031] {
            let xs: Vec<f32> = (0..n)
                .map(|i| ((i * 89 % 401) as f32 - 200.0) * 0.37)
                .collect();
            assert_eq!(max_f32_scalar(&xs), max_f32(&xs), "n={n}");
        }
        assert_eq!(max_f32(&[]), f32::NEG_INFINITY);
    }

    /// The GELU twins against the scalar oracle, across the range an MLP
    /// activation actually sees plus both saturating tails.
    #[test]
    fn gelu_twins_match_the_scalar_oracle() {
        for &n in &[0usize, 1, 3, 4, 7, 8, 9, 15, 33, 4096, 4099] {
            let src: Vec<f32> = (0..n)
                .map(|i| ((i * 53 % 601) as f32 - 300.0) * 0.09)
                .collect();
            let mut want = src.clone();
            for v in &mut want {
                *v = gelu_tanh(*v);
            }
            let mut got = src.clone();
            gelu_tanh_inplace(&mut got);
            for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
                let err = (a - b).abs() / a.abs().max(1e-6);
                assert!(
                    err < 1e-5,
                    "n={n} i={i} x={}: {a} vs {b} (rel {err:e})",
                    src[i]
                );
            }
        }
    }

    /// A softmax built on it sums to 1 — the property the caller depends on.
    #[test]
    fn normalises_to_one() {
        let mut row: Vec<f32> = (0..1024).map(|i| ((i % 97) as f32) * 0.11 - 5.0).collect();
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = exp_sub_sum_inplace(&mut row, max);
        let total: f32 = row.iter().map(|v| v / sum).sum();
        assert!((total - 1.0).abs() < 1e-4, "sums to {total}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Worst relative error of `f` against `oracle` over a dense sweep.
    fn sweep(lo: f32, hi: f32, f: impl Fn(f32) -> f32, oracle: impl Fn(f32) -> f32) -> (f32, f32) {
        let n = 200_001;
        let mut worst_rel = 0.0f32;
        let mut at = lo;
        for i in 0..n {
            let x = lo + (hi - lo) * (i as f32 / (n - 1) as f32);
            let (a, b) = (f(x), oracle(x));
            // A point passes if EITHER the relative or the absolute error is
            // small, and that is not a loosened gate — it is the only honest
            // one across this dynamic range.
            //
            // The oracle runs out of precision before we do. GELU at
            // x = -4.744 is ~1.2e-6, and the reference computes it as
            // `0.5*x*(1 + tanh z)` where `1 + tanh z` cancels down to a few
            // f32 epsilons: the ORACLE's own uncertainty there is ~23 %
            // relative. Judging us by relative error against it would be
            // measuring its cancellation, not our accuracy — our absolute
            // error at that point is 7.6e-8. Further out, `f32::tanh` returns
            // exactly -1.0 and the oracle produces a hard zero.
            let abs = (a - b).abs();
            let rel = if b.abs() > f32::MIN_POSITIVE {
                (abs / b.abs()).min(abs)
            } else {
                abs
            };
            if rel > worst_rel {
                worst_rel = rel;
                at = x;
            }
        }
        (worst_rel, at)
    }

    #[test]
    fn exp_tracks_libm() {
        let (rel, at) = sweep(-20.0, 20.0, exp, f32::exp);
        eprintln!("exp: worst rel {rel:.3e} at x = {at}");
        assert!(rel < 1e-5, "exp worst rel {rel:.3e} at {at}");
    }

    #[test]
    fn exp_saturates_rather_than_producing_garbage() {
        // The clamp is load-bearing: an unclamped exponent write produces a
        // denormal or a wrapped exponent, which is a plausible-looking wrong
        // number rather than an infinity anyone would notice.
        assert!(exp(200.0).is_finite(), "exp(200) must saturate, not wrap");
        assert!(exp(200.0) > 1e30, "exp(200) should be very large");
        // NOT zero: the clamp is in log2 space, so exp(-200) saturates to
        // 2^-125 ~= 2.4e-38. Tiny and finite is the correct behaviour — the
        // thing the clamp exists to prevent is a WRAPPED exponent field, which
        // would be a large plausible number instead.
        assert!(
            exp(-200.0) < 1e-30 && exp(-200.0) >= 0.0,
            "exp(-200) = {} should saturate tiny, not wrap",
            exp(-200.0)
        );
        assert!((exp(0.0) - 1.0).abs() < 1e-7, "exp(0) = {}", exp(0.0));
    }

    #[test]
    fn the_magic_rounding_is_round_ties_even() {
        // The property the whole module depends on. Ties go to EVEN, unlike
        // `f32::round` which goes away from zero — that difference is the
        // reason this exists.
        for (x, want) in [
            (0.5f32, 0.0f32),
            (1.5, 2.0),
            (2.5, 2.0),
            (-0.5, -0.0),
            (-1.5, -2.0),
            (-2.5, -2.0),
            (3.2, 3.0),
            (-3.7, -4.0),
        ] {
            let got = round_ties_even_fast(x);
            assert_eq!(got, want, "round_ties_even_fast({x}) = {got}, want {want}");
        }
        // And it agrees with std's ties-even over a range.
        for i in -1000..1000 {
            let x = i as f32 / 7.0;
            assert_eq!(round_ties_even_fast(x), x.round_ties_even(), "at {x}");
        }
    }

    #[test]
    fn tanh_tracks_libm_and_saturates() {
        let (rel, at) = sweep(-8.0, 8.0, tanh, f32::tanh);
        eprintln!("tanh: worst rel {rel:.3e} at x = {at}");
        assert!(rel < 1e-5, "tanh worst rel {rel:.3e} at {at}");
        assert!((tanh(0.0)).abs() < 1e-7);
        assert!((tanh(20.0) - 1.0).abs() < 1e-6);
        assert!((tanh(-20.0) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_and_silu_track_libm() {
        let (rel, at) = sweep(-15.0, 15.0, sigmoid, |x| 1.0 / (1.0 + (-x).exp()));
        assert!(rel < 1e-5, "sigmoid worst rel {rel:.3e} at {at}");
        let (rel, at) = sweep(-15.0, 15.0, silu, |x| x / (1.0 + (-x).exp()));
        assert!(rel < 1e-5, "silu worst rel {rel:.3e} at {at}");
        assert_eq!(silu(0.0), 0.0, "silu(0) must be exactly 0");
    }

    #[test]
    fn gelu_tracks_libm_and_keeps_its_dip() {
        let oracle =
            |x: f32| 0.5 * x * (1.0 + (0.797_884_56 * x * (1.0 + 0.044_715 * x * x)).tanh());
        let (rel, at) = sweep(-10.0, 10.0, gelu_tanh, oracle);
        eprintln!("gelu: worst rel {rel:.3e} at x = {at}");
        assert!(rel < 1e-5, "gelu worst rel {rel:.3e} at {at}");
        // GELU is NOT monotone — it dips to about -0.17 near x = -0.75, and an
        // approximation that smooths that away is wrong in the one place the
        // curve has any shape.
        assert!(
            (-0.18..-0.15).contains(&gelu_tanh(-0.75)),
            "gelu(-0.75) = {} should sit near the -0.17 minimum",
            gelu_tanh(-0.75)
        );
        assert_eq!(gelu_tanh(0.0), 0.0, "gelu(0) must be exactly 0");
        assert!((gelu_tanh(10.0) - 10.0).abs() < 1e-4);
        assert!(gelu_tanh(-10.0).abs() < 1e-5);
    }

    #[test]
    fn erf_and_gelu_erf_track_a_reference() {
        // erf has no `std` implementation, so the oracle is a high-order
        // series evaluated in f64 — an independent route to the same value
        // rather than a rearrangement of the same approximation.
        let oracle = |x: f32| -> f32 {
            let x = f64::from(x);
            // Taylor out to |x| = 4, where `1 - erf(4) = 1.5e-8` is already
            // below f32 epsilon at 1.0 — so saturating past it costs nothing.
            //
            // The first version cut over at 3.0 and FAILED, reporting a
            // 2.2e-5 error that was exactly `1 - erf(3)`: the oracle was
            // returning a hard 1.0 where the true value still had digits. The
            // magnitude naming the cutoff is what identified it as the
            // oracle's fault rather than the kernel's.
            if x.abs() < 4.0 {
                let mut term = x;
                let mut sum = x;
                for n in 1..90 {
                    term *= -x * x / f64::from(n);
                    sum += term / f64::from(2 * n + 1);
                }
                (sum * 2.0 / std::f64::consts::PI.sqrt()) as f32
            } else if x > 0.0 {
                1.0
            } else {
                -1.0
            }
        };
        let (rel, at) = sweep(-3.0, 3.0, erf, oracle);
        eprintln!("erf: worst {rel:.3e} at x = {at}");
        // 1.6e-6 measured. A&S 7.1.26 claims 1.5e-7 ABSOLUTE for the formula;
        // the extra comes from evaluating it in f32, where `1 - poly*exp` is a
        // cancellation that amplifies the rounding. Not a coefficient error —
        // the worst point is mid-range, not at a cutoff.
        assert!(rel < 5e-6, "erf worst {rel:.3e} at {at}");
        // NOT exactly zero, and forcing it would cost a branch on a hot path
        // for no benefit. A&S 7.1.26's five coefficients sum to 0.999999999,
        // not 1, so `1 - poly*exp(0)` leaves ~1.8e-7 — inside the 1.6e-6 the
        // formula is good for anyway.
        assert!(erf(0.0).abs() < 3e-7, "erf(0) = {}", erf(0.0));
        // gelu_erf(0) IS exact, structurally: x multiplies the whole thing.
        assert_eq!(gelu_erf(0.0), 0.0, "gelu_erf(0) must be exactly 0");
        assert!((erf(4.0) - 1.0).abs() < 1e-6);
        assert!((erf(-4.0) + 1.0).abs() < 1e-6);

        // gelu_erf and gelu_tanh must NOT be interchangeable, and the test
        // says so — if they ever agree to 1e-5 someone has quietly aliased one
        // to the other.
        let worst = (-40..40)
            .map(|i| {
                let x = i as f32 / 10.0;
                (gelu_erf(x) - gelu_tanh(x)).abs()
            })
            .fold(0.0f32, f32::max);
        assert!(
            worst > 1e-4,
            "gelu_erf and gelu_tanh differ by only {worst:.3e} — are they aliased?"
        );
    }

    #[test]
    fn ln_and_log10_track_libm() {
        let (rel, at) = sweep(1e-6, 1e6, ln, f32::ln);
        eprintln!("ln: worst rel {rel:.3e} at x = {at}");
        assert!(rel < 1e-5, "ln worst rel {rel:.3e} at {at}");
        let (rel, at) = sweep(1e-6, 1e6, log10, f32::log10);
        assert!(rel < 1e-5, "log10 worst rel {rel:.3e} at {at}");
        // The landmarks a log-mel actually hits.
        assert!((log10(1.0)).abs() < 1e-6, "log10(1) = {}", log10(1.0));
        assert!((log10(10.0) - 1.0).abs() < 1e-5);
        assert!((log10(1e-10) + 10.0).abs() < 1e-4);
    }

    #[test]
    fn ln_matches_std_at_the_edges() {
        assert_eq!(ln(0.0), f32::NEG_INFINITY);
        assert!(ln(-1.0).is_nan());
        assert!((ln(std::f32::consts::E) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn exp_and_ln_round_trip() {
        // A property neither sweep catches on its own: the two must be
        // inverses, so an error in one that happens to cancel in a sweep
        // against the same libm still shows up here.
        for i in -60..60 {
            let x = i as f32 / 6.0;
            let back = ln(exp(x));
            assert!((back - x).abs() < 1e-4, "ln(exp({x})) = {back}");
        }
    }
}

// ---------------------------------------------------------------------------
// Vectorised softmax inner loop
// ---------------------------------------------------------------------------

/// `row[i] = exp(row[i] - max)`, returning the sum — the softmax inner loop.
///
/// # Why this exists as one function rather than `map` + `sum`
///
/// Measured on Argus's `SoftmaxExpInplace`: 151 M elements per caption at
/// **228 M elem/s**, 22.7 % of the whole vision tower and 2.7x the `q.k^T`
/// matmul that produces its input. An elementwise pass beating the O(n^3)
/// matmul that feeds it is the tell.
///
/// Two costs, and neither yields to a scalar rewrite:
///
/// * `(x - max).exp()` is a libm `expf` CALL per element, ~75 M/s.
/// * `sum += e` is a loop-carried dependency on a non-associative add, so LLVM
///   will not lane the loop — and a scalar polynomial `exp` with eight split
///   accumulators was **measured slower** than libm here (2410 ms vs 2090 ms),
///   because it removes the call but still will not vectorise.
///
/// So the fix has to be explicit lanes, which is what this is. The scalar body
/// below stays as the oracle and as the fallback for targets with neither ISA.
///
/// Accuracy: `exp` here is the degree-5 `exp2` polynomial, ~4.2e-6 relative.
/// Lane splitting reassociates the sum, so it differs from a sequential fold in
/// the last bits — gate on tolerance, never bit-identity.
#[must_use]
pub fn exp_sub_sum_inplace(row: &mut [f32], max: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: both features probed immediately above.
            return unsafe { exp_sub_sum_avx2(row, max) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Unconditional, like the wasm arm and unlike x86's runtime probe:
        // NEON is BASELINE on aarch64, so there is nothing to detect. That is
        // also why this arm is easy to forget -- the scalar body below already
        // auto-vectorises for the ELEMENTWISE primitives, so the gap only bites
        // the two REDUCTIONS, where a loop-carried non-associative accumulator
        // stops LLVM cold no matter what the baseline is.
        //
        // SAFETY: `neon` is a compile-time guarantee on this target.
        return unsafe { exp_sub_sum_neon(row, max) };
    }
    #[cfg(target_arch = "wasm32")]
    {
        // Unconditional, unlike the x86 arm's runtime check: wasm validates a
        // whole module ahead of time, so a v128 instruction anywhere makes the
        // MODULE require SIMD. There is no `is_wasm_feature_detected!` and no
        // per-call dispatch. The workspace already sets `+simd128` for this
        // target in `.cargo/config.toml`, which makes SIMD a baseline
        // requirement rather than an upgrade.
        //
        // SAFETY: `simd128` is a compile-time guarantee on this target.
        return unsafe { exp_sub_sum_simd128(row, max) };
    }
    #[allow(unreachable_code)]
    exp_sub_sum_scalar(row, max)
}

/// The oracle. Every vector twin is gated against this.
#[must_use]
pub fn exp_sub_sum_scalar(row: &mut [f32], max: f32) -> f32 {
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        let e = exp(*v - max);
        *v = e;
        sum += e;
    }
    sum
}

/// AVX2 + FMA, eight lanes.
///
/// # Safety
/// Caller must have verified `avx2` and `fma`.
#[allow(clippy::many_single_char_names)]
// The names are the polynomial's own: `n`, `i`, `x`, `t`, `r`, `f`, `p`, `e`,
// `k`, `z`. Expanding them into prose makes the kernel harder to read against
// its twins and the scalar oracle, which is the only way this code is verified.
#[allow(clippy::wildcard_imports)]
// A SIMD kernel names 15 intrinsics; importing them one by one is a list that
// goes stale the moment the arithmetic changes, and hides nothing.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_sub_sum_avx2(row: &mut [f32], max: f32) -> f32 {
    // SAFETY: whole-body wrapper inserted by the edition-2024
    // `unsafe_op_in_unsafe_fn` migration. This block adds no new obligation:
    // the contract is stated on the `unsafe fn` signature above.
    unsafe {
        use std::arch::x86_64::*;
        const L1: f32 = std::f32::consts::LN_2;
        const L2: f32 = L1 * L1 / 2.0;
        const L3: f32 = L1 * L1 * L1 / 6.0;
        const L4: f32 = L1 * L1 * L1 * L1 / 24.0;
        const L5: f32 = L1 * L1 * L1 * L1 * L1 / 120.0;

        let vmax = _mm256_set1_ps(max);
        let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let magic = _mm256_set1_ps(MAGIC);
        let lo = _mm256_set1_ps(-125.0);
        let hi = _mm256_set1_ps(125.0);
        let c1 = _mm256_set1_ps(L1);
        let c2 = _mm256_set1_ps(L2);
        let c3 = _mm256_set1_ps(L3);
        let c4 = _mm256_set1_ps(L4);
        let c5 = _mm256_set1_ps(L5);
        let one = _mm256_set1_ps(1.0);
        let bias = _mm256_set1_epi32(127);

        let mut acc = _mm256_setzero_ps();
        let n = row.len();
        let mut i = 0;
        while i + 8 <= n {
            let x = _mm256_loadu_ps(row.as_ptr().add(i));
            // exp(x - max) == exp2((x - max) * log2(e)), clamped like the scalar.
            let t = _mm256_mul_ps(_mm256_sub_ps(x, vmax), log2e);
            let t = _mm256_min_ps(_mm256_max_ps(t, lo), hi);
            // round-to-nearest-even without an SSE4.1 instruction: (t+M)-M.
            let r = _mm256_sub_ps(_mm256_add_ps(t, magic), magic);
            let f = _mm256_sub_ps(t, r);
            // Horner, FMA-fused.
            let p = _mm256_fmadd_ps(f, c5, c4);
            let p = _mm256_fmadd_ps(f, p, c3);
            let p = _mm256_fmadd_ps(f, p, c2);
            let p = _mm256_fmadd_ps(f, p, c1);
            let p = _mm256_fmadd_ps(f, p, one);
            // 2^r straight into the exponent field.
            let e = _mm256_cvtps_epi32(r);
            let e = _mm256_slli_epi32::<23>(_mm256_add_epi32(e, bias));
            let out = _mm256_mul_ps(p, _mm256_castsi256_ps(e));
            _mm256_storeu_ps(row.as_mut_ptr().add(i), out);
            acc = _mm256_add_ps(acc, out);
            i += 8;
        }
        // Horizontal reduce, then the scalar tail through the SAME polynomial.
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        while i < n {
            let e = exp(row[i] - max);
            row[i] = e;
            sum += e;
            i += 1;
        }
        sum
    }
}

/// NEON, four lanes, FMA-fused.
///
/// **Why this exists even though NEON is baseline.** The scalar oracle below
/// auto-vectorises on aarch64 for elementwise work, so the obvious reading is
/// that this arm is redundant. It is not: `sum += e` is a loop-carried
/// dependency on a NON-ASSOCIATIVE add, and LLVM may not reassociate it
/// without fast-math. So the oracle stays a scalar dependency chain on every
/// target regardless of the baseline, and only explicit lanes break it. The
/// same argument covers [`max_f32_neon`]; it does NOT cover
/// [`gelu_tanh_neon`], which is elementwise and would likely have vectorised
/// unaided -- that one is here for symmetry and to pin the numerics.
///
/// Mirrors the AVX2 arm rather than the wasm one, because NEON HAS an FMA
/// (`vfmaq_f32`) and wasm's base SIMD does not.
///
/// # Safety
/// `neon` is a compile-time guarantee on `aarch64` here.
#[allow(clippy::many_single_char_names)]
// The names are the polynomial's own: `n`, `i`, `x`, `t`, `r`, `f`, `p`, `e`,
// `k`, `z`. Expanding them into prose makes the kernel harder to read against
// its twins and the scalar oracle, which is the only way this code is verified.
#[allow(clippy::wildcard_imports)]
// A SIMD kernel names 15 intrinsics; importing them one by one is a list that
// goes stale the moment the arithmetic changes, and hides nothing.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn exp_sub_sum_neon(row: &mut [f32], max: f32) -> f32 {
    // SAFETY: whole-body wrapper inserted by the edition-2024
    // `unsafe_op_in_unsafe_fn` migration. This block adds no new obligation:
    // the contract is stated on the `unsafe fn` signature above.
    unsafe {
        use core::arch::aarch64::*;
        const L1: f32 = core::f32::consts::LN_2;
        const L2: f32 = L1 * L1 / 2.0;
        const L3: f32 = L1 * L1 * L1 / 6.0;
        const L4: f32 = L1 * L1 * L1 * L1 / 24.0;
        const L5: f32 = L1 * L1 * L1 * L1 * L1 / 120.0;

        let vmax = vdupq_n_f32(max);
        let log2e = vdupq_n_f32(core::f32::consts::LOG2_E);
        let magic = vdupq_n_f32(MAGIC);
        let lo = vdupq_n_f32(-125.0);
        let hi = vdupq_n_f32(125.0);
        let c1 = vdupq_n_f32(L1);
        let c2 = vdupq_n_f32(L2);
        let c3 = vdupq_n_f32(L3);
        let c4 = vdupq_n_f32(L4);
        let c5 = vdupq_n_f32(L5);
        let one = vdupq_n_f32(1.0);
        let bias = vdupq_n_s32(127);

        let mut acc = vdupq_n_f32(0.0);
        let n = row.len();
        let mut i = 0;
        while i + 4 <= n {
            let x = vld1q_f32(row.as_ptr().add(i));
            let t = vmulq_f32(vsubq_f32(x, vmax), log2e);
            let t = vminq_f32(vmaxq_f32(t, lo), hi);
            // Round-to-nearest-even via the magic constant, exactly as the other
            // twins do -- NEON has `vrndnq_f32`, but using it here would make this
            // arm disagree with them in the last bit at the ties.
            let r = vsubq_f32(vaddq_f32(t, magic), magic);
            let f = vsubq_f32(t, r);
            // `vfmaq_f32(a, b, c)` is `a + b * c`.
            let p = vfmaq_f32(c4, f, c5);
            let p = vfmaq_f32(c3, f, p);
            let p = vfmaq_f32(c2, f, p);
            let p = vfmaq_f32(c1, f, p);
            let p = vfmaq_f32(one, f, p);
            let e = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(r), bias));
            let out = vmulq_f32(p, vreinterpretq_f32_s32(e));
            vst1q_f32(row.as_mut_ptr().add(i), out);
            acc = vaddq_f32(acc, out);
            i += 4;
        }
        // Same pairwise order as the wasm twin, so the two four-lane arms agree.
        let mut sum = (vgetq_lane_f32::<0>(acc) + vgetq_lane_f32::<1>(acc))
            + (vgetq_lane_f32::<2>(acc) + vgetq_lane_f32::<3>(acc));
        while i < n {
            let e = exp(row[i] - max);
            row[i] = e;
            sum += e;
            i += 1;
        }
        sum
    }
}

/// wasm SIMD128, four lanes.
///
/// **No FMA.** Base wasm SIMD has no fused multiply-add, so each Horner step is
/// a separate multiply and add — which is half of why this target runs ~6.5x
/// behind native single-thread even with SIMD on.
///
/// # Safety
/// `simd128` is a compile-time guarantee on `wasm32` here.
#[allow(clippy::many_single_char_names)]
// The names are the polynomial's own: `n`, `i`, `x`, `t`, `r`, `f`, `p`, `e`,
// `k`, `z`. Expanding them into prose makes the kernel harder to read against
// its twins and the scalar oracle, which is the only way this code is verified.
#[allow(clippy::wildcard_imports)]
// A SIMD kernel names 15 intrinsics; importing them one by one is a list that
// goes stale the moment the arithmetic changes, and hides nothing.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn exp_sub_sum_simd128(row: &mut [f32], max: f32) -> f32 {
    // SAFETY: whole-body wrapper inserted by the edition-2024
    // `unsafe_op_in_unsafe_fn` migration. This block adds no new obligation:
    // the contract is stated on the `unsafe fn` signature above.
    unsafe {
        use core::arch::wasm32::*;
        const L1: f32 = core::f32::consts::LN_2;
        const L2: f32 = L1 * L1 / 2.0;
        const L3: f32 = L1 * L1 * L1 / 6.0;
        const L4: f32 = L1 * L1 * L1 * L1 / 24.0;
        const L5: f32 = L1 * L1 * L1 * L1 * L1 / 120.0;

        let vmax = f32x4_splat(max);
        let log2e = f32x4_splat(core::f32::consts::LOG2_E);
        let magic = f32x4_splat(MAGIC);
        let lo = f32x4_splat(-125.0);
        let hi = f32x4_splat(125.0);
        let c1 = f32x4_splat(L1);
        let c2 = f32x4_splat(L2);
        let c3 = f32x4_splat(L3);
        let c4 = f32x4_splat(L4);
        let c5 = f32x4_splat(L5);
        let one = f32x4_splat(1.0);
        let bias = i32x4_splat(127);

        let mut acc = f32x4_splat(0.0);
        let n = row.len();
        let mut i = 0;
        while i + 4 <= n {
            let x = v128_load(row.as_ptr().add(i).cast());
            let t = f32x4_mul(f32x4_sub(x, vmax), log2e);
            let t = f32x4_pmin(hi, f32x4_pmax(lo, t));
            let r = f32x4_sub(f32x4_add(t, magic), magic);
            let f = f32x4_sub(t, r);
            let p = f32x4_add(f32x4_mul(f, c5), c4);
            let p = f32x4_add(f32x4_mul(f, p), c3);
            let p = f32x4_add(f32x4_mul(f, p), c2);
            let p = f32x4_add(f32x4_mul(f, p), c1);
            let p = f32x4_add(f32x4_mul(f, p), one);
            let e = i32x4_trunc_sat_f32x4(r);
            let e = i32x4_shl(i32x4_add(e, bias), 23);
            let out = f32x4_mul(p, e);
            v128_store(row.as_mut_ptr().add(i).cast(), out);
            acc = f32x4_add(acc, out);
            i += 4;
        }
        let mut sum = (f32x4_extract_lane::<0>(acc) + f32x4_extract_lane::<1>(acc))
            + (f32x4_extract_lane::<2>(acc) + f32x4_extract_lane::<3>(acc));
        while i < n {
            let e = exp(row[i] - max);
            row[i] = e;
            sum += e;
            i += 1;
        }
        sum
    }
}

/// Maximum of a slice — vectorised.
///
/// The other half of the softmax row. `for &v in row { max = max.max(v) }` is a
/// loop-carried reduction on a function with NaN semantics, so LLVM will not
/// lane it any more than it lanes the sum. It is a full pass over the same
/// 50 MB-per-layer score tensor that `exp_sub_sum_inplace` then walks again.
///
/// Returns `f32::NEG_INFINITY` for an empty slice, matching a fold from that
/// identity. Lane-splitting a MAX is exact — `max` is associative and
/// commutative on non-NaN floats — so unlike the sum this twin is gated by
/// `assert_eq!`, not by tolerance.
#[must_use]
pub fn max_f32(xs: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: probed immediately above.
            return unsafe { max_f32_avx2(xs) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `neon` is a compile-time guarantee on this target.
        return unsafe { max_f32_neon(xs) };
    }
    #[cfg(target_arch = "wasm32")]
    {
        // SAFETY: `simd128` is a compile-time guarantee on this target.
        return unsafe { max_f32_simd128(xs) };
    }
    #[allow(unreachable_code)]
    max_f32_scalar(xs)
}

/// The oracle.
#[must_use]
pub fn max_f32_scalar(xs: &[f32]) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for &v in xs {
        m = m.max(v);
    }
    m
}

/// # Safety
/// Caller must have verified `avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_f32_avx2(xs: &[f32]) -> f32 {
    // SAFETY: whole-body wrapper inserted by the edition-2024
    // `unsafe_op_in_unsafe_fn` migration. This block adds no new obligation:
    // the contract is stated on the `unsafe fn` signature above.
    unsafe {
        use std::arch::x86_64::{_mm256_loadu_ps, _mm256_max_ps, _mm256_storeu_ps};
        let n = xs.len();
        if n < 8 {
            return max_f32_scalar(xs);
        }
        let mut acc = _mm256_loadu_ps(xs.as_ptr());
        let mut i = 8;
        while i + 8 <= n {
            acc = _mm256_max_ps(acc, _mm256_loadu_ps(xs.as_ptr().add(i)));
            i += 8;
        }
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut m = lanes[0];
        for &v in &lanes[1..] {
            m = m.max(v);
        }
        while i < n {
            m = m.max(xs[i]);
            i += 1;
        }
        m
    }
}

/// NEON, four lanes. Exact, like the other max twins.
///
/// # Safety
/// `neon` is a compile-time guarantee on `aarch64` here.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn max_f32_neon(xs: &[f32]) -> f32 {
    // SAFETY: whole-body wrapper inserted by the edition-2024
    // `unsafe_op_in_unsafe_fn` migration. This block adds no new obligation:
    // the contract is stated on the `unsafe fn` signature above.
    unsafe {
        use core::arch::aarch64::{vld1q_f32, vmaxq_f32, vmaxvq_f32};
        let n = xs.len();
        if n < 4 {
            return max_f32_scalar(xs);
        }
        let mut acc = vld1q_f32(xs.as_ptr());
        let mut i = 4;
        while i + 4 <= n {
            acc = vmaxq_f32(acc, vld1q_f32(xs.as_ptr().add(i)));
            i += 4;
        }
        let mut m = vmaxvq_f32(acc);
        while i < n {
            m = m.max(xs[i]);
            i += 1;
        }
        m
    }
}

/// # Safety
/// `simd128` is a compile-time guarantee on `wasm32` here.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn max_f32_simd128(xs: &[f32]) -> f32 {
    // SAFETY: whole-body wrapper inserted by the edition-2024
    // `unsafe_op_in_unsafe_fn` migration. This block adds no new obligation:
    // the contract is stated on the `unsafe fn` signature above.
    unsafe {
        use core::arch::wasm32::{f32x4_extract_lane, f32x4_pmax, v128_load};
        let n = xs.len();
        if n < 4 {
            return max_f32_scalar(xs);
        }
        // `f32x4_pmax` is the raw max: it returns the second operand when either is
        // NaN, which matches nothing in particular — but these are attention
        // scores, never NaN, and the scalar oracle gate covers the range we feed.
        let mut acc = v128_load(xs.as_ptr().cast());
        let mut i = 4;
        while i + 4 <= n {
            acc = f32x4_pmax(acc, v128_load(xs.as_ptr().add(i).cast()));
            i += 4;
        }
        let mut m = f32x4_extract_lane::<0>(acc);
        m = m.max(f32x4_extract_lane::<1>(acc));
        m = m.max(f32x4_extract_lane::<2>(acc));
        m = m.max(f32x4_extract_lane::<3>(acc));
        while i < n {
            m = m.max(xs[i]);
            i += 1;
        }
        m
    }
}

/// `xs[i] = gelu_tanh(xs[i])`, vectorised.
///
/// The activation between a transformer MLP's two projections, so it runs over
/// `seq * 4 * hidden` elements per layer. Argus's tower had an AVX2 twin for it
/// and **no wasm twin at all**, so the browser took a scalar loop — the same
/// asymmetry the softmax had.
///
/// `gelu_tanh` is branch-free (`x / (1 + exp(-2z))`), so unlike `tanh` there is
/// no small-|x| special case to reproduce; the twins are the same expression in
/// lanes. Gated by tolerance against the scalar oracle.
pub fn gelu_tanh_inplace(xs: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: probed immediately above.
            unsafe { gelu_tanh_avx2(xs) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `neon` is a compile-time guarantee on this target.
        unsafe { gelu_tanh_neon(xs) };
        return;
    }
    #[cfg(target_arch = "wasm32")]
    {
        // SAFETY: `simd128` is a compile-time guarantee on this target.
        unsafe { gelu_tanh_simd128(xs) };
        return;
    }
    #[allow(unreachable_code)]
    for v in xs.iter_mut() {
        *v = gelu_tanh(*v);
    }
}

#[allow(clippy::excessive_precision)]
// `0.797_884_56` is `SQRT_2_OVER_PI` from the scalar oracle above, verbatim.
// Truncating it is bit-identical (0x2a424c3f either way), but it would make the
// twin stop matching the oracle ON THE PAGE, and reading them side by side is
// how the twins are checked.
/// # Safety
/// Caller must have verified `avx2` and `fma`.
#[allow(clippy::many_single_char_names)]
// The names are the polynomial's own: `n`, `i`, `x`, `t`, `r`, `f`, `p`, `e`,
// `k`, `z`. Expanding them into prose makes the kernel harder to read against
// its twins and the scalar oracle, which is the only way this code is verified.
#[allow(clippy::wildcard_imports)]
// A SIMD kernel names 15 intrinsics; importing them one by one is a list that
// goes stale the moment the arithmetic changes, and hides nothing.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_tanh_avx2(xs: &mut [f32]) {
    // SAFETY: whole-body wrapper inserted by the edition-2024
    // `unsafe_op_in_unsafe_fn` migration. This block adds no new obligation:
    // the contract is stated on the `unsafe fn` signature above.
    unsafe {
        use std::arch::x86_64::*;
        const L1: f32 = std::f32::consts::LN_2;
        const L2: f32 = L1 * L1 / 2.0;
        const L3: f32 = L1 * L1 * L1 / 6.0;
        const L4: f32 = L1 * L1 * L1 * L1 / 24.0;
        const L5: f32 = L1 * L1 * L1 * L1 * L1 / 120.0;
        let sq2pi = _mm256_set1_ps(0.797_884_56);
        let k = _mm256_set1_ps(0.044_715);
        let one = _mm256_set1_ps(1.0);
        let m2 = _mm256_set1_ps(-2.0);
        let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let magic = _mm256_set1_ps(MAGIC);
        let lo = _mm256_set1_ps(-125.0);
        let hi = _mm256_set1_ps(125.0);
        let (c1, c2, c3, c4, c5) = (
            _mm256_set1_ps(L1),
            _mm256_set1_ps(L2),
            _mm256_set1_ps(L3),
            _mm256_set1_ps(L4),
            _mm256_set1_ps(L5),
        );
        let bias = _mm256_set1_epi32(127);
        let n = xs.len();
        let mut i = 0;
        while i + 8 <= n {
            let x = _mm256_loadu_ps(xs.as_ptr().add(i));
            // z = sqrt(2/pi) * x * (1 + k x^2)
            let x2 = _mm256_mul_ps(x, x);
            let z = _mm256_mul_ps(_mm256_mul_ps(sq2pi, x), _mm256_fmadd_ps(k, x2, one));
            // exp(-2z)
            let t = _mm256_mul_ps(_mm256_mul_ps(m2, z), log2e);
            let t = _mm256_min_ps(_mm256_max_ps(t, lo), hi);
            let r = _mm256_sub_ps(_mm256_add_ps(t, magic), magic);
            let f = _mm256_sub_ps(t, r);
            let p = _mm256_fmadd_ps(f, c5, c4);
            let p = _mm256_fmadd_ps(f, p, c3);
            let p = _mm256_fmadd_ps(f, p, c2);
            let p = _mm256_fmadd_ps(f, p, c1);
            let p = _mm256_fmadd_ps(f, p, one);
            let e = _mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_cvtps_epi32(r), bias));
            let ex = _mm256_mul_ps(p, _mm256_castsi256_ps(e));
            _mm256_storeu_ps(
                xs.as_mut_ptr().add(i),
                _mm256_div_ps(x, _mm256_add_ps(one, ex)),
            );
            i += 8;
        }
        while i < n {
            xs[i] = gelu_tanh(xs[i]);
            i += 1;
        }
    }
}

#[allow(clippy::excessive_precision)]
// `0.797_884_56` is `SQRT_2_OVER_PI` from the scalar oracle above, verbatim.
// Truncating it is bit-identical (0x2a424c3f either way), but it would make the
// twin stop matching the oracle ON THE PAGE, and reading them side by side is
// how the twins are checked.
/// NEON, four lanes, FMA-fused.
///
/// # Safety
/// `neon` is a compile-time guarantee on `aarch64` here.
#[allow(clippy::many_single_char_names)]
// The names are the polynomial's own: `n`, `i`, `x`, `t`, `r`, `f`, `p`, `e`,
// `k`, `z`. Expanding them into prose makes the kernel harder to read against
// its twins and the scalar oracle, which is the only way this code is verified.
#[allow(clippy::wildcard_imports)]
// A SIMD kernel names 15 intrinsics; importing them one by one is a list that
// goes stale the moment the arithmetic changes, and hides nothing.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gelu_tanh_neon(xs: &mut [f32]) {
    // SAFETY: whole-body wrapper inserted by the edition-2024
    // `unsafe_op_in_unsafe_fn` migration. This block adds no new obligation:
    // the contract is stated on the `unsafe fn` signature above.
    unsafe {
        use core::arch::aarch64::*;
        const L1: f32 = core::f32::consts::LN_2;
        const L2: f32 = L1 * L1 / 2.0;
        const L3: f32 = L1 * L1 * L1 / 6.0;
        const L4: f32 = L1 * L1 * L1 * L1 / 24.0;
        const L5: f32 = L1 * L1 * L1 * L1 * L1 / 120.0;
        let sq2pi = vdupq_n_f32(0.797_884_56);
        let k = vdupq_n_f32(0.044_715);
        let one = vdupq_n_f32(1.0);
        let m2 = vdupq_n_f32(-2.0);
        let log2e = vdupq_n_f32(core::f32::consts::LOG2_E);
        let magic = vdupq_n_f32(MAGIC);
        let lo = vdupq_n_f32(-125.0);
        let hi = vdupq_n_f32(125.0);
        let (c1, c2, c3, c4, c5) = (
            vdupq_n_f32(L1),
            vdupq_n_f32(L2),
            vdupq_n_f32(L3),
            vdupq_n_f32(L4),
            vdupq_n_f32(L5),
        );
        let bias = vdupq_n_s32(127);
        let n = xs.len();
        let mut i = 0;
        while i + 4 <= n {
            let x = vld1q_f32(xs.as_ptr().add(i));
            let x2 = vmulq_f32(x, x);
            let z = vmulq_f32(vmulq_f32(sq2pi, x), vfmaq_f32(one, k, x2));
            let t = vmulq_f32(vmulq_f32(m2, z), log2e);
            let t = vminq_f32(vmaxq_f32(t, lo), hi);
            let r = vsubq_f32(vaddq_f32(t, magic), magic);
            let f = vsubq_f32(t, r);
            let p = vfmaq_f32(c4, f, c5);
            let p = vfmaq_f32(c3, f, p);
            let p = vfmaq_f32(c2, f, p);
            let p = vfmaq_f32(c1, f, p);
            let p = vfmaq_f32(one, f, p);
            let e = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(r), bias));
            let ex = vmulq_f32(p, vreinterpretq_f32_s32(e));
            vst1q_f32(xs.as_mut_ptr().add(i), vdivq_f32(x, vaddq_f32(one, ex)));
            i += 4;
        }
        while i < n {
            xs[i] = gelu_tanh(xs[i]);
            i += 1;
        }
    }
}

#[allow(clippy::excessive_precision)]
// `0.797_884_56` is `SQRT_2_OVER_PI` from the scalar oracle above, verbatim.
// Truncating it is bit-identical (0x2a424c3f either way), but it would make the
// twin stop matching the oracle ON THE PAGE, and reading them side by side is
// how the twins are checked.
/// # Safety
/// `simd128` is a compile-time guarantee on `wasm32` here.
#[allow(clippy::many_single_char_names)]
// The names are the polynomial's own: `n`, `i`, `x`, `t`, `r`, `f`, `p`, `e`,
// `k`, `z`. Expanding them into prose makes the kernel harder to read against
// its twins and the scalar oracle, which is the only way this code is verified.
#[allow(clippy::wildcard_imports)]
// A SIMD kernel names 15 intrinsics; importing them one by one is a list that
// goes stale the moment the arithmetic changes, and hides nothing.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn gelu_tanh_simd128(xs: &mut [f32]) {
    // SAFETY: whole-body wrapper inserted by the edition-2024
    // `unsafe_op_in_unsafe_fn` migration. This block adds no new obligation:
    // the contract is stated on the `unsafe fn` signature above.
    unsafe {
        use core::arch::wasm32::*;
        const L1: f32 = core::f32::consts::LN_2;
        const L2: f32 = L1 * L1 / 2.0;
        const L3: f32 = L1 * L1 * L1 / 6.0;
        const L4: f32 = L1 * L1 * L1 * L1 / 24.0;
        const L5: f32 = L1 * L1 * L1 * L1 * L1 / 120.0;
        let sq2pi = f32x4_splat(0.797_884_56);
        let k = f32x4_splat(0.044_715);
        let one = f32x4_splat(1.0);
        let m2 = f32x4_splat(-2.0);
        let log2e = f32x4_splat(core::f32::consts::LOG2_E);
        let magic = f32x4_splat(MAGIC);
        let lo = f32x4_splat(-125.0);
        let hi = f32x4_splat(125.0);
        let (c1, c2, c3, c4, c5) = (
            f32x4_splat(L1),
            f32x4_splat(L2),
            f32x4_splat(L3),
            f32x4_splat(L4),
            f32x4_splat(L5),
        );
        let bias = i32x4_splat(127);
        let n = xs.len();
        let mut i = 0;
        while i + 4 <= n {
            let x = v128_load(xs.as_ptr().add(i).cast());
            let x2 = f32x4_mul(x, x);
            let z = f32x4_mul(f32x4_mul(sq2pi, x), f32x4_add(f32x4_mul(k, x2), one));
            let t = f32x4_mul(f32x4_mul(m2, z), log2e);
            let t = f32x4_pmin(hi, f32x4_pmax(lo, t));
            let r = f32x4_sub(f32x4_add(t, magic), magic);
            let f = f32x4_sub(t, r);
            let p = f32x4_add(f32x4_mul(f, c5), c4);
            let p = f32x4_add(f32x4_mul(f, p), c3);
            let p = f32x4_add(f32x4_mul(f, p), c2);
            let p = f32x4_add(f32x4_mul(f, p), c1);
            let p = f32x4_add(f32x4_mul(f, p), one);
            let e = i32x4_shl(i32x4_add(i32x4_trunc_sat_f32x4(r), bias), 23);
            let ex = f32x4_mul(p, e);
            v128_store(
                xs.as_mut_ptr().add(i).cast(),
                f32x4_div(x, f32x4_add(one, ex)),
            );
            i += 4;
        }
        while i < n {
            xs[i] = gelu_tanh(xs[i]);
            i += 1;
        }
    }
}
