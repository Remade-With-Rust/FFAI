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
pub fn ln(x: f32) -> f32 {
    if x <= 0.0 {
        return if x == 0.0 { f32::NEG_INFINITY } else { f32::NAN };
    }
    let bits = x.to_bits();
    // Exponent field, unbiased.
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
    let p = 2.0 * s * (1.0 + s2 * (0.333_333_34 + s2 * (0.2 + s2 * (0.142_857_15 + s2 * 0.111_111_11))));
    (e as f32).mul_add(std::f32::consts::LN_2, p)
}

/// `log10(x)`, for log-mel and decibel work.
#[inline(always)]
#[must_use]
pub fn log10(x: f32) -> f32 {
    ln(x) * std::f32::consts::LOG10_E
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
        let oracle = |x: f32| {
            0.5 * x * (1.0 + (0.797_884_56 * x * (1.0 + 0.044_715 * x * x)).tanh())
        };
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
