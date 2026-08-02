//! Prometheus telemetry hooks — Stage 1 (harvest).
//!
//! # Why Diana is a Prometheus target
//!
//! Prometheus is the private refinery for `remade_ffmpeg_rs`: harvest real
//! telemetry, discover a closed form with symbolic regression, simplify it to
//! minimal-operation form with E-graphs, PROVE its error bound with SMT, and
//! forge it back as generated Rust behind the usual gates. Its first keeper
//! was MP3's `x^(3/4)` — "the hot transcendental of the quantize stage", found
//! at 62.7 % of encode and strength-reduced bit-exactly.
//!
//! Diana has the same shape of problem in `crate::silu::exp_fast`:
//!
//! * it is **30.7 % of serial detect**, the largest single line item, after
//!   three rounds of hand optimisation;
//! * it is now **COMPUTE-bound, not memory-bound** — measured at 1.12-1.61
//!   Gelem/s against a ~2.4 Gelem/s ceiling implied by its own op count, so
//!   the only remaining lever is FEWER OPERATIONS;
//! * fewer operations means a shorter polynomial, which is an ACCURACY trade
//!   — and the oracle here is tight enough that one FMA fusion breached it by
//!   2 %. Guessing a shorter polynomial is not safe; a proved error bound is.
//!
//! That last point is why the refinery earns its place rather than a
//! hand-fitted minimax polynomial: `prom-prove` bounds the error over the
//! range the harvest shows activations ACTUALLY occupy, instead of over the
//! `[-125, 125]` the current clamp defends.
//!
//! # The contract, from `Prometheus/docs/telemetry-hooks.md`
//!
//! * **Off by default** — never in `default = [...]`.
//! * **Zero cost when off** — the hooks compile to nothing.
//! * **Additive only** — a hook may observe; it may never change a decision.
//!   An instrumented engine and a stock engine must produce identical output,
//!   which the oracle and determinism tests already enforce.

/// Record one activation input. Compiled away entirely without the feature.
#[cfg(not(feature = "prometheus-telemetry"))]
#[inline(always)]
pub fn observe_silu_input(_x: f32) {}

/// Reset the histogram. No-op without the feature.
#[cfg(not(feature = "prometheus-telemetry"))]
#[inline(always)]
pub fn reset() {}

/// Dump the harvested distribution. `None` without the feature.
#[cfg(not(feature = "prometheus-telemetry"))]
#[inline(always)]
pub fn dump() -> Option<String> {
    None
}

#[cfg(feature = "prometheus-telemetry")]
mod on {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Histogram bounds. Wider than any plausible activation so the tails
    /// are COUNTED rather than clipped — the whole question this harvest
    /// answers is how much of `[-125, 125]` is actually used, and a
    /// histogram that silently clamps would answer it wrongly.
    const LO: f32 = -40.0;
    const HI: f32 = 40.0;
    const BINS: usize = 512;

    static HIST: [AtomicU64; BINS] = {
        #[allow(clippy::declare_interior_mutable_const)]
        const Z: AtomicU64 = AtomicU64::new(0);
        [Z; BINS]
    };
    static BELOW: AtomicU64 = AtomicU64::new(0);
    static ABOVE: AtomicU64 = AtomicU64::new(0);
    static COUNT: AtomicU64 = AtomicU64::new(0);

    /// Observe one input. Relaxed ordering: this is a histogram, not a
    /// synchronisation primitive, and exact inter-thread ordering would cost
    /// more than the information is worth.
    #[inline]
    pub fn observe_silu_input(x: f32) {
        COUNT.fetch_add(1, Ordering::Relaxed);
        if x < LO {
            BELOW.fetch_add(1, Ordering::Relaxed);
        } else if x >= HI {
            ABOVE.fetch_add(1, Ordering::Relaxed);
        } else {
            let b = (((x - LO) / (HI - LO)) * BINS as f32) as usize;
            HIST[b.min(BINS - 1)].fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn reset() {
        for h in &HIST {
            h.store(0, Ordering::Relaxed);
        }
        BELOW.store(0, Ordering::Relaxed);
        ABOVE.store(0, Ordering::Relaxed);
        COUNT.store(0, Ordering::Relaxed);
    }

    /// CSV: `bin_lo,bin_hi,count` plus the out-of-range tallies, which is the
    /// `Dataset` shape `prom-distill` consumes.
    pub fn dump() -> Option<String> {
        let total = COUNT.load(Ordering::Relaxed);
        if total == 0 {
            return None;
        }
        let mut s = String::from("# silu input distribution harvested from real images\n");
        s.push_str(&format!("# total={total} below_{LO}={} above_{HI}={}\n",
                            BELOW.load(Ordering::Relaxed), ABOVE.load(Ordering::Relaxed)));
        s.push_str("bin_lo,bin_hi,count\n");
        let w = (HI - LO) / BINS as f32;
        for (i, h) in HIST.iter().enumerate() {
            let c = h.load(Ordering::Relaxed);
            if c > 0 {
                s.push_str(&format!("{:.4},{:.4},{c}\n", LO + i as f32 * w, LO + (i + 1) as f32 * w));
            }
        }
        Some(s)
    }
}

#[cfg(feature = "prometheus-telemetry")]
pub use on::{dump, observe_silu_input, reset};
