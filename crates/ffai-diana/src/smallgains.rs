//! Gains too small to measure alone — and the switch that measures them together.
//!
//! # Why this module exists
//!
//! The measurement discipline in this repo says: prune on arithmetic before
//! building, and if the expected pipeline gain lands under the noise floor,
//! the experiment cannot succeed. That rule is right, and applied without
//! care it quietly forbids an entire class of real improvement — because
//! **1 % from fifty functions is 1.005^50 = 1.28x, and not one of those
//! fifty is individually resolvable.** Refusing every gain that cannot be
//! seen in isolation is how a codebase stays behind forever.
//!
//! The resolution is not to relax the gate. It is to change what gets
//! gated: measure the STACK, not the brick.
//!
//! # The rule
//!
//! A change qualifies as a small gain when ALL of:
//!
//! 1. it is **correct on its own terms** — it does not trade accuracy for
//!    the speed, and the oracle still passes bit-for-bit or within its band;
//! 2. its expected pipeline gain is computed as `stage share x speedup` and
//!    **written down**, so the stack's predicted total can be checked
//!    against the stack's measured total;
//! 3. it is reachable from [`disabled`], so the whole accumulated set can be
//!    turned off in one process and A/B'd as a group.
//!
//! Point 3 is the load-bearing one. Each gain lands with its arithmetic and
//! **no speed claim**. When the predicted total clears the harness's
//! resolution — this box needs roughly 5 % — the stack is measured against
//! `FFAI_DIANA_NO_SMALLGAINS=1` with the usual paired ABBA, and THAT number
//! is the one that gets published. If the measured total falls short of the
//! predicted total, the arithmetic was wrong somewhere and the register says
//! where to look.
//!
//! # The register
//!
//! | # | change | stage share | speedup | predicted |
//! |---|---|---:|---:|---:|
//! | 1 | top-k decodes `max_detections`, not always 300 | 2.1 % | ~1.3x | **~0.5 %** |
//! | 2 | blocked transpose instead of candle's generic one | 0.3 % | ~1.4x | **~0.1 %** |
//!
//! Predicted stack total: **~0.6 %** — under this box's ~5 % resolution, so
//! nothing is claimed yet.
//!
//! Entry 2 is the register's reason for existing, stated plainly. Candle's
//! `t().contiguous()` measured **4.3x slower than a blocked loop** on large
//! shapes; at attention's smaller ones it is ~1.4x, worth ~0.1 % of the
//! pipeline. Nothing about that is resolvable on this box, and it was taken
//! anyway on principle: being several times slower than a plain loop at
//! anything is a defect whether or not it currently sits on a hot path. It
//! is also BYTE-IDENTICAL — a transpose permutes floats rather than
//! computing with them — so there is no accuracy question to trade against.
//!
//! Entry 1 is worth reading as the template: it was justified by PARITY
//! first (Ultralytics feeds `max_det` into both top-k stages, so decoding
//! 300 and trimming was a different algorithm from the reference), and the
//! speed was the bonus. A small gain that also fixes a correctness mismatch
//! is the easy case; the register exists for the ones that are only speed.

/// True when the accumulated small gains should be turned OFF, restoring the
/// pre-optimization behaviour for a stack-level A/B.
///
/// One switch for the whole set on purpose: fifty individual toggles would
/// each be unmeasurable, which is the problem this module exists to solve.
#[inline]
pub fn disabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(u8::MAX);
    match CACHED.load(Ordering::Relaxed) {
        u8::MAX => {
            let off = std::env::var("FFAI_DIANA_NO_SMALLGAINS").is_ok_and(|v| v == "1");
            CACHED.store(off as u8, Ordering::Relaxed);
            off
        }
        v => v == 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_gains_are_on_by_default() {
        // The env var is absent in a normal test run, so the accumulated
        // gains must be active — a stack that silently defaults to OFF would
        // make every downstream measurement wrong in the same direction.
        assert!(!disabled());
    }
}
