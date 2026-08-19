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
//! | 3 | `silu` skips rayon when it has a single chunk | 15.2 % | ~1.08x | **~1.1 %** |
//!
//! | 4 | `silu`'s AVX2 path stops writing its output twice | 15.2 % | ~1.15x | **~2.0 %** |
//!
//! Predicted stack total: **~3.7 %** — under this box's ~5 % resolution, so
//! nothing is claimed yet.
//!
//! Entry 4 is a redundancy, not a trade. The AVX2 branch did
//! `v.resize(len, 0.0)` and then had the kernel overwrite every element —
//! **10.38 M elements per image, 41.5 MiB written twice** — while the comment
//! directly above it recorded removing exactly that double write from the
//! neighbouring `collect` path. It was fixed there and never here.
//!
//! Writing into uninitialised capacity removes one of the two writes. The
//! output is unchanged bit for bit, because the zeros were never read: every
//! element is written by the kernel before anything observes it.
//!
//! No speed is claimed. A closely related change — removing the im2col
//! buffer's zero-fill, 216.9 MiB and a much larger arithmetic prize —
//! measured **z = -0.22, exactly chance**, because the pipeline's zeroed
//! allocations largely come from fresh OS pages that arrive zero for free.
//! This one may well do the same. It is taken because a buffer written twice
//! is a defect at any measured size, and it is registered here so the batch
//! carries the verdict.
//!
//! Entry 3 came out of a thread-scaling sweep and is the clearest case yet
//! for measuring the stack rather than the brick. `silu` is called ~1305
//! times per image, and most of those tensors are smaller than one rayon
//! chunk — so `par_chunks_mut` forked a job, handed the whole buffer to a
//! single worker, and joined, ~1305 times, to arrive back on the calling
//! thread. It is the one bucket in the profile that got **SLOWER with more
//! cores**: 0.292 s at one thread against 0.315 s at twenty-four. A stage
//! that costs more the more cores you give it is paying for parallelism it
//! never receives, and the arithmetic for removing that is 0.023 s of a
//! 2.065 s pipeline — about 1.1 %, which this box resolves at 108 %.
//!
//! It was taken anyway, for the same reason as entry 2: negative scaling is
//! a defect at any magnitude, and the fix costs one branch.
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
            CACHED.store(u8::from(off), Ordering::Relaxed);
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
