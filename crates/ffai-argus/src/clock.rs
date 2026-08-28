//! `Instant` on native, a zero clock on wasm.
//!
//! `std::time::Instant::now()` **panics** on `wasm32-unknown-unknown`: the
//! target has no clock behind it. Argus calls it 51 times across `decode`,
//! `engine`, `siglip` and `text`, and they sit inside the forward pass rather
//! than behind a profiling flag — so in a browser they are a crash on the
//! first tile, not a slow path.
//!
//! This is a drop-in for `std::time::Instant`: same `now()` / `elapsed()`
//! surface, so a call site changes its IMPORT and nothing else. On native it
//! is a newtype that inlines away and behaviour is byte-identical; on wasm
//! `elapsed()` is always `Duration::ZERO`.
//!
//! A `cfg` rather than a feature: this is a property of the target, not a
//! choice a consumer should be able to get wrong.
//!
//! **Zero is a safe reading for every consumer here, and that was checked
//! rather than assumed.** Every Argus site feeds a REPORT — per-stage and
//! per-token millisecond tables printed behind `FFAI_ARGUS_PROFILE` — so on
//! wasm the tables read all-zero rather than lying about a number the target
//! cannot take. No control flow branches on these durations.

use core::time::Duration;

/// A monotonic instant, or nothing at all on a target without a clock.
#[derive(Clone, Copy, Debug)]
pub struct Instant {
    #[cfg(not(target_arch = "wasm32"))]
    t0: std::time::Instant,
}

impl Instant {
    #[must_use]
    #[inline]
    pub fn now() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            t0: std::time::Instant::now(),
        }
    }

    /// Time since [`Self::now`] — always `ZERO` on wasm.
    #[must_use]
    #[inline]
    pub fn elapsed(&self) -> Duration {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.t0.elapsed()
        }
        #[cfg(target_arch = "wasm32")]
        {
            Duration::ZERO
        }
    }
}
