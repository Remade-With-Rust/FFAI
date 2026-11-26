//! A coarse stage profiler for prefill and decode: atomic nanosecond buckets, on only with
//! `MINERVA_PROF=1` (read once). Scopes are per projection and per layer phase (tens per layer a
//! batch, never per element), so the timer's own cost is negligible; when off, a scope is one
//! relaxed load.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

/// The stages timed.
#[derive(Clone, Copy, Debug)]
pub enum Stage {
    /// The whole batch.
    Total = 0,
    /// The ternary GEMMs.
    Gemm = 1,
    /// The `[rows, tokens]` transpose and `finish` after each GEMM.
    Finish = 2,
    /// The attention region (scores, softmax, value sums).
    Attention = 3,
    /// Decode: every step that ends in a next token (forward pass, final norm and head).
    DecTotal = 4,
    /// Decode: the ternary mat-vecs (each join of projections timed as one).
    DecTernary = 5,
    /// Decode: the attention region.
    DecAttention = 6,
    /// Decode: the final norm and the head (screen, or the full head).
    DecHead = 7,
}

/// Stage names, in [`Stage`] order.
pub const NAMES: [&str; 8] = [
    "total",
    "gemm",
    "finish",
    "attention",
    "decode",
    "ternary",
    "attention",
    "head",
];
static NS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MINERVA_PROF").is_some())
}

/// A running scope; adds its time to its stage when dropped.
pub struct Scope(Option<(Stage, Instant)>);

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some((s, t)) = self.0 {
            NS[s as usize].fetch_add(
                u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX),
                Relaxed,
            );
        }
    }
}

/// Start timing `stage` (a no-op unless `MINERVA_PROF` is set).
#[must_use]
pub fn scope(stage: Stage) -> Scope {
    Scope(on().then(|| (stage, Instant::now())))
}

/// `(stage, seconds)` so far, when profiling is on.
#[must_use]
pub fn report() -> Option<Vec<(&'static str, f64)>> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "nanoseconds to seconds, for a report"
    )]
    on().then(|| {
        NAMES
            .iter()
            .zip(&NS)
            .map(|(n, v)| (*n, v.load(Relaxed) as f64 / 1e9))
            .collect()
    })
}
