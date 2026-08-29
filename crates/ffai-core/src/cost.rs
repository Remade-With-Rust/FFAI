//! A **deterministic** cost model — counters, not a stopwatch.
//!
//! Lives in `ffai-core` so every engine shares one vocabulary of work. It was
//! written for `ffai-argus`'s vision tower and moved here unchanged, because
//! the thing it measures — how much work of which KIND — is not specific to a
//! vision tower, and a second copy would drift the way three copies of
//! `exp` did (see [`crate::fastmath`]).
//!
//! # Why counters and not a stopwatch
//!
//! Wall-clock on this box swings ±12 % run to run, which is larger than most
//! of the wins worth having. Round 2 opened by chasing exactly that: a change
//! that must be faster (less work per tile, same threading) measured *slower*,
//! and three repeats disagreed with each other.
//!
//! A number that changes when nothing changed cannot decide anything. So the
//! unit of account here is not milliseconds but **work**:
//!
//! | counter | what it is | why it is the honest unit |
//! |---|---|---|
//! | `matmul_flops` | `2*m*k*n` summed | the arithmetic that must happen |
//! | `matmul_calls` | GEMM invocations | per-call setup and weight packing |
//! | `elem_ops` | elementwise element-visits | the memory-bound half |
//! | `transcendental` | `exp`/`tanh` calls | ~40 ns each, scalar, no SIMD |
//! | `bytes_moved` | read + written | what actually binds a 50 MB tensor |
//! | `copies` / `copy_bytes` | layout copies | `contiguous`, `to_vec`, `cat` |
//!
//! Every one is **exactly reproducible**: same input, same counts, on any
//! machine, under any load. A win is a counter that went down, and a
//! regression is one that went up — neither needs a quiet box, an ABBA
//! interleave, or a z-score.
//!
//! This does not replace timing; it replaces timing *as the decision
//! procedure*. Wall-clock still says whether a counter reduction was worth
//! having, but it is no longer what tells us whether the change did anything.
//!
//! # The rule these counters encode
//!
//! Not all work costs the same, and the ratios are stable enough to reason
//! with. On this box, measured once: a matmul FLOP retires at ~660 GF/s, an
//! elementwise element-visit at ~10 GB/s (2.5 G elem/s), and a scalar `tanhf`
//! at ~80 M/s. So **one transcendental costs roughly 30 elementwise visits and
//! ~8000 FLOPs of matmul time.** That is why replacing `tanhf` with `expf` was
//! the largest single win of round 1 and why adding threads to a 3 MB
//! `LayerNorm` was not.

use std::sync::atomic::{AtomicU64, Ordering};

/// One counter set. Global and atomic: the tower runs on several threads, and
/// a per-thread count would have to be summed by hand at exactly the moment
/// the answer is wanted.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Costs {
    pub matmul_flops: u64,
    pub matmul_calls: u64,
    pub elem_ops: u64,
    pub elem_calls: u64,
    /// Scalar libm calls (`tanhf`, `expf` in a plain loop): ~75 M/s here.
    pub transcendental: u64,
    /// Vectorised transcendentals inside a candle kernel: ~2.7 G/s here — 36x
    /// cheaper each, so they are a different unit of work entirely.
    pub transcendental_vec: u64,
    pub bytes_moved: u64,
    pub copies: u64,
    pub copy_bytes: u64,
}

macro_rules! counters {
    ($($name:ident),* $(,)?) => {
        $(static $name: AtomicU64 = AtomicU64::new(0);)*
    };
}
counters!(
    MATMUL_FLOPS,
    MATMUL_CALLS,
    ELEM_OPS,
    ELEM_CALLS,
    TRANSCENDENTAL,
    TRANSCENDENTAL_VEC,
    BYTES_MOVED,
    COPIES,
    COPY_BYTES
);

/// Counting is off unless a probe turns it on, so the shipping path pays
/// nothing but one relaxed load per instrumented site.
static ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Enable counting and zero the counters. Returns the previous state.
pub fn start() -> bool {
    for c in [
        &MATMUL_FLOPS,
        &MATMUL_CALLS,
        &ELEM_OPS,
        &ELEM_CALLS,
        &TRANSCENDENTAL,
        &TRANSCENDENTAL_VEC,
        &BYTES_MOVED,
        &COPIES,
        &COPY_BYTES,
    ] {
        c.store(0, Ordering::Relaxed);
    }
    ON.swap(true, Ordering::Relaxed)
}

/// Stop counting and read the totals.
pub fn stop() -> Costs {
    ON.store(false, Ordering::Relaxed);
    Costs {
        matmul_flops: MATMUL_FLOPS.load(Ordering::Relaxed),
        matmul_calls: MATMUL_CALLS.load(Ordering::Relaxed),
        elem_ops: ELEM_OPS.load(Ordering::Relaxed),
        elem_calls: ELEM_CALLS.load(Ordering::Relaxed),
        transcendental: TRANSCENDENTAL.load(Ordering::Relaxed),
        transcendental_vec: TRANSCENDENTAL_VEC.load(Ordering::Relaxed),
        bytes_moved: BYTES_MOVED.load(Ordering::Relaxed),
        copies: COPIES.load(Ordering::Relaxed),
        copy_bytes: COPY_BYTES.load(Ordering::Relaxed),
    }
}

#[inline]
fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Record a matmul of `(m,k) x (k,n)`, `batch` times.
pub fn matmul(batch: u64, m: u64, k: u64, n: u64) {
    if !on() {
        return;
    }
    MATMUL_CALLS.fetch_add(batch, Ordering::Relaxed);
    MATMUL_FLOPS.fetch_add(2 * batch * m * k * n, Ordering::Relaxed);
    // Read both operands, write the result. The output is what makes q·kᵀ
    // expensive: 64 wide in, 1024 wide out.
    BYTES_MOVED.fetch_add(4 * batch * (m * k + k * n + m * n), Ordering::Relaxed);
}

/// Record an elementwise pass over `n` elements: `reads` in, `writes` out.
pub fn elementwise(n: u64, reads: u64, writes: u64) {
    if !on() {
        return;
    }
    ELEM_CALLS.fetch_add(1, Ordering::Relaxed);
    ELEM_OPS.fetch_add(n, Ordering::Relaxed);
    BYTES_MOVED.fetch_add(4 * n * (reads + writes), Ordering::Relaxed);
}

/// Record `n` VECTORISED transcendental calls — inside a candle kernel.
pub fn transcendental_vector(n: u64) {
    if !on() {
        return;
    }
    TRANSCENDENTAL_VEC.fetch_add(n, Ordering::Relaxed);
}

/// Record `n` scalar transcendental calls (`exp`, `tanh`, …).
///
/// Counted apart from `elem_ops` because they are not the same unit of work:
/// one is ~30x the other on this box, so folding them together would let a
/// change that trades 3 M `tanhf` for 3 M `expf` look like it did nothing.
pub fn transcendental_scalar(n: u64) {
    if !on() {
        return;
    }
    TRANSCENDENTAL.fetch_add(n, Ordering::Relaxed);
}

/// Record a layout copy of `n` elements — `contiguous`, `cat`, a `Vec` round
/// trip. Pure overhead: no arithmetic, and it is the cost `to_vec1` +
/// `from_vec` hides.
pub fn copy(n: u64) {
    if !on() {
        return;
    }
    COPIES.fetch_add(1, Ordering::Relaxed);
    COPY_BYTES.fetch_add(4 * n * 2, Ordering::Relaxed);
    BYTES_MOVED.fetch_add(4 * n * 2, Ordering::Relaxed);
}

impl Costs {
    /// A weighted total, in units of "matmul-FLOP equivalents".
    ///
    /// The weights come from one measurement of this box (see the module
    /// docs): 660 GF/s matmul, 2.5 G elementwise visits/s, 80 M
    /// transcendental/s. They are a MODEL, not a timing — their job is to stop
    /// a change that trades 12 M elementwise visits for 3 M transcendentals
    /// from reading as an improvement because one counter fell.
    ///
    /// Deterministic like everything else here: same input, same number.
    #[must_use]
    pub const fn weighted(&self) -> u64 {
        const ELEM_WEIGHT: u64 = 264; // 660e9 / 2.5e9
        const TRANS_WEIGHT: u64 = 8800; // 660e9 / 75e6  (scalar libm)
        const TRANS_VEC_WEIGHT: u64 = 244; // 660e9 / 2.7e9 (vectorised)
        self.matmul_flops
            + self.elem_ops * ELEM_WEIGHT
            + self.transcendental * TRANS_WEIGHT
            + self.transcendental_vec * TRANS_VEC_WEIGHT
    }

    /// Human-readable, aligned for diffing two runs by eye.
    #[must_use]
    pub fn report(&self, label: &str) -> String {
        format!(
            "{label:<22} matmul {:>7.1} GF in {:>4} calls | elem {:>7.1} M in {:>3} calls | \
             transc {:>6.1} M scalar / {:>6.1} M vec | moved {:>7.1} MB | copies {:>3} ({:>6.1} MB) | weighted {:>8.1} G",
            self.matmul_flops as f64 / 1e9,
            self.matmul_calls,
            self.elem_ops as f64 / 1e6,
            self.elem_calls,
            self.transcendental as f64 / 1e6,
            self.transcendental_vec as f64 / 1e6,
            self.bytes_moved as f64 / 1e6,
            self.copies,
            self.copy_bytes as f64 / 1e6,
            self.weighted() as f64 / 1e9,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counters are GLOBAL, and `cargo test` runs tests in parallel — so
    /// two cost tests racing each other read one another's increments. Caught
    /// exactly that way: `counting_is_off_until_asked` failed because a
    /// concurrent test had counting enabled.
    ///
    /// Global state is the right design for the counters (the tower runs on
    /// several threads and per-thread totals would have to be summed by hand
    /// at exactly the wrong moment), so the tests take a lock instead.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn guard() -> std::sync::MutexGuard<'static, ()> {
        SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[test]
    fn counting_is_off_until_asked() {
        let _g = guard();
        // `start()` zeroes; `stop()` only disables. Zero first, or this reads
        // whatever the last test left behind.
        start();
        let _ = stop();
        matmul(1, 10, 10, 10);
        transcendental_scalar(99);
        assert_eq!(stop(), Costs::default(), "counted while disabled");
    }

    #[test]
    fn a_matmul_costs_two_flops_per_multiply_add() {
        let _g = guard();
        start();
        matmul(1, 2, 3, 4);
        let c = stop();
        assert_eq!(c.matmul_flops, 2 * 2 * 3 * 4);
        assert_eq!(c.matmul_calls, 1);
        // reads both operands, writes the result
        assert_eq!(c.bytes_moved, 4 * (2 * 3 + 3 * 4 + 2 * 4));
    }

    #[test]
    fn the_weighting_stops_a_bad_trade_reading_as_a_win() {
        let _g = guard();
        // Trading 12 M elementwise visits for 3 M transcendentals lowers
        // `elem_ops` and would look like progress on any single counter.
        start();
        elementwise(12_000_000, 1, 1);
        let before = stop();
        start();
        transcendental_scalar(3_000_000);
        let after = stop();
        assert!(
            after.weighted() > before.weighted(),
            "3 M transcendentals ({}) should outweigh 12 M elementwise visits ({})",
            after.weighted(),
            before.weighted()
        );
    }

    #[test]
    fn counters_are_reproducible_across_runs() {
        let _g = guard();
        // The whole premise: same work, same numbers, every time.
        let run = || {
            start();
            matmul(12, 1024, 64, 1024);
            elementwise(3_145_728, 1, 1);
            transcendental_scalar(3_145_728);
            copy(786_432);
            stop()
        };
        assert_eq!(run(), run(), "counters are not deterministic");
    }
}
