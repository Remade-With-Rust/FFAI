//! Where the intra-image parallelism goes — and the measurement that says
//! it should not always go to the same place.
//!
//! Every hand-written kernel here fans out with `par_chunks_mut`, once per
//! layer. `YOLO26n` has ~60 convolutions and ~60 activations, so a single
//! image pays on the order of a hundred fork-joins, each one waking every
//! worker in the pool and waiting for the slowest to arrive. Measured cost
//! of that, at the n tier on a 16-physical/24-logical box, as CPU time per
//! image (the metric that survives a loaded machine, since CPU time does not
//! accrue while descheduled):
//!
//! | rayon threads | CPU ms/image | vs serial |
//! |---:|---:|---:|
//! | 1 | **363** | — |
//! | 8 | 523 | 1.44x |
//! | 16 | 641 | 1.77x |
//! | 24 | 844 | **2.32x** |
//!
//! **The work is 363 ms. At 24 threads we spend 844 ms to do it.** The
//! overhead is close to linear in thread count (~20 ms per thread), which is
//! the signature of a per-thread cost paid at every barrier, not of a slow
//! kernel. Restricting to the P-cores of this hybrid CPU recovers 1.22x of
//! it and no more, so E-core IPC is a contributor and not the cause.
//!
//! ## The sign-flip, which is why this is a dispatch and not a constant
//!
//! That overhead buys something real when one image is all there is: a
//! paired ABBA A/B of a 24-thread pool against an 8-thread pool over the
//! same images in one process measured **9/24 for 8 threads, z = -1.22,
//! median 0.979x** — fewer threads is not faster on wall. For LATENCY the
//! fan-out is worth its cost, because the alternative is leaving 15 cores
//! idle.
//!
//! For THROUGHPUT it is pure waste. When N images are in flight there is
//! already enough work to fill every core, and the nested fan-out only adds
//! barriers: `detect_batch` fans out across images while each image's
//! kernels fan out again into the same pool. Serial kernels inside a
//! parallel batch do the same arithmetic in 363 ms of CPU instead of 844.
//!
//! **Same knob, opposite optima, decided by which mode the caller is in.**
//! That is the `codec-content-adaptive-dispatch` trigger — a sign-flip is a
//! dispatch signal, not a tuning value — with the workload mode standing in
//! for the content class. So the choice is made per call, not per build:
//! `detect` fans out, `detect_batch` does not.

use std::cell::Cell;

thread_local! {
    /// Set for the duration of one image's work inside a parallel batch.
    ///
    /// A thread-local rather than a parameter because the flag has to reach
    /// four kernels through candle's `CustomOp1` boundary, which takes no
    /// context of ours. It is set and cleared by [`serial_scope`] on the
    /// same thread that runs the work, so a panic mid-image cannot leave it
    /// set for the next task (see the guard).
    static SERIAL: Cell<bool> = const { Cell::new(false) };
}

/// True when the current thread should run kernels serially because the
/// caller has already saturated the machine at a coarser grain.
#[inline]
pub fn serial_kernels() -> bool {
    // wasm32 has no threads to spawn, so the fan-out arm does not exist. This
    // is not a concession: the table above says serial is 363 ms of CPU
    // against 844 ms at 24 threads, so wasm gets the arm that was already
    // winning on CPU. Only wall-clock parallelism is lost.
    #[cfg(target_arch = "wasm32")]
    {
        return true;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        SERIAL.with(Cell::get)
    }
}

/// Run `f` with intra-image parallelism disabled on this thread.
///
/// Restores the previous value on the way out, including on unwind — the
/// guard's `Drop` does it, so an error return or a panic inside a kernel
/// cannot leak the flag onto a rayon worker that will be reused for the
/// next image.
pub fn serial_scope<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SERIAL.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(SERIAL.with(Cell::get));
    SERIAL.with(|s| s.set(!nested_par_forced()));
    f()
}

/// `FFAI_DIANA_NESTED_PAR=1` restores the old nested fan-out.
///
/// Kept because a reverted-or-kept decision that cannot be re-measured is a
/// decision that gets re-litigated. The A/B this exists for is in
/// `examples/batch_ab.rs`; the knob costs one relaxed atomic load per image.
fn nested_par_forced() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(u8::MAX);
    match CACHED.load(Ordering::Relaxed) {
        u8::MAX => {
            let on = std::env::var("FFAI_DIANA_NESTED_PAR").is_ok_and(|v| v == "1");
            CACHED.store(u8::from(on), Ordering::Relaxed);
            on
        }
        v => v == 1,
    }
}

/// The pool a SINGLE-image `detect` fans out into.
///
/// # Why this is not rayon's default pool
///
/// Every kernel here fans out once per layer, so one image pays on the order
/// of a hundred fork-joins. A barrier's cost grows with the number of workers
/// that must be woken and waited for, while the work behind each barrier does
/// NOT grow — most activation tensors are small. Past a few workers the pool
/// is being paid to arrive, not to compute.
///
/// Measured on a 16-physical / 24-logical box, n tier, min-of-80 per run,
/// arms interleaved to break the warming trend (which is large here: a
/// 10-rep version of this same sweep drifted 140 ms -> 97 ms across a
/// session and would have been read as a thread effect):
///
/// | 24 threads | 4 threads |
/// |---:|---:|
/// | 101.2 | 83.2 |
/// | 100.6 | 85.5 |
/// | 101.5 | 81.7 |
/// | 101.1 | 79.5 |
/// | 99.6 | 83.4 |
/// | 102.2 | 82.4 |
///
/// **6/6, median 1.22x, and the distributions do not overlap** — the slowest
/// 4-thread run (85.5 ms) beats the fastest 24-thread run (99.6 ms). A
/// narrow sweep puts the optimum in a flat basin from 3 to 6 workers, with 2
/// clearly too few.
///
/// ## The tension with the A/B recorded above, stated rather than buried
///
/// The module docs record a paired in-process ABBA of **8 threads against
/// 24** finding no wall difference (9/24, z = -1.22, median 0.979x). This
/// does not overturn that: it tests **4**, not 8, and 8 sits at the edge of
/// the basin where the effect is small. The two results are consistent with
/// an optimum near 4 that 8 barely captures. The earlier measurement is the
/// methodologically stronger design (same process, paired over identical
/// images); this one is cross-process, and its defence is that 80 reps
/// amortise pool spin-up and the arms are interleaved and fully separated.
///
/// The spin-up confound was checked directly rather than assumed away: at 10
/// reps the gap is 1.18x and at 80 reps it is 1.22x. Amortising the one-time
/// cost STRENGTHENED the effect, which is the opposite of what a spin-up
/// artifact does.
///
/// Not used by `detect_batch`: a batch already fills every core across
/// images, and [`serial_scope`] turns the per-layer fan-out off entirely.
/// `FFAI_DIANA_THREADS=0` installs NO pool of our own, so kernels run on
/// rayon's global pool.
///
/// This is the arm the 1.21x measurement never had. That A/B compared our
/// pool at 4 against our pool at 24 — BOTH installing a pool — so it priced
/// the pool's WIDTH and never the pool's EXISTENCE.
///
/// The hypothesis was that the pool's EXISTENCE costs something: candle 0.11
/// keeps its own 24-thread pool, so every matmul called from one of our
/// workers is a cross-pool handoff, ~1320 of them per image.
///
/// Measured on CPU TIME — which a contended box cannot inflate — 8 pairs,
/// ABBA, arms fully separated:
///
/// | | CPU per image |
/// |---|---:|
/// | pool of 4 | **199-250 ms** |
/// | no pool (global rayon) | **761-826 ms** |
///
/// **The pool cuts CPU work by 3.5x.** The handoff hypothesis is refuted and
/// its opposite is true: without our pool, 24 global rayon workers AND
/// candle's 24 contend for 24 cores, and three quarters of the machine goes
/// to barriers. This is the strongest evidence for the pool by some margin —
/// the earlier 1.21x compared widths (4 against 24) and could not see it,
/// because both arms already had a pool.
///
/// It also reframes the utilisation figure. ~225 ms of CPU in ~100 ms of wall
/// is ~2.2 cores busy, and that is not idle capacity going to waste: the
/// useful work is simply SMALL, and spreading it wider costs more in barriers
/// than it recovers. Reaching 53 ms would need either ~4.2 cores busy on this
/// work — near-perfect scaling — or less work.
#[must_use] 
pub fn no_pool() -> bool {
    std::env::var("FFAI_DIANA_THREADS").is_ok_and(|v| v == "0")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn latency_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
        // Env first: the basin's location is a property of the machine and
        // the model, and a deployment on different hardware should be able to
        // re-measure it with `examples/scaling.rs` and set it.
        let n = std::env::var("FFAI_DIANA_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .filter(|&n| n > 0)
            .unwrap_or_else(|| (cores / 6).clamp(3, 6));
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("diana-{i}"))
            .build()
            .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().expect("default pool"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_parallel() {
        assert!(!serial_kernels());
    }

    #[test]
    fn scope_sets_and_restores() {
        assert!(!serial_kernels());
        serial_scope(|| assert!(serial_kernels()));
        assert!(!serial_kernels());
    }

    #[test]
    fn nested_scopes_restore_the_outer_value() {
        serial_scope(|| {
            serial_scope(|| assert!(serial_kernels()));
            assert!(serial_kernels(), "inner scope must not clear the outer one");
        });
        assert!(!serial_kernels());
    }

    /// The flag must not survive a panic, or a rayon worker reused for the
    /// next image would silently run it serially inside a LATENCY call.
    #[test]
    fn unwinding_restores_the_flag() {
        let r = std::panic::catch_unwind(|| {
            serial_scope(|| panic!("kernel blew up"));
        });
        assert!(r.is_err());
        assert!(!serial_kernels(), "flag leaked past an unwind");
    }
}
