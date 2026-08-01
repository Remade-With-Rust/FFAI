//! Where the intra-image parallelism goes — and the measurement that says
//! it should not always go to the same place.
//!
//! Every hand-written kernel here fans out with `par_chunks_mut`, once per
//! layer. YOLO26n has ~60 convolutions and ~60 activations, so a single
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
    SERIAL.with(Cell::get)
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
            CACHED.store(on as u8, Ordering::Relaxed);
            on
        }
        v => v == 1,
    }
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
