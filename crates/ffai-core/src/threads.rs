//! How many threads this process may actually run on.
//!
//! `std::thread::available_parallelism` ignores the process affinity mask on
//! Windows: under `start /affinity 4` (one core) it still reports every
//! logical CPU. rayon sizes its global pool the same way, and candle hands
//! gemm `Parallelism::Rayon(physical cores)` on every matmul, so a process
//! pinned to one core ran two dozen threads on it.
//!
//! PdfCraft measured what sizing the pool to the mask is worth (ABBA, output
//! identical): on 8 cores, 0.927x the time (10/10 rounds) and peak memory
//! 1 959 -> 1 240 MB; on 1 core, time neutral and memory 1 285 -> 747 MB. It
//! could not do it itself, because reading the mask is an FFI call and it
//! forbids `unsafe`. This module is that one call, behind a safe function.
//!
//! Use [`affinity_pool`] and run the engine inside it:
//!
//! ```no_run
//! let pool = ffai_core::threads::affinity_pool().expect("thread pool");
//! pool.install(|| {
//!     // engine.recognize(&image, &options) — every rayon task the engine and
//!     // candle spawn now runs on this pool's threads.
//! });
//! ```

/// The number of CPUs this process may run on: the affinity mask on Windows,
/// `available_parallelism` elsewhere (which already honours affinity and
/// cgroup quotas on Linux). At least 1.
#[must_use]
pub fn available_threads() -> usize {
    #[cfg(windows)]
    if let Some(n) = windows_affinity() {
        return n;
    }
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

#[cfg(windows)]
fn windows_affinity() -> Option<usize> {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessAffinityMask};
    let (mut process, mut system) = (0usize, 0usize);
    // SAFETY: `GetCurrentProcess` returns a pseudo-handle that is always valid
    // for the calling process and needs no closing; both out-pointers point at
    // live, writable `usize`s for the duration of the call.
    #[allow(unsafe_code)]
    let ok = unsafe { GetProcessAffinityMask(GetCurrentProcess(), &raw mut process, &raw mut system) };
    // The mask covers the process's processor group (at most 64 CPUs).
    (ok != 0 && process != 0).then(|| process.count_ones() as usize)
}

/// A rayon pool with [`available_threads`] threads.
///
/// # Errors
/// When rayon cannot spawn the threads.
pub fn affinity_pool() -> Result<rayon::ThreadPool, rayon::ThreadPoolBuildError> {
    rayon::ThreadPoolBuilder::new().num_threads(available_threads()).build()
}

#[cfg(test)]
mod tests {
    #[test]
    fn at_least_one_and_at_most_the_machine() {
        let n = super::available_threads();
        assert!(n >= 1);
        // On Windows the mask can only narrow the count, never widen it.
        #[cfg(windows)]
        assert!(n <= std::thread::available_parallelism().map_or(usize::MAX, std::num::NonZeroUsize::get));
    }
}
