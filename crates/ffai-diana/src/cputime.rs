//! CPU time with usable resolution.
//!
//! # The defect this replaces
//!
//! `GetProcessTimes` reports in FILETIME ticks but only ADVANCES on the
//! scheduler's clock interrupt — 15.625 ms on Windows. It was only noticed
//! here because a set of readings were exact multiples of it: 281.2 ms is 18
//! ticks, 265.6 is 17, 250.0 is 16. At the ~250 ms scale this campaign
//! measures, that is **+/-5 %** — fine for the 2x gap it was used on,
//! useless for the 8 % ones the kernel work will produce, and worse than
//! useless because the quantisation is invisible unless you happen to divide
//! by 15.625.
//!
//! # What replaces it
//!
//! `QueryProcessCycleTime` returns CPU *cycles*, which increment continuously
//! rather than on a timer tick. It measures the same quantity in a different
//! unit — work actually executed on a core, not wall time — and it does not
//! accrue while descheduled, which is the property that made CPU time the
//! right instrument on this contended box in the first place.
//!
//! Cycles are also the more honest unit for kernel work: they do not move
//! when the CPU changes frequency under thermal or power management, which
//! millisecond figures silently do.
//!
//! A `cycles_per_sec` calibration is provided for reporting in familiar
//! units, measured once against the wall clock rather than assumed from the
//! nominal frequency — this part is a Raptor Lake mobile chip whose actual
//! clock is nothing like its label.

#[cfg(windows)]
mod imp {
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn QueryProcessCycleTime(h: isize, cycles: *mut u64) -> i32;
    }

    /// CPU cycles consumed by this process across all threads.
    ///
    /// Returns `None` if the call fails, rather than a zero that would enter
    /// a median looking like data.
    pub fn cycles() -> Option<u64> {
        let mut c = 0u64;
        // SAFETY: `GetCurrentProcess` yields a pseudo-handle that needs no
        // release, and `c` is a valid writable u64 for the call's duration.
        let ok = unsafe { QueryProcessCycleTime(GetCurrentProcess(), &mut c) };
        (ok != 0).then_some(c)
    }
}

#[cfg(not(windows))]
mod imp {
    /// No cycle counter wired for this platform yet. `None` rather than a
    /// plausible-looking wrong number.
    pub fn cycles() -> Option<u64> {
        None
    }
}

pub use imp::cycles;

/// Cycles per second, measured against the wall clock.
///
/// Calibrated rather than taken from the CPU's nominal frequency: the label
/// on a mobile part describes neither its base nor its boost behaviour under
/// load, and this box is under load whenever anything is measured.
pub fn calibrate() -> Option<f64> {
    let t = std::time::Instant::now();
    let c0 = cycles()?;
    // Busy-wait: sleeping would stop accruing cycles, which is the whole
    // point of the counter and would calibrate it to zero.
    let mut spin = 0u64;
    while t.elapsed() < std::time::Duration::from_millis(120) {
        spin = spin.wrapping_add(1);
        std::hint::black_box(spin);
    }
    let c1 = cycles()?;
    let secs = t.elapsed().as_secs_f64();
    (c1 > c0 && secs > 0.0).then(|| (c1 - c0) as f64 / secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counter must advance on work, and advance FINELY — the whole
    /// reason it exists is that the API it replaces moved in 15.625 ms steps.
    #[cfg(windows)]
    #[test]
    fn cycles_advance_at_sub_quantum_resolution() {
        let Some(a) = cycles() else { return };
        // ~1 ms of work: far below the 15.625 ms tick that GetProcessTimes
        // could not resolve at all.
        let t = std::time::Instant::now();
        let mut s = 0u64;
        while t.elapsed() < std::time::Duration::from_millis(1) {
            s = s.wrapping_add(1);
            std::hint::black_box(s);
        }
        let Some(b) = cycles() else { return };
        assert!(b > a, "cycle counter did not advance over ~1 ms of work");
    }

    #[cfg(windows)]
    #[test]
    fn calibration_lands_in_a_plausible_range() {
        let Some(hz) = calibrate() else { return };
        // 0.5-8 GHz brackets every plausible core clock; this is a sanity
        // check on the calibration, not a claim about the part.
        assert!(hz > 0.5e9 && hz < 8.0e9, "implausible cycles/sec: {hz:.3e}");
    }
}
