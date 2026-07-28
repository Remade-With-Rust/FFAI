//! Peak-memory measurement — the fourth gate's instrument.
//!
//! `GateKind::Footprint` has printed `SKIP` since Phase 0 with the note
//! "peak-memory instrumentation lands in Mercury M2". That is not a neutral
//! omission: `GateReport::all_passed` treats a skipped gate as not passed, so
//! the harness has been unable to clear a verdict on any run, ever. It also
//! means two shipped optimizations whose entire value is memory have been
//! worth nothing on paper — the int8 decoder variant (accuracy-free,
//! speed-neutral, ~4x smaller weights) and the f16 cross-attention cache
//! (18.4 -> 9.2 MB per token). Both were kept on arguments the ledger could
//! not record.
//!
//! **What is measured.** Peak working set — the high-water mark of resident
//! physical memory — for our engine's own process and for each reference's
//! subprocess. Peak rather than current, because a transcription's cost is
//! set by its worst moment, not by whatever it happens to hold at the end.
//!
//! **Why the reference needs different handling.** Our engine runs in-process,
//! so we ask the OS about ourselves. A reference is a subprocess that has
//! usually already exited by the time we want the number. On Windows a process
//! handle stays valid after exit until it is closed, and the memory counters
//! persist with it, so the measurement is taken after `wait()` while the
//! `Child` still owns the handle. That is why `Reference::exec` cannot use
//! `Command::output()`, which consumes the child and drops the handle.
//!
//! **Platforms without an implementation return `None`**, and the gate then
//! reports `SKIP` with an honest reason rather than inventing a number.

/// Peak resident memory of a process, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeakBytes(pub u64);

impl PeakBytes {
    pub fn mib(self) -> f64 {
        self.0 as f64 / (1024.0 * 1024.0)
    }
}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;

    // PROCESS_MEMORY_COUNTERS. Field order is load-bearing — the struct is
    // written by the kernel, and `cb` must match the size we declare.
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    // K32GetProcessMemoryInfo lives in kernel32 (Windows 7+), so this needs no
    // psapi link directive and no build script.
    unsafe extern "system" {
        fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
    }

    fn query(handle: *mut c_void) -> Option<u64> {
        let mut c = ProcessMemoryCounters {
            cb: size_of::<ProcessMemoryCounters>() as u32,
            ..Default::default()
        };
        // SAFETY: `handle` is a live process handle (either the pseudo-handle
        // from GetCurrentProcess, or one owned by a `Child` that has not been
        // dropped). `c` is a correctly-sized, fully-initialized struct whose
        // `cb` field declares its own size, which is the contract the call
        // documents.
        let ok = unsafe { K32GetProcessMemoryInfo(handle, &mut c, c.cb) };
        (ok != 0).then_some(c.peak_working_set_size as u64)
    }

    pub fn peak_self() -> Option<u64> {
        // SAFETY: returns a pseudo-handle; needs no close.
        query(unsafe { GetCurrentProcess() })
    }

    pub fn peak_child(child: &std::process::Child) -> Option<u64> {
        use std::os::windows::io::AsRawHandle;
        query(child.as_raw_handle() as *mut c_void)
    }

    // ---- process-TREE measurement -------------------------------------
    //
    // A reference is usually not one process. `whisper-cpp-tiny-greedy-t24`
    // runs `python.exe adapter.py --bin whisper-cli.exe`, so the process we
    // spawn is a launcher and the actual work happens in a grandchild.
    // Measuring the direct child reported **5 MiB for a reference that loads a
    // 77.7 MB model** — a number that cannot be true, next to a 127x ratio
    // that also cannot be true.
    //
    // A Job Object fixes the scope: descendants inherit job membership, so the
    // job's peak covers the whole tree however deep it goes, and no adapter has
    // to cooperate by reporting its own memory.

    #[repr(C)]
    #[derive(Default)]
    struct JobBasicLimitInformation {
        per_process_user_time_limit: i64,
        per_job_user_time_limit: i64,
        limit_flags: u32,
        minimum_working_set_size: usize,
        maximum_working_set_size: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct IoCounters {
        read_operation_count: u64,
        write_operation_count: u64,
        other_operation_count: u64,
        read_transfer_count: u64,
        write_transfer_count: u64,
        other_transfer_count: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct JobExtendedLimitInformation {
        basic: JobBasicLimitInformation,
        io: IoCounters,
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }

    const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: u32 = 9;
    const JOB_OBJECT_BASIC_PROCESS_ID_LIST: u32 = 3;
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;

    // Variable-length in the API; 256 slots is far more than any reference
    // tree here and keeps this a plain stack struct.
    #[repr(C)]
    struct JobBasicProcessIdList {
        number_of_assigned_processes: u32,
        number_of_process_ids_in_list: u32,
        process_id_list: [usize; 256],
    }

    unsafe extern "system" {
        fn CreateJobObjectW(attrs: *mut c_void, name: *const u16) -> *mut c_void;
        fn AssignProcessToJobObject(job: *mut c_void, process: *mut c_void) -> i32;
        fn QueryInformationJobObject(
            job: *mut c_void,
            class: u32,
            info: *mut c_void,
            len: u32,
            returned: *mut u32,
        ) -> i32;
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut c_void;
        fn CloseHandle(h: *mut c_void) -> i32;
    }

    /// A job the spawned process and all its descendants belong to.
    pub struct Job(*mut c_void);

    // SAFETY: a Windows job handle names a process-wide kernel object, and the
    // calls made through it here (QueryInformationJobObject,
    // AssignProcessToJobObject) are thread-safe. The handle is closed exactly
    // once, in `Drop`. Sharing it with the sampling thread is therefore sound;
    // the raw pointer is only `!Send` because the compiler cannot see that.
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    impl Job {
        pub fn create() -> Option<Self> {
            // SAFETY: null attributes and a null name request an unnamed job
            // with default security, which is what the API documents.
            let h = unsafe { CreateJobObjectW(std::ptr::null_mut(), std::ptr::null()) };
            (!h.is_null()).then_some(Job(h))
        }

        /// Assign a freshly-spawned child. Descendants it creates AFTER this
        /// call inherit membership; anything it spawned in the microseconds
        /// before would be missed, which is why this runs immediately after
        /// `spawn`.
        pub fn assign(&self, child: &std::process::Child) -> bool {
            use std::os::windows::io::AsRawHandle;
            // SAFETY: `self.0` is a live job handle owned by this struct and
            // the child handle is owned by a `Child` that is still alive.
            unsafe { AssignProcessToJobObject(self.0, child.as_raw_handle() as *mut c_void) != 0 }
        }

        /// Peak COMMITTED memory across the job.
        ///
        /// **Do not use this as a footprint figure.** Calibration
        /// (`examples/footprint_calibrate.rs`) shows `whisper-cli --help` —
        /// which prints a help message and exits — reporting **2953 MiB** here
        /// against 8 MiB of working set. That is OpenBLAS committing per-thread
        /// buffers it never touches; commit counts address space that was
        /// reserved, not memory that occupies RAM. Reporting it as "footprint"
        /// would have claimed a 7x memory advantage built almost entirely out
        /// of pages the reference never faulted in.
        ///
        /// Kept because it is the right question for "will this fit in a
        /// commit limit", which is not the question this gate asks.
        pub fn peak_commit(&self) -> Option<u64> {
            let mut info = JobExtendedLimitInformation::default();
            let mut returned = 0u32;
            // SAFETY: `info` is a correctly-shaped, fully-initialized struct
            // and its length is passed alongside, per the API contract.
            let ok = unsafe {
                QueryInformationJobObject(
                    self.0,
                    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                    &mut info as *mut _ as *mut c_void,
                    size_of::<JobExtendedLimitInformation>() as u32,
                    &mut returned,
                )
            };
            (ok != 0).then_some(info.peak_job_memory_used as u64)
        }
    }

    impl Job {
        /// Summed CURRENT working set across every process in the job, right
        /// now. Sample it repeatedly and keep the maximum: that maximum is the
        /// tree's peak resident memory, which is the figure a footprint claim
        /// should rest on.
        ///
        /// It has to be sampled rather than read at the end, because a
        /// process's counters die with it and the interesting process (the
        /// grandchild doing the work) exits before we look.
        pub fn working_set_now(&self) -> Option<u64> {
            let mut list = JobBasicProcessIdList {
                number_of_assigned_processes: 0,
                number_of_process_ids_in_list: 0,
                process_id_list: [0; 256],
            };
            // SAFETY: correctly-shaped struct, its own size passed alongside.
            let ok = unsafe {
                QueryInformationJobObject(
                    self.0,
                    JOB_OBJECT_BASIC_PROCESS_ID_LIST,
                    &mut list as *mut _ as *mut c_void,
                    size_of::<JobBasicProcessIdList>() as u32,
                    std::ptr::null_mut(),
                )
            };
            if ok == 0 {
                return None;
            }
            let n = (list.number_of_process_ids_in_list as usize).min(256);
            let mut total = 0u64;
            for &pid in &list.process_id_list[..n] {
                // SAFETY: pid comes from the kernel's own list; a failed open
                // (the process just exited) yields null and is skipped.
                let h = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid as u32) };
                if h.is_null() {
                    continue;
                }
                let mut c = ProcessMemoryCounters {
                    cb: size_of::<ProcessMemoryCounters>() as u32,
                    ..Default::default()
                };
                // SAFETY: `h` is a live handle we just opened and own.
                if unsafe { K32GetProcessMemoryInfo(h, &mut c, c.cb) } != 0 {
                    total += c.working_set_size as u64;
                }
                // SAFETY: handle opened above, not used after this.
                unsafe { CloseHandle(h) };
            }
            Some(total)
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            // SAFETY: handle created by CreateJobObjectW and owned here.
            unsafe { CloseHandle(self.0) };
        }
    }

    /// Peak COMMITTED memory of the current process — the same quantity the
    /// job reports, so our side and the reference side are the same metric.
    pub fn peak_commit_self() -> Option<u64> {
        let mut c = ProcessMemoryCounters {
            cb: size_of::<ProcessMemoryCounters>() as u32,
            ..Default::default()
        };
        // SAFETY: as in `query` — pseudo-handle, correctly-sized struct.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
        (ok != 0).then_some(c.peak_pagefile_usage as u64)
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn peak_self() -> Option<u64> {
        // Linux exposes VmHWM (the peak resident set) in /proc/self/status.
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        parse_vm_hwm(&status)
    }

    /// A child's peak is not readable after exit on Linux the way it is on
    /// Windows — /proc/<pid> is gone. Sampling it while alive is the portable
    /// answer and is not implemented yet, so the gate SKIPs rather than
    /// reporting a number it did not measure.
    pub fn peak_child(_child: &std::process::Child) -> Option<u64> {
        None
    }

    pub(super) fn parse_vm_hwm(status: &str) -> Option<u64> {
        let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }

    pub fn peak_commit_self() -> Option<u64> {
        peak_self()
    }

    /// No cgroup/job equivalent wired up here yet, so a tree measurement is
    /// not available and the gate skips rather than reporting a partial one.
    pub struct Job;

    impl Job {
        pub fn create() -> Option<Self> {
            None
        }
        pub fn assign(&self, _child: &std::process::Child) -> bool {
            false
        }
        pub fn peak_commit(&self) -> Option<u64> {
            None
        }
        pub fn working_set_now(&self) -> Option<u64> {
            None
        }
    }
}

/// Peak resident memory of the current process.
pub fn peak_self() -> Option<PeakBytes> {
    imp::peak_self().map(PeakBytes)
}

/// Peak resident memory of a child process.
///
/// Must be called while the `Child` still owns its handle — after `wait()` is
/// fine, after `drop` is not.
pub fn peak_child(child: &std::process::Child) -> Option<PeakBytes> {
    imp::peak_child(child).map(PeakBytes)
}

/// Peak **committed** memory of the current process.
///
/// This, not the working set, is what the engine side reports — it is the same
/// quantity [`Job::peak_commit`] reports for a reference's whole process tree,
/// and comparing a working set against a job's commit would be the metric
/// version of pairing numbers taken at different operating points.
pub fn peak_commit_self() -> Option<PeakBytes> {
    imp::peak_commit_self().map(PeakBytes)
}

/// A process tree to measure as one unit. See [`imp::Job`].
pub use imp::Job;

/// Whether this platform can measure at all, so the gate can say why it
/// skipped instead of silently reporting nothing.
pub fn supported() -> bool {
    cfg!(windows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn our_own_peak_is_plausible() {
        let Some(p) = peak_self() else {
            assert!(!supported(), "peak_self returned None on a supported platform");
            return;
        };
        // A running test process holds at least a megabyte and nothing like a
        // terabyte. The point is to catch a garbage struct layout, which would
        // show up as an absurd number rather than a failure.
        assert!(p.0 > 1 << 20, "implausibly small peak: {} bytes", p.0);
        assert!(p.0 < 1 << 40, "implausibly large peak: {} bytes", p.0);
    }

    #[test]
    fn peak_grows_when_memory_is_touched() {
        let Some(before) = peak_self() else { return };
        // Touch every page so the pages are genuinely resident, not just
        // reserved — a lazily-allocated Vec would not move the working set.
        let mut big = vec![0u8; 192 << 20];
        for i in (0..big.len()).step_by(4096) {
            big[i] = 1;
        }
        std::hint::black_box(&big);
        let after = peak_self().expect("supported once before succeeded");
        assert!(
            after.0 > before.0,
            "peak did not rise after touching 192 MiB: {} -> {}",
            before.0,
            after.0
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn vm_hwm_parses() {
        let sample = "Name:\tx\nVmHWM:\t  123456 kB\nVmRSS:\t  100 kB\n";
        assert_eq!(imp::parse_vm_hwm(sample), Some(123456 * 1024));
    }
}
