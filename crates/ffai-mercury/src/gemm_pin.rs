//! Pin gemm's once-per-process cache detection to a fixed core.
//!
//! candle's CPU matmul runs on the `gemm` crate. On first use, gemm reads the
//! cache sizes with CPUID and caches them for the rest of the process
//! (`gemm_common::cache::CACHE_INFO`). CPUID answers for the core the calling
//! thread happens to be on. On a hybrid CPU the core types disagree: on an
//! i7-14650HX a P-core reports a 48 KiB 12-way L1 and an E-core a 32 KiB
//! 8-way L1. gemm derives its k-block from the L1, so a multiply with an
//! inner dimension above 512 splits its sum differently (k=960 is one block
//! of 960 on a P-core and two of 480 on an E-core) and the last bits move.
//!
//! The result was a per-process coin flip: whichever core the first matmul
//! landed on chose the output. Piper TTS showed two exact waveforms per seed,
//! the second in about 3-5 % of unpinned runs; pinning a run to E-cores gave
//! the second waveform every time, and pinning it to P-cores never did.
//!
//! The fix forces the detection early, on a helper thread pinned to the
//! lowest logical CPU this process may use. Unpinned, that is CPU 0, which is
//! a P-core on Intel hybrid parts. The answer is then the same on every run
//! with the same affinity. Where pinning is unavailable the detection still
//! runs here, which is no worse than before.
//!
//! The direct `gemm-common` dependency must resolve to the SAME version
//! candle-core links, or this initializes a different static and does
//! nothing. Keep it in step with candle's `gemm` (0.19 for candle 0.11).

use std::sync::Once;

/// Fix gemm's cache detection to the lowest allowed logical CPU.
///
/// Idempotent and cheap after the first call. Mercury's model loaders call
/// it. An application that runs other candle code before loading a Mercury
/// model should call it first, because gemm's detection is process-wide.
pub fn pin_gemm_blocking() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let t = std::thread::Builder::new()
                .name("ffai-gemm-pin".into())
                .spawn(|| {
                    pin_current_thread_to_lowest_cpu();
                    let _ = *gemm_common::cache::CACHE_INFO;
                });
            match t {
                Ok(h) => {
                    let _ = h.join();
                }
                Err(_) => {
                    let _ = *gemm_common::cache::CACHE_INFO;
                }
            }
        }
    });
}

#[cfg(windows)]
#[allow(unsafe_code)]
fn pin_current_thread_to_lowest_cpu() {
    type Handle = *mut core::ffi::c_void;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> Handle;
        fn GetCurrentThread() -> Handle;
        fn GetProcessAffinityMask(p: Handle, proc_mask: *mut usize, sys_mask: *mut usize) -> i32;
        fn SetThreadAffinityMask(t: Handle, mask: usize) -> usize;
    }
    let (mut proc_mask, mut sys_mask) = (0usize, 0usize);
    // SAFETY: Class D (OS FFI). The pseudo-handles from GetCurrentProcess /
    // GetCurrentThread are always valid for the calling process and thread
    // and need no close. Both out-pointers are live, aligned locals. A zero
    // return means failure and leaves the thread unpinned, which is handled.
    unsafe {
        if GetProcessAffinityMask(GetCurrentProcess(), &raw mut proc_mask, &raw mut sys_mask) == 0
            || proc_mask == 0
        {
            return;
        }
        let lowest = proc_mask & proc_mask.wrapping_neg();
        let _ = SetThreadAffinityMask(GetCurrentThread(), lowest);
    }
}

#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
fn pin_current_thread_to_lowest_cpu() {
    // glibc and musl cpu_set_t: 1024 bits.
    #[repr(C)]
    struct CpuSet([u64; 16]);
    unsafe extern "C" {
        fn sched_getaffinity(pid: i32, size: usize, set: *mut CpuSet) -> i32;
        fn sched_setaffinity(pid: i32, size: usize, set: *const CpuSet) -> i32;
    }
    let mut set = CpuSet([0; 16]);
    // SAFETY: Class D (OS FFI). pid 0 names the calling thread. `set` is a
    // live, aligned local of exactly the size passed. A nonzero return means
    // failure and leaves the thread unpinned, which is handled.
    unsafe {
        if sched_getaffinity(0, core::mem::size_of::<CpuSet>(), &raw mut set) != 0 {
            return;
        }
        let Some((w, bits)) = set.0.iter().enumerate().find(|(_, b)| **b != 0) else {
            return;
        };
        let mut one = CpuSet([0; 16]);
        one.0[w] = bits & bits.wrapping_neg();
        let _ = sched_setaffinity(0, core::mem::size_of::<CpuSet>(), &raw const one);
    }
}

#[cfg(not(any(windows, target_os = "linux", target_arch = "wasm32")))]
fn pin_current_thread_to_lowest_cpu() {}

#[cfg(test)]
mod tests {
    #[test]
    fn pin_is_idempotent_and_initializes_gemm() {
        super::pin_gemm_blocking();
        super::pin_gemm_blocking();
        #[cfg(not(target_arch = "wasm32"))]
        assert!(gemm_common::cache::CACHE_INFO[0].cache_bytes > 0);
    }
}
