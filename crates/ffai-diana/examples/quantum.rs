//! Demonstrate the 15.625 ms quantum, and that cycles do not have it.
fn main() {
    let hz = ffai_diana::cputime::calibrate().unwrap_or(f64::NAN);
    println!("calibrated {:.3} GHz\n", hz / 1e9);
    println!("{:>10} {:>16} {:>16}", "work ms", "GetProcessTimes", "QueryProcessCycleTime");
    for ms in [1u64, 4, 8, 20, 60] {
        let (c0, f0) = (ffai_diana::cputime::cycles().unwrap_or(0), filetime_ms());
        let t = std::time::Instant::now();
        let mut s = 0u64;
        while t.elapsed() < std::time::Duration::from_millis(ms) {
            s = s.wrapping_add(1);
            std::hint::black_box(s);
        }
        let (c1, f1) = (ffai_diana::cputime::cycles().unwrap_or(0), filetime_ms());
        println!("{ms:>10} {:>15.2}ms {:>15.2}ms", f1 - f0, (c1 - c0) as f64 / hz * 1e3);
    }
}
fn filetime_ms() -> f64 {
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn GetCurrentProcess() -> isize;
            fn GetProcessTimes(h: isize, c: *mut u64, e: *mut u64, k: *mut u64, u: *mut u64) -> i32;
        }
        let (mut a, mut b, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
        if unsafe { GetProcessTimes(GetCurrentProcess(), &mut a, &mut b, &mut k, &mut u) } != 0 {
            return (k + u) as f64 * 1e-4;
        }
    }
    f64::NAN
}
