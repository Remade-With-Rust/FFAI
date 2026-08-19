//! Memory bandwidth as a function of WORKING SET — the number that decides
//! whether im2col's traffic is a DRAM floor or an L3 one.
//!
//! A 216.9 MiB/image im2col figure was priced against a roofline measured
//! with a 64 MiB copy, i.e. DRAM. But this machine has 30 MB of L3 and the
//! n-tier col buffer is 11 MB per layer, so that traffic may never leave
//! cache. Same counter, wildly different ceiling.
use std::time::Instant;

fn main() {
    println!("{:>10} {:>12} {:>12}", "buffer", "GB/s", "note");
    for mb in [1usize, 4, 8, 11, 24, 64, 128] {
        let n = mb * 1024 * 1024 / 4;
        let src = vec![1.0f32; n];
        let mut dst = vec![0.0f32; n];
        for _ in 0..3 { dst.copy_from_slice(&src); std::hint::black_box(&dst); }
        let mut best = f64::MAX;
        for _ in 0..9 {
            let t = Instant::now();
            dst.copy_from_slice(&src);
            std::hint::black_box(&dst);
            best = best.min(t.elapsed().as_secs_f64());
        }
        let gbs = (n as f64 * 8.0) / best / 1e9;
        let note = if mb <= 24 { "L2/L3 resident" } else { "DRAM" };
        println!("{mb:>8} MB {gbs:>12.1} {note:>12}");
    }
}
