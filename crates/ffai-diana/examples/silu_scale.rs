//! Does SiLU scale in ISOLATION, or only fail to scale in the pipeline?
//!
//! The stage profile says SiLU scales 1.24x on four workers while 91.3 % of
//! its elements sit in calls that DO fan out. Those two facts do not fit, and
//! the campaign's ranked spine puts SiLU top at 9.7 ms of serial time per
//! image, so the discrepancy is worth one probe.
//!
//! Two explanations, and this separates them:
//!
//! * **the kernel does not scale** — elementwise work at ~15 operations per
//!   8 bytes may saturate cache ports, in which case extra workers cannot
//!   help and the 9.7 ms is not recoverable;
//! * **the pipeline does not let it scale** — per-call fork-join at 87 calls
//!   per image, or the profiler's own tax on a bucket with 87 scope entries,
//!   in which case the kernel is fine and the wrapper is the target.
//!
//! Isolated, at the size the pipeline actually uses (10.38 M elements over 52
//! fanning-out calls = ~200 k each), min-of-N so the box's tail cannot
//! dominate.
//!
//! # WARNING: this probe is CACHE-WARM and the pipeline is not
//!
//! It reuses ONE buffer for every iteration, so after the first the data is
//! resident and the kernel never pays for a cold read. The pipeline's SiLU
//! takes a fresh activation each call.
//!
//! Its 8.33 Gelem/s at four threads is therefore a CEILING, not a target, and
//! the 1.25 ms/image derived from it is a lower bound. The pipeline's 0.79
//! Gelem/s is the honest figure; the gap between them is part per-call
//! machinery and part cold data, and THIS PROBE CANNOT SEPARATE THEM.
//!
//! To separate them, rotate over enough distinct buffers to evict L2 between
//! iterations. See `docs/whys/diana-latency.md` — that experiment must come
//! before any buffer-pool work.
use candle_core::{Device, Tensor};
use std::time::Instant;

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let reps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let xs: Vec<f32> = (0..n).map(|i| (i % 977) as f32 * 0.01 - 4.0).collect();
    // The real entry point, SliceOp wrapper included — that wrapper is one of
    // the two suspects, so measuring past it would answer the wrong question.
    let x = Tensor::from_vec(xs, (1, n), &Device::Cpu).unwrap();

    for threads in [1usize, 2, 4, 6, 8] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            let out = pool.install(|| ffai_diana::silu::silu(&x).unwrap());
            let ms = t.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&out);
            best = best.min(ms);
        }
        let gelem = n as f64 / (best / 1e3) / 1e9;
        if threads == 1 {
            BASE.with(|b| b.set(best));
        }
        let sp = BASE.with(|b| b.get()) / best;
        println!("threads={threads:<2} min {best:7.3} ms   {gelem:5.2} Gelem/s   speedup {sp:.2}x");
    }
}

thread_local! {
    static BASE: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}
