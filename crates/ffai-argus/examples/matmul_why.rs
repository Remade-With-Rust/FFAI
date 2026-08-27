//! Why do our matmuls run at ~300 GF/s in situ and ~500-570 isolated?
//!
//! Two hypotheses, both testable without touching the tower:
//!
//! A. **Cache pollution.** A layer writes a 50 MB score matrix between its
//!    GEMMs. If that evicts the packed weight panels and activations, the next
//!    GEMM restarts cold.
//! B. **Weight re-packing.** A GEMM packs its B matrix into a blocked layout
//!    before multiplying. If candle/gemm re-packs on every call, that work is
//!    repeated 12 layers x 17 tiles = 204 times per weight, per caption.
//!
//! Arm 1 runs a GEMM back-to-back (weight hot, nothing else touched).
//! Arm 2 runs the same GEMM with a 50 MB buffer touched between calls.
//! Arm 3 rotates over 12 DIFFERENT weights, as the real 12-layer tower does.
use candle_core::{Device, Tensor};
use std::time::Instant;

#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    const M: usize = 1024;
    const K: usize = 768;
    const N: usize = 3072;
    let gf = 2.0 * (M * K * N) as f64 / 1e9;

    let x = Tensor::rand(-1.0f32, 1.0, (M, K), &d)?;
    let w = Tensor::rand(-1.0f32, 1.0, (K, N), &d)?;
    // Twelve distinct weights, like twelve layers.
    let ws: Vec<Tensor> = (0..12)
        .map(|_| Tensor::rand(-1.0f32, 1.0, (K, N), &d).expect("w"))
        .collect();
    // A score-matrix-sized scratch, as a layer materialises between its GEMMs.
    let scores = Tensor::rand(-1.0f32, 1.0, (12, 1024, 1024), &d)?;

    let rate = |ms: f64| gf / (ms / 1e3);
    let run = |f: &dyn Fn()| -> f64 {
        f();
        let mut b = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            f();
            b = b.min(t.elapsed().as_secs_f64() * 1e3);
        }
        b
    };

    let a1 = run(&|| { std::hint::black_box(x.matmul(&w).expect("mm").dims()); });
    println!("  {:<46} {a1:>8.2} ms {:>7.0} GF/s", "A1  same weight, nothing else touched", rate(a1));

    let a2 = run(&|| {
        // Touch the 50 MB score matrix, as a real layer does between GEMMs.
        std::hint::black_box(scores.sum_all().expect("sum"));
        std::hint::black_box(x.matmul(&w).expect("mm").dims());
    });
    let a2n = run(&|| { std::hint::black_box(scores.sum_all().expect("sum")); });
    println!("  {:<46} {:>8.2} ms {:>7.0} GF/s   (50 MB sweep costs {a2n:.2} ms)",
        "A2  same weight, 50 MB touched between", a2 - a2n, rate(a2 - a2n));

    let i = std::sync::atomic::AtomicUsize::new(0);
    let a3 = run(&|| {
        let k = i.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::hint::black_box(x.matmul(&ws[k % 12]).expect("mm").dims());
    });
    println!("  {:<46} {a3:>8.2} ms {:>7.0} GF/s", "A3  rotating 12 weights (cold each time)", rate(a3));

    println!("\n  A1 vs A3 isolates WEIGHT residency; A1 vs A2 isolates cache pollution.");
    println!("  In situ the tower measures ~315 GF/s on this shape.");
    Ok(())
}
