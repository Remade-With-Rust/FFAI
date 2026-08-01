//! Is candle's matmul doing well on OUR shapes, or taking a bad path?
//!
//! gemm is 18.4% of serial detect and it is the framework's code, not ours.
//! Before writing a kernel it is worth knowing whether the existing one is
//! near peak for the shapes we hand it — a framework picking a vector path
//! for a matrix problem is a known trap and costs multiples, not percent.
//!
//! Shapes are the real ones: `w[c_out, K] x col[K, ohw]` for the n tier.
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    println!("{:>22} {:>10} {:>12} {:>10}", "shape (M x K x N)", "ms", "GFLOP/s", "note");
    // (c_out, K = c_in*9 or c_in, ohw)
    let cases = [
        (16usize, 27usize, 102400usize, "stem 3x3, 640->320"),
        (32, 144, 25600, "l1 3x3 s2"),
        (64, 288, 25600, "l2 3x3"),
        (128, 576, 6400, "deep 3x3"),
        (256, 1152, 1600, "deepest 3x3"),
        (64, 64, 25600, "1x1 pointwise"),
        (256, 256, 1600, "1x1 deep"),
    ];
    for (m, k, n, note) in cases {
        let a = Tensor::zeros((m, k), DType::F32, &dev)?;
        let b = Tensor::zeros((k, n), DType::F32, &dev)?;
        for _ in 0..3 { let _ = a.matmul(&b)?; }
        let mut best = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            std::hint::black_box(a.matmul(&b)?);
            best = best.min(t.elapsed().as_secs_f64());
        }
        let gflops = 2.0 * m as f64 * k as f64 * n as f64 / best / 1e9;
        println!("{:>22} {:>10.3} {:>12.1} {:>10}", format!("{m}x{k}x{n}"), best * 1e3, gflops, note);
    }
    Ok(())
}
