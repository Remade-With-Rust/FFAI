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
    // M and K co-vary in the real layers, so the first table could not tell
    // which one drives efficiency. These two sweeps hold one fixed.
    let cases = [
        (16usize, 27usize, 102400usize, "real: stem"),
        (256, 1152, 1600, "real: deepest"),
        // K fixed at 27, M swept: does M matter?
        (16, 27, 102400, "K=27  M=16"),
        (64, 27, 102400, "K=27  M=64"),
        (256, 27, 102400, "K=27  M=256"),
        // M fixed at 16, K swept: does K matter?
        (16, 27, 102400, "M=16  K=27"),
        (16, 288, 25600, "M=16  K=288"),
        (16, 1152, 6400, "M=16  K=1152"),
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
