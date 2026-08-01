//! What does one candle `matmul` CALL cost, independent of its arithmetic?
//!
//! Tiling im2col so each block stays in L2 would replace one big GEMM per
//! convolution with ~10 small ones. That is only worth doing if the
//! per-call cost is small against the bandwidth it saves, so this prices
//! the call before anything is built — `expected gain = saving - tax`.
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    // Layer-0-ish shapes: w [16, 27], col [27, N].
    let w = Tensor::zeros((16usize, 27usize), DType::F32, &dev)?;
    for n in [102400usize, 10240, 4855, 1024] {
        let col = Tensor::zeros((27usize, n), DType::F32, &dev)?;
        for _ in 0..3 { let _ = w.matmul(&col)?; }
        let mut best = f64::MAX;
        for _ in 0..9 {
            let t = Instant::now();
            let y = w.matmul(&col)?;
            std::hint::black_box(&y);
            best = best.min(t.elapsed().as_secs_f64());
        }
        let tiles = (102400 + n - 1) / n;
        println!("N={n:>7}  {:>8.3} ms/call   {tiles:>3} tiles to cover 102400 -> {:>8.3} ms total",
                 best * 1e3, best * 1e3 * tiles as f64);
    }
    Ok(())
}
