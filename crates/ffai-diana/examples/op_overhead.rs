//! What does one custom-op CALL cost, independent of its arithmetic?
//!
//! The serial profile still puts silu at 30.3% of detect AFTER a kernel fix
//! that made it 4.71x faster in isolation. A share that will not fall when
//! the kernel gets four times faster is not being spent in the kernel.
//!
//! One image runs ~1305 silu calls and ~1440 convolution calls through
//! `SliceOp` (candle's `CustomOp1`). If each call carries a fixed cost, that
//! cost is multiplied by ~2745 and would dominate everything.
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    println!("{:>12} {:>14} {:>14} {:>12}", "elements", "silu us/call", "kernel us", "overhead us");
    for n in [16usize, 256, 4096, 65536, 1 << 20] {
        let x = Tensor::zeros(n, DType::F32, &dev)?;
        for _ in 0..5 { let _ = ffai_diana::silu::silu(&x)?; }
        let iters = if n > 65536 { 50 } else { 2000 };
        let t = Instant::now();
        for _ in 0..iters { std::hint::black_box(ffai_diana::silu::silu(&x)?); }
        let per = t.elapsed().as_secs_f64() / iters as f64 * 1e6;
        // The kernel itself, from the corrected microbench: ~1.30 Gelem/s.
        let kernel = n as f64 / 1.30e9 * 1e6;
        println!("{n:>12} {per:>14.2} {kernel:>14.2} {:>12.2}", per - kernel);
    }
    Ok(())
}
