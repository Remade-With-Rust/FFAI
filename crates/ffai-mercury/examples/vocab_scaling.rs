//! The vocabulary projection is 38.6% of decode. It reads 40 MB (f16) per
//! token for ONE row of output — pure bandwidth. Against the corrected
//! ceiling (33.57 GB/s, ggml's memcpy bench) we hit ~49%. Does it scale?
use ffai_core::candle::{DType, Device, Tensor};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let (d, vocab) = (384usize, 51864usize);
    let x = Tensor::zeros((1, d), DType::F32, &dev)?.to_dtype(DType::F16)?;
    let w = Tensor::zeros((d, vocab), DType::F32, &dev)?.to_dtype(DType::F16)?;
    let bytes = (d * vocab * 2) as f64;
    let mut b = f64::MAX;
    for _ in 0..60 {
        let t = Instant::now();
        std::hint::black_box(x.matmul(&w).unwrap());
        b = b.min(t.elapsed().as_secs_f64());
    }
    println!(
        "  GEMV (1x384)@(384x51864) f16: {:6.2} ms -> {:5.1} GB/s  ({:.0}% of 33.6 GB/s ceiling)",
        b * 1e3,
        bytes / b / 1e9,
        bytes / b / 1e9 / 33.57 * 100.0
    );
    // Same bytes, but as a batched GEMM (8 rows) — does candle's matmul do
    // better when m>1, i.e. is the GEMV path the weak one?
    for m in [1usize, 2, 4, 8] {
        let xm = Tensor::zeros((m, d), DType::F32, &dev)?.to_dtype(DType::F16)?;
        let mut bb = f64::MAX;
        for _ in 0..40 {
            let t = Instant::now();
            std::hint::black_box(xm.matmul(&w).unwrap());
            bb = bb.min(t.elapsed().as_secs_f64());
        }
        println!(
            "  m={m}: {:6.2} ms  {:5.1} GB/s  ({:.2} ms per output row)",
            bb * 1e3,
            bytes / bb / 1e9,
            bb * 1e3 / m as f64
        );
    }
    // A raw parallel read of the same buffer: the achievable floor.
    let raw: Vec<f32> = vec![0.0; d * vocab];
    let mut bc = f64::MAX;
    for _ in 0..40 {
        let t = Instant::now();
        let s: f32 = raw.iter().sum();
        std::hint::black_box(s);
        bc = bc.min(t.elapsed().as_secs_f64());
    }
    println!(
        "  serial f32 read of same elems: {:6.2} ms -> {:5.1} GB/s",
        bc * 1e3,
        (d * vocab * 4) as f64 / bc / 1e9
    );
    Ok(())
}
