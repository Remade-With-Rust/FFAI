//! Does the self-attention KV cache cost O(n^2)? Every token appends via
//! Tensor::cat, which allocates a new buffer and copies the whole cache.
use ffai_core::candle::{DType, Device, Tensor};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let d = 384usize;
    let step = Tensor::zeros((1, 1, d), DType::F32, &dev)?;
    for &n in &[1usize, 10, 25, 50, 100, 200] {
        let cache = Tensor::zeros((1, n, d), DType::F32, &dev)?;
        let mut b = f64::MAX;
        for _ in 0..200 {
            let t = Instant::now();
            std::hint::black_box(Tensor::cat(&[&cache, &step], 1).unwrap());
            b = b.min(t.elapsed().as_secs_f64());
        }
        println!(
            "  cache len {n:4}: cat = {:7.2} us   ({:.1} KB copied)",
            b * 1e6,
            (n + 1) * d * 4 / 1024
        );
    }
    // total cost of growing 1..N by cat, vs one preallocated write
    for &n in &[50usize, 224] {
        let mut total = 0.0;
        for len in 1..n {
            let cache = Tensor::zeros((1, len, d), DType::F32, &dev)?;
            let t = Instant::now();
            std::hint::black_box(Tensor::cat(&[&cache, &step], 1).unwrap());
            total += t.elapsed().as_secs_f64();
        }
        println!(
            "  growing to {n:4} tokens by cat: {:8.2} us total  (x2 for K and V = {:.2} us)",
            total * 1e6,
            total * 2e6
        );
    }
    Ok(())
}
