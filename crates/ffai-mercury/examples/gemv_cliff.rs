//! Does candle's m=1 cliff apply to every decoder matmul, or only the big one?
use ffai_core::candle::{DType, Device, Tensor};
use std::time::Instant;
fn probe(name: &str, k: usize, n: usize, dt: DType, dev: &Device) {
    let w = Tensor::zeros((k, n), DType::F32, dev)
        .unwrap()
        .to_dtype(dt)
        .unwrap();
    let mut best1 = f64::MAX;
    let mut best4 = f64::MAX;
    let x1 = Tensor::zeros((1, k), DType::F32, dev)
        .unwrap()
        .to_dtype(dt)
        .unwrap();
    let x4 = Tensor::zeros((4, k), DType::F32, dev)
        .unwrap()
        .to_dtype(dt)
        .unwrap();
    for _ in 0..300 {
        let t = Instant::now();
        std::hint::black_box(x1.matmul(&w).unwrap());
        best1 = best1.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        std::hint::black_box(x4.matmul(&w).unwrap());
        best4 = best4.min(t.elapsed().as_secs_f64());
    }
    println!(
        "  {name:<28} m=1 {:8.2} us   m=4 {:8.2} us   ratio {:5.2}x {}",
        best1 * 1e6,
        best4 * 1e6,
        best1 / best4,
        if best1 / best4 > 1.15 {
            "<- PAD WINS"
        } else {
            ""
        }
    );
}
fn main() {
    let dev = Device::Cpu;
    let d = 384usize;
    println!("candle matmul, m=1 vs m=4 (absolute time for MORE work):");
    probe("attn proj 384x384 f32", d, d, DType::F32, &dev);
    probe("mlp fc1 384x1536 f32", d, d * 4, DType::F32, &dev);
    probe("mlp fc2 1536x384 f32", d * 4, d, DType::F32, &dev);
    probe("vocab 384x51864 f16", d, 51864, DType::F16, &dev);
    probe("vocab 384x51864 f32", d, 51864, DType::F32, &dev);
}
