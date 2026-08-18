use ffai_core::candle::{Device, Tensor};
use std::time::Instant;
fn t(n: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut b = f64::MAX;
    for _ in 0..n {
        let s = Instant::now();
        f();
        b = b.min(s.elapsed().as_secs_f64());
    }
    b
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    let x = Tensor::randn(0f32, 1., (1500, 384), &d)?.contiguous()?;
    let w1 = Tensor::randn(0f32, 1., (384, 1536), &d)?.contiguous()?;
    let w2 = Tensor::randn(0f32, 1., (1536, 384), &d)?.contiguous()?;
    let h = x.matmul(&w1)?;
    let f1 = 2.0 * 1500.0 * 384.0 * 1536.0;
    let a = t(9, || {
        std::hint::black_box(x.matmul(&w1).unwrap());
    });
    let b = t(9, || {
        std::hint::black_box(h.matmul(&w2).unwrap());
    });
    let g = t(9, || {
        std::hint::black_box(ffai_mercury::asr::text_decoder::fast_gelu(&h).unwrap());
    });
    println!(
        "fc1  (1500,384)@(384,1536)  {:6.2} ms  {:5.0} GFLOP/s",
        a * 1e3,
        f1 / a / 1e9
    );
    println!(
        "fc2  (1500,1536)@(1536,384) {:6.2} ms  {:5.0} GFLOP/s",
        b * 1e3,
        f1 / b / 1e9
    );
    println!("gelu (1500,1536)            {:6.2} ms", g * 1e3);
    println!(
        "--> one layer = {:6.2} ms, x4 layers = {:6.2} ms",
        (a + b + g) * 1e3,
        (a + b + g) * 4e3
    );
    println!("    profile says encoder mlp = 49-59 ms for 4 layers");
    Ok(())
}
