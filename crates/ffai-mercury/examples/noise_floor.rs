//! Is this machine fit to measure on? Run before believing any A/B.
//! codec-six-whys-unknowns, depth 6: everything below the spread is a coin flip.
use ffai_core::candle::{Device, Tensor};
use std::time::Instant;
fn spread(label: &str, mut f: impl FnMut() -> f64) {
    let mut v: Vec<f64> = (0..11).map(|_| f()).collect();
    v.sort_by(f64::total_cmp);
    let (min, med, max) = (v[0], v[5], v[10]);
    println!(
        "{label:<26} min {min:7.2} med {med:7.2} max {max:7.2} ms   spread {:5.0}%",
        (max - min) / med * 100.0
    );
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let a = Tensor::randn(0f32, 1., (1500, 384), &dev)?;
    let b = Tensor::randn(0f32, 1., (384, 1536), &dev)?;
    let big = Tensor::randn(0f32, 1., (1024, 1024), &dev)?;
    spread("encoder fc1 (compute)", || {
        let t = Instant::now();
        let o = a.matmul(&b).unwrap();
        std::hint::black_box(&o);
        t.elapsed().as_secs_f64() * 1e3
    });
    spread("1024^3 matmul (compute)", || {
        let t = Instant::now();
        let o = big.matmul(&big).unwrap();
        std::hint::black_box(&o);
        t.elapsed().as_secs_f64() * 1e3
    });
    spread("16M copy (memory)", || {
        let t = Instant::now();
        let o = big.copy().unwrap();
        std::hint::black_box(&o);
        t.elapsed().as_secs_f64() * 1e3
    });
    println!("\nVERDICT: usable for A/Bs larger than the worst spread above.");
    Ok(())
}
