//! The conv front end runs at 104 GFLOP/s while every matmul in this pipeline
//! runs at 350-550. Which conv, and is a matmul form faster?
use std::time::Instant;
use ffai_core::candle::{Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module};
fn t(n: usize, mut f: impl FnMut()) -> f64 {
    f(); let mut b = f64::MAX;
    for _ in 0..n { let s = Instant::now(); f(); b = b.min(s.elapsed().as_secs_f64()); }
    b
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    let mel = Tensor::randn(0f32, 1., (1, 80, 3000), &d)?.contiguous()?;
    let w1 = Tensor::randn(0f32, 1., (384, 80, 3), &d)?.contiguous()?;
    let b1 = Tensor::randn(0f32, 1., 384, &d)?;
    let c1 = Conv1d::new(w1, Some(b1), Conv1dConfig { padding: 1, ..Default::default() });
    let h = c1.forward(&mel)?;
    let w2 = Tensor::randn(0f32, 1., (384, 384, 3), &d)?.contiguous()?;
    let b2 = Tensor::randn(0f32, 1., 384, &d)?;
    let c2 = Conv1d::new(w2, Some(b2), Conv1dConfig { padding: 1, stride: 2, ..Default::default() });

    let t1 = t(5, || { std::hint::black_box(c1.forward(&mel).unwrap()); });
    let t2 = t(5, || { std::hint::black_box(c2.forward(&h).unwrap()); });
    let f1 = 2.0*80.0*384.0*3.0*3000.0;
    let f2 = 2.0*384.0*384.0*3.0*1500.0;
    println!("conv1 (80->384,k3,s1)  {:6.2} ms  {:5.0} GFLOP/s", t1*1e3, f1/t1/1e9);
    println!("conv2 (384->384,k3,s2) {:6.2} ms  {:5.0} GFLOP/s", t2*1e3, f2/t2/1e9);

    // What does the SAME arithmetic cost as a plain GEMM of equal size?
    let a = Tensor::randn(0f32, 1., (3000, 240), &d)?.contiguous()?;
    let b = Tensor::randn(0f32, 1., (240, 384), &d)?.contiguous()?;
    let g1 = t(5, || { std::hint::black_box(a.matmul(&b).unwrap()); });
    let a2 = Tensor::randn(0f32, 1., (1500, 1152), &d)?.contiguous()?;
    let b2m = Tensor::randn(0f32, 1., (1152, 384), &d)?.contiguous()?;
    let g2 = t(5, || { std::hint::black_box(a2.matmul(&b2m).unwrap()); });
    println!("\nsame FLOPs as GEMM (what im2col+matmul would cost, minus the gather):");
    println!("  (3000,240)@(240,384)   {:6.2} ms  {:5.0} GFLOP/s", g1*1e3, f1/g1/1e9);
    println!("  (1500,1152)@(1152,384) {:6.2} ms  {:5.0} GFLOP/s", g2*1e3, f2/g2/1e9);
    println!("\n  headroom: conv1 {:.1}x, conv2 {:.1}x", t1/g1, t2/g2);
    Ok(())
}
