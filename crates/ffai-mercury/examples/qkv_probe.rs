//! Three-probe re-test of the q/k/v fusion refutation.
//! Axis 1: fused vs separate. Axis 2: ORIENTATION. Axis 3: does the
//! transposed form hand the kernel K's layout for free?
use std::time::Instant;
use ffai_core::candle::{Device, Tensor};
fn t(n: usize, mut f: impl FnMut()) -> f64 {
    f(); let mut b = f64::MAX;
    for _ in 0..n { let s = Instant::now(); f(); b = b.min(s.elapsed().as_secs_f64()); }
    b
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    let (l, n) = (1500usize, 384usize);
    let x = Tensor::randn(0f32, 1., (l, n), &d)?.contiguous()?;
    let w = Tensor::randn(0f32, 1., (n, n), &d)?.contiguous()?;
    let w3 = Tensor::randn(0f32, 1., (n, 3 * n), &d)?.contiguous()?;
    let w3t = Tensor::randn(0f32, 1., (3 * n, n), &d)?.contiguous()?;

    // A: what ships today -- 3 separate (1500,384)@(384,384)
    let a = t(7, || {
        for _ in 0..3 { std::hint::black_box(x.matmul(&w).unwrap()); }
    });
    // plus the K transpose the kernel needs: (1,6,1500,64)->(1,6,64,1500)
    let k4 = x.reshape((1, l, 6, 64))?.transpose(1, 2)?.contiguous()?;
    let ktr = t(7, || { std::hint::black_box(k4.transpose(2, 3).unwrap().contiguous().unwrap()); });

    // B: fused, SAME orientation -- (1500,384)@(384,1152), split is strided
    let b = t(7, || { std::hint::black_box(x.matmul(&w3).unwrap()); });
    let fused = x.matmul(&w3)?;
    let split = t(7, || {
        for i in 0..3 { std::hint::black_box(fused.narrow(1, i*n, n).unwrap().contiguous().unwrap()); }
    });

    // C: fused, TRANSPOSED -- (1152,384)@(384,1500). Splits along dim 0 are
    // CONTIGUOUS, and K lands as (384,1500) = (6,64,1500), the kernel layout.
    let xt = x.t()?.contiguous()?;
    let xt_cost = t(7, || { std::hint::black_box(x.t().unwrap().contiguous().unwrap()); });
    let c = t(7, || { std::hint::black_box(w3t.matmul(&xt).unwrap()); });

    println!("A  3 separate matmuls      {:6.2} ms", a*1e3);
    println!("   + K transpose           {:6.2} ms   => {:6.2} ms", ktr*1e3, (a+ktr)*1e3);
    println!("B  fused (1500,384)@(384,1152) {:6.2} ms", b*1e3);
    println!("   + 3 strided splits      {:6.2} ms   => {:6.2} ms", split*1e3, (b+split)*1e3);
    println!("C  x^T once                {:6.2} ms", xt_cost*1e3);
    println!("   fused (1152,384)@(384,1500) {:6.2} ms  <- splits CONTIGUOUS, K free", c*1e3);
    println!("                                       => {:6.2} ms", (xt_cost+c)*1e3);
    Ok(())
}
