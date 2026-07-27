//! What does encoder attention cost BESIDE the fused kernel?
//! Stage is 83 ms for 4 layers; the kernel probe says ~10.4 ms/layer.
use std::time::Instant;
use ffai_core::candle::{Device, Tensor};
fn t(n: usize, mut f: impl FnMut()) -> f64 {
    f(); let mut b = f64::MAX;
    for _ in 0..n { let s = Instant::now(); f(); b = b.min(s.elapsed().as_secs_f64()); }
    b
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    let x = Tensor::randn(0f32, 1., (1500, 384), &d)?.contiguous()?;
    let w = Tensor::randn(0f32, 1., (384, 384), &d)?.contiguous()?;
    let proj = t(9, || { std::hint::black_box(x.matmul(&w).unwrap()); });
    let k = x.matmul(&w)?;
    // split_heads: (1,1500,384) -> (1,6,1500,64)
    let sh = t(9, || {
        std::hint::black_box(k.reshape((1, 1500, 6, 64)).unwrap().transpose(1, 2).unwrap());
    });
    let heads = k.reshape((1, 1500, 6, 64))?.transpose(1, 2)?;
    let cont = t(9, || { std::hint::black_box(heads.contiguous().unwrap()); });
    let kt = t(9, || { std::hint::black_box(heads.transpose(2, 3).unwrap().contiguous().unwrap()); });
    println!("per layer:");
    println!("  one projection  (1500,384)@(384,384)   {:6.2} ms  (x4 = {:6.2})", proj*1e3, proj*4e3);
    println!("  split_heads (view only)                {:6.2} ms", sh*1e3);
    println!("  .contiguous() on (1,6,1500,64)         {:6.2} ms", cont*1e3);
    println!("  transpose(2,3) + .contiguous()  [K]    {:6.2} ms  <- strided gather", kt*1e3);
    let glue = proj*4.0 + cont*2.0 + kt;
    println!("\n  projections+copies/layer {:6.2} ms  x4 layers = {:6.2} ms", glue*1e3, glue*4e3);
    println!("  fused kernel  x4 layers  = {:6.2} ms", 10.44*4.0);
    println!("  => accounted {:6.2} ms vs stage 83 ms", glue*4e3 + 41.8);
    Ok(())
}
