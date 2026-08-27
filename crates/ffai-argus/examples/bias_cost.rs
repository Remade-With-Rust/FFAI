//! Is the BIAS add the missing half of every vision matmul?
//!
//! In situ `fc1` costs 15.35 ms/layer; a bare `matmul` of the identical shape
//! costs 8.57 ms. The tower calls `candle_nn::Linear::forward`, which is a
//! matmul **plus `broadcast_add(bias)`** — and candle's binary ops are not
//! parallel. This prices that difference at every projection in the layer.
use candle_core::{Device, Module, Tensor};
use std::time::Instant;

#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

fn t(f: &dyn Fn()) -> f64 {
    f();
    let mut b = f64::INFINITY;
    for _ in 0..7 {
        let s = Instant::now();
        f();
        b = b.min(s.elapsed().as_secs_f64() * 1e3);
    }
    b
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    const SEQ: usize = 1024;
    const HID: usize = 768;
    println!("  {:<34} {:>9} {:>9} {:>9} {:>8}", "projection", "matmul", "+bias", "bias cost", "x12x17");
    println!("  {:-<34} {:->9} {:->9} {:->9} {:->8}", "", "", "", "", "");
    let mut total = 0f64;
    for (name, k, n) in [("qkv  768->2304", HID, 3 * HID), ("fc1  768->3072", HID, 4 * HID),
                         ("fc2  3072->768", 4 * HID, HID), ("out  768->768", HID, HID)] {
        let x = Tensor::rand(-1.0f32, 1.0, (SEQ, k), &d)?;
        let w = Tensor::rand(-1.0f32, 1.0, (n, k), &d)?;
        let b = Tensor::rand(-1.0f32, 1.0, n, &d)?;
        let wt = w.t()?.contiguous()?;
        let lin = candle_nn::Linear::new(w.clone(), Some(b.clone()));
        let bare = t(&|| { std::hint::black_box(x.matmul(&wt).expect("mm").dims()); });
        let full = t(&|| { std::hint::black_box(lin.forward(&x).expect("lin").dims()); });
        let cost = full - bare;
        total += cost;
        println!("  {name:<34} {bare:>8.2}ms {full:>8.2}ms {cost:>8.2}ms {:>7.0}ms",
            cost * 12.0 * 17.0);
    }
    println!("\n  bias adds per layer: {total:.2} ms   ->  x12 layers x17 tiles = {:.1} s",
        total * 12.0 * 17.0 / 1e3);
    println!("  A whole-tile forward measures ~1076 ms, so a caption is ~18.3 s of tower.");
    Ok(())
}
