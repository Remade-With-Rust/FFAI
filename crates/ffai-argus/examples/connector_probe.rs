//! The connector is 5.8 % of a tile (78.9 ms, `tile_batching_ab`). Where?
//!
//! It is a pixel-shuffle (pure reshapes) plus one matmul, so 78.9 ms is a lot.
//! Three candidates, all visible in `vision::connect`: TWO separate
//! `.contiguous()` copies in the shuffle, a `to_dtype` on the projection weight
//! per call, and `broadcast_matmul` — which `text.rs` already measured at 33x
//! slower than a plain 2D matmul when the batch dim is 1.
use candle_core::{Device, Tensor};
use std::time::Instant;

fn t(f: &mut dyn FnMut() -> candle_core::Result<Tensor>) -> f64 {
    let _ = f().expect("warm");
    let mut b = f64::INFINITY;
    for _ in 0..10 {
        let s = Instant::now();
        let o = f().expect("op");
        std::hint::black_box(o.dims());
        b = b.min(s.elapsed().as_secs_f64());
    }
    b * 1e3
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    const SIDE: usize = 32;
    const DIM: usize = 768;
    const S: usize = 4;
    let hidden = Tensor::rand(-1.0f32, 1.0, (1, SIDE * SIDE, DIM), &d)?;
    let proj = Tensor::rand(-1.0f32, 1.0, (576, DIM * S * S), &d)?;
    let shuffled = Tensor::rand(-1.0f32, 1.0, (1, (SIDE / S) * (SIDE / S), DIM * S * S), &d)?;

    println!("connector: (1,{},{DIM}) -> (1,{},{})\n", SIDE * SIDE, (SIDE / S) * (SIDE / S), DIM * S * S);

    // --- the shuffle, as written: two transposes, two contiguous copies ----
    let two = t(&mut || {
        let x = hidden.reshape((1, SIDE, SIDE, DIM))?;
        let x = x.reshape((1, SIDE, SIDE / S, DIM * S))?;
        let x = x.transpose(1, 2)?.contiguous()?;
        let x = x.reshape((1, SIDE / S, SIDE / S, DIM * S * S))?;
        let x = x.transpose(1, 2)?.contiguous()?;
        x.reshape((1, (SIDE / S) * (SIDE / S), DIM * S * S))
    });
    println!("  {:<46} {two:>8.3} ms", "shuffle: 2 transposes + 2 contiguous [ours]");

    // --- the matmul, three ways -------------------------------------------
    let bm = t(&mut || {
        let w = proj.to_dtype(shuffled.dtype())?.t()?;
        shuffled.broadcast_matmul(&w)
    });
    println!("  {:<46} {bm:>8.3} ms", "proj: to_dtype + broadcast_matmul [ours]");

    let bm2 = t(&mut || {
        let w = proj.t()?;
        shuffled.broadcast_matmul(&w)
    });
    println!("  {:<46} {bm2:>8.3} ms", "proj: broadcast_matmul, no to_dtype");

    let flat = t(&mut || {
        let (b, n, k) = shuffled.dims3()?;
        shuffled.reshape((b * n, k))?.matmul(&proj.t()?)?.reshape((b, n, 576))
    });
    println!("  {:<46} {flat:>8.3} ms", "proj: flatten to 2D + matmul");

    println!("\n  shuffle+proj as written : {:>8.3} ms", two + bm);
    println!("  shuffle+proj best above : {:>8.3} ms   ({:.2}x)", two + flat, (two + bm) / (two + flat));
    Ok(())
}
