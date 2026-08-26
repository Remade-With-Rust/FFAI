//! Does batching tiles help the GEMM, and by how much — the CEILING.
//!
//! `tile_batching_ab` runs whole towers and this box gives it a 4x spread, so
//! its 1.149x is below the noise floor and settles nothing. This asks the much
//! narrower question that batching actually turns on: **does a GEMM with 8x the
//! rows run at a better rate than eight GEMMs of 1x?**
//!
//! That is measurable in a few ms per point and it BOUNDS the win: batching
//! cannot beat the per-row-rate improvement shown here, whatever else it does.
use candle_core::{Device, Tensor};
use std::time::Instant;

fn t(f: &mut dyn FnMut() -> candle_core::Result<Tensor>) -> f64 {
    let _ = f().expect("warm");
    let mut b = f64::INFINITY;
    for _ in 0..7 {
        let s = Instant::now();
        let o = f().expect("op");
        std::hint::black_box(o.dims());
        b = b.min(s.elapsed().as_secs_f64());
    }
    b * 1e3
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::Cpu;
    // The four linears a SigLIP layer runs, at (in, out).
    let shapes: &[(&str, usize, usize)] = &[
        ("qkv    768->2304", 768, 2304),
        ("fc1    768->3072", 768, 3072),
        ("fc2   3072->768", 3072, 768),
        ("out    768->768", 768, 768),
    ];
    println!("GEMM rate vs batched rows (1 tile = 1024 rows)\n");
    println!("  {:<20} {:>8} {:>10} {:>10} {:>10}", "linear", "tiles", "ms", "GF/s", "vs 1 tile");
    println!("  {:-<20} {:->8} {:->10} {:->10} {:->10}", "", "", "", "", "");
    for &(name, k, n) in shapes {
        let w = Tensor::rand(-1.0f32, 1.0, (k, n), &d)?;
        let mut base = 0f64;
        for tiles in [1usize, 2, 4, 8] {
            let m = 1024 * tiles;
            let x = Tensor::rand(-1.0f32, 1.0, (m, k), &d)?;
            let ms = t(&mut || x.matmul(&w));
            let gfs = 2.0 * (m * k * n) as f64 / 1e9 / (ms / 1e3);
            if tiles == 1 {
                base = gfs;
            }
            println!("  {name:<20} {tiles:>8} {ms:>10.2} {gfs:>10.0} {:>9.2}x", gfs / base);
        }
    }
    println!("\n  The `vs 1 tile` column IS the batching ceiling for the linears —");
    println!("  they are 51 % of a layer. If it is ~1.0x, batching cannot pay for");
    println!("  its footprint (384 MiB of attention tensor at chunk 8) and the");
    println!("  question is closed regardless of what a noisy whole-tower A/B says.");
    Ok(())
}
