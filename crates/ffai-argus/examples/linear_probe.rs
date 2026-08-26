//! Our `linear()` costs 6.35 ms where the raw GEMM costs 3.42 ms. Why?
//!
//! `text::linear` is `x.reshape((b*s,i)).matmul(&w.t()).reshape((b,s,o))`.
//! The reshapes are contiguous and free, so the suspect is `w.t()` — a
//! transposed VIEW of the checkpoint's `(out, in)` weight. `transpose_probe`
//! found `.t()` free at m=1 and near-free at m=1142, so this re-tests it at the
//! exact shape and against a weight pre-transposed ONCE at load.
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
    for &(name, m, k, n) in &[
        ("mlp gate/up  1142x576 -> 1536", 1142usize, 576usize, 1536usize),
        ("mlp down    1142x1536 -> 576", 1142, 1536, 576),
        ("q proj       1142x576 -> 576", 1142, 576, 576),
    ] {
        let x3 = Tensor::rand(-1.0f32, 1.0, (1, m, k), &d)?;
        let w_out_in = Tensor::rand(-1.0f32, 1.0, (n, k), &d)?;   // checkpoint layout
        let w_in_out = w_out_in.t()?.contiguous()?;               // pre-transposed once
        let gf = 2.0 * (m * k * n) as f64 / 1e9;

        let a = t(&mut || {
            let (b, s, i) = x3.dims3()?;
            x3.reshape((b * s, i))?.matmul(&w_out_in.t()?)?.reshape((b, s, n))
        });
        let b_ = t(&mut || {
            let (b, s, i) = x3.dims3()?;
            x3.reshape((b * s, i))?.matmul(&w_in_out)?.reshape((b, s, n))
        });
        let c = t(&mut || x3.reshape((m, k))?.matmul(&w_in_out));

        println!("{name}");
        println!("   {:<44} {a:>8.2} ms  {:>6.0} GF/s", "ours: reshape + matmul(w.t()) + reshape", gf / (a / 1e3));
        println!("   {:<44} {b_:>8.2} ms  {:>6.0} GF/s", "same, weight PRE-TRANSPOSED at load", gf / (b_ / 1e3));
        println!("   {:<44} {c:>8.2} ms  {:>6.0} GF/s", "bare 2D matmul, pre-transposed", gf / (c / 1e3));
        println!("   {:<44} {:>8.2}x", "-> pre-transposing is worth", a / b_);
        println!();
    }
    Ok(())
}
