//! Is candle's GEMM slow at the TEXT tower's shapes, or is prefill's cost
//! somewhere other than the matmuls?
//!
//! # Why ask
//!
//! `gemm_probe` measured candle at **589-702 GF/s** on vision shapes — parity
//! with PyTorch, which is why "candle's GEMM is slow" was refuted in §19.
//! `text_scaling` then measured the whole text tower at **61-90 GF/s**, and at
//! seq 128 — where the score matrix is 0.6 MB and attention cannot be the
//! explanation — it is still 84.6 GF/s.
//!
//! Those two numbers cannot both describe the same matmul implementation
//! unless the SHAPE is what differs. This prices the exact shapes each tower
//! runs, side by side, so the answer is a measurement rather than a story:
//!
//! * if the text shapes also reach ~660 GF/s, the matmuls are fine and
//!   prefill's cost is in the ops BETWEEN them (norms, RoPE, repeat_kv, the
//!   KV-cache concat) — and this file has refuted a hypothesis;
//! * if the text shapes are much slower, the lever is the GEMM path itself,
//!   on 25 % of a caption.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example gemm_shapes
//! ```
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn bench(dev: &Device, m: usize, k: usize, n: usize, batch: usize) -> (f64, f64) {
    let a = Tensor::zeros((batch, m, k), DType::F32, dev).expect("a");
    let b = Tensor::zeros((batch, k, n), DType::F32, dev).expect("b");
    let _ = a.matmul(&b).expect("warm");
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        let c = a.matmul(&b).expect("matmul");
        std::hint::black_box(c.dims());
        best = best.min(t.elapsed().as_secs_f64());
    }
    let gflop = 2.0 * (batch * m * k * n) as f64 / 1e9;
    (best * 1e3, gflop / best)
}

fn main() {
    let dev = Device::Cpu;
    println!("candle matmul, best of 5, f32\n");
    println!("  {:<34} {:>10} {:>10}", "shape", "ms", "GF/s");
    println!("  {:-<34} {:->10} {:->10}", "", "", "");

    let cases: &[(&str, usize, usize, usize, usize)] = &[
        // ---- VISION tower (SigLIP-base): the shapes already at parity -------
        ("VISION qkv    1024x768 @ 768x2304", 1024, 768, 2304, 1),
        ("VISION fc1    1024x768 @ 768x3072", 1024, 768, 3072, 1),
        ("VISION fc2    1024x3072 @ 3072x768", 1024, 3072, 768, 1),
        ("VISION q.k^T  x12 1024x64 @ 64x1024", 1024, 64, 1024, 12),
        // ---- TEXT tower (SmolLM2-135M): hidden 576, inter 1536, 9/3 heads ---
        ("TEXT   qkv    1142x576 @ 576x960", 1142, 576, 960, 1),
        ("TEXT   proj   1142x576 @ 576x576", 1142, 576, 576, 1),
        ("TEXT   mlp-up 1142x576 @ 576x1536", 1142, 576, 1536, 1),
        ("TEXT   mlp-dn 1142x1536 @ 1536x576", 1142, 1536, 576, 1),
        ("TEXT   q.k^T  x9 1142x64 @ 64x1142", 1142, 64, 1142, 9),
        ("TEXT   lm_head 1x576 @ 576x49280", 1, 576, 49280, 1),
        // ---- the DECODE step: m=1, the shape 32 generated tokens run --------
        ("DECODE qkv    1x576 @ 576x960", 1, 576, 960, 1),
        ("DECODE mlp-up 1x576 @ 576x1536", 1, 576, 1536, 1),
    ];
    for &(name, m, k, n, batch) in cases {
        let (ms, gfs) = bench(&dev, m, k, n, batch);
        println!("  {name:<34} {ms:>10.2} {gfs:>10.1}");
    }

    println!();
    println!("  If the TEXT rows land near the VISION rows, the matmuls are NOT the");
    println!("  gap and prefill's 61-90 GF/s is being spent between them — on the");
    println!("  norms, RoPE, repeat_kv and the KV-cache concat. That would make the");
    println!("  lever the same class as the vision win: kernels candle runs on one");
    println!("  core, not arithmetic.");
}
