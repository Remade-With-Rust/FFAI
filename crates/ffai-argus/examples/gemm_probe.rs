//! Where does Argus's CPU time actually go — our code, or the tensor spine?
//!
//! `ffai bench vlm` says the engine is 2.4x slower than the PyTorch reference
//! at matched weights and matched decode config. The obvious structural
//! suspect — running the vision tower 17 times at batch 1 instead of once at
//! batch 17 — was measured and **refuted**: chunking gains 1.07x at best
//! (`tile_batching_ab`). So the gap is not the shape of our calls.
//!
//! That leaves the multiply itself. `SigLIP`'s forward is dominated by four
//! dense products per layer, and if candle's CPU GEMM runs at a fraction of
//! MKL's throughput then the gap is inherited from the tensor spine and no
//! amount of Argus-level restructuring will close it.
//!
//! This prices exactly that, on the shapes the tower actually uses, so the
//! claim "the gap is GEMM" becomes a number instead of an assertion. Compare
//! against the matching PyTorch probe printed at the end.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example gemm_probe
//! ```

use candle_core::{DType, Device, Tensor};
use std::time::Instant;

/// `(m, k) x (k, n)` — one product, `2*m*k*n` flops.
fn probe(device: &Device, m: usize, k: usize, n: usize, iters: usize) -> f64 {
    let a = Tensor::rand(-1.0f32, 1.0f32, (m, k), device).expect("a");
    let b = Tensor::rand(-1.0f32, 1.0f32, (k, n), device).expect("b");
    // Warm-up, discarded: the first product pays for candle's lazily-built
    // rayon pool, and charging that to the measurement inflates it.
    let _ = a.matmul(&b).expect("warm");

    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        let c = a.matmul(&b).expect("matmul");
        // Force materialisation: candle is eager, but reading one element
        // guarantees the work is not somehow deferred behind a lazy view.
        let _ = c.get(0).expect("row");
        best = best.min(t.elapsed().as_secs_f64());
    }
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    flops / best / 1e9
}

fn main() {
    let device = Device::Cpu;
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset (all cores)".into());
    println!("candle {} CPU f32 GEMM — RAYON_NUM_THREADS={threads}\n", env!("CARGO_PKG_VERSION"));

    // The shapes SigLIP-base actually runs, per tile: 1024 patches, 768 hidden,
    // 3072 MLP. qkv/proj are (1024,768)x(768,768); the MLP is the wide pair.
    let cases: &[(&str, usize, usize, usize)] = &[
        ("attn qkv/proj  (1024,768)x(768,768)", 1024, 768, 768),
        ("mlp up         (1024,768)x(768,3072)", 1024, 768, 3072),
        ("mlp down       (1024,3072)x(3072,768)", 1024, 3072, 768),
        ("batch-8 qkv    (8192,768)x(768,768)", 8192, 768, 768),
    ];
    println!("{:<40} {:>10}", "shape", "GFLOP/s");
    for &(name, m, k, n) in cases {
        println!("{name:<40} {:>10.1}", probe(&device, m, k, n, 20));
    }

    println!(
        "\nCompare with PyTorch on the same box and the same shapes:\n\
         \n  .venv-argus/Scripts/python.exe -c \"\
         import torch,time\\n\
         torch.set_num_threads(torch.get_num_threads())\\n\
         for m,k,n in [(1024,768,768),(1024,768,3072),(1024,3072,768),(8192,768,768)]:\\n\
         \\x20 a=torch.rand(m,k);b=torch.rand(k,n);a@b\\n\
         \\x20 t=min((lambda:(lambda s:(a@b,time.perf_counter()-s)[1])(time.perf_counter()))() for _ in range(20))\\n\
         \\x20 print(f'{{m}}x{{k}}x{{n}}: {{2*m*k*n/t/1e9:.1f}} GFLOP/s')\"\n"
    );
    println!(
        "If candle is well under PyTorch here, the 2.4x is INHERITED from the \n\
         tensor spine — it is a candle CPU-backend number, not an Argus one, and \n\
         restructuring Argus cannot fix it. Note also that `DType::F32` is what \n\
         Arm 2 pins, so this is the matched comparison, not a quantization story."
    );
    let _ = DType::F32;
}
