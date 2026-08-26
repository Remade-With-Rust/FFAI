//! Why is prefill 25 % of a caption when its FLOPs say 0.5 s?
//!
//! # The arithmetic that does not close
//!
//! `stage_split` measures prefill at **4282 ms** for 1142 tokens. Priced by
//! FLOPs the same work is ~333 GFLOP (30 layers x ~11.1 GFLOP at hidden 576,
//! 9 heads, 3 kv-heads, intermediate 1536), and this box does 660 GF/s on
//! candle's GEMM — parity with PyTorch, measured in `gemm_probe`. That is
//! **~0.5 s against 4282 ms, an 8.5x gap.**
//!
//! A ratio that implies an implausible per-unit cost indicts the
//! DECOMPOSITION, not the code (`rusty-curiosity`). So rather than guess which
//! term is missing, this sweeps ONE variable — sequence length — and reads the
//! exponent off the result:
//!
//! | if cost scales as | the dominant term is |
//! |---|---|
//! | `O(n)` | weights: streaming 30 layers of parameters, memory-bound |
//! | `O(n^2)` | attention: the score matrix and its softmax |
//!
//! Both are actionable and they are actionable in OPPOSITE directions, which
//! is exactly why guessing was not good enough. `ms/token` is printed
//! alongside: flat means linear, rising means quadratic.
//!
//! Deterministic where it can be: the per-length FLOP and byte counts come
//! from the config, so the model column is reproducible anywhere. Only the
//! wall-clock column is machine-dependent, and it is min-of-3.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example text_scaling
//! ```
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))?;
    let snaps = std::path::Path::new(&home)
        .join(".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots");
    let snap = std::fs::read_dir(&snaps)?
        .flatten()
        .next()
        .ok_or("no snapshot in the HF cache")?
        .path();
    let device = Device::Cpu;
    let config_json = std::fs::read_to_string(snap.join("config.json"))?;
    let mut dec = ffai_argus::decode::TextDecoder::load(
        &snap.join("model.safetensors"),
        &config_json,
        &device,
    )?;

    // Geometry, read from the checkpoint rather than assumed.
    let v: serde_json::Value = serde_json::from_str(&config_json)?;
    let t = v.get("text_config").unwrap_or(&v);
    let g = |k: &str| t.get(k).and_then(serde_json::Value::as_u64).unwrap_or(0);
    let (layers, hidden, heads, kv_heads, inter) = (
        g("num_hidden_layers"),
        g("hidden_size"),
        g("num_attention_heads"),
        g("num_key_value_heads").max(1),
        g("intermediate_size"),
    );
    let hdim = hidden / heads.max(1);
    println!(
        "text tower: {layers} layers, hidden {hidden}, {heads} heads ({kv_heads} kv), \
         head_dim {hdim}, inter {inter}\n"
    );

    println!("  {:>6}  {:>10}  {:>10}  {:>9}  {:>9}  {:>9}", "seq", "ms", "ms/token", "GFLOP", "score MB", "GF/s");
    println!("  {:->6}  {:->10}  {:->10}  {:->9}  {:->9}  {:->9}", "", "", "", "", "", "");

    let mut prev: Option<(f64, f64)> = None; // (seq, ms)
    for &seq in &[128usize, 256, 512, 1024, 1142] {
        let embeds = Tensor::zeros((1, seq, hidden as usize), DType::F32, &device)?;
        let n = seq as u64;

        // Per-layer FLOPs, from the config. Gated MLP (gate+up+down).
        let qkv = 2 * n * hidden * (hidden + 2 * kv_heads * hdim);
        let attn = 2 * heads * n * hdim * n * 2; // q.k^T and attn.v
        let proj = 2 * n * hidden * hidden;
        let mlp = 3 * 2 * n * hidden * inter;
        let gflop = layers as f64 * (qkv + attn + proj + mlp) as f64 / 1e9;
        // The score matrix, per layer, once.
        let score_mb = heads as f64 * (n * n) as f64 * 4.0 / 1e6;

        let mut best = f64::INFINITY;
        for _ in 0..3 {
            dec.reset();
            let t0 = Instant::now();
            let out = dec.forward_embeds(&embeds, 0)?;
            std::hint::black_box(out.dims());
            best = best.min(t0.elapsed().as_secs_f64());
        }
        let ms = best * 1e3;
        println!(
            "  {seq:>6}  {ms:>10.1}  {:>10.3}  {gflop:>9.1}  {score_mb:>9.1}  {:>9.1}",
            ms / seq as f64,
            gflop / best,
        );
        if let Some((ps, pms)) = prev {
            let ratio = (ms / pms).log2() / (seq as f64 / ps).log2();
            println!("  {:>6}  {:>10}  exponent vs previous point: n^{ratio:.2}", "", "");
        }
        prev = Some((seq as f64, ms));
    }

    println!();
    println!("  READ IT: an exponent near 1.0 means prefill is LINEAR — the cost is");
    println!("  streaming 30 layers of weights, and the lever is weight traffic");
    println!("  (fewer passes, better reuse, batching). Near 2.0 means attention's");
    println!("  score matrix dominates and the lever is the same one the vision");
    println!("  tower has: do not materialise it.");
    println!();
    println!("  The GF/s column is the honest check on the FLOP model: if it is far");
    println!("  under the ~660 GF/s this box reaches on GEMM, the FLOPs are not what");
    println!("  is being paid for, and no amount of matmul tuning will help.");
    Ok(())
}
