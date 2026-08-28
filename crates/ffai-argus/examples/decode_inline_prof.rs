//! Per-op cost inside the real DECODE loop — the half nobody has profiled.
//!
//! # Why decode needs its own instrument
//!
//! `text_inline_prof` profiles PREFILL: one pass over 1142 positions, where
//! every matmul is `M = 1142` and the tower is compute-bound. Decode is a
//! different machine. `M = 1` — every projection degenerates to a
//! matrix-VECTOR product, so arithmetic intensity collapses and the step is
//! bound by **streaming the weights**, not by multiplying them.
//!
//! That means the two phases can want opposite things, and a conclusion drawn
//! on prefill shapes does not transfer. The fused-QKV question is the standing
//! example: it was measured at seq 64 / 512 / 1142 and refuted at all three,
//! but seq 1 was never in that table.
//!
//! # The floor this is measured against
//!
//! SmolLM2-135M in f32 is ~540 MB of weights. Every decode step reads all of
//! them once, so at a realistic ~20 GB/s that is **~27 ms/token before any
//! arithmetic**. Comparing the measured step against that number says whether
//! the remaining cost is bandwidth (nothing to win without changing the dtype)
//! or overhead (winnable).
//!
//! ```sh
//! FFAI_TEXT_PROFILE=1 cargo run --release -p ffai-argus --example decode_inline_prof
//! ```
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::time::Instant;

const PROMPT: usize = 1142;
const STEPS: usize = 24;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))?;
    let snap = std::fs::read_dir(
        std::path::Path::new(&home)
            .join(".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots"),
    )?
    .flatten()
    .next()
    .ok_or("no snapshot")?
    .path();
    let weights = snap.join("model.safetensors");
    let cj = std::fs::read_to_string(snap.join("config.json"))?;
    let d = Device::Cpu;
    let v: serde_json::Value = serde_json::from_str(&cj)?;
    let t = v.get("text_config").unwrap_or(&v);
    let g = |k: &str, dv: u64| t.get(k).and_then(serde_json::Value::as_u64).unwrap_or(dv);
    let gf = |k: &str, dv: f64| t.get(k).and_then(serde_json::Value::as_f64).unwrap_or(dv);
    let heads = g("num_attention_heads", 9) as usize;
    let hidden = g("hidden_size", 576) as usize;
    let cfg = ffai_argus::text::Cfg {
        layers: g("num_hidden_layers", 30) as usize,
        hidden,
        heads,
        kv_heads: g("num_key_value_heads", 3) as usize,
        head_dim: hidden / heads,
        inter: g("intermediate_size", 1536) as usize,
        eps: gf("rms_norm_eps", 1e-5),
        rope_theta: gf("rope_theta", 100_000.0) as f32,
        max_pos: g("max_position_embeddings", 8192) as usize,
    };
    let layers = cfg.layers;
    let inter = cfg.inter;
    let kv_heads = cfg.kv_heads;
    let head_dim = cfg.head_dim;
    // SAFETY: mapped file owned by the model cache, not mutated here.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, &d)?
    };
    let mut tower = ffai_argus::text::TextTower::load(&vb, cfg, &d)?;

    let prompt = Tensor::rand(-1.0f32, 1.0, (1, PROMPT, hidden), &d)?;
    let step = Tensor::rand(-1.0f32, 1.0, (1, 1, hidden), &d)?;

    // Warm, then discard: the first pass pages weights in, and attributing
    // that to "decode" would flatter every later step.
    tower.reset();
    let _ = tower.forward(&prompt, 0)?;
    let _ = tower.forward(&step, PROMPT)?;
    let _ = ffai_argus::text::prof::take();

    // Prefill again to build a real cache, and throw its profile away — we
    // want the STEPS only.
    tower.reset();
    let _ = tower.forward(&prompt, 0)?;
    let _ = ffai_argus::text::prof::take();

    let t0 = Instant::now();
    for i in 0..STEPS {
        let _ = tower.forward(&step, PROMPT + i)?;
    }
    let whole = t0.elapsed().as_secs_f64() * 1e3;
    let rows = ffai_argus::text::prof::take();
    let per = whole / STEPS as f64;
    if rows.is_empty() {
        // THE PROFILER'S OWN TAX. Run this arm too and compare: 480 timer
        // pairs and mutex acquisitions per token is negligible against a
        // 1142-position prefill and easy to assume is negligible here as
        // well — but a decode step is three orders of magnitude smaller, so
        // the same fixed cost is a completely different fraction of it.
        println!("
  DECODE with profiling OFF: {per:.2} ms/token over {STEPS} steps");
        println!("  (compare against the profiled run to price the instrument itself)");
        return Ok(());
    }
    let sum: f64 = rows.iter().map(|r| r.1).sum();
    println!("\nOUR text DECODE, {STEPS} steps at seq 1, cache primed to {PROMPT}, {layers} layers\n");
    println!("  {:<24} {:>10} {:>10} {:>8}", "op (all layers)", "ms total", "ms/token", "share");
    println!("  {:-<24} {:->10} {:->10} {:->8}", "", "", "", "");
    for (n, ms) in &rows {
        println!(
            "  {n:<24} {ms:>10.1} {:>10.2} {:>7.1}%",
            ms / STEPS as f64,
            100.0 * ms / whole
        );
    }
    println!("  {:-<24} {:->10} {:->10} {:->8}", "", "", "", "");
    println!("  {:<24} {sum:>10.1} {:>10.2} {:>7.1}%", "accounted", sum / STEPS as f64, 100.0 * sum / whole);
    println!(
        "  {:<24} {:>10.1} {:>10.2} {:>7.1}%",
        "UNACCOUNTED",
        whole - sum,
        (whole - sum) / STEPS as f64,
        100.0 * (whole - sum) / whole
    );
    println!("  {:<24} {whole:>10.1} {per:>10.2}", "whole decode");

    // ---- the bandwidth floor ------------------------------------------------
    //
    // Per step every weight is read exactly once. Counting them is exact and
    // machine-independent; only the assumed GB/s is not.
    let per_layer = hidden * (heads * head_dim)            // q
        + 2 * hidden * (kv_heads * head_dim)               // k, v
        + hidden * (heads * head_dim)                      // o
        + 3 * hidden * inter                               // gate, up, down
        + 2 * hidden; // the two RMSNorm weights
    let params = layers * per_layer;
    let bytes = params as f64 * 4.0;
    println!("\n  Weight bytes read per token: {:.0} MB ({:.1} M params, f32)", bytes / 1e6, params as f64 / 1e6);
    for bw in [15.0f64, 20.0, 25.0] {
        println!(
            "    at {bw:>4.0} GB/s -> {:>6.2} ms/token floor   (we are at {per:.2}, {:.2}x the floor)",
            bytes / (bw * 1e9) * 1e3,
            per / (bytes / (bw * 1e9) * 1e3)
        );
    }
    println!(
        "\n  If we sit near 1.0x the floor, decode is BANDWIDTH-bound and the only\n  \
         lever is fewer bytes (dtype), not better code. Well above it means overhead."
    );
    Ok(())
}
