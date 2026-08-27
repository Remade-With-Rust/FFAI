//! Per-op cost measured INSIDE the real prefill, not in isolation.
//!
//! `text_ops_now` summed isolated ops to ~940 ms against a measured 1283 ms.
//! Isolated ops run warm and uncontended; the real forward does neither. This
//! times each op where it actually runs, so the parts must add to the whole.
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::time::Instant;

/// The SAME allocator `ffai` declares — see `examples/alloc_ab.rs`.
///
/// An example is a binary, and a binary with no `#[global_allocator]` gets the
/// system one. Every figure this crate published from an example was therefore
/// taken under an allocator production does not use, and measured **1.15x
/// pessimistic** on prefill (1180.9 ms system vs 1028.4 ms rusty_alloc, 3/3
/// interleaved rounds). A caption allocates a 50 MB score tensor 204 times in
/// the vision tower alone, so this is not a rounding difference.
#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

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
    // SAFETY: mapped file owned by the model cache, not mutated here.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, &d)?
    };
    let mut tower = ffai_argus::text::TextTower::load(&vb, cfg, &d)?;

    const SEQ: usize = 1142;
    let x = Tensor::rand(-1.0f32, 1.0, (1, SEQ, hidden), &d)?;
    // Warm: page the weights in, then discard whatever the warm run recorded.
    tower.reset();
    let _ = tower.forward(&x, 0)?;
    let _ = ffai_argus::text::prof::take();

    tower.reset();
    let t0 = Instant::now();
    let _ = tower.forward(&x, 0)?;
    let whole = t0.elapsed().as_secs_f64() * 1e3;
    let rows = ffai_argus::text::prof::take();
    if rows.is_empty() {
        println!("no samples — run with FFAI_TEXT_PROFILE=1");
        return Ok(());
    }
    let sum: f64 = rows.iter().map(|r| r.1).sum();
    println!("OUR text prefill, seq {SEQ}, {} layers\n", cfg.layers);
    println!("  {:<24} {:>10} {:>8}", "op (all layers)", "ms", "share");
    println!("  {:-<24} {:->10} {:->8}", "", "", "");
    for (n, ms) in &rows {
        println!("  {n:<24} {ms:>10.1} {:>7.1}%", 100.0 * ms / whole);
    }
    println!("  {:-<24} {:->10} {:->8}", "", "", "");
    println!("  {:<24} {sum:>10.1} {:>7.1}%", "accounted", 100.0 * sum / whole);
    println!("  {:<24} {:>10.1} {:>7.1}%", "UNACCOUNTED", whole - sum, 100.0 * (whole - sum) / whole);
    println!("  {:<24} {whole:>10.1}", "whole prefill");
    println!("\n  PyTorch does this same prefill in 675 ms (measured directly).");
    Ok(())
}
