//! Per-op cost measured INSIDE the real tower, not in isolation.
//!
//! `vision_ops_now` sums isolated ops. For the text tower that method missed
//! 18 % of the real forward, and the miss was allocation churn — a fixable
//! win nobody had priced. The vision tower has never had the equivalent check.
//! One tile, kernels parallel (single-tile mode), so the parts must add up.
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::time::Instant;

#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))?;
    let snap = std::fs::read_dir(std::path::Path::new(&home).join(
        ".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots",
    ))?.flatten().next().ok_or("no snapshot")?.path();
    let d = Device::Cpu;
    let cj = std::fs::read_to_string(snap.join("config.json"))?;
    let (vcfg, _s) = ffai_argus::vision::vision_config_from_json(&cj)?;
    // SAFETY: mapped file owned by the model cache.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(
        std::slice::from_ref(&snap.join("model.safetensors")), DType::F32, &d)? };
    let tower = ffai_argus::siglip::VisionTower::new(&vcfg, vb.pp("model.vision_model"))?;
    let x = Tensor::rand(-1.0f32, 1.0, (1, 3, vcfg.image_size, vcfg.image_size), &d)?;

    let _ = tower.forward(&x)?;              // warm
    let _ = ffai_argus::siglip::prof::take();

    let t0 = Instant::now();
    let _ = tower.forward(&x)?;
    let whole = t0.elapsed().as_secs_f64() * 1e3;
    let rows = ffai_argus::siglip::prof::take();
    if rows.is_empty() {
        println!("no samples — run with FFAI_VIS_PROFILE=1");
        return Ok(());
    }
    let sum: f64 = rows.iter().map(|r| r.1).sum();
    println!("ONE TILE through the real tower ({} layers)\n", vcfg.num_hidden_layers);
    println!("  {:<22} {:>9} {:>8}", "op (all layers)", "ms", "share");
    println!("  {:-<22} {:->9} {:->8}", "", "", "");
    for (n, ms) in &rows {
        println!("  {n:<22} {ms:>9.1} {:>7.1}%", 100.0 * ms / whole);
    }
    println!("  {:-<22} {:->9} {:->8}", "", "", "");
    println!("  {:<22} {sum:>9.1} {:>7.1}%", "accounted", 100.0 * sum / whole);
    println!("  {:<22} {:>9.1} {:>7.1}%", "UNACCOUNTED", whole - sum, 100.0 * (whole - sum) / whole);
    println!("  {:<22} {whole:>9.1}", "whole tile");
    let _ = DType::F32;
    Ok(())
}
