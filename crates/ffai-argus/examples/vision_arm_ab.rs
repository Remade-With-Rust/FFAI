//! Interleaved ABBA A/B for ONE vision arm, both halves in one process.
//!
//! # Why this shape
//!
//! This box has read the same vision code at 8172, 9380, 10327, 12029, 13815
//! and 14382 ms inside a single session — a **1.76x spread for identical
//! code**. A "before" from one build against an "after" from another cannot
//! resolve a 5 % effect; it resolves which half of the afternoon each ran in.
//!
//! So every arm is a runtime toggle, both settings live in one binary, and this
//! alternates them **ABBA** over the same tower and the same input. A thermal
//! ramp or a background compile lands on A and B equally.
//!
//! Every `off` arm is a real implementation that computes the same tensor — the
//! null-arm rule. An arm whose "off" branch merely skips work measures nothing.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example vision_arm_ab -- head_attn
//! ```
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::time::Instant;

const ROUNDS: usize = 6;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arm = std::env::args().nth(1).unwrap_or_else(|| "head_attn".into());
    // `attn_block` is a SIZE, not a flag: its ON arm is the block size under
    // test and its OFF arm is 0, the unblocked path. Pass the size as the
    // second argument to sweep it.
    let block: usize =
        std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(256);
    let set_block = move |on: bool| {
        let prev = ffai_argus::siglip::set_attn_block(if on { block } else { 0 });
        prev != 0
    };
    if arm == "attn_block" {
        println!("  blocked attention: ON = BLOCK {block}, OFF = BLOCK 0 (unblocked)");
    }

    let toggle: Box<dyn Fn(bool) -> bool> = match arm.as_str() {
        "fuse_bias" => Box::new(ffai_argus::siglip::set_fuse_bias),
        "head_attn" => Box::new(ffai_argus::siglip::set_head_attn),
        "inplace_softmax" => Box::new(ffai_argus::siglip::set_inplace_softmax),
        "late_norm" => Box::new(ffai_argus::siglip::set_late_normalize),
        "fused_ln" => Box::new(ffai_argus::siglip::set_fused_ln),
        "attn_block" => Box::new(set_block),
        other => return Err(format!("unknown arm {other}").into()),
    };

    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))?;
    let snap = std::fs::read_dir(std::path::Path::new(&home).join(
        ".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots",
    ))?
    .flatten()
    .next()
    .ok_or("no snapshot")?
    .path();
    let device = Device::Cpu;
    let cj = std::fs::read_to_string(snap.join("config.json"))?;
    let (cfg, _s) = ffai_argus::vision::vision_config_from_json(&cj)?;
    // SAFETY: mapped file owned by the model cache.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            std::slice::from_ref(&snap.join("model.safetensors")),
            DType::F32,
            &device,
        )?
    };
    let tower = ffai_argus::siglip::VisionTower::new(&cfg, vb.pp("model.vision_model"))?;
    let px = Tensor::rand(-1.0f32, 1.0, (1, 3, cfg.image_size, cfg.image_size), &device)?;

    // Correctness BEFORE speed: the two arms must agree, or the faster one is
    // not an optimisation, it is a different model.
    let on = {
        let p = toggle(true);
        let o = tower.forward(&px)?;
        toggle(p);
        o
    };
    let off = {
        let p = toggle(false);
        let o = tower.forward(&px)?;
        toggle(p);
        o
    };
    let diff = (&on - &off)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let scale = on.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("  arm `{arm}`: max |on - off| = {diff:.3e}  (values up to {scale:.3e})");
    if diff > 1e-3 * scale.max(1.0) {
        return Err(format!("arms disagree by {diff:.3e} — not a like-for-like A/B").into());
    }

    let mut run = |on: bool| -> Result<f64, Box<dyn std::error::Error>> {
        let prev = toggle(on);
        let t = Instant::now();
        let out = tower.forward(&px)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(out.dims());
        toggle(prev);
        Ok(ms)
    };

    for on in [true, false] {
        run(on)?; // warm both
    }

    let (mut a, mut b) = (Vec::new(), Vec::new());
    for r in 0..ROUNDS {
        // The order flips every round, so a monotone drift cannot favour
        // whichever arm happens to go first.
        if r % 2 == 0 {
            a.push(run(true)?);
            b.push(run(false)?);
            b.push(run(false)?);
            a.push(run(true)?);
        } else {
            b.push(run(false)?);
            a.push(run(true)?);
            a.push(run(true)?);
            b.push(run(false)?);
        }
        println!("  round {}/{ROUNDS}", r + 1);
    }

    let stat = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        (v[0], v[v.len() / 2])
    };
    let (amin, amed) = stat(&mut a);
    let (bmin, bmed) = stat(&mut b);
    println!("\n  ONE TILE, 12 layers — {} samples per arm\n", a.len());
    println!("  {:<18} {:>10} {:>10}", "arm", "min ms", "median ms");
    println!("  {:-<18} {:->10} {:->10}", "", "", "");
    println!("  {:<18} {bmin:>10.1} {bmed:>10.1}", format!("{arm} OFF"));
    println!("  {:<18} {amin:>10.1} {amed:>10.1}", format!("{arm} ON"));
    println!("  {:-<18} {:->10} {:->10}", "", "", "");
    println!("  {:<18} {:>9.3}x {:>9.3}x", "speedup", bmin / amin, bmed / amed);
    println!(
        "\n  Per caption (17 tiles): {:.2} s -> {:.2} s, saving {:.0} ms",
        bmin * 17.0 / 1e3,
        amin * 17.0 / 1e3,
        (bmin - amin) * 17.0
    );
    Ok(())
}
