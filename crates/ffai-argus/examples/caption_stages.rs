//! Where does a whole caption go, stage by stage?
//!
//! # Why this is not `vision_inline_prof`
//!
//! `vision_inline_prof` profiles ONE tower forward, op by op. That is the right
//! instrument for optimising a layer, and the wrong one for deciding WHAT to
//! optimise: it can only ever tell you which op inside the tower is largest,
//! never whether the tower is where the caption's time actually goes.
//!
//! Every optimisation this crate has landed recently was chosen from inside the
//! tower. That is a sampling bias — the tower is simply the thing that had a
//! profiler attached. Preprocessing cuts 17 tiles with two Lanczos resizes
//! each, the connector runs a pixel shuffle and a projection, and prefill pushes
//! 1142 tokens through 30 decoder layers. None of those appear in a per-op
//! vision profile at all.
//!
//! So: one whole caption, the real engine, every stage the trace reports, with
//! the tower's per-tile spread included — because 17 tiles that should cost the
//! same and do not is itself a finding.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example caption_stages
//! ```
use ffai_core::engine::{VlmEngine, VlmOptions};
use ffai_core::types::{ImageBuffer, PixelFormat};
use std::path::Path;

const IMG: usize = 384;

fn reference_image() -> ImageBuffer {
    let mut data = vec![0u8; IMG * IMG * 3];
    let mut i = 0;
    for y in 0..IMG {
        let fy = y as f64 / IMG as f64;
        for x in 0..IMG {
            let fx = x as f64 / IMG as f64;
            let r = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * fx).sin();
            let g = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * fy + 1.0).sin();
            let b = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * (fx + fy) + 2.0).sin();
            data[i] = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            data[i + 1] = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            data[i + 2] = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            i += 3;
        }
    }
    ImageBuffer { width: IMG as u32, height: IMG as u32, format: PixelFormat::Rgb8, data }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifests = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let engine = ffai_argus::SmolVlm::with_manifest_dir(manifests);
    let img = reference_image();
    let opts = VlmOptions { max_new_tokens: Some(32), ..VlmOptions::default() };

    // Warm: the first caption pays lazy init that no later one does, and
    // attributing that to "preprocess" would send the next round of work at
    // the wrong stage entirely.
    let _ = engine.describe_image_traced(&img, &opts)?;

    let (text, t) = engine.describe_image_traced(&img, &opts)?;
    let steps: f64 = t.step_ms.iter().sum();
    let total = t.preprocess_ms
        + t.tower_ms
        + t.assemble_ms
        + t.prefill_ms
        + steps
        + t.detokenize_ms;

    let mut rows = vec![
        ("preprocess (17 tiles, Lanczos)", t.preprocess_ms),
        ("tower (SigLIP + connector)", t.tower_ms),
        ("assemble (template/tokenize/splice)", t.assemble_ms),
        ("prefill (1 pass, whole prompt)", t.prefill_ms),
        ("generate (all tokens)", steps),
        ("detokenize", t.detokenize_ms),
    ];
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));

    println!("\n  ONE CAPTION — {} tiles, {} prompt tokens, {} generated\n",
        t.tiles, t.prompt_tokens, t.step_ms.len());
    println!("  {:<38} {:>9} {:>8}", "stage", "ms", "share");
    println!("  {:-<38} {:->9} {:->8}", "", "", "");
    for (name, ms) in &rows {
        println!("  {name:<38} {ms:>9.1} {:>7.1}%", 100.0 * ms / total);
    }
    println!("  {:-<38} {:->9} {:->8}", "", "", "");
    println!("  {:<38} {total:>9.1}", "TOTAL");

    // Per-tile spread. The tiles are all the same shape, so anything but a flat
    // line here is scheduling, not work.
    let mut per = t.tower_per_tile_ms.clone();
    if !per.is_empty() {
        per.sort_by(f64::total_cmp);
        let sum: f64 = per.iter().sum();
        println!(
            "\n  tower per tile: min {:.0}  median {:.0}  max {:.0} ms  (sum {:.0}, spread {:.2}x)",
            per[0],
            per[per.len() / 2],
            per[per.len() - 1],
            sum,
            per[per.len() - 1] / per[0].max(1e-9)
        );
        println!(
            "  {} tiles run on {} workers, so the sum EXCEEDS the wall time by design.",
            per.len(),
            std::thread::available_parallelism().map_or(1, std::num::NonZero::get) / 4
        );
    }
    println!("\n  caption: {text:?}");
    Ok(())
}
