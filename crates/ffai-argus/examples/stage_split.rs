//! What share of a caption is vision, MEASURED end to end?
//!
//! The cost model prices one tile; the demo priced a caption before round 2
//! landed. Neither answers "what is vision's share now", and every decision
//! about where to optimise next depends on it. Re-anchor before investigating
//! is the rule; this is the instrument for it.
use ffai_core::engine::VlmEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or("root")?;
    let engine = ffai_argus::SmolVlm::with_manifest_dir(root.join("models"));
    // Image and budget are arguments so this can be pointed at the SAME file
    // and the SAME token budget as `corpora/refs/smolvlm_hf_profile.py`. A
    // head-to-head where the two arms caption different images, or generate
    // different numbers of tokens, is not a head-to-head.
    let arg_img = std::env::args().nth(1);
    let arg_max = std::env::args().nth(2).and_then(|v| v.parse::<usize>().ok());
    let path = arg_img.map_or_else(
        || root.join("corpora/clips/carmenta-render/page-00.png"),
        std::path::PathBuf::from,
    );
    let img = ffai_media::load_image(&path)?;
    println!("image: {}  ({}x{})", path.display(), img.width, img.height);
    let opts = ffai_core::engine::VlmOptions {
        prompt: Some("What is written in this image?".into()),
        max_new_tokens: Some(arg_max.unwrap_or(24)),
        ..Default::default()
    };
    // Warm: the first call pays weight load, which is not part of a caption.
    let _ = engine.describe_image(&img, &opts)?;

    let mut best: Option<ffai_argus::engine::CaptionTrace> = None;
    for _ in 0..3 {
        let (_, tr) = engine.describe_image_traced(&img, &opts)?;
        if best.as_ref().is_none_or(|b| tr.total_ms() < b.total_ms()) {
            best = Some(tr);
        }
    }
    let t = best.ok_or("no run")?;
    let total = t.total_ms();
    let rows = [
        ("preprocess", t.preprocess_ms),
        ("VISION (tower)", t.tower_ms),
        ("assemble", t.assemble_ms),
        ("prefill", t.prefill_ms),
        ("generate", t.decode_ms()),
        ("detokenize", t.detokenize_ms),
    ];
    println!("caption stage split, min of 3, warm ({} tiles, {} prompt tokens)\n",
             t.tiles, t.prompt_tokens);
    for (name, ms) in rows {
        println!("  {name:<16} {ms:>9.0} ms  {:>5.1}%", ms / total * 100.0);
    }
    println!("  {:-<16} {total:>9.0} ms", "total");
    println!("\n  VISION SHARE: {:.1}%", t.tower_ms / total * 100.0);
    Ok(())
}
