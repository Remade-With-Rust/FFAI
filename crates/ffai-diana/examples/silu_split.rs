//! How much of SiLU runs on ONE worker?
use ffai_core::engine::{DetectEngine, DetectOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tier = std::env::args().nth(1).unwrap_or_else(|| "n".into());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let img = ffai_media::load_image(&root.join("corpora/clips/diana-coco/coco-032.png"))?;
    let engine = ffai_diana::engine::Yolo26::build(&tier, ffai_diana::image::Geometry::Rect, root.join("models"));
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };
    engine.detect(&img, &opts)?;
    let _ = ffai_diana::silu::take_silu_split();
    engine.detect(&img, &opts)?;
    let (se, sc, pe, pc) = ffai_diana::silu::take_silu_split();
    let tot = (se + pe).max(1);
    println!("tier {tier}: SiLU per image");
    println!("  SERIAL   (one chunk): {sc:>5} calls, {:>8.2} M elements  {:5.1}%", se as f64 / 1e6, se as f64 / tot as f64 * 100.0);
    println!("  PARALLEL (fans out) : {pc:>5} calls, {:>8.2} M elements  {:5.1}%", pe as f64 / 1e6, pe as f64 / tot as f64 * 100.0);
    println!();
    println!("  SiLU is ~15.2% of the pipeline. The serial share of THAT is {:.1}%,", se as f64 / tot as f64 * 100.0);
    println!("  so {:.1}% of the whole pipeline runs SiLU on a single worker.", 15.2 * se as f64 / tot as f64);
    Ok(())
}
