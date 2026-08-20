//! Load and run every tier whose weights are present.
//!
//! The graph code is tier-agnostic: widths and repeat counts come from the
//! checkpoint's own scale via `config::Dims`. This is the proof — the same
//! binary building n and s from the same source, and a strict load that
//! fails closed if the derivation is wrong for any of them.
//!
//! ```sh
//! cargo run --release -p ffai-diana --example tiers
//! ```

use ffai_core::engine::{DetectEngine, DetectOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf();
    let img = root.join("corpora/clips/diana-coco/coco-032.png");
    let image = ffai_media::load_image(&img)?;
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };

    println!("{:<10} {:>10} {:>9} {:>8}  top detection", "TIER", "params", "warm ms", "dets");
    for tier in ffai_diana::engine::TIERS {
        let engine = ffai_diana::engine::Yolo26::build(
            tier,
            ffai_diana::image::Geometry::Rect,
            root.join("models"),
        );
        // Weights are AGPL and user-converted, so an absent tier is expected
        // rather than a failure — say which command produces it.
        match engine.detect(&image, &opts) {
            Err(e) => {
                let msg = e.to_string();
                let short = msg.lines().next().unwrap_or(&msg);
                println!("{tier:<10} {:>10} {short}", "-");
            }
            Ok(_) => {
                let t = std::time::Instant::now();
                let out = engine.detect(&image, &opts)?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let names = engine.class_names();
                let top = out
                    .detections
                    .first().map_or_else(|| "none".into(), |d| {
                        format!(
                            "{} {:.3}",
                            names.get(d.class_id as usize).map_or("?", String::as_str),
                            d.confidence
                        )
                    });
                println!(
                    "{tier:<10} {:>10} {ms:>9.1} {:>8}  {top}",
                    "loaded",
                    out.detections.len()
                );
            }
        }
    }
    Ok(())
}
