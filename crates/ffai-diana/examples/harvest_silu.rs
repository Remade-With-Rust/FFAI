//! Prometheus Stage 1: harvest the distribution silu ACTUALLY sees.
//!
//! `exp_fast` clamps to [-125, 125] and carries a degree-5 polynomial to be
//! accurate across all of it. Real activations do not use that range. How
//! much they DO use decides how short a polynomial can replace it — and
//! `prom-prove` can bound the error over the harvested interval instead of
//! over a defensive one.
//!
//! ```sh
//! cargo run --release -p ffai-diana --features prometheus-telemetry \
//!     --example harvest_silu > Prometheus-dataset.csv
//! ```
use ffai_core::engine::{DetectEngine, DetectOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tier = std::env::args().nth(1).unwrap_or_else(|| "n".into());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let mut clips: Vec<_> = std::fs::read_dir(root.join("corpora/clips/diana-coco"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    clips.sort();
    clips.truncate(12);
    let engine = ffai_diana::engine::Yolo26::build(&tier, ffai_diana::image::Geometry::Rect, root.join("models"));
    let opts = DetectOptions { confidence: 0.001, max_detections: 100, ..Default::default() };
    // Warm OUTSIDE the harvest so lazy init does not colour the distribution.
    engine.detect(&ffai_media::load_image(&clips[0])?, &opts)?;
    ffai_diana::telemetry::reset();
    for p in &clips {
        engine.detect(&ffai_media::load_image(p)?, &opts)?;
    }
    match ffai_diana::telemetry::dump() {
        Some(csv) => print!("{csv}"),
        None => eprintln!("no telemetry — build with --features prometheus-telemetry"),
    }
    Ok(())
}
