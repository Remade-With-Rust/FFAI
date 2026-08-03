//! The README's example, compiled.
//!
//! A published README is permanent, so its code is kept here where `cargo
//! build --examples` type-checks it against the real API rather than against
//! memory. The first draft of it named a `class_name` field that does not
//! exist — `Detection` carries `class_id`, and names come from the engine.
use ffai_core::engine::{DetectEngine, DetectOptions};
use ffai_diana::{engine::Yolo26, image::Geometry};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
        root.join("corpora/clips/diana-coco/coco-032.png").to_string_lossy().into_owned()
    });
    let models = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("models");

    let engine = Yolo26::build("n", Geometry::Rect, models);
    let image = ffai_media::load_image(std::path::Path::new(&path))?;
    let found = engine.detect(&image, &DetectOptions::default())?;
    let names = engine.class_names();

    for d in &found.detections {
        println!(
            "{} {:.2} @ [{:.0} {:.0} {:.0} {:.0}]",
            names[d.class_id as usize], d.confidence, d.x0, d.y0, d.x1, d.y1
        );
    }
    Ok(())
}
