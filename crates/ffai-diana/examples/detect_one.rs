//! Detect on one image and print the boxes — the smoke test before any
//! corpus number is trusted.
//!
//! ```sh
//! cargo run --release -p ffai-diana --example detect_one -- <image> [conf]
//! ```

use ffai_core::engine::{DetectEngine, DetectOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: detect_one <image> [conf]")?;
    let conf: f32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.25);

    let image = ffai_media::load_image(std::path::Path::new(&path))?;
    let engine = ffai_diana::engine::Yolo26::new();
    let opts = DetectOptions { confidence: conf, ..Default::default() };

    let t = std::time::Instant::now();
    let out = engine.detect(&image, &opts)?;
    let cold = t.elapsed();
    let t = std::time::Instant::now();
    let out2 = engine.detect(&image, &opts)?;
    let warm = t.elapsed();

    let names = engine.class_names();
    println!(
        "{} ({}x{}) — {} detections at conf >= {conf}",
        path,
        image.width,
        image.height,
        out.detections.len()
    );
    for d in &out.detections {
        println!(
            "  {:<16} {:.3}  [{:7.1}, {:7.1}, {:7.1}, {:7.1}]",
            names.get(d.class_id as usize).map_or("?", String::as_str),
            d.confidence,
            d.x0,
            d.y0,
            d.x1,
            d.y1
        );
    }
    println!(
        "first call {:.0} ms (includes load) · warm {:.0} ms · deterministic: {}",
        cold.as_secs_f64() * 1000.0,
        warm.as_secs_f64() * 1000.0,
        out.detections == out2.detections
    );
    Ok(())
}
