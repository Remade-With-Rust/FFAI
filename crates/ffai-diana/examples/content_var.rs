//! Does Diana's cost vary with CONTENT, or only with SIZE?
//!
//! That distinction decides whether content-adaptive dispatch applies at all.
//! The trigger is "per-unit cost varies 5-15x with content" — in a codec it
//! does, because a busy block searches harder than a flat one. In a
//! fixed-compute CNN it should not: the graph does the same FLOPs whatever the
//! pixels say.
//!
//! Grouped by letterbox size so SIZE is held constant and only content moves.
use ffai_core::engine::{DetectEngine, DetectOptions};
use ffai_diana::engine::Yolo26;
use ffai_diana::image::Geometry;
use std::collections::BTreeMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let dir = root.join("corpora/clips/diana-coco");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    paths.sort();
    paths.truncate(30);

    let eng = Yolo26::build("n", Geometry::Rect, root.join("models"));
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };

    // (letterbox_h) -> [(ms, detections)]
    let mut by_size: BTreeMap<u32, Vec<(f64, usize)>> = BTreeMap::new();
    for p in &paths {
        let img = ffai_media::load_image(p)?;
        eng.detect(&img, &opts)?;
        let mut ms = f64::MAX;
        let mut nd = 0;
        for _ in 0..7 {
            let t = Instant::now();
            let out = eng.detect(&img, &opts)?;
            ms = ms.min(t.elapsed().as_secs_f64() * 1e3);
            nd = out.detections.len();
        }
        // Group by the SHORT side, which is what the rect letterbox varies.
        let short = img.width.min(img.height);
        let lb = short + (640 - short) % 32;
        by_size.entry(lb).or_default().push((ms, nd));
    }

    println!("{:>10} {:>6} {:>9} {:>9} {:>7} {:>14}", "letterbox", "n", "min ms", "max ms", "spread", "detections");
    for (lb, v) in &by_size {
        if v.len() < 2 {
            continue;
        }
        let lo = v.iter().map(|x| x.0).fold(f64::MAX, f64::min);
        let hi = v.iter().map(|x| x.0).fold(f64::MIN, f64::max);
        let dmin = v.iter().map(|x| x.1).min().unwrap();
        let dmax = v.iter().map(|x| x.1).max().unwrap();
        println!("{lb:>10} {:>6} {lo:>9.2} {hi:>9.2} {:>6.2}x {dmin:>6}-{dmax:<7}", v.len(), hi / lo);
    }
    println!("\n  spread at FIXED letterbox size = content sensitivity.");
    println!("  dispatch needs 5-15x. Anything near 1x means the graph does not care what it sees.");
    Ok(())
}
