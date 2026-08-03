//! Does an allocator that keeps pages MAPPED remove the per-call cost?
//!
//! Measured on this box: **58,634 page faults per image**. At 4 KiB each that
//! is ~234 MiB of freshly-faulted pages, against 293.7 MiB/image of
//! allocation — so essentially every byte allocated is arriving on a page the
//! process has to fault in, which means the system allocator is returning
//! memory to the OS and re-faulting it rather than keeping it mapped.
//!
//! That is the only mechanism found in this campaign whose magnitude could
//! account for the remaining gap: 58 k soft faults at even 0.3 us each is
//! ~17 ms of a 69 ms image.
//!
//! One line tests it. If the fault count collapses and the wall follows, the
//! per-call cost was never candle's machinery or SiLU's wrapper — it was the
//! page table.
use ffai_core::engine::{DetectEngine, DetectOptions};
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tier = std::env::args().nth(1).unwrap_or_else(|| "n".into());
    let reps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let img = ffai_media::load_image(&root.join("corpora/clips/diana-coco/coco-032.png"))?;
    let engine = ffai_diana::engine::Yolo26::build(&tier, ffai_diana::image::Geometry::Rect, root.join("models"));
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };
    engine.detect(&img, &opts)?;

    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        engine.detect(&img, &opts)?;
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("mimalloc  tier {tier}  min-of-{reps} = {best:.1} ms");
    Ok(())
}
