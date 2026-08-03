//! What speedup do 24 cores actually buy?
//!
//! The roofline accounted for ~42 ms of a 142 ms image: ~20 ms of convolution
//! arithmetic, ~13 ms of allocation traffic, ~9.5 ms of im2col traffic. The
//! remaining ~100 ms was attributed to "framework overhead", which is a label,
//! not a finding.
//!
//! There is a cheaper explanation that was never measured: the work is there
//! and the cores are not being used. Thread COUNT was A/B'd earlier in this
//! campaign and refuted — but that experiment varied the pool size around its
//! default and found no difference, which is a different question from "how
//! much does the pool buy over one thread". A pipeline that scales 2x on 24
//! cores looks identical under pool-size perturbation and has an order of
//! magnitude sitting idle.
//!
//! Min-of-N, because the floor is the honest number for a scaling curve: the
//! tail is scheduler noise and this box's clock drifts more than the effect.
// The shipped allocator; examples do not inherit the binary's.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use ffai_core::engine::{DetectEngine, DetectOptions};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tier = std::env::args().nth(1).unwrap_or_else(|| "n".into());
    let reps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(7);
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
    println!("threads={:2}  tier {tier}  min-of-{reps} = {best:.1} ms", rayon::current_num_threads());
    Ok(())
}
