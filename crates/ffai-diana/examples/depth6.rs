//! Depth-6 probe: is the M-D1 speed gap a fair measurement?
//!
//! Run BEFORE profiling anything (codec-six-whys-unknowns: "depth 6 costs
//! minutes and repeatedly invalidates days of downstream work"). Three
//! questions, in the skill's order:
//!
//!   1. Do both arms do identical work?  -> thread counts, input geometry
//!   2. What is the noise floor?         -> repeat our own arm against itself
//!   3. Does micro agree with in-context? -> per-stage vs whole-detect
//!
//! ```sh
//! cargo run --release -p ffai-diana --example depth6 -- <image.png>
//! ```

use ffai_core::engine::{DetectEngine, DetectOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "corpora/clips/diana-coco/coco-032.png".into());
    let image = ffai_media::load_image(std::path::Path::new(&path))?;
    let engine = ffai_diana::engine::Yolo26::new();
    let opts = DetectOptions { confidence: 0.001, max_detections: 100, ..Default::default() };

    println!("image {} ({}x{})", path, image.width, image.height);
    println!(
        "RAYON_NUM_THREADS={:?}  available_parallelism={:?}",
        std::env::var("RAYON_NUM_THREADS").ok(),
        std::thread::available_parallelism().map(|n| n.get()).ok()
    );

    // Warm (loads weights, fills caches) — never timed.
    engine.detect(&image, &opts)?;

    // Q2: noise floor of our own arm against itself.
    let mut times = Vec::new();
    for _ in 0..15 {
        let t = std::time::Instant::now();
        engine.detect(&image, &opts)?;
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    let (min, med, max) = (times[0], times[times.len() / 2], times[times.len() - 1]);
    println!(
        "\nself-vs-self over {} runs: min {min:.1} ms · median {med:.1} ms · max {max:.1} ms \
         · spread {:.2}x",
        times.len(),
        max / min
    );
    println!(
        "  => any A/B delta smaller than {:.1} ms ({:.1}%) is not a result",
        max - min,
        (max - min) / med * 100.0
    );

    // Q3 / depth-2 preview: split preprocessing from the network, so the
    // campaign starts from an absolute-cost ranking rather than a guess.
    // Ranked by ABSOLUTE ms — the worst RATIO is rarely the best target.
    use candle_core::Device;
    let dev = Device::Cpu;
    let mut lb = Vec::new();
    for _ in 0..15 {
        let t = std::time::Instant::now();
        let _ = ffai_diana::image::letterbox(&image, 640, &dev)?;
        lb.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    lb.sort_by(|a, b| a.total_cmp(b));
    println!(
        "\nletterbox alone: min {:.2} ms · median {:.2} ms  ({:.1}% of a {med:.0} ms detect)",
        lb[0],
        lb[lb.len() / 2],
        lb[lb.len() / 2] / med * 100.0
    );
    Ok(())
}
