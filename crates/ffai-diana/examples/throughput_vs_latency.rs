//! Are "warm throughput" and "p50 latency" the same function? (They are not.)
//!
//! Both currently read ~2.5-3.2x behind, which invites treating them as one
//! number. They are different questions with different levers:
//!
//! - **p50 latency** is the single-image critical path. Only making one
//!   image faster helps.
//! - **warm throughput** is images per second, and a deployment processing
//!   many images can run them CONCURRENTLY. Nothing about the per-image
//!   path has to change.
//!
//! Two measured facts say concurrency is the unexploited axis here:
//! 24 cores buy the single-image path only **1.42x** (so intra-image
//! parallelism is nearly exhausted), and PyTorch reaches its number on
//! **8 intra-op threads** while we spend 24 to be slower. Idle capacity plus
//! independent work units is exactly the "threads dwarf micro-SIMD when the
//! units are independent" case.
//!
//! The structural asymmetry that makes this ours to take: `DetectEngine` is
//! `Send + Sync`, so N concurrent detections share ONE 95 MiB model in one
//! process. The PyTorch path cannot do that — Python's GIL forces
//! multiprocessing, and each worker carries its own ~310 MiB copy.
//!
//! ```sh
//! cargo run --release -p ffai-diana --example throughput_vs_latency
//! RAYON_NUM_THREADS=4 cargo run --release -p ffai-diana --example throughput_vs_latency
//! ```

use ffai_bench::corpus::Manifest;
use ffai_core::engine::{DetectEngine, DetectOptions};
use ffai_core::types::ImageBuffer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf();
    let manifest = Manifest::load(&root.join("corpora/diana-coco-v2.toml"))?;
    let engine = ffai_diana::engine::Yolo26::with_manifest_dir(root.join("models"));
    let opts = DetectOptions { confidence: 0.001, max_detections: 100, ..Default::default() };

    // Decode once, outside every timed region — the same rule the harness uses.
    let images: Vec<ImageBuffer> = manifest
        .holdout()
        .map(|c| ffai_media::load_image(&manifest.clip_path(c)))
        .collect::<Result<_, _>>()?;
    let n = images.len();
    engine.detect(&images[0], &opts)?; // warm weights + allocator

    println!(
        "{n} images · RAYON_NUM_THREADS={} · cores={}",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into()),
        std::thread::available_parallelism().map(|c| c.get()).unwrap_or(0)
    );

    // --- SEQUENTIAL: one image at a time, intra-op parallelism only. ---
    // This is exactly what `ffai bench detect` measures today.
    let mut per_image = Vec::with_capacity(n);
    let t0 = std::time::Instant::now();
    for im in &images {
        let t = std::time::Instant::now();
        let _ = engine.detect(im, &opts)?;
        per_image.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let seq = t0.elapsed().as_secs_f64();
    per_image.sort_by(|a, b| a.total_cmp(b));
    let p50 = per_image[n / 2];
    let p95 = per_image[(n as f64 * 0.95) as usize];

    // Keep the sequential results so concurrency can be proven not to
    // change them — a throughput win that alters output is not a win.
    let seq_out: Vec<Vec<_>> = images
        .iter()
        .map(|im| engine.detect(im, &opts).map(|o| o.detections))
        .collect::<Result<_, _>>()?;

    // --- CONCURRENT: the first-class batch path. ---
    let t0 = std::time::Instant::now();
    let par_out: Vec<Vec<_>> = engine
        .detect_batch(&images, &opts)?
        .into_iter()
        .map(|o| o.detections)
        .collect();
    let par = t0.elapsed().as_secs_f64();
    let total_dets: usize = par_out.iter().map(|v| v.len()).sum();
    assert_eq!(
        seq_out, par_out,
        "concurrency CHANGED the detections — the engine is not safely shareable"
    );

    println!("\n{:<26} {:>10} {:>12}", "MODE", "img/s", "wall s");
    println!("{:<26} {:>10.2} {:>12.3}", "sequential (today)", n as f64 / seq, seq);
    println!("{:<26} {:>10.2} {:>12.3}", "concurrent (par_iter)", n as f64 / par, par);
    println!(
        "\np50 {p50:.1} ms · p95 {p95:.1} ms  (sequential — the LATENCY number, \
         unchanged by concurrency)"
    );
    println!("concurrency speedup on THROUGHPUT: {:.2}x", seq / par);
    println!("detections {total_dets} (identical work in both modes)");
    println!(
        "\nreference, same corpus: PyTorch 15.30 img/s (8 threads), \
         ORT 33.24 img/s, ours sequential 6.06"
    );
    Ok(())
}
