//! Does holding the corpus resident slow inference? A working-set sweep.
//!
//! The bench pre-decodes every clip and holds them live through the timed
//! loop, so the model's weights and activations share L3 with the whole
//! corpus. The reference decodes one frame, uses it, frees it.
//!
//! That is a candidate explanation for a gap nothing else explains: one image
//! repeated reads 45 ms, forty-five different images read a p50 of 81, while
//! Ultralytics reads 52.7 and 55 for the same pair. If the cause is residency
//! rather than content, latency should rise with the NUMBER of images held —
//! and flatten once they no longer fit.
//!
//! Every image is letterboxed to the same 640x640, so tensor shapes are
//! identical at every K. The only variable is how much memory is live.
use ffai_core::engine::{DetectEngine, DetectOptions};
use ffai_diana::engine::Yolo26;
use ffai_diana::image::Geometry;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let dir = root.join("corpora/clips/diana-coco");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    paths.sort();
    paths.truncate(45);

    // Load ONE image to size things and warm the engine, then drop it. The
    // first version of this held all 45 in an `all` vec and took K from it —
    // so 48 MiB stayed resident in EVERY arm and the working set never
    // varied. That is why K=1 read 121 ms against the 45 ms the same work
    // takes elsewhere, and why the sweep looked flat: there was no sweep.
    let probe = ffai_media::load_image(&paths[0])?;
    let bytes_each: usize = probe.data.len();
    let eng = Yolo26::build("n", Geometry::Square, root.join("models"));
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };
    eng.detect(&probe, &opts)?;
    eng.detect(&probe, &opts)?;
    drop(probe);

    println!("{:>4} {:>12} {:>10} {:>10} {:>9}", "K", "resident MiB", "min ms", "p50 ms", "vs K=1");
    let mut base = 0f64;
    // Descending, then the K=1 arm runs LAST — so if it still looks fast it
    // is not warmup, which is the confound the first version had.
    for k in [45usize, 32, 16, 8, 4, 2, 1] {
        // Hold exactly K images live; cycle through them so each detect sees a
        // different one, as the bench does.
        // Load exactly K from disk for this arm. Nothing else is resident.
        let held: Vec<_> = paths.iter().take(k).map(|p| ffai_media::load_image(p).unwrap()).collect();
        let iters = 45;
        let mut times = Vec::with_capacity(iters);
        for i in 0..iters {
            let img = &held[i % k];
            let t = Instant::now();
            std::hint::black_box(eng.detect(img, &opts)?);
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (mn, p50) = (times[0], times[times.len() / 2]);
        if k == 1 {
            base = p50;
        } else if base == 0.0 {
            base = p50; // first arm seen; rewritten when K=1 lands
        }
        println!("{k:>4} {:>11.1} {mn:>10.1} {p50:>10.1} {:>8.2}x",
                 (k * bytes_each) as f64 / 1048576.0, p50 / base);
        std::hint::black_box(&held);
    }
    println!("\n  identical 640x640 tensors at every K — the ONLY variable is resident bytes.");
    println!("  rising p50 = residency costs; flat p50 = the bench gap is something else.");
    Ok(())
}
