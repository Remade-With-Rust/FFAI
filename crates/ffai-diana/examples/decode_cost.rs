//! D6 work-parity probe: how much work is the reference doing that our
//! timed region is not?
//!
//! `crates/ffai-bench/src/runner.rs` decodes every clip BEFORE starting the
//! per-image timer, then times `detect()` alone. The Ultralytics adapter
//! times `model.predict(path)` — a FILE PATH, so the decode happens inside
//! its timed region. The two arms therefore do different work, and the
//! difference is in our favour, which is the case that deserves the check
//! first rather than last.
//!
//! This is a COUNT-shaped question answered with a duration only because the
//! quantity IS a duration; what makes it admissible is that it is the same
//! decoder, on the same files, that the harness runs.
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

    // Warm the page cache so this measures the DECODER, not the disk — the
    // reference pays a warm-cache read too, having just been handed the same
    // files by the same harness.
    for p in &paths {
        let _ = std::fs::read(p)?;
    }

    let mut best = vec![f64::MAX; paths.len()];
    for _ in 0..5 {
        for (i, p) in paths.iter().enumerate() {
            let t = Instant::now();
            let img = ffai_media::load_image(p)?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&img);
            best[i] = best[i].min(ms);
        }
    }
    let mut s = best.clone();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = best.iter().sum();
    println!("{} PNGs, min-of-5 each", paths.len());
    println!("  decode p50 {:.1} ms · mean {:.1} ms · total {:.0} ms", s[s.len() / 2], sum / best.len() as f64, sum);
    println!("  the reference pays this INSIDE its timed region; our harness pays it OUTSIDE");
    Ok(())
}
