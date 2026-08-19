//! Rect vs square, ONE instrument, interleaved.
//!
//! The bench put Diana at 35 ms rect and 81 ms square and I called that
//! superlinear. It was not: rect had read 35, 50 and 53 across three runs and
//! square was measured ONCE, so a best-case reading was compared against a
//! single sample.
//!
//! This measures both geometries in one process, alternating, min-of-N, so the
//! ratio is paired and the box's drift lands on both arms. Ultralytics goes
//! 40 -> 55 ms on the same corpus (1.38x) and square has 1.43x the pixels, so
//! anything near 1.4x here is linear and there is nothing geometry-specific
//! to find.
use ffai_core::engine::{DetectEngine, DetectOptions};
use ffai_diana::engine::Yolo26;
use ffai_diana::image::Geometry;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let img = ffai_media::load_image(&root.join("corpora/clips/diana-coco/coco-032.png"))?;
    let n: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(60);
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };

    let rect = Yolo26::build("n", Geometry::Rect, root.join("models"));
    let square = Yolo26::build("n", Geometry::Square, root.join("models"));
    rect.detect(&img, &opts)?;
    square.detect(&img, &opts)?;

    let (mut br, mut bs) = (f64::MAX, f64::MAX);
    for i in 0..n {
        // Alternate the order every iteration so "second is warmer" cancels.
        // The two blocks share a first line but run the arms in opposite
        // order, which is the entire point of the interleave.
        #[allow(clippy::branches_sharing_code)]
        if i % 2 == 0 {
            let t = Instant::now();
            std::hint::black_box(rect.detect(&img, &opts)?);
            br = br.min(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            std::hint::black_box(square.detect(&img, &opts)?);
            bs = bs.min(t.elapsed().as_secs_f64() * 1e3);
        } else {
            let t = Instant::now();
            std::hint::black_box(square.detect(&img, &opts)?);
            bs = bs.min(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            std::hint::black_box(rect.detect(&img, &opts)?);
            br = br.min(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    // SOLO passes, kept but NOT usable as a control — they run AFTER 2n
    // interleaved detects and read SLOWER than the arms they were meant to
    // check (135 ms rect solo against 62 ms interleaved). That is an ordering
    // confound: the machine degrades across a long run, which is exactly what
    // the alternation above exists to cancel and what these passes do not.
    //
    // Left in place with this note rather than deleted, because the temptation
    // to "just also measure it solo" is what produced them, and the numbers
    // look authoritative until you notice solo is slower than paired.
    let mut sr = f64::MAX;
    for _ in 0..n {
        let t = Instant::now();
        std::hint::black_box(rect.detect(&img, &opts)?);
        sr = sr.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut ss = f64::MAX;
    for _ in 0..n {
        let t = Instant::now();
        std::hint::black_box(square.detect(&img, &opts)?);
        ss = ss.min(t.elapsed().as_secs_f64() * 1e3);
    }

    println!("source {}x{}", img.width, img.height);
    println!("  rect   min-of-{n} = {br:6.2} ms");
    println!("  square min-of-{n} = {bs:6.2} ms");
    println!("  ratio  {:.2}x   (square has 1.43x the pixels; Ultralytics reads 1.38x)", bs / br);
    println!("SOLO (one geometry at a time, same harness):");
    println!("  rect   min-of-{n} = {sr:6.2} ms");
    println!("  square min-of-{n} = {ss:6.2} ms");
    println!("  ratio  {:.2}x", ss / sr);
    println!("  interleaving inflates rect {:.2}x, square {:.2}x", br / sr, bs / ss);
    Ok(())
}
