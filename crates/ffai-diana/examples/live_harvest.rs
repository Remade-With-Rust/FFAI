//! Harvest the change gate's threshold on Diana's own content.
//!
//! Carmenta's v1 gate failed because its statistic separated real events from
//! noise by a hair — 2.3 against a 0.9 floor. No threshold survives that. So
//! the question this answers is not "what number" but "**do the two classes
//! separate at all**", and by how many orders of magnitude.
//!
//! Two classes, built from a real corpus image rather than a synthetic
//! pattern, because the whole point is the statistics of actual content:
//!
//! * **HOLD** — the same frame with sensor-style noise added. Must gate.
//! * **MOVE** — the frame translated by 1, 2, 4, 8 px. Must NOT gate; one
//!   pixel of camera shake already moves every edge in the scene.
//!
//! Also prices the gate itself. A gate that costs a tenth of the model saves
//! nothing on a stream that mostly changes, so `gate_ms` against `detect_ms`
//! is what says whether this is worth having at all.

use ffai_core::engine::{DetectEngine, DetectOptions};
use ffai_core::types::{ImageBuffer, PixelFormat};
use ffai_diana::engine::Yolo26;
use ffai_diana::image::Geometry;
use ffai_diana::live::{changed_fraction, LiveConfig, LiveSession};
use std::time::Instant;

/// Deterministic ±amplitude noise. A fixed pattern, not an RNG: the harvest
/// must produce the same numbers on every run or it cannot be quoted.
fn with_noise(src: &ImageBuffer, amp: i16) -> ImageBuffer {
    let mut out = src.clone();
    for (i, p) in out.data.iter_mut().enumerate() {
        let sign = if (i * 2654435761) % 2 == 0 { 1 } else { -1 };
        let mag = ((i * 40503) % (amp as usize + 1)) as i16;
        *p = (i16::from(*p) + sign * mag).clamp(0, 255) as u8;
    }
    out
}

/// Translate right/down by `dx`,`dy`, edge-clamped — what camera shake or a
/// slow pan does to every pixel in the frame at once.
fn translated(src: &ImageBuffer, dx: usize, dy: usize) -> ImageBuffer {
    let step = match src.format {
        PixelFormat::Gray8 => 1,
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
    };
    let (w, h) = (src.width as usize, src.height as usize);
    let mut out = src.clone();
    for y in 0..h {
        let sy = y.saturating_sub(dy);
        for x in 0..w {
            let sx = x.saturating_sub(dx);
            for c in 0..step {
                out.data[(y * w + x) * step + c] = src.data[(sy * w + sx) * step + c];
            }
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let img = ffai_media::load_image(&root.join("corpora/clips/diana-coco/coco-032.png"))?;
    let delta: u8 = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(ffai_diana::live::DEFAULT_PIXEL_DELTA);

    println!("frame {}x{}, pixel_delta = {delta}\n", img.width, img.height);

    println!("HOLD — must gate");
    let mut worst_hold = 0f32;
    for amp in [0i16, 1, 2, 4, 6] {
        let f = with_noise(&img, amp);
        let frac = changed_fraction(&img, &f, delta);
        worst_hold = worst_hold.max(frac);
        println!("  noise +/-{amp:<2}      changed {:>9.6}%", frac * 100.0);
    }

    println!("\nMOVE — must NOT gate");
    let mut best_move = 1f32;
    for d in [1usize, 2, 4, 8] {
        let f = translated(&img, d, d);
        let frac = changed_fraction(&img, &f, delta);
        best_move = best_move.min(frac);
        println!("  shift {d} px       changed {:>9.6}%", frac * 100.0);
    }

    println!("\nSEPARATION");
    println!("  worst HOLD  {:.6}%", worst_hold * 100.0);
    println!("  weakest MOVE {:.6}%", best_move * 100.0);
    if worst_hold > 0.0 {
        println!("  ratio        {:.0}x", best_move / worst_hold);
    } else {
        println!("  ratio        infinite — no HOLD frame moved a single pixel past the delta");
    }
    println!(
        "  default threshold {:.3}% sits {}",
        ffai_diana::live::DEFAULT_CHANGE_FRACTION * 100.0,
        if ffai_diana::live::DEFAULT_CHANGE_FRACTION > worst_hold
            && ffai_diana::live::DEFAULT_CHANGE_FRACTION < best_move
        {
            "BETWEEN the classes"
        } else {
            "OUTSIDE the classes — the constant is wrong for this content"
        }
    );

    // What the gate costs against what it saves.
    let engine = Yolo26::build("n", Geometry::Rect, root.join("models"));
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };
    engine.detect(&img, &opts)?; // warm

    let mut det_ms = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        engine.detect(&img, &opts)?;
        det_ms = det_ms.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let held = with_noise(&img, 2);
    let mut gate_ms = f64::MAX;
    for _ in 0..200 {
        let t = Instant::now();
        let f = changed_fraction(&img, &held, delta);
        gate_ms = gate_ms.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(f);
    }
    println!("\nCOST (min-of-N)");
    println!("  detect      {det_ms:7.3} ms");
    println!("  gate        {gate_ms:7.3} ms   ({:.2}% of a detect)", gate_ms / det_ms * 100.0);
    println!("  break-even  a stream must be > {:.2}% static for the gate to pay", gate_ms / det_ms * 100.0);

    // End to end on a realistic sequence: mostly static with periodic motion,
    // which is what a fixed camera actually produces.
    let mut seq: Vec<ImageBuffer> = Vec::new();
    for i in 0..60 {
        if i % 15 == 0 && i > 0 {
            seq.push(translated(&img, 4, 4));
        } else {
            seq.push(with_noise(&img, 2));
        }
    }
    let eng = std::sync::Arc::new(Yolo26::build("n", Geometry::Rect, root.join("models")));
    let mut s = LiveSession::new(eng, LiveConfig::default(), opts.clone());
    let t = Instant::now();
    for f in &seq {
        s.process(f)?;
    }
    let gated_wall = t.elapsed().as_secs_f64() * 1e3;
    let st = s.stats().clone();

    // The ungated arm, MEASURED rather than estimated as det_ms x frames.
    // Same session type, same code path, threshold 0 so nothing ever gates —
    // which keeps the diff itself in BOTH arms and prices only the skipping.
    let eng2 = std::sync::Arc::new(Yolo26::build("n", Geometry::Rect, root.join("models")));
    let never = LiveConfig { change_fraction: 0.0, ..Default::default() };
    let mut s2 = LiveSession::new(eng2, never, opts.clone());
    let t = Instant::now();
    for f in &seq {
        s2.process(f)?;
    }
    let ungated_wall = t.elapsed().as_secs_f64() * 1e3;

    println!("
SEQUENCE — 60 frames, motion every 15th");
    println!("  processed {} · gated {} · forced {}", st.processed, st.gated, st.forced);
    println!("  skip rate {:.1}%", st.skip_rate() * 100.0);
    println!("  ungated   {ungated_wall:7.0} ms  ({} model runs)", s2.stats().processed);
    println!("  gated     {gated_wall:7.0} ms  ({} model runs)", st.processed);
    println!("  speedup   {:.2}x", ungated_wall / gated_wall);
    Ok(())
}
