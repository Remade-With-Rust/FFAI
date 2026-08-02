//! How many of the pool's workers are actually busy?
//!
//! The fusion campaign proved nothing here is memory-bandwidth-bound, and
//! left two parallel-efficiency numbers unexplained: the pipeline gets 1.53x
//! from 24 cores, and ~65% of it is serial by Amdahl. "Serial" was inferred
//! from a scaling curve, never measured directly.
//!
//! `QueryProcessCycleTime` sums cycles across every thread in the process.
//! Divided by wall time and by the calibrated single-core rate, it gives the
//! MEAN NUMBER OF CORES BUSY — a direct utilisation figure that needs no
//! Amdahl fit and no second arm to compare against.
//!
//! A 4-worker pool doing perfectly parallel work reads 4.0. Reading 1.2 means
//! the fan-out is decorative and there is a 3x sitting idle.
//!
//! # Read the MAXIMUM across runs, not the mean
//!
//! This is a ratio of CPU time to WALL time, so it is only as clean as the
//! wall. A competing process steals cores: our wall grows, our CPU does not,
//! and the ratio falls. Contention can therefore only DEPRESS this number,
//! never inflate it — which makes the highest value observed the least
//! contaminated one, and a valid lower bound on how idle the pool is.
//!
//! First readings on this box ranged 0.65 to 2.01 with another project's
//! benchmark occupying ~12 cores. Take 2.01: **at best half of a 4-worker
//! pool is doing anything.** The true figure on a quiet box is somewhere at
//! or above that, and needs one to pin down.
use ffai_core::engine::{DetectEngine, DetectOptions};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tier = std::env::args().nth(1).unwrap_or_else(|| "n".into());
    let reps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(40);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let img = ffai_media::load_image(&root.join("corpora/clips/diana-coco/coco-032.png"))?;
    let engine = ffai_diana::engine::Yolo26::build(&tier, ffai_diana::image::Geometry::Rect, root.join("models"));
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };
    engine.detect(&img, &opts)?;

    let hz = match ffai_diana::cputime::calibrate() {
        Some(h) => h,
        None => {
            eprintln!("no process cycle counter on this platform");
            return Ok(());
        }
    };

    let c0 = ffai_diana::cputime::cycles().unwrap_or(0);
    let t = Instant::now();
    for _ in 0..reps {
        engine.detect(&img, &opts)?;
    }
    let wall = t.elapsed().as_secs_f64();
    let cycles = ffai_diana::cputime::cycles().unwrap_or(0) - c0;

    let cpu_secs = cycles as f64 / hz;
    let busy = cpu_secs / wall;
    println!("tier {tier}, {reps} images");
    println!("  wall        {:8.1} ms  ({:.1} ms/image)", wall * 1e3, wall * 1e3 / reps as f64);
    println!("  process CPU {:8.1} ms  ({:.1} ms/image)", cpu_secs * 1e3, cpu_secs * 1e3 / reps as f64);
    println!("  MEAN CORES BUSY: {busy:.2}");
    println!();
    println!("  A 4-worker pool at perfect efficiency reads 4.00.");
    println!("  Serial-equivalent fraction (Amdahl, p=4): {:.0}%", ((4.0 - busy) / 3.0 * 100.0).clamp(0.0, 100.0));
    Ok(())
}
