//! Peak memory for one caption, under the real engine and worker count.
//!
//! # Why this is TWO processes and not an interleaved A/B
//!
//! Every other instrument in this crate interleaves its arms in one process,
//! because timing on this box drifts up to 1.76x. Peak RSS cannot be measured
//! that way: it is a **high-water mark**, so once the greedier arm has run, the
//! leaner arm's peak is indistinguishable from it. Each arm needs a fresh
//! process.
//!
//! That is fine here, and it is worth being clear why: peak RSS is not a
//! stopwatch. It does not care whether the box is quiet, warm, or compiling
//! something else. Two separate runs are a valid comparison for this quantity,
//! and would not be for a duration.
//!
//! Run each arm in its own process and read `PeakWorkingSet64` from outside,
//! since the peak belongs to the process rather than to any point inside it.
//! `FFAI_ARGUS_LATE_NORM` selects the arm.
use ffai_core::engine::{VlmEngine, VlmOptions};
use ffai_core::types::{ImageBuffer, PixelFormat};
use std::path::Path;

const IMG: usize = 384;

fn reference_image() -> ImageBuffer {
    let mut data = vec![0u8; IMG * IMG * 3];
    let mut i = 0;
    for y in 0..IMG {
        let fy = y as f64 / IMG as f64;
        for x in 0..IMG {
            let fx = x as f64 / IMG as f64;
            let r = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * fx).sin();
            let g = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * fy + 1.0).sin();
            let b = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * (fx + fy) + 2.0).sin();
            data[i] = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            data[i + 1] = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            data[i + 2] = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            i += 3;
        }
    }
    ImageBuffer { width: IMG as u32, height: IMG as u32, format: PixelFormat::Rgb8, data }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifests = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let engine = ffai_argus::SmolVlm::with_manifest_dir(manifests);
    let opts = VlmOptions { max_new_tokens: Some(32), ..VlmOptions::default() };
    let text = engine.describe_image(&reference_image(), &opts)?;

    // The peak is read by the CALLER from the OS, not from in here: pulling
    // `ffai-bench` in as a dependency of `ffai-argus` just to read one counter
    // would invert the dependency direction — the bench measures the engine,
    // never the reverse.
    println!(
        "  LATE_NORM={}  caption {text:?}",
        std::env::var("FFAI_ARGUS_LATE_NORM").unwrap_or_else(|_| "default".into()),
    );
    Ok(())
}
