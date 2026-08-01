//! Paired A/B: rff's PNG decoder against the `png`-crate path it replaced.
//!
//! The swap was justified by ownership (Principle 7 — codecs come from home)
//! and by capability (rff sets `EXPAND | STRIP_16`, so palette and 16-bit
//! PNGs decode instead of erroring). Speed is the third claim and the only
//! one that needs a number.
//!
//! Measurement follows the rules this repo's campaigns paid for: arms
//! ALTERNATED per round so neither samples a busier minute, MINIMUM over
//! rounds rather than mean (foreign load only ever adds time, so the minimum
//! is the floor of the code's own cost), and a byte-equality check first —
//! a faster decoder that returns different pixels is not a faster decoder,
//! and that check is a standing test in this crate rather than a courtesy.
//!
//! ```sh
//! cargo run --release -p ffai-media --example png_ab -- [dir] [rounds]
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use ffai_core::types::{ImageBuffer, PixelFormat};

/// The pre-rff implementation, kept as the comparison arm.
fn load_png_reference(path: &Path) -> Option<ImageBuffer> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());
    if info.bit_depth != png::BitDepth::Eight {
        return None;
    }
    let format = match info.color_type {
        png::ColorType::Grayscale => PixelFormat::Gray8,
        png::ColorType::Rgb => PixelFormat::Rgb8,
        png::ColorType::Rgba => PixelFormat::Rgba8,
        png::ColorType::GrayscaleAlpha => {
            buf = buf.chunks_exact(2).map(|p| p[0]).collect();
            PixelFormat::Gray8
        }
        png::ColorType::Indexed => return None,
    };
    Some(ImageBuffer { width: info.width, height: info.height, format, data: buf })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "corpora/clips/diana-coco-v3".into());
    let rounds: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(7);

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(root.join(&dir))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    paths.sort();
    paths.truncate(40);
    if paths.is_empty() {
        return Err(format!("no PNGs in {dir}").into());
    }

    // Equality BEFORE timing. If the pixels differ the timing is meaningless.
    let mut bytes = 0usize;
    for p in &paths {
        let a = ffai_media::load_image(p)?;
        let Some(b) = load_png_reference(p) else { continue };
        assert!(a.data == b.data && a.format == b.format, "{} differs", p.display());
        bytes += a.data.len();
    }

    let (mut rff_best, mut png_best) = (f64::MAX, f64::MAX);
    for r in 0..rounds {
        let run_rff = || {
            let t = Instant::now();
            for p in &paths {
                std::hint::black_box(ffai_media::load_image(p).unwrap());
            }
            t.elapsed().as_secs_f64() * 1e3
        };
        let run_png = || {
            let t = Instant::now();
            for p in &paths {
                std::hint::black_box(load_png_reference(p));
            }
            t.elapsed().as_secs_f64() * 1e3
        };
        // Alternate, so "the second one runs warmer" cancels across rounds.
        let (a, b) = if r % 2 == 0 {
            let a = run_rff();
            (a, run_png())
        } else {
            let b = run_png();
            (run_rff(), b)
        };
        rff_best = rff_best.min(a);
        png_best = png_best.min(b);
    }

    println!("{dir} · {} images · {:.1} MiB decoded · best of {rounds}", paths.len(), bytes as f64 / 1048576.0);
    println!("  rff-codec-png : {rff_best:8.2} ms   ({:.1} MiB/s)", bytes as f64 / 1048576.0 / (rff_best / 1e3));
    println!("  png crate     : {png_best:8.2} ms   ({:.1} MiB/s)", bytes as f64 / 1048576.0 / (png_best / 1e3));
    println!("  rff is {:.3}x the png crate's time  ({})", rff_best / png_best,
             if rff_best < png_best { "rff FASTER" } else { "rff slower" });
    Ok(())
}
