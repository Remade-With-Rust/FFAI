//! Where is Diana's cache cliff?
//!
//! Matched-geometry benching found Diana scaling **superlinearly** with input
//! size — 2.31x the time for 1.43x the pixels, where Ultralytics scales 1.38x
//! and is therefore linear in the work. Every other measurement in this
//! campaign was taken at ONE input size, so nothing in the profile could have
//! seen it.
//!
//! A cliff has a location. This sweeps input size and reports **ms per
//! megapixel** — flat means linear, rising means the working set has crossed
//! something.
//!
//! Two defects in the first version of this file, both worth keeping in view:
//! it swept SIDE LENGTH of a square input, and a square input letterboxes to
//! 640x640 whatever its size — so it measured one point six times. And it used
//! min-of-5 on a box whose within-run spread reaches 2.8x, which reads roughly
//! double the true floor. Aspect ratio is the variable that moves the
//! letterbox, and the minimum needs enough samples to find.
use ffai_core::engine::{DetectEngine, DetectOptions};
use ffai_core::types::{ImageBuffer, PixelFormat};
use ffai_diana::engine::Yolo26;
use ffai_diana::image::Geometry;
use std::time::Instant;

/// Nearest-neighbour resize; content fidelity is irrelevant here, only shape.
fn resized(src: &ImageBuffer, w: usize, h: usize) -> ImageBuffer {
    let step = match src.format {
        PixelFormat::Gray8 => 1,
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
    };
    let (sw, sh) = (src.width as usize, src.height as usize);
    let mut data = vec![0u8; w * h * step];
    for y in 0..h {
        let sy = y * sh / h;
        for x in 0..w {
            let sx = x * sw / w;
            for c in 0..step {
                data[(y * w + x) * step + c] = src.data[(sy * sw + sx) * step + c];
            }
        }
    }
    ImageBuffer { width: w as u32, height: h as u32, format: src.format, data }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let src = ffai_media::load_image(&root.join("corpora/clips/diana-coco/coco-032.png"))?;
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };

    println!("{:>6} {:>12} {:>9} {:>13} {:>8}", "in h", "letterbox", "Mpx", "ms (min-of-25)", "ms/Mpx");
    // Width pinned at 640, height swept. A SQUARE input always letterboxes to
    // 640x640 whatever its size — the first version of this swept side length
    // and measured one point six times. Aspect ratio is the variable that
    // actually moves the letterbox.
    let eng_rect = Yolo26::build("n", Geometry::Rect, root.join("models"));
    let mut prev: Option<(f64, f64)> = None;
    for h in [160usize, 224, 288, 352, 416, 480, 544, 608, 640] {
        let img = resized(&src, 640, h);
        let _ = eng_rect.detect(&img, &opts)?;
        let mut ms = f64::MAX;
        for _ in 0..25 {
            let t = Instant::now();
            let out = eng_rect.detect(&img, &opts)?;
            ms = ms.min(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&out);
        }
        // Rect: the long side goes to 640, the short side to the smallest
        // multiple of 32 containing it.
        let lb_h = h + (640 - h) % 32;
        let mpx = (640 * lb_h) as f64 / 1e6;
        let per = ms / mpx;
        // No marker. A single point 10% above its neighbour is this box's
        // noise, not a knee — the first run of this flagged 480 at 296 ms/Mpx
        // while 544 came back at 219 and 160 sat at 237. A cliff is a
        // SUSTAINED rise across several sizes; anything else is the spread.
        let arrow = "";
        println!("{:>6} {:>7}x{:<4} {mpx:>9.3} {ms:>13.2} {per:>8.2}{arrow}", h, 640, lb_h);
        prev = Some((mpx, per));
    }
    println!("\n  flat ms/Mpx = linear in the work. Rising = the working set crossed something.");
    Ok(())
}
