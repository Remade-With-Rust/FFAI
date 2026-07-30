//! Content classification for per-unit dispatch (mission plan §4).
//!
//! Carmenta has now measured THREE independent per-content sign-flips —
//! recognition lineage, word-segmentation strategy, and engine choice — each
//! one a case where the better option on synthetic text is the worse option
//! on photographs. Averaging them away would discard both wins, so the
//! pipeline dispatches instead, on a signal cheap enough to run per frame.
//!
//! **The signal: adjacent-pixel flatness.** The fraction of horizontally
//! adjacent pixel pairs that are EXACTLY equal. Rendered and screen-captured
//! text has large exactly-flat regions; a camera sensor essentially never
//! produces exact equality at scale, because shot noise perturbs every
//! photosite independently.
//!
//! Measured on this project's four corpora:
//!
//! | corpus | flatness |
//! |---|---|
//! | render (synthetic print) | 0.881 – 0.943 |
//! | frames (synthetic HUD) | 0.974 – 0.995 |
//! | capture (GDI ClearType) | 0.974 |
//! | CORD (real photographs) | 0.103 – 0.507 |
//!
//! The classes are separated by a 0.37-wide EMPTY band, so [`THRESHOLD`] is
//! not a fitted constant — it is the middle of a gap, which is why this
//! dispatch is expected to hold on content neither corpus contains.

use ffai_core::types::{ImageBuffer, PixelFormat};

/// What kind of source this image is, for dispatch purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentKind {
    /// Rendered or screen-captured: exact flat regions, clean uniform gaps.
    Rendered,
    /// Camera capture: sensor noise, tilt, variable spacing.
    Photographic,
}

/// Flatness above this counts as rendered. The middle of the measured empty
/// band (photos top out at 0.51, rendered bottoms out at 0.88).
pub const THRESHOLD: f32 = 0.70;

/// Fraction of horizontally-adjacent pixel pairs that are exactly equal.
/// Rows and columns are subsampled on large inputs — the statistic is a
/// ratio, so it is scale-free, and this keeps the cost near-zero per frame.
pub fn flatness(img: &ImageBuffer) -> f32 {
    let (w, h) = (img.width as usize, img.height as usize);
    if w < 2 || h < 1 {
        return 1.0;
    }
    let bpp = img.format.bytes_per_pixel();
    // Sample the first channel only: for RGB, sensor noise shows in every
    // channel, and comparing one is enough to detect exact flatness.
    let row_step = if h > 512 { h / 256 } else { 1 }.max(1);
    let col_step = if w > 512 { 2 } else { 1 };
    let (mut equal, mut total) = (0usize, 0usize);
    for y in (0..h).step_by(row_step) {
        let row = y * w * bpp;
        let mut x = 0;
        while x + col_step < w {
            let a = img.data[row + x * bpp];
            let b = img.data[row + (x + col_step) * bpp];
            equal += usize::from(a == b);
            total += 1;
            x += col_step;
        }
    }
    if total == 0 { 1.0 } else { equal as f32 / total as f32 }
}

/// Classify, honouring `FFAI_CONTENT=rendered|photo|auto` for A/B and for
/// callers who know their source better than a heuristic can.
pub fn classify(img: &ImageBuffer) -> ContentKind {
    match std::env::var("FFAI_CONTENT").as_deref() {
        Ok("rendered") => return ContentKind::Rendered,
        Ok("photo") | Ok("photographic") => return ContentKind::Photographic,
        _ => {}
    }
    if flatness(img) >= THRESHOLD {
        ContentKind::Rendered
    } else {
        ContentKind::Photographic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(data: Vec<u8>, w: u32, h: u32) -> ImageBuffer {
        ImageBuffer { width: w, height: h, format: PixelFormat::Gray8, data }
    }

    #[test]
    fn flat_synthetic_reads_as_rendered() {
        // A rendered page: mostly exact-white background with some text runs.
        let mut d = vec![255u8; 200 * 100];
        for y in 40..50 {
            for x in 20..120 {
                d[y * 200 + x] = if x % 3 == 0 { 20 } else { 255 };
            }
        }
        let i = img(d, 200, 100);
        assert!(flatness(&i) > THRESHOLD, "flatness {}", flatness(&i));
        assert_eq!(classify(&i), ContentKind::Rendered);
    }

    #[test]
    fn noisy_capture_reads_as_photographic() {
        // A camera frame: every pixel perturbed, so exact equality is rare.
        let mut seed = 0x1234_5678u32;
        let d: Vec<u8> = (0..200 * 100)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (180 + (seed >> 24) % 40) as u8
            })
            .collect();
        let i = img(d, 200, 100);
        assert!(flatness(&i) < THRESHOLD, "flatness {}", flatness(&i));
        assert_eq!(classify(&i), ContentKind::Photographic);
    }
}
