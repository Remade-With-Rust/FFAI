//! Straightening a slightly rotated page (`OcrOptions::deskew`).
//!
//! Detection and recognition both cut axis-aligned boxes. On a page scanned a
//! couple of degrees off square, a long line climbs or falls by more than its
//! own height across the page, so its crop takes in the lines above and below
//! and the recognizer reads noise: PdfCraft measured 0.41 % CER at 0.5
//! degrees, 0.68 % at 1, 93.3 % at 2 and 98.8 % at 5 on rendered pages, and
//! 61 % on a phone photo tilted 1.5 degrees.
//!
//! The engine measures the skew from the detector's own regions
//! ([`crate::mobiledet::skew_degrees`]), and a page at least [`MIN_DEGREES`]
//! off is rotated back ONCE, read, and its boxes mapped back onto the input.
//! A square page is never resampled, so it reads exactly as before.
//!
//! # Coordinates
//!
//! Boxes stay in the INPUT image's coordinates, as with `auto_orient`. A
//! straightened line is a rectangle; on the input page it is that rectangle
//! turned by the skew, and the reported box is its axis-aligned bounds.
//! `OcrOutput::skew_degrees` carries the angle, and [`rotated_corners`]
//! recovers the turned rectangle from a reported box, which is what a text
//! layer drawn along the page's slope needs.

use ffai_core::types::{BoundingBox, ImageBuffer};

use crate::par::prelude::*;

/// Below this the page is read as it is.
pub const MIN_DEGREES: f32 = 0.3;

/// A rotation about the image centres: the point `q` of the straightened image
/// shows the point `centre + R(angle) (q - straight_centre)` of the input,
/// `R` the rotation by `degrees` clockwise on screen (y down).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Skew {
    pub degrees: f32,
    pub centre: (f32, f32),
    pub straight_centre: (f32, f32),
}

impl Skew {
    /// The input-image point that `(x, y)` of the straightened image shows.
    #[must_use]
    pub fn back(&self, (x, y): (f32, f32)) -> (f32, f32) {
        let (s, c) = self.degrees.to_radians().sin_cos();
        let (dx, dy) = (x - self.straight_centre.0, y - self.straight_centre.1);
        (self.centre.0 + c * dx - s * dy, self.centre.1 + s * dx + c * dy)
    }

    /// A box of the straightened image as the axis-aligned bounds of its
    /// turned rectangle on the input, clamped to the input's `w x h`.
    #[must_use]
    pub fn back_box(&self, b: BoundingBox, w: f32, h: f32) -> BoundingBox {
        let corners = [
            (b.x, b.y),
            (b.x + b.width, b.y),
            (b.x + b.width, b.y + b.height),
            (b.x, b.y + b.height),
        ]
        .map(|p| self.back(p));
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for (x, y) in corners {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        let (x0, y0) = (x0.clamp(0.0, w), y0.clamp(0.0, h));
        let (x1, y1) = (x1.clamp(0.0, w), y1.clamp(0.0, h));
        BoundingBox { x: x0, y: y0, width: (x1 - x0).max(0.0), height: (y1 - y0).max(0.0) }
    }
}

/// Rotate `img` by `-degrees` about its centre onto a canvas large enough to
/// hold all of it, so text the skew carried past an edge is not cut off. The
/// new corners are filled with the page's own paper colour (the median of its
/// border pixels), not black, which the detector would read as ink.
#[must_use]
pub fn straighten(img: &ImageBuffer, degrees: f32) -> (ImageBuffer, Skew) {
    let (w, h) = (img.width as usize, img.height as usize);
    let bpp = img.format.bytes_per_pixel();
    let (s, c) = degrees.to_radians().sin_cos();
    let nw = (w as f32 * c.abs() + h as f32 * s.abs()).ceil() as usize;
    let nh = (w as f32 * s.abs() + h as f32 * c.abs()).ceil() as usize;
    let skew = Skew {
        degrees,
        centre: (w as f32 / 2.0, h as f32 / 2.0),
        straight_centre: (nw as f32 / 2.0, nh as f32 / 2.0),
    };
    let paper = paper_colour(img);
    let mut out = vec![0u8; nw * nh * bpp];
    out.par_chunks_mut(nw * bpp).enumerate().for_each(|(qy, row)| {
        for qx in 0..nw {
            // Pixel centres: sample the input where this output centre lands.
            let (px, py) = skew.back((qx as f32 + 0.5, qy as f32 + 0.5));
            let (sx, sy) = (px - 0.5, py - 0.5);
            let dst = &mut row[qx * bpp..(qx + 1) * bpp];
            if sx < -1.0 || sy < -1.0 || sx > w as f32 || sy > h as f32 {
                dst.copy_from_slice(&paper[..bpp]);
                continue;
            }
            let (x0, y0) = (sx.floor(), sy.floor());
            let (fx, fy) = (sx - x0, sy - y0);
            let tap = |x: f32, y: f32, ch: usize| -> f32 {
                if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
                    f32::from(paper[ch])
                } else {
                    f32::from(img.data[(y as usize * w + x as usize) * bpp + ch])
                }
            };
            for (ch, d) in dst.iter_mut().enumerate() {
                let top = tap(x0, y0, ch) * (1.0 - fx) + tap(x0 + 1.0, y0, ch) * fx;
                let bot = tap(x0, y0 + 1.0, ch) * (1.0 - fx) + tap(x0 + 1.0, y0 + 1.0, ch) * fx;
                *d = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
            }
        }
    });
    (
        ImageBuffer {
            width: u32::try_from(nw).unwrap_or(u32::MAX),
            height: u32::try_from(nh).unwrap_or(u32::MAX),
            format: img.format,
            data: out,
        },
        skew,
    )
}

/// Per-channel median of the image's border pixels (up to 4 channels).
fn paper_colour(img: &ImageBuffer) -> [u8; 4] {
    let (w, h) = (img.width as usize, img.height as usize);
    let bpp = img.format.bytes_per_pixel();
    let mut out = [255u8; 4];
    if w == 0 || h == 0 {
        return out;
    }
    let mut idx: Vec<usize> = Vec::with_capacity(2 * (w + h));
    idx.extend((0..w).map(|x| x));
    idx.extend((0..w).map(|x| (h - 1) * w + x));
    idx.extend((0..h).map(|y| y * w));
    idx.extend((0..h).map(|y| y * w + w - 1));
    for (ch, o) in out.iter_mut().enumerate().take(bpp) {
        let mut v: Vec<u8> = idx.iter().map(|&i| img.data[i * bpp + ch]).collect();
        v.sort_unstable();
        *o = v[v.len() / 2];
    }
    out
}

/// The four corners, on the input page, of the turned rectangle whose
/// axis-aligned bounds are `b`, for a page skewed `degrees` (as
/// `OcrOutput::skew_degrees` reports it). In order: top-left, top-right,
/// bottom-right, bottom-left of the straightened rectangle.
///
/// A `W x H` rectangle turned by `a` has bounds `w = W|cos a| + H|sin a|` and
/// `h = W|sin a| + H|cos a|`, which this solves for `W` and `H`; the centre is
/// shared. Exact up to the clamping at the page edge.
#[must_use]
pub fn rotated_corners(b: BoundingBox, degrees: f32) -> [(f32, f32); 4] {
    let (s, c) = degrees.to_radians().sin_cos();
    let (sa, ca) = (s.abs(), c.abs());
    let det = ca * ca - sa * sa;
    let (cx, cy) = (b.x + b.width / 2.0, b.y + b.height / 2.0);
    let (rw, rh) = if det.abs() < 1e-3 {
        (b.width, b.height)
    } else {
        (
            ((b.width * ca - b.height * sa) / det).max(0.0),
            ((b.height * ca - b.width * sa) / det).max(0.0),
        )
    };
    [(-rw / 2.0, -rh / 2.0), (rw / 2.0, -rh / 2.0), (rw / 2.0, rh / 2.0), (-rw / 2.0, rh / 2.0)]
        .map(|(dx, dy)| (cx + c * dx - s * dy, cy + s * dx + c * dy))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffai_core::types::PixelFormat;

    #[test]
    fn back_box_and_rotated_corners_invert() {
        let img = ImageBuffer { width: 2000, height: 3000, format: PixelFormat::Gray8, data: vec![255; 6_000_000] };
        let (straight, skew) = straighten(&img, 2.0);
        assert!(straight.width > 2000 && straight.height > 3000);
        // A line on the straightened page, mapped back and recovered.
        let line = BoundingBox { x: 400.0, y: 900.0, width: 1200.0, height: 40.0 };
        let on_input = skew.back_box(line, 2000.0, 3000.0);
        let corners = rotated_corners(on_input, 2.0);
        let want = [
            (line.x, line.y),
            (line.x + line.width, line.y),
            (line.x + line.width, line.y + line.height),
            (line.x, line.y + line.height),
        ]
        .map(|p| skew.back(p));
        for (g, w) in corners.iter().zip(&want) {
            assert!((g.0 - w.0).abs() < 0.05 && (g.1 - w.1).abs() < 0.05, "{g:?} vs {w:?}");
        }
    }

    #[test]
    fn straighten_keeps_the_paper_colour() {
        let img = ImageBuffer { width: 64, height: 32, format: PixelFormat::Rgb8, data: vec![200; 64 * 32 * 3] };
        let (s, _) = straighten(&img, 5.0);
        assert!(s.data.iter().all(|&v| v == 200), "corners must be filled with paper, not black");
    }
}
