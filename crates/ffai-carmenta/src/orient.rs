//! Page orientation: which quarter turn makes a page upright
//! (docs/plans/commercial-gaps.md, gap 5).
//!
//! Scanners store portrait forms sideways, and phone photos arrive in any of
//! four turns. Read sideways, a clean page came back with a fraction of its
//! lines at 0.38-0.58 confidence, with no error to say why. The engine-side
//! decision ([`crate::engine::CraftCrnn::detect_orientation`]) uses two
//! signals the models already produce:
//!
//! * **axis**, from detection: the detector draws WIDE boxes around text lines
//!   that run horizontally and TALL ones around lines that run vertically, so
//!   box aspect, weighted by area, separates {0, 180} from {90, 270};
//! * **which way up**, from recognition: a sample of the largest boxes is read
//!   in both candidate turns, and the upright turn reads with clearly higher
//!   confidence (0.15 or more on every page a caller measured).
//!
//! This module holds the pure geometry: rotation and mapping boxes back.

use ffai_core::types::{BoundingBox, ImageBuffer};

/// Rotate an image `turns` quarter turns CLOCKWISE (0-3; taken mod 4).
#[must_use]
pub fn rotate(img: &ImageBuffer, turns: u8) -> ImageBuffer {
    let turns = turns % 4;
    if turns == 0 {
        return img.clone();
    }
    let (w, h) = (img.width as usize, img.height as usize);
    let bpp = img.format.bytes_per_pixel();
    let (nw, nh) = if turns == 2 { (w, h) } else { (h, w) };
    let mut out = vec![0u8; w * h * bpp];
    for y in 0..h {
        let src_row = &img.data[y * w * bpp..(y + 1) * w * bpp];
        for x in 0..w {
            let (nx, ny) = match turns {
                1 => (h - 1 - y, x),
                2 => (w - 1 - x, h - 1 - y),
                _ => (y, w - 1 - x),
            };
            let d = (ny * nw + nx) * bpp;
            out[d..d + bpp].copy_from_slice(&src_row[x * bpp..(x + 1) * bpp]);
        }
    }
    ImageBuffer {
        width: nw as u32,
        height: nh as u32,
        format: img.format,
        data: out,
    }
}

/// Keep every `f`-th pixel in each direction (nearest-neighbour decimation).
/// For the axis vote only: box ASPECT survives a 2-3x reduction intact, and
/// detection cost falls with the square of it.
#[must_use]
pub fn decimate(img: &ImageBuffer, f: usize) -> ImageBuffer {
    if f <= 1 {
        return img.clone();
    }
    let (w, h) = (img.width as usize, img.height as usize);
    let (nw, nh) = (w.div_ceil(f), h.div_ceil(f));
    let bpp = img.format.bytes_per_pixel();
    let mut data = Vec::with_capacity(nw * nh * bpp);
    for y in (0..h).step_by(f) {
        for x in (0..w).step_by(f) {
            let s = (y * w + x) * bpp;
            data.extend_from_slice(&img.data[s..s + bpp]);
        }
    }
    ImageBuffer {
        width: u32::try_from(nw).unwrap_or(u32::MAX),
        height: u32::try_from(nh).unwrap_or(u32::MAX),
        format: img.format,
        data,
    }
}

/// Map a box found in the image rotated `turns` clockwise back to the frame
/// of the ORIGINAL `(orig_w, orig_h)` image. Boxes are edge coordinates
/// (`x .. x + width`), so a quarter turn maps them exactly.
#[must_use]
pub fn unrotate_box(b: BoundingBox, turns: u8, orig_w: f32, orig_h: f32) -> BoundingBox {
    let turns = turns % 4;
    // Size of the rotated frame the box lives in.
    let (rw, rh) = if turns % 2 == 1 {
        (orig_h, orig_w)
    } else {
        (orig_w, orig_h)
    };
    // Rotating the rotated frame by (4 - turns) clockwise quarter turns
    // returns to the original. One clockwise turn of a frame (W, H) sends the
    // point (x, y) to (H - y, x).
    let mut corners = [(b.x, b.y), (b.x + b.width, b.y + b.height)];
    let (mut fw, mut fh) = (rw, rh);
    for _ in 0..(4 - turns) % 4 {
        for c in &mut corners {
            *c = (fh - c.1, c.0);
        }
        (fw, fh) = (fh, fw);
    }
    let (x0, x1) = (
        corners[0].0.min(corners[1].0),
        corners[0].0.max(corners[1].0),
    );
    let (y0, y1) = (
        corners[0].1.min(corners[1].1),
        corners[0].1.max(corners[1].1),
    );
    BoundingBox {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffai_core::types::PixelFormat;

    fn img(w: u32, h: u32) -> ImageBuffer {
        ImageBuffer {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            data: (0..w * h).map(|i| i as u8).collect(),
        }
    }

    #[test]
    fn four_quarter_turns_are_the_identity_and_two_are_a_half_turn() {
        let a = img(5, 3);
        let mut r = a.clone();
        for _ in 0..4 {
            r = rotate(&r, 1);
        }
        assert_eq!(r.data, a.data);
        assert_eq!(rotate(&rotate(&a, 1), 1).data, rotate(&a, 2).data);
        assert_eq!(rotate(&rotate(&a, 3), 1).data, a.data);
        let q = rotate(&a, 1);
        assert_eq!((q.width, q.height), (3, 5));
    }

    #[test]
    fn a_clockwise_turn_moves_the_top_left_pixel_to_the_top_right() {
        // 2x1 [10, 20] turned clockwise is 1x2 with 10 on top.
        let a = ImageBuffer {
            width: 2,
            height: 1,
            format: PixelFormat::Gray8,
            data: vec![10, 20],
        };
        assert_eq!(rotate(&a, 1).data, vec![10, 20]);
        assert_eq!(rotate(&a, 3).data, vec![20, 10]);
    }

    /// A box drawn around specific pixels in a rotated image maps back onto
    /// the same pixels in the original, for every turn.
    #[test]
    fn unrotate_box_lands_on_the_same_pixels() {
        let (w, h) = (7usize, 4usize);
        // Mark a 2x1 block of distinct values at (x 2..4, y 1..2).
        let mut data = vec![0u8; w * h];
        data[w + 2] = 200;
        data[w + 3] = 201;
        let orig = ImageBuffer {
            width: w as u32,
            height: h as u32,
            format: PixelFormat::Gray8,
            data,
        };
        for turns in 0..4u8 {
            let r = rotate(&orig, turns);
            let (rw, rh) = (r.width as usize, r.height as usize);
            let marked: Vec<(usize, usize)> = (0..rh)
                .flat_map(|y| (0..rw).map(move |x| (x, y)))
                .filter(|&(x, y)| r.data[y * rw + x] >= 200)
                .collect();
            let x0 = marked.iter().map(|p| p.0).min().unwrap();
            let x1 = marked.iter().map(|p| p.0).max().unwrap() + 1;
            let y0 = marked.iter().map(|p| p.1).min().unwrap();
            let y1 = marked.iter().map(|p| p.1).max().unwrap() + 1;
            let b = BoundingBox {
                x: x0 as f32,
                y: y0 as f32,
                width: (x1 - x0) as f32,
                height: (y1 - y0) as f32,
            };
            let back = unrotate_box(b, turns, w as f32, h as f32);
            assert_eq!(
                (back.x, back.y, back.width, back.height),
                (2.0, 1.0, 2.0, 1.0),
                "turns {turns}"
            );
        }
    }
}
