//! Letterbox preprocessing: `ImageBuffer` → the network's input tensor.
//!
//! Reproduces Ultralytics' `LetterBox` in the **square** geometry M-D0
//! pinned (`--rect off`): scale to fit, centre-pad with 114, `/255`, CHW,
//! RGB. The rectangular default is deliberately not implemented — M-D0
//! measured the two geometries disagreeing by 1.5–1.8 pp mAP in opposite
//! directions by tier (mission plan §8.1), so which one this engine
//! implements is a recorded decision, not an accident.
//!
//! The inverse transform is returned with the tensor rather than
//! recomputed by the caller, and travels onward in
//! [`ffai_core::types::DetectOutput`].

use crate::par::prelude::*;
use candle_core::{Device, Result as CandleResult, Tensor};
use ffai_core::types::{ImageBuffer, Letterbox, PixelFormat};

/// The pad value Ultralytics uses. Not 0: a black border would look like
/// content to the detector at the image edge.
pub const PAD_VALUE: f32 = 114.0 / 255.0;

/// How much padding the letterbox adds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Geometry {
    /// Pad to a full `size x size` square. What M-D0 pinned for the parity
    /// gate, because an unpinned geometry made the `.pt` and ORT reference
    /// rows disagree by 1.5-1.8 pp.
    Square,
    /// Pad only to the next multiple of `STRIDE` — Ultralytics' own default.
    ///
    /// A 428x640 photo becomes 448x640 rather than 640x640, so **30% of the
    /// compute stops being spent on grey padding**. M-D0's own board shows
    /// this is better on BOTH axes at the n tier: 70.14 mAP50 against 68.65,
    /// and 18.26 img/s against 15.30. Square is dominated here; pinning it
    /// for the GATE and shipping it as the DEFAULT were two decisions that
    /// got conflated.
    Rect,
}

/// The network's total stride — the padded size must be a multiple of it,
/// or the three feature-map levels do not tile the input.
pub const STRIDE: usize = 32;

/// Letterbox `image` into a CHW f32 tensor in `[0, 1]`.
///
/// The letterbox output shape `(height, width)` for a source, WITHOUT resizing.
///
/// Shares its arithmetic with [`letterbox_with`] line for line, so the number a
/// progress line prints cannot drift from the tensor the model actually sees.
/// Ultralytics prints this shape height-first, and so does the CLI.
#[must_use] 
pub fn letterbox_shape(w0: usize, h0: usize, size: usize, geometry: Geometry) -> (usize, usize) {
    let scale = (size as f32 / w0 as f32).min(size as f32 / h0 as f32);
    let nw = ((w0 as f32 * scale).round() as usize).max(1).min(size);
    let nh = ((h0 as f32 * scale).round() as usize).max(1).min(size);
    match geometry {
        Geometry::Square => (size, size),
        Geometry::Rect => (nh + (size - nh) % STRIDE, nw + (size - nw) % STRIDE),
    }
}

// Returns the tensor and the transform needed to map detections back.
pub fn letterbox(
    image: &ImageBuffer,
    size: usize,
    device: &Device,
) -> CandleResult<(Tensor, Letterbox)> {
    letterbox_with(image, size, Geometry::Square, device)
}

pub fn letterbox_with(
    image: &ImageBuffer,
    size: usize,
    geometry: Geometry,
    device: &Device,
) -> CandleResult<(Tensor, Letterbox)> {
    let (w0, h0) = (image.width as usize, image.height as usize);
    let scale = (size as f32 / w0 as f32).min(size as f32 / h0 as f32);
    let nw = ((w0 as f32 * scale).round() as usize).max(1).min(size);
    let nh = ((h0 as f32 * scale).round() as usize).max(1).min(size);
    // Reproduces Ultralytics' `auto=True` rule: the padding is
    // `(size - n) mod stride`, so the result is the SMALLEST multiple of the
    // stride that still contains the scaled image.
    let (out_w, out_h) = match geometry {
        Geometry::Square => (size, size),
        Geometry::Rect => (nw + (size - nw) % STRIDE, nh + (size - nh) % STRIDE),
    };
    let (size_w, size_h) = (out_w, out_h);
    // Ultralytics computes the pad as `round((size - n) / 2 - 0.1)`, which
    // biases a half-pixel remainder toward the top-left. Reproduced rather
    // than rounded the obvious way: a one-pixel offset shifts every box by
    // one pixel, which is small enough to pass a smoke test and large
    // enough to cost mAP.
    let pad_x = (((out_w - nw) as f32 / 2.0 - 0.1).round() as usize).min(out_w - nw);
    let pad_y = (((out_h - nh) as f32 / 2.0 - 0.1).round() as usize).min(out_h - nh);

    let channels = match image.format {
        PixelFormat::Gray8 => 1,
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
    };
    // The canvas is NOT filled with the pad value and then overwritten: the
    // content rectangle is written in full by the vertical pass below, so
    // only the pad bands (top, bottom, left, right) are filled. 860 000
    // floats of constant fill per call, two thirds of which were overwritten.
    let n_out = 3 * size_h * size_w;
    let mut out: Vec<f32> = Vec::with_capacity(n_out);
    // SAFETY: `f32` is plain data; the pad bands are filled just below and the
    // content rectangle `[pad_y, pad_y + nh) x [pad_x, pad_x + nw)` of every
    // channel is written by `vfill` before the tensor is read.
    #[allow(unsafe_code, clippy::uninit_vec)]
    unsafe {
        out.set_len(n_out);
    }
    for chan in out.chunks_mut(size_h * size_w) {
        chan[..pad_y * size_w].fill(PAD_VALUE);
        chan[(pad_y + nh) * size_w..].fill(PAD_VALUE);
        for row in chan[pad_y * size_w..(pad_y + nh) * size_w].chunks_mut(size_w) {
            row[..pad_x].fill(PAD_VALUE);
            row[pad_x + nw..].fill(PAD_VALUE);
        }
    }

    // Horizontal sample positions, computed ONCE.
    //
    // They were recomputed inside the pixel loop, and the pixel loop runs
    // 273,920 times at the n tier for a 640x428 letterbox — so a divide, a
    // floor and two clamps were being redone for every column of every row.
    // They depend only on `x`, so `nw` of them is the whole set.
    let xs: Vec<(usize, usize, f32)> = (0..nw)
        .map(|x| {
            let sx = ((x as f32 + 0.5) / scale - 0.5).clamp(0.0, (w0 - 1) as f32);
            let x0 = sx.floor() as usize;
            (x0, (x0 + 1).min(w0 - 1), sx - x0 as f32)
        })
        .collect();

    // Nearest-neighbour would be cheaper and visibly worse; bilinear is
    // what the reference resize does.
    //
    // Parallel over (channel, row). This loop was entirely SERIAL while the
    // rest of the engine fanned out over four workers, and at 12.6 % of the
    // pipeline it was the largest single-threaded region left — `pre` is also
    // the one stage whose profiled share is fully trustworthy, because it
    // runs once per image and carries no per-call profiler tax.
    //
    // The arithmetic per output element is UNCHANGED and evaluated in the
    // same order, so the result is byte-identical; only who evaluates it
    // moves. The `/ 255.0` stays a division for that reason — `* (1.0/255.0)`
    // is a different function in the last ulp, and this graph has already had
    // one oracle breached by exactly that class of substitution.
    // `u8 -> f32 / 255` has 256 possible answers. Computed once each, here,
    // with the SAME expression the per-pixel code used — so every table entry
    // is the bit pattern the convert-and-divide produced, and the output is
    // byte-identical. The pixel loop did the conversion and the division
    // four times per output element (3.3 M times per 640x425 image); now it
    // does four loads.
    let lut: [f32; 256] = std::array::from_fn(|v| f32::from(v as u8) / 255.0);
    // Bilinear as the two passes it is: horizontal on each SOURCE row the
    // output touches, then vertical between two of those. The one-pass form
    // interpolated each source row horizontally TWICE — once as some output
    // row's `y0` and again as the previous row's `y1` — and for an upscale
    // more often still. Same two expressions per element, same operand
    // order, so the bytes are identical; each horizontal result is simply
    // computed once and read twice.
    //
    // Which source rows: `y0` and `y1` of every content row, de-duplicated.
    // For a downscale that is a fraction of the source; never more than `h0`.
    let vert: Vec<(usize, usize, f32)> = (0..nh)
        .map(|y| {
            let sy = ((y as f32 + 0.5) / scale - 0.5).clamp(0.0, (h0 - 1) as f32);
            let y0 = sy.floor() as usize;
            (y0, (y0 + 1).min(h0 - 1), sy - y0 as f32)
        })
        .collect();
    let mut pos = vec![usize::MAX; h0];
    let mut rows_needed: Vec<usize> = Vec::with_capacity(nh + 1);
    for &(y0, y1, _) in &vert {
        for yy in [y0, y1] {
            if pos[yy] == usize::MAX {
                pos[yy] = rows_needed.len();
                rows_needed.push(yy);
            }
        }
    }
    let nrows = rows_needed.len();
    // Pass 1: hbuf[c][k][x] = horizontal lerp of source row rows_needed[k].
    let mut hbuf = vec![0f32; 3 * nrows * nw];
    // Byte offsets of the two taps within a source row, computed once per
    // column rather than once per (row, column): `x * channels` is the same
    // multiply 1 278 times over for every column otherwise.
    let xoff: Vec<(u32, u32, f32)> =
        xs.iter().map(|&(x0, x1, fx)| ((x0 * channels) as u32, (x1 * channels) as u32, fx)).collect();
    let row_bytes = w0 * channels;
    let hfill = |c: usize, yy: usize, dst: &mut [f32]| {
        // Gray8 broadcasts its single channel; Rgba8 drops alpha.
        let src_c = if channels == 1 { 0 } else { c };
        let line = &image.data[yy * row_bytes..(yy + 1) * row_bytes];
        // The tap offsets are `x * channels + src_c` with `x <= w0 - 1` and
        // `src_c < channels`, so every one is below `row_bytes == line.len()`:
        // the bound is established where `xs` is built, which is why the
        // per-element check can go. It was two checked gathers per output
        // (8.2 M Ir on the line census, a third of the pass).
        debug_assert!(xoff.iter().all(|&(a, b, _)| (a as usize + src_c) < line.len() && (b as usize + src_c) < line.len()));
        for (d, &(o0, o1, fx)) in dst.iter_mut().zip(&xoff) {
            // SAFETY: bounds argued just above; `lut` is indexed by a `u8`.
            #[allow(unsafe_code)]
            let (a, b) = unsafe {
                (
                    lut[*line.get_unchecked(o0 as usize + src_c) as usize],
                    lut[*line.get_unchecked(o1 as usize + src_c) as usize],
                )
            };
            *d = a * (1.0 - fx) + b * fx;
        }
    };
    // Pass 2: one output row from two horizontal rows.
    let plane = size_h * size_w;
    let vfill = |y: usize, hb: &[f32], row: &mut [f32]| {
        let (y0, y1, fy) = vert[y];
        let top = &hb[pos[y0] * nw..(pos[y0] + 1) * nw];
        let bot = &hb[pos[y1] * nw..(pos[y1] + 1) * nw];
        let dst = &mut row[pad_x..pad_x + nw];
        for ((d, &t), &b) in dst.iter_mut().zip(top).zip(bot) {
            *d = t * (1.0 - fy) + b * fy;
        }
    };
    if crate::parallel::serial_kernels() {
        for (c, hb) in hbuf.chunks_mut(nrows * nw).enumerate() {
            for (k, dst) in hb.chunks_mut(nw).enumerate() {
                hfill(c, rows_needed[k], dst);
            }
        }
        for (c, chan) in out.chunks_mut(plane).enumerate() {
            let hb = &hbuf[c * nrows * nw..(c + 1) * nrows * nw];
            for (y, row) in chan[pad_y * size_w..(pad_y + nh) * size_w].chunks_mut(size_w).enumerate() {
                vfill(y, hb, row);
            }
        }
    } else {
        // Parallel over (channel, source row), then over (channel, output
        // row). Three channels alone would leave a four-worker pool a
        // quarter idle, and the rows are independent.
        hbuf.par_chunks_mut(nw).enumerate().for_each(|(i, dst)| {
            let (c, k) = (i / nrows, i % nrows);
            hfill(c, rows_needed[k], dst);
        });
        let hbuf = &hbuf;
        out.par_chunks_mut(plane).enumerate().for_each(|(c, chan)| {
            let hb = &hbuf[c * nrows * nw..(c + 1) * nrows * nw];
            chan[pad_y * size_w..(pad_y + nh) * size_w]
                .par_chunks_mut(size_w)
                .enumerate()
                .for_each(|(y, row)| vfill(y, hb, row));
        });
    }

    let tensor = Tensor::from_vec(out, (1, 3, size_h, size_w), device)?;
    let lb = Letterbox {
        scale,
        pad_x: pad_x as f32,
        pad_y: pad_y as f32,
        orig_width: image.width,
        orig_height: image.height,
    };
    Ok((tensor, lb))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, v: u8) -> ImageBuffer {
        ImageBuffer {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            data: vec![v; (w * h * 3) as usize],
        }
    }

    #[test]
    fn square_input_needs_no_padding() {
        let (t, lb) = letterbox(&solid(320, 320, 255), 640, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1, 3, 640, 640]);
        assert_eq!(lb.scale, 2.0);
        assert_eq!((lb.pad_x, lb.pad_y), (0.0, 0.0));
    }

    #[test]
    fn wide_input_pads_top_and_bottom_only() {
        // 640x360 -> scale 1.0, nh 360, pad_y = round((640-360)/2 - 0.1) = 140
        let (_, lb) = letterbox(&solid(640, 360, 128), 640, &Device::Cpu).unwrap();
        assert_eq!(lb.scale, 1.0);
        assert_eq!(lb.pad_x, 0.0);
        assert_eq!(lb.pad_y, 140.0);
    }

    #[test]
    fn padding_uses_114_not_zero() {
        let (t, lb) = letterbox(&solid(640, 360, 255), 640, &Device::Cpu).unwrap();
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // A row inside the pad band, well away from the content.
        let top_left = v[0];
        assert!((top_left - PAD_VALUE).abs() < 1e-6, "pad value {top_left}");
        // A row inside the content band is the image's own value.
        let mid = v[(lb.pad_y as usize + 10) * 640 + 320];
        assert!((mid - 1.0).abs() < 1e-3, "content value {mid}");
    }

    #[test]
    fn inverse_maps_a_padded_box_back_to_the_original() {
        let (_, lb) = letterbox(&solid(640, 360, 128), 640, &Device::Cpu).unwrap();
        // A detection at the top of the content band maps to y = 0.
        let (x, y) = lb.invert(0.0, 140.0);
        assert!((x - 0.0).abs() < 1e-4 && (y - 0.0).abs() < 1e-4, "got ({x}, {y})");
        let (x, y) = lb.invert(640.0, 500.0);
        assert!((x - 640.0).abs() < 1e-4 && (y - 360.0).abs() < 1e-4, "got ({x}, {y})");
    }

    #[test]
    fn grayscale_broadcasts_to_three_channels() {
        let img = ImageBuffer {
            width: 4,
            height: 4,
            format: PixelFormat::Gray8,
            data: vec![200u8; 16],
        };
        let (t, _) = letterbox(&img, 8, &Device::Cpu).unwrap();
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want = 200.0 / 255.0;
        // Centre pixel of each channel plane should carry the same value.
        for c in 0..3 {
            let s = v[c * 64 + 4 * 8 + 4];
            assert!((s - want).abs() < 1e-3, "channel {c} = {s}");
        }
    }
}
