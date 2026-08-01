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
/// Returns the tensor and the transform needed to map detections back.
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
    let mut out = vec![PAD_VALUE; 3 * size_h * size_w];
    // Nearest-neighbour would be cheaper and visibly worse; bilinear is
    // what the reference resize does.
    for y in 0..nh {
        // Source coordinate of this destination row's centre.
        let sy = ((y as f32 + 0.5) / scale - 0.5).clamp(0.0, (h0 - 1) as f32);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(h0 - 1);
        let fy = sy - y0 as f32;
        for x in 0..nw {
            let sx = ((x as f32 + 0.5) / scale - 0.5).clamp(0.0, (w0 - 1) as f32);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(w0 - 1);
            let fx = sx - x0 as f32;
            let dst = (y + pad_y) * size_w + (x + pad_x);
            for c in 0..3 {
                // Gray8 broadcasts its single channel; Rgba8 drops alpha.
                let src_c = if channels == 1 { 0 } else { c };
                let at = |yy: usize, xx: usize| -> f32 {
                    image.data[(yy * w0 + xx) * channels + src_c] as f32 / 255.0
                };
                let top = at(y0, x0) * (1.0 - fx) + at(y0, x1) * fx;
                let bot = at(y1, x0) * (1.0 - fx) + at(y1, x1) * fx;
                out[c * size_h * size_w + dst] = top * (1.0 - fy) + bot * fy;
            }
        }
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
