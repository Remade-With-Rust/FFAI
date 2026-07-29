//! Image preprocessing for the det+rec core: grayscale conversion, bilinear
//! resize, CRAFT normalization, CRNN crop preparation. All pure CPU-side
//! math on `f32` buffers; tensors are built at the boundary.

use candle_core::{Device, Result as CandleResult, Tensor};
use ffai_core::error::{Error, Result};
use ffai_core::types::{ImageBuffer, PixelFormat};

/// Grayscale plane (0..255 f32) from any supported pixel format, ITU-R 601
/// luma coefficients — the same conversion cv2's BGR2GRAY applies in the
/// reference pipeline.
pub fn to_gray_f32(img: &ImageBuffer) -> Result<Vec<f32>> {
    let n = (img.width * img.height) as usize;
    let data = &img.data;
    Ok(match img.format {
        PixelFormat::Gray8 => data.iter().map(|&p| p as f32).collect(),
        PixelFormat::Rgb8 => (0..n)
            .map(|i| {
                0.299 * data[3 * i] as f32 + 0.587 * data[3 * i + 1] as f32 + 0.114 * data[3 * i + 2] as f32
            })
            .collect(),
        PixelFormat::Rgba8 => (0..n)
            .map(|i| {
                0.299 * data[4 * i] as f32 + 0.587 * data[4 * i + 1] as f32 + 0.114 * data[4 * i + 2] as f32
            })
            .collect(),
    })
}

/// Bilinear resize of a single-channel plane, align_corners=false semantics
/// (the PyTorch/OpenCV default this whole lineage was trained under).
pub fn resize_bilinear(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    let mut out = vec![0f32; dw * dh];
    for oy in 0..dh {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(sh - 1);
        let y1 = (y0 + 1).min(sh - 1);
        let wy = fy - fy.floor();
        for ox in 0..dw {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(sw - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let wx = fx - fx.floor();
            let top = src[y0 * sw + x0] * (1.0 - wx) + src[y0 * sw + x1] * wx;
            let bot = src[y1 * sw + x0] * (1.0 - wx) + src[y1 * sw + x1] * wx;
            out[oy * dw + ox] = top * (1.0 - wy) + bot * wy;
        }
    }
    out
}

/// Catmull-Rom bicubic resize of a single-channel plane (a = -0.5), the
/// sharpness class of the reference stack's Lanczos crop resizing. Used on
/// the recognition path only; detection keeps bilinear, matching its own
/// reference preprocessing.
pub fn resize_bicubic(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    fn kernel(x: f32) -> f32 {
        const A: f32 = -0.5;
        let x = x.abs();
        if x < 1.0 {
            (A + 2.0) * x * x * x - (A + 3.0) * x * x + 1.0
        } else if x < 2.0 {
            A * x * x * x - 5.0 * A * x * x + 8.0 * A * x - 4.0 * A
        } else {
            0.0
        }
    }
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    let clamp = |v: i64, n: usize| v.clamp(0, n as i64 - 1) as usize;
    let mut out = vec![0f32; dw * dh];
    for oy in 0..dh {
        let fy = (oy as f32 + 0.5) * sy - 0.5;
        let y0 = fy.floor() as i64;
        let wy: [f32; 4] = std::array::from_fn(|k| kernel(fy - (y0 - 1 + k as i64) as f32));
        for ox in 0..dw {
            let fx = (ox as f32 + 0.5) * sx - 0.5;
            let x0 = fx.floor() as i64;
            let wx: [f32; 4] = std::array::from_fn(|k| kernel(fx - (x0 - 1 + k as i64) as f32));
            let mut acc = 0f32;
            for (ky, wyv) in wy.iter().enumerate() {
                let row = clamp(y0 - 1 + ky as i64, sh) * sw;
                for (kx, wxv) in wx.iter().enumerate() {
                    acc += src[row + clamp(x0 - 1 + kx as i64, sw)] * wyv * wxv;
                }
            }
            out[oy * dw + ox] = acc;
        }
    }
    out
}

/// CRAFT input from a grayscale plane: optional scale (canvas 2560, mag 1.0 —
/// EasyOCR's detect defaults), pad to multiples of 32 with black, replicate
/// grey to RGB, normalize per channel with ImageNet mean/std.
///
/// Returns the (1,3,H,W) tensor and the scale that maps ORIGINAL image
/// coordinates to tensor coordinates (boxes divide by it on the way back).
pub fn craft_input(gray: &[f32], w: usize, h: usize, device: &Device) -> Result<(Tensor, f32)> {
    const CANVAS: f32 = 2560.0;
    let long_side = w.max(h) as f32;
    let scale = det_scale().min(CANVAS / long_side);

    let (rw, rh) = (((w as f32 * scale) as usize).max(1), ((h as f32 * scale) as usize).max(1));
    let resized = resize_bilinear(gray, w, h, rw, rh);
    let (cw, ch) = (rw.div_ceil(32) * 32, rh.div_ceil(32) * 32);

    // ImageNet mean/std per channel, applied to the grey value replicated to
    // R=G=B; padding is black (0), normalized like any other pixel — exactly
    // what the reference does (pads raw, then normalizes the whole canvas).
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let mut chw = vec![0f32; 3 * cw * ch];
    for (c, plane) in chw.chunks_exact_mut(cw * ch).enumerate() {
        let (mean, std) = (MEAN[c], STD[c]);
        let pad = (0.0 - mean) / std;
        plane.fill(pad);
        for y in 0..rh {
            for x in 0..rw {
                plane[y * cw + x] = (resized[y * rw + x] / 255.0 - mean) / std;
            }
        }
    }
    let t = Tensor::from_vec(chw, (1, 3, ch, cw), device).map_err(candle_err)?;
    Ok((t, scale))
}

/// CRNN input from a grayscale plane crop: resize to height 64 preserving
/// aspect, normalize (x/255 - 0.5)/0.5. Returns (1,1,64,W).
pub fn crnn_input(
    gray: &[f32],
    img_w: usize,
    img_h: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    device: &Device,
) -> Result<Tensor> {
    let (x1, y1) = (x1.min(img_w), y1.min(img_h));
    if x1 <= x0 + 1 || y1 <= y0 + 1 {
        return Err(Error::Other("degenerate crop".into()));
    }
    let (cw, ch) = (x1 - x0, y1 - y0);
    let mut crop = vec![0f32; cw * ch];
    for y in 0..ch {
        crop[y * cw..(y + 1) * cw].copy_from_slice(&gray[(y0 + y) * img_w + x0..(y0 + y) * img_w + x1]);
    }
    let out_w = ((cw as f32 * 64.0 / ch as f32).round() as usize).max(8);
    // Bicubic, not bilinear: recognition crops UPSCALE (~25 px lines to
    // h=64), and bilinear smears single-dot glyphs — measured as a
    // systematic '.'->':' / '.'->'' confusion on the train split. The
    // reference stack resizes crops with PIL Lanczos; Catmull-Rom is the
    // same sharpness class.
    let resized = resize_bicubic(&crop, cw, ch, out_w, 64);
    let norm: Vec<f32> = resized.iter().map(|&p| (p / 255.0 - 0.5) / 0.5).collect();
    Tensor::from_vec(norm, (1, 1, 64, out_w), device).map_err(candle_err)
}

/// Detection scale (default 1.0 = native resolution, the reference's
/// mag_ratio). `FFAI_DET_SCALE` overrides for the speed campaign's sweeps:
/// the profiler puts CRAFT's forward at 62-89% of recognition time and VGG
/// cost is ~linear in pixels, so half scale is ~4x less detection work.
/// Recognition always crops from the ORIGINAL image, so this trades
/// detection recall only — gated on corpus CER like every knob.
fn det_scale() -> f32 {
    use std::sync::OnceLock;
    static SCALE: OnceLock<f32> = OnceLock::new();
    *SCALE.get_or_init(|| {
        std::env::var("FFAI_DET_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0)
    })
}

pub fn candle_err(e: candle_core::Error) -> Error {
    Error::Other(format!("candle: {e}"))
}

/// Bridge candle results into ffai results at the module boundary.
pub fn ok<T>(r: CandleResult<T>) -> Result<T> {
    r.map_err(candle_err)
}
