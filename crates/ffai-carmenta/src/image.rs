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
    let long_side = w.max(h) as f32;
    craft_input_scaled(gray, w, h, det_effective_scale(long_side), device)
}

/// `craft_input` with the scale supplied explicitly. Oracles use this: a
/// stage oracle must test the NET against its pinned fixture, never the
/// scaling POLICY — adaptive scaling silently re-scaled the 640x640 fixture
/// to 2x and the oracle caught it, which is the whole point of having one.
pub fn craft_input_scaled(
    gray: &[f32],
    w: usize,
    h: usize,
    requested: f32,
    device: &Device,
) -> Result<(Tensor, f32)> {
    const CANVAS: f32 = 2560.0;
    let long_side = w.max(h) as f32;
    let scale = requested.min(CANVAS / long_side);

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
    crnn_input_patch(&crop, cw, ch, device)
}

/// Is this crop LIGHT INK ON DARK PAPER — white text in a coloured pill, a
/// reversed-out heading, a dark callout box?
///
/// Ink is the MINORITY of the pixels: a line of text covers far less area than
/// the space around it. So split at the mean and ask which side is smaller —
/// if the light pixels are the minority, they are the ink, and the crop is
/// inverted relative to the printed-document convention the CRNN was trained
/// on.
///
/// Mean rather than Otsu on purpose: this decides a binary flip, the two classes
/// are far apart whenever the question is live, and the mean costs one pass
/// where Otsu costs a histogram. §8.145 measured Otsu separability at 0.882 on
/// exactly these crops — they are cleanly bimodal, which is what makes the cheap
/// split safe.
pub fn is_reversed(crop: &[f32], w: usize, h: usize) -> bool {
    if crop.len() < w * h || w < 3 || h < 3 {
        return false;
    }
    // The BORDER is the background: a line box surrounds its text, so the top
    // and bottom rows and the left and right columns are paper almost by
    // construction. Compare that ring against the interior and the polarity
    // falls out — dark ring with a lighter middle means light ink on dark paper.
    //
    // NOT a pixel majority, which was the first cut and was wrong: a tightly
    // cropped BOLD heading is legitimately more than half ink, so the majority
    // rule inverted ordinary dark-on-light text and destroyed it. §8.149 measured
    // that at +0.397 pp on control pages from polarity alone — a step that cannot
    // change how MUCH is read, only what, which is what proved it was misfiring
    // rather than over-reaching.
    let mut ring = 0.0f32;
    let mut nring = 0usize;
    for x in 0..w {
        ring += crop[x] + crop[(h - 1) * w + x];
        nring += 2;
    }
    for y in 1..h - 1 {
        ring += crop[y * w] + crop[y * w + w - 1];
        nring += 2;
    }
    let ring = ring / nring as f32;
    let all = crop.iter().sum::<f32>() / crop.len() as f32;
    // Require a real margin: a crop whose border and interior agree has no
    // decidable polarity, and guessing on it is how the majority rule went wrong.
    ring + 16.0 < all
}

/// The CRNN tensor for an already-extracted patch (straight or straightened).
pub fn crnn_input_patch(crop: &[f32], cw: usize, ch: usize, device: &Device) -> Result<Tensor> {
    // POLARITY (§8.146). The recognizer is trained on dark ink over light paper;
    // a reversed heading is a different image to it, not a stylistic variant.
    // §8.145 measured those crops at mean confidence 0.916 against 0.972
    // upright on the coloured-heading pages, and 0.781 against 0.948 elsewhere,
    // producing 'Let" $ Spell' for "Let's Spell" and single-letter garbage.
    //
    // Gated because it is a change to every crop's preprocessing and the gate is
    // how it gets measured before it becomes the default — the same shape as
    // FFAI_BODY_ONLY and FFAI_REC_LANG.
    // CONTRAST (§8.146). Separability is not the same as dynamic range, and
    // Otsu is scale-invariant so it cannot tell them apart: red-on-orange
    // separates perfectly (eta 0.88) and still lands at ink 0.45 / paper 0.65,
    // a mid-grey mush, where the training distribution is ink 0.1 / paper 0.9.
    // §8.145 measured 6.2 % of crops on the coloured-heading pages below a 0.35
    // span against 1.6 % elsewhere, and those crops read at 0.888 / 0.709 mean
    // confidence against 0.975 / 0.959 for the rest.
    //
    // Stretching to the full range is a no-op on a crop that already spans it,
    // so the gate covers both steps and neither needs its own threshold.
    let norm = std::env::var("FFAI_CROP_NORM").as_deref() == Ok("1");
    let owned: Vec<f32>;
    let crop = if norm {
        let flipped: Vec<f32> = if is_reversed(crop, cw, ch) {
            crop.iter().map(|&p| 255.0 - p).collect()
        } else {
            crop.to_vec()
        };
        // Percentile anchors, not min/max: a single dust speck or blown
        // highlight would otherwise set the whole scale.
        let mut sorted = flipped.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo = sorted[sorted.len() / 50];
        let hi = sorted[sorted.len() - 1 - sorted.len() / 50];
        // Stretch ONLY a crop that is actually compressed (§8.149).
        //
        // The first cut stretched every crop, and a crop already spanning
        // 10..250 still scales by 1.06 — so all 26 631 crops moved when the 6 %
        // with a real problem were the target. §8.148 measured the cost of that:
        // -0.573 pp on the pages it was aimed at, +0.184 pp everywhere else.
        //
        // `FFAI_CROP_NORM_SPAN` is the trigger, as a fraction of full range.
        // Above it the crop is left EXACTLY alone, so the flag cannot perturb a
        // page that never had a contrast problem.
        //
        // A crop with no real range is blank or nearly so; stretching that
        // amplifies sensor noise into glyph-shaped garbage, hence the floor.
        let trigger = std::env::var("FFAI_CROP_NORM_SPAN")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.35)
            * 255.0;
        let span = hi - lo;
        owned = if span > 24.0 && span < trigger {
            let scale = 255.0 / span;
            flipped.iter().map(|&p| ((p - lo) * scale).clamp(0.0, 255.0)).collect()
        } else {
            flipped
        };
        &owned[..]
    } else {
        crop
    };
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

/// ADAPTIVE detection scale (the CORD spike fix): normalize the long side
/// toward a target so small photos get magnified above CRAFT's measured
/// ~8px glyph floor (7 of the 13 worst per-clip spikes were 576x864
/// receipts with ~10px text) and camera-resolution monsters get capped
/// (10.5s / 7.8GB class). `FFAI_DET_TARGET=0` disables (fixed-scale
/// behaviour); FFAI_DET_SCALE still multiplies on top for sweeps.
fn det_target() -> f32 {
    use std::sync::OnceLock;
    static T: OnceLock<f32> = OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("FFAI_DET_TARGET").ok().and_then(|v| v.parse().ok()).unwrap_or(1536.0)
    })
}

pub(crate) fn det_effective_scale(long_side: f32) -> f32 {
    let base = det_scale();
    let t = det_target();
    if t <= 0.0 {
        return base;
    }
    base * (t / long_side).clamp(0.375, 2.0)
}

/// Horizontal ink extent of a single-line strip: columns whose maximum
/// deviation from the strip's median level exceeds a small threshold, padded
/// by one strip-height each side (context for the recognizer). Falls back to
/// the full width when nothing exceeds the threshold.
pub fn ink_extent(gray: &[f32], w: usize, h: usize) -> (usize, usize) {
    let mut sample: Vec<f32> = gray.iter().copied().step_by(97).collect();
    sample.sort_by(|a, b| a.total_cmp(b));
    let median = sample.get(sample.len() / 2).copied().unwrap_or(128.0);
    const DEV: f32 = 25.0;
    let mut first = None;
    let mut last = None;
    for x in 0..w {
        let ink = (0..h).any(|y| (gray[y * w + x] - median).abs() > DEV);
        if ink {
            if first.is_none() {
                first = Some(x);
            }
            last = Some(x);
        }
    }
    match (first, last) {
        (Some(a), Some(b)) => {
            let pad = 2 * h; // two strip-heights of context each side
            (a.saturating_sub(pad), (b + 1 + pad).min(w))
        }
        _ => (0, w),
    }
}

/// Word x-ranges within a single-line band of the ORIGINAL image, by ink
/// column projection: a run of ink-free columns wider than 30% of the band
/// height is a word gap. Replaces map-space splitting for word-level
/// recognizers — CRAFT's region/affinity maps are per-character gaussians
/// whose dips cut words mid-glyph (measured: "October" -> "Oo"+"ccober" on
/// region-only, truncations like "Exa[ctly]" on region|affinity), while
/// image-space gaps at this rendering are unambiguous.
pub fn split_ink_words(
    gray: &[f32],
    img_w: usize,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
) -> Vec<(usize, usize)> {
    let h = y1.saturating_sub(y0).max(1);
    let min_gap = ((h as f32) * 0.30).max(2.0) as usize;
    let mut sample: Vec<f32> = (y0..y1)
        .flat_map(|y| gray[y * img_w + x0..y * img_w + x1].iter().copied())
        .step_by(53)
        .collect();
    sample.sort_by(|a, b| a.total_cmp(b));
    let median = sample.get(sample.len() / 2).copied().unwrap_or(128.0);
    const DEV: f32 = 25.0;
    let ink: Vec<bool> = (x0..x1)
        .map(|x| (y0..y1).any(|y| (gray[y * img_w + x] - median).abs() > DEV))
        .collect();
    let mut words = Vec::new();
    let (mut start, mut gap) = (None::<usize>, 0usize);
    for (i, &on) in ink.iter().enumerate() {
        if on {
            if start.is_none() {
                start = Some(i);
            }
            gap = 0;
        } else if let Some(st) = start {
            gap += 1;
            if gap >= min_gap {
                words.push((x0 + st, x0 + i + 1 - gap));
                start = None;
                gap = 0;
            }
        }
    }
    if let Some(st) = start {
        words.push((x0 + st, x0 + ink.len() - gap.min(ink.len())));
    }
    words
}

/// CRAFT input from the FULL-COLOR frame. The gray path (above) replicates
/// luma to three channels — correct for grayscale sources, but on real
/// photographs it ERASES chroma contrast: red-on-white or stamped text can
/// be near-isoluminant, its region scores collapse, and no threshold
/// recovers it (measured: 63% coverage vs paddle on CORD, bit-identical
/// across threshold sweeps — the threshold-insensitivity was the tell).
/// Same scale/pad/normalize contract as `craft_input`.
pub fn craft_input_color(img: &ImageBuffer, device: &Device) -> Result<(Tensor, f32)> {
    let (w, h) = (img.width as usize, img.height as usize);
    let bpp = img.format.bytes_per_pixel();
    if bpp == 1 {
        let gray = to_gray_f32(img)?;
        return craft_input(&gray, w, h, device);
    }
    const CANVAS: f32 = 2560.0;
    let long_side = w.max(h) as f32;
    let scale = det_effective_scale(long_side).min(CANVAS / long_side);
    let (rw, rh) = (((w as f32 * scale) as usize).max(1), ((h as f32 * scale) as usize).max(1));
    let (cw, ch) = (rw.div_ceil(32) * 32, rh.div_ceil(32) * 32);
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let mut chw = vec![0f32; 3 * cw * ch];
    for c in 0..3 {
        let plane: Vec<f32> = (0..w * h).map(|i| img.data[i * bpp + c] as f32).collect();
        let resized = resize_bilinear(&plane, w, h, rw, rh);
        let (mean, std) = (MEAN[c], STD[c]);
        let dst = &mut chw[c * cw * ch..(c + 1) * cw * ch];
        dst.fill((0.0 - mean) / std);
        for y in 0..rh {
            for x in 0..rw {
                dst[y * cw + x] = (resized[y * rw + x] / 255.0 - mean) / std;
            }
        }
    }
    let t = Tensor::from_vec(chw, (1, 3, ch, cw), device).map_err(candle_err)?;
    Ok((t, scale))
}

/// Input for PP-OCRv5 mobile-det, reproducing `DetResizeForTest`'s policy.
///
/// Returns the tensor and the per-axis scale from source to network pixels,
/// which the postprocess needs to map boxes back. The axes scale
/// independently — paddle resizes straight to the /32-rounded size rather than
/// resizing proportionally and padding, so `sx != sy` in general.
///
/// ## The policy is a MINIMUM side, not a maximum, and that is the whole ball game
///
/// The obvious reading of `inference.yml` — "resize_long: 960", so scale the
/// long side to 960 — is wrong, and wrong by a factor of 17 in pixel count.
/// The reference resizes only to bring the SHORT side UP to `min_side`, leaves
/// larger images alone, and then caps the long side at `max_side`. A 2376x4224
/// receipt therefore reaches the detector at 2240x4000, not 544x960.
///
/// Measured, not inferred: the reference logs
/// `Resized image size (2376x4224) exceeds max_side_limit of 4000`, which can
/// only happen if the ratio was still 1.0 when the cap was applied. Running our
/// port at 544x960 merged an entire receipt into one 1903x1781 blob — and so
/// did paddle's own postprocess on paddle's own probability map at that size.
/// The detector was never the problem; the input was.
///
/// One faithful oddity, deliberate: channels are **BGR**. PaddleOCR decodes BGR
/// and then normalizes with ImageNet's *RGB* statistics, so mean 0.485 lands on
/// the blue channel. It looks like a bug and is not one — it is what these
/// weights were trained against.
pub fn mobiledet_input(
    img: &ImageBuffer,
    min_side: usize,
    device: &Device,
) -> Result<(Tensor, f32, f32)> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];

    let (w, h) = (img.width as usize, img.height as usize);
    let max_side: usize = std::env::var("FFAI_DET_MAX_SIDE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4000);

    // A shape-aware floor was tried here — a smaller `min_side` for band STRIPS
    // (aspect > 4) than for pages — on the theory that LIVE's 1280x45 bands were
    // what a page-shaped floor mishandles. It measured INERT: LIVE stayed at
    // 1.83 % while a global lower floor reaches 1.62 %, which locates the effect
    // in the full-frame calibration call, not in the strips. Reverted rather
    // than kept as a plausible-looking knob that buys nothing. See §8.20.
    let short = w.min(h) as f32;
    let ratio = if (short as usize) < min_side { min_side as f32 / short } else { 1.0 };
    let (mut rw, mut rh) = ((w as f32 * ratio).trunc(), (h as f32 * ratio).trunc());
    if rw.max(rh) > max_side as f32 {
        let cap = max_side as f32 / rw.max(rh);
        rw = (rw * cap).trunc();
        rh = (rh * cap).trunc();
    }
    let round32 = |v: f32| (((v / 32.0).round() as usize) * 32).max(32);
    let (cw, ch) = (round32(rw), round32(rh));

    let bpp = img.format.bytes_per_pixel();
    let mut chw = vec![0f32; 3 * cw * ch];
    for c in 0..3 {
        // Tensor channel 0 is blue; a grey source replicates its single plane.
        let src_c = if bpp == 1 { 0 } else { 2 - c };
        let plane: Vec<f32> = (0..w * h).map(|i| img.data[i * bpp + src_c] as f32).collect();
        let resized = resize_bilinear(&plane, w, h, cw, ch);
        let (mean, std) = (MEAN[c], STD[c]);
        let dst = &mut chw[c * cw * ch..(c + 1) * cw * ch];
        for (d, &s) in dst.iter_mut().zip(&resized) {
            *d = (s / 255.0 - mean) / std;
        }
    }
    let t = Tensor::from_vec(chw, (1, 3, ch, cw), device).map_err(candle_err)?;
    Ok((t, cw as f32 / w as f32, ch as f32 / h as f32))
}

pub fn candle_err(e: candle_core::Error) -> Error {
    Error::Other(format!("candle: {e}"))
}

/// Bridge candle results into ffai results at the module boundary.
pub fn ok<T>(r: CandleResult<T>) -> Result<T> {
    r.map_err(candle_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The BORDER is the background, not the pixel majority — the distinction
    /// that cost +0.397 pp on control pages before it was found (§8.149).
    #[test]
    fn reversed_polarity_detection() {
        // 10x10, light paper with a dark glyph in the middle.
        let mut normal = vec![240.0f32; 100];
        for y in 3..7 { for x in 3..7 { normal[y * 10 + x] = 20.0; } }
        assert!(!is_reversed(&normal, 10, 10));

        // The SAME polarity with MORE ink than paper — a bold glyph. A pixel
        // majority calls this reversed and destroys it; the border does not.
        let mut bold = vec![240.0f32; 100];
        for y in 1..9 { for x in 1..9 { bold[y * 10 + x] = 20.0; } }
        assert!(bold.iter().filter(|&&v| v < 128.0).count() * 2 > bold.len());
        assert!(!is_reversed(&bold, 10, 10));

        // Genuinely reversed: dark border, light glyph.
        let mut rev = vec![20.0f32; 100];
        for y in 3..7 { for x in 3..7 { rev[y * 10 + x] = 240.0; } }
        assert!(is_reversed(&rev, 10, 10));

        // Degenerate input must not panic or claim a polarity.
        assert!(!is_reversed(&[], 0, 0));
        assert!(!is_reversed(&[128.0; 100], 10, 10));
    }

}
