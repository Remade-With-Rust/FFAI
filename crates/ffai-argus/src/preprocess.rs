//! Image -> `pixel_values`: the content path between a decoded file and the
//! vision tower.
//!
//! Steps 3 and 4 deliberately fed the tower the reference's own
//! `pixel_values`, so that a tensor mismatch could be attributed to the tower
//! rather than to a resize. This module is the brick they isolated out, and it
//! gets its own gate for the same reason.
//!
//! # The spec, pinned from `preprocessor_config.json`
//!
//! | step | value |
//! |---|---|
//! | convert to RGB | `do_convert_rgb: true` |
//! | resize longest edge | **2048** |
//! | resample filter | **`1` = LANCZOS** |
//! | split into tiles | `max_image_size.longest_edge: 512` |
//! | plus a global thumbnail | -> **17** tiles for a square image |
//! | rescale | `1/255` |
//! | normalize | mean 0.5, std 0.5 -> `[-1, 1]` |
//!
//! That arithmetic is what produces 17 tiles: the image is scaled UP to
//! 2048x2048, cut into sixteen 512x512 tiles, and a thumbnail is appended.
//!
//! # Why Lanczos is written out rather than substituted
//!
//! The tree already has `carmenta::image::resize_bilinear` and a Catmull-Rom
//! `resize_bicubic`, each chosen for a measured reason in its own path. Neither
//! is this one. Substituting a different filter changes every `pixel_values`
//! and therefore every tensor downstream — silently, because the output is
//! still a plausible image. So the filter the reference declares is the filter
//! implemented, and it is gated against the reference's own tensor.
//!
//! The convention matters as much as the kernel. PIL's resampler, which is
//! what `resample: 1` means here:
//!
//! * maps output pixel `i` to input centre `(i + 0.5) * scale`, where
//!   `scale = in / out` — a HALF-PIXEL-CENTRED mapping. Using `i * scale`
//!   instead shifts the whole image by half a pixel and is the classic
//!   off-by-half that survives visual inspection;
//! * widens the kernel when DOWNSCALING (`filter_scale = max(1, scale)`) so
//!   the filter low-passes rather than aliases, and leaves it at 1 when
//!   upscaling;
//! * normalises the weights to sum to 1, so brightness is preserved;
//! * resamples horizontally then vertically — separable, and it is what makes
//!   the cost `O(w*h*support)` instead of `O(w*h*support^2)`.

use crate::par::prelude::*;

/// Lanczos-3 kernel: `sinc(x) * sinc(x/3)`, zero outside `|x| < 3`.
#[must_use]
pub fn lanczos3(x: f32) -> f32 {
    const A: f32 = 3.0;
    if x == 0.0 {
        return 1.0;
    }
    let ax = x.abs();
    if ax >= A {
        return 0.0;
    }
    let px = std::f32::consts::PI * x;
    // sinc(x) * sinc(x/a), written as the product of two normalised sincs.
    (px.sin() / px) * ((px / A).sin() / (px / A))
}

/// Precomputed weights for one output axis.
struct Taps {
    /// For each output index: the first input index its window touches.
    starts: Vec<usize>,
    /// Flattened weights, `width` per output index.
    weights: Vec<f32>,
    width: usize,
}

/// Build the resampling taps for one axis, PIL-style.
fn build_taps(src: usize, dst: usize) -> Taps {
    let scale = src as f32 / dst as f32;
    // Widen the kernel only when shrinking. Upscaling keeps support 3.
    let filter_scale = if scale < 1.0 { 1.0 } else { scale };
    let support = 3.0 * filter_scale;
    let width = (support.ceil() as usize) * 2 + 1;

    let mut starts = Vec::with_capacity(dst);
    let mut weights = vec![0.0f32; dst * width];
    for i in 0..dst {
        // HALF-PIXEL CENTRED — see the module docs. `i * scale` here is the
        // half-pixel shift that looks fine and is wrong.
        let center = (i as f32 + 0.5) * scale;
        let xmin = ((center - support + 0.5).floor().max(0.0)) as usize;
        let xmax = (((center + support + 0.5).floor()) as usize).min(src);
        let n = xmax.saturating_sub(xmin);
        let row = &mut weights[i * width..i * width + width];
        let mut sum = 0.0f32;
        for (k, slot) in row.iter_mut().take(n.min(width)).enumerate() {
            let x = (xmin + k) as f32 - center + 0.5;
            let w = lanczos3(x / filter_scale);
            *slot = w;
            sum += w;
        }
        // Normalise so the filter preserves brightness. A kernel that sums to
        // 0.98 darkens the image by 2 % everywhere, which no visual check
        // catches and every tensor comparison does.
        if sum != 0.0 {
            for w in row.iter_mut().take(n.min(width)) {
                *w /= sum;
            }
        }
        starts.push(xmin);
    }
    Taps {
        starts,
        weights,
        width,
    }
}

/// Separable Lanczos resize of an interleaved `channels`-plane image.
///
/// Input and output are `f32` in whatever units the caller uses; this does no
/// rescaling of its own, so it can be applied before or after normalisation
/// without changing meaning.
#[must_use]
pub fn resize_lanczos(
    src: &[f32],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    channels: usize,
) -> Vec<f32> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    // Horizontal pass: (sh, sw) -> (sh, dw).
    let hx = build_taps(sw, dw);
    let mut mid = vec![0.0f32; sh * dw * channels];
    for y in 0..sh {
        for x in 0..dw {
            let start = hx.starts[x];
            let row = &hx.weights[x * hx.width..x * hx.width + hx.width];
            for c in 0..channels {
                let mut acc = 0.0f32;
                for (k, &w) in row.iter().enumerate() {
                    if w == 0.0 {
                        continue;
                    }
                    let sx = (start + k).min(sw - 1);
                    acc += w * src[(y * sw + sx) * channels + c];
                }
                mid[(y * dw + x) * channels + c] = acc;
            }
        }
    }
    // Vertical pass: (sh, dw) -> (dh, dw).
    let vy = build_taps(sh, dh);
    let mut out = vec![0.0f32; dh * dw * channels];
    for y in 0..dh {
        let start = vy.starts[y];
        let row = &vy.weights[y * vy.width..y * vy.width + vy.width];
        for x in 0..dw {
            for c in 0..channels {
                let mut acc = 0.0f32;
                for (k, &w) in row.iter().enumerate() {
                    if w == 0.0 {
                        continue;
                    }
                    let sy = (start + k).min(sh - 1);
                    acc += w * mid[(sy * dw + x) * channels + c];
                }
                out[(y * dw + x) * channels + c] = acc;
            }
        }
    }
    out
}

/// The tile grid a given image size produces, before the thumbnail.
///
/// Returned as `(rows, cols)` so the caller can build the `<row_r_col_c>`
/// markers — the prompt assembly and the pixel tiling MUST agree, and deriving
/// both from one function is what keeps them agreeing.
#[must_use]
pub const fn tile_grid(width: usize, height: usize, max_edge: usize) -> (usize, usize) {
    if width <= max_edge && height <= max_edge {
        return (0, 0);
    }
    (height.div_ceil(max_edge), width.div_ceil(max_edge))
}

/// Longest-edge resize target, preserving aspect ratio.
#[must_use]
pub fn fit_longest_edge(width: usize, height: usize, longest: usize) -> (usize, usize) {
    if width.max(height) == longest {
        return (width, height);
    }
    let scale = longest as f32 / width.max(height) as f32;
    (
        ((width as f32 * scale).round() as usize).max(1),
        ((height as f32 * scale).round() as usize).max(1),
    )
}

/// Rescale to `[0,1]` then normalize to `[-1,1]`, matching
/// `rescale_factor 1/255` with mean and std 0.5.
#[must_use]
pub fn normalize_u8(pixels: &[u8]) -> Vec<f32> {
    // (x/255 - 0.5) / 0.5 == x/127.5 - 1. Written as the fused form because
    // the two-step version rounds twice in f32 for no benefit.
    pixels.iter().map(|&v| f32::from(v) / 127.5 - 1.0).collect()
}


/// PIL's PRECISION_BITS: `32 - 8 - 2`. Coefficients are held as `i32` scaled
/// by `1 << 22`, which leaves room for a u8 sample times a full kernel without
/// overflowing i32.
const PRECISION_BITS: i32 = 32 - 8 - 2;

/// Quantise one axis' normalised f64 weights to PIL's fixed-point form.
///
/// PIL rounds AWAY FROM ZERO (`+0.5` for positives, `-0.5` for negatives)
/// rather than using `round()`'s banker-ish behaviour on ties. Lanczos weights
/// are frequently negative — that is what makes it sharpen — so the negative
/// branch is not a formality.
fn quantise(w: f64) -> i32 {
    let scaled = w * f64::from(1 << PRECISION_BITS);
    if w < 0.0 {
        (scaled - 0.5) as i32
    } else {
        (scaled + 0.5) as i32
    }
}

/// Taps in PIL's fixed-point form, computed in f64 then quantised.
struct FixedTaps {
    starts: Vec<usize>,
    lens: Vec<usize>,
    k: Vec<i32>,
    ksize: usize,
}

fn build_fixed_taps(src: usize, dst: usize) -> FixedTaps {
    // f64 throughout, matching PIL: the coefficients are computed and
    // NORMALISED in double, and only then quantised. Normalising after
    // quantisation would make the weights sum to slightly off 1<<22 and tint
    // the whole image.
    let scale = src as f64 / dst as f64;
    let filter_scale = if scale < 1.0 { 1.0 } else { scale };
    let support = 3.0 * filter_scale;
    let ksize = (support.ceil() as usize) * 2 + 1;

    let mut starts = Vec::with_capacity(dst);
    let mut lens = Vec::with_capacity(dst);
    let mut k = vec![0i32; dst * ksize];
    let inv = 1.0 / filter_scale;

    for xx in 0..dst {
        let center = (xx as f64 + 0.5) * scale;
        // `(int)(v + 0.5)` in C truncates toward zero; both values are
        // non-negative here after the clamp, so this matches.
        let xmin = ((center - support + 0.5) as isize).max(0) as usize;
        let xmax = (((center + support + 0.5) as isize).max(0) as usize).min(src);
        let n = xmax.saturating_sub(xmin);

        let mut w = vec![0.0f64; ksize];
        let mut ww = 0.0f64;
        for (x, slot) in w.iter_mut().enumerate().take(n) {
            let v = lanczos3_f64(((x + xmin) as f64 - center + 0.5) * inv);
            *slot = v;
            ww += v;
        }
        if ww != 0.0 {
            for slot in w.iter_mut().take(n) {
                *slot /= ww;
            }
        }
        for (x, &v) in w.iter().enumerate() {
            k[xx * ksize + x] = quantise(v);
        }
        starts.push(xmin);
        lens.push(n);
    }
    FixedTaps {
        starts,
        lens,
        k,
        ksize,
    }
}

/// f64 Lanczos-3, for coefficient computation.
fn lanczos3_f64(x: f64) -> f64 {
    const A: f64 = 3.0;
    if x == 0.0 {
        return 1.0;
    }
    if x.abs() >= A {
        return 0.0;
    }
    let px = std::f64::consts::PI * x;
    (px.sin() / px) * ((px / A).sin() / (px / A))
}

/// `clamp(acc >> PRECISION_BITS, 0, 255)` — PIL's `clip8`, which is a lookup
/// table over the same arithmetic.
const fn clip8(acc: i32) -> u8 {
    let v = acc >> PRECISION_BITS;
    if v <= 0 {
        0
    } else if v >= 255 {
        255
    } else {
        v as u8
    }
}

/// **PIL-faithful** Lanczos resize on `u8` samples.
///
/// This is the one the content path uses. The `f32` [`resize_lanczos`] above
/// remains as the readable scalar twin and as the thing that proves the
/// CONVENTIONS (half-pixel centres, kernel widening, weight normalisation) —
/// but it is not what the reference computes, and the difference is not
/// academic: it left a residual of exactly one quantisation level, which
/// **flipped a generated token at step 5**.
///
/// Three details carry that difference, and each is invisible in a
/// floating-point idealisation:
///
/// 1. **Coefficients are quantised to `i32` at `1 << 22`** after being
///    normalised in `f64`.
/// 2. **Accumulation is integer**, seeded with `1 << (PRECISION_BITS - 1)` so
///    the final shift rounds instead of truncating.
/// 3. **The intermediate is `u8`.** PIL runs the horizontal pass into an
///    8-bit image and then resamples THAT vertically — so the round-off
///    happens twice, on purpose. Keeping f32 between the passes is "more
///    accurate" and produces a different picture.
#[must_use]
pub fn resize_lanczos_u8(
    src: &[u8],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    channels: usize,
) -> Vec<u8> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    // Horizontal: (sh, sw) -> (sh, dw), quantised to u8.
    let mid: Vec<u8> = if sw == dw {
        src.to_vec()
    } else {
        let t = build_fixed_taps(sw, dw);
        let mut out = vec![0u8; sh * dw * channels];
        // One task per output row. Rows share only immutable inputs, so this
        // is bit-identical to the serial loop — the arithmetic within a row is
        // untouched and rows never interact.
        out.par_chunks_mut(dw * channels)
            .enumerate()
            .for_each(|(y, orow)| {
                for x in 0..dw {
                    let (start, n) = (t.starts[x], t.lens[x]);
                    let row = &t.k[x * t.ksize..x * t.ksize + t.ksize];
                    for c in 0..channels {
                        let mut acc: i32 = 1 << (PRECISION_BITS - 1);
                        for (kk, &w) in row.iter().enumerate().take(n) {
                            acc += i32::from(src[(y * sw + start + kk) * channels + c]) * w;
                        }
                        orow[x * channels + c] = clip8(acc);
                    }
                }
            });
        out
    };
    if sh == dh {
        return mid;
    }
    // Vertical: (sh, dw) -> (dh, dw), on the u8 intermediate.
    let t = build_fixed_taps(sh, dh);
    let mut out = vec![0u8; dh * dw * channels];
    out.par_chunks_mut(dw * channels)
        .enumerate()
        .for_each(|(y, orow)| {
            let (start, n) = (t.starts[y], t.lens[y]);
            let row = &t.k[y * t.ksize..y * t.ksize + t.ksize];
            for x in 0..dw {
                for c in 0..channels {
                    let mut acc: i32 = 1 << (PRECISION_BITS - 1);
                    for (kk, &w) in row.iter().enumerate().take(n) {
                        acc += i32::from(mid[((start + kk) * dw + x) * channels + c]) * w;
                    }
                    orow[x * channels + c] = clip8(acc);
                }
            }
        });
    out
}

/// The vision encoder wants both dimensions to be exact multiples of the tile
/// size, so the split is a clean grid rather than a grid plus an odd remainder.
///
/// This is `Idefics3ImageProcessor.resize_for_vision_encoder`, and its
/// asymmetry is deliberate: the LONGER edge is rounded up first, the shorter
/// edge is derived from the aspect ratio and THEN rounded up. Rounding both
/// independently would distort a non-square image differently at each size.
#[must_use]
pub fn vision_encoder_size(width: usize, height: usize, tile: usize) -> (usize, usize) {
    let aspect = width as f64 / height as f64;
    if width >= height {
        let w = width.div_ceil(tile) * tile;
        let h = ((w as f64 / aspect) as usize).div_ceil(tile) * tile;
        (w, h.max(tile))
    } else {
        let h = height.div_ceil(tile) * tile;
        let w = ((h as f64 * aspect) as usize).div_ceil(tile) * tile;
        (w.max(tile), h)
    }
}

/// One image, preprocessed into what the tower and the prompt both need.
///
/// `rows`/`cols` are carried alongside the pixels precisely because the prompt
/// assembly must emit `<row_r_col_c>` markers that agree with this tiling. Two
/// independent derivations of the same grid is how they silently disagree.
pub struct Preprocessed {
    /// `(tiles, 3, tile, tile)` planar CHW, normalized to `[-1, 1]`.
    pub pixel_values: Vec<f32>,
    pub tiles: usize,
    pub rows: usize,
    pub cols: usize,
    pub tile: usize,
}

/// The size an image is resized to before tiling — step 2 of the content path.
///
/// Reported rather than recomputed by callers because the two-step rule is not
/// guessable from the output: the longest edge goes to 2048 FIRST, and only
/// then is each edge rounded up to a tile multiple. A viewer told "17 tiles"
/// and shown a 512x512 source has no way to see where the 2048 came from.
#[must_use]
pub fn resized_size(width: usize, height: usize) -> (usize, usize) {
    let (aw, ah) = fit_longest_edge(width, height, 2048);
    vision_encoder_size(aw, ah, 512)
}

/// The tile geometry an image WOULD get, without touching a pixel.
///
/// Returns `(tiles, rows, cols)`. Every step here is arithmetic on two
/// integers, so it costs nothing — which is the point: the caller can price a
/// prompt before committing to the resizes and the vision tower that would
/// produce it. Finding out a prompt is too long *after* running the tower over
/// two hundred frames is four minutes to learn something derivable up front.
#[must_use]
pub fn tile_geometry(width: usize, height: usize, split: bool) -> (usize, usize, usize) {
    const LONGEST: usize = 2048;
    const TILE: usize = 512;
    let (aw, ah) = fit_longest_edge(width, height, LONGEST);
    let (bw, bh) = vision_encoder_size(aw, ah, TILE);
    let (rows, cols) = if split { tile_grid(bw, bh, TILE) } else { (0, 0) };
    (rows * cols + 1, rows, cols)
}

/// Decoded RGB8 -> `pixel_values`, the whole Idefics3 content path.
///
/// The ORDER is the part worth stating, because it is not the obvious one and
/// each step changes the pixels:
///
/// 1. resize so the longest edge is `longest` (2048) — this UPSCALES a small
///    image, which is why a 512x512 input yields 17 tiles rather than 1;
/// 2. resize again so both edges are multiples of `tile` — usually a no-op
///    after step 1 for common aspect ratios, and never one for odd ones;
/// 3. cut the exact `rows x cols` grid of tiles;
/// 4. append a global thumbnail, which is the step-2 image resized DOWN to one
///    tile — **not** the original. That distinction was found the expensive
///    way: assuming the original left a residual of exactly one quantisation
///    level, which is too small to look wrong and too large to be noise.
///
/// Rescale/normalize come last, on `u8`, so every resize happens in the domain
/// PIL resizes in.
#[must_use]
pub fn preprocess_rgb8(rgb: &[u8], width: usize, height: usize) -> Preprocessed {
    preprocess_rgb8_opts(rgb, width, height, true)
}

/// The same path, with tile splitting optional.
///
/// # Why video turns splitting OFF
///
/// Splitting is what makes a still image legible: seventeen 512x512 tiles at
/// 64 tokens each is **1088 image tokens**, and the model reads fine print
/// because it sees the page at 2048px. For video that arithmetic inverts. The
/// text tower's `max_position_embeddings` is **8192**, so a split frame caps a
/// window at seven frames before the prompt does not fit at all — and seven
/// frames is not a window, it is a slideshow with a memory problem.
///
/// Unsplit, a frame is ONE tile: **64 tokens**. The same 8192 positions then
/// hold a hundred frames. Sixteen frames of temporal context beats one frame
/// of fine print when the question is "what happens in this clip", and the
/// reference implementations make the same trade for the same reason.
///
/// The unsplit tile is **exactly the global thumbnail the split path already
/// produces** — the same two resizes, the same final 512x512 — rather than a
/// second, subtly different route to a small image. That matters: the
/// thumbnail is gated bit-exactly against the reference (§16), so the video
/// path inherits that gate instead of needing its own.
#[must_use]
pub fn preprocess_rgb8_opts(
    rgb: &[u8],
    width: usize,
    height: usize,
    split: bool,
) -> Preprocessed {
    const LONGEST: usize = 2048;
    const TILE: usize = 512;

    let (aw, ah) = fit_longest_edge(width, height, LONGEST);
    let a = resize_lanczos_u8(rgb, width, height, aw, ah, 3);
    let (bw, bh) = vision_encoder_size(aw, ah, TILE);
    let b = resize_lanczos_u8(&a, aw, ah, bw, bh, 3);

    let (rows, cols) = if split {
        tile_grid(bw, bh, TILE)
    } else {
        // rows = cols = 0 is the shape `PromptLayout::image_block` already
        // encodes as "thumbnail only, no grid" — so the prompt and the pixels
        // agree here for the same reason they agree in the split case: one
        // derivation, used by both.
        (0, 0)
    };
    let per = TILE * TILE * 3;
    let tiles = rows * cols + 1;
    let mut pixel_values = vec![0.0f32; tiles * per];

    let mut write_tile = |idx: usize, src: &[u8]| {
        let norm = normalize_u8(src);
        // Interleaved HWC -> planar CHW: the tower's layout.
        let base = idx * per;
        for c in 0..3 {
            for i in 0..TILE * TILE {
                pixel_values[base + c * TILE * TILE + i] = norm[i * 3 + c];
            }
        }
    };

    let mut tile_buf = vec![0u8; per];
    for r in 0..rows {
        for c in 0..cols {
            for y in 0..TILE {
                let s = ((r * TILE + y) * bw + c * TILE) * 3;
                tile_buf[y * TILE * 3..(y + 1) * TILE * 3].copy_from_slice(&b[s..s + TILE * 3]);
            }
            write_tile(r * cols + c, &tile_buf);
        }
    }
    // The thumbnail is LAST, matching the prompt's `<global-img>` placement
    // after every `<row_r_col_c>` block.
    let thumb = resize_lanczos_u8(&b, bw, bh, TILE, TILE, 3);
    write_tile(rows * cols, &thumb);

    Preprocessed {
        pixel_values,
        tiles,
        rows,
        cols,
        tile: TILE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kernel_is_one_at_zero_and_zero_at_the_integers() {
        assert!((lanczos3(0.0) - 1.0).abs() < 1e-6);
        // sinc has zeros at every non-zero integer inside the support.
        for k in [1.0f32, 2.0] {
            assert!(lanczos3(k).abs() < 1e-5, "lanczos3({k}) should vanish");
            assert!(lanczos3(-k).abs() < 1e-5);
        }
        // …and nothing outside the support.
        assert_eq!(lanczos3(3.0), 0.0);
        assert_eq!(lanczos3(4.5), 0.0);
    }

    #[test]
    fn weights_sum_to_one_at_every_output_position() {
        // Brightness preservation, the property a visual check cannot see.
        for (src, dst) in [(512, 2048), (2048, 512), (300, 512), (512, 300)] {
            let t = build_taps(src, dst);
            for i in 0..dst {
                let s: f32 = t.weights[i * t.width..(i + 1) * t.width].iter().sum();
                assert!(
                    (s - 1.0).abs() < 1e-4,
                    "{src}->{dst} position {i} sums to {s}"
                );
            }
        }
    }

    #[test]
    fn a_constant_image_survives_resizing_unchanged() {
        // The strongest cheap invariant: normalised weights times a constant
        // must give that constant back, at any scale, in both directions.
        for (sw, sh, dw, dh) in [(16, 16, 64, 64), (64, 64, 16, 16), (10, 20, 33, 7)] {
            let src = vec![0.37f32; sw * sh * 3];
            let out = resize_lanczos(&src, sw, sh, dw, dh, 3);
            assert_eq!(out.len(), dw * dh * 3);
            let worst = out.iter().map(|v| (v - 0.37).abs()).fold(0.0f32, f32::max);
            assert!(worst < 1e-4, "{sw}x{sh}->{dw}x{dh} drifted by {worst}");
        }
    }

    #[test]
    fn an_identity_resize_is_a_copy() {
        let src: Vec<f32> = (0..48).map(|i| i as f32).collect();
        assert_eq!(resize_lanczos(&src, 4, 4, 4, 4, 3), src);
    }

    #[test]
    fn the_grid_is_what_produced_seventeen_tiles() {
        // 512 -> upscaled to 2048 -> 4x4, plus the thumbnail = 17.
        assert_eq!(fit_longest_edge(512, 512, 2048), (2048, 2048));
        assert_eq!(tile_grid(2048, 2048, 512), (4, 4));
        assert_eq!(4 * 4 + 1, 17);
        // A small image is thumbnail-only, which the prompt assembly encodes
        // as rows = cols = 0.
        assert_eq!(tile_grid(400, 300, 512), (0, 0));
    }

    #[test]
    fn normalisation_lands_on_minus_one_to_one() {
        let v = normalize_u8(&[0, 255, 128]);
        assert!((v[0] + 1.0).abs() < 1e-6);
        assert!((v[1] - 1.0).abs() < 1e-6);
        assert!(v[2].abs() < 0.01);
    }

    #[test]
    fn the_fixed_point_kernel_preserves_a_constant_image() {
        // The integer path has two extra ways to drift that the f32 twin does
        // not: the coefficients are rounded to i32, and the accumulator is
        // shifted. A constant image catches both — the quantised weights must
        // still sum to 1<<22 closely enough that the shift lands on the same
        // level.
        for (sw, sh, dw, dh) in [(16, 16, 64, 64), (64, 64, 16, 16), (10, 20, 33, 7)] {
            let src = vec![97u8; sw * sh * 3];
            let out = resize_lanczos_u8(&src, sw, sh, dw, dh, 3);
            assert_eq!(out.len(), dw * dh * 3);
            let worst = out.iter().map(|&v| i32::from(v) - 97).map(i32::abs).max();
            assert_eq!(worst, Some(0), "{sw}x{sh}->{dw}x{dh} drifted off 97");
        }
    }

    #[test]
    fn quantisation_rounds_away_from_zero_in_both_directions() {
        // Lanczos weights go negative — that is what makes it sharpen — so the
        // negative branch is load-bearing, not a formality. `as i32` alone
        // truncates toward zero and would bias every negative lobe upward.
        let unit = f64::from(1 << PRECISION_BITS);
        assert_eq!(quantise(1.0), 1 << PRECISION_BITS);
        assert_eq!(quantise(-1.0), -(1 << PRECISION_BITS));
        assert_eq!(quantise(0.5 / unit), 1);
        assert_eq!(quantise(-0.5 / unit), -1);
    }

    #[test]
    fn clip8_clamps_rather_than_wrapping() {
        // A sharpening kernel overshoots at edges: the accumulator genuinely
        // goes below 0 and above 255. Wrapping there turns a bright edge into
        // a black one, which is very visible and very hard to attribute.
        assert_eq!(clip8(-5 << PRECISION_BITS), 0);
        assert_eq!(clip8(300 << PRECISION_BITS), 255);
        assert_eq!(clip8(128 << PRECISION_BITS), 128);
        // The `1 << (PRECISION_BITS - 1)` seed makes the shift ROUND.
        assert_eq!(clip8((1 << PRECISION_BITS) - 1 + (1 << (PRECISION_BITS - 1))), 1);
    }

    #[test]
    fn an_identity_resize_is_a_copy_in_the_fixed_point_path_too() {
        let src: Vec<u8> = (0..48).map(|i| i as u8).collect();
        assert_eq!(resize_lanczos_u8(&src, 4, 4, 4, 4, 3), src);
    }

    #[test]
    fn the_vision_encoder_size_rounds_the_long_edge_first() {
        // Both edges land on a multiple of the tile, and the LONG edge is the
        // one rounded first — rounding them independently distorts a
        // non-square image differently at each size.
        assert_eq!(vision_encoder_size(2048, 2048, 512), (2048, 2048));
        assert_eq!(vision_encoder_size(2048, 1536, 512), (2048, 1536));
        for (w, h) in [(2048, 1153), (1000, 700), (513, 511), (100, 3000)] {
            let (a, b) = vision_encoder_size(w, h, 512);
            assert_eq!(a % 512, 0, "{w}x{h} -> {a}x{b}: width not a tile multiple");
            assert_eq!(b % 512, 0, "{w}x{h} -> {a}x{b}: height not a tile multiple");
            assert!(a >= 512 && b >= 512, "{w}x{h} -> {a}x{b}: degenerate");
        }
    }

    #[test]
    fn preprocessing_fills_every_tile_it_promises() {
        // The tile count, the buffer length and the grid must agree. A tile
        // left at its zeroed initial value is mid-gray after normalisation —
        // a plausible-looking image the tower would happily caption.
        let (w, h) = (300usize, 700usize);
        let px: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
        let out = preprocess_rgb8(&px, w, h);
        assert_eq!(out.pixel_values.len(), out.tiles * 3 * out.tile * out.tile);
        assert_eq!(out.tiles, out.rows * out.cols + 1);
        let per = 3 * out.tile * out.tile;
        for t in 0..out.tiles {
            let tile = &out.pixel_values[t * per..(t + 1) * per];
            assert!(
                tile.iter().any(|&v| v.abs() > 1e-6),
                "tile {t} is entirely zero — an unfilled buffer, not an image"
            );
        }
    }
}
