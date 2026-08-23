//! The content-path gate: image bytes -> `pixel_values`, against the reference.
//!
//! Steps 3, 4 and 5 all fed the tower the reference's own `pixel_values` so
//! that a failure could be attributed. This is the brick they isolated out.
//!
//! The comparison is deliberately staged from cheapest to hardest, so a
//! failure says WHICH part is wrong rather than merely that something is:
//!
//! 1. **Tile count and geometry** — arithmetic, no filter involved.
//! 2. **The thumbnail** — for a 512x512 input the thumbnail is a 512-longest-edge
//!    resize, i.e. a NO-OP, so it isolates rescale+normalize from the resampler
//!    entirely. If this fails, the filter is innocent.
//! 3. **The upscaled tiles** — the only stage where Lanczos actually runs.

use std::path::{Path, PathBuf};

use ffai_argus::preprocess::{fit_longest_edge, preprocess_rgb8, tile_grid};

const IMG: usize = 512;

/// The same deterministic pattern the dumpers use. Mirrored here rather than
/// shared, which is the point: if the two drift, the gate fails loudly instead
/// of comparing an image against itself.
fn reference_image() -> Vec<u8> {
    let mut px = vec![0u8; IMG * IMG * 3];
    let mut i = 0;
    for y in 0..IMG {
        let fy = y as f64 / IMG as f64;
        for x in 0..IMG {
            let fx = x as f64 / IMG as f64;
            let r = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * fx).sin();
            let g = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * fy + 1.0).sin();
            let b = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * (fx + fy) + 2.0).sin();
            px[i] = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            px[i + 1] = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            px[i + 2] = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            i += 3;
        }
    }
    px
}

fn oracle() -> Option<(PathBuf, serde_json::Value)> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.oracle/smolvlm-prompt");
    let v = serde_json::from_str(&std::fs::read_to_string(d.join("prompt.json")).ok()?).ok()?;
    Some((d, v))
}

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect()
}

/// Interleaved HWC f32 -> planar CHW, the layout the tower consumes.
fn to_chw(hwc: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h * 3];
    for c in 0..3 {
        for i in 0..w * h {
            out[c * w * h + i] = hwc[i * 3 + c];
        }
    }
    out
}

/// One quantisation level of 255 in `[-1, 1]` units: `1/127.5`.
const ONE_LEVEL: f32 = 7.843_2e-3;

/// Magnitude AND count, because only the pair is a real gate.
///
/// Returns `(max_abs, fraction of values that differ at all)`.
///
/// # Why this is not `assert_eq!(max_abs, 0.0)`
///
/// It was, briefly, and it could not pass. Our resampler is **bit-identical to
/// PIL** — `resize_oracle.rs` gates exactly that, zero differing pixels in both
/// directions. But the reference tensor here was not produced by PIL:
/// `AutoProcessor` defaults to the torchvision-backed *fast* processor, which
/// reimplements the same fixed-point algorithm and rounds a handful of
/// coefficients the other way. Measured directly, **PIL's own output differs
/// from this reference on ~20 values per border tile out of 786,432** — 0.0025%,
/// all of them ULP-boundary ties.
///
/// So exact equality is unattainable against this fixture, and demanding it
/// would be demanding that we disagree with PIL.
///
/// Bounding only the MAGNITUDE would be too weak in the other direction: the
/// float resampler this replaced was also within one level, everywhere, and it
/// produced 8/32 correct tokens. What separates the two is how MANY values
/// differ — twenty, versus all of them. Hence both bounds.
fn stats(name: &str, got: &[f32], want: &[f32]) -> (f32, f64) {
    let max_abs = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let differing = got
        .iter()
        .zip(want)
        .filter(|(a, b)| (*a - *b).abs() > 1e-4)
        .count();
    let (mut se, mut sr) = (0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(want) {
        se += f64::from(a - b) * f64::from(a - b);
        sr += f64::from(b) * f64::from(b);
    }
    let snr = 10.0 * (sr / se.max(f64::MIN_POSITIVE)).log10();
    let frac = differing as f64 / got.len() as f64;
    eprintln!(
        "  {name:22} n={:<8} max_abs={max_abs:.3e}  differ={differing} ({:.4}%)  SNR={snr:.1} dB",
        got.len(),
        frac * 100.0
    );
    (max_abs, frac)
}

/// At most one quantisation level, on at most 0.1% of values.
///
/// 0.1% is ~40x the measured PIL-vs-torchvision tie rate and ~1000x below a
/// wrong filter, which differs essentially everywhere.
fn assert_within_tie_noise(name: &str, max_abs: f32, frac: f64) {
    assert!(
        max_abs <= ONE_LEVEL * 1.01,
        "{name}: max_abs {max_abs:.3e} exceeds one quantisation level \
         ({ONE_LEVEL:.3e}) — that is the RESAMPLER, not a rounding tie"
    );
    assert!(
        frac < 1e-3,
        "{name}: {:.4}% of values differ. Each is within one level, but a \
         tie-break disagreement between two fixed-point implementations is ~0.003%. \
         This many means the whole image moved — which is what the FLOAT resampler \
         looked like, and it produced 8/32 correct tokens.",
        frac * 100.0
    );
}

/// Geometry first: no filter involved, so a failure here is pure arithmetic.
#[test]
fn the_geometry_matches_the_reference_tiling() {
    let Some((_, doc)) = oracle() else {
        eprintln!("SKIP: no prompt oracle");
        return;
    };
    let pv: Vec<usize> = doc["embeds"]["pixel_values"]["shape"]
        .as_array()
        .expect("shape")
        .iter()
        .map(|x| x.as_u64().expect("dim") as usize)
        .collect();
    let (tiles, c, h, w) = (pv[1], pv[2], pv[3], pv[4]);

    let (rw, rh) = fit_longest_edge(IMG, IMG, 2048);
    let (rows, cols) = tile_grid(rw, rh, 512);
    assert_eq!((rw, rh), (2048, 2048), "longest edge 2048 upscales 512");
    assert_eq!((rows, cols), (4, 4));
    assert_eq!(rows * cols + 1, tiles, "16 tiles plus the global thumbnail");
    assert_eq!((c, h, w), (3, 512, 512));
}

/// The thumbnail — and a hypothesis that was wrong in an informative way.
///
/// **The first version assumed the thumbnail was the untouched original**, on
/// the reasoning that a 512-longest-edge resize of a 512x512 image is a no-op.
/// It missed by `max_abs = 7.843e-3`, which in `[-1,1]` units is about ONE
/// LEVEL of 255 — far too small for a wrong image and far too large for f32
/// noise.
///
/// The magnitude was the diagnosis. Idefics3 resizes to `longest_edge` 2048
/// first, and the global thumbnail is that upscaled image brought back DOWN to
/// 512 — not the original. Lanczos up-then-down is near-identity, not
/// identity. Being wrong bought a better test: this stage is the only one that
/// exercises the resampler in the DOWNSCALE direction, where the kernel widens
/// (`filter_scale = max(1, scale)`) and the tiles' upscale path never goes.
#[test]
fn the_thumbnail_matches_the_downscaled_round_trip() {
    let Some((dir, doc)) = oracle() else {
        eprintln!("SKIP: no prompt oracle");
        return;
    };
    if !dir.join("pixel_values.f32").exists() {
        eprintln!("SKIP: dump lacks pixel_values");
        return;
    }
    let pv = shape(&doc);
    let (tiles, per_tile) = (pv[1], pv[2] * pv[3] * pv[4]);
    let all = read_f32(&dir.join("pixel_values.f32"));

    let ours = preprocess_rgb8(&reference_image(), IMG, IMG);
    assert_eq!(ours.tiles, tiles, "tile count");

    // The thumbnail is the LAST tile — the same ordering the prompt assembly
    // encodes, where `<global-img>` follows every `<row_r_col_c>`.
    let want = &all[(tiles - 1) * per_tile..tiles * per_tile];
    let got = &ours.pixel_values[(tiles - 1) * per_tile..tiles * per_tile];

    eprintln!("thumbnail (Lanczos 512 -> 2048 -> 512):");
    let (max_abs, frac) = stats("thumbnail", got, want);
    assert_within_tie_noise("thumbnail", max_abs, frac);
}

/// The tiles: the only stage where the resampler runs in the UPSCALE direction.
#[test]
fn the_upscaled_tiles_match_the_reference() {
    let Some((dir, doc)) = oracle() else {
        eprintln!("SKIP: no prompt oracle");
        return;
    };
    if !dir.join("pixel_values.f32").exists() {
        eprintln!("SKIP: dump lacks pixel_values");
        return;
    }
    let pv = shape(&doc);
    let per_tile = pv[2] * pv[3] * pv[4];
    let all = read_f32(&dir.join("pixel_values.f32"));
    let ours = preprocess_rgb8(&reference_image(), IMG, IMG);
    assert_eq!((ours.rows, ours.cols), (4, 4));

    eprintln!("upscaled tiles (Lanczos 512 -> 2048):");
    let mut worst = 0.0f32;
    let mut worst_frac = 0.0f64;
    let mut interior_worst = 0.0f32;
    let mut interior_differing = 0.0f64;
    for r in 0..ours.rows {
        for c in 0..ours.cols {
            let idx = r * ours.cols + c;
            let got = &ours.pixel_values[idx * per_tile..(idx + 1) * per_tile];
            let want = &all[idx * per_tile..(idx + 1) * per_tile];
            let interior = r > 0 && c > 0 && r + 1 < ours.rows && c + 1 < ours.cols;
            let (m, f) = stats(&format!("tile r{}c{}", r + 1, c + 1), got, want);
            worst = worst.max(m);
            worst_frac = worst_frac.max(f);
            if interior {
                interior_worst = interior_worst.max(m);
                interior_differing += f;
            }
        }
    }
    eprintln!("worst tile: max_abs {worst:.3e}, {:.4}% differing", worst_frac * 100.0);
    assert_within_tie_noise("tiles", worst, worst_frac);
    // The four INTERIOR tiles touch no image border, so no boundary window is
    // truncated and no tie can arise there: they must be exact. This is what
    // localised the original bug — the interior was already bit-exact while
    // every border tile was not, which said "boundary handling", not "kernel".
    // Judged on the COUNT, not the magnitude: `interior_worst` is ~5.9e-8,
    // which is f32 epsilon from the CHW transpose and the normalise, not a
    // resampling difference. What matters is that zero values differ by
    // anything real.
    assert_eq!(
        interior_differing, 0.0,
        "the four interior tiles must be EXACT. They touch no image border, so \
         no resampling window is truncated and no tie can arise. A difference \
         HERE is the kernel itself rather than a tie-break — which is exactly \
         how the original bug was localised: the interior was already bit-exact \
         while every border tile was not."
    );
    assert!(
        interior_worst < 1e-6,
        "interior tiles drifted by {interior_worst:.3e} — beyond f32 epsilon"
    );
}

/// A non-square image, where the two resize steps stop being the same step.
///
/// The square case hides `vision_encoder_size` entirely — 512x512 goes to
/// 2048x2048, which is already a multiple of 512, so step 2 is a no-op and a
/// missing step 2 would still pass every test above.
#[test]
fn a_non_square_image_gets_a_rectangular_grid() {
    // 800x600 -> longest edge 2048 -> 2048x1536 -> 4 cols x 3 rows = 12 + 1.
    let (w, h) = (800usize, 600usize);
    let mut px = vec![0u8; w * h * 3];
    for (i, v) in px.iter_mut().enumerate() {
        *v = (i % 251) as u8;
    }
    let out = preprocess_rgb8(&px, w, h);
    assert_eq!(fit_longest_edge(w, h, 2048), (2048, 1536));
    assert_eq!((out.rows, out.cols), (3, 4));
    assert_eq!(out.tiles, 3 * 4 + 1);
    assert_eq!(out.pixel_values.len(), out.tiles * 3 * 512 * 512);
    // Every value must be a real normalised sample, not a zeroed hole.
    assert!(out.pixel_values.iter().all(|v| (-1.0..=1.0).contains(v)));
    assert!(
        tile_grid(2048, 1536, 512) == (3, 4),
        "the grid the prompt markers use must be the grid the pixels use"
    );
}

fn shape(doc: &serde_json::Value) -> Vec<usize> {
    doc["embeds"]["pixel_values"]["shape"]
        .as_array()
        .expect("shape")
        .iter()
        .map(|x| x.as_u64().expect("dim") as usize)
        .collect()
}
