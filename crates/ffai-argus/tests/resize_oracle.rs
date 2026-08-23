//! The resampler alone, against PIL's raw `u8` output.
//!
//! The tiled comparison in `preprocess_oracle` measures the resize THROUGH the
//! tile cut, the rescale and the CHW transpose. That is the right gate for the
//! content path and the wrong instrument for a resampler bug: it reports one
//! number per tile and cannot say WHICH pixels moved.
//!
//! This is the single-variable version — `u8` in, `u8` out, no other stage —
//! and it reports the differences by DISTANCE FROM THE EDGE, because that is
//! the axis the tiled gate implicated: the four interior tiles came out
//! bit-exact while all twelve border tiles did not.
//!
//! Regenerate the fixtures with `python corpora/refs/dump_pil_resize.py`.

use std::path::{Path, PathBuf};

use ffai_argus::preprocess::resize_lanczos_u8;

fn dir() -> Option<PathBuf> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.oracle/pil-resize");
    d.join("up_2048.rgb8").exists().then_some(d)
}

/// Differences bucketed by how far the pixel is from the nearest image edge.
///
/// A resampler whose kernel is right but whose boundary window is wrong puts
/// every difference within `support` pixels of an edge. A resampler that is
/// simply wrong spreads them everywhere. The histogram tells the two apart in
/// one run.
fn report(name: &str, got: &[u8], want: &[u8], w: usize, h: usize) -> usize {
    assert_eq!(got.len(), want.len(), "{name}: size");
    let mut diffs = 0usize;
    let mut worst = 0i32;
    let mut by_edge = [0usize; 8];
    let mut interior = 0usize;
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let i = (y * w + x) * 3 + c;
                let d = i32::from(got[i]) - i32::from(want[i]);
                if d != 0 {
                    diffs += 1;
                    worst = worst.max(d.abs());
                    let e = x.min(y).min(w - 1 - x).min(h - 1 - y);
                    if e < 8 {
                        by_edge[e] += 1;
                    } else {
                        interior += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "  {name:14} {diffs:>9} / {:<9} differ  worst={worst}",
        got.len()
    );
    eprintln!("    by distance from edge: {by_edge:?}  interior(>=8): {interior}");
    diffs
}

#[test]
fn our_lanczos_is_bit_identical_to_pil() {
    let Some(d) = dir() else {
        eprintln!("SKIP: no PIL resize fixture — run corpora/refs/dump_pil_resize.py");
        return;
    };
    let src = std::fs::read(d.join("src_512.rgb8")).expect("src");
    let up = std::fs::read(d.join("up_2048.rgb8")).expect("up");
    let down = std::fs::read(d.join("down_512.rgb8")).expect("down");

    eprintln!("resampler vs PIL, u8 in / u8 out:");
    let ours_up = resize_lanczos_u8(&src, 512, 512, 2048, 2048, 3);
    let n_up = report("512 -> 2048", &ours_up, &up, 2048, 2048);
    // Feed PIL's own upscale in, so the downscale is measured on its own rather
    // than inheriting whatever the upscale got wrong.
    let ours_down = resize_lanczos_u8(&up, 2048, 2048, 512, 512, 3);
    let n_down = report("2048 -> 512", &ours_down, &down, 512, 512);

    assert_eq!(
        (n_up, n_down),
        (0, 0),
        "the resampler must be BIT-IDENTICAL to PIL, not merely close: one \
         quantisation level of difference is enough to change a generated token."
    );
}

/// The shapes the content path never exercises, and why they matter.
///
/// The two resizes Argus actually performs are 512 -> 2048 and 2048 -> 512:
/// both exact 4x ratios, both square. That is a weak gate for a resampler.
/// An exact integer ratio puts every output centre at a tidy offset, so a
/// half-pixel convention error can partly cancel; the tap count never varies,
/// so an off-by-one in the window width never shows; and only one of the two
/// downscales, so the kernel widening (`filter_scale = max(1, scale)`) has a
/// single witness.
///
/// These cost nothing to dump and cover all three.
#[test]
fn our_lanczos_matches_pil_at_awkward_shapes_too() {
    let Some(d) = dir() else {
        eprintln!("SKIP: no PIL resize fixture — run corpora/refs/dump_pil_resize.py");
        return;
    };
    let Ok(cases) = std::fs::read_to_string(d.join("cases.txt")) else {
        eprintln!("SKIP: fixture predates the shape sweep — re-run the dumper");
        return;
    };
    let src512 = std::fs::read(d.join("src_512.rgb8")).expect("src");
    let up2048 = std::fs::read(d.join("up_2048.rgb8")).expect("up");

    eprintln!("resampler vs PIL at awkward shapes:");
    let mut total = 0usize;
    for line in cases.lines().filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 5, "cases.txt line: {line}");
        let name = f[0];
        let (sw, sh, dw, dh) = (
            f[1].parse::<usize>().expect("sw"),
            f[2].parse::<usize>().expect("sh"),
            f[3].parse::<usize>().expect("dw"),
            f[4].parse::<usize>().expect("dh"),
        );
        // The dumper resizes from either the 512 original or its 2048 upscale;
        // which one is recoverable from the recorded source dimensions.
        let src: &[u8] = if (sw, sh) == (512, 512) {
            &src512
        } else if (sw, sh) == (2048, 2048) {
            &up2048
        } else {
            panic!("{name}: unknown source {sw}x{sh}");
        };
        let want = std::fs::read(d.join(format!("{name}.rgb8"))).expect("case");
        let got = resize_lanczos_u8(src, sw, sh, dw, dh, 3);
        total += report(name, &got, &want, dw, dh);
    }
    assert_eq!(
        total, 0,
        "a non-integer ratio or a non-square target diverges while the 4x square \
         cases pass. That points at the tap WINDOW (xmin/xmax rounding, or the \
         width cap) rather than at the kernel, which the square cases already \
         prove."
    );
}
