//! Full-graph oracle for the depth task.
//!
//! Backbone and neck are shared with detect and already gated across five
//! tiers, so what this actually tests is [`DepthHead`] — but it tests it
//! **through the whole graph**, because that is where the interesting failures
//! live. A bilinear convention that is wrong by half a pixel, or a
//! `ConvTranspose2d` weight read as `[out, in]` instead of `[in, out]`,
//! produces a depth map that looks entirely plausible and is simply wrong.
//! Neither shows up in a unit test of the op alone; both show up here.
//!
//! The fixture is the letterboxed TENSOR the reference consumed, not the
//! source image, so this compares graphs rather than two independent
//! preprocessing implementations. Diana's letterbox is gated separately.
//!
//! Skips loudly when the converted weights are absent — they are AGPL and
//! never vendored, so a clean checkout cannot have them.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use ffai_diana::backbone::Backbone;
use ffai_diana::config::Dims;
use ffai_diana::depth_head::DepthHead;
use ffai_diana::neck::Neck;
use std::path::PathBuf;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("corpora/refs/fixtures")
}

fn weights() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent()?.parent()?.to_path_buf();
    let candidates = [
        root.join("corpora/cache/yolo26n-depth-diana.safetensors"),
        dirs_cache().join("yolo26n-depth-diana/yolo26n-depth-diana.safetensors"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

fn dirs_cache() -> PathBuf {
    std::env::var("LOCALAPPDATA")
        .map(|d| PathBuf::from(d).join("ffai/models"))
        .unwrap_or_else(|_| PathBuf::from("/tmp/ffai/models"))
}

/// Read the oracle written by the dump script: a raw f32 `.npy`.
fn read_npy_f32(path: &PathBuf) -> (Vec<usize>, Vec<f32>) {
    let raw = std::fs::read(path).expect("oracle npy");
    assert_eq!(&raw[..6], b"\x93NUMPY", "not a npy file");
    let hlen = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    let header = std::str::from_utf8(&raw[10..10 + hlen]).unwrap();
    let shape: Vec<usize> = header
        .split("'shape': (")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let body = &raw[10 + hlen..];
    let vals: Vec<f32> = body
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (shape, vals)
}

#[test]
fn depth_matches_ultralytics() {
    let Some(w) = weights() else {
        eprintln!(
            "SKIP depth_matches_ultralytics: converted weights absent. \
             Run `python tools/diana_convert.py --model yolo26n-depth` \
             (checkpoints are AGPL and never vendored)."
        );
        return;
    };
    let fx = fixtures();
    let (in_path, or_path) = (fx.join("diana_depth_input.bin"), fx.join("diana_depth_oracle.npy"));
    if !in_path.exists() || !or_path.exists() {
        eprintln!("SKIP depth_matches_ultralytics: oracle fixtures absent");
        return;
    }

    let dev = Device::Cpu;
    let raw = std::fs::read(&in_path).unwrap();
    let xs: Vec<f32> =
        raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let hw = ((xs.len() / 3) as f64).sqrt() as usize;
    let x = Tensor::from_vec(xs, (1, 3, hw, hw), &dev).unwrap();

    // SAFETY: mmap of a file this test just located; candle requires it.
    #[allow(unsafe_code)]
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&w), DType::F32, &dev).unwrap()
    };
    let dims = Dims::for_scale("n").unwrap();
    let backbone = Backbone::new(vb.clone(), dims).unwrap();
    let neck = Neck::new(vb.clone(), dims).unwrap();
    // model.23 is the head's index in the graph, the same slot Detect occupies.
    let head = DepthHead::new(vb.pp("model.23"), [64, 128, 256], 256).unwrap();

    let b = backbone.forward(&x).unwrap();
    let n = neck.forward(&b).unwrap();
    let got = head.forward(&[n.p3.clone(), n.p4.clone(), n.p5.clone()]).unwrap();

    let (oshape, want) = read_npy_f32(&or_path);
    let gv = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(
        gv.len(),
        want.len(),
        "depth map size differs: got {:?}, oracle {:?}",
        got.dims(),
        oshape
    );

    // Relative error against a metric-depth map, whose values span roughly
    // 1.5-14 m here. An absolute bound would be meaningless across that range.
    let mut worst = 0f32;
    let mut worst_i = 0usize;
    for (i, (a, e)) in gv.iter().zip(&want).enumerate() {
        let rel = (a - e).abs() / e.abs().max(1e-3);
        if rel > worst {
            worst = rel;
            worst_i = i;
        }
    }
    let mean: f32 = gv.iter().zip(&want).map(|(a, e)| (a - e).abs()).sum::<f32>() / gv.len() as f32;
    eprintln!(
        "depth oracle: worst rel {worst:.3e} at {worst_i}, mean abs {mean:.4} m, \
         range got [{:.2}, {:.2}] oracle [{:.2}, {:.2}]",
        gv.iter().cloned().fold(f32::MAX, f32::min),
        gv.iter().cloned().fold(f32::MIN, f32::max),
        want.iter().cloned().fold(f32::MAX, f32::min),
        want.iter().cloned().fold(f32::MIN, f32::max),
    );
    assert!(
        worst < 1e-3,
        "depth diverges from Ultralytics by {worst:.3e} relative at index {worst_i} \
         (got {}, want {}). Check: bilinear align_corners scale is (in-1)/(out-1); \
         ConvTranspose2d weight layout is [in, out, kh, kw]; the clamp is applied \
         BEFORE exp.",
        gv[worst_i],
        want[worst_i]
    );
}


/// The ENGINE path: manifest resolution, letterbox, graph, and the depth map
/// a caller actually receives.
///
/// The oracle above grades the graph from a pre-letterboxed tensor. This
/// grades everything around it — and would catch a manifest that resolves to
/// the wrong weights, a letterbox that disagrees with the reference, or a
/// stride-4 map reported with the wrong dimensions.
#[test]
fn depth_engine_runs_end_to_end() {
    use ffai_core::engine::{DepthEngine, DepthOptions};
    use ffai_diana::depth_engine::Yolo26Depth;
    use ffai_diana::image::Geometry;

    if weights().is_none() {
        eprintln!("SKIP depth_engine_runs_end_to_end: converted weights absent");
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf();
    let img = root.join("corpora/refs/fixtures/diana_photo_input.png");
    if !img.exists() {
        eprintln!("SKIP depth_engine_runs_end_to_end: fixture image absent");
        return;
    }
    let image = ffai_media::load_image(&img).expect("load fixture");
    let eng = Yolo26Depth::build("n", Geometry::Rect, root.join("models"));

    let out = eng.depth(&image, &DepthOptions::default()).expect("depth");
    assert_eq!(out.width * out.height, out.depth.len(), "map size disagrees with its dims");
    let (lo, hi) = out.range().expect("finite range");
    // Metric depth from the released weights: metres, positive, and bounded
    // by the head's own clamp — exp(-4) to exp(5), 0.018 to 148 m.
    assert!(lo > 0.0 && hi < 149.0, "depth out of the head's clamp range: {lo}..{hi}");
    assert!(hi > lo, "degenerate depth range");
    eprintln!("engine: {}x{} depth {lo:.2}-{hi:.2} m", out.width, out.height);

    // Determinism — the same discipline the detect path is held to.
    let again = eng.depth(&image, &DepthOptions::default()).expect("depth twice");
    assert_eq!(out.depth, again.depth, "depth is not deterministic across runs");

    // Full resolution maps back onto the source image.
    let full = eng
        .depth(&image, &DepthOptions { full_resolution: true })
        .expect("full-res depth");
    assert_eq!(full.width, image.width as usize);
    assert_eq!(full.height, image.height as usize);
    let covered = full.depth.iter().filter(|v| v.is_finite()).count();
    assert!(
        covered * 100 / full.depth.len() >= 99,
        "full-res map covers only {}% of the source",
        covered * 100 / full.depth.len()
    );
}


/// ALL FIVE tiers, through the ENGINE.
///
/// The depth head is width-256 at n, s, m, l and x alike — only the backbone
/// and neck widths scale, feeding `proj` different input channels. So one
/// code path should serve every tier, and this is what turns "should" into a
/// number. It is not a formality: the c3k promotion is a tier-DEPENDENT
/// behaviour in this same graph that looked like a no-op until it was
/// measured, and it was caught by exactly this shape of test.
#[test]
fn depth_matches_ultralytics_across_tiers() {
    use ffai_core::engine::{DepthEngine, DepthOptions};
    use ffai_diana::depth_engine::Yolo26Depth;
    use ffai_diana::image::Geometry;

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf();
    let img = root.join("corpora/refs/fixtures/diana_photo_input.png");
    if !img.exists() {
        eprintln!("SKIP tiers: fixture image absent");
        return;
    }
    let image = ffai_media::load_image(&img).expect("load fixture");

    let mut ran = 0;
    for tier in ["n", "s", "m", "l", "x"] {
        let oracle = match tier {
            "n" => fixtures().join("diana_depth_oracle.npy"),
            t => fixtures().join(format!("diana_depth_oracle_{t}.npy")),
        };
        let st = root.join(format!("corpora/cache/yolo26{tier}-depth-diana.safetensors"));
        if !oracle.exists() || !st.exists() {
            eprintln!("SKIP tier {tier}: weights or oracle absent");
            continue;
        }
        let eng = Yolo26Depth::build(tier, Geometry::Rect, root.join("models"));
        let got = eng.depth(&image, &DepthOptions::default()).expect("depth");
        let (_, want) = read_npy_f32(&oracle);
        assert_eq!(got.depth.len(), want.len(), "tier {tier}: map size differs");

        let mut worst = 0f32;
        for (a, e) in got.depth.iter().zip(&want) {
            worst = worst.max((a - e).abs() / e.abs().max(1e-3));
        }
        let (lo, hi) = got.range().unwrap();
        eprintln!("tier {tier}: worst rel {worst:.3e}, range {lo:.2}-{hi:.2} m");
        assert!(
            worst < 1e-3,
            "tier {tier} diverges from Ultralytics by {worst:.3e} relative —              the head is width-256 at every tier, so a tier-dependent failure              is in the backbone/neck widths or in proj's input channels"
        );
        ran += 1;
    }
    assert!(ran > 0, "no tier had both weights and an oracle; nothing was verified");
}

/// The from-bytes path must build the SAME engine as the filesystem path.
///
/// This is the API that stands between Diana compiling for wasm — it does —
/// and running there, which it cannot while weights arrive through
/// `std::fs`. It also serves embedded targets whose weights live in flash.
///
/// Byte-identical output is the gate, not "close": the two constructors
/// differ only in where the bytes came from, so any difference is a bug in
/// one of them.
#[test]
fn from_bytes_matches_the_filesystem_path() {
    use ffai_core::engine::{DepthEngine, DepthOptions};
    use ffai_diana::depth_engine::Yolo26Depth;
    use ffai_diana::image::Geometry;

    let Some(w) = weights() else {
        eprintln!("SKIP from_bytes: converted weights absent");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf();
    let img_path = root.join("corpora/refs/fixtures/diana_photo_input.png");
    let manifest = root.join("models/yolo26n-depth-diana.json");
    if !img_path.exists() || !manifest.exists() {
        eprintln!("SKIP from_bytes: fixture or manifest absent");
        return;
    }
    let image = ffai_media::load_image(&img_path).expect("fixture");

    let disk = Yolo26Depth::build("n", Geometry::Rect, root.join("models"));
    let want = disk.depth(&image, &DepthOptions::default()).expect("disk path");

    let bytes = std::fs::read(&w).expect("safetensors");
    let json = std::fs::read_to_string(&manifest).expect("manifest");
    let mem = Yolo26Depth::from_bytes("n", Geometry::Rect, bytes, &json).expect("from_bytes");
    let got = mem.depth(&image, &DepthOptions::default()).expect("bytes path");

    assert_eq!(got.width, want.width);
    assert_eq!(got.height, want.height);
    for (i, (a, b)) in got.depth.iter().zip(&want.depth).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "from_bytes diverged at {i}: {a} vs {b}");
    }

    // A manifest for the wrong tier must be refused, not silently loaded to
    // whatever shapes happen to fit.
    let bytes = std::fs::read(&w).expect("safetensors");
    let json = std::fs::read_to_string(&manifest).expect("manifest");
    assert!(
        Yolo26Depth::from_bytes("x", Geometry::Rect, bytes, &json).is_err(),
        "from_bytes accepted a tier that disagrees with the manifest"
    );
}
