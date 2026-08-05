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
