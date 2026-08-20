//! Determinism: same input, same bytes out — every run, every thread count.
//!
//! This is the parity claim that CAN be byte-exact, and unlike agreement
//! with PyTorch it is entirely within our control. Piper cannot offer it for
//! TTS (it samples noise in-graph); PyTorch cannot offer it for detection
//! across thread counts, because its kernels reassociate differently per
//! schedule. Diana can, and a caller who needs a reproducible pipeline —
//! caching, auditing, regression-diffing a ledger claim — needs exactly
//! this.
//!
//! What makes it hold, and what would break it:
//!
//! - our kernels partition by rayon over **disjoint output ranges**, so no
//!   two threads accumulate into the same cell and the schedule cannot
//!   change the arithmetic;
//! - the GEMM is called with fixed shapes, so its internal blocking is a
//!   function of the shapes alone;
//! - the two-stage top-k sorts by `total_cmp`, a **total order**, so ties
//!   resolve by position rather than by whichever thread finished first.
//!
//! A future brick that introduced a parallel reduction over a shared
//! accumulator (a common way to speed up a reduction) would break this
//! silently and this test is what would catch it.

use ffai_core::engine::{DetectEngine, DetectOptions};

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/ffai-diana has two ancestors")
        .to_path_buf()
}

/// A stable digest of a detection set: the exact bits of every field.
fn digest(out: &ffai_core::types::DetectOutput) -> Vec<u8> {
    let mut v = Vec::with_capacity(out.detections.len() * 24);
    for d in &out.detections {
        v.extend_from_slice(&d.x0.to_le_bytes());
        v.extend_from_slice(&d.y0.to_le_bytes());
        v.extend_from_slice(&d.x1.to_le_bytes());
        v.extend_from_slice(&d.y1.to_le_bytes());
        v.extend_from_slice(&d.confidence.to_le_bytes());
        v.extend_from_slice(&d.class_id.to_le_bytes());
    }
    v
}

#[test]
fn same_input_gives_byte_identical_detections() {
    let root = repo_root();
    let img = root.join("corpora/clips/diana-coco/coco-032.png");
    let weights = root.join("corpora/cache/yolo26n-diana.safetensors");
    if !img.exists() || !weights.exists() {
        eprintln!(
            "SKIP determinism: corpus or weights absent.\n  \
             python tools/diana_coco_corpus.py\n  \
             .venv-diana/Scripts/python.exe tools/diana_convert.py --model yolo26n"
        );
        return;
    }

    let image = ffai_media::load_image(&img).expect("load image");
    // `Yolo26::new()` resolves `models/` relative to the CWD, which under
    // `cargo test` is the crate directory, not the repo root.
    let engine = ffai_diana::engine::Yolo26::with_manifest_dir(root.join("models"));
    // The bench decode settings — the deepest tail, so the top-k tie
    // resolution is exercised, not just the confident head.
    let opts = DetectOptions { confidence: 0.001, max_detections: 300, ..Default::default() };

    let first = digest(&engine.detect(&image, &opts).expect("detect"));
    assert!(!first.is_empty(), "no detections to compare");

    for run in 1..6 {
        let d = digest(&engine.detect(&image, &opts).expect("detect"));
        assert_eq!(
            d.len(),
            first.len(),
            "run {run}: detection COUNT changed between identical calls"
        );
        assert!(
            d == first,
            "run {run}: output is not byte-identical to run 0 — something in the \
             forward pass depends on thread scheduling (a shared accumulator, or a \
             sort that is not a total order)"
        );
    }
    // Printed so the SAME digest can be compared ACROSS PROCESSES at
    // different thread counts — rayon's pool is fixed at startup, so the
    // thread-count claim cannot be tested inside one process:
    //   RAYON_NUM_THREADS=1  cargo test -p ffai-diana --test determinism -- --nocapture
    //   RAYON_NUM_THREADS=24 cargo test -p ffai-diana --test determinism -- --nocapture
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in &first {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    eprintln!(
        "determinism: {} detections, byte-identical across 6 runs · digest {h:016x} \
         (threads={})",
        first.len() / 24,
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into())
    );
}

/// A second, independent image — so the property is not an accident of one
/// input's particular detection count.
#[test]
fn determinism_holds_on_a_second_image() {
    let root = repo_root();
    let img = root.join("corpora/clips/diana-coco/coco-000.png");
    let weights = root.join("corpora/cache/yolo26n-diana.safetensors");
    if !img.exists() || !weights.exists() {
        eprintln!("SKIP: corpus or weights absent");
        return;
    }
    let image = ffai_media::load_image(&img).expect("load image");
    // `Yolo26::new()` resolves `models/` relative to the CWD, which under
    // `cargo test` is the crate directory, not the repo root.
    let engine = ffai_diana::engine::Yolo26::with_manifest_dir(root.join("models"));
    let opts = DetectOptions { confidence: 0.001, max_detections: 300, ..Default::default() };
    let a = digest(&engine.detect(&image, &opts).expect("detect"));
    let b = digest(&engine.detect(&image, &opts).expect("detect"));
    assert_eq!(a, b, "coco-000: output not byte-identical between two calls");
}
