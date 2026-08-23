//! The bytes constructor must be the filesystem constructor with a different
//! door — Step 1 of `docs/plans/carmenta-wasm-plan.md`.
//!
//! `CraftCrnn::from_bytes` exists because a browser has neither a filesystem
//! nor an mmap. The risk in adding a second way to build an engine is that it
//! becomes a LENIENT way: skipping a validation, defaulting a charset,
//! loading a different dtype. So the gate is not "it works" but **the two
//! constructors produce identical text on the same image**, which is the same
//! shape as Diana's `from_bytes_matches_the_filesystem_path`.
//!
//! SKIPs loudly when weights are absent, like `oracles.rs`: both are fetched
//! artifacts and a fresh checkout should not fail on their absence.

use ffai_carmenta::engine::{CraftCrnn, DetStage, RecStage, WeightBytes};
use ffai_core::engine::{OcrEngine, OcrOptions};
use std::path::PathBuf;

fn model_file(model: &str, file: &str) -> Option<PathBuf> {
    let p = ffai_models::cache_dir().join("models").join(model).join(file);
    p.exists().then_some(p)
}

/// `CraftCrnn::new()` resolves `models/` relative to the CWD, and `cargo test`
/// runs from the CRATE directory rather than the workspace root — so the
/// filesystem arm has to be pointed at the manifests explicitly or it fails
/// with an i/o error that looks like missing weights.
fn manifests() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn test_image() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpora/clips/carmenta-capture/cap-000.png")
}

/// CRAFT + CRNN: the default pair, and the one with a compile-time charset.
#[test]
fn craft_crnn_from_bytes_matches_the_filesystem_path() {
    let (Some(craft), Some(crnn)) = (
        model_file("craft-mlt", "craft.safetensors"),
        model_file("crnn-english-g2", "crnn.safetensors"),
    ) else {
        eprintln!("SKIP from_bytes: craft/crnn weights not in cache");
        return;
    };
    let image = test_image();
    if !image.exists() {
        eprintln!("SKIP from_bytes: test image missing");
        return;
    }

    let img = ffai_media::load_image(&image).expect("test image loads");
    let opts = OcrOptions::default();

    let from_path = CraftCrnn::with_manifest_dir(&manifests());
    let expected = from_path.recognize(&img, &opts).expect("filesystem path recognizes");

    let from_bytes = CraftCrnn::from_bytes(
        DetStage::Craft,
        RecStage::Crnn,
        WeightBytes {
            craft: Some(std::fs::read(&craft).expect("craft bytes")),
            crnn: Some(std::fs::read(&crnn).expect("crnn bytes")),
            ..WeightBytes::default()
        },
    )
    .expect("bytes path builds");
    let actual = from_bytes.recognize(&img, &opts).expect("bytes path recognizes");

    assert_eq!(
        expected.text(),
        actual.text(),
        "the two constructors differ in what they read from the same image"
    );
}

/// The pair a browser should actually ship: 4.7 MB of detector against
/// CRAFT's 83 MB, and the only one whose charset also arrives as bytes — the
/// place an off-by-one would shift every decoded character (§8.168).
#[test]
fn mobiledet_svtr_from_bytes_matches_the_filesystem_path() {
    let (Some(det), Some(rec), Some(charset)) = (
        model_file("ppocrv5-mobile-det", "det-fused.safetensors"),
        model_file("ppocrv5-mobile-rec", "rec.safetensors"),
        model_file("ppocrv5-mobile-rec", "charset.txt"),
    ) else {
        eprintln!("SKIP from_bytes: ppocrv5 mobile weights not in cache");
        return;
    };
    let image = test_image();
    if !image.exists() {
        eprintln!("SKIP from_bytes: test image missing");
        return;
    }

    let img = ffai_media::load_image(&image).expect("test image loads");
    let opts = OcrOptions::default();

    let from_path =
        CraftCrnn::variant_in(RecStage::Svtr, DetStage::MobileDet, &manifests());
    let expected = from_path.recognize(&img, &opts).expect("filesystem path recognizes");

    let from_bytes = CraftCrnn::from_bytes(
        DetStage::MobileDet,
        RecStage::Svtr,
        WeightBytes {
            mobiledet: Some(std::fs::read(&det).expect("det bytes")),
            svtr: Some((
                std::fs::read(&rec).expect("rec bytes"),
                std::fs::read_to_string(&charset).expect("charset text"),
            )),
            ..WeightBytes::default()
        },
    )
    .expect("bytes path builds");
    let actual = from_bytes.recognize(&img, &opts).expect("bytes path recognizes");

    assert_eq!(
        expected.text(),
        actual.text(),
        "the two constructors differ in what they read from the same image"
    );
}

/// A missing blob must name what it wanted. The failure mode this guards is a
/// browser build that silently recognizes nothing because a fetch 404'd.
#[test]
fn a_missing_blob_is_an_error_that_says_which() {
    // `CraftCrnn` is not `Debug`, so match rather than `expect_err`.
    let msg = match CraftCrnn::from_bytes(DetStage::Craft, RecStage::Crnn, WeightBytes::default()) {
        Ok(_) => panic!("no weights supplied must not build"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("craft"), "error should name the missing weights, got: {msg}");
}
