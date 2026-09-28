//! Contracts a production caller relies on, from docs/plans/commercial-gaps.md:
//!
//! * **gap 6: determinism.** The same page read twice gives the same text and
//!   the same boxes, so a stored extraction can be re-derived and audited.
//! * **gaps 7 and 10: offline resolution.** `CraftCrnn::offline` needs no
//!   manifest directory, never touches the network, and reads exactly what
//!   the cache-backed constructor reads.
//! * **gap 3: charset constraint.** With every character allowed the
//!   constrained decoder IS the free decoder; with a restricted set, nothing
//!   outside it is emitted.
//!
//! Weights are fetched artifacts, so these SKIP loudly when absent, like
//! `oracles.rs` and `from_bytes.rs`.

use ffai_carmenta::engine::{CraftCrnn, DetStage, RecStage};
use ffai_core::engine::{OcrEngine, OcrOptions};
use ffai_core::types::{ImageBuffer, OcrOutput};
use std::path::PathBuf;

fn weights_root() -> PathBuf {
    ffai_models::cache_dir().join("models")
}

fn have(model: &str, file: &str) -> bool {
    weights_root().join(model).join(file).exists()
}

fn have_mobiledet_svtr() -> bool {
    have("ppocrv5-mobile-det", "det-fused.safetensors")
        && have("ppocrv5-mobile-rec", "rec.safetensors")
        && have("ppocrv5-mobile-rec", "charset.txt")
}

fn have_mobiledet_crnn() -> bool {
    have("ppocrv5-mobile-det", "det-fused.safetensors")
        && have("crnn-english-g2", "crnn.safetensors")
}

fn manifests() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

/// A real document page with enough lines to take the PARALLEL recognition
/// path (three or more), which is where a non-deterministic reduction would
/// show up if there were one.
fn doc_page() -> Option<ImageBuffer> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../corpora/clips/carmenta-omnidoc/omni-0000.png");
    p.exists()
        .then(|| ffai_media::load_image(&p).expect("doc page loads"))
}

/// Every line as (text, bbox) — the parts of the output that must be exact.
fn lines(out: &OcrOutput) -> Vec<(String, Option<[i64; 4]>)> {
    out.blocks
        .iter()
        .flat_map(|b| &b.lines)
        .map(|l| {
            let bb = l
                .bbox
                .map(|b| [b.x as i64, b.y as i64, b.width as i64, b.height as i64]);
            (l.text.clone(), bb)
        })
        .collect()
}

fn confidences(out: &OcrOutput) -> Vec<Option<f32>> {
    out.blocks
        .iter()
        .flat_map(|b| &b.lines)
        .map(|l| l.confidence)
        .collect()
}

#[test]
fn the_same_page_read_twice_is_identical() {
    let Some(img) = doc_page() else {
        eprintln!("SKIP determinism: doc page missing");
        return;
    };
    if !have_mobiledet_svtr() {
        eprintln!("SKIP determinism: mobiledet-svtr weights not in cache");
        return;
    }
    let opts = OcrOptions::default();
    // Two independent instances as well as two calls on one, so neither lazy
    // loading nor any per-instance state can hide a difference.
    let a = CraftCrnn::variant_in(RecStage::Svtr, DetStage::MobileDet, &manifests());
    let b = CraftCrnn::variant_in(RecStage::Svtr, DetStage::MobileDet, &manifests());
    let r1 = a.recognize(&img, &opts).unwrap();
    let r2 = a.recognize(&img, &opts).unwrap();
    let r3 = b.recognize(&img, &opts).unwrap();

    let l1 = lines(&r1);
    assert!(
        l1.len() >= 3,
        "page should exercise the parallel path, got {} lines",
        l1.len()
    );
    assert_eq!(l1, lines(&r2), "text or boxes changed between two calls");
    assert_eq!(
        l1,
        lines(&r3),
        "text or boxes changed between two instances"
    );

    // Confidences: §8.46 measured ~1e-4 drift when OTHER processes compete for
    // the CPU. Measured here rather than assumed, and bounded well below any
    // threshold a caller would gate on.
    let drift = confidences(&r1)
        .iter()
        .zip(confidences(&r2).iter().chain(confidences(&r3).iter()))
        .filter_map(|(x, y)| Some((x.as_ref()? - y.as_ref()?).abs()))
        .fold(0f32, f32::max);
    eprintln!(
        "determinism: {} lines identical x3, max confidence drift {drift:e}",
        l1.len()
    );
    assert!(
        drift <= 1e-3,
        "confidence drift {drift} is large enough to flip a gate"
    );
}

#[test]
fn offline_reads_what_the_cache_backed_constructor_reads() {
    let Some(img) = doc_page() else {
        eprintln!("SKIP offline: doc page missing");
        return;
    };
    if !have_mobiledet_svtr() {
        eprintln!("SKIP offline: mobiledet-svtr weights not in cache");
        return;
    }
    let opts = OcrOptions::default();
    let cached = CraftCrnn::variant_in(RecStage::Svtr, DetStage::MobileDet, &manifests());
    let offline = CraftCrnn::offline(RecStage::Svtr, DetStage::MobileDet, &weights_root());
    offline
        .preload()
        .expect("offline engine loads from the cache layout");
    assert_eq!(
        lines(&cached.recognize(&img, &opts).unwrap()),
        lines(&offline.recognize(&img, &opts).unwrap()),
        "offline resolution read different weights or manifests"
    );
}

#[test]
fn offline_with_no_weights_fails_at_preload_naming_the_path() {
    let empty = std::env::temp_dir().join(format!("ffai-offline-empty-{}", std::process::id()));
    std::fs::create_dir_all(&empty).unwrap();
    let e = CraftCrnn::offline(RecStage::Svtr, DetStage::MobileDet, &empty);
    let msg = match e.preload() {
        Ok(()) => panic!("an empty weights directory must not load"),
        Err(err) => err.to_string(),
    };
    let want = empty
        .join("ppocrv5-mobile-det")
        .join("det-fused.safetensors");
    assert!(
        msg.contains(&want.display().to_string()),
        "error must name the expected file {}: {msg}",
        want.display()
    );
    let _ = std::fs::remove_dir_all(&empty);
}

#[test]
fn a_fully_permissive_charset_is_the_free_decode() {
    let Some(img) = doc_page() else {
        eprintln!("SKIP charset: doc page missing");
        return;
    };
    if !have_mobiledet_crnn() {
        eprintln!("SKIP charset: mobiledet-crnn weights not in cache");
        return;
    }
    let e = CraftCrnn::variant_in(RecStage::Crnn, DetStage::MobileDet, &manifests());
    let free = e.recognize(&img, &OcrOptions::default()).unwrap();
    let all = OcrOptions {
        charset: Some(ffai_carmenta::crnn::CHARSET.to_string()),
        ..OcrOptions::default()
    };
    let masked = e.recognize(&img, &all).unwrap();
    assert_eq!(
        lines(&free),
        lines(&masked),
        "allowing every class changed the text"
    );
    assert_eq!(
        confidences(&free),
        confidences(&masked),
        "allowing every class changed confidence"
    );
}

#[test]
fn a_digit_charset_emits_nothing_outside_it() {
    let Some(img) = doc_page() else {
        eprintln!("SKIP charset: doc page missing");
        return;
    };
    if !have_mobiledet_svtr() {
        eprintln!("SKIP charset: mobiledet-svtr weights not in cache");
        return;
    }
    let allowed = "0123456789$,.- ";
    let e = CraftCrnn::variant_in(RecStage::Svtr, DetStage::MobileDet, &manifests());
    let free = e.recognize(&img, &OcrOptions::default()).unwrap();
    let digits = e
        .recognize(
            &img,
            &OcrOptions {
                charset: Some(allowed.into()),
                ..OcrOptions::default()
            },
        )
        .unwrap();
    let text = digits.text();
    assert!(
        text.chars().all(|c| allowed.contains(c) || c == '\n'),
        "constrained output contains a character outside {allowed:?}: {text:?}"
    );
    // A page of prose forced into digits: the model did not want these
    // characters, and the confidence must say so.
    let mean = |o: &OcrOutput| {
        let c: Vec<f32> = confidences(o).into_iter().flatten().collect();
        c.iter().sum::<f32>() / c.len().max(1) as f32
    };
    eprintln!(
        "charset: free mean conf {:.3}, digits-only {:.3}",
        mean(&free),
        mean(&digits)
    );
    assert!(
        mean(&digits) < mean(&free),
        "forced characters should read as less confident"
    );
}

#[test]
fn a_character_the_model_cannot_emit_is_an_error_naming_it() {
    let Some(img) = doc_page() else {
        eprintln!("SKIP charset: doc page missing");
        return;
    };
    if !have_mobiledet_crnn() {
        eprintln!("SKIP charset: mobiledet-crnn weights not in cache");
        return;
    }
    let e = CraftCrnn::variant_in(RecStage::Crnn, DetStage::MobileDet, &manifests());
    // The English CRNN has no CJK classes.
    let err = e
        .recognize(
            &img,
            &OcrOptions {
                charset: Some("0123中".into()),
                ..OcrOptions::default()
            },
        )
        .expect_err("an unrepresentable character must be refused")
        .to_string();
    assert!(err.contains('中'), "error should name the character: {err}");
}
