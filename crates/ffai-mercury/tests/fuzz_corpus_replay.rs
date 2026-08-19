//! Replay the fuzz regression corpus as ordinary tests (gate H-27).
//!
//! The corpus under `fuzz/corpus/` holds every input that has ever crashed this
//! crate, plus the seeds. Until now it was only ever replayed by the NIGHTLY,
//! LINUX-ONLY `cargo fuzz` job — so a change that reintroduced a fixed crash
//! could sit green in PR CI for up to a day, and could not be reproduced at all
//! on a developer machine without a nightly toolchain and a working libFuzzer.
//! (It does not build on Windows at all: `onig_sys` fails to compile under the
//! sanitizer flags.)
//!
//! These tests call the SAME entry points the fuzz targets call, on the SAME
//! bytes, with none of that machinery. They are not a substitute for fuzzing —
//! they explore nothing new — but they turn the corpus into a regression suite
//! that runs everywhere, on every PR, in about a millisecond.
//!
//! A corpus file that no longer round-trips is a FINDING, not a stale fixture.

use std::path::{Path, PathBuf};

fn corpus_dir(target: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz").join("corpus").join(target)
}

/// Read every input for a target. Returns an empty vec if the directory is
/// absent, so a shallow checkout does not fail the suite.
fn inputs(target: &str) -> Vec<(String, Vec<u8>)> {
    let dir = corpus_dir(target);
    let Ok(entries) = std::fs::read_dir(&dir) else { return Vec::new() };
    let mut out: Vec<(String, Vec<u8>)> = entries
        .filter_map(Result::ok)
        .filter(|e| e.path().is_file())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            std::fs::read(e.path()).ok().map(|b| (name, b))
        })
        .collect();
    // Deterministic order, so a failure names the same file run to run.
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Mirrors `fuzz_targets/onnx_parse.rs`: parsing is TOTAL — every byte string is
/// either an error or a Graph, never a panic.
#[test]
fn onnx_parse_corpus_is_total() {
    let cases = inputs("onnx_parse");
    assert!(!cases.is_empty(), "onnx_parse corpus is empty - regression seeds lost?");
    for (name, bytes) in cases {
        let _ = ffai_mercury::tts::onnx::parse(&bytes);
        // Reaching here without unwinding IS the assertion.
        eprintln!("onnx_parse: {name} ({} bytes) ok", bytes.len());
    }
}

/// Mirrors `fuzz_targets/mel_compute.rs`, including its shape contract. This is
/// the target that already caught one real defect (`compute(&[])` indexing an
/// empty slice inside `reflect_pad`), so its seeds are load-bearing.
#[test]
fn mel_compute_corpus_holds_shape_contract() {
    use ffai_mercury::asr::mel::MelSpectrogram;
    let cases = inputs("mel_compute");
    assert!(!cases.is_empty(), "mel_compute corpus is empty - regression seeds lost?");
    let m = MelSpectrogram::new(80);
    for (name, bytes) in cases {
        let samples: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let chunk = m.compute(&samples);
        assert_eq!(
            chunk.data.len(),
            chunk.n_mels * chunk.n_frames,
            "{name}: mel buffer is not n_mels * n_frames"
        );
        assert_eq!(
            chunk.n_frames,
            MelSpectrogram::n_frames(samples.len()),
            "{name}: frame count disagrees with the advertised formula"
        );
    }
}

/// Mirrors `fuzz_targets/normalize_text.rs`: total over any `&str`, and
/// byte-stable across calls.
#[test]
fn normalize_text_corpus_is_total_and_deterministic() {
    use ffai_mercury::tts::normalize::normalize;
    let cases = inputs("normalize_text");
    assert!(!cases.is_empty(), "normalize_text corpus is empty - regression seeds lost?");
    for (name, bytes) in cases {
        let Ok(text) = std::str::from_utf8(&bytes) else { continue };
        let a = normalize(text);
        let b = normalize(text);
        assert_eq!(a, b, "{name}: normalize is not deterministic");
    }
}

/// Mirrors `fuzz_targets/lexicon_parse.rs`. `Lexicon::load` takes a path, so the
/// bytes are materialised — same shape as the fuzz target, and the same reason:
/// there is no `from_bytes` entry point yet.
#[test]
fn lexicon_parse_corpus_is_total() {
    let cases = inputs("lexicon_parse");
    assert!(!cases.is_empty(), "lexicon_parse corpus is empty - regression seeds lost?");
    let dir = std::env::temp_dir().join(format!("ffai-lexicon-replay-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    for (name, bytes) in cases {
        let path = dir.join(format!("{name}.dict"));
        std::fs::write(&path, &bytes).expect("write corpus input");
        let _ = ffai_mercury::tts::lexicon::Lexicon::load(&path);
        let _ = std::fs::remove_file(&path);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
