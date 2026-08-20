//! Stage oracle: Mercury's mel front-end vs openai-whisper's own output.
//!
//! This is the test that makes the front-end defensible rather than lucky.
//! A WER regression can come from any stage; a passing mel oracle removes
//! one box from the search in seconds (mission plan §7).
//!
//! The fixture is regenerated with:
//!
//! ```sh
//! .venv-bench/Scripts/python corpora/refs/dump_whisper_mel.py \
//!     --out crates/ffai-mercury/tests/fixtures/mel_oracle_80.f32
//! ```
//!
//! The input is a deterministic formula shared by both sides, so the fixture
//! carries no audio and no license.

use ffai_mercury::asr::mel::{MelSpectrogram, SAMPLE_RATE};

/// Must match `reference_signal` in corpora/refs/dump_whisper_mel.py exactly.
fn reference_signal(n: usize, sample_rate: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            let chirp = (2.0 * std::f64::consts::PI * (200.0 + 1500.0 * t) * t).sin();
            let tone = 0.5 * (2.0 * std::f64::consts::PI * 3000.0 * t).sin();
            (0.6 * (chirp + tone)) as f32
        })
        .collect()
}

fn load_fixture(path: &str) -> Option<(usize, usize, Vec<f32>)> {
    let bytes = std::fs::read(path).ok()?;
    let n_mels = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let n_frames = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    let data: Vec<f32> = bytes[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();
    assert_eq!(
        data.len(),
        n_mels * n_frames,
        "fixture header disagrees with its payload"
    );
    Some((n_mels, n_frames, data))
}

#[test]
fn mel_matches_openai_whisper() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mel_oracle_80.f32"
    );
    let Some((n_mels, n_frames, expected)) = load_fixture(path) else {
        panic!("missing oracle fixture {path} — regenerate with corpora/refs/dump_whisper_mel.py");
    };

    let samples = reference_signal(SAMPLE_RATE, SAMPLE_RATE);
    let ours = MelSpectrogram::new(n_mels).compute(&samples);

    assert_eq!(ours.n_mels, n_mels);
    assert_eq!(
        ours.n_frames, n_frames,
        "frame count must match torch.stft's"
    );

    let mut max_abs = 0.0f32;
    let mut worst = (0usize, 0usize);
    for m in 0..n_mels {
        for f in 0..n_frames {
            let i = m * n_frames + f;
            let d = (ours.data[i] - expected[i]).abs();
            if d > max_abs {
                max_abs = d;
                worst = (m, f);
            }
        }
    }
    let mean_abs: f32 = ours
        .data
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / expected.len() as f32;

    // Tolerance covers f32 accumulation-order differences between our FFT and
    // torch's, not modelling differences. Values live in roughly [-1, 1]
    // after Whisper's final scaling, so 1e-3 is ~0.05% of full scale.
    assert!(
        max_abs < 1e-3,
        "mel diverges from openai-whisper: max |Δ| = {max_abs:.3e} at mel band {}, frame {} \
         (mean |Δ| = {mean_abs:.3e})",
        worst.0,
        worst.1
    );
}
