//! Property tests for the invariants this crate advertises (gate H-28).
//!
//! Scope is deliberately the MODEL-FREE surface, so these run in CI where no
//! weights are cached: the audio front-end and text normalization. That is also
//! where the real trust boundary now sits — model files are trusted input
//! (see docs/threat-model.md), but the audio and text a CALLER hands us are not.
//! A caller may legitimately pass an empty buffer, a single sample, NaNs, or
//! infinities, and none of that may panic.

use ffai_mercury::asr::mel::{self, MelSpectrogram};
use ffai_mercury::tts::normalize::normalize;
use proptest::prelude::*;

/// Samples including the values a real decoder can produce: NaN and +/-inf.
fn any_samples(max: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        prop_oneof![
            4 => (-1.0f32..1.0),
            1 => prop::num::f32::ANY,          // NaN, inf, subnormals
        ],
        0..max,
    )
}

proptest! {
    /// `pad_or_trim_to` is documented as "zero-pad or truncate to `target`".
    /// The length contract is what every downstream shape calculation assumes.
    #[test]
    fn pad_or_trim_to_always_returns_target_len(
        samples in any_samples(4096),
        target in 0usize..8192,
    ) {
        prop_assert_eq!(mel::pad_or_trim_to(&samples, target).len(), target);
    }

    /// Truncation must preserve the prefix; padding must be zeros.
    #[test]
    fn pad_or_trim_to_preserves_prefix_and_zero_fills(
        samples in prop::collection::vec(-1.0f32..1.0, 0..512),
        target in 0usize..1024,
    ) {
        let out = mel::pad_or_trim_to(&samples, target);
        let take = samples.len().min(target);
        prop_assert_eq!(&out[..take], &samples[..take]);
        prop_assert!(out[take..].iter().all(|&x| x == 0.0));
    }

    /// The front end must be TOTAL over caller audio. Short buffers are the
    /// interesting case: `compute` reflect-pads by N_FFT/2, and reflect padding
    /// of a buffer shorter than the pad width is where an out-of-bounds index
    /// would live if there were one.
    #[test]
    fn mel_compute_never_panics(samples in any_samples(2048)) {
        let m = MelSpectrogram::new(80);
        let chunk = m.compute(&samples);
        // Shape contract: data is (n_mels, n_frames) row-major.
        prop_assert_eq!(chunk.n_frames, MelSpectrogram::n_frames(samples.len()));
        prop_assert_eq!(chunk.data.len(), chunk.n_mels * chunk.n_frames);
    }

    /// Byte-stable determinism is an advertised property of this crate — the
    /// 0.7.0 version bump exists because synthesis output changed. Here it is
    /// pinned at the front end, where it can be checked without weights.
    #[test]
    fn mel_compute_is_bit_deterministic(samples in any_samples(1024)) {
        let m = MelSpectrogram::new(80);
        let a = m.compute(&samples);
        let b = m.compute(&samples);
        prop_assert_eq!(a.data.len(), b.data.len());
        // to_bits(), not ==: NaN != NaN, and we are asserting BYTE stability.
        prop_assert!(
            a.data.iter().zip(b.data.iter()).all(|(x, y)| x.to_bits() == y.to_bits()),
            "two identical calls produced different bytes"
        );
    }

    /// `resized` is a pad-or-truncate on the frame axis; same length contract.
    #[test]
    fn mel_resized_has_requested_frames(
        samples in any_samples(1024),
        frames in 0usize..64,
    ) {
        let chunk = MelSpectrogram::new(80).compute(&samples);
        let r = chunk.resized(frames);
        prop_assert_eq!(r.n_frames, frames);
        prop_assert_eq!(r.data.len(), r.n_mels * frames);
    }

    /// Text normalization is total: any string a caller can build, including
    /// control characters, lone surrogught-free unicode and huge digit runs.
    #[test]
    fn normalize_never_panics(text in ".*") {
        let _ = normalize(&text);
    }

    /// ...and deterministic, for the same byte-stability reason.
    #[test]
    fn normalize_is_deterministic(text in ".*") {
        prop_assert_eq!(normalize(&text), normalize(&text));
    }
}
