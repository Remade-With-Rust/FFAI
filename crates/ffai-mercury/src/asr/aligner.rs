//! Loads the CTC acoustic model and turns a transcript into word timings.
//!
//! The glue between [`super::wav2vec2`] (which produces emissions) and
//! [`super::align`] (which aligns text to them). Kept apart from both so the
//! algorithm stays model-free and the model stays transcript-free; this is the
//! only place that knows both exist.

use std::path::Path;

use ffai_core::candle::{DType, Device};
use ffai_core::error::{Error, Result};
use ffai_core::types::TimedSegment;

use super::align::{align_words, CtcAlphabet};
use super::wav2vec2::{Config, Wav2Vec2Ctc};

/// Default manifest name. English only — the alignment model is per-language
/// and shipping one while claiming coverage of others would be a lie the
/// output format cannot express.
pub const DEFAULT_MODEL: &str = "wav2vec2-base-960h";

pub struct Aligner {
    model: Wav2Vec2Ctc,
    alphabet: CtcAlphabet,
    sample_rate: usize,
}

impl Aligner {
    /// Load from a manifest directory, the same way the ASR model is loaded.
    pub fn from_manifest_dir(dir: &Path, name: &str, device: Device) -> Result<Self> {
        let manifests = ffai_models::load_dir(dir)?;
        let manifest = manifests.into_iter().find(|m| m.name == name).ok_or_else(|| {
            Error::Model(format!(
                "no model manifest named `{name}` in {} — word timestamps need a CTC \
                 alignment model (see models/{DEFAULT_MODEL}.toml)",
                dir.display()
            ))
        })?;
        let resolved = manifest.fetch()?;
        let weights = resolved.file("model.safetensors")?.to_path_buf();

        // SAFETY: as in `model.rs` — the Hugging Face cache blob is immutable
        // and never written by FFai, which is the condition mmap requires.
        #[allow(unsafe_code)]
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)
                .map_err(|e| Error::Model(format!("mapping alignment weights: {e}")))?
        };
        let model = Wav2Vec2Ctc::load(Config::base_960h(), vb, device)?;
        Ok(Aligner {
            model,
            alphabet: CtcAlphabet::wav2vec2_english(),
            sample_rate: super::mel::SAMPLE_RATE,
        })
    }

    /// Align each segment's own text against its own audio.
    ///
    /// Per segment rather than whole-file, matching WhisperX: it bounds the
    /// trellis (which is `frames × characters`, and quadratic growth on an
    /// hour of audio is not a rounding error), and it keeps a
    /// misrecognised word from dragging every later timestamp with it.
    ///
    /// A segment that cannot be aligned — too short for its own text, or
    /// emissions that go degenerate — is **skipped, not faked**. Its words are
    /// absent from the output rather than being assigned the segment's own
    /// bounds, which would look like a successful alignment and be wrong.
    pub fn align_segments(
        &self,
        samples: &[f32],
        segments: &[TimedSegment<String>],
    ) -> Vec<TimedSegment<String>> {
        let sr = self.sample_rate as f64;
        let mut words = Vec::new();
        for seg in segments {
            let start = ((seg.start.max(0.0) * sr) as usize).min(samples.len());
            let end = ((seg.end.max(0.0) * sr).ceil() as usize).clamp(start, samples.len());
            let slice = &samples[start..end];
            if self.model.config().output_frames(slice.len()) == 0 {
                continue;
            }
            let Ok(emissions) = self.model.emissions(slice, self.sample_rate) else {
                continue;
            };
            if let Ok(mut w) = align_words(&emissions, &seg.value, &self.alphabet, seg.start) {
                words.append(&mut w);
            }
        }
        words
    }
}
