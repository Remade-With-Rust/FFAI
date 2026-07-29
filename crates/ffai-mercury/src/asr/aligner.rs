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

        // FFAI_ALIGN_DTYPE selects how the 95M-parameter alignment model is
        // stored. Both alternatives to f32 were measured.
        //
        // `q8_0` — WORKS, and is a real trade rather than a free win:
        //
        //     steady   509 -> 266 MiB  (-48 %)
        //     peak     725 -> 500 MiB  (-31 %)
        //     quality  identical: 1105/1105 words contained, 100 % coverage
        //     speed    63.8 -> 183.7 s over the four-file gate (2.9x SLOWER)
        //
        // Per-file it measures at parity (11.9 vs 12.0 s on one 83 s file;
        // 28.8 vs 28.8 s on one 185 s file), and only the multi-file run in a
        // single process shows the cost. That discrepancy is NOT explained —
        // an arena trim was suspected and removing it changed nothing — and it
        // is recorded as open rather than smoothed over.
        //
        // So it is opt-in: halve the memory when memory is what binds, keep
        // f32 when time is. Neither is the obviously right default.
        //
        // `f16` — a documented DEAD END:
        //
        //   f16 emissions go degenerate and forced alignment finds no valid
        //   path on any segment. Not a wiring fault — the input-dtype cast
        //   that used to make this fail as "dtype mismatch in conv1d" is
        //   fixed — but a numerical one. wav2vec2-base was not trained for
        //   half precision and its conv-stack activations do not survive the
        //   range.
        //
        // The next attempt at C3 should be mixed precision (f16 storage, f32
        // accumulation) or int8 with per-channel scales, not a blanket dtype
        // switch. Left switchable so that work starts from a measurement
        // rather than repeating this one.
        let mode = std::env::var("FFAI_ALIGN_DTYPE").unwrap_or_default();
        // Weights are read as f32 whatever the mode: Q8_0 quantizes FROM f32,
        // and f16 is only kept as the documented dead end above.
        let dtype = if mode == "f16" { DType::F16 } else { DType::F32 };
        let quantized = mode == "q8_0";

        // SAFETY: as in `model.rs` — the Hugging Face cache blob is immutable
        // and never written by FFai, which is the condition mmap requires.
        #[allow(unsafe_code)]
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)
                .map_err(|e| Error::Model(format!("mapping alignment weights: {e}")))?
        };
        if std::env::var_os("FFAI_DEBUG_TOKENS").is_some() {
            eprintln!("[align] dtype={dtype:?} quantized={quantized}");
        }
        let model = Wav2Vec2Ctc::load_with(Config::base_960h(), vb, device, quantized)?;
        // NO arena trim here, and the reason is worth keeping.
        //
        // The f32 weights are dead once quantized, so handing their pages back
        // looks obviously right. It is not: a trim before any work means every
        // subsequent forward re-faults the quantized weights it is about to
        // use. Measured on the long-form gate — four files in one process —
        // that cost 66s -> 185s, a 2.8x slowdown, while per-file CLI runs
        // (one trim, one file) showed parity and hid it completely.
        //
        // This is the same rule whisper_candle already documents: trim after
        // the first COMPLETE pass, never before work, "where trimming what you
        // are about to touch again only buys page faults". The Whisper path
        // gets it right; this one had to learn it twice.
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
    ///
    /// But skipping *every* segment is a broken model, not a quiet edge case,
    /// and this used to return an empty list for it. That is how an f16 build
    /// producing nothing at all looked like "no words" instead of a failure —
    /// and it is only the gate's **coverage** column that caught it, because
    /// containment over zero words is trivially 100 %. If nothing aligns, say
    /// so.
    pub fn align_segments(
        &self,
        samples: &[f32],
        segments: &[TimedSegment<String>],
    ) -> Result<Vec<TimedSegment<String>>> {
        let sr = self.sample_rate as f64;
        let mut words = Vec::new();
        let mut attempted = 0usize;
        let mut failed = 0usize;
        let mut first_error: Option<String> = None;
        for seg in segments {
            let start = ((seg.start.max(0.0) * sr) as usize).min(samples.len());
            let end = ((seg.end.max(0.0) * sr).ceil() as usize).clamp(start, samples.len());
            let slice = &samples[start..end];
            if self.model.config().output_frames(slice.len()) == 0 {
                continue;
            }
            attempted += 1;
            match self
                .model
                .emissions(slice, self.sample_rate)
                .map_err(|e| e.to_string())
                .and_then(|e| align_words(&e, &seg.value, &self.alphabet, seg.start))
            {
                Ok(mut w) => words.append(&mut w),
                Err(e) => {
                    failed += 1;
                    first_error.get_or_insert(e);
                }
            }
        }

        if std::env::var_os("FFAI_DEBUG_TOKENS").is_some() {
            eprintln!(
                "[align] segments={} attempted={attempted} failed={failed} words={} first_err={:?}",
                segments.len(),
                words.len(),
                first_error
            );
        }

        // Segments present but none even attempted means every one was too
        // short for a single frame — a configuration error, not a hard clip.
        if !segments.is_empty() && attempted == 0 {
            return Err(Error::Model(format!(
                "alignment attempted no segments: all {} were shorter than one model frame",
                segments.len()
            )));
        }
        // Every segment failing is a broken model. One or two failing is a
        // hard segment, which is what the skip is for.
        if attempted > 0 && failed == attempted {
            return Err(Error::Model(format!(
                "alignment failed on all {attempted} segments — the model produced nothing                  usable. First error: {}",
                first_error.as_deref().unwrap_or("(none reported)")
            )));
        }
        Ok(words)
    }
}
