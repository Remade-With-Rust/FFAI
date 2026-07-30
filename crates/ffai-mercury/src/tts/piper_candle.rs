//! `piper-candle`: the piper voice family on candle, end to end in Rust.
//!
//! Composes the independent stages exactly as the mission plan's §2 tree
//! promises: [`super::phonemize`] (pure-Rust G2P, gated by M-T1's
//! substitution test) → [`super::phoneme_ids`] (the voice's id map) →
//! [`super::vits`] (candle inference, gated by M-T2's stage oracles).
//!
//! Two deliberate differences from piper itself:
//! - **Deterministic by default.** All noise is seeded (seed 0); the same
//!   call produces the same bytes. piper samples in-graph and cannot do
//!   this (mission plan §3.3).
//! - Synthesis knobs default to the VOICE CONFIG's own values, so matched
//!   comparisons against piper compare implementations, not settings.
//!
//! Weights: converted from the voice's own `.onnx` by
//! `corpora/refs/dump_piper_weights.py` into the model cache — see
//! `models/piper-vits-lessac-medium.toml`. Weights are data, never vendored.

use std::path::PathBuf;
use std::sync::OnceLock;

use ffai_core::engine::{EngineInfo, EngineStatus, Task, TtsEngine, TtsOptions};
use ffai_core::error::{Error, Result};
use ffai_core::types::AudioBuffer;

use super::phonemize::Phonemizer;
use super::vits::{SynthesisOptions, Vits};

/// The default (and currently only) converted voice.
pub const DEFAULT_VOICE: &str = "piper-vits-lessac-medium";

pub struct PiperCandle {
    loaded: OnceLock<Result<Loaded>>,
}

struct Loaded {
    vits: Vits,
    phonemizer: Phonemizer,
}

impl PiperCandle {
    pub fn new() -> Self {
        PiperCandle { loaded: OnceLock::new() }
    }

    fn load(&self) -> Result<&Loaded> {
        self.loaded
            .get_or_init(|| {
                let dir = model_dir();
                let vits = Vits::load(&dir).map_err(|e| {
                    Error::Model(format!(
                        "piper-candle voice not converted: {e} — run \
                         corpora/refs/dump_piper_weights.py (see models/{DEFAULT_VOICE}.toml)"
                    ))
                })?;
                let phonemizer = Phonemizer::from_manifest_dir(std::path::Path::new("models"))?;
                Ok(Loaded { vits, phonemizer })
            })
            .as_ref()
            .map_err(clone_err)
    }
}

impl PiperCandle {
    /// The phonemes this engine will feed the model for `text`, one entry per
    /// sentence — the G2P's actual output.
    ///
    /// Exposed because it is the stage most worth inspecting: pronunciation
    /// bugs are invisible in a waveform and obvious in IPA. It runs the SAME
    /// chunker and phonemizer [`TtsEngine::synthesize`] does, so what a caller
    /// displays is what the model was given, not a re-derivation that could
    /// drift from it.
    pub fn phonemes(&self, text: &str) -> Result<Vec<String>> {
        let loaded = self.load()?;
        super::chunk::sentences(text)
            .iter()
            .map(|s| loaded.phonemizer.phonemize(s))
            .collect()
    }

    /// The voice's native sample rate, once loaded.
    pub fn sample_rate(&self) -> Result<u32> {
        Ok(self.load()?.vits.sample_rate)
    }
}

impl Default for PiperCandle {
    fn default() -> Self {
        Self::new()
    }
}

fn model_dir() -> PathBuf {
    ffai_models::cache_dir().join("models").join(DEFAULT_VOICE)
}

fn clone_err(e: &Error) -> Error {
    Error::Model(e.to_string())
}

impl TtsEngine for PiperCandle {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "piper-candle".into(),
            task: Task::Tts,
            status: EngineStatus::Experimental,
            description: format!(
                "piper voices (VITS) on candle, pure Rust — voice {DEFAULT_VOICE}, \
                 deterministic (seeded noise)"
            ),
        }
    }

    fn synthesize(&self, text: &str, opts: &TtsOptions) -> Result<AudioBuffer> {
        if let Some(voice) = &opts.voice {
            if voice != DEFAULT_VOICE && voice != "en_US-lessac-medium" {
                return Err(Error::Model(format!(
                    "voice `{voice}` is not converted — {DEFAULT_VOICE} is the M-T2 voice; \
                     the tier sweep lands in M-T4"
                )));
            }
        }
        let loaded = self.load()?;
        let defaults = loaded.vits.defaults;
        let synth = SynthesisOptions {
            // speed is the user knob; length_scale is its reciprocal. The
            // noise knobs default to the voice's own values (`None`).
            length_scale: defaults.length_scale / opts.speed.max(0.1) as f64,
            noise_scale: opts.noise_scale.map(f64::from).unwrap_or(defaults.noise_scale),
            noise_w: opts.noise_w.map(f64::from).unwrap_or(defaults.noise_w),
            seed: opts.seed,
        };

        let sample_rate = loaded.vits.sample_rate;
        let gap = ((opts.sentence_silence_s.max(0.0)) * sample_rate as f32) as usize;
        let mut samples: Vec<f32> = Vec::new();
        let mut total_unknown = 0usize;
        for (i, sentence) in super::chunk::sentences(text).iter().enumerate() {
            let ipa = loaded.phonemizer.phonemize(sentence)?;
            let (ids, unknown) = loaded.vits.id_map.sentence_to_ids(&ipa);
            total_unknown += unknown;
            if i > 0 {
                samples.extend(std::iter::repeat_n(0.0f32, gap));
            }
            samples.extend(loaded.vits.synthesize_ids(&ids, &synth)?);
        }
        if total_unknown > 0 {
            eprintln!(
                "piper-candle: {total_unknown} phoneme(s) outside the voice's id map, skipped"
            );
        }
        Ok(AudioBuffer { samples, sample_rate, channels: 1 })
    }
}
