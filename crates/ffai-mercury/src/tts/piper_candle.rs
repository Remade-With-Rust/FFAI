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

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ffai_core::engine::{EngineInfo, EngineStatus, Task, TtsEngine, TtsOptions};
use ffai_core::error::{Error, Result};
use ffai_core::types::AudioBuffer;
use ffai_models::ModelManifest;

use super::phonemize::Phonemizer;
use super::vits::{SynthesisOptions, Vits};

/// The default (and currently only) converted voice.
pub const DEFAULT_VOICE: &str = "piper-vits-lessac-medium";

pub struct PiperCandle {
    /// `None` = use the compiled-in manifests. `Some(dir)` = read them from
    /// there instead, for a caller who wants to point at their own.
    manifest_dir: Option<PathBuf>,
    loaded: OnceLock<Result<Loaded>>,
}

struct Loaded {
    vits: Vits,
    phonemizer: Phonemizer,
}

impl PiperCandle {
    /// The default engine: compiled-in manifests, weights from the cache.
    /// Works from any working directory.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            manifest_dir: None,
            loaded: OnceLock::new(),
        }
    }

    /// Read the `cmudict` and voice manifests from `dir` instead of using the
    /// compiled-in copies — for a caller who ships their own manifests (a
    /// different voice, a relocated cache, a pinned checksum).
    pub fn with_manifest_dir(dir: impl AsRef<Path>) -> Self {
        Self {
            manifest_dir: Some(dir.as_ref().to_path_buf()),
            loaded: OnceLock::new(),
        }
    }

    /// A manifest by name: from `manifest_dir` when one was given, otherwise
    /// the copy compiled into [`crate::manifests`].
    fn manifest(&self, name: &str) -> Result<ModelManifest> {
        crate::manifests::resolve(self.manifest_dir.as_deref(), name)
    }

    fn load(&self) -> Result<&Loaded> {
        self.loaded
            .get_or_init(|| {
                let voice = self.manifest(DEFAULT_VOICE)?;
                // Fetch the files piper itself ships and read the ONNX
                // directly — no conversion step, no Python, no ONNX runtime.
                let resolved = voice.fetch().map_err(|e| {
                    Error::Model(format!(
                        "piper-candle could not obtain the voice: {e}\n  \
                         fetch it explicitly with: ffai models --fetch {DEFAULT_VOICE}"
                    ))
                })?;
                let onnx = resolved
                    .files
                    .iter()
                    .find(|(n, _)| n.ends_with(".onnx"))
                    .map(|(_, p)| p.clone())
                    .ok_or_else(|| Error::Model("voice manifest lists no .onnx".into()))?;
                let config = resolved
                    .files
                    .iter()
                    .find(|(n, _)| n.ends_with(".onnx.json"))
                    .map(|(_, p)| p.clone())
                    .ok_or_else(|| Error::Model("voice manifest lists no .onnx.json".into()))?;
                let vits = Vits::load_onnx(&onnx, &config)?;
                let phonemizer = Phonemizer::from_manifest(&self.manifest("cmudict")?)?;
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

#[allow(dead_code)] // default-voice path helper, retained for the manifest-free fallback
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
        if let Some(voice) = &opts.voice
            && voice != DEFAULT_VOICE
            && voice != "en_US-lessac-medium"
        {
            return Err(Error::Model(format!(
                "voice `{voice}` is not converted — {DEFAULT_VOICE} is the M-T2 voice; \
                     the tier sweep lands in M-T4"
            )));
        }
        let loaded = self.load()?;
        let defaults = loaded.vits.defaults;
        let synth = SynthesisOptions {
            // speed is the user knob; length_scale is its reciprocal. The
            // noise knobs default to the voice's own values (`None`).
            length_scale: defaults.length_scale / f64::from(opts.speed.max(0.1)),
            noise_scale: opts.noise_scale.map_or(defaults.noise_scale, f64::from),
            noise_w: opts.noise_w.map_or(defaults.noise_w, f64::from),
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
        Ok(AudioBuffer {
            samples,
            sample_rate,
            channels: 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_voice_and_lexicon_resolve_without_a_working_directory() {
        // Parsing and drift are covered once for all ten manifests in
        // `crate::manifests`; what matters here is that THIS engine's two
        // names resolve with no directory at all.
        let voice = PiperCandle::new().manifest(DEFAULT_VOICE).expect("voice");
        assert_eq!(voice.name, DEFAULT_VOICE);
        assert_eq!(voice.task, "tts");
        // Fetched from the voice repo piper itself publishes, as an ONNX we
        // read directly — if this stops being true the setup story changes.
        assert_eq!(voice.hf_repo.as_deref(), Some("rhasspy/piper-voices"));
        assert!(voice.files.iter().any(|f| f.name.ends_with(".onnx")));

        let dict = PiperCandle::new().manifest("cmudict").expect("cmudict");
        assert!(dict.files.iter().any(|f| f.name == "cmudict.dict"));
    }
}
