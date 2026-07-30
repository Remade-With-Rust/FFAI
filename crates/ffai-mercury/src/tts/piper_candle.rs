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

/// The manifests this engine needs, COMPILED IN.
///
/// They live in the crate (`manifests/`) rather than being read from a
/// `models/` directory beside the caller, because a library must not depend
/// on its consumer's working directory. `ffai-mercury` on crates.io has no
/// repo checkout to read, and the same `Path::new("models")` assumption is
/// why a plain `cargo add` + `PiperCandle::new()` failed outside this tree.
///
/// The repo's `models/*.toml` remain the copies `ffai models` lists; the test
/// at the bottom of this file fails if the two drift apart.
const CMUDICT_MANIFEST: &str = include_str!("../../manifests/cmudict.toml");
const VOICE_MANIFEST: &str = include_str!("../../manifests/piper-vits-lessac-medium.toml");

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
    pub fn new() -> Self {
        PiperCandle { manifest_dir: None, loaded: OnceLock::new() }
    }

    /// Read the `cmudict` and voice manifests from `dir` instead of using the
    /// compiled-in copies — for a caller who ships their own manifests (a
    /// different voice, a relocated cache, a pinned checksum).
    pub fn with_manifest_dir(dir: impl AsRef<Path>) -> Self {
        PiperCandle {
            manifest_dir: Some(dir.as_ref().to_path_buf()),
            loaded: OnceLock::new(),
        }
    }

    /// A manifest by name: from `manifest_dir` when one was given, otherwise
    /// the compiled-in copy.
    fn manifest(&self, name: &str, embedded: &str) -> Result<ModelManifest> {
        match &self.manifest_dir {
            Some(dir) => ModelManifest::load(&dir.join(format!("{name}.toml"))),
            None => ModelManifest::from_toml(embedded),
        }
    }

    fn load(&self) -> Result<&Loaded> {
        self.loaded
            .get_or_init(|| {
                let voice = self.manifest(DEFAULT_VOICE, VOICE_MANIFEST)?;
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
                let phonemizer =
                    Phonemizer::from_manifest(&self.manifest("cmudict", CMUDICT_MANIFEST)?)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifests_parse_and_name_what_they_claim() {
        let voice = ModelManifest::from_toml(VOICE_MANIFEST).expect("voice manifest parses");
        assert_eq!(voice.name, DEFAULT_VOICE);
        assert_eq!(voice.task, "tts");
        // The licence field is the whole point of a manifest (principle 4);
        // an empty one would ship a voice with no licence surfaced.
        assert!(!voice.license.is_empty());

        let dict = ModelManifest::from_toml(CMUDICT_MANIFEST).expect("cmudict manifest parses");
        assert_eq!(dict.name, "cmudict");
        assert!(dict.files.iter().any(|f| f.name == "cmudict.dict"));
    }

    #[test]
    fn embedded_manifests_match_the_repo_copies() {
        // Two copies exist on purpose: the crate needs its own (a published
        // crate has no repo checkout to read), and `models/` is what
        // `ffai models` lists. This test is the guard against them drifting.
        //
        // Skipped when the repo files are absent — that is the packaged-crate
        // case, where there is nothing to drift from.
        for (embedded, name) in
            [(CMUDICT_MANIFEST, "cmudict.toml"), (VOICE_MANIFEST, "piper-vits-lessac-medium.toml")]
        {
            let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models").join(name);
            let Ok(text) = std::fs::read_to_string(&repo) else { continue };
            assert_eq!(
                text.replace("\r\n", "\n"),
                embedded.replace("\r\n", "\n"),
                "crates/ffai-mercury/manifests/{name} has drifted from models/{name} — \
                 copy the repo version over the crate's"
            );
        }
    }
}
