//! `whisper-candle` — Mercury's reference ASR engine.
//!
//! Composes the independent stages: [`mel`] → [`model`] → [`decoder`] →
//! [`ffai_core::types::Transcript`]. The engine itself holds no algorithm;
//! it wires stages and manages the loaded model, which is exactly the
//! division that makes each stage separately testable.

use std::path::PathBuf;
use std::sync::Mutex;

use ffai_core::engine::{AsrEngine, AsrOptions, EngineInfo, EngineStatus, Task};
use ffai_core::error::{Error, Result};
use ffai_core::types::{AudioBuffer, Transcript};

use super::decoder::{self, DecodeConfig};
use super::mel::{self, MelSpectrogram};
use super::model::LoadedWhisper;
use super::text_decoder::Precision;

/// OpenAI Whisper running on candle.
///
/// The model is loaded lazily on first use and reused across calls, so
/// per-clip timing measures inference rather than weight loading (see
/// docs/benchmarking.md on warm vs end-to-end throughput).
pub struct WhisperCandle {
    manifest_dir: PathBuf,
    model_name: String,
    precision: Precision,
    // Mutex because decoding mutates the KV cache; the engine trait is
    // `Send + Sync` so callers can share one engine across threads.
    state: Mutex<Option<State>>,
}

struct State {
    whisper: LoadedWhisper,
    front_end: MelSpectrogram,
}

impl WhisperCandle {
    /// Default: `whisper-tiny-en` from the repo's `models/` directory — the
    /// M1 bring-up target, matching the M0 baseline configuration.
    pub fn new() -> Self {
        Self::with_model("models", "whisper-tiny-en", Precision::F32)
    }

    pub fn with_model(
        manifest_dir: impl Into<PathBuf>,
        model_name: impl Into<String>,
        precision: Precision,
    ) -> Self {
        WhisperCandle {
            manifest_dir: manifest_dir.into(),
            model_name: model_name.into(),
            precision,
            state: Mutex::new(None),
        }
    }
}

impl Default for WhisperCandle {
    fn default() -> Self {
        Self::new()
    }
}

impl AsrEngine for WhisperCandle {
    fn info(&self) -> EngineInfo {
        // The default model is the bare engine name; other sizes get a
        // suffix, so `--engine whisper-candle-base` selects base.en.
        let mut name = match self.model_name.as_str() {
            "whisper-tiny-en" => "whisper-candle".to_string(),
            other => format!(
                "whisper-candle-{}",
                other.trim_start_matches("whisper-").trim_end_matches("-en")
            ),
        };
        if self.precision != Precision::F32 {
            name.push('-');
            name.push_str(self.precision.label());
        }
        EngineInfo {
            name,
            task: Task::Asr,
            status: EngineStatus::Experimental,
            description: format!(
                "OpenAI Whisper on candle ({}, decoder {}) — greedy decode",
                self.model_name,
                self.precision.label()
            ),
        }
    }

    fn transcribe(&self, audio: &AudioBuffer, opts: &AsrOptions) -> Result<Transcript> {
        if opts.word_timestamps || opts.diarize {
            return Err(Error::Other(
                "--word-timestamps and --diarize are the WhisperX layer, scheduled for Mercury \
                 M3 (see docs/mercury-mission-plan.md §3.2)"
                    .into(),
            ));
        }

        let mut guard = self.state.lock().map_err(|_| {
            Error::Other("whisper-candle state lock poisoned by an earlier panic".into())
        })?;
        if guard.is_none() {
            let whisper = LoadedWhisper::from_manifest_dir(
                &self.manifest_dir,
                &self.model_name,
                ffai_core::best_device(),
                self.precision,
            )?;
            let front_end = MelSpectrogram::new(whisper.n_mels());
            *guard = Some(State { whisper, front_end });
        }
        let state = guard.as_mut().expect("state initialized above");

        // Whisper is a 16 kHz mono model; anything else is resampled/downmixed
        // by ffai-media before it reaches here.
        if audio.sample_rate != mel::SAMPLE_RATE as u32 {
            return Err(Error::Media(format!(
                "whisper-candle needs {} Hz audio, got {} Hz — resample on ingest",
                mel::SAMPLE_RATE,
                audio.sample_rate
            )));
        }
        let mono = audio.to_mono();

        let language = match (&opts.language, state.whisper.is_english_only()) {
            (Some(code), false) => Some(state.whisper.tokenizer.language(code).ok_or_else(|| {
                Error::Other(format!("model has no language token for `{code}`"))
            })?),
            // An .en model has no language slot; forcing English is a no-op
            // rather than an error.
            (Some(code), true) if !code.eq_ignore_ascii_case("en") => {
                return Err(Error::Other(format!(
                    "model `{}` is English-only but language `{code}` was requested",
                    state.whisper.name
                )))
            }
            _ => None,
        };
        let cfg = DecodeConfig { language, translate: opts.translate, ..Default::default() };

        let mut segments = Vec::new();
        for (start_sample, window) in decoder::windows(&mono.samples) {
            let offset_secs = start_sample as f64 / mel::SAMPLE_RATE as f64;
            let window_secs = window.len() as f64 / mel::SAMPLE_RATE as f64;
            // Pad in the sample domain, before the spectrogram — see
            // mel::pad_or_trim for why the distinction matters.
            //
            // The full 30 s window is NOT waste that can simply be trimmed:
            // shortening the encoder input was measured and pruned (mission
            // plan §6.4) — it induces decoder repetition loops and took corpus
            // WER from 3.00 % to 268 %. Whisper is trained exclusively on 30 s
            // contexts and the decoder's cross-attention depends on it.
            let chunk = super::profile::timed(&super::profile::profile().mel, || {
                state.front_end.compute(&mel::pad_or_trim(window))
            });
            segments.extend(decoder::decode_window(
                &mut state.whisper,
                &chunk,
                offset_secs,
                &cfg,
                window_secs,
            )?);
        }

        // TRIED AND REVERTED: `ffai_core::release_load_arena()` here, to hand
        // back the ~240 MiB of one-time transients the allocator keeps after
        // weight loading. It measured FLAT — 345.1 MiB held either way — at
        // both call sites tried (after load, and after the first complete
        // pass). The trim itself works (resident drops to 0.2 MiB), but the
        // following passes fault the same pages straight back, so the memory
        // is evidently touched again rather than being dead.
        //
        // That contradicts the standalone probe, where trimming after four
        // passes and repeating them re-settles at 102 MiB. Both measurements
        // are real and they disagree about WHICH pages the work touches;
        // resolving it needs allocation profiling, not another guess. Left
        // reverted rather than shipped inert, with the numbers here so the
        // next attempt starts from them (see examples/mem_claims.rs).
        Ok(Transcript {
            language: opts.language.clone().or_else(|| {
                state.whisper.is_english_only().then(|| "en".to_string())
            }),
            segments,
        })
    }
}
