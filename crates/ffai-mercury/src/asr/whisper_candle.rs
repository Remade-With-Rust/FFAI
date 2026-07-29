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
use super::aligner::{Aligner, DEFAULT_MODEL as ALIGN_MODEL};
use super::diarize;
use super::diarizer::{Diarizer, DEFAULT_MODEL as DIARIZE_MODEL};
use super::registry::SpeakerRegistry;
use super::text_decoder::Precision;
use super::vad;

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
    /// Loaded on first `--diarize` call and never before.
    ///
    /// Separate lock from `aligner` so requesting one stage never serialises
    /// the other, and lazy so the flag's promise holds: without `--diarize`
    /// the 83 MB speaker model is not fetched, not read, and not resident.
    diarizer: Mutex<Option<std::sync::Arc<Diarizer>>>,
    /// Speaker identities carried between calls when `persist_speakers` is
    /// set. Untouched otherwise, so the default path allocates nothing.
    speakers: Mutex<Option<SpeakerRegistry>>,
    /// Loaded on first `--word-timestamps` call and never before.
    ///
    /// Separate from `state` so the ASR path cannot be blocked behind
    /// alignment, and lazy so the flag's promise holds: without it, the
    /// 95M-parameter model is not fetched, not mapped, and not resident.
    aligner: Mutex<Option<std::sync::Arc<Aligner>>>,
}

struct State {
    whisper: LoadedWhisper,
    front_end: MelSpectrogram,
    /// Whether the one-time arena release has run yet.
    warmed: bool,
}

impl WhisperCandle {
    /// The alignment model, loaded on first use.
    ///
    /// Held behind an `Arc` so the lock is released before the (slow)
    /// alignment runs — otherwise one thread asking for word timestamps would
    /// serialise every other thread's plain transcription behind it.
    fn aligner(&self) -> Result<std::sync::Arc<Aligner>> {
        let mut guard = self
            .aligner
            .lock()
            .map_err(|_| Error::Other("aligner lock poisoned by an earlier panic".into()))?;
        if guard.is_none() {
            *guard = Some(std::sync::Arc::new(Aligner::from_manifest_dir(
                &self.manifest_dir,
                ALIGN_MODEL,
                ffai_core::best_device(),
            )?));
        }
        Ok(guard.as_ref().expect("initialized above").clone())
    }

    /// The speaker model, loaded on first use. See [`Self::aligner`] for why
    /// this hands back an `Arc` rather than holding the lock across the work.
    fn diarizer(&self) -> Result<std::sync::Arc<Diarizer>> {
        let mut guard = self
            .diarizer
            .lock()
            .map_err(|_| Error::Other("diarizer lock poisoned by an earlier panic".into()))?;
        if guard.is_none() {
            *guard = Some(std::sync::Arc::new(Diarizer::from_manifest_dir(
                &self.manifest_dir,
                DIARIZE_MODEL,
                ffai_core::best_device(),
            )?));
        }
        Ok(guard.as_ref().expect("initialized above").clone())
    }

    /// Forget every remembered speaker.
    ///
    /// Call this when a new recording starts. A registry that carries
    /// identities from the last meeting into this one will match the wrong
    /// people together, and unlike in-call clustering it never reconsiders.
    pub fn reset_speakers(&self) {
        if let Ok(mut guard) = self.speakers.lock() {
            *guard = None;
        }
    }

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
            aligner: Mutex::new(None),
            diarizer: Mutex::new(None),
            speakers: Mutex::new(None),
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
            *guard = Some(State { whisper, front_end, warmed: false });
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

        // Window selection is the whole of the VAD feature. With it off this
        // is the fixed 30 s grid, byte for byte what it has always been; with
        // it on the windows are speech-shaped, and stretches of silence longer
        // than the packing width are never handed to the encoder at all.
        //
        // Windows carry their absolute start sample either way, so segment
        // timestamps land in the original audio's time base with no remapping.
        // FFAI_VAD overrides the flag in BOTH directions for callers that
        // cannot set it — the bench harness builds `AsrOptions` itself, and a
        // feature that is on by default needs an off switch to be measured
        // against, not just an on switch. `FFAI_VAD=0` restores the
        // fixed-grid behaviour; anything else forces segmentation. Same A/B
        // override pattern as FFAI_ALLOW_NONSPEECH; not a user-facing switch.
        let vad_on = match std::env::var("FFAI_VAD").ok().as_deref() {
            Some("0" | "off" | "false") => false,
            Some(_) => true,
            None => opts.vad,
        };
        // Detected once and kept: diarization needs the same speech regions
        // the windows were cut from. Recomputing them there would let the two
        // drift apart under a threshold change and attribute speaker turns to
        // audio the transcript never saw.
        let regions = if vad_on || opts.diarize {
            vad::detect(&mono.samples, opts.vad_threshold)
        } else {
            Vec::new()
        };
        let windows: Vec<(usize, &[f32])> = if vad_on {
            vad::pack(&regions, opts.vad_chunk_secs as f64)
                .into_iter()
                .map(|w| {
                    let sr = mel::SAMPLE_RATE as f64;
                    let start = ((w.start * sr) as usize).min(mono.samples.len());
                    let end = ((w.end * sr).ceil() as usize).clamp(start, mono.samples.len());
                    (start, &mono.samples[start..end])
                })
                .collect()
        } else {
            decoder::windows(&mono.samples).collect()
        };
        // No speech found is a valid answer, and the common one on a silent
        // chunk. It returns an empty transcript without loading the encoder.
        let did_work = !windows.is_empty();

        let mut segments = Vec::new();
        for (start_sample, window) in windows {
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

        // Hand back the load arena, once, after the FIRST complete pass.
        //
        // Loading weights churns far more heap than the model keeps — dtype
        // conversions, quantization scratch, the safetensors mapping — and the
        // first inference adds its own on top. Freed is not returned: those
        // pages stay counted against the process for its whole life.
        //
        // Measured on tiny.en (examples/mem_claims.rs): the process holds
        // **347 MiB** without this and **102 MiB** with it, while the passes
        // that follow fault back only what they actually touch. Against
        // whisper.cpp's 194 MiB steady that is the difference between using
        // 2.2x the reference's memory and using half of it.
        //
        // TIMING IS LOAD-BEARING and cost a wrong conclusion once: trimming
        // after all four clips instead of after the first leaves 347 MiB,
        // because by then the transients have been touched again. Once, here,
        // right after the first pass — never in the decode path, where
        // trimming what you are about to touch again only buys page faults.
        // Gated on `did_work`: with VAD on, a silent first call decodes
        // nothing, and trimming then would hand back an arena the first real
        // inference is about to fault straight back in — the same
        // timing-is-load-bearing mistake described above, arrived at from the
        // other direction.
        let first_pass = !state.warmed && did_work;
        state.warmed |= did_work;
        let language = opts
            .language
            .clone()
            .or_else(|| state.whisper.is_english_only().then(|| "en".to_string()));
        drop(guard);
        if first_pass {
            ffai_core::release_load_arena();
        }

        // `None` when not asked for — an absent result, distinct from an
        // empty one. Alignment runs after decoding because it needs the text:
        // it is aligning a known transcript, not recognising one.
        let words = if opts.word_timestamps {
            let aligner = self.aligner()?;
            Some(aligner.align_segments(&mono.samples, &segments)?)
        } else {
            None
        };

        // Diarization runs last: it needs the speech regions (above) and is
        // independent of the text, so a failure here cannot corrupt a
        // transcript that already succeeded.
        let speakers = if opts.diarize {
            let diarizer = self.diarizer()?;
            let turns = if opts.persist_speakers {
                let mut guard = self.speakers.lock().map_err(|_| {
                    Error::Other("speaker registry lock poisoned by an earlier panic".into())
                })?;
                let registry = guard.get_or_insert_with(|| {
                    SpeakerRegistry::new(opts.diarize_threshold, opts.max_speakers)
                });
                diarizer.diarize_streaming(
                    &mono.samples,
                    &regions,
                    opts.diarize_threshold,
                    opts.max_speakers,
                    registry,
                )
            } else {
                diarizer.diarize(
                    &mono.samples,
                    &regions,
                    opts.diarize_threshold,
                    opts.max_speakers,
                )
            };
            Some(diarize::labelled_turns(&turns))
        } else {
            None
        };

        Ok(Transcript { language, segments, words, speakers })
    }
}
