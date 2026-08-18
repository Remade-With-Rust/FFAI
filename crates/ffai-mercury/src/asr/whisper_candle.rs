//! `whisper-candle` — Mercury's reference ASR engine.
//!
//! Composes the independent stages: [`mel`] → [`model`] → [`decoder`] →
//! [`ffai_core::types::Transcript`]. The engine itself holds no algorithm;
//! it wires stages and manages the loaded model, which is exactly the
//! division that makes each stage separately testable.

//! Cast policy (gate H-15): `cast_possible_truncation`, `cast_sign_loss` and
//! `cast_possible_wrap` are allowed in this module. Every value converted here
//! is a MODEL-INTERNAL dimension, index or accumulator - bounded by weights the
//! loader has already validated - not a number read from caller input. The lint
//! stays DENIED in the untrusted-surface modules (`mel`, `fbank`, `onnx`,
//! `normalize`, `lexicon`, `chunk`, `phonemize`, `phoneme_ids`), which is where
//! this audit's arithmetic defects were actually found.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use std::path::PathBuf;
use std::sync::Mutex;

use ffai_core::engine::{AsrEngine, AsrOptions, EngineInfo, EngineStatus, Task};
use ffai_core::error::{Error, Result};
use ffai_core::types::{AudioBuffer, Transcript};

use super::aligner::{Aligner, DEFAULT_MODEL as ALIGN_MODEL};
use super::decoder::{self, DecodeConfig};
use super::diarize;
use super::diarizer::{DEFAULT_MODEL as DIARIZE_MODEL, Diarizer};
use super::mel::{self, MelSpectrogram};
use super::model::LoadedWhisper;
use super::registry::SpeakerRegistry;
use super::text_decoder::Precision;
use super::vad;

/// `OpenAI` Whisper running on candle.
///
/// The model is loaded lazily on first use and reused across calls, so
/// per-clip timing measures inference rather than weight loading (see
/// docs/benchmarking.md on warm vs end-to-end throughput).
pub struct WhisperCandle {
    /// `None` = the manifests compiled into the crate; `Some(dir)` = read
    /// them from there instead.
    manifest_dir: Option<PathBuf>,
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
    /// Embeddings already computed this session, in absolute stream time.
    ///
    /// Beside the registry because they are the same lifetime: both are
    /// per-session streaming identity, and `reset_speakers` must clear both
    /// or a new recording would cluster against the previous one's audio.
    stream: Mutex<Option<super::diarize::StreamState>>,
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
            *guard = Some(std::sync::Arc::new(Aligner::from_manifest_source(
                self.manifest_dir.as_deref(),
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
            *guard = Some(std::sync::Arc::new(Diarizer::from_manifest_source(
                self.manifest_dir.as_deref(),
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
        // The window history goes too: it is the audio those identities were
        // learned from, and keeping it would let a new recording cluster
        // against the previous one's speech.
        if let Ok(mut guard) = self.stream.lock() {
            *guard = None;
        }
    }

    /// Default: `whisper-tiny-en` — the M1 bring-up target, matching the M0
    /// baseline configuration.
    ///
    /// Manifests come from the copies compiled into the crate
    /// ([`crate::manifests`]), so this works from any working directory; a
    /// library that resolved them from a relative `models/` path would work
    /// only inside this repo.
    #[must_use]
    pub fn new() -> Self {
        Self::model("whisper-tiny-en", Precision::F32)
    }

    /// A named size/precision, using the compiled-in manifests.
    pub fn model(model_name: impl Into<String>, precision: Precision) -> Self {
        Self {
            manifest_dir: None,
            model_name: model_name.into(),
            precision,
            state: Mutex::new(None),
            aligner: Mutex::new(None),
            diarizer: Mutex::new(None),
            speakers: Mutex::new(None),
            stream: Mutex::new(None),
        }
    }

    /// A named size/precision whose manifests are read from `manifest_dir`
    /// instead — for a caller shipping their own (a different checkpoint, a
    /// pinned checksum, a relocated cache).
    pub fn with_model(
        manifest_dir: impl Into<PathBuf>,
        model_name: impl Into<String>,
        precision: Precision,
    ) -> Self {
        Self {
            manifest_dir: Some(manifest_dir.into()),
            model_name: model_name.into(),
            precision,
            state: Mutex::new(None),
            aligner: Mutex::new(None),
            diarizer: Mutex::new(None),
            speakers: Mutex::new(None),
            stream: Mutex::new(None),
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

    // The speaker-table guard is deliberately held across clustering: the table
    // must not change under it, which is the entire point of persisting speakers
    // between calls. Tightening the scope would reintroduce the race.
    #[allow(clippy::significant_drop_tightening)]
    fn transcribe(&self, audio: &AudioBuffer, opts: &AsrOptions) -> Result<Transcript> {
        let mut guard = self.state.lock().map_err(|_| {
            Error::Other("whisper-candle state lock poisoned by an earlier panic".into())
        })?;
        if guard.is_none() {
            let whisper = LoadedWhisper::from_manifest_source(
                self.manifest_dir.as_deref(),
                &self.model_name,
                ffai_core::best_device(),
                self.precision,
            )?;
            let front_end = MelSpectrogram::new(whisper.n_mels());
            *guard = Some(State {
                whisper,
                front_end,
                warmed: false,
            });
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
            (Some(code), false) => {
                Some(state.whisper.tokenizer.language(code).ok_or_else(|| {
                    Error::Other(format!("model has no language token for `{code}`"))
                })?)
            }
            // An .en model has no language slot; forcing English is a no-op
            // rather than an error.
            (Some(code), true) if !code.eq_ignore_ascii_case("en") => {
                return Err(Error::Other(format!(
                    "model `{}` is English-only but language `{code}` was requested",
                    state.whisper.name
                )));
            }
            _ => None,
        };
        let cfg = DecodeConfig {
            language,
            translate: opts.translate,
            ..Default::default()
        };

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
        if std::env::var_os("FFAI_DEBUG_TOKENS").is_some() && vad_on {
            for w in vad::pack(&regions, f64::from(opts.vad_chunk_secs)) {
                eprintln!(
                    "[vad window] {:.2}-{:.2}s ({:.1}s)",
                    w.start,
                    w.end,
                    w.end - w.start
                );
            }
        }
        // No speech found is a valid answer, and the common one on a silent
        // chunk. It returns an empty transcript without loading the encoder.
        let sr = mel::SAMPLE_RATE as f64;
        let did_work = if vad_on {
            !regions.is_empty()
        } else {
            !mono.samples.is_empty()
        };

        // THE SEEK LOOP. A window's decode can end before the window does —
        // the model emits <|endoftext|> early, most often at a discontinuity.
        // The old fixed-window design accepted that as final and silently
        // dropped the tail: on the long-form corpus a whole 5.7 s utterance
        // vanished that way, and the fixed 30 s grid had the same failure at
        // different positions. openai-whisper and whisper.cpp both RESUME
        // from the last timestamped segment; so does this.
        //
        // Each iteration decodes one window of up to `vad_chunk_secs` (≤30 s)
        // starting at `point` — or, with VAD, at the first speech at or after
        // `point`, which is what preserves the skip-silence win. If the
        // decode stops more than SEEK_TAIL_SECS short of the window's end,
        // the next window starts where it stopped rather than after the
        // window. The audio handed to the model is still always padded to
        // the full 30 s context (mission plan §6.4: shortening the encoder
        // input was measured at 268 % WER and pruned).
        const SEEK_TAIL_SECS: f64 = 1.0;
        let mut segments = Vec::new();
        // Previous-window text, carried into the next window's prompt as
        // `<|startofprev|>` context (openai-whisper's
        // `condition_on_previous_text`, on by default in both references).
        // Root cause it addresses, reproduced on longform-02 at 82.3 s: a
        // window that resumes mid-sentence decodes the sentence tail with no
        // memory of its head, overshoots the tail segment's end-timestamp by
        // seconds, and the short utterance under the overshoot is swallowed —
        // and the overshoot also shrinks the coverage-repair hole below its
        // 1 s threshold, so the repair pass cannot catch what it was built
        // for. Reset after any high-temperature rung, as openai-whisper does,
        // so a barely-accepted transcript does not poison the next window.
        // Read the environment directly, once per call, NOT through a cached
        // knob: these are per-transcribe decisions (one read per file, never
        // per token), and `ab_clips` flips its B arm by setting the variable
        // between two `transcribe` calls in one process — a `knobs::Flag`
        // caches its first read and would silently give that harness two
        // identical arms.
        // `FFAI_PREV_CONTEXT=on` — opt-in, and gated OFF by measurement:
        // on the (spliced) long-form corpus it took in-context WER 13.16 %
        // -> 36.90 % (31 worse / 4 better, z = +4.56). See
        // docs/whys/adaptive-context.md before defaulting this on.
        let prev_context_on = std::env::var("FFAI_PREV_CONTEXT").as_deref() == Ok("on");
        // Default ON — gated 2026-07-29 (docs/whys/adaptive-context.md):
        // per-clip WER/CER neutral on both holdouts (|z| <= 0.93, 171-180 of
        // 200 clips byte-unchanged), byte-identical on long-form, clean-clip
        // throughput 1.47x. `FFAI_ADAPTIVE_CTX=off` reproduces the old path.
        let adaptive_ctx_on = std::env::var("FFAI_ADAPTIVE_CTX").as_deref() != Ok("off");
        let mut prev_text: Vec<u32> = Vec::new();
        let total_secs = mono.samples.len() as f64 / sr;
        let window_secs_max = f64::from(opts.vad_chunk_secs).clamp(1.0, mel::CHUNK_SECONDS as f64);
        let mut point = 0.0f64;
        // The 1e-6 epsilon is precisely the guard against float accumulation
        // that makes this loop terminate cleanly at the end of the audio; a
        // bare `<` is what would be wrong here.
        #[allow(clippy::while_float)]
        while point < total_secs - 1e-6 {
            let window_start = if vad_on {
                // First speech at or after `point`; silence between is never
                // encoded, which is the VAD speed win.
                match regions.iter().find(|r| r.end > point + 1e-6) {
                    Some(r) => r.start.max(point),
                    None => break,
                }
            } else {
                point
            };
            let start_sample = ((window_start * sr) as usize).min(mono.samples.len());
            let end_sample =
                ((window_start + window_secs_max) * sr).min(mono.samples.len() as f64) as usize;
            if end_sample <= start_sample {
                break;
            }
            let window = &mono.samples[start_sample..end_sample];
            let offset_secs = start_sample as f64 / sr;
            let window_secs = window.len() as f64 / sr;
            let mut wcfg = cfg.clone();
            if prev_context_on {
                wcfg.prev_text_tokens = prev_text.clone();
            }

            // ADAPTIVE CONTEXT. The encoder is O(n) and, on ordinary
            // utterances, ~70 % of its work is the silence that pads every
            // window to 30 s (mission plan §6.4). Encode a bucketed context
            // sized to the audio actually present, with the timestamp
            // grammar masked at the encoded extent; if the greedy decode
            // fails the confidence/repetition guards, fall through to the
            // full-context path below — which is byte-for-byte today's
            // behaviour, so the worst case is a small extra cost, not a
            // different transcript. The §6.4 prune (268 % WER) truncated
            // with no margin, no timestamp mask, and no guards to escalate
            // through; this path has all three.
            let mut outcome = None;
            if adaptive_ctx_on && let Some(ctx_secs) = adaptive_ctx_secs(window_secs) {
                let mut scfg = wcfg.clone();
                scfg.max_timestamp_secs = Some(ctx_secs);
                // The short-context arm must be HARDER to satisfy than
                // the temperature ladder: -1.0 asks "is this transcript
                // salvageable", the right bar for retrying at a higher
                // temperature, not for trusting a reduced context. On
                // test-other (noisy) the model at 10 s context degrades
                // exactly on the windows it scores worst — measured
                // per-clip: CER 38 worsened / 23 improved at the -1.0
                // bar. Requiring the model's own confidence to clear a
                // stricter bar routes hard audio back to the trained
                // 30 s regime; clean speech scores -0.24..-0.48 and
                // keeps the speed.
                scfg.logprob_threshold = scfg.logprob_threshold.max(
                    std::env::var("FFAI_CTX_LOGPROB")
                        .ok()
                        .and_then(|v| v.parse::<f32>().ok())
                        .unwrap_or(-0.5),
                );
                // A hair under the bar, not comfortably under it: with a
                // 0.15 margin the windows the -0.5 bar rejects mostly sat
                // in [-0.65, -0.5] and decoded to completion before being
                // rejected — the doomed decode the abort exists to kill.
                // 0.05 keeps a token's worth of wobble room; a decode
                // that dips below THAT and recovers merely escalates to
                // the full-context path, which is a speed cost on that
                // window, never a quality one.
                scfg.early_abort_logprob = Some(scfg.logprob_threshold - 0.05);
                let chunk = super::profile::timed(&super::profile::profile().mel, || {
                    state
                        .front_end
                        .compute(&mel::pad_or_trim_to(window, (ctx_secs * sr) as usize))
                });
                outcome = decoder::decode_window_strict(
                    &mut state.whisper,
                    &chunk,
                    offset_secs,
                    &scfg,
                    window_secs,
                )?;
                // GEOMETRY GUARDS. The confidence/repetition guards catch
                // bad TEXT; a reduced context also produces bad
                // TIMESTAMPS on audio the model finds hard, and those
                // poison the pipeline around the decode rather than the
                // transcript itself. Reproduced on test-other
                // 7975-280063-0002: the short-context decode transcribed
                // the whole window correctly but closed its segment at
                // 1.2 s of 7.0 s of speech — the seek loop trusted that,
                // resumed at 1.2 s, and re-decoded the same speech into a
                // duplicate (WER > 1). Escalate to the full context when
                // the decode's geometry is untrustworthy: it stops short
                // of the VAD speech extent, or a segment claims a span
                // its own word count cannot cover.
                if let Some(o) = &outcome
                    && !o.segments.is_empty()
                {
                    let window_end = offset_secs + window_secs;
                    let speech_end = if vad_on {
                        regions
                            .iter()
                            .filter(|r| r.start < window_end)
                            .map(|r| r.end.min(window_end))
                            .fold(offset_secs, f64::max)
                    } else {
                        window_end
                    };
                    let decoded_to = o.segments.iter().map(|s| s.end).fold(offset_secs, f64::max);
                    let unfinished = speech_end - decoded_to > SEEK_TAIL_SECS;
                    let stretched = o.segments.iter().any(|s| {
                        let words = s.value.split_whitespace().count() as f64;
                        s.end - s.start > words + 2.0 // 1 word/s floor + slack
                    });
                    if unfinished || stretched {
                        outcome = None;
                    }
                }
                if outcome.is_none() {
                    ADAPTIVE_ESCALATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if std::env::var_os("FFAI_DEBUG_TOKENS").is_some() {
                        eprintln!(
                            "[ctx] window at {offset_secs:.2}s rejected at {ctx_secs:.0}s \
                                 context — escalating to 30s"
                        );
                    }
                }
            }
            let outcome = if let Some(o) = outcome {
                o
            } else {
                // Pad in the sample domain, before the spectrogram — see
                // mel::pad_or_trim for why the distinction matters.
                let chunk = super::profile::timed(&super::profile::profile().mel, || {
                    state.front_end.compute(&mel::pad_or_trim(window))
                });
                decoder::decode_window(&mut state.whisper, &chunk, offset_secs, &wcfg, window_secs)?
            };
            if prev_context_on {
                if outcome.temperature > 0.5 {
                    prev_text.clear();
                } else {
                    // Accumulate, as openai-whisper does; the prompt builder
                    // keeps only the last 223 tokens, so bound the buffer at
                    // that horizon rather than growing with the file.
                    prev_text.extend_from_slice(&outcome.text_tokens);
                    let excess = prev_text.len().saturating_sub(223);
                    if excess > 0 {
                        prev_text.drain(..excess);
                    }
                }
            }
            let mut new_segments = outcome.segments;

            // Where did the decode actually get to, in absolute seconds?
            let decoded_to = new_segments
                .iter()
                .map(|s| s.end)
                .fold(offset_secs, f64::max);
            let window_end_secs = end_sample as f64 / sr;
            point = if window_end_secs - decoded_to > SEEK_TAIL_SECS && decoded_to > offset_secs {
                if std::env::var_os("FFAI_DEBUG_TOKENS").is_some() {
                    eprintln!(
                        "[seek] window {offset_secs:.2}-{window_end_secs:.2}s decoded to                          {decoded_to:.2}s — resuming there"
                    );
                }
                decoded_to
            } else {
                // Reached the end (or made no progress at all, where resuming
                // would loop forever): advance past the whole window.
                window_end_secs
            };
            segments.append(&mut new_segments);
        }

        // COVERAGE REPAIR. The seek loop guarantees no window TAIL is lost,
        // but the model can still skip content MID-window: the timestamp
        // grammar lets it jump over a discontinuity with one legal <t1><t2>
        // pair, and the decode then runs to the window's end so seek never
        // fires. On the long-form corpus that swallowed whole utterances.
        //
        // VAD already knows where the speech is, which makes it a checksum on
        // the decode: any detected speech region that no emitted segment even
        // OVERLAPS gets one repair window of its own. Segments past the
        // region's end are already covered by the main pass and are dropped
        // rather than duplicated. One pass, no recursion — a region the
        // repair decode ALSO cannot transcribe is left uncovered rather than
        // retried forever.
        if vad_on {
            let mut repaired: Vec<ffai_core::types::TimedSegment<String>> = Vec::new();
            // Uncovered sub-spans, not per-region coverage fractions: VAD can
            // merge neighbouring utterances into one region (a 0.5 s gap sits
            // right at the bridge boundary once hysteresis tails are counted),
            // and a skip inside a merged region hides behind the rest of the
            // region's perfectly good coverage. Measured: the per-region
            // fraction test fired zero times on a corpus with two known skips.
            // 1.0 s, and a 0.5 s floor was TRIED AND REVERTED (2026-07-30):
            // the last known long-form drop ("YOU DO ME A GREAT HONOUR",
            // longform-02) leaves NO hole at any floor — the following
            // segment abuts the overshooting one exactly (84.64/84.64), so
            // the 3.3 s utterance is absorbed between two contiguous,
            // individually-plausible spans. That class is structurally
            // invisible to span-based coverage; catching it needs word-level
            // coverage (the aligner). Meanwhile 0.5 s false-fired 2-3 times
            // across the short-clip corpora where 1.0 s fires zero. Knob
            // kept for re-measurement.
            let min_hole_secs: f64 = std::env::var("FFAI_REPAIR_MIN_HOLE")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| *v > 0.0)
                .unwrap_or(1.0);
            const MAX_REPAIRS: usize = 8;
            let mut holes: Vec<(f64, f64)> = Vec::new();
            for region in &regions {
                if region.end - region.start < min_hole_secs {
                    continue;
                }
                // Segments overlapping this region, in time order — with each
                // segment's claimed span CLIPPED to what its word count can
                // plausibly cover. Segment end-timestamps lie in exactly the
                // way that defeats this check: both drops on the long-form
                // corpus hid behind a stretched span — "Their masters said
                // Mrs. Neverband." claiming 12.4 s (0.5 words/s) covered a
                // swallowed 9.9 s utterance, and a +2.3 s overshoot shrank
                // the other hole to 0.97 s, under the 1 s floor. Nobody
                // speaks below ~1 word/s for seconds on end, so a span
                // beyond `words / 1.0 + 1 s slack` is treated as unclaimed
                // for COVERAGE purposes only; the emitted segment itself is
                // untouched.
                const MIN_WORDS_PER_SEC: f64 = 1.0;
                const CLAIM_SLACK_SECS: f64 = 1.0;
                let mut cover: Vec<(f64, f64)> = segments
                    .iter()
                    .filter(|s| s.start < region.end && s.end > region.start)
                    .map(|s| {
                        let words = s.value.split_whitespace().count() as f64;
                        let plausible_end = s.start + words / MIN_WORDS_PER_SEC + CLAIM_SLACK_SECS;
                        (
                            s.start.max(region.start),
                            s.end.min(plausible_end).min(region.end),
                        )
                    })
                    .filter(|(a, b)| b > a)
                    .collect();
                cover.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut cursor = region.start;
                for (cs, ce) in cover {
                    if cs - cursor > min_hole_secs {
                        holes.push((cursor, cs));
                    }
                    cursor = cursor.max(ce);
                }
                if region.end - cursor > min_hole_secs {
                    holes.push((cursor, region.end));
                }
            }
            for (hole_start, hole_end) in holes.into_iter().take(MAX_REPAIRS) {
                // Start the repair window a little BEFORE the hole: when the
                // hole was exposed by the word-rate clip, its start is where
                // the previous segment's plausible claim runs out — which is
                // fuzzy by construction, and usually a word or two into the
                // swallowed utterance. Measured on longform-04: without the
                // backoff the repair recovered "the Nautilus, then conducts…"
                // and left the utterance's first four words in the hole's
                // shadow. Duplicates stay bounded because kept segments must
                // OVERLAP the hole itself, not merely precede its end.
                const REPAIR_BACKOFF_SECS: f64 = 2.0;
                let win_start = (hole_start - REPAIR_BACKOFF_SECS).max(0.0);
                let start_sample = ((win_start * sr) as usize).min(mono.samples.len());
                let end_sample =
                    ((win_start + window_secs_max) * sr).min(mono.samples.len() as f64) as usize;
                if end_sample <= start_sample {
                    continue;
                }
                let window = &mono.samples[start_sample..end_sample];
                let offset_secs = start_sample as f64 / sr;
                let chunk = super::profile::timed(&super::profile::profile().mel, || {
                    state.front_end.compute(&mel::pad_or_trim(window))
                });
                // Repair decodes stay at the full 30 s context and without
                // previous-text conditioning: a hole is by definition audio
                // the main pass got wrong, so the repair gets the model's
                // most-trained configuration rather than the fastest one.
                let new_segments = decoder::decode_window(
                    &mut state.whisper,
                    &chunk,
                    offset_secs,
                    &cfg,
                    window.len() as f64 / sr,
                )?
                .segments;
                if std::env::var_os("FFAI_DEBUG_TOKENS").is_some() {
                    eprintln!(
                        "[repair] hole {hole_start:.2}-{hole_end:.2}s — re-decoded, {} kept",
                        new_segments
                            .iter()
                            .filter(|s| s.start < hole_end && s.end > hole_start)
                            .count()
                    );
                }
                // Keep only segments that overlap the hole: anything past it
                // duplicates the main pass, and anything wholly inside the
                // 2 s backoff region re-transcribes speech the main pass
                // already has — the false-fire injection path a lower hole
                // floor would otherwise open.
                repaired.extend(
                    new_segments
                        .into_iter()
                        .filter(|s| s.start < hole_end && s.end > hole_start),
                );
            }
            if !repaired.is_empty() {
                segments.append(&mut repaired);
                segments.sort_by(|a, b| a.start.total_cmp(&b.start));
            }
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
                // Incremental only when the caller actually says where it is
                // in the stream. Without an offset every buffer claims to
                // start at 0, so "already processed" would be meaningless and
                // the state would suppress windows it had never seen.
                let mut sguard = self.stream.lock().map_err(|_| {
                    Error::Other("stream state lock poisoned by an earlier panic".into())
                })?;
                let incr = std::env::var("FFAI_DIARIZE_INCREMENTAL").as_deref() != Ok("off");
                let state = (incr && opts.stream_offset_secs > 0.0)
                    .then(|| sguard.get_or_insert_with(super::diarize::StreamState::new));
                diarizer.diarize_streaming(
                    &mono.samples,
                    &regions,
                    opts.diarize_threshold,
                    opts.max_speakers,
                    registry,
                    opts.stream_offset_secs,
                    state,
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

        Ok(Transcript {
            language,
            segments,
            words,
            speakers,
        })
    }
}

/// How many adaptive-context windows this process has escalated to the full
/// 30 s context. Instrumented like [`decoder::no_speech_drops`]: the number
/// that says whether the short-context arm is earning its keep on a given
/// content class, without a debug env var.
pub static ADAPTIVE_ESCALATIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// The encoder context, in seconds, for a window holding `window_secs` of
/// real audio — or `None` when only the full 30 s context will do.
///
/// One second of margin so speech never sits flush against the encoded
/// edge, rounded up to a 5 s bucket so the encoder sees a handful of
/// shapes rather than a new one per clip, floored at 10 s because very
/// short contexts sit furthest from the model's training distribution —
/// the §6.4 prune truncated to the audio's exact length and destabilized
/// the decoder. Within a bucket the remainder is real zero-padding, mel'd
/// exactly as the 30 s path pads.
fn adaptive_ctx_secs(window_secs: f64) -> Option<f64> {
    let bucket = (((window_secs + 1.0) / 5.0).ceil() * 5.0).max(10.0);
    // Cap the attempt at the 15 s bucket. The prize scales with (30 −
    // bucket) while a rejected attempt's cost scales with the bucket, so a
    // 20-25 s attempt risks the most encode to save the least — negative
    // expected value at any realistic escalation rate.
    (bucket <= 15.0).then_some(bucket)
}

#[cfg(test)]
mod adaptive_ctx_tests {
    use super::adaptive_ctx_secs;

    #[test]
    fn buckets_are_floored_stepped_and_capped() {
        // Short clips land on the 10 s floor.
        assert_eq!(adaptive_ctx_secs(2.0), Some(10.0));
        assert_eq!(adaptive_ctx_secs(6.5), Some(10.0));
        // 9.0 + 1.0 margin = 10.0 exactly — still the floor bucket.
        assert_eq!(adaptive_ctx_secs(9.0), Some(10.0));
        // Above the floor, 5 s steps with the 1 s margin applied first.
        assert_eq!(adaptive_ctx_secs(12.3), Some(15.0));
        // Past the 15 s bucket the attempt risks more encode than it can
        // save — the normal 30 s path runs directly.
        assert_eq!(adaptive_ctx_secs(19.0), None);
        assert_eq!(adaptive_ctx_secs(23.9), None);
        assert_eq!(adaptive_ctx_secs(29.5), None);
        assert_eq!(adaptive_ctx_secs(30.0), None);
    }
}

#[cfg(test)]
mod reset_tests {
    use super::*;

    /// `reset_speakers` must clear the WINDOW HISTORY as well as the
    /// registry.
    ///
    /// The two are one lifetime: the history is the audio those identities
    /// were learned from. Leaving it behind would let a new recording cluster
    /// against the previous one's speech, and — worse — a new session that
    /// starts at t=0 would sit before the old `processed_to`, which is
    /// exactly the rewind case that used to yield no labels at all.
    ///
    /// No weights are touched: the engine loads lazily, so this exercises the
    /// bookkeeping without fetching an 83 MB model.
    #[test]
    fn reset_speakers_clears_the_stream_history_too() {
        let engine = WhisperCandle::new();
        {
            let mut g = engine.stream.lock().expect("fresh lock");
            let st = g.get_or_insert_with(super::super::diarize::StreamState::new);
            st.extend(vec![(0.0, 1.5, vec![1.0f32; 4])], 10.0);
            assert_eq!(st.len(), 1, "seeded");
        }
        {
            let mut g = engine.speakers.lock().expect("fresh lock");
            *g = Some(SpeakerRegistry::new(0.8, None));
        }

        engine.reset_speakers();

        assert!(
            engine.stream.lock().expect("lock").is_none(),
            "window history must not survive a reset — a new recording would \
             cluster against the old one's audio"
        );
        assert!(
            engine.speakers.lock().expect("lock").is_none(),
            "registry cleared"
        );
    }
}
