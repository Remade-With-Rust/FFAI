//! The Whisper decode loop: encoder output → token ids → timed segments.
//!
//! Greedy decoding with the timestamp grammar (M1). Beam search and
//! temperature fallback land in M2; the seams for both are marked below.

use candle_transformers::models::whisper as m;
use ffai_core::candle::{IndexOp, Tensor};
use ffai_core::error::{Error, Result};
use ffai_core::types::TimedSegment;

use super::mel::{self, MelChunk};
use super::model::LoadedWhisper;

/// Whisper's `max_initial_timestamp` of 1.0 s, in 20 ms timestamp steps.
const MAX_INITIAL_TIMESTAMP_INDEX: u32 = 50;

/// How the decode loop should behave for one call.
#[derive(Debug, Clone)]
pub struct DecodeConfig {
    /// Forced language token id (multilingual models only).
    pub language: Option<u32>,
    pub translate: bool,
    /// Emit timestamp tokens and split the output into timed segments.
    pub timestamps: bool,
    /// Hard cap on generated tokens per 30 s window.
    pub max_tokens: usize,
    /// Temperature ladder for fallback. The first entry is the greedy pass;
    /// later entries are sampled retries. Empty = greedy only.
    pub temperatures: Vec<f32>,
    /// Retry when the mean per-token log-probability falls below this.
    pub logprob_threshold: f32,
    /// Drop a window entirely when P(`<|nospeech|>`) exceeds this *and* the
    /// transcript is also low-confidence. Whisper hallucinates fluent text on
    /// silence ("you", "Thank you.") — the model tells you it is silence via
    /// this token, and without reading it there is no way to know. Matches
    /// `no_speech_threshold` in openai-whisper and `-nth` in whisper.cpp.
    /// Set above 1.0 to disable.
    pub no_speech_threshold: f32,
    /// Apply Whisper's `SuppressTokens` list — the ~90 symbol tokens that
    /// annotate audio rather than transcribe it (`♪`, `((`, `[`).
    ///
    /// **Off by default, matching whisper.cpp** (`--suppress-nst` is opt-in
    /// there) and diverging from openai-whisper, which applies the list. With
    /// it on, a cough cannot be written as `(coughs)` — the model is denied
    /// the tokens and falls through to the nearest phonetic rendering
    /// (`Hah!`), which reads as a transcription error rather than an
    /// annotation.
    ///
    /// This default is a **priced decision, not a free one**. LibriSpeech
    /// test-clean, 134 clips, tiny.en:
    ///
    ///   suppressed (openai-whisper)   WER 7.77  CER 3.25
    ///   allowed    (whisper.cpp, us)  WER 7.99  CER 3.27
    ///
    /// 0.22 pp — 2.8 % relative — because on clean speech the model sometimes
    /// spends an annotation where words belong. We take that cost to match the
    /// reference we benchmark against, and carry recovering it as open quality
    /// work rather than pretending it is not there. Set this to `true` for
    /// transcripts that must contain words only; it is the better setting for
    /// WER and the worse one for reading a live transcript.
    pub suppress_non_speech: bool,
    /// Conditional annotations: allow the `SuppressTokens` list **only** when
    /// `P(<|nospeech|>)` at the first decode position reaches this value.
    ///
    /// The all-or-nothing choice above is a bad trade in both directions.
    /// Suppressing always turns a cough into `Hah!` — a transcription error
    /// wearing the costume of a word. Allowing always lets the model spend an
    /// annotation where words belong on clean speech, which measured at
    /// 0.22 pp WER on test-clean.
    ///
    /// The model already tells us which case it is, and we already compute
    /// the number for the no-speech gate and then throw it away. Measured on
    /// tiny.en:
    ///
    ///   clean speech      0.004 - 0.032
    ///   cough / laughter  0.288 - 0.454
    ///   digital silence   0.949   (dropped outright by `no_speech_threshold`)
    ///
    /// A threshold of 0.10 sits ~3x above the loudest speech reading and ~3x
    /// below the quietest non-speech event. `None` keeps the binary
    /// behaviour; `Some(t)` overrides [`Self::suppress_non_speech`].
    pub annotation_threshold: Option<f32>,
    /// Retry when the transcript looks like a repetition loop (see
    /// [`repetition_ratio`]).
    pub repetition_threshold: f32,
    /// Text tokens from the previous window(s), fed to the decoder as
    /// `<|startofprev|>` context — openai-whisper's
    /// `condition_on_previous_text`, which both references ship by default.
    /// Empty = no conditioning. Capped to the last 223 tokens at prompt
    /// build (half the decoder context minus the control token), matching
    /// openai-whisper.
    pub prev_text_tokens: Vec<u32>,
    /// Suppress timestamp tokens beyond this many seconds. Set when the
    /// encoder ran on a context shorter than the full 30 s window, so the
    /// decoder cannot place a timestamp past the audio that was actually
    /// encoded. `None` = the grammar's usual 30 s range.
    pub max_timestamp_secs: Option<f64>,
    /// Stop generating as soon as the running mean log-probability falls to
    /// this value (checked from the 14th token). Only set on the
    /// short-context rung, where a decode headed for rejection is pure
    /// waste — without this, ~60 % of noisy windows paid a full doomed
    /// decode before escalating, and the escalation tax ate the speed win.
    /// Must sit BELOW [`Self::logprob_threshold`]: aborting can only route
    /// the window to the full-context path it was already headed for.
    pub early_abort_logprob: Option<f32>,
    /// Hypotheses to carry in the search. `1` is greedy decoding — the path
    /// every measurement in this repo has used, and the one the references
    /// are pinned to for a like-for-like comparison. `5` is what all three
    /// references run by DEFAULT, so it is the setting to use when the
    /// question is "how good can Mercury be" rather than "is our
    /// implementation equivalent".
    pub beam_size: usize,
    /// Exponent for beam-search length normalization (`logprob / len^alpha`),
    /// applied when picking the winner, never while expanding. `0.0` scores
    /// on raw cumulative log-probability.
    pub length_penalty: f32,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        DecodeConfig {
            language: None,
            translate: false,
            timestamps: true,
            // Whisper's own limit: half the decoder context.
            max_tokens: 224,
            // openai-whisper's default ladder.
            temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            logprob_threshold: -1.0,
            no_speech_threshold: 0.6,
            suppress_non_speech: false,
            // Off until the corpus says otherwise. This is Q1 in the open
            // campaign, and a knob that has not been through the gate does
            // not become a default on the strength of six probe clips —
            // which is precisely the mistake logged four times already.
            // FFAI_ANNOT_THRESHOLD turns it on for the A/B.
            annotation_threshold: std::env::var("FFAI_ANNOT_THRESHOLD")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .filter(|t| (0.0..=1.0).contains(t)),
            repetition_threshold: 2.4,
            prev_text_tokens: Vec::new(),
            max_timestamp_secs: None,
            early_abort_logprob: None,
            // Greedy by default: it is what every ledger line to date was
            // measured with, and flipping the default would silently
            // invalidate the comparison the whole bench rests on. Beam search
            // is opt-in until its own corpus gate says otherwise.
            beam_size: std::env::var("FFAI_BEAM_SIZE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|n| (1..=16).contains(n))
                .unwrap_or(1),
            length_penalty: std::env::var("FFAI_LENGTH_PENALTY")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .filter(|v| (0.0..=2.0).contains(v))
                .unwrap_or(1.0),
        }
    }
}

/// Everything one window's decode produced, beyond the segments themselves.
///
/// `text_tokens` and `temperature` exist for the caller's previous-window
/// conditioning: the next window's prompt wants this window's text, unless
/// this window only passed at a high temperature — openai-whisper resets its
/// context after any rung above 0.5, because conditioning on a sampled,
/// barely-accepted transcript propagates its errors forward.
pub struct WindowOutcome {
    pub segments: Vec<TimedSegment<String>>,
    /// The accepted rung's text tokens (timestamps and controls stripped).
    pub text_tokens: Vec<u32>,
    /// The temperature of the rung that produced the accepted result.
    pub temperature: f32,
}

/// Which rungs of the fallback ladder a decode may use.
#[derive(Clone, Copy, PartialEq)]
enum Ladder {
    /// The full temperature ladder; always produces a result.
    Full,
    /// The greedy rung only; `None` when its guards reject, so the caller
    /// can escalate — used by the adaptive-context path, whose escalation
    /// arm is the full 30 s context rather than a higher temperature.
    GreedyOnly,
}

/// Decode one 30-second window that starts at `offset_secs` in the source.
///
/// Returns segments with absolute timestamps. When `timestamps` is off, a
/// single segment spanning the window is returned.
pub fn decode_window(
    whisper: &mut LoadedWhisper,
    chunk: &MelChunk,
    offset_secs: f64,
    cfg: &DecodeConfig,
    window_secs: f64,
) -> Result<WindowOutcome> {
    Ok(
        decode_window_inner(whisper, chunk, offset_secs, cfg, window_secs, Ladder::Full)?
            .expect("Ladder::Full always yields a result"),
    )
}

/// [`decode_window`], but greedy-rung-only: `None` means the confidence or
/// repetition guard rejected it and the caller should re-decode at the full
/// context. The audio handed in may be padded to less than 30 s **only**
/// through this entry point — set [`DecodeConfig::max_timestamp_secs`] to the
/// encoded extent so the timestamp grammar cannot point past the audio.
pub fn decode_window_strict(
    whisper: &mut LoadedWhisper,
    chunk: &MelChunk,
    offset_secs: f64,
    cfg: &DecodeConfig,
    window_secs: f64,
) -> Result<Option<WindowOutcome>> {
    decode_window_inner(
        whisper,
        chunk,
        offset_secs,
        cfg,
        window_secs,
        Ladder::GreedyOnly,
    )
}

fn decode_window_inner(
    whisper: &mut LoadedWhisper,
    chunk: &MelChunk,
    offset_secs: f64,
    cfg: &DecodeConfig,
    window_secs: f64,
    ladder: Ladder,
) -> Result<Option<WindowOutcome>> {
    let mel_tensor = whisper
        .mel_tensor(chunk)
        .map_err(|e| Error::Model(format!("building mel tensor: {e}")))?;
    let p = super::profile::profile();
    let use_candle_encoder = std::env::var_os("FFAI_CANDLE_ENCODER").is_some();
    let audio_features = super::profile::timed(&p.encoder, || {
        if use_candle_encoder {
            whisper
                .model
                .encoder
                .forward(&mel_tensor, true)
                .map_err(|e| Error::Model(format!("encoder forward: {e}")))
        } else {
            whisper
                .encoder
                .forward(&mel_tensor)
                .map_err(|e| Error::Model(format!("encoder forward: {e}")))
        }
    })?;

    // The encoder runs f32 and the decoder may run f16; convert once per
    // window rather than per token.
    let audio_features = audio_features
        .to_dtype(whisper.decoder_dtype)
        .map_err(tensor_err)?;
    let Some((tokens, temperature)) = decode_with_fallback(whisper, &audio_features, cfg, ladder)?
    else {
        return Ok(None);
    };
    // Set FFAI_DEBUG_TOKENS=1 to see the raw token stream — the first thing
    // worth looking at when a transcript is empty or garbled.
    if std::env::var_os("FFAI_DEBUG_TOKENS").is_some() {
        let tk = &whisper.tokenizer;
        let rendered: Vec<String> = tokens
            .iter()
            .take(24)
            .map(|&t| {
                if tk.is_timestamp(t) {
                    format!("<{:.2}>", tk.timestamp_secs(t))
                } else if t == tk.eot {
                    "<eot>".to_string()
                } else {
                    format!("{:?}", tk.decode(&[t]).unwrap_or_default())
                }
            })
            .collect();
        eprintln!("[tokens n={}] {}", tokens.len(), rendered.join(" "));
    }
    let text_tokens: Vec<u32> = tokens
        .iter()
        .copied()
        .filter(|&t| t < whisper.tokenizer.eot)
        .collect();
    let segments = segments_from_tokens(whisper, &tokens, offset_secs, cfg, window_secs)?;
    Ok(Some(WindowOutcome {
        segments,
        text_tokens,
        temperature,
    }))
}

/// Whisper's temperature fallback: decode greedily, and if the result looks
/// unreliable, re-decode at rising temperature until it does not.
///
/// Both openai-whisper and whisper.cpp do this by default; taking the greedy
/// result unconditionally (as we did through M1) accepts every low-confidence
/// segment as final. Two signals trigger a retry, matching the reference:
/// a poor mean log-probability, and a transcript that has collapsed into
/// repetition.
///
/// The ladder returns the first acceptable result, or — if none is acceptable
/// — the best-scoring one, so a hard segment still yields its most plausible
/// transcription rather than nothing.
fn decode_with_fallback(
    whisper: &mut LoadedWhisper,
    audio_features: &Tensor,
    cfg: &DecodeConfig,
    mode: Ladder,
) -> Result<Option<(Vec<u32>, f32)>> {
    let full: &[f32] = if cfg.temperatures.is_empty() {
        &[0.0]
    } else {
        &cfg.temperatures
    };
    let ladder: &[f32] = match mode {
        Ladder::Full => full,
        Ladder::GreedyOnly => &full[..1],
    };
    let mut best: Option<(f32, Vec<u32>, f32)> = None;

    for &temperature in ladder {
        // Beam search only at temperature 0. Above it the ladder is
        // deliberately sampling to escape a degenerate greedy result, and a
        // beam search over a sampled distribution is neither one strategy nor
        // the other — openai-whisper switches to plain sampling at t > 0 for
        // the same reason.
        let (tokens, avg_logprob, no_speech_prob) = if cfg.beam_size > 1 && temperature == 0.0 {
            decode_beam(whisper, audio_features, cfg, cfg.beam_size)?
        } else {
            decode_once(whisper, audio_features, cfg, temperature)?
        };

        // The no-speech gate, and a *deliberate divergence* from the
        // reference. openai-whisper requires BOTH a high `<|nospeech|>`
        // probability and a poor `avg_logprob`, vetoing the skip whenever the
        // transcript scores well. On silence that veto always fires: the
        // hallucination is two tokens long ("you"), and short outputs score
        // well by construction — measured -0.72 against a -1.0 bar. The
        // reference's own gate cannot catch the case it exists for.
        //
        // Measured on tiny.en, 12 LibriSpeech clips vs digital silence:
        //
        //   no_speech_prob   speech 0.010..0.076   silence 0.949
        //   avg_logprob      speech -0.24..-0.48   silence -0.72
        //
        // The first separates cleanly, with the 0.6 threshold sitting 8x above
        // the loudest speech reading; the second does not separate at all. So
        // we gate on the variable that carries the signal and drop the one
        // that does not. `no_speech_prob` comes from raw logits and does not
        // move with temperature, so the first rung decides it.
        if std::env::var_os("FFAI_DEBUG_TOKENS").is_some() {
            eprintln!(
                "[gate] no_speech_prob={no_speech_prob:.4} avg_logprob={avg_logprob:.4} \
                 (thresholds {} / {})",
                cfg.no_speech_threshold, cfg.logprob_threshold
            );
        }
        if no_speech_prob > cfg.no_speech_threshold {
            NO_SPEECH_DROPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(Some((Vec::new(), temperature)));
        }

        let text = whisper.tokenizer.decode(&tokens).unwrap_or_default();
        let looping = repetition_ratio(&text) > cfg.repetition_threshold;
        let confident = avg_logprob > cfg.logprob_threshold;

        if confident && !looping {
            return Ok(Some((tokens, temperature)));
        }
        if best
            .as_ref()
            .is_none_or(|(score, _, _)| avg_logprob > *score)
        {
            best = Some((avg_logprob, tokens, temperature));
        }
    }
    match mode {
        // No rung was acceptable: hand back the best-scoring one, so a hard
        // segment still yields its most plausible transcription.
        Ladder::Full => Ok(Some(
            best.map(|(_, tokens, temperature)| (tokens, temperature))
                .unwrap_or((Vec::new(), 1.0)),
        )),
        // The greedy rung failed its guards: the caller escalates (to the
        // full context) instead of accepting a rejected transcript.
        Ladder::GreedyOnly => Ok(None),
    }
}

/// How many windows the no-speech gate has discarded in this process.
///
/// Instrumented, not decorative: the ablation that says whether VAD is doing
/// its job is "with `--vad` on, does this stay near zero?" A gate that keeps
/// firing means silence is still reaching the model and the detector is
/// mis-tuned — which is a thing we want to be told, so the gate stays on as
/// defence in depth rather than being removed once VAD lands.
pub static NO_SPEECH_DROPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Read the drop counter. See [`NO_SPEECH_DROPS`].
pub fn no_speech_drops() -> usize {
    NO_SPEECH_DROPS.load(std::sync::atomic::Ordering::Relaxed)
}

/// A cheap stand-in for Whisper's zlib compression-ratio check: the ratio of
/// total words to *distinct* words. A healthy sentence sits near 1.0-1.5; a
/// decoder stuck in a loop climbs without bound. This avoids pulling in a
/// compression dependency to detect something this structural.
pub fn repetition_ratio(text: &str) -> f32 {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 8 {
        return 1.0;
    }
    let distinct: std::collections::HashSet<&str> = words.iter().copied().collect();
    words.len() as f32 / distinct.len().max(1) as f32
}

/// One hypothesis under beam search.
struct Beam {
    /// The full token sequence, prompt included (so `forward` can be fed the
    /// prompt once and one token per step thereafter).
    tokens: Vec<u32>,
    /// Sum of per-token log-probabilities — the quantity beams are ranked on
    /// before any length normalization.
    logprob: f32,
    /// This beam's decoder cache, so it can be resumed without re-feeding.
    state: super::text_decoder::DecoderState,
    /// Set once the beam has emitted end-of-text; finished beams stop
    /// expanding but still compete for the final answer.
    done: bool,
}

/// Beam search over the same grammar greedy decoding uses.
///
/// Whisper's references (openai-whisper, faster-whisper, whisper.cpp) all
/// default to `beam_size=5`; Mercury has been greedy since M1, and every
/// benchmark in this repo pins the references to greedy so the comparison
/// measures implementations rather than decoding strategies. That is the
/// honest comparison and it also caps what we can claim about accuracy —
/// closing that is what this function is for.
///
/// The search is the textbook one, with the two details that matter:
///
/// - **Per-beam decoder state.** Each hypothesis owns a cache snapshot
///   ([`super::text_decoder::DecoderState`]), so expanding a beam costs one
///   token's forward pass rather than a re-feed of its whole prefix.
/// - **Length normalization at selection time only.** Beams are *expanded* on
///   raw cumulative log-probability, but the winner is chosen on
///   `logprob / len^alpha`. Ranking mid-search on the normalized score
///   biases toward whichever beam happens to be shortest at that step.
///
/// Returns the same triple as [`decode_once`] so the fallback ladder and the
/// no-speech gate are untouched.
fn decode_beam(
    whisper: &mut LoadedWhisper,
    audio_features: &Tensor,
    cfg: &DecodeConfig,
    beam_size: usize,
) -> Result<(Vec<u32>, f32, f32)> {
    let eot = whisper.tokenizer.eot;
    let no_speech_tok = whisper.tokenizer.no_speech as usize;
    let english_only = whisper.is_english_only();

    const N_TEXT_CTX: usize = 448;
    let initial =
        whisper
            .tokenizer
            .initial_tokens(cfg.language, cfg.translate, cfg.timestamps, english_only);
    let mut prompt = Vec::new();
    if !cfg.prev_text_tokens.is_empty() {
        let budget = N_TEXT_CTX.saturating_sub(cfg.max_tokens + initial.len() + 1);
        if budget > 0 {
            prompt.push(whisper.tokenizer.prev);
            let skip = cfg.prev_text_tokens.len().saturating_sub(budget);
            prompt.extend_from_slice(&cfg.prev_text_tokens[skip..]);
        }
    }
    prompt.extend(initial);
    let prompt_len = prompt.len();

    let non_speech = if cfg.suppress_non_speech {
        whisper.tokenizer.non_speech_tokens()
    } else {
        Vec::new()
    };

    let p = super::profile::profile();
    whisper.decoder.reset();

    // Step 0 is shared: one forward over the prompt seeds every beam, so the
    // prompt is encoded once no matter how wide the search is.
    let logits = super::profile::timed(&p.decoder, || {
        whisper.decoder.forward(&prompt, audio_features)
    })
    .map_err(|e| Error::Model(format!("decoder forward: {e}")))?;
    let mut values: Vec<f32> = logits
        .to_dtype(ffai_core::candle::DType::F32)
        .and_then(|t| t.to_vec1())
        .map_err(tensor_err)?;
    let no_speech_prob = {
        let denom = logsumexp(&values);
        values
            .get(no_speech_tok)
            .map_or(0.0, |&v| (v - denom).exp())
    };
    apply_logit_filters(&mut values, &[], &whisper.tokenizer, cfg, &non_speech);
    let seed_state = whisper.decoder.save();

    let mut beams: Vec<Beam> = top_k(&values, beam_size)
        .into_iter()
        .map(|(token, lp)| {
            let mut tokens = prompt.clone();
            tokens.push(token);
            Beam {
                tokens,
                logprob: lp,
                state: seed_state.clone(),
                done: token == eot,
            }
        })
        .collect();

    for _ in 1..cfg.max_tokens {
        if beams.iter().all(|b| b.done) {
            break;
        }
        let mut candidates: Vec<Beam> = Vec::new();
        for beam in &beams {
            if beam.done {
                // A finished beam is carried forward unchanged; it still
                // competes on score but consumes no more compute.
                candidates.push(Beam {
                    tokens: beam.tokens.clone(),
                    logprob: beam.logprob,
                    state: beam.state.clone(),
                    done: true,
                });
                continue;
            }
            whisper.decoder.restore(&beam.state);
            let last = [*beam.tokens.last().expect("seeded with one token")];
            let logits = super::profile::timed(&p.decoder, || {
                whisper.decoder.forward(&last, audio_features)
            })
            .map_err(|e| Error::Model(format!("decoder forward: {e}")))?;
            let state = whisper.decoder.save();
            let mut values: Vec<f32> = super::profile::timed(&p.sampling, || {
                logits
                    .to_dtype(ffai_core::candle::DType::F32)
                    .and_then(|t| t.to_vec1())
            })
            .map_err(tensor_err)?;
            super::profile::timed(&p.sampling, || {
                apply_logit_filters(
                    &mut values,
                    &beam.tokens[prompt_len..],
                    &whisper.tokenizer,
                    cfg,
                    &non_speech,
                );
            });
            for (token, lp) in top_k(&values, beam_size) {
                let mut tokens = beam.tokens.clone();
                tokens.push(token);
                candidates.push(Beam {
                    tokens,
                    logprob: beam.logprob + lp,
                    state: state.clone(),
                    done: token == eot,
                });
            }
        }
        // Expand on RAW cumulative logprob (see the doc comment).
        candidates.sort_by(|a, b| b.logprob.total_cmp(&a.logprob));
        candidates.truncate(beam_size);
        beams = candidates;
    }

    // Select on the LENGTH-NORMALIZED score.
    let best = beams
        .iter()
        .max_by(|a, b| beam_score(a, prompt_len, cfg).total_cmp(&beam_score(b, prompt_len, cfg)))
        .ok_or_else(|| Error::Model("beam search produced no hypotheses".into()))?;

    let generated = best.tokens.len().saturating_sub(prompt_len);
    let avg_logprob = if generated > 0 {
        best.logprob / generated as f32
    } else {
        f32::NEG_INFINITY
    };
    p.count_tokens(generated);
    Ok((
        best.tokens[prompt_len..].to_vec(),
        avg_logprob,
        no_speech_prob,
    ))
}

/// `logprob / len^alpha` — Google NMT-style length normalization.
///
/// Without it, cumulative log-probability is monotonically decreasing in
/// length, so the search systematically prefers to stop early. `alpha = 0`
/// reproduces raw cumulative scoring.
fn beam_score(beam: &Beam, prompt_len: usize, cfg: &DecodeConfig) -> f32 {
    let len = beam.tokens.len().saturating_sub(prompt_len).max(1) as f32;
    if cfg.length_penalty == 0.0 {
        beam.logprob
    } else {
        beam.logprob / len.powf(cfg.length_penalty)
    }
}

/// The `k` highest-scoring tokens with their LOG-probabilities.
///
/// Normalizes over the filtered distribution, so a beam's cumulative score is
/// comparable with any other beam's: `apply_logit_filters` sets rejected
/// tokens to `-inf`, and those drop out of the `logsumexp` rather than
/// silently stealing mass.
fn top_k(values: &[f32], k: usize) -> Vec<(u32, f32)> {
    let denom = logsumexp(values);
    let mut idx: Vec<u32> = (0..values.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| values[b as usize].total_cmp(&values[a as usize]));
    idx.into_iter()
        .take(k)
        .filter(|&i| values[i as usize].is_finite())
        .map(|i| (i, values[i as usize] - denom))
        .collect()
}

/// Token generation at a given temperature. The KV cache is reset per window.
fn decode_once(
    whisper: &mut LoadedWhisper,
    audio_features: &Tensor,
    cfg: &DecodeConfig,
    temperature: f32,
) -> Result<(Vec<u32>, f32, f32)> {
    let eot = whisper.tokenizer.eot;
    let no_speech_tok = whisper.tokenizer.no_speech as usize;
    let english_only = whisper.is_english_only();
    // Previous-window context: `<|startofprev|> <text…>` ahead of the usual
    // prompt. The budget is exact, not openai's fixed 223: every Whisper
    // decoder has 448 positions, and prompt + generation must fit them —
    // `<|startofprev|>` + prev text + the SOT sequence + `max_tokens`
    // generated. A cap of 223 overflowed position 448 on any window that
    // generated to its 224-token limit, which is exactly the windows hard
    // enough to be running with fallback context in the first place.
    const N_TEXT_CTX: usize = 448;
    let initial =
        whisper
            .tokenizer
            .initial_tokens(cfg.language, cfg.translate, cfg.timestamps, english_only);
    let mut tokens = Vec::new();
    if !cfg.prev_text_tokens.is_empty() {
        let budget = N_TEXT_CTX.saturating_sub(cfg.max_tokens + initial.len() + 1);
        if budget > 0 {
            tokens.push(whisper.tokenizer.prev);
            let skip = cfg.prev_text_tokens.len().saturating_sub(budget);
            tokens.extend_from_slice(&cfg.prev_text_tokens[skip..]);
        }
    }
    tokens.extend(initial);
    let prompt_len = tokens.len();

    // The reference twin stays one env var away, so any suspected regression
    // in the fast path can be A/B'd against candle in seconds.
    let use_candle = std::env::var_os("FFAI_CANDLE_DECODER").is_some();
    if !use_candle {
        whisper.decoder.reset();
    }

    // Whisper's SuppressTokens list, resolved once per window rather than
    // per token (it is ~90 tokenizer lookups).
    // FFAI_ALLOW_NONSPEECH lifts SuppressTokens, which is what separates our
    // output from whisper.cpp's on non-speech events: they leave these tokens
    // enabled by default (`--suppress-nst` is opt-in), so their model writes
    // "[coughing]" where ours, unable to spell it, falls through to the
    // nearest phonetic rendering ("Hahaha").
    // Mutable because the *conditional* mode decides it at step 0, once
    // `P(<|nospeech|>)` is known. Starts suppressed so the gate has to be
    // opened deliberately rather than defaulting open.
    let mut non_speech = if cfg.suppress_non_speech
        || (cfg.annotation_threshold.is_some()
            && std::env::var_os("FFAI_ALLOW_NONSPEECH").is_none())
    {
        whisper.tokenizer.non_speech_tokens()
    } else if std::env::var_os("FFAI_ALLOW_NONSPEECH").is_some() {
        Vec::new()
    } else {
        Vec::new()
    };

    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut logprob_sum = 0.0f32;
    let mut generated = 0usize;
    let mut no_speech_prob = 0.0f32;

    let p = super::profile::profile();
    for step in 0..cfg.max_tokens {
        let logits = if use_candle {
            // Reference path: re-feed the whole sequence every step.
            let input = Tensor::new(tokens.as_slice(), &whisper.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| Error::Model(format!("building token tensor: {e}")))?;
            let hidden = super::profile::timed(&p.decoder, || {
                whisper
                    .model
                    .decoder
                    .forward(&input, audio_features, step == 0)
                    .map_err(|e| Error::Model(format!("decoder forward: {e}")))
            })?;
            // `forward` returns hidden states (batch, seq, d_model) — NOT
            // logits. Projecting to the vocabulary is a separate step;
            // skipping it makes argmax pick over d_model instead of
            // vocab_size, which decodes as fluent-looking garbage.
            let (_, seq_len, _) = hidden.dims3().map_err(tensor_err)?;
            whisper
                .model
                .decoder
                .final_linear(&hidden.i((..1, seq_len - 1..)).map_err(tensor_err)?)
                .map_err(|e| Error::Model(format!("final_linear: {e}")))?
                .i(0)
                .and_then(|t| t.i(0))
                .map_err(tensor_err)?
        } else {
            // Fast path: the whole prompt once, then one token per step.
            let feed: &[u32] = if step == 0 {
                &tokens
            } else {
                &tokens[tokens.len() - 1..]
            };
            super::profile::timed(&p.decoder, || {
                whisper
                    .decoder
                    .forward(feed, audio_features)
                    .map_err(|e| Error::Model(format!("decoder forward: {e}")))
            })?
        };
        // `P(<|nospeech|>)` at the first position, read from UNFILTERED logits
        // — `apply_logit_filters` drives that token to -inf, so reading it
        // afterwards always yields zero. Position 0 is the only place it is
        // defined.
        //
        // It is read HERE rather than inside the sampling closure because the
        // conditional annotation gate has to act on it *before* the first
        // token is filtered: the annotation opens with `[` or `(`, so a
        // decision made one step later is a decision made too late.
        let step0_no_speech = if step == 0 {
            let values: Vec<f32> = logits
                .to_dtype(ffai_core::candle::DType::F32)
                .and_then(|t| t.to_vec1())
                .map_err(tensor_err)?;
            let denom = logsumexp(&values);
            values
                .get(no_speech_tok)
                .map_or(0.0, |&v| (v - denom).exp())
        } else {
            0.0
        };
        if step == 0
            && let Some(threshold) = cfg.annotation_threshold
        {
            // Open the gate only when the model itself says this is not
            // ordinary speech. Measured on tiny.en: clean speech reads
            // 0.004-0.032, a cough or laugh 0.29-0.45, silence 0.95 (and
            // silence never gets here — the no-speech gate drops it).
            if step0_no_speech >= threshold {
                non_speech.clear();
            }
        }

        let (next, token_logprob, ns_prob) =
            super::profile::timed(&p.sampling, || -> Result<(u32, f32, f32)> {
                // Logit filtering and sampling always run in f32, whatever the
                // weights are stored as: the vocabulary softmax is where
                // half-precision rounding would actually change a token choice.
                let mut values: Vec<f32> = logits
                    .to_dtype(ffai_core::candle::DType::F32)
                    .and_then(|t| t.to_vec1())
                    .map_err(tensor_err)?;
                // `<|nospeech|>` is read for the caller's benefit only when the
                // conditional gate is off; with it on the value is needed BEFORE
                // filtering (see the block above this closure) and is passed in.
                let ns = if step == 0 { step0_no_speech } else { 0.0 };
                apply_logit_filters(
                    &mut values,
                    &tokens[prompt_len..],
                    &whisper.tokenizer,
                    cfg,
                    &non_speech,
                );
                let chosen = if temperature > 0.0 {
                    sample(&values, temperature, &mut rng)
                } else {
                    argmax(&values)
                };
                // log P(chosen) = logit - logsumexp(logits), the quantity the
                // fallback threshold is defined on.
                let lp = values[chosen as usize] - logsumexp(&values);
                Ok((chosen, lp, ns))
            })?;
        if step == 0 {
            no_speech_prob = ns_prob;
        }
        tokens.push(next);
        logprob_sum += token_logprob;
        generated += 1;
        if next == eot {
            break;
        }
        // Short-context rung only: a decode whose running mean is already
        // clearly under the acceptance bar is headed for rejection — stop
        // paying for it. 14 tokens before judging, so one bad opening word
        // cannot trigger it.
        if let Some(bar) = cfg.early_abort_logprob
            && generated >= 14
            && logprob_sum / generated as f32 <= bar
        {
            break;
        }
    }
    p.count_tokens(tokens.len() - prompt_len);
    let avg = if generated > 0 {
        logprob_sum / generated as f32
    } else {
        f32::NEG_INFINITY
    };
    Ok((tokens[prompt_len..].to_vec(), avg, no_speech_prob))
}

/// Temperature sampling from raw logits, with a deterministic RNG so a run is
/// reproducible — a benchmark that cannot be repeated is not a measurement.
fn sample(logits: &[f32], temperature: f32, rng: &mut u64) -> u32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return 0;
    }
    let weights: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max) / temperature).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    // xorshift64*: tiny, deterministic, and adequate for token sampling.
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    let mut point = (*rng >> 11) as f32 / (1u64 << 53) as f32 * total;
    for (i, &w) in weights.iter().enumerate() {
        point -= w;
        if point <= 0.0 {
            return i as u32;
        }
    }
    argmax(logits)
}

/// Whisper's decoding rules, applied to the logits before sampling.
///
/// Greedy argmax over raw logits is not Whisper's decoder — it is a decoder
/// that happens to work most of the time. Without these filters the model can
/// select `<|endoftext|>` at the very first position and emit an empty
/// transcript, or sample control tokens into the middle of a sentence. Both
/// were observed on base.en before this landed.
///
/// Ported from `whisper/decoding.py`: `SuppressBlank`, `SuppressTokens`, and
/// `ApplyTimestampRules`.
fn apply_logit_filters(
    logits: &mut [f32],
    generated: &[u32],
    tk: &super::tokenizer::WhisperTokenizer,
    cfg: &DecodeConfig,
    non_speech: &[u32],
) {
    let vocab = logits.len() as u32;
    let suppress = |logits: &mut [f32], token: u32| {
        if let Some(slot) = logits.get_mut(token as usize) {
            *slot = f32::NEG_INFINITY;
        }
    };
    let suppress_range = |logits: &mut [f32], range: std::ops::Range<u32>| {
        for token in range {
            if let Some(slot) = logits.get_mut(token as usize) {
                *slot = f32::NEG_INFINITY;
            }
        }
    };

    // Control tokens that are never valid output.
    for token in [tk.sot, tk.prev, tk.no_speech, tk.no_timestamps] {
        suppress(logits, token);
    }
    // SuppressTokens: symbols that annotate audio rather than transcribe it.
    for &token in non_speech {
        suppress(logits, token);
    }

    // SuppressBlank: the transcript may not open with a space or with
    // end-of-text.
    if generated.is_empty() {
        suppress(logits, tk.eot);
        if let Some(space) = tk.space {
            suppress(logits, space);
        }
    }

    if !cfg.timestamps {
        suppress_range(logits, tk.timestamp_begin..vocab);
        return;
    }

    // Timestamps come in pairs, except immediately before end-of-text.
    let last_is_ts = generated.last().is_some_and(|&t| tk.is_timestamp(t));
    let penultimate_is_ts = generated.len() < 2 || tk.is_timestamp(generated[generated.len() - 2]);
    if last_is_ts {
        if penultimate_is_ts {
            // A pair just closed: the next token must be text.
            suppress_range(logits, tk.timestamp_begin..vocab);
        } else {
            // A segment just opened: the next token must close it (or end).
            suppress_range(logits, 0..tk.eot);
        }
    }

    // Timestamps must not run backwards.
    if let Some(&last) = generated.iter().rev().find(|&&t| tk.is_timestamp(t)) {
        suppress_range(logits, tk.timestamp_begin..last.min(vocab));
    }

    // `max_initial_timestamp`: a window may not open more than 1 s in. Without
    // this, a model can open a segment deep inside the window and then have
    // nothing coherent to say — base.en opened at 7.00 s and immediately
    // emitted end-of-text.
    if generated.is_empty() {
        let last_allowed = tk.timestamp_begin + MAX_INITIAL_TIMESTAMP_INDEX;
        suppress_range(logits, (last_allowed + 1).min(vocab)..vocab);
    }

    // When the encoder ran on a shortened context, a timestamp past the
    // encoded extent points at audio the model never saw. The 2026-07 prune
    // of the variable-length window (mission plan §6.4) decoded without this
    // mask; timestamps drifting past the real audio is one of the ways a
    // truncated window destabilizes.
    if let Some(max_secs) = cfg.max_timestamp_secs {
        let last_allowed = tk
            .timestamp_begin
            .saturating_add((max_secs / 0.02).floor().max(0.0) as u32);
        suppress_range(logits, (last_allowed + 1).min(vocab)..vocab);
    }

    // THE rule that makes Whisper emit `<|0.00|>` first: if the total
    // probability mass on timestamps exceeds the single best text token,
    // a timestamp must be sampled.
    //
    // Comparing in log space, log_softmax's normalizer is common to both
    // sides and cancels, so raw logits suffice:
    //     logsumexp(timestamp logits)  vs  max(text logits)
    //
    // Without this, greedy argmax picks the locally-likeliest *text* token
    // and the timestamp grammar never starts — which surfaced as tiny.en
    // prefixing every transcript with "." and base.en emitting nothing at
    // all on some clips.
    let ts = tk.timestamp_begin as usize;
    if ts < logits.len() {
        let timestamp_mass = logsumexp(&logits[ts..]);
        let best_text = logits[..ts]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if timestamp_mass > best_text {
            suppress_range(logits, 0..tk.timestamp_begin);
        }
    }
}

/// Numerically stable `log(sum(exp(xs)))`, skipping suppressed (-inf) entries.
fn logsumexp(xs: &[f32]) -> f32 {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return f32::NEG_INFINITY;
    }
    let sum: f32 = xs.iter().map(|&x| (x - max).exp()).sum();
    max + sum.ln()
}

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or_default()
}

fn tensor_err(e: ffai_core::candle::Error) -> Error {
    Error::Model(format!("tensor op: {e}"))
}

/// Split a token stream on timestamp-token pairs into timed segments.
fn segments_from_tokens(
    whisper: &LoadedWhisper,
    tokens: &[u32],
    offset_secs: f64,
    cfg: &DecodeConfig,
    window_secs: f64,
) -> Result<Vec<TimedSegment<String>>> {
    let tk = &whisper.tokenizer;
    if !cfg.timestamps {
        let text = tk.decode(tokens)?;
        return Ok(if text.trim().is_empty() {
            Vec::new()
        } else {
            vec![TimedSegment {
                start: offset_secs,
                end: offset_secs + window_secs,
                value: text,
                confidence: None,
            }]
        });
    }

    let mut segments = Vec::new();
    let mut current_start: Option<f64> = None;
    let mut buffer: Vec<u32> = Vec::new();

    for &token in tokens {
        if token == tk.eot {
            break;
        }
        if tk.is_timestamp(token) {
            let time = tk.timestamp_secs(token);
            match current_start {
                // Closing timestamp: flush the buffered text as a segment.
                Some(start) => {
                    let text = tk.decode(&buffer)?;
                    if !text.trim().is_empty() {
                        segments.push(TimedSegment {
                            start: offset_secs + start,
                            end: offset_secs + time,
                            value: text,
                            confidence: None,
                        });
                    }
                    buffer.clear();
                    current_start = None;
                }
                None => current_start = Some(time),
            }
        } else {
            buffer.push(token);
        }
    }

    // Unterminated tail: the model ran out of tokens mid-segment.
    if !buffer.is_empty() {
        let text = tk.decode(&buffer)?;
        if !text.trim().is_empty() {
            let start = current_start.unwrap_or(0.0);
            segments.push(TimedSegment {
                start: offset_secs + start,
                end: offset_secs + window_secs,
                value: text,
                confidence: None,
            });
        }
    }
    Ok(segments)
}

/// Split samples into 30-second windows, the model's fixed context size.
/// The final partial window is padded by [`MelChunk::to_context_window`].
pub fn windows(samples: &[f32]) -> impl Iterator<Item = (usize, &[f32])> {
    samples
        .chunks(mel::N_SAMPLES)
        .enumerate()
        .map(|(i, chunk)| (i * mel::N_SAMPLES, chunk))
}

/// Type marker so `m` stays referenced even when the model module changes.
#[allow(dead_code)]
type WhisperModel = m::model::Whisper;

#[cfg(test)]
mod beam_tests {
    use super::*;

    fn cfg_with(beam: usize, penalty: f32) -> DecodeConfig {
        DecodeConfig {
            beam_size: beam,
            length_penalty: penalty,
            ..DecodeConfig::default()
        }
    }

    /// `top_k` must return LOG-probabilities normalized over the FILTERED
    /// distribution. If it returned raw logits, beams of different lengths
    /// would accumulate scores on different scales and the comparison
    /// between them would be meaningless — a bug that still produces
    /// plausible transcripts, which is why it is pinned here.
    #[test]
    fn top_k_returns_normalized_logprobs_in_rank_order() {
        // Two live tokens and one suppressed: exp(1)+exp(2) is the whole
        // mass, so the top token's logprob is 2 - logsumexp([1,2]).
        let values = vec![1.0f32, 2.0, f32::NEG_INFINITY];
        let got = top_k(&values, 3);
        assert_eq!(got.len(), 2, "-inf tokens must not be selectable");
        assert_eq!(got[0].0, 1, "highest logit first");
        let denom = (1.0f32.exp() + 2.0f32.exp()).ln();
        assert!((got[0].1 - (2.0 - denom)).abs() < 1e-5, "got {}", got[0].1);
        // A normalized distribution's probabilities sum to 1.
        let mass: f32 = got.iter().map(|(_, lp)| lp.exp()).sum();
        assert!(
            (mass - 1.0).abs() < 1e-5,
            "probabilities must sum to 1, got {mass}"
        );
    }

    /// Length normalization must not change the ranking of equal-length
    /// beams, and must favour the longer of two beams with equal cumulative
    /// logprob — that is the whole point of dividing by `len^alpha`.
    #[test]
    fn length_penalty_favours_longer_at_equal_cumulative_score() {
        let state = super::super::text_decoder::DecoderState::empty();
        let mk = |n: usize, lp: f32| Beam {
            tokens: vec![0u32; 3 + n], // 3-token prompt
            logprob: lp,
            state: state.clone(),
            done: false,
        };
        let short = mk(2, -4.0);
        let long = mk(8, -4.0);

        // alpha = 0 -> raw cumulative: a tie.
        let c0 = cfg_with(5, 0.0);
        assert_eq!(beam_score(&short, 3, &c0), beam_score(&long, 3, &c0));

        // alpha = 1 -> divided by length: the longer beam wins, because the
        // same total surprise spread over more tokens is a better sequence.
        let c1 = cfg_with(5, 1.0);
        assert!(
            beam_score(&long, 3, &c1) > beam_score(&short, 3, &c1),
            "length penalty must reward the longer hypothesis at equal cumulative score"
        );
    }

    /// The default must stay greedy. Every ledger line in this repo was
    /// measured with greedy decoding and the references are pinned to it;
    /// flipping this default would silently invalidate that comparison.
    #[test]
    fn default_is_greedy() {
        // Guard against a stray environment leaking into the assertion.
        if std::env::var_os("FFAI_BEAM_SIZE").is_some() {
            return;
        }
        assert_eq!(DecodeConfig::default().beam_size, 1);
    }
}
