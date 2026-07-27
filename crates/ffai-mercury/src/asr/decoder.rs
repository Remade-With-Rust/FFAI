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
    /// Retry when the transcript looks like a repetition loop (see
    /// [`repetition_ratio`]).
    pub repetition_threshold: f32,
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
            repetition_threshold: 2.4,
        }
    }
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
) -> Result<Vec<TimedSegment<String>>> {
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
    let tokens = decode_with_fallback(whisper, &audio_features, cfg)?;
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
    segments_from_tokens(whisper, &tokens, offset_secs, cfg, window_secs)
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
) -> Result<Vec<u32>> {
    let ladder: &[f32] = if cfg.temperatures.is_empty() { &[0.0] } else { &cfg.temperatures };
    let mut best: Option<(f32, Vec<u32>)> = None;

    for &temperature in ladder {
        let (tokens, avg_logprob) = decode_once(whisper, audio_features, cfg, temperature)?;
        let text = whisper.tokenizer.decode(&tokens).unwrap_or_default();
        let looping = repetition_ratio(&text) > cfg.repetition_threshold;
        let confident = avg_logprob > cfg.logprob_threshold;

        if confident && !looping {
            return Ok(tokens);
        }
        if best.as_ref().is_none_or(|(score, _)| avg_logprob > *score) {
            best = Some((avg_logprob, tokens));
        }
    }
    Ok(best.map(|(_, tokens)| tokens).unwrap_or_default())
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

/// Token generation at a given temperature. The KV cache is reset per window.
fn decode_once(
    whisper: &mut LoadedWhisper,
    audio_features: &Tensor,
    cfg: &DecodeConfig,
    temperature: f32,
) -> Result<(Vec<u32>, f32)> {
    let eot = whisper.tokenizer.eot;
    let english_only = whisper.is_english_only();
    let mut tokens = whisper.tokenizer.initial_tokens(
        cfg.language,
        cfg.translate,
        cfg.timestamps,
        english_only,
    );
    let prompt_len = tokens.len();

    // The reference twin stays one env var away, so any suspected regression
    // in the fast path can be A/B'd against candle in seconds.
    let use_candle = std::env::var_os("FFAI_CANDLE_DECODER").is_some();
    if !use_candle {
        whisper.decoder.reset();
    }

    // Whisper's SuppressTokens list, resolved once per window rather than
    // per token (it is ~90 tokenizer lookups).
    let non_speech = whisper.tokenizer.non_speech_tokens();

    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut logprob_sum = 0.0f32;
    let mut generated = 0usize;

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
            let feed: &[u32] = if step == 0 { &tokens } else { &tokens[tokens.len() - 1..] };
            super::profile::timed(&p.decoder, || {
                whisper
                    .decoder
                    .forward(feed, audio_features)
                    .map_err(|e| Error::Model(format!("decoder forward: {e}")))
            })?
        };
        let (next, token_logprob) = super::profile::timed(&p.sampling, || -> Result<(u32, f32)> {
            // Logit filtering and sampling always run in f32, whatever the
            // weights are stored as: the vocabulary softmax is where
            // half-precision rounding would actually change a token choice.
            let mut values: Vec<f32> = logits
                .to_dtype(ffai_core::candle::DType::F32)
                .and_then(|t| t.to_vec1())
                .map_err(tensor_err)?;
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
            Ok((chosen, lp))
        })?;
        tokens.push(next);
        logprob_sum += token_logprob;
        generated += 1;
        if next == eot {
            break;
        }
    }
    p.count_tokens(tokens.len() - prompt_len);
    let avg = if generated > 0 { logprob_sum / generated as f32 } else { f32::NEG_INFINITY };
    Ok((tokens[prompt_len..].to_vec(), avg))
}

/// Temperature sampling from raw logits, with a deterministic RNG so a run is
/// reproducible — a benchmark that cannot be repeated is not a measurement.
fn sample(logits: &[f32], temperature: f32, rng: &mut u64) -> u32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return 0;
    }
    let weights: Vec<f32> = logits.iter().map(|&l| ((l - max) / temperature).exp()).collect();
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
    let penultimate_is_ts = generated.len() < 2
        || tk.is_timestamp(generated[generated.len() - 2]);
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
        let best_text = logits[..ts].iter().copied().fold(f32::NEG_INFINITY, f32::max);
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
