//! Forced alignment — put a timestamp on every word.
//!
//! **Why Whisper cannot do this itself.** Whisper emits timestamp *tokens*
//! chosen by the same sampler that picks words, at a 20 ms quantisation, and
//! only at segment granularity. They drift, they round, and they describe a
//! phrase rather than a word. Anything that needs to know when a particular
//! word was said — a caption that highlights, an edit that cuts on a word, a
//! speaker label attached to the right speaker — needs a different mechanism.
//!
//! **The mechanism.** Run a phoneme-level CTC acoustic model over the audio to
//! get per-frame character probabilities, then find the single most likely
//! alignment of the *known* transcript against those probabilities. Because
//! the text is already known, this is not recognition: it is a shortest-path
//! problem over a trellis, solved exactly by dynamic programming. Same
//! approach as `WhisperX` (Bain et al., Interspeech 2023) and torchaudio's
//! forced-alignment pipeline.
//!
//! **This module is deliberately model-free.** It takes emissions as data and
//! knows nothing about where they came from. That is what makes the hard part
//! testable before the expensive part exists: every function here is exercised
//! against synthetic emissions in the tests below, with no weights, no fetch,
//! and no candle. The wav2vec2 port that produces real emissions is a separate
//! piece of work (Mercury-X §C2) and cannot break this one.

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

use std::collections::HashMap;

use ffai_core::types::TimedSegment;

/// Per-frame log-probabilities from a CTC acoustic model.
///
/// Row-major `frames × vocab`. Log-probabilities, not logits: the alignment
/// sums them along a path, and summing unnormalised logits would compare
/// paths of different lengths on different scales.
#[derive(Debug, Clone)]
pub struct Emissions {
    pub frames: usize,
    pub vocab: usize,
    /// `frames * vocab` log-probs, row-major.
    pub data: Vec<f32>,
    /// Seconds of audio per frame (wav2vec2-base: 0.02).
    pub frame_secs: f64,
}

impl Emissions {
    pub fn new(
        frames: usize,
        vocab: usize,
        data: Vec<f32>,
        frame_secs: f64,
    ) -> Result<Self, String> {
        if data.len() != frames * vocab {
            return Err(format!(
                "emissions are {}×{} = {} values but {} were given",
                frames,
                vocab,
                frames * vocab,
                data.len()
            ));
        }
        Ok(Self {
            frames,
            vocab,
            data,
            frame_secs,
        })
    }

    #[inline]
    fn at(&self, frame: usize, token: u32) -> f32 {
        self.data[frame * self.vocab + token as usize]
    }
}

/// One aligned token and the frames it occupies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenSpan {
    pub token: u32,
    /// Inclusive first frame.
    pub start: usize,
    /// Exclusive last frame.
    pub end: usize,
    /// Mean per-frame probability over the token's own (non-blank) frames.
    pub score: f32,
}

/// The character inventory a CTC model was trained on.
///
/// wav2vec2's English alphabet is uppercase letters plus an apostrophe, with
/// a literal `|` standing in for space. Lowercase text and punctuation are not
/// in the inventory at all, so tokenising has to fold them away — and what it
/// folds away is exactly what the alignment cannot place.
#[derive(Debug, Clone)]
pub struct CtcAlphabet {
    pub blank: u32,
    pub word_sep: Option<u32>,
    lookup: HashMap<char, u32>,
}

impl CtcAlphabet {
    /// Build from an ordered character list; index in the list is the token id.
    #[must_use]
    pub fn new(chars: &[char], blank: u32, word_sep: Option<char>) -> Self {
        let lookup: HashMap<char, u32> = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
        let word_sep = word_sep.and_then(|c| lookup.get(&c).copied());
        Self {
            blank,
            word_sep,
            lookup,
        }
    }

    /// The standard wav2vec2 English CTC alphabet, in its canonical order.
    #[must_use]
    pub fn wav2vec2_english() -> Self {
        let chars: Vec<char> = "<pad><s></s><unk>|ETAONIHSRDLUMWCFGYPBVK'XJQZ"
            .chars()
            .collect::<Vec<_>>()
            // The four control tokens above occupy ids 0..4 as single slots,
            // not as their spelled-out characters — rebuild the list properly.
            .into_iter()
            .collect();
        // Explicit table rather than parsing the string above, which is only
        // kept as documentation of the upstream ordering.
        let _ = chars;
        let table: Vec<char> = vec![
            '\u{0}', // <pad>, the CTC blank
            '\u{1}', // <s>
            '\u{2}', // </s>
            '\u{3}', // <unk>
            '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F',
            'G', 'Y', 'P', 'B', 'V', 'K', '\'', 'X', 'J', 'Q', 'Z',
        ];
        Self::new(&table, 0, Some('|'))
    }

    /// Map text onto token ids, folding case and dropping anything the model
    /// has no symbol for.
    ///
    /// Returns the ids alongside the *kept* characters, because the caller
    /// needs to know which characters survived in order to reassemble words
    /// from the aligned tokens. Silently dropping a character and then
    /// indexing the original string is how word boundaries end up off by one.
    #[must_use]
    pub fn tokenize(&self, text: &str) -> (Vec<u32>, Vec<char>) {
        let mut ids = Vec::new();
        let mut kept = Vec::new();
        let mut last_was_sep = true; // suppress a leading separator
        for ch in text.chars() {
            let upper = ch.to_ascii_uppercase();
            if upper.is_whitespace() {
                if let Some(sep) = self.word_sep
                    && !last_was_sep
                {
                    ids.push(sep);
                    kept.push(' ');
                    last_was_sep = true;
                }
                continue;
            }
            if let Some(&id) = self.lookup.get(&upper) {
                ids.push(id);
                kept.push(upper);
                last_was_sep = false;
            }
            // Unknown characters (punctuation, digits, accents) are dropped:
            // the model cannot emit them, so no alignment exists for them.
        }
        // A trailing separator aligns to nothing and only creates an empty word.
        if last_was_sep && !ids.is_empty() {
            ids.pop();
            kept.pop();
        }
        (ids, kept)
    }
}

/// Align a known token sequence to emissions, returning one span per token.
///
/// Exact Viterbi over the trellis: `trellis[t][j]` is the best log-probability
/// of having consumed frames `0..=t` while sitting on token `j`. From a state
/// the path may **stay** — emitting either the blank or the token again — or
/// **advance** to the next token. That is the standard forced-alignment
/// formulation; it does not model the CTC rule that a repeated symbol needs an
/// intervening blank, which matters for decoding and not for aligning a
/// transcript that is already known.
///
/// Fails rather than guessing when the audio is too short to hold the text:
/// fewer frames than tokens has no valid path, and returning a degenerate
/// alignment there would put every word at the same timestamp.
pub fn forced_align(
    emissions: &Emissions,
    tokens: &[u32],
    blank: u32,
) -> Result<Vec<TokenSpan>, String> {
    let (t_n, j_n) = (emissions.frames, tokens.len());
    if j_n == 0 {
        return Ok(Vec::new());
    }
    if t_n < j_n {
        return Err(format!(
            "cannot align {j_n} tokens to {t_n} frames — the audio is shorter than the text"
        ));
    }
    if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= emissions.vocab) {
        return Err(format!(
            "token id {bad} is outside the {}-symbol vocabulary",
            emissions.vocab
        ));
    }

    const NEG: f32 = f32::NEG_INFINITY;
    let mut trellis = vec![NEG; t_n * j_n];
    // `advanced[t][j]`: did the best path into (t, j) come from token j-1?
    let mut advanced = vec![false; t_n * j_n];
    // `emitted[t][j]`: did the best path into (t, j) emit the token rather
    // than the blank? Recorded so a token's span can exclude the blank frames
    // that merely padded it, which otherwise stretch every word to touch its
    // neighbour.
    let mut emitted = vec![false; t_n * j_n];

    // At frame 0 the path sits on token 0, but it may be emitting the token
    // OR the blank — audio almost never begins exactly on a phoneme.
    //
    // Forcing `emitted[0] = true` here made the first token's span always
    // start at frame 0, so the first word absorbed every leading silent frame
    // in its segment. Measured on the long-form corpus, that placed first
    // words a consistent 0.16-0.24 s early: VAD pads each speech region by
    // SPEECH_PAD_MS (150 ms) and one alignment frame is 20 ms, which is
    // exactly the offset observed. A systematic bias, not scatter — and
    // invisible on a single short clip with no leading pad.
    let first_token = emissions.at(0, tokens[0]);
    let first_blank = emissions.at(0, blank);
    if first_token >= first_blank {
        trellis[0] = first_token;
        emitted[0] = true;
    } else {
        trellis[0] = first_blank;
        emitted[0] = false;
    }

    for t in 1..t_n {
        for j in 0..j_n.min(t + 1) {
            let tok = emissions.at(t, tokens[j]);
            let blk = emissions.at(t, blank);

            let prev_same = trellis[(t - 1) * j_n + j];
            // Staying may emit the blank or repeat the token; take whichever
            // the acoustics prefer at this frame.
            let (stay_score, stay_emitted) = if prev_same == NEG {
                (NEG, false)
            } else if tok > blk {
                (prev_same + tok, true)
            } else {
                (prev_same + blk, false)
            };

            let adv_score = if j == 0 {
                NEG
            } else {
                let prev = trellis[(t - 1) * j_n + j - 1];
                if prev == NEG { NEG } else { prev + tok }
            };

            let idx = t * j_n + j;
            if adv_score > stay_score {
                trellis[idx] = adv_score;
                advanced[idx] = true;
                emitted[idx] = true;
            } else {
                trellis[idx] = stay_score;
                emitted[idx] = stay_emitted;
            }
        }
    }

    if trellis[(t_n - 1) * j_n + (j_n - 1)] == NEG {
        return Err("no valid alignment path — emissions may be degenerate".into());
    }

    // Backtrace: walk frames backwards, recording which token owns each and
    // whether it actually emitted there.
    let mut owner = vec![0usize; t_n];
    let mut was_emit = vec![false; t_n];
    let mut j = j_n - 1;
    for t in (0..t_n).rev() {
        let idx = t * j_n + j;
        owner[t] = j;
        was_emit[t] = emitted[idx];
        if advanced[idx] && j > 0 {
            j -= 1;
        }
    }

    // Collect each token's frames. A token's span is the frames where it
    // actually emitted; if it only ever rode along under a blank, fall back to
    // the frames assigned to it so it still gets a position.
    let mut spans = Vec::with_capacity(j_n);
    for (j, &token) in tokens.iter().enumerate() {
        let assigned: Vec<usize> = (0..t_n).filter(|&t| owner[t] == j).collect();
        if assigned.is_empty() {
            // Should be unreachable given t_n >= j_n, but a silently missing
            // token would corrupt every word boundary after it.
            return Err(format!("token {j} received no frames during backtrace"));
        }
        let emitting: Vec<usize> = assigned.iter().copied().filter(|&t| was_emit[t]).collect();
        let frames = if emitting.is_empty() {
            &assigned
        } else {
            &emitting
        };
        let start = frames[0];
        let end = frames[frames.len() - 1] + 1;
        let score = frames
            .iter()
            .map(|&t| emissions.at(t, token).exp())
            .sum::<f32>()
            / frames.len() as f32;
        spans.push(TokenSpan {
            token,
            start,
            end,
            score,
        });
    }
    Ok(spans)
}

/// Group aligned character spans into words.
///
/// `kept` is the character each token came from, as returned by
/// [`CtcAlphabet::tokenize`] — the two are index-parallel by construction, and
/// pairing them any other way is how a word ends up carrying its neighbour's
/// timing.
#[must_use]
pub fn words_from_spans(
    spans: &[TokenSpan],
    kept: &[char],
    frame_secs: f64,
    time_offset: f64,
) -> Vec<TimedSegment<String>> {
    let mut words: Vec<TimedSegment<String>> = Vec::new();
    let mut text = String::new();
    let mut first: Option<usize> = None;
    let mut last = 0usize;
    let mut score_sum = 0.0f32;
    let mut score_n = 0usize;

    let flush = |text: &mut String,
                 first: &mut Option<usize>,
                 last: usize,
                 score_sum: &mut f32,
                 score_n: &mut usize,
                 out: &mut Vec<TimedSegment<String>>| {
        if let Some(f) = first.take() {
            if !text.is_empty() {
                out.push(TimedSegment {
                    start: (f as f64).mul_add(frame_secs, time_offset),
                    end: (last as f64).mul_add(frame_secs, time_offset),
                    value: std::mem::take(text),
                    confidence: (*score_n > 0).then(|| *score_sum / *score_n as f32),
                });
            }
            text.clear();
        }
        *score_sum = 0.0;
        *score_n = 0;
    };

    for (span, &ch) in spans.iter().zip(kept.iter()) {
        if ch == ' ' {
            flush(
                &mut text,
                &mut first,
                last,
                &mut score_sum,
                &mut score_n,
                &mut words,
            );
            continue;
        }
        if first.is_none() {
            first = Some(span.start);
        }
        last = span.end;
        text.push(ch);
        score_sum += span.score;
        score_n += 1;
    }
    flush(
        &mut text,
        &mut first,
        last,
        &mut score_sum,
        &mut score_n,
        &mut words,
    );
    words
}

/// End-to-end: emissions plus known text in, word timings out.
pub fn align_words(
    emissions: &Emissions,
    text: &str,
    alphabet: &CtcAlphabet,
    time_offset: f64,
) -> Result<Vec<TimedSegment<String>>, String> {
    let (tokens, kept) = alphabet.tokenize(text);
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    let spans = forced_align(emissions, &tokens, alphabet.blank)?;
    Ok(words_from_spans(
        &spans,
        &kept,
        emissions.frame_secs,
        time_offset,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build emissions that put `plan[t]` at high probability on frame `t`
    /// and spread the rest thinly. Log-probabilities, roughly normalised.
    fn emissions_for(plan: &[u32], vocab: usize, frame_secs: f64) -> Emissions {
        let mut data = vec![0.0f32; plan.len() * vocab];
        for (t, &want) in plan.iter().enumerate() {
            for v in 0..vocab {
                data[t * vocab + v] = if v as u32 == want {
                    (0.9f32).ln()
                } else {
                    (0.1f32 / vocab as f32).ln()
                };
            }
        }
        Emissions::new(plan.len(), vocab, data, frame_secs).expect("shape checked")
    }

    #[test]
    fn shape_mismatch_is_rejected() {
        assert!(Emissions::new(3, 4, vec![0.0; 11], 0.02).is_err());
    }

    #[test]
    fn aligns_a_clean_two_token_sequence() {
        // blank=0. Plan: token 1 for 2 frames, blank, token 2 for 2 frames.
        let plan = [1, 1, 0, 2, 2];
        let e = emissions_for(&plan, 4, 0.02);
        let spans = forced_align(&e, &[1, 2], 0).expect("alignable");
        assert_eq!(spans.len(), 2);
        assert_eq!((spans[0].start, spans[0].end), (0, 2), "{spans:?}");
        assert_eq!((spans[1].start, spans[1].end), (3, 5), "{spans:?}");
    }

    #[test]
    fn leading_silence_is_not_absorbed_by_the_first_word() {
        // Regression: the first token used to be FORCED to emit at frame 0,
        // so a word always started at the beginning of its segment however
        // much silence preceded it. On real audio that placed every
        // segment-initial word ~0.17 s early — VAD's 150 ms pad plus a frame.
        let plan = [0, 0, 0, 0, 0, 1, 1, 2, 2];
        let e = emissions_for(&plan, 4, 0.02);
        let spans = forced_align(&e, &[1, 2], 0).expect("alignable");
        assert!(
            spans[0].start >= 4,
            "first word absorbed {} leading blank frames: {spans:?}",
            spans[0].start
        );
    }

    #[test]
    fn blank_padding_is_excluded_from_a_span() {
        // The blank frames between tokens must not be absorbed into either
        // token's span, or every word touches its neighbour.
        let plan = [1, 0, 0, 0, 2];
        let e = emissions_for(&plan, 4, 0.02);
        let spans = forced_align(&e, &[1, 2], 0).expect("alignable");
        assert_eq!(spans[0].end, 1, "leading token absorbed blanks: {spans:?}");
        assert_eq!(
            spans[1].start, 4,
            "trailing token absorbed blanks: {spans:?}"
        );
    }

    #[test]
    fn spans_are_monotonic_and_non_overlapping() {
        let plan = [1, 1, 0, 2, 0, 3, 3, 3, 0, 1];
        let e = emissions_for(&plan, 5, 0.02);
        let spans = forced_align(&e, &[1, 2, 3, 1], 0).expect("alignable");
        for pair in spans.windows(2) {
            assert!(pair[0].end <= pair[1].start, "overlap in {spans:?}");
        }
    }

    #[test]
    fn audio_shorter_than_text_is_an_error_not_a_guess() {
        let e = emissions_for(&[1, 2], 5, 0.02);
        let err = forced_align(&e, &[1, 2, 3, 4], 0).expect_err("must refuse");
        assert!(err.contains("shorter than the text"), "{err}");
    }

    #[test]
    fn out_of_range_token_is_rejected() {
        let e = emissions_for(&[1, 2, 3], 4, 0.02);
        assert!(forced_align(&e, &[1, 99], 0).is_err());
    }

    #[test]
    fn empty_text_aligns_to_nothing() {
        let e = emissions_for(&[1, 2, 3], 4, 0.02);
        assert!(forced_align(&e, &[], 0).expect("ok").is_empty());
    }

    #[test]
    fn alphabet_folds_case_and_drops_unknown_symbols() {
        let a = CtcAlphabet::wav2vec2_english();
        let (ids, kept) = a.tokenize("Hi, there!");
        let text: String = kept.iter().collect();
        assert_eq!(text, "HI THERE", "punctuation should be dropped: {text:?}");
        assert_eq!(
            ids.len(),
            kept.len(),
            "ids and chars must stay index-parallel"
        );
        assert_eq!(ids[2], a.word_sep.expect("separator exists"));
    }

    #[test]
    fn alphabet_suppresses_leading_and_trailing_separators() {
        let a = CtcAlphabet::wav2vec2_english();
        let (ids, kept) = a.tokenize("  hi  ");
        assert_eq!(kept.iter().collect::<String>(), "HI");
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn words_are_split_on_the_separator_with_their_own_times() {
        let a = CtcAlphabet::wav2vec2_english();
        let (tokens, kept) = a.tokenize("AB C");
        // One frame per token, in order, so timings are trivially checkable.
        let e = emissions_for(&tokens, 40, 0.02);
        let spans = forced_align(&e, &tokens, a.blank).expect("alignable");
        let words = words_from_spans(&spans, &kept, 0.02, 0.0);
        assert_eq!(words.len(), 2, "{words:?}");
        assert_eq!(words[0].value, "AB");
        assert_eq!(words[1].value, "C");
        assert!(words[0].end <= words[1].start, "{words:?}");
    }

    #[test]
    fn time_offset_shifts_words_into_the_source_time_base() {
        let a = CtcAlphabet::wav2vec2_english();
        let (tokens, kept) = a.tokenize("AB C");
        let e = emissions_for(&tokens, 40, 0.02);
        let spans = forced_align(&e, &tokens, a.blank).expect("alignable");
        let base = words_from_spans(&spans, &kept, 0.02, 0.0);
        let shifted = words_from_spans(&spans, &kept, 0.02, 10.0);
        for (b, s) in base.iter().zip(shifted.iter()) {
            assert!((s.start - b.start - 10.0).abs() < 1e-9);
            assert!((s.end - b.end - 10.0).abs() < 1e-9);
        }
    }

    #[test]
    fn align_words_end_to_end() {
        let a = CtcAlphabet::wav2vec2_english();
        let (tokens, _) = a.tokenize("HELLO WORLD");
        let e = emissions_for(&tokens, 40, 0.02);
        let words = align_words(&e, "Hello, world!", &a, 0.0).expect("aligns");
        assert_eq!(
            words.iter().map(|w| w.value.as_str()).collect::<Vec<_>>(),
            ["HELLO", "WORLD"]
        );
        assert!(words[0].confidence.is_some());
    }
}
