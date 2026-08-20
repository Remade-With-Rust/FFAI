//! Voice activity detection — find speech, so silence never reaches the model.
//!
//! **Why this exists.** Whisper hallucinates on silence: fed an empty window
//! it emits fluent text (`you`, `Thank you.`). Mercury gates that downstream
//! by reading `P(<|nospeech|>)` after decoding (see [`super::decoder`]), which
//! works but pays for a full encoder pass and a decode over audio containing
//! nothing. VAD is the upstream answer — segment on speech first, and the
//! model never sees the silence at all.
//!
//! **What this detector is, honestly.** An energy detector with an adaptive
//! noise floor. It is not a learned model and does not pretend to be one: it
//! separates speech from silence and from steady room tone reliably, and it
//! will lose to a trained VAD on non-stationary noise — a café, a passing
//! car, music under speech.
//!
//! That is a deliberate staging choice, not an oversight. The pipeline, the
//! flag surface and the segment contract are the hard-to-change parts; the
//! detector behind them is one function. Phase B swaps in a Silero-class model
//! (see [`mercury-X-mission.md`](../../../../docs/mercury-X-mission.md) §B)
//! behind this exact signature. Building the pipeline against a model that
//! does not exist yet in candle — `candle-transformers` 0.11 ships no VAD, no
//! wav2vec2, and no speaker model — would have blocked every downstream stage
//! on a port.
//!
//! Frame geometry matches the mel front-end (25 ms window, 10 ms hop) so
//! region boundaries land on spectrogram frames rather than between them.

use ffai_core::types::TimedSegment;

use super::mel::{HOP_LENGTH, N_FFT, SAMPLE_RATE};

/// Default speech threshold. Mid-scale: see [`margin_db`] for what it means
/// to this detector.
pub const DEFAULT_THRESHOLD: f32 = 0.5;

/// Default packing width, in seconds. Whisper's context is 30 s and the
/// encoder processes that whether or not it is full, so there is nothing to
/// gain by packing tighter.
pub const DEFAULT_CHUNK_SECS: f32 = 30.0;

/// Speech shorter than this is a click, a breath, or a door — not an
/// utterance worth an encoder pass.
const MIN_SPEECH_MS: f64 = 250.0;

/// Gaps shorter than this are within-utterance pauses, not boundaries.
/// Bridging them first stops a single sentence fragmenting into a dozen
/// regions at every inter-word gap.
const MIN_SILENCE_MS: f64 = 300.0;

/// Widen each region by this much. Speech onsets ramp, and an energy
/// threshold necessarily fires a little late and releases a little early;
/// without padding the detector clips plosives and word-final fricatives,
/// which is a WER regression rather than a saving.
const SPEECH_PAD_MS: f64 = 150.0;

/// The noise floor is the 10th percentile of frame energy — robust to both a
/// clip that is mostly silence and one that is mostly speech, where a mean
/// would be dragged to the majority class.
const FLOOR_PERCENTILE: f64 = 0.10;

/// No frame below this is ever speech, whatever the adaptive floor says.
///
/// This guard is load-bearing. On digital silence every frame sits near
/// -100 dBFS, so the floor lands there too and `floor + margin` becomes
/// -90 dBFS — a threshold that dither or a stray denormal can cross. The
/// absolute bar makes pure silence unconditionally silent.
const ABS_SILENCE_DBFS: f32 = -70.0;

/// Energy floor for the dB conversion, so a digitally-silent frame is a
/// finite number rather than `-inf` polluting the percentile sort.
const MIN_DBFS: f32 = -100.0;

/// A packed window of audio to hand the encoder: contiguous, and known to
/// contain speech.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadWindow {
    pub start: f64,
    pub end: f64,
}

/// Map the user-facing 0..1 threshold onto this detector's actual knob: how
/// far above the noise floor a frame must sit to count as speech.
///
/// The mapping is stated rather than hidden because the two quantities are
/// not the same kind of thing. A learned VAD's threshold is a probability;
/// this one is a decibel margin. Phase B's model replaces this function with
/// a real probability and the flag keeps its meaning.
const fn margin_db(threshold: f32) -> f32 {
    16.0f32.mul_add(threshold.clamp(0.0, 1.0), 2.0)
}

/// Per-frame RMS energy in dBFS, on the mel front-end's frame geometry.
fn frame_energies(samples: &[f32]) -> Vec<f32> {
    if samples.len() < N_FFT {
        return Vec::new();
    }
    let n = (samples.len() - N_FFT) / HOP_LENGTH + 1;
    (0..n)
        .map(|i| {
            let frame = &samples[i * HOP_LENGTH..i * HOP_LENGTH + N_FFT];
            let sum_sq: f32 = frame.iter().map(|s| s * s).sum();
            let rms = (sum_sq / N_FFT as f32).sqrt();
            if rms <= 0.0 {
                MIN_DBFS
            } else {
                (20.0 * rms.log10()).max(MIN_DBFS)
            }
        })
        .collect()
}

// The index cast is guarded twice: the `is_empty` early return below makes
// `len - 1` safe, and the result is `.min(len - 1)` clamped before indexing, so
// a NaN or out-of-range `p` cannot escape the slice. Rust also saturates
// float->int casts rather than wrapping.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn percentile(sorted: &[f32], p: f64) -> f32 {
    if sorted.is_empty() {
        return MIN_DBFS;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Speech-to-floor energy contrast in dB: how far the loud frames (p90) sit
/// above the noise floor (p10). Clean recordings show large contrast —
/// speech against near-silence; noisy channels compress it, because the
/// "silence" is noise. One pass over the samples, microseconds against the
/// encoder pass it helps route, so the adaptive-context dispatcher can send
/// low-contrast (noisy) windows straight to the full 30 s context without
/// paying for a doomed short-context attempt first.
#[must_use]
pub fn energy_contrast_db(samples: &[f32]) -> f32 {
    let energies = frame_energies(samples);
    if energies.is_empty() {
        return 0.0;
    }
    let mut sorted = energies;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile(&sorted, 0.90) - percentile(&sorted, FLOOR_PERCENTILE)
}

/// Find speech regions in 16 kHz mono samples.
///
/// Returns non-overlapping, ascending regions. An empty result means "no
/// speech here" and is a valid, common answer — silence, room tone, and a
/// closed microphone all produce it.
#[must_use]
pub fn detect(samples: &[f32], threshold: f32) -> Vec<TimedSegment<()>> {
    let energies = frame_energies(samples);
    if energies.is_empty() {
        return Vec::new();
    }

    let mut sorted = energies.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = percentile(&sorted, FLOOR_PERCENTILE);

    let margin = margin_db(threshold);
    let onset = (floor + margin).max(ABS_SILENCE_DBFS);
    // Hysteresis: release lower than we trigger, so a frame hovering at the
    // boundary does not chop one utterance into a stutter of regions.
    let offset = margin.mul_add(0.6, floor).max(ABS_SILENCE_DBFS);

    let secs_per_frame = HOP_LENGTH as f64 / SAMPLE_RATE as f64;
    let total_secs = samples.len() as f64 / SAMPLE_RATE as f64;

    // Hysteresis scan.
    let mut regions: Vec<(f64, f64)> = Vec::new();
    let mut in_speech = false;
    let mut start_frame = 0usize;
    for (i, &db) in energies.iter().enumerate() {
        if !in_speech && db > onset {
            in_speech = true;
            start_frame = i;
        } else if in_speech && db < offset {
            in_speech = false;
            regions.push((
                start_frame as f64 * secs_per_frame,
                i as f64 * secs_per_frame,
            ));
        }
    }
    if in_speech {
        regions.push((start_frame as f64 * secs_per_frame, total_secs));
    }
    if regions.is_empty() {
        return Vec::new();
    }

    // Bridge short gaps BEFORE dropping short regions: two halves of one word
    // separated by a stop consonant must become one region and survive, not
    // be discarded separately for being individually too short.
    let min_silence = MIN_SILENCE_MS / 1000.0;
    let mut bridged: Vec<(f64, f64)> = vec![regions[0]];
    for &(s, e) in &regions[1..] {
        let last = bridged.last_mut().expect("seeded above");
        if s - last.1 < min_silence {
            last.1 = e;
        } else {
            bridged.push((s, e));
        }
    }

    let min_speech = MIN_SPEECH_MS / 1000.0;
    let pad = SPEECH_PAD_MS / 1000.0;
    let mut out: Vec<TimedSegment<()>> = Vec::new();
    for (s, e) in bridged.into_iter().filter(|(s, e)| e - s >= min_speech) {
        let s = (s - pad).max(0.0);
        let e = (e + pad).min(total_secs);
        // Padding can make neighbours touch; keep the contract that regions
        // are non-overlapping.
        match out.last_mut() {
            Some(prev) if s <= prev.end => prev.end = e.max(prev.end),
            _ => out.push(TimedSegment {
                start: s,
                end: e,
                value: (),
                confidence: None,
            }),
        }
    }
    out
}

/// Pack speech regions into contiguous windows of at most `chunk_secs`.
///
/// Follows `WhisperX`'s `merge_chunks`: a window closes when *adding* the next
/// region would make it exceed `chunk_secs`, so regions are packed together
/// rather than cut at a fixed grid. A region longer than `chunk_secs` — a
/// monologue with no pause — is split, because the encoder's context is fixed
/// and a longer window cannot be represented.
///
/// Windows are **contiguous spans**, not spliced speech: a gap shorter than
/// the packing width stays inside its window and does get encoded. Splicing
/// silence out would compress harder, but it needs a timestamp remap and it
/// feeds Whisper discontinuities it was never trained on. The saving here is
/// the silence that falls *between* windows, which is where the long tails
/// actually are.
#[must_use]
pub fn pack(regions: &[TimedSegment<()>], chunk_secs: f64) -> Vec<VadWindow> {
    let chunk = if chunk_secs > 0.0 {
        chunk_secs
    } else {
        f64::from(DEFAULT_CHUNK_SECS)
    };
    let mut out = Vec::new();
    let mut cur: Option<(f64, f64)> = None;

    for region in regions {
        let mut start = region.start;
        // `while` rather than `if`: a region longer than one window yields
        // several pieces, and each is packed by the same rule.
        loop {
            let piece_end = (start + chunk).min(region.end);
            match cur {
                None => cur = Some((start, piece_end)),
                Some((cs, ce)) => {
                    if piece_end - cs > chunk {
                        out.push(VadWindow { start: cs, end: ce });
                        cur = Some((start, piece_end));
                    } else {
                        cur = Some((cs, piece_end));
                    }
                }
            }
            start = piece_end;
            if start >= region.end {
                break;
            }
        }
    }
    if let Some((s, e)) = cur {
        out.push(VadWindow { start: s, end: e });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(secs: f64) -> Vec<f32> {
        vec![0.0; (secs * SAMPLE_RATE as f64) as usize]
    }

    /// Deterministic band-limited buzz — stands in for speech energy without
    /// needing an audio fixture on disk.
    fn tone(secs: f64, amp: f32) -> Vec<f32> {
        let n = (secs * SAMPLE_RATE as f64) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                amp * ((2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * 700.0 * t).sin())
            })
            .collect()
    }

    #[test]
    fn digital_silence_has_no_speech() {
        assert!(detect(&silence(5.0), DEFAULT_THRESHOLD).is_empty());
    }

    #[test]
    fn steady_room_tone_has_no_speech() {
        // Uniform low-level noise: the adaptive floor sits at the tone's own
        // level, so nothing clears the margin. This is the case a fixed
        // threshold gets wrong.
        let n = (5.0 * SAMPLE_RATE as f64) as usize;
        let mut rng = 12345u64;
        let hiss: Vec<f32> = (0..n)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                ((rng >> 11) as f32 / (1u64 << 53) as f32 - 0.5) * 0.002
            })
            .collect();
        assert!(detect(&hiss, DEFAULT_THRESHOLD).is_empty());
    }

    #[test]
    fn finds_speech_between_silences() {
        let mut audio = silence(2.0);
        audio.extend(tone(1.5, 0.3));
        audio.extend(silence(2.0));
        let regions = detect(&audio, DEFAULT_THRESHOLD);
        assert_eq!(regions.len(), 1, "expected one region, got {regions:?}");
        // Padded, so bounds are near the tone rather than exactly on it.
        assert!(
            regions[0].start < 2.0 && regions[0].start > 1.5,
            "{regions:?}"
        );
        assert!(regions[0].end > 3.5 && regions[0].end < 4.0, "{regions:?}");
    }

    #[test]
    fn brief_gap_does_not_split_one_utterance() {
        let mut audio = silence(1.0);
        audio.extend(tone(1.0, 0.3));
        audio.extend(silence(0.1)); // shorter than MIN_SILENCE_MS
        audio.extend(tone(1.0, 0.3));
        audio.extend(silence(1.0));
        assert_eq!(detect(&audio, DEFAULT_THRESHOLD).len(), 1);
    }

    #[test]
    fn long_gap_does_split() {
        let mut audio = silence(1.0);
        audio.extend(tone(1.0, 0.3));
        audio.extend(silence(2.0));
        audio.extend(tone(1.0, 0.3));
        audio.extend(silence(1.0));
        assert_eq!(detect(&audio, DEFAULT_THRESHOLD).len(), 2);
    }

    fn seg(start: f64, end: f64) -> TimedSegment<()> {
        TimedSegment {
            start,
            end,
            value: (),
            confidence: None,
        }
    }

    #[test]
    fn pack_merges_regions_inside_one_window() {
        let out = pack(&[seg(0.0, 2.0), seg(25.0, 27.0)], 30.0);
        assert_eq!(
            out,
            vec![VadWindow {
                start: 0.0,
                end: 27.0
            }]
        );
    }

    #[test]
    fn pack_separates_regions_further_apart_than_a_window() {
        // The 38 s of silence between them is never encoded — the point of
        // the whole exercise.
        let out = pack(&[seg(0.0, 2.0), seg(40.0, 42.0)], 30.0);
        assert_eq!(
            out,
            vec![
                VadWindow {
                    start: 0.0,
                    end: 2.0
                },
                VadWindow {
                    start: 40.0,
                    end: 42.0
                }
            ]
        );
    }

    #[test]
    fn pack_splits_a_region_longer_than_a_window() {
        let out = pack(&[seg(0.0, 70.0)], 30.0);
        assert_eq!(out.len(), 3);
        assert!(
            out.iter().all(|w| w.end - w.start <= 30.0 + 1e-9),
            "{out:?}"
        );
        assert_eq!(out[0].start, 0.0);
        assert_eq!(out[2].end, 70.0);
    }

    #[test]
    fn pack_of_nothing_is_nothing() {
        assert!(pack(&[], 30.0).is_empty());
    }
}
