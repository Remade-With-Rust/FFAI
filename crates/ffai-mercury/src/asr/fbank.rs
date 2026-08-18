//! SpeechBrain-compatible filterbank features, for [`super::speaker`].
//!
//! **This is not [`super::mel`].** Whisper's log-mel and `SpeechBrain`'s Fbank
//! are both "80-dim mel features" and they differ in four places, every one
//! of which changes the numbers:
//!
//! | | Whisper (`mel.rs`) | `SpeechBrain` (here) |
//! |---|---|---|
//! | window | Hann | **Hamming** |
//! | STFT padding | reflect | **constant (zeros)** |
//! | mel filters | Slaney-normalised | **unnormalised triangles** |
//! | log | `log10`, clamp to `max-8`, `(x+4)/4` | **`10·log10`, 80 dB floor** |
//!
//! Feeding Whisper's mel to an ECAPA-TDNN produces embeddings that are
//! finite, well-shaped, and cluster beautifully — into groups that have
//! nothing to do with who was speaking. There is no error to catch it, which
//! is why this module exists rather than a parameter on the existing one.
//!
//! Transcribed from `speechbrain/processing/features.py`: `STFT` defaults
//! (`window_fn=torch.hamming_window`, `center=True`, `pad_mode="constant"`)
//! and `Filterbank` defaults (`power_spectrogram=2`, `log_mel=True`,
//! `amin=1e-10`, `ref_value=1.0`, `top_db=80`, `filter_shape="triangular"`).

use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

pub const SAMPLE_RATE: usize = 16_000;
/// 25 ms at 16 kHz.
pub const N_FFT: usize = 400;
/// 10 ms at 16 kHz.
pub const HOP_LENGTH: usize = 160;

const AMIN: f32 = 1e-10;
const TOP_DB: f32 = 80.0;
/// `power_spectrogram = 2` selects the power spectrum, and `SpeechBrain` pairs
/// that with a multiplier of 10 (20 is for magnitude).
const DB_MULTIPLIER: f32 = 10.0;

fn to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Periodic Hamming window, matching `torch.hamming_window(n)` whose
/// `periodic` argument defaults to true.
///
/// The symmetric variant divides by `n - 1` instead of `n`. The difference is
/// one sample of taper and it is enough to move every coefficient.
fn hamming(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            0.46f32.mul_add(
                -(2.0 * std::f32::consts::PI * i as f32 / n as f32).cos(),
                0.54,
            )
        })
        .collect()
}

/// Unnormalised triangular mel filterbank.
///
/// `SpeechBrain` builds `n_mels + 2` mel-spaced points, takes the inner
/// `n_mels` as centres and the successive differences as bandwidths, then
/// forms `max(0, min(slope + 1, -slope + 1))` with `slope = (f - centre) /
/// band`. Deliberately **not** area-normalised: dividing by filter width here
/// (as Whisper's Slaney filters do) rescales every band and shifts the whole
/// feature distribution the network was trained on.
fn filterbank(n_mels: usize, n_stft: usize, f_min: f32, f_max: f32) -> Vec<Vec<f32>> {
    let mel_lo = to_mel(f_min);
    let mel_hi = to_mel(f_max);
    let points: Vec<f32> = (0..n_mels + 2)
        .map(|i| to_hz(mel_lo + (mel_hi - mel_lo) * i as f32 / (n_mels + 1) as f32))
        .collect();

    // `band[i] = points[i+1] - points[i]`, then drop the last; centres are
    // the interior points. Both end up with exactly `n_mels` entries.
    let bands: Vec<f32> = (0..n_mels).map(|i| points[i + 1] - points[i]).collect();
    let centres: Vec<f32> = points[1..=n_mels].to_vec();

    let all_freqs: Vec<f32> = (0..n_stft)
        .map(|i| (SAMPLE_RATE / 2) as f32 * i as f32 / (n_stft - 1) as f32)
        .collect();

    (0..n_mels)
        .map(|m| {
            let (centre, band) = (centres[m], bands[m].max(f32::EPSILON));
            all_freqs
                .iter()
                .map(|&f| {
                    let slope = (f - centre) / band;
                    (slope + 1.0).min(-slope + 1.0).max(0.0)
                })
                .collect()
        })
        .collect()
}

/// The feature extractor. Build once, reuse — it owns an FFT plan and the
/// filterbank.
pub struct Fbank {
    n_mels: usize,
    window: Vec<f32>,
    filters: Vec<Vec<f32>>,
    fft: Arc<dyn Fft<f32>>,
}

impl Fbank {
    #[must_use]
    pub fn new(n_mels: usize) -> Self {
        let n_stft = N_FFT / 2 + 1;
        Self {
            n_mels,
            window: hamming(N_FFT),
            filters: filterbank(n_mels, n_stft, 0.0, (SAMPLE_RATE / 2) as f32),
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
        }
    }

    #[must_use]
    pub const fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Frames produced for `n` samples, with centred framing.
    #[must_use]
    pub const fn n_frames(n: usize) -> usize {
        n / HOP_LENGTH + 1
    }

    /// `samples` → `(features, frames)`, row-major `frames × n_mels`.
    ///
    /// Applies sentence-level mean normalisation, which is `SpeechBrain`'s
    /// `InputNormalization(norm_type="sentence", std_norm=False)` — subtract
    /// the per-utterance mean of each feature, do **not** divide by the
    /// standard deviation. Skipping it leaves a channel offset the network
    /// never saw in training.
    pub fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let frames = Self::n_frames(samples.len());
        if frames == 0 {
            return (Vec::new(), 0);
        }
        // `center=True` with `pad_mode="constant"`: zero-pad by n_fft/2 each
        // side so frame t is centred at t*hop. Whisper reflect-pads here.
        let pad = N_FFT / 2;
        let mut padded = vec![0.0f32; samples.len() + 2 * pad];
        padded[pad..pad + samples.len()].copy_from_slice(samples);

        let n_stft = N_FFT / 2 + 1;
        let mut out = vec![0.0f32; frames * self.n_mels];
        let mut scratch = vec![Complex32::new(0.0, 0.0); N_FFT];
        let mut power = vec![0.0f32; n_stft];

        for frame in 0..frames {
            let start = frame * HOP_LENGTH;
            for (i, slot) in scratch.iter_mut().enumerate() {
                let v = padded.get(start + i).copied().unwrap_or(0.0) * self.window[i];
                *slot = Complex32::new(v, 0.0);
            }
            self.fft.process(&mut scratch);
            for (bin, p) in power.iter_mut().enumerate() {
                // power_spectrogram = 2: magnitude squared.
                *p = scratch[bin].norm_sqr();
            }
            for (m, filter) in self.filters.iter().enumerate() {
                let energy: f32 = filter.iter().zip(power.iter()).map(|(w, p)| w * p).sum();
                out[frame * self.n_mels + m] = DB_MULTIPLIER * energy.max(AMIN).log10();
            }
        }

        // top_db: nothing more than 80 dB below the loudest value survives,
        // so a near-silent frame cannot dominate the mean subtraction below.
        let peak = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if peak.is_finite() {
            let floor = peak - TOP_DB;
            for v in &mut out {
                *v = v.max(floor);
            }
        }

        // Sentence-level mean normalisation, per feature dimension.
        for m in 0..self.n_mels {
            let mut sum = 0.0f32;
            for f in 0..frames {
                sum += out[f * self.n_mels + m];
            }
            let mean = sum / frames as f32;
            for f in 0..frames {
                out[f * self.n_mels + m] -= mean;
            }
        }

        (out, frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_scale_roundtrips() {
        for hz in [0.0f32, 100.0, 1000.0, 4000.0, 8000.0] {
            assert!((to_hz(to_mel(hz)) - hz).abs() < 1e-2, "{hz}");
        }
    }

    #[test]
    fn hamming_is_periodic_not_symmetric() {
        // The two variants differ only in the divisor — `n` vs `n - 1` — so at
        // n = 400 the endpoints differ by 6e-5 and no endpoint check
        // distinguishes them reliably. At n = 4 the difference is structural:
        //   periodic  (÷4): [0.08, 0.54, 1.00, 0.54]  — peaks at exactly 1
        //   symmetric (÷3): [0.08, 0.77, 0.77, 0.08]  — never reaches 1
        let w = hamming(4);
        assert!((w[0] - 0.08).abs() < 1e-6, "{w:?}");
        assert!((w[2] - 1.0).abs() < 1e-6, "not periodic: {w:?}");
        assert!((w[1] - 0.54).abs() < 1e-6, "not periodic: {w:?}");
    }

    #[test]
    fn filters_are_not_area_normalised() {
        // Whisper's Slaney filters sum to a constant area; these peak at 1.0
        // regardless of width. Conflating them is the failure this guards.
        let filters = filterbank(80, N_FFT / 2 + 1, 0.0, 8000.0);
        assert_eq!(filters.len(), 80);
        // A triangle reaches exactly 1.0 only where a bin lands on the centre
        // frequency, which discrete bins essentially never do — so this is a
        // "close to the apex" check, not an equality one.
        let peak = filters[40].iter().copied().fold(0.0f32, f32::max);
        assert!(
            (0.8..=1.0).contains(&peak),
            "peak should approach 1.0, got {peak}"
        );
        // Wide (high-frequency) filters must have MORE total weight than
        // narrow ones — which is exactly what normalisation would remove.
        let low: f32 = filters[5].iter().sum();
        let high: f32 = filters[70].iter().sum();
        assert!(high > low * 2.0, "low {low} high {high}");
    }

    #[test]
    fn filters_are_non_negative_and_cover_the_band() {
        let filters = filterbank(80, N_FFT / 2 + 1, 0.0, 8000.0);
        assert!(filters.iter().flatten().all(|v| *v >= 0.0));
        // Every filter must have some support, or a feature dimension is dead.
        for (i, f) in filters.iter().enumerate() {
            assert!(f.iter().any(|v| *v > 0.0), "filter {i} is empty");
        }
    }

    #[test]
    fn frame_count_follows_the_hop() {
        let fb = Fbank::new(80);
        let (feats, frames) = fb.compute(&vec![0.0; SAMPLE_RATE]);
        assert_eq!(frames, Fbank::n_frames(SAMPLE_RATE));
        assert_eq!(feats.len(), frames * 80);
        assert!(
            frames >= 100,
            "1 s at 10 ms hop should be ~101 frames, got {frames}"
        );
    }

    #[test]
    fn mean_normalisation_centres_each_dimension() {
        let fb = Fbank::new(80);
        let n = SAMPLE_RATE / 2;
        let tone: Vec<f32> = (0..n)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.5
            })
            .collect();
        let (feats, frames) = fb.compute(&tone);
        for m in 0..80 {
            let mean: f32 = (0..frames).map(|f| feats[f * 80 + m]).sum::<f32>() / frames as f32;
            assert!(mean.abs() < 1e-3, "dimension {m} mean {mean}");
        }
    }

    #[test]
    fn a_tone_concentrates_energy_in_a_few_bands() {
        let fb = Fbank::new(80);
        let n = SAMPLE_RATE / 2;
        let tone: Vec<f32> = (0..n)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.5
            })
            .collect();
        let (feats, frames) = fb.compute(&tone);
        // Look at a mid-utterance frame; find its argmax band.
        let f = frames / 2;
        let row = &feats[f * 80..(f + 1) * 80];
        let peak = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("non-empty")
            .0;
        // 1 kHz sits in the lower-middle of an 80-band 0-8 kHz mel scale.
        assert!((20..55).contains(&peak), "1 kHz landed in band {peak}");
    }

    #[test]
    fn silence_does_not_produce_nan() {
        let fb = Fbank::new(80);
        let (feats, _) = fb.compute(&vec![0.0; SAMPLE_RATE / 10]);
        assert!(feats.iter().all(|v| v.is_finite()), "NaN or inf on silence");
    }
}
