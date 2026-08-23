//! Whisper's audio front-end: PCM → log-mel spectrogram.
//!
//! This is Mercury's first independent stage (mission plan §2). It owns the
//! STFT and the mel filterbank outright rather than borrowing them, because
//! the front-end is where streaming and SIMD work will land later, and
//! because a stage we own is a stage we can oracle-test on its own — a WER
//! regression should be attributable to *one* box in minutes.
//!
//! It reproduces `whisper/audio.py::log_mel_spectrogram` exactly:
//!
//! ```text
//! hann(400, periodic) → STFT(n_fft=400, hop=160, center, reflect-pad)
//!   → |X|² (dropping the final frame, as torch's `stft[..., :-1]`)
//!   → slaney mel filterbank (librosa defaults: htk=false, norm=slaney)
//!   → log10, clamped at 1e-10
//!   → floored at (global max − 8)
//!   → (x + 4) / 4
//! ```
//!
//! Verified against openai-whisper's own output by
//! `tests/oracle_mel.rs` — see `docs/mercury-mission-plan.md` §6.2.

use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

/// Whisper operates exclusively at 16 kHz.
pub const SAMPLE_RATE: usize = 16_000;
/// STFT window length in samples (25 ms).
pub const N_FFT: usize = 400;
/// STFT hop in samples (10 ms).
pub const HOP_LENGTH: usize = 160;
/// The model's fixed context window, in seconds.
pub const CHUNK_SECONDS: usize = 30;
/// Samples in one context window.
pub const N_SAMPLES: usize = SAMPLE_RATE * CHUNK_SECONDS;
/// Mel frames in one context window.
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH;

/// Convert Hz to the Slaney mel scale (librosa's `htk=False` default:
/// linear below 1 kHz, logarithmic above).
fn hz_to_mel(freq: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    if freq >= MIN_LOG_HZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() / (6.4f32.ln() / 27.0)
    } else {
        freq / F_SP
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * (6.4f32.ln() / 27.0)).exp()
    } else {
        mel * F_SP
    }
}

/// A mel filterbank: `n_mels` triangular filters over `N_FFT/2 + 1` bins,
/// row-major.
#[derive(Debug, Clone)]
pub struct MelFilters {
    pub n_mels: usize,
    pub n_bins: usize,
    pub(crate) weights: Vec<f32>,
}

impl MelFilters {
    /// Build the librosa-default filterbank (`htk=False`, `norm="slaney"`),
    /// which is what Whisper ships in `mel_filters.npz`.
    #[must_use]
    pub fn new(n_mels: usize) -> Self {
        let n_bins = N_FFT / 2 + 1;
        let nyquist = SAMPLE_RATE as f32 / 2.0;
        let fft_freqs: Vec<f32> = (0..n_bins)
            .map(|i| i as f32 * SAMPLE_RATE as f32 / N_FFT as f32)
            .collect();

        // n_mels + 2 band edges, evenly spaced on the mel scale.
        let (mel_min, mel_max) = (hz_to_mel(0.0), hz_to_mel(nyquist));
        let edges: Vec<f32> = (0..n_mels + 2)
            .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
            .collect();

        let mut weights = vec![0.0f32; n_mels * n_bins];
        for m in 0..n_mels {
            let (left, center, right) = (edges[m], edges[m + 1], edges[m + 2]);
            // Slaney normalization: equal AREA per filter, not equal peak.
            let enorm = 2.0 / (right - left);
            for (k, &freq) in fft_freqs.iter().enumerate() {
                let rising = (freq - left) / (center - left);
                let falling = (right - freq) / (right - center);
                weights[m * n_bins + k] = enorm * rising.min(falling).max(0.0);
            }
        }
        Self {
            n_mels,
            n_bins,
            weights,
        }
    }

    fn row(&self, m: usize) -> &[f32] {
        &self.weights[m * self.n_bins..(m + 1) * self.n_bins]
    }
}

/// The log-mel front-end. Construct once and reuse: it owns the FFT plan and
/// the filterbank, both of which are expensive to rebuild per call.
pub struct MelSpectrogram {
    filters: MelFilters,
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl MelSpectrogram {
    #[must_use]
    pub fn new(n_mels: usize) -> Self {
        // torch.hann_window defaults to periodic=True: denominator N, not N-1.
        let window: Vec<f32> = (0..N_FFT)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32).cos()))
            .collect();
        Self {
            filters: MelFilters::new(n_mels),
            window,
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
        }
    }

    #[must_use]
    pub const fn n_mels(&self) -> usize {
        self.filters.n_mels
    }

    /// The original scalar projection, kept as the fallback and the oracle:
    /// if a tensor cannot be built the front end must still produce identical
    /// output rather than fail.
    fn filterbank_scalar(filters: &MelFilters, power: &[f32], n_frames: usize) -> Vec<f32> {
        let mut mel = vec![0.0f32; filters.n_mels * n_frames];
        for m in 0..filters.n_mels {
            let filter = filters.row(m);
            for frame in 0..n_frames {
                let mut sum = 0.0f32;
                for (bin, &w) in filter.iter().enumerate() {
                    sum += w * power[bin * n_frames + frame];
                }
                mel[m * n_frames + frame] = sum;
            }
        }
        mel
    }

    /// Frames produced for `n_samples` of audio: `n_samples / HOP_LENGTH`.
    #[must_use]
    pub const fn n_frames(n_samples: usize) -> usize {
        n_samples / HOP_LENGTH
    }

    /// Compute the log-mel spectrogram: `(n_mels, n_frames)` row-major.
    #[must_use]
    pub fn compute(&self, samples: &[f32]) -> MelChunk {
        let n_frames = Self::n_frames(samples.len());
        let n_bins = self.filters.n_bins;

        // center=True with reflect padding, matching torch.stft.
        let padded = reflect_pad(samples, N_FFT / 2);

        // Power spectrum stored BIN-MAJOR, (n_bins, n_frames). The natural
        // frame-major layout would force a transpose before the filterbank
        // matmul below; writing it transposed here costs nothing, since the
        // FFT output is scattered either way.
        let mut power = vec![0.0f32; n_bins * n_frames];
        let mut scratch = vec![Complex32::new(0.0, 0.0); N_FFT];
        for frame in 0..n_frames {
            let start = frame * HOP_LENGTH;
            for i in 0..N_FFT {
                scratch[i] = Complex32::new(padded[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut scratch);
            for bin in 0..n_bins {
                power[bin * n_frames + frame] = scratch[bin].norm_sqr();
            }
        }

        // Filterbank projection: (n_mels, n_bins) @ (n_bins, n_frames).
        //
        // This is a matmul, and it was hand-written as a scalar triple loop —
        // 87 % of all mel time at 5.1 GFLOP/s, against the ~600 GFLOP/s the
        // tuned GEMM reaches. Handing it to candle turns the single largest
        // cost in the front end into one of the smallest. A hot loop that is
        // secretly a matmul is worth looking for anywhere it appears.
        let n_mels = self.filters.n_mels;
        let device = ffai_core::candle::Device::Cpu;
        let mel = match (
            ffai_core::candle::Tensor::from_slice(&self.filters.weights, (n_mels, n_bins), &device),
            ffai_core::candle::Tensor::from_slice(&power, (n_bins, n_frames), &device),
        ) {
            (Ok(f), Ok(p)) => f
                .matmul(&p)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .unwrap_or_else(|_| Self::filterbank_scalar(&self.filters, &power, n_frames)),
            _ => Self::filterbank_scalar(&self.filters, &power, n_frames),
        };
        let mut mel = mel;

        // log10 with a floor, then a dynamic-range clamp 8 decades below the
        // GLOBAL peak, then scale into roughly [-1, 1].
        let mut peak = f32::MIN;
        for v in &mut mel {
            // `fastmath::log10` — one libm call per mel bin per frame
            // otherwise, and the loop cannot vectorise around a call.
            *v = ffai_core::fastmath::log10(v.max(1e-10));
            peak = peak.max(*v);
        }
        let floor = peak - 8.0;
        for v in &mut mel {
            *v = (v.max(floor) + 4.0) / 4.0;
        }

        MelChunk {
            n_mels,
            n_frames,
            data: mel,
        }
    }
}

/// A computed log-mel spectrogram, `(n_mels, n_frames)` row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct MelChunk {
    pub n_mels: usize,
    pub n_frames: usize,
    pub data: Vec<f32>,
}

impl MelChunk {
    /// Pad or truncate to `frames`.
    #[must_use]
    pub fn resized(&self, frames: usize) -> Self {
        let mut data = vec![0.0f32; self.n_mels * frames];
        for m in 0..self.n_mels {
            let take = self.n_frames.min(frames);
            data[m * frames..m * frames + take]
                .copy_from_slice(&self.data[m * self.n_frames..m * self.n_frames + take]);
        }
        Self {
            n_mels: self.n_mels,
            n_frames: frames,
            data,
        }
    }
}

/// Zero-pad or truncate audio to exactly one context window, Whisper's
/// `pad_or_trim`.
///
/// Padding must happen HERE, in the sample domain, not on the finished
/// spectrogram. Two things go wrong otherwise: the pad value in log-mel space
/// is the floor, not 0.0, and the dynamic-range clamp in [`MelSpectrogram::compute`]
/// takes a *global* max — computing it over only the speech frames and then
/// appending padding normalizes the window differently than Whisper does.
#[must_use]
pub fn pad_or_trim(samples: &[f32]) -> Vec<f32> {
    pad_or_trim_to(samples, N_SAMPLES)
}

/// Zero-pad or truncate to `target` samples.
#[must_use]
pub fn pad_or_trim_to(samples: &[f32], target: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; target];
    let take = samples.len().min(target);
    out[..take].copy_from_slice(&samples[..take]);
    out
}

// NOTE: a variable-length encoder window (trimming the 30 s context to the
// audio actually present) was implemented, measured, and PRUNED here — see
// mission plan §6.4. The encoder is O(n) and 69 % of its work is padding, so
// the arithmetic promised ~3.2x on the stage; in practice it destabilized the
// decoder into repetition loops and corpus WER went 3.00 % -> 268 %.
// A cheap, correct-looking change that a speed-only measurement would have
// shipped.

/// Reflect-pad both ends by `pad` samples (numpy/torch `mode="reflect"`:
/// the edge sample itself is not repeated).
fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let n = samples.len();
    // An empty signal has nothing to reflect, and BOTH loops below are unsound
    // for n == 0: `reflect_index` returns 0, which is not a valid index into an
    // empty slice, and `n - 1 - i.min(n - 1)` underflows. `compute` produces
    // zero frames for an empty buffer and never reads this padding, but
    // `AudioBuffer::samples` is caller-supplied and an empty one must not panic
    // the library. Silence of the padded length keeps the function TOTAL.
    //
    // Found by tests/properties.rs (gate H-28) on its first run; minimal
    // failing input was `samples = []`.
    if n == 0 {
        return vec![0.0; 2 * pad];
    }
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        out.push(samples[reflect_index(pad - i, n)]);
    }
    out.extend_from_slice(samples);
    for i in 1..=pad {
        out.push(samples[reflect_index(n - 1 - i.min(n - 1), n)]);
    }
    out
}

fn reflect_index(i: usize, n: usize) -> usize {
    if n == 0 { 0 } else { i.min(n - 1) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_scale_roundtrips_across_the_breakpoint() {
        for hz in [0.0f32, 100.0, 999.0, 1000.0, 1001.0, 4000.0, 8000.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((back - hz).abs() < 1e-2, "hz {hz} -> {back}");
        }
        // The scale is continuous at the 1 kHz linear/log breakpoint.
        assert!((hz_to_mel(999.999) - hz_to_mel(1000.0)).abs() < 1e-3);
    }

    #[test]
    fn filterbank_has_expected_shape_and_support() {
        let f = MelFilters::new(80);
        assert_eq!(f.n_mels, 80);
        assert_eq!(f.n_bins, N_FFT / 2 + 1);
        // Every filter must have some energy, and none may be negative.
        for m in 0..80 {
            let row = f.row(m);
            assert!(
                row.iter().all(|&w| w >= 0.0),
                "filter {m} has negative weight"
            );
            assert!(row.iter().any(|&w| w > 0.0), "filter {m} is empty");
        }
    }

    #[test]
    fn frame_count_matches_whisper_context_window() {
        assert_eq!(MelSpectrogram::n_frames(N_SAMPLES), N_FRAMES);
        assert_eq!(N_FRAMES, 3000);
    }

    #[test]
    fn reflect_pad_mirrors_without_repeating_the_edge() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let padded = reflect_pad(&x, 2);
        assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn silence_produces_the_floor_value_everywhere() {
        let mel = MelSpectrogram::new(80);
        let out = mel.compute(&vec![0.0f32; SAMPLE_RATE]);
        assert_eq!(out.n_mels, 80);
        assert_eq!(out.n_frames, 100);
        // Pure silence is uniform, so max == min and the clamp makes it flat.
        let first = out.data[0];
        assert!(out.data.iter().all(|v| (v - first).abs() < 1e-6));
    }

    #[test]
    fn a_tone_concentrates_energy_in_a_few_mel_bands() {
        let mel = MelSpectrogram::new(80);
        let samples: Vec<f32> = (0..SAMPLE_RATE)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let out = mel.compute(&samples);
        // Energy per mel band, averaged over frames.
        let band_energy: Vec<f32> = (0..out.n_mels)
            .map(|m| {
                out.data[m * out.n_frames..(m + 1) * out.n_frames]
                    .iter()
                    .sum::<f32>()
                    / out.n_frames as f32
            })
            .collect();
        let loudest = band_energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        // 440 Hz sits in the linear part of the mel scale: 440/(200/3) = 6.6
        // mel, and the band spacing puts it in the low bands.
        assert!(
            loudest < 20,
            "440 Hz landed in mel band {loudest}, expected a low band"
        );
    }

    #[test]
    fn pad_or_trim_produces_exactly_one_context_window() {
        assert_eq!(pad_or_trim(&[1.0, 2.0]).len(), N_SAMPLES);
        assert_eq!(pad_or_trim(&vec![1.0f32; N_SAMPLES * 2]).len(), N_SAMPLES);
        let padded = pad_or_trim(&[1.0, 2.0]);
        assert_eq!(&padded[..3], &[1.0, 2.0, 0.0]);
    }

    #[test]
    fn padding_audio_and_padding_mel_are_not_the_same_thing() {
        // The bug this guards: log-mel silence is the floor value, not 0.0,
        // and the dynamic-range clamp uses a global max. Padding the
        // spectrogram instead of the audio gets both wrong.
        let mel = MelSpectrogram::new(80);
        let speech: Vec<f32> = (0..SAMPLE_RATE)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let correct = mel.compute(&pad_or_trim(&speech));
        let wrong = mel.compute(&speech).resized(N_FRAMES);
        assert_eq!(correct.n_frames, N_FRAMES);
        let pad_region = N_FRAMES - 100;
        assert!(
            (correct.data[pad_region] - wrong.data[pad_region]).abs() > 0.1,
            "padding domain should change the padded region materially"
        );
    }

    #[test]
    fn resized_truncates_each_mel_row_independently() {
        let long = MelChunk {
            n_mels: 2,
            n_frames: 5,
            data: (0..10).map(|i| i as f32).collect(),
        };
        let cut = long.resized(3);
        assert_eq!(cut.data, vec![0.0, 1.0, 2.0, 5.0, 6.0, 7.0]);
    }
}
