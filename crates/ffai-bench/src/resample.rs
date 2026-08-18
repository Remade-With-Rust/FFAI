//! Harness-side resampling: every TTS implementation's audio is converted to
//! the judge's required format (16 kHz mono) by THIS code, identically, so
//! no implementation's score depends on its native sample rate or on some
//! external tool's resampler. This is measurement plumbing, not product —
//! engines still reject wrong-rate audio rather than silently resampling
//! (ffai-media's documented policy).
//!
//! The kernel is a Hann-windowed sinc (32 taps per side), cutoff at 45 % of
//! the lower Nyquist. Not audiophile-grade, but flat where speech lives and
//! ~70 dB of alias rejection — far below what a WER judge can hear. The
//! tests assert CONTENT (a tone survives with its frequency and amplitude),
//! not shapes, per the `audio_encoder.rs` lesson.

use ffai_core::types::AudioBuffer;

/// Taps per side of the sinc kernel.
const HALF_TAPS: usize = 32;

/// Convert to the round-trip judge's input format: mono, `dst_rate` Hz.
#[must_use]
pub fn to_judge_format(audio: &AudioBuffer, dst_rate: u32) -> AudioBuffer {
    let mono = audio.to_mono();
    if mono.sample_rate == dst_rate {
        return mono;
    }
    AudioBuffer {
        samples: resample(&mono.samples, mono.sample_rate, dst_rate),
        sample_rate: dst_rate,
        channels: 1,
    }
}

/// Windowed-sinc resampling of a mono signal.
fn resample(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    // Sample rates come from a file header, so neither is guaranteed non-zero.
    // `src_rate == 0` makes `ratio` 0, `len / 0` infinity, and `floor() as usize`
    // SATURATES to usize::MAX - which then becomes `Vec::with_capacity(usize::MAX)`
    // and aborts the process. `dst_rate == 0` silently produces an empty signal.
    // Neither is a meaningful resample, so refuse both up front.
    if src_rate == 0 || dst_rate == 0 {
        return Vec::new();
    }
    let ratio = f64::from(src_rate) / f64::from(dst_rate);
    // Anti-alias at the LOWER of the two Nyquists (in units of the source
    // rate), with margin for the finite kernel's transition band.
    let cutoff = 0.45 * (f64::from(dst_rate.min(src_rate)) / f64::from(src_rate));
    let out_len = ((samples.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);

    for n in 0..out_len {
        // Position of this output sample on the source timeline.
        let center = n as f64 * ratio;
        let left = (center.floor() as isize) - (HALF_TAPS as isize) + 1;
        let mut acc = 0.0f64;
        let mut norm = 0.0f64;
        for k in 0..(2 * HALF_TAPS) {
            let idx = left + k as isize;
            if idx < 0 || idx as usize >= samples.len() {
                continue;
            }
            let t = center - idx as f64; // distance in source samples
            let w = windowed_sinc(t, cutoff);
            acc += f64::from(samples[idx as usize]) * w;
            norm += w;
        }
        // Normalizing by the kernel sum keeps unity gain at DC and at the
        // edges where the kernel is truncated by the signal boundary.
        out.push(if norm.abs() > 1e-12 {
            (acc / norm) as f32
        } else {
            0.0
        });
    }
    out
}

/// `sinc(2·cutoff·t) · 2·cutoff`, Hann-windowed over ±`HALF_TAPS`.
fn windowed_sinc(t: f64, cutoff: f64) -> f64 {
    let x = 2.0 * cutoff * t;
    let sinc = if x.abs() < 1e-9 {
        1.0
    } else {
        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
    };
    // Hann window over the kernel's support.
    let w = 0.5 + 0.5 * (std::f64::consts::PI * t / HALF_TAPS as f64).cos();
    if t.abs() >= HALF_TAPS as f64 {
        0.0
    } else {
        2.0 * cutoff * sinc * w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f64, rate: u32, secs: f64) -> AudioBuffer {
        let n = (rate as f64 * secs) as usize;
        AudioBuffer {
            samples: (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / rate as f64).sin() as f32)
                .collect(),
            sample_rate: rate,
            channels: 1,
        }
    }

    /// Count sign changes — 2·freq·secs for a clean tone.
    fn zero_crossings(samples: &[f32]) -> usize {
        samples
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count()
    }

    fn rms(samples: &[f32]) -> f64 {
        (samples
            .iter()
            .map(|s| (*s as f64) * (*s as f64))
            .sum::<f64>()
            / samples.len() as f64)
            .sqrt()
    }

    #[test]
    fn a_1khz_tone_survives_22050_to_16000_with_frequency_and_amplitude() {
        // The exact conversion the judge pass performs on piper audio. A shape
        // assertion (right length, right rate) would pass on garbage; the
        // content assertions are the test.
        let src = tone(1000.0, 22_050, 1.0);
        let out = to_judge_format(&src, 16_000);
        assert_eq!(out.sample_rate, 16_000);
        assert_eq!(out.channels, 1);
        assert!(
            (out.samples.len() as f64 - 16_000.0).abs() < 8.0,
            "{}",
            out.samples.len()
        );

        // Frequency preserved: ~2000 zero crossings for 1 kHz over 1 s.
        let zc = zero_crossings(&out.samples);
        assert!(
            (1990..=2010).contains(&zc),
            "zero crossings {zc}, want ~2000"
        );

        // Amplitude preserved: RMS of a unit sine is 1/sqrt(2) ~ 0.707.
        let r = rms(&out.samples);
        assert!((r - 0.707).abs() < 0.02, "rms {r}, want ~0.707");
    }

    #[test]
    fn content_above_the_target_nyquist_is_rejected_not_aliased() {
        // 9 kHz is representable at 22.05 kHz but NOT at 16 kHz. A naive
        // (linear) resampler folds it back into the audible band as an alias
        // the judge would transcribe as noise; a correct one removes it.
        let src = tone(9_000.0, 22_050, 1.0);
        let out = to_judge_format(&src, 16_000);
        let r = rms(&out.samples);
        assert!(r < 0.05, "9 kHz tone should be suppressed, rms {r}");
    }

    #[test]
    fn stereo_is_downmixed_and_same_rate_passes_through() {
        let mono = tone(440.0, 16_000, 0.5);
        let stereo = AudioBuffer {
            samples: mono.samples.iter().flat_map(|s| [*s, *s]).collect(),
            sample_rate: 16_000,
            channels: 2,
        };
        let out = to_judge_format(&stereo, 16_000);
        assert_eq!(out.channels, 1);
        assert_eq!(out.samples.len(), mono.samples.len());
        // Identical L/R halves must average back to the original signal.
        let max_delta = out
            .samples
            .iter()
            .zip(&mono.samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_delta < 1e-6,
            "downmix changed the signal, max delta {max_delta}"
        );
    }

    /// A zero sample rate must not become `Vec::with_capacity(usize::MAX)`.
    ///
    /// `ratio = src_rate / dst_rate`; `src_rate == 0` makes it 0, `len / 0`
    /// infinity, and `floor() as usize` SATURATES rather than wrapping - so the
    /// capacity request aborted the process. Rates come from a file header, so
    /// zero is reachable input, not a hypothetical.
    #[test]
    fn zero_sample_rates_do_not_request_an_absurd_allocation() {
        let samples = vec![0.1f32; 64];
        assert!(resample(&samples, 0, 16_000).is_empty(), "src_rate 0");
        assert!(resample(&samples, 16_000, 0).is_empty(), "dst_rate 0");
        assert!(resample(&samples, 0, 0).is_empty(), "both 0");
        // A real ratio still resamples.
        assert!(!resample(&samples, 32_000, 16_000).is_empty());
    }
}
