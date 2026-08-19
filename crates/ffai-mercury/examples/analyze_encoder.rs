//! Encoder analyzer — quantify before optimizing.
//!
//! The M2 profile left the audio encoder at ~42 % of transcription time on
//! tiny.en. Before touching it, this answers the questions that decide *which*
//! optimization is even applicable (codec-analyzer methodology):
//!
//! 1. **Scaling law.** Is encoder cost linear in input frames, or quadratic?
//!    Self-attention over 1500 positions is O(n²); the convolutional front end
//!    and the MLPs are O(n). Which one dominates decides whether shortening
//!    the input is a large win or a rounding error.
//! 2. **The padding ceiling.** Whisper pads every window to 30 s. On a corpus
//!    of short utterances most encoder work is therefore spent on silence.
//!    This measures exactly how much — the upper bound on what a
//!    variable-length encoder could recover.
//! 3. **Model-size scaling.** tiny.en (4 layers, d_model 384) vs base.en
//!    (6 layers, 512) separates per-layer cost from fixed cost.
//! 4. **Cache-boundedness.** Per-frame throughput across input sizes: if it
//!    degrades as the working set grows, the stage is memory-bound and a
//!    locality refactor applies. If it is flat or improves, it is not, and
//!    that whole family of work can be ruled out cheaply.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example analyze_encoder
//! ```

use std::time::Instant;

use ffai_core::candle::Tensor;
use ffai_mercury::asr::mel::{self, MelSpectrogram};
use ffai_mercury::asr::model::LoadedWhisper;

/// Best-of-N wall clock, the honest number (noise floor is the minimum).
fn best_of<T>(n: usize, mut f: impl FnMut() -> T) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..n {
        let t0 = Instant::now();
        let out = f();
        std::hint::black_box(&out);
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models_dir = std::path::Path::new("models");
    let device = ffai_core::best_device();

    for model_name in ["whisper-tiny-en", "whisper-base-en"] {
        println!("\n================ {model_name} ================");
        let whisper = match LoadedWhisper::from_manifest_dir(
            models_dir,
            model_name,
            device.clone(),
            ffai_mercury::asr::text_decoder::Precision::F32,
        ) {
            Ok(w) => w,
            Err(e) => {
                println!("skipping: {e}");
                continue;
            }
        };
        println!(
            "layers={} d_model={} heads={} n_mels={}",
            whisper.config.encoder_layers,
            whisper.config.d_model,
            whisper.config.encoder_attention_heads,
            whisper.config.num_mel_bins,
        );

        // Real content, not synthetic: a synthetic clip's stage shares can be
        // off by 2-3x (codec-analyzer, 2026-07-16).
        let clip = "corpora/clips/librispeech-test-clean/audio/3575-170457-0008.wav";
        let audio = match ffai_media::load_audio(std::path::Path::new(clip)) {
            Ok(a) => a.to_mono(),
            Err(e) => {
                println!("skipping: {e} (run the corpus prep first)");
                continue;
            }
        };
        let front_end = MelSpectrogram::new(whisper.n_mels());
        let full = front_end.compute(&mel::pad_or_trim(&audio.samples));

        // ---- 1 & 4: scaling law + cache-boundedness in one sweep ----
        // The encoder halves its sequence in conv2 (stride 2), so N mel frames
        // become N/2 attention positions.
        println!("\nSCALING SWEEP (mel frames -> encoder positions)");
        println!(
            "{:>7} {:>7} {:>10} {:>12} {:>10}",
            "FRAMES", "POS", "MS", "US/POS", "VS LINEAR"
        );
        let mut baseline_us_per_pos = 0.0f64;
        for (i, &frames) in [188usize, 375, 750, 1500, 2250, 3000].iter().enumerate() {
            let chunk = full.resized(frames);
            let tensor = whisper.mel_tensor(&chunk)?;
            let secs = best_of(3, || {
                whisper.encoder.forward(&tensor).expect("encoder forward")
            });
            let positions = frames / 2;
            let us_per_pos = secs * 1e6 / positions as f64;
            if i == 0 {
                baseline_us_per_pos = us_per_pos;
            }
            println!(
                "{frames:>7} {positions:>7} {:>10.2} {us_per_pos:>12.3} {:>9.2}x",
                secs * 1000.0,
                us_per_pos / baseline_us_per_pos,
            );
        }
        println!(
            "  reading: 'vs linear' flat => O(n) and NOT cache-bound; rising => O(n^2)\n\
             \x20 attention dominates, or the working set is outgrowing cache."
        );

        // ---- 2: the padding ceiling on the real corpus ----
        if let Ok(manifest) = ffai_bench::corpus::Manifest::load(std::path::Path::new(
            "corpora/librispeech-test-clean-v1.toml",
        )) {
            let mut speech = 0.0f64;
            let mut windows = 0usize;
            for clip in &manifest.clips {
                if let Ok(a) = ffai_media::load_audio(&manifest.clip_path(clip)) {
                    speech += a.duration_secs();
                    windows += (a.duration_secs() / 30.0).ceil().max(1.0) as usize;
                }
            }
            let encoded = windows as f64 * 30.0;
            println!(
                "\nPADDING CEILING (real corpus)\n\
                 \x20 speech {speech:.1}s across {} clips -> {windows} x 30s windows = {encoded:.0}s encoded\n\
                 \x20 {:.1}% of encoder work is spent on padding; a variable-length\n\
                 \x20 encoder could recover at most {:.2}x on this stage.",
                manifest.clips.len(),
                (1.0 - speech / encoded) * 100.0,
                encoded / speech,
            );
        }

        // ---- 3: fixed vs per-layer cost, deployment config ----
        let tensor = whisper.mel_tensor(&full)?;
        let deploy = best_of(5, || {
            whisper.encoder.forward(&tensor).expect("encoder forward")
        });
        println!(
            "\nDEPLOYMENT CONFIG (3000 mel frames, best of 5): {:.2} ms  ({:.3} ms/layer)",
            deploy * 1000.0,
            deploy * 1000.0 / whisper.config.encoder_layers as f64,
        );

        // Keep the tensor alive so the timing above isn't optimized away.
        std::hint::black_box(&tensor as *const Tensor);
    }
    Ok(())
}
