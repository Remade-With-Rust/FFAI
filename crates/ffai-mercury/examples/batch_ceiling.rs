//! M-X2: does batching the encoder actually buy anything on this CPU?
//!
//! The prune in `docs/whys/mx2-batching.md` rests on two claims. The first —
//! every corpus clip is a single window — was measured directly. The second —
//! the encoder is compute-saturated so batching cannot help — was taken from
//! the roofline table in mission-plan §6.10 rather than measured *as
//! batching*. A borrowed number is not a probe, so this measures it.
//!
//! Our own encoder cannot be used for the test: `conv1d_gemm` flattens and
//! indexes as `(cin, l_in)` with no batch stride, so a batched call silently
//! returns item 0. candle's reference encoder goes through `candle_nn::conv1d`
//! and is genuinely batch-capable, which makes it the right instrument — it
//! answers "does this MACHINE reward batching this SHAPE", which is the
//! question, and it answers it without first committing to a kernel rewrite.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example batch_ceiling
//! ```

use std::time::Instant;

use ffai_core::candle::{DType, Tensor};
use ffai_mercury::asr::mel::{self, MelSpectrogram};
use ffai_mercury::asr::model::LoadedWhisper;
use ffai_mercury::asr::text_decoder::Precision;

fn main() {
    let mut whisper = LoadedWhisper::from_manifest_dir(
        std::path::Path::new("models"),
        "whisper-tiny-en",
        ffai_core::best_device(),
        Precision::F32,
    )
    .expect("model loads");

    let front_end = MelSpectrogram::new(whisper.n_mels());
    // One window's worth of audio; content is irrelevant to timing, and a
    // fixed pattern keeps the run reproducible.
    let samples: Vec<f32> = (0..mel::N_SAMPLES)
        .map(|i| ((i as f32) * 0.001).sin() * 0.1)
        .collect();
    let chunk = front_end.compute(&mel::pad_or_trim(&samples));
    let one = whisper.mel_tensor(&chunk).expect("mel tensor");
    let (_, n_mels, frames) = one.dims3().expect("3-D mel");

    println!("mel window: {n_mels} x {frames}   (one 30 s context)\n");

    // Warm the allocator and any lazily-built kernels so the first timed
    // batch is not paying setup the others avoid.
    for _ in 0..2 {
        let _ = whisper.model.encoder.forward(&one, true).expect("warmup");
    }

    println!("{:>5}  {:>10}  {:>12}  {:>10}  {:>9}", "batch", "total ms", "ms/window", "vs b=1", "verdict");
    println!("{}", "-".repeat(56));

    let mut baseline_per_window = 0.0f64;
    for &b in &[1usize, 2, 4, 8] {
        // Stack b copies into (b, n_mels, frames).
        let batched = if b == 1 {
            one.clone()
        } else {
            Tensor::cat(&vec![&one; b], 0).expect("stack").to_dtype(DType::F32).expect("dtype")
        };

        // Best-of-3: the fastest run is the one least disturbed by whatever
        // else the machine was doing, the same rule the bench harness uses.
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            let out = whisper.model.encoder.forward(&batched, true).expect("encoder");
            // Force materialisation; candle is lazy enough that timing a
            // graph build instead of a computation is a real hazard.
            let _ = out.sum_all().expect("materialise").to_scalar::<f32>().expect("scalar");
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }

        let per_window = best / b as f64;
        if b == 1 {
            baseline_per_window = per_window;
        }
        let speedup = baseline_per_window / per_window;
        let verdict = if speedup >= 1.15 {
            "REAL"
        } else if speedup >= 1.05 {
            "marginal"
        } else {
            "none"
        };
        println!("{b:>5}  {best:>10.1}  {per_window:>12.2}  {speedup:>9.3}x  {verdict:>9}");
    }

    println!(
        "\nPrune stands if per-window time is flat: batching more windows into\n\
         one call buys nothing when the machine is already saturated at b=1.\n\
         A speedup >= 1.15x at any batch size would REFUTE the prune and make\n\
         extending conv1d_gemm and the transposed-K path worth costing."
    );
}
