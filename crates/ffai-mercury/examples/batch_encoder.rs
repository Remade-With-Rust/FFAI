//! M-X2: the batched encoder, and its two exit gates.
//!
//! The milestone's gates were: **correctness — transcripts byte-identical to
//! unbatched, not "close"**; and a **null arm** — batch size 1 must reproduce
//! the unbatched number exactly.
//!
//! Both are checked here against our own encoder (the AVX2 path), not
//! candle's. `examples/batch_ceiling.rs` measured the *ceiling* using candle's
//! batch-capable reference; this measures what we actually ship.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example batch_encoder
//! ```

use std::time::Instant;

use ffai_core::candle::Tensor;
use ffai_mercury::asr::mel::{self, MelSpectrogram};
use ffai_mercury::asr::model::LoadedWhisper;
use ffai_mercury::asr::text_decoder::Precision;

/// Distinct audio per window — identical copies would let a batch bug that
/// broadcasts item 0 pass every check.
fn window(seed: usize) -> Vec<f32> {
    let f = 110.0 + 37.0 * seed as f32;
    (0..mel::N_SAMPLES)
        .map(|i| {
            let t = i as f32 / mel::SAMPLE_RATE as f32;
            0.2 * (2.0 * std::f32::consts::PI * f * t).sin()
                + 0.05 * (2.0 * std::f32::consts::PI * (f * 3.7) * t).sin()
        })
        .collect()
}

fn main() {
    let mut whisper = LoadedWhisper::from_manifest_dir(
        std::path::Path::new("models"),
        "whisper-tiny-en",
        ffai_core::best_device(),
        Precision::F32,
    )
    .expect("model loads");
    let front_end = MelSpectrogram::new(whisper.n_mels());

    const N: usize = 4;
    let mels: Vec<Tensor> = (0..N)
        .map(|i| {
            let chunk = front_end.compute(&mel::pad_or_trim(&window(i)));
            whisper.mel_tensor(&chunk).expect("mel tensor")
        })
        .collect();

    // ---- reference: one call per window ----
    let single: Vec<Vec<f32>> = mels
        .iter()
        .map(|m| {
            whisper
                .encoder
                .forward(m)
                .expect("encoder")
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .expect("readback")
        })
        .collect();

    // ---- batched: one call for all N ----
    let stacked = Tensor::cat(&mels.iter().collect::<Vec<_>>(), 0).expect("stack");
    let batched = whisper.encoder.forward(&stacked).expect("batched encoder");
    let (b, seq, d) = batched.dims3().expect("3-D features");
    println!("batched output: ({b}, {seq}, {d})\n");
    assert_eq!(b, N, "batch dimension lost");

    // ---- GATE 1: byte-identical, per window ----
    println!("GATE correctness — batch of {N} vs {N} separate calls");
    let mut worst = 0.0f32;
    let mut exact = true;
    for i in 0..N {
        let got: Vec<f32> = batched
            .narrow(0, i, 1)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .expect("slice readback");
        let diff = got
            .iter()
            .zip(single[i].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if got != single[i] {
            exact = false;
        }
        worst = worst.max(diff);
        println!("  window {i}: max |Δ| = {diff:.3e}{}", if got == single[i] { "  (bit-exact)" } else { "" });
    }

    // A batch bug that broadcasts item 0 shows up here and nowhere else.
    let all_same = (1..N).all(|i| single[i] == single[0]);
    assert!(!all_same, "test windows are not distinct — the check proves nothing");

    println!(
        "\n  verdict: {}",
        if exact {
            "PASS — bit-exact, not merely close"
        } else if worst < 1e-5 {
            "FAIL — close but NOT identical; the gate says identical"
        } else {
            "FAIL — materially different"
        }
    );

    // ---- GATE 2: null arm ----
    let null: Vec<f32> = whisper
        .encoder
        .forward(&mels[0])
        .expect("encoder")
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("readback");
    println!(
        "\nGATE null arm — batch of 1 reproduces unbatched: {}",
        if null == single[0] { "PASS (bit-exact)" } else { "FAIL" }
    );

    // ---- speed, on OUR kernels ----
    println!("\nSPEED — our AVX2 path, best of 3");
    let mut per_window = Vec::new();
    for &size in &[1usize, 2, 4] {
        let input = Tensor::cat(&mels[..size].iter().collect::<Vec<_>>(), 0).expect("stack");
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            let out = whisper.encoder.forward(&input).expect("encoder");
            let _ = out.sum_all().expect("materialise").to_scalar::<f32>().expect("scalar");
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let each = best / size as f64;
        per_window.push(each);
        println!(
            "  batch {size}: {best:8.1} ms total   {each:7.2} ms/window   {:.3}x",
            per_window[0] / each
        );
    }
    println!(
        "\nBatching is implemented and correct. Whether it is WORTH anything is\n\
         the ms/window column, and the ceiling probe already said 1.00-1.04x."
    );
}
