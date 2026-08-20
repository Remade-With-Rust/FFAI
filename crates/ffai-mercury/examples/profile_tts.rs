//! M-T3 speed instrument: where does synthesis time go, per stage — and is
//! candle's conv1d the bottleneck it looks like?
//!
//! ```text
//! cargo run --release -p ffai-mercury --example profile_tts
//! ```
//!
//! Two measurements, per the codec-analyzer rule that a stage split names
//! WHERE and a microbench names WHY:
//! 1. Stage timing over real holdout sentences (text encoder / duration
//!    predictor / flow / decoder), best-of-3 per stage.
//! 2. A conv-vs-matmul probe at the decoder's exact hot shapes: the same
//!    arithmetic as conv1d, expressed as im2col + matmul through candle's
//!    tuned GEMM. If the ratio is large, the lever is layout, not SIMD.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ffai_core::candle::{Device, Tensor};
use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::{GaussRng, SynthesisOptions, Vits};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::var("FFAI_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("ffai"));
    let vits = Vits::load(&cache.join("models").join("piper-vits-lessac-medium"))?;
    let phonemizer = Phonemizer::from_manifest_dir(Path::new("models"))?;

    // 20 holdout sentences: enough to average, quick to iterate.
    let manifest =
        ffai_bench::corpus::Manifest::load(Path::new("corpora/harvard-sentences-v1.toml"))?;
    let texts: Vec<String> = manifest
        .holdout()
        .take(20)
        .map(|c| {
            std::fs::read_to_string(manifest.clip_path(c))
                .unwrap()
                .trim()
                .to_string()
        })
        .collect();
    let ids_list: Vec<Vec<i64>> = texts
        .iter()
        .map(|t| {
            vits.id_map
                .sentence_to_ids(&phonemizer.phonemize(t).unwrap())
                .0
        })
        .collect();

    // ---- stage timing, best-of-3 whole-corpus passes ----
    let mut best = [f64::MAX; 4];
    let mut audio_secs = 0f64;
    for _ in 0..3 {
        let (mut t_enc, mut t_dp, mut t_flow, mut t_dec) = (0f64, 0f64, 0f64, 0f64);
        audio_secs = 0.0;
        for ids in &ids_list {
            let t0 = Instant::now();
            let (m_p, logs_p, hidden) = vits.text_encoder(ids)?;
            t_enc += t0.elapsed().as_secs_f64();

            let t0 = Instant::now();
            let mut rng = GaussRng::new(0);
            let w = vits.durations(&hidden, 0.8, 1.0, &mut rng)?;
            t_dp += t0.elapsed().as_secs_f64();

            let t0 = Instant::now();
            let (m_e, _logs_e) = vits.expand_prior(&m_p, &logs_p, &w)?;
            let z = vits.flow_reverse(&m_e)?;
            t_flow += t0.elapsed().as_secs_f64();

            let t0 = Instant::now();
            let audio = vits.decode(&z)?;
            t_dec += t0.elapsed().as_secs_f64();
            audio_secs += audio.len() as f64 / vits.sample_rate as f64;
        }
        for (b, t) in best.iter_mut().zip([t_enc, t_dp, t_flow, t_dec]) {
            *b = b.min(t);
        }
    }
    let total: f64 = best.iter().sum();
    println!("stage timing, 20 sentences, {audio_secs:.1}s audio, best-of-3:");
    for (name, t) in ["text_encoder", "duration_pred", "flow", "decoder"]
        .iter()
        .zip(best)
    {
        println!(
            "  {name:<14} {:>8.1} ms  {:>5.1} %",
            t * 1000.0,
            100.0 * t / total
        );
    }
    println!(
        "  {:<14} {:>8.1} ms  -> {:.1}x realtime",
        "total",
        total * 1000.0,
        audio_secs / total
    );

    // ---- conv1d vs im2col+matmul at the decoder's hot shapes ----
    println!("\nconv1d vs im2col+matmul (per call, best of 5):");
    let dev = Device::Cpu;
    for (c_in, c_out, k, len, label) in [
        (128usize, 128usize, 5usize, 1520usize, "resblock L1"),
        (64, 64, 5, 12160, "resblock L2"),
        (32, 32, 7, 48640, "resblock L3"),
        (192, 384, 5, 190, "flow in_layer"),
        (192, 384, 1, 190, "flow res_skip"),
    ] {
        let x = Tensor::randn(0f32, 1f32, (1, c_in, len), &dev)?;
        let w = Tensor::randn(0f32, 1f32, (c_out, c_in, k), &dev)?;
        let pad = (k - 1) / 2;

        let conv_ms = best_ms(5, || {
            x.conv1d(&w, pad, 1, 1, 1).unwrap();
        });

        let direct_ms = best_ms(5, || {
            ffai_mercury::tts::decoder_kernels::conv1d_direct(&x, &w, None, pad, 1).unwrap();
        });

        // im2col: [C_in*K, L] gathered once, then one [C_out, C_in*K] GEMM.
        let mm_ms = best_ms(5, || {
            let xp = x.pad_with_zeros(2, pad, pad).unwrap();
            let cols: Vec<Tensor> = (0..k).map(|i| xp.narrow(2, i, len).unwrap()).collect();
            let stacked = Tensor::cat(&cols, 1).unwrap(); // [1, C_in*K, L] (k-major per channel)
            let w2 = w
                .transpose(1, 2)
                .unwrap()
                .reshape((c_out, c_in * k))
                .unwrap(); // matches [K,C_in] ordering? probe only: FLOP-equivalent
            w2.broadcast_matmul(&stacked).unwrap();
        });
        let flops = 2.0 * (c_out * c_in * k * len) as f64;
        println!(
            "  {label:<22} conv {conv_ms:>7.2} ms ({:>5.1} GF/s)   matmul {mm_ms:>7.2} ms ({:>5.1} GF/s)   direct {direct_ms:>7.2} ms ({:>5.1} GF/s)",
            flops / conv_ms / 1e6,
            flops / mm_ms / 1e6,
            flops / direct_ms / 1e6,
        );
    }
    Ok(())
}

fn best_ms(n: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..n {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
    }
    best
}
