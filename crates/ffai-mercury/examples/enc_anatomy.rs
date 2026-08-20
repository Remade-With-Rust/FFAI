//! D3b/D4b: the text encoder is 483 ms at 4.27 cores — the lowest occupancy
//! left after the duration-predictor fix, and 24% of the pipeline.
//!
//! Splits the 6-layer stack into attention / feed-forward / LayerNorm and
//! reports achieved GFLOP/s per part, plus each part's core occupancy. A part
//! at single-digit GFLOP/s and ~1 core is dispatch-bound; a part at high
//! occupancy but still slow is a kernel problem. The two want opposite fixes,
//! which is why guessing here is expensive.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example enc_anatomy
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::Vits;

#[cfg(windows)]
unsafe extern "system" {
    fn GetCurrentProcess() -> isize;
    fn GetProcessTimes(h: isize, c: *mut u64, e: *mut u64, k: *mut u64, u: *mut u64) -> i32;
}

fn cpu_secs() -> f64 {
    #[cfg(windows)]
    unsafe {
        let (mut c, mut e, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
        if GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) == 0 {
            return 0.0;
        }
        (k + u) as f64 * 1e-7
    }
    #[cfg(not(windows))]
    0.0
}

const HIDDEN: f64 = 192.0;
const LAYERS: usize = 6;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::var("FFAI_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("ffai"));
    let vits = Vits::load(&cache.join("models").join("piper-vits-lessac-medium"))?;
    let phonemizer = Phonemizer::from_manifest_dir(Path::new("models"))?;

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

    // Each layer's input, captured once so every part is measured on the real
    // activations rather than a synthetic tensor.
    let mut inputs = Vec::new();
    for ids in &ids_list {
        let (_m, _l, hidden) = vits.text_encoder(ids)?;
        inputs.push((hidden, ids.len()));
    }
    let total_t: usize = inputs.iter().map(|(_, t)| *t).sum();
    println!(
        "20 sentences, {total_t} phoneme columns (mean {:.1})",
        total_t as f64 / 20.0
    );

    let bench =
        |label: &str, gflop: f64, f: &dyn Fn() -> Result<(), Box<dyn std::error::Error>>| {
            // Warm.
            let _ = f();
            let (mut best, mut best_cpu) = (f64::MAX, f64::MAX);
            for _ in 0..7 {
                let (t0, c0) = (Instant::now(), cpu_secs());
                let _ = f();
                let w = t0.elapsed().as_secs_f64();
                if w < best {
                    best = w;
                    best_cpu = cpu_secs() - c0;
                }
            }
            println!(
                "  {label:<16} {:>8.1} ms  {:>6.2} cores  {:>7.1} GFLOP/s",
                best * 1000.0,
                best_cpu / best,
                gflop / best
            );
            best
        };

    let t = total_t as f64;
    // Per column, summed over 6 layers.
    // attention: fused QKV 192->576, out proj 192->192
    let g_attn = LAYERS as f64 * t * (HIDDEN * 3.0 * HIDDEN * 2.0 + HIDDEN * HIDDEN * 2.0) / 1e9;
    // ffn: conv k3 192->768 then 768->192
    let g_ffn = LAYERS as f64 * t * (HIDDEN * 768.0 * 3.0 * 2.0 * 2.0) / 1e9;

    let whole = bench("whole encoder", g_attn + g_ffn, &|| {
        for ids in &ids_list {
            std::hint::black_box(vits.text_encoder(ids)?);
        }
        Ok(())
    });
    let attn = bench("attention x6", g_attn, &|| {
        for (x, t_len) in &inputs {
            for l in 0..LAYERS {
                std::hint::black_box(vits.enc_attn_probe(l, x, *t_len)?);
            }
        }
        Ok(())
    });
    let ffn = bench("ffn x6", g_ffn, &|| {
        for (x, _) in &inputs {
            for l in 0..LAYERS {
                std::hint::black_box(vits.enc_ffn_probe(l, x)?);
            }
        }
        Ok(())
    });
    let norm = bench("layernorm x6", 0.0, &|| {
        for (x, _) in &inputs {
            for l in 0..LAYERS {
                std::hint::black_box(vits.enc_norm_probe(l, x)?);
            }
        }
        Ok(())
    });

    println!(
        "\n  parts sum {:.1} ms vs whole {:.1} ms -> residue {:.1} ms ({:.0}%)",
        (attn + ffn + norm) * 1000.0,
        whole * 1000.0,
        (whole - attn - ffn - norm) * 1000.0,
        100.0 * (whole - attn - ffn - norm) / whole
    );
    Ok(())
}
