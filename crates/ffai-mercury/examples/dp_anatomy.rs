//! D4 for the duration predictor: it runs at 0.88 cores while flow and decoder
//! reach 11+, and costs 16% of wall. Which op inside it owns that, and is the
//! cost arithmetic or candle tensor plumbing?
//!
//! The distinction decides the fix. If it is arithmetic, the lever is a kernel
//! (this campaign already has flat AVX2 paths for flow and decoder). If it is
//! plumbing — allocation, reshape, broadcast, dispatch on tensors of ~60
//! columns — no kernel helps and the lever is de-plumbing.
//!
//! Reports ns and the achieved GFLOP/s next to a rough FLOP count, because a
//! stage running at single-digit GFLOP/s on a machine that does hundreds is
//! overhead-bound by definition.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example dp_anatomy
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::{GaussRng, Vits};

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

    // Hidden states are the duration predictor's input; compute them once so
    // the text encoder's cost is not folded into this measurement.
    let mut hiddens = Vec::new();
    for ids in &ids_list {
        let (_m, _l, hidden) = vits.text_encoder(ids)?;
        hiddens.push(hidden);
    }
    let total_t: usize = hiddens.iter().map(|h| h.dim(2).unwrap()).sum();
    println!(
        "20 sentences, {total_t} phoneme columns total (mean {:.1})",
        total_t as f64 / 20.0
    );

    // Warm.
    for h in &hiddens {
        let mut rng = GaussRng::new(0);
        vits.durations(h, 0.8, 1.0, &mut rng)?;
    }

    let mut best = f64::MAX;
    for _ in 0..7 {
        let t0 = Instant::now();
        for h in &hiddens {
            let mut rng = GaussRng::new(0);
            std::hint::black_box(vits.durations(h, 0.8, 1.0, &mut rng)?);
        }
        best = best.min(t0.elapsed().as_secs_f64());
    }
    println!("durations() whole:      {:>8.2} ms", best * 1000.0);

    // The conditioning half (pre -> DDSConv -> proj) is where essentially all
    // the multiply-accumulate lives: 192 channels against 2 in the flow half.
    let mut best_cond = f64::MAX;
    for _ in 0..7 {
        let t0 = Instant::now();
        for h in &hiddens {
            std::hint::black_box(vits.dp_conditioning_probe(h)?);
        }
        best_cond = best_cond.min(t0.elapsed().as_secs_f64());
    }
    // FLOPs for the conditioning path, per column:
    //   pre    1x1 conv 192->192          = 192*192*2
    //   dds    3 x [depthwise k5 192      = 192*5*2
    //               1x1 conv 192->192     = 192*192*2]
    //   proj   1x1 conv 192->192          = 192*192*2
    let per_col = 192.0 * 192.0 * 2.0 * 5.0 + 3.0 * 192.0 * 5.0 * 2.0;
    let gflop = per_col * total_t as f64 / 1e9;
    println!(
        "  conditioning:         {:>8.2} ms  ({:.3} GFLOP -> {:.1} GFLOP/s)",
        best_cond * 1000.0,
        gflop,
        gflop / best_cond
    );
    println!(
        "  flows + tail:         {:>8.2} ms  ({:.0}% of durations())",
        (best - best_cond) * 1000.0,
        100.0 * (best - best_cond) / best
    );
    Ok(())
}
