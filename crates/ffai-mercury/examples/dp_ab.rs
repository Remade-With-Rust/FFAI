//! Interleaved paired A/B for the duration predictor's conv routing.
//!
//! Sequential process-per-arm A/B was useless here: three runs of the same two
//! arms read NEW 210.8 / OLD 312.8, then NEW 191.6 / OLD 277.3, then NEW 273.7 /
//! OLD 240.4 — a reversal, with 30-43% spread INSIDE each arm. The two arms
//! were sampling different machines.
//!
//! So: both arms in ONE process, alternating every round, ABBA-ordered so any
//! "the second one runs warmer" effect cancels instead of accumulating. The
//! statistic is the paired win rate, which is a fair coin under the null:
//!
//!     z = (wins - N/2) / (0.5 * sqrt(N))
//!
//! |z| > 2 is a verdict regardless of how far the medians drifted.
//!
//! `FFAI_AB_NULL=1` makes both arms identical — the harness's own resolution
//! floor. Any effect smaller than what the null reports is not a result.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example dp_ab
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::{GaussRng, Vits};

/// Flip the routing knob `Vits::conv` reads. Both arms live in one binary so
/// they experience identical machine drift.
fn set_arm_old(old: bool) {
    // SAFETY: single-threaded at the point of the flip; the rayon pool is idle
    // between rounds.
    unsafe {
        if old {
            std::env::set_var("FFAI_DP_DIRECT_1X1", "1");
        } else {
            std::env::remove_var("FFAI_DP_DIRECT_1X1");
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let null = std::env::var("FFAI_AB_NULL").is_ok();
    let rounds: usize = std::env::var("FFAI_AB_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(31);

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
    let mut hiddens = Vec::new();
    for ids in &ids_list {
        let (_m, _l, hidden) = vits.text_encoder(ids)?;
        hiddens.push(hidden);
    }

    let run_once = |old: bool| -> Result<f64, Box<dyn std::error::Error>> {
        set_arm_old(old && !null);
        let t0 = Instant::now();
        for h in &hiddens {
            let mut rng = GaussRng::new(0);
            std::hint::black_box(vits.durations(h, 0.8, 1.0, &mut rng)?);
        }
        Ok(t0.elapsed().as_secs_f64())
    };

    // Warm both arms so neither pays first-touch costs inside the measurement.
    run_once(false)?;
    run_once(true)?;

    let mut new_wins = 0usize;
    let (mut new_times, mut old_times) = (Vec::new(), Vec::new());
    for r in 0..rounds {
        // ABBA: alternate which arm goes first each round.
        let (t_new, t_old) = if r % 2 == 0 {
            let a = run_once(false)?;
            let b = run_once(true)?;
            (a, b)
        } else {
            let b = run_once(true)?;
            let a = run_once(false)?;
            (a, b)
        };
        if t_new < t_old {
            new_wins += 1;
        }
        new_times.push(t_new);
        old_times.push(t_old);
    }

    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2] * 1000.0
    };
    let n = rounds as f64;
    let z = (new_wins as f64 - n / 2.0) / (0.5 * n.sqrt());
    let (mn, mo) = (median(&mut new_times), median(&mut old_times));

    println!(
        "{} rounds{}",
        rounds,
        if null {
            "  [NULL ARM — both arms identical]"
        } else {
            ""
        }
    );
    println!(
        "  new median {mn:8.2} ms   old median {mo:8.2} ms   ratio {:.3}x",
        mo / mn
    );
    println!(
        "  new wins {new_wins}/{rounds}  z = {z:+.2}  -> {}",
        if z.abs() > 2.0 {
            if z > 0.0 {
                "NEW faster (real)"
            } else {
                "OLD faster (real)"
            }
        } else {
            "INSIDE NOISE — no verdict"
        }
    );
    Ok(())
}
