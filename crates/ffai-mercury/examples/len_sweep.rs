//! Item #4 + the reopened item #3: find the CROSSOVER for every parallel/serial
//! guard in the TTS path, instead of asserting one.
//!
//! Three parallel decisions were made this campaign and all three sit on the
//! same axis — WORK PER TASK, which is driven by sequence length:
//!   * dp dense convs  -> parallel WON  at T~88
//!   * attention rows  -> parallel WON  at T~88
//!   * spline columns  -> parallel LOST at T~88
//!
//! Per codec-content-adaptive-dispatch, that mixed outcome is not a verdict on
//! the losing feature; it is an unfinished dispatch. A loss localizes WHERE the
//! premise breaks, and the honest endpoints are (a) find the threshold that
//! makes the loser non-negative while the winners keep winning, or (b) prove no
//! cheap signal separates them. The guards currently read `>= 32`, a number I
//! reasoned to rather than measured — and reasoning is exactly what got the
//! spline wrong.
//!
//! Sweeps T and reports, per knob, the paired win rate of parallel over serial.
//! The crossover is where z crosses +2.
//!
//! Synthetic lengths are legitimate HERE (and only here): these kernels are
//! dense with no data-dependent branching, so cost depends on shape, not on
//! content. The same shortcut would be invalid for stage-share or quality work.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example len_sweep
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::{GaussRng, Vits};

fn set(knob: &str, on: bool) {
    // SAFETY: single-threaded at the flip; no rayon region live.
    unsafe {
        if on {
            std::env::set_var(knob, "1");
        } else {
            std::env::remove_var(knob);
        }
    }
}

/// Paired ABBA win rate of "knob unset" (parallel) over "knob set" (serial).
fn paired<F>(knob: &str, rounds: usize, mut run: F) -> (f64, f64, f64)
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    let mut once = |serial: bool| -> f64 {
        set(knob, serial);
        let t0 = Instant::now();
        let _ = run();
        t0.elapsed().as_secs_f64()
    };
    once(false);
    once(true);
    let mut par_wins = 0usize;
    let (mut pv, mut sv) = (Vec::new(), Vec::new());
    for r in 0..rounds {
        let (p, s) = if r % 2 == 0 {
            let a = once(false);
            let b = once(true);
            (a, b)
        } else {
            let b = once(true);
            let a = once(false);
            (a, b)
        };
        if p < s {
            par_wins += 1;
        }
        pv.push(p);
        sv.push(s);
    }
    pv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = rounds as f64;
    let z = (par_wins as f64 - n / 2.0) / (0.5 * n.sqrt());
    (sv[rounds / 2] / pv[rounds / 2], z, pv[rounds / 2] * 1000.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rounds: usize =
        std::env::var("FFAI_AB_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    let cache = std::env::var("FFAI_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("ffai"));
    let vits = Vits::load(&cache.join("models").join("piper-vits-lessac-medium"))?;
    let phonemizer = Phonemizer::from_manifest_dir(Path::new("models"))?;
    let manifest =
        ffai_bench::corpus::Manifest::load(Path::new("corpora/harvard-sentences-v1.toml"))?;
    let seed_ids = manifest
        .holdout()
        .take(1)
        .map(|c| {
            let t = std::fs::read_to_string(manifest.clip_path(c)).unwrap();
            vits.id_map.sentence_to_ids(&phonemizer.phonemize(t.trim()).unwrap()).0
        })
        .next()
        .unwrap();

    println!("paired parallel-vs-serial by sequence length ({rounds} rounds each)");
    println!("  a guard should sit where z first exceeds +2\n");
    println!(
        "  {:>6}  {:>22}  {:>22}",
        "T", "attention (FFAI_SERIAL_ATTN)", "spline (FFAI_PAR_SPLINE inv)"
    );

    for &t_len in &[16usize, 32, 64, 128, 256, 512, 1024] {
        // Build ids of the requested length by tiling a real sentence.
        let ids: Vec<i64> = (0..t_len).map(|i| seed_ids[i % seed_ids.len()]).collect();
        let (_m, _l, hidden) = vits.text_encoder(&ids)?;
        let real_t = hidden.dim(2)?;

        let (r_attn, z_attn, ms_attn) = paired("FFAI_SERIAL_ATTN", rounds, || {
            for l in 0..6 {
                std::hint::black_box(vits.enc_attn_probe(l, &hidden, real_t)?);
            }
            Ok(())
        });

        // The spline knob is inverted: FFAI_PAR_SPLINE=1 ENABLES parallel, so
        // "knob set" is the parallel arm. Report it in the same orientation as
        // attention (ratio > 1 and z > 0 mean parallel wins) by flipping.
        let (r_sp, z_sp, ms_sp) = paired("FFAI_PAR_SPLINE", rounds, || {
            let mut rng = GaussRng::new(0);
            std::hint::black_box(vits.durations(&hidden, 0.8, 1.0, &mut rng)?);
            Ok(())
        });
        let (r_sp, z_sp) = (1.0 / r_sp, -z_sp);

        println!(
            "  {real_t:>6}  {:>8.3}x z {:>+5.2} ({:>5.1}ms)  {:>8.3}x z {:>+5.2} ({:>5.1}ms)",
            r_attn, z_attn, ms_attn, r_sp, z_sp, ms_sp
        );
    }
    println!("\n  ratio > 1 and z > +2 => parallel is genuinely faster at that length");
    Ok(())
}
