//! Interleaved paired A/B at the WHOLE-PIPELINE level, for any routing knob.
//!
//! This is the gate that decides. A stage-level harness that runs one stage
//! back-to-back keeps that stage's weights hot in cache in a way the real
//! pipeline never does, and it can report a win the pipeline does not see: the
//! parallel attention kernel measured 1.345x (z = +3.41) stage-level and about
//! 2% across separate whole-pipeline runs. Neither number settles it, because
//! the stage harness is cache-warm and the pipeline runs were sequential arms
//! sampling different machines.
//!
//! So: full synthesis, both arms, one process, ABBA order, paired win rate.
//!
//!   FFAI_AB_KNOB=FFAI_SERIAL_ATTN cargo run --release -p ffai-mercury \
//!       --example pipe_ab
//!
//! `FFAI_AB_NULL=1` runs both arms identical to establish the harness floor.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::{GaussRng, Vits};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let knob = std::env::var("FFAI_AB_KNOB").unwrap_or_else(|_| "FFAI_SERIAL_ATTN".into());
    let null = std::env::var("FFAI_AB_NULL").is_ok();
    let rounds: usize =
        std::env::var("FFAI_AB_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(21);

    let set_arm = |on: bool| {
        // SAFETY: single-threaded at the flip; the rayon pool is idle between rounds.
        unsafe {
            if on && !null {
                std::env::set_var(&knob, "1");
            } else {
                std::env::remove_var(&knob);
            }
        }
    };

    let cache = std::env::var("FFAI_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("ffai"));
    let vits = Vits::load(&cache.join("models").join("piper-vits-lessac-medium"))?;
    let phonemizer = Phonemizer::from_manifest_dir(Path::new("models"))?;

    let manifest =
        ffai_bench::corpus::Manifest::load(Path::new("corpora/harvard-sentences-v1.toml"))?;
    let ids_list: Vec<Vec<i64>> = manifest
        .holdout()
        .take(20)
        .map(|c| {
            let t = std::fs::read_to_string(manifest.clip_path(c)).unwrap();
            vits.id_map.sentence_to_ids(&phonemizer.phonemize(t.trim()).unwrap()).0
        })
        .collect();

    let synth = |on: bool| -> Result<f64, Box<dyn std::error::Error>> {
        set_arm(on);
        let t0 = Instant::now();
        for ids in &ids_list {
            let (m_p, logs_p, hidden) = vits.text_encoder(ids)?;
            let mut rng = GaussRng::new(0);
            let w = vits.durations(&hidden, 0.8, 1.0, &mut rng)?;
            let (m_e, _) = vits.expand_prior(&m_p, &logs_p, &w)?;
            let z = vits.flow_reverse(&m_e)?;
            std::hint::black_box(vits.decode(&z)?);
        }
        Ok(t0.elapsed().as_secs_f64())
    };

    synth(false)?;
    synth(true)?;

    let mut off_wins = 0usize;
    let (mut off_times, mut on_times) = (Vec::new(), Vec::new());
    for r in 0..rounds {
        // "off" = knob unset = the NEW behaviour; "on" = knob set = the OLD one.
        let (t_off, t_on) = if r % 2 == 0 {
            let a = synth(false)?;
            let b = synth(true)?;
            (a, b)
        } else {
            let b = synth(true)?;
            let a = synth(false)?;
            (a, b)
        };
        if t_off < t_on {
            off_wins += 1;
        }
        off_times.push(t_off);
        on_times.push(t_on);
    }

    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2] * 1000.0
    };
    let n = rounds as f64;
    let z = (off_wins as f64 - n / 2.0) / (0.5 * n.sqrt());
    let (mo, mn) = (median(&mut off_times), median(&mut on_times));

    println!("knob={knob} {rounds} rounds{}", if null { "  [NULL ARM]" } else { "" });
    println!("  knob-unset median {mo:8.1} ms   knob-set median {mn:8.1} ms   ratio {:.3}x", mn / mo);
    println!(
        "  knob-unset wins {off_wins}/{rounds}  z = {z:+.2}  -> {}",
        if z.abs() > 2.0 {
            if z > 0.0 { "UNSET faster (real)" } else { "SET faster (real)" }
        } else {
            "INSIDE NOISE — no verdict"
        }
    );
    Ok(())
}
