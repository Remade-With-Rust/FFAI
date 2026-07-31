//! Paying the refutation debt: is spreading rayon onto E-cores actually
//! harmless, or did I kill that hypothesis with a bad instrument?
//!
//! The original refutation compared THREE SEPARATE SEQUENTIAL RUNS (8 P-cores
//! 16.85xRT, 8P+8E 18.25x, unpinned 20.89x) — the very design I later proved
//! unreliable for the dp A/B, where sequential arms reversed their verdict
//! between runs. A refutation is permanent in a way a confirmation is not, so
//! it deserves better evidence than the thing it refuted.
//!
//! Two upgrades:
//!   * PAIRED. `SetProcessAffinityMask` can be called repeatedly, so both
//!     affinity arms alternate inside one process, ABBA-ordered, and the
//!     verdict is a paired win rate.
//!   * VARIED AXIS. Straggling should bite hardest where parallel regions are
//!     SHORT (many fork/joins, each waiting on the slowest worker) and least
//!     where they are long. So the same A/B runs at three granularities:
//!     whole pipeline, decoder only (long regions), attention only (short
//!     regions). If E-cores were hurting anywhere, attention is where it shows.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example ecore_ab
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::{GaussRng, Vits};

#[cfg(windows)]
unsafe extern "system" {
    fn GetCurrentProcess() -> isize;
    fn SetProcessAffinityMask(h: isize, mask: usize) -> i32;
    fn SetPriorityClass(h: isize, class: u32) -> i32;
}

/// 8 P-cores, one thread each (logical 0,2,..,14).
const P_ONLY: usize = 0x5555;
/// 8 P-cores + all 8 E-cores.
const P_AND_E: usize = 0xFF_5555;

fn pin(mask: usize) {
    #[cfg(windows)]
    unsafe {
        SetProcessAffinityMask(GetCurrentProcess(), mask);
    }
    #[cfg(not(windows))]
    let _ = mask;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(windows)]
    unsafe {
        SetPriorityClass(GetCurrentProcess(), 0x0000_0080); // HIGH_PRIORITY_CLASS
    }
    let null = std::env::var("FFAI_AB_NULL").is_ok();
    let rounds: usize =
        std::env::var("FFAI_AB_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(21);

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

    // Pre-compute inputs for the narrower granularities.
    let mut hiddens = Vec::new();
    let mut zs = Vec::new();
    for ids in &ids_list {
        let (m_p, logs_p, hidden) = vits.text_encoder(ids)?;
        let mut rng = GaussRng::new(0);
        let w = vits.durations(&hidden, 0.8, 1.0, &mut rng)?;
        let (m_e, _) = vits.expand_prior(&m_p, &logs_p, &w)?;
        zs.push(vits.flow_reverse(&m_e)?);
        hiddens.push((hidden, ids.len()));
    }

    let workloads: Vec<(&str, Box<dyn Fn() -> Result<(), Box<dyn std::error::Error>>>)> = vec![
        (
            "whole pipeline",
            Box::new(|| {
                for ids in &ids_list {
                    let (m_p, logs_p, hidden) = vits.text_encoder(ids)?;
                    let mut rng = GaussRng::new(0);
                    let w = vits.durations(&hidden, 0.8, 1.0, &mut rng)?;
                    let (m_e, _) = vits.expand_prior(&m_p, &logs_p, &w)?;
                    let z = vits.flow_reverse(&m_e)?;
                    std::hint::black_box(vits.decode(&z)?);
                }
                Ok(())
            }),
        ),
        (
            "decoder only (long regions)",
            Box::new(|| {
                for z in &zs {
                    std::hint::black_box(vits.decode(z)?);
                }
                Ok(())
            }),
        ),
        (
            "attention only (short regions)",
            Box::new(|| {
                for (x, t_len) in &hiddens {
                    for l in 0..6 {
                        std::hint::black_box(vits.enc_attn_probe(l, x, *t_len)?);
                    }
                }
                Ok(())
            }),
        ),
    ];

    println!("paired affinity A/B — P-only (0x{P_ONLY:x}) vs P+E (0x{P_AND_E:x})");
    if null {
        println!("[NULL ARM — both arms use the same mask]");
    }
    for (label, work) in &workloads {
        let run = |use_p_only: bool| -> Result<f64, Box<dyn std::error::Error>> {
            pin(if use_p_only && !null { P_ONLY } else { P_AND_E });
            let t0 = Instant::now();
            work()?;
            Ok(t0.elapsed().as_secs_f64())
        };
        run(true)?;
        run(false)?;

        let mut pe_wins = 0usize;
        let (mut p_times, mut pe_times) = (Vec::new(), Vec::new());
        for r in 0..rounds {
            let (tp, tpe) = if r % 2 == 0 {
                let a = run(true)?;
                let b = run(false)?;
                (a, b)
            } else {
                let b = run(false)?;
                let a = run(true)?;
                (a, b)
            };
            if tpe < tp {
                pe_wins += 1;
            }
            p_times.push(tp);
            pe_times.push(tpe);
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2] * 1000.0
        };
        let n = rounds as f64;
        let z = (pe_wins as f64 - n / 2.0) / (0.5 * n.sqrt());
        let (mp, mpe) = (median(&mut p_times), median(&mut pe_times));
        println!(
            "  {label:<32} P-only {mp:7.1} ms  P+E {mpe:7.1} ms  ratio {:.3}x  \
             P+E wins {pe_wins}/{rounds}  z {z:+.2}  -> {}",
            mp / mpe,
            if z.abs() > 2.0 {
                if z > 0.0 { "E-cores HELP" } else { "E-cores HURT (straggler real)" }
            } else {
                "inside noise"
            }
        );
    }
    pin(P_AND_E);
    Ok(())
}
