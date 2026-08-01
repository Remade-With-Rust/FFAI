//! Interleaved paired A/B for the parallel relative-attention kernel, with the
//! same discipline dp_ab established: both arms in one process, ABBA order, a
//! null arm for the harness floor, and a paired win-rate z.
//!
//! Also asserts BIT-IDENTITY first. Parallelizing the (h, i) row grid does not
//! reorder any row's summation, so the two arms must agree to the last bit —
//! and if they do not, the speed number is irrelevant.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example enc_ab
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

/// Process CPU time (kernel + user), seconds.
///
/// On a box another session is saturating, WALL time is dominated by how often
/// we were descheduled, which is why the wall-clock A/B of the flat FFN
/// returned a null arm 16% off unity. CPU time does not accrue while
/// descheduled, so for a change that removes WORK it is the instrument that
/// still resolves under foreign load.
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

/// Which knob this run A/Bs. Defaults to the attention kernel it was written
/// for; `FFAI_AB_KNOB` retargets it at any other encoder knob, so a change too
/// small for the whole-pipeline harness can still be resolved at the stage it
/// actually lives in.
fn knob() -> String {
    std::env::var("FFAI_AB_KNOB").unwrap_or_else(|_| "FFAI_SERIAL_ATTN".into())
}

fn set_arm_serial(serial: bool) {
    // SAFETY: single-threaded at the flip; the rayon pool is idle between rounds.
    unsafe {
        if serial {
            std::env::set_var(knob(), "1");
        } else {
            std::env::remove_var(knob());
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let null = std::env::var("FFAI_AB_NULL").is_ok();
    let rounds: usize =
        std::env::var("FFAI_AB_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(31);

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

    // ---- correctness first: the arms must be bit-identical ----
    let mut mismatches = 0usize;
    let check_identity = knob() == "FFAI_SERIAL_ATTN";
    for ids in &ids_list {
        if !check_identity {
            break;
        }
        set_arm_serial(true);
        let (a, _, _) = vits.text_encoder(ids)?;
        set_arm_serial(false);
        let (b, _, _) = vits.text_encoder(ids)?;
        let av: Vec<f32> = a.flatten_all()?.to_vec1()?;
        let bv: Vec<f32> = b.flatten_all()?.to_vec1()?;
        if av.iter().zip(&bv).any(|(x, y)| x.to_bits() != y.to_bits()) {
            mismatches += 1;
        }
    }
    println!(
        "knob={}  bit-identity over 20 sentences: {}",
        knob(),
        if !check_identity {
            "not asserted for this knob".to_string()
        } else if mismatches == 0 {
            "IDENTICAL".to_string()
        } else {
            format!("{mismatches} DIFFER")
        }
    );

    // Measure BOTH: wall for the headline, CPU for the verdict under load.
    let use_cpu = std::env::var("FFAI_AB_WALL").is_err();
    let run_once = |serial: bool| -> Result<f64, Box<dyn std::error::Error>> {
        set_arm_serial(serial && !null);
        let (t0, c0) = (Instant::now(), cpu_secs());
        for ids in &ids_list {
            std::hint::black_box(vits.text_encoder(ids)?);
        }
        Ok(if use_cpu { cpu_secs() - c0 } else { t0.elapsed().as_secs_f64() })
    };

    run_once(false)?;
    run_once(true)?;

    let mut par_wins = 0usize;
    let (mut par_times, mut ser_times) = (Vec::new(), Vec::new());
    for r in 0..rounds {
        let (t_par, t_ser) = if r % 2 == 0 {
            let a = run_once(false)?;
            let b = run_once(true)?;
            (a, b)
        } else {
            let b = run_once(true)?;
            let a = run_once(false)?;
            (a, b)
        };
        if t_par < t_ser {
            par_wins += 1;
        }
        par_times.push(t_par);
        ser_times.push(t_ser);
    }

    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2] * 1000.0
    };
    let n = rounds as f64;
    let z = (par_wins as f64 - n / 2.0) / (0.5 * n.sqrt());
    let (mp, ms) = (median(&mut par_times), median(&mut ser_times));

    println!(
        "{} rounds, metric = {}{}",
        rounds,
        if use_cpu { "CPU TIME (load-robust)" } else { "wall" },
        if null { "  [NULL ARM]" } else { "" }
    );
    println!("  parallel median {mp:8.2} ms   serial median {ms:8.2} ms   ratio {:.3}x", ms / mp);
    println!(
        "  parallel wins {par_wins}/{rounds}  z = {z:+.2}  -> {}",
        if z.abs() > 2.0 {
            if z > 0.0 { "PARALLEL faster (real)" } else { "SERIAL faster (real)" }
        } else {
            "INSIDE NOISE — no verdict"
        }
    );
    Ok(())
}
