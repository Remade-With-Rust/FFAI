//! Which of the round-3/4 kernels actually owns the +0.09 pp WER shift?
//!
//! Our WER read 0.0591 for six consecutive runs, then 0.0599 twice. The engine
//! is seeded and deterministic, so that is a real effect of the numerical
//! changes — but THREE landed together:
//!
//!   * conv1d -> GEMM (k=3 im2col+matmul, k=1 direct)  — exact arithmetic,
//!     different summation order (rounding only)
//!   * fused flat LayerNorm                            — same, rounding only
//!   * A&S 7.1.26 GELU                                 — a genuine APPROXIMATION
//!
//! I have been asserting the GELU is the culprit because it is the only
//! approximation. That is a mechanism, not a measurement. This turns each knob
//! on ALONE against the all-candle baseline and reports the audio deviation it
//! contributes, so the attribution is measured before anything is "fixed".
//!
//! ```text
//! cargo run --release -p ffai-mercury --example quality_attrib
//! ```

use std::path::{Path, PathBuf};

use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::{GaussRng, Vits};

const KNOBS: [&str; 3] = ["FFAI_CANDLE_FFN", "FFAI_CANDLE_LN", "FFAI_CANDLE_GELU"];

fn set(knob: &str, on: bool) {
    // SAFETY: single-threaded at the flip; no rayon region is live.
    unsafe {
        if on {
            std::env::set_var(knob, "1");
        } else {
            std::env::remove_var(knob);
        }
    }
}

/// `candle_on` lists the knobs left on the CANDLE path; everything else runs
/// our kernel.
fn configure(candle_on: &[&str]) {
    for k in KNOBS {
        set(k, candle_on.contains(&k));
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::var("FFAI_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("ffai"));
    let vits = Vits::load(&cache.join("models").join("piper-vits-lessac-medium"))?;
    let phonemizer = Phonemizer::from_manifest_dir(Path::new("models"))?;
    let manifest =
        ffai_bench::corpus::Manifest::load(Path::new("corpora/harvard-sentences-v1.toml"))?;
    let n: usize = std::env::var("FFAI_N").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    let ids_list: Vec<Vec<i64>> = manifest
        .clips
        .iter()
        .take(n)
        .map(|c| {
            let t = std::fs::read_to_string(manifest.clip_path(c)).unwrap();
            vits.id_map.sentence_to_ids(&phonemizer.phonemize(t.trim()).unwrap()).0
        })
        .collect();
    println!("{} sentences\n", ids_list.len());

    let synth = |vits: &Vits, ids: &[i64]| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (m_p, logs_p, hidden) = vits.text_encoder(ids)?;
        let w = vits.durations(&hidden, 0.8, 1.0, &mut GaussRng::new(0))?;
        let (m_e, _) = vits.expand_prior(&m_p, &logs_p, &w)?;
        let z = vits.flow_reverse(&m_e)?;
        Ok(vits.decode(&z)?)
    };

    // Reference: every knob on the candle path.
    configure(&KNOBS);
    let mut base = Vec::new();
    let mut base_w = Vec::new();
    for ids in &ids_list {
        let (m_p, logs_p, hidden) = vits.text_encoder(ids)?;
        base_w.push(vits.durations(&hidden, 0.8, 1.0, &mut GaussRng::new(0))?);
        let _ = (&m_p, &logs_p);
        base.push(synth(&vits, ids)?);
    }

    // Each arm names the knobs left on CANDLE, so exactly one of ours is live.
    let arms: [(&str, Vec<&str>); 4] = [
        ("GEMM convs only", vec!["FFAI_CANDLE_LN", "FFAI_CANDLE_GELU"]),
        ("flat LayerNorm only", vec!["FFAI_CANDLE_FFN", "FFAI_CANDLE_GELU"]),
        ("A&S GELU only", vec!["FFAI_CANDLE_FFN", "FFAI_CANDLE_LN"]),
        ("all three (shipped)", vec![]),
    ];

    println!("  {:<22} {:>12} {:>12} {:>10}", "arm", "max|delta|", "rms", "w_ceil");
    for (label, candle_on) in &arms {
        configure(candle_on);
        let (mut worst, mut sq, mut cnt) = (0f32, 0f64, 0usize);
        let mut wdiff = 0usize;
        let mut lendiff = 0usize;
        for (i, ids) in ids_list.iter().enumerate() {
            let (_m, _l, hidden) = vits.text_encoder(ids)?;
            let w = vits.durations(&hidden, 0.8, 1.0, &mut GaussRng::new(0))?;
            if w != base_w[i] {
                wdiff += 1;
            }
            let a = synth(&vits, ids)?;
            if a.len() != base[i].len() {
                lendiff += 1;
                continue;
            }
            for (x, y) in a.iter().zip(&base[i]) {
                let d = (x - y).abs();
                worst = worst.max(d);
                sq += (d as f64) * (d as f64);
                cnt += 1;
            }
        }
        let rms = (sq / cnt.max(1) as f64).sqrt();
        println!(
            "  {label:<22} {worst:>12.3e} {rms:>12.3e} {:>10}",
            if wdiff == 0 && lendiff == 0 { "exact".to_string() } else { format!("{wdiff} differ") }
        );
    }
    configure(&[]);
    println!("\n  arms are vs the all-candle baseline; w_ceil 'exact' means every");
    println!("  duration matched, so any deviation is waveform-only.");
    Ok(())
}
