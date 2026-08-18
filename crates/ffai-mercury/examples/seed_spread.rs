//! Is our 6.00% WER a quality gap, or one draw from a distribution?
//!
//! The bench compares our SINGLE deterministic WER against piper's, but piper
//! samples noise inside its ONNX graph and cannot repeat a run — its ledger
//! history spans 4.8-6.5%. Ours is seeded, so it reports the same number every
//! time. Comparing a fixed point to one draw from a spread tells you very
//! little unless you know how wide OUR spread is.
//!
//! This synthesizes the same holdout at several seeds and judges each through
//! the same frozen whisper.cpp the bench uses. If our seed-to-seed spread is
//! comparable to the measured 0.46 pp "gap", then the gap is sampling noise and
//! the honest verdict is parity.
//!
//! NOTE this is a MEASUREMENT, not a tuning knob. Picking the best seed would
//! not generalize: the seed fixes a noise draw per utterance, so a seed that
//! flatters the train split has no mechanism to help unseen sentences. That is
//! overfitting, and it is deliberately not done here.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example seed_spread
//! ```

use std::path::{Path, PathBuf};

use ffai_bench::metrics::wer_with;
use ffai_bench::normalize::Mode;
use ffai_bench::reference::ReferenceFile;
use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::{SynthesisOptions, Vits};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::var("FFAI_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("ffai"));
    let vits = Vits::load(&cache.join("models").join("piper-vits-lessac-medium"))?;
    let phonemizer = Phonemizer::from_manifest_dir(Path::new("models"))?;
    let manifest =
        ffai_bench::corpus::Manifest::load(Path::new("corpora/harvard-sentences-v1.toml"))?;

    let n: usize = std::env::var("FFAI_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(134);
    let clips: Vec<_> = manifest.holdout().take(n).collect();
    let truths: Vec<(String, String)> = clips
        .iter()
        .map(|c| {
            (
                c.id.clone(),
                std::fs::read_to_string(manifest.clip_path(c))
                    .unwrap()
                    .trim()
                    .to_string(),
            )
        })
        .collect();
    println!("{} holdout clips", truths.len());

    let judge = ReferenceFile::load(Path::new("corpora/references.toml"))?
        .for_task("tts-judge")
        .next()
        .cloned()
        .ok_or("no tts-judge in corpora/references.toml")?;

    let outdir = PathBuf::from("bench/seed-spread");
    let seeds: Vec<u64> = std::env::var("FFAI_SEEDS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0, 1, 2, 3, 4]);

    let mut wers = Vec::new();
    for seed in &seeds {
        let dir = outdir.join(format!("seed{seed}"));
        std::fs::create_dir_all(&dir)?;
        let mut paths = Vec::new();
        for (id, text) in &truths {
            let ids = vits.id_map.sentence_to_ids(&phonemizer.phonemize(text)?).0;
            // Call the REAL synthesis entry point. A first cut of this probe
            // hand-rolled the stages and fed `m_exp` straight to flow_reverse,
            // silently dropping the ACOUSTIC noise (`noise_scale`, 0.667 by
            // default) that `synthesize_ids` applies -- so it measured duration
            // noise only and read ~0.85 pp better than the bench. The engine's
            // own defaults, with only the seed varying, is the whole point.
            let opts = SynthesisOptions {
                seed: *seed,
                ..vits.defaults
            };
            let audio = ffai_core::types::AudioBuffer {
                samples: vits.synthesize_ids(&ids, &opts)?,
                sample_rate: vits.sample_rate,
                channels: 1,
            };
            let wav = dir.join(format!("{id}.wav"));
            ffai_media::save_wav(&wav, &ffai_bench::resample::to_judge_format(&audio, 16_000))?;
            paths.push(wav);
        }
        eprintln!("judging seed {seed} ({} wavs) ...", paths.len());
        let batch = judge.run_batch(&paths)?;
        // Per-clip WER averaged over clips — the same aggregation the bench
        // and the substitution gate use, so this number is comparable to them.
        let mut per_clip = Vec::new();
        for ((id, truth), wav) in truths.iter().zip(&paths) {
            match batch.text_for(wav) {
                Some(hyp) => per_clip.push(wer_with(truth, hyp, Mode::English)),
                None => eprintln!("  {id}: judge returned no transcript"),
            }
        }
        let w = per_clip.iter().sum::<f64>() / per_clip.len().max(1) as f64;
        println!("  seed {seed:>3}   WER {:.4} %", w * 100.0);
        wers.push(w);
    }

    wers.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (lo, hi) = (wers[0], wers[wers.len() - 1]);
    let med = wers[wers.len() / 2];
    println!(
        "\n  our spread across {} seeds: {:.4} .. {:.4} %  (median {:.4}, range {:.3} pp)",
        seeds.len(),
        lo * 100.0,
        hi * 100.0,
        med * 100.0,
        (hi - lo) * 100.0
    );
    println!("  piper's own ledger history spans 4.8-6.5 % on this corpus.");
    Ok(())
}
