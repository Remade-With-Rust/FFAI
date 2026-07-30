//! The M-T1 substitution gate (docs/mercury-tts-mission.md §6.2): feed OUR
//! phonemes through PIPER'S OWN runtime and judge the audio against espeak's
//! phonemes through the same runtime. Synthesis quality is held constant, so
//! the round-trip WER difference prices the phonemizer and nothing else —
//! the exit criterion is our arm inside the 5 % relative band.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example tts_substitution_gate
//! ```
//!
//! Pipeline: phonemize holdout → piper_substitute.py renders both arms at
//! zero noise (deterministic) → harness resamples to the judge format →
//! the frozen `tts-judge` from corpora/references.toml transcribes both →
//! WER/CER vs corpus text under the English normalizer. No self-grading:
//! the judge is whisper.cpp, pinned.

use std::path::{Path, PathBuf};
use std::process::Command;

use ffai_bench::metrics::{cer_with, wer_with};
use ffai_bench::normalize::Mode;
use ffai_bench::reference::ReferenceFile;
use ffai_mercury::tts::phonemize::Phonemizer;

const BAND: f64 = 1.05;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest =
        ffai_bench::corpus::Manifest::load(Path::new("corpora/harvard-sentences-v1.toml"))?;
    let phonemizer = Phonemizer::from_manifest_dir(Path::new("models"))?;
    let outdir = PathBuf::from("bench/mt1-substitution");
    std::fs::create_dir_all(&outdir)?;

    // 1. Phonemize the HOLDOUT and write ours.jsonl.
    let mut ours_jsonl = String::new();
    let mut truths: Vec<(String, String)> = Vec::new();
    for clip in manifest.holdout() {
        let text = std::fs::read_to_string(manifest.clip_path(clip))?;
        let ipa = phonemizer.phonemize(text.trim())?;
        ours_jsonl.push_str(&serde_json::json!({"id": clip.id, "ipa": ipa}).to_string());
        ours_jsonl.push('\n');
        truths.push((clip.id.clone(), text.trim().to_string()));
    }
    let ours_path = outdir.join("ours.jsonl");
    std::fs::write(&ours_path, &ours_jsonl)?;
    eprintln!("phonemized {} holdout sentences", truths.len());

    // 2. Render both arms through piper's runtime, zero noise.
    let status = Command::new(".venv-bench/Scripts/python.exe")
        .args(["corpora/refs/piper_substitute.py", "--fixtures"])
        .arg("corpora/fixtures/harvard-espeak-phonemes-v1.jsonl")
        .arg("--ours")
        .arg(&ours_path)
        .args(["--model", ".piper-voices/en_US-lessac-medium.onnx", "--outdir"])
        .arg(&outdir)
        .status()?;
    if !status.success() {
        return Err(format!("piper_substitute.py failed: {status}").into());
    }

    // 3. Resample both arms to the judge format (the same code path
    //    `ffai bench tts` uses for every implementation).
    let judge = ReferenceFile::load(Path::new("corpora/references.toml"))?
        .for_task("tts-judge")
        .next()
        .cloned()
        .ok_or("no tts-judge in corpora/references.toml")?;
    let mut results: Vec<(String, f64, f64, usize)> = Vec::new();
    let mut transcripts: Vec<Vec<(String, f64, String)>> = Vec::new();
    for arm in ["espeak", "ours"] {
        let arm_dir = outdir.join(arm);
        let judge_dir = outdir.join(format!("{arm}-judge16k"));
        std::fs::create_dir_all(&judge_dir)?;
        let mut judge_paths = Vec::new();
        for (id, _) in &truths {
            let wav = arm_dir.join(format!("{id}.wav"));
            let audio = ffai_media::load_audio(&wav)?;
            let jwav = judge_dir.join(format!("{id}.wav"));
            ffai_media::save_wav(&jwav, &ffai_bench::resample::to_judge_format(&audio, 16_000))?;
            judge_paths.push(jwav);
        }
        eprintln!("judging arm `{arm}` ({} wavs) ...", judge_paths.len());
        let batch = judge.run_batch(&judge_paths)?;

        let (mut wers, mut cers, mut scored) = (Vec::new(), Vec::new(), 0usize);
        let mut arm_transcripts = Vec::new();
        for ((id, truth), jwav) in truths.iter().zip(&judge_paths) {
            match batch.text_for(jwav) {
                Some(hyp) => {
                    scored += 1;
                    let w = wer_with(truth, hyp, Mode::English);
                    wers.push(w);
                    cers.push(cer_with(truth, hyp, Mode::English));
                    arm_transcripts.push((id.clone(), w, hyp.to_string()));
                }
                None => {
                    eprintln!("  {id}: judge returned no transcript");
                    arm_transcripts.push((id.clone(), f64::NAN, String::new()));
                }
            }
        }
        let wer = wers.iter().sum::<f64>() / wers.len().max(1) as f64;
        let cer = cers.iter().sum::<f64>() / cers.len().max(1) as f64;
        results.push((arm.to_string(), wer, cer, scored));
        transcripts.push(arm_transcripts);
    }

    // Attribution: sentences where the arms scored differently, with both
    // judge transcripts — this names the words the phonemizer costs.
    println!("\nsentences where the arms differ (truth / espeak-arm / ours-arm):");
    for (e, o) in transcripts[0].iter().zip(transcripts[1].iter()) {
        if (e.1 - o.1).abs() > 1e-9 {
            let truth = &truths.iter().find(|(id, _)| *id == e.0).unwrap().1;
            println!(
                "  {}  esp {:.0}% / ours {:.0}%\n    txt: {truth}\n    esp: {}\n    our: {}",
                e.0,
                e.1 * 100.0,
                o.1 * 100.0,
                e.2,
                o.2
            );
        }
    }

    // 4. Verdict.
    println!("\nM-T1 SUBSTITUTION GATE — {} holdout sentences, zero-noise, judge: {}\n", truths.len(), judge.name);
    println!("{:<24} {:>8} {:>8} {:>8}", "ARM", "WER%", "CER%", "SCORED");
    for (arm, wer, cer, scored) in &results {
        println!(
            "{:<24} {:>8.2} {:>8.2} {:>8}",
            format!("piper <- {arm}"),
            wer * 100.0,
            cer * 100.0,
            scored
        );
    }
    let espeak_wer = results[0].1;
    let ours_wer = results[1].1;
    let limit = espeak_wer * BAND;
    let pass = ours_wer <= limit;
    println!(
        "\nband: ours {:.2}% vs espeak {:.2}% x {:.2} = {:.2}% -> {}",
        ours_wer * 100.0,
        espeak_wer * 100.0,
        BAND,
        limit * 100.0,
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(())
}
