//! Mercury vs whisper.cpp at the SAME model size, scored by the same code.
//!
//! The size ladder shows small.en more than halving tiny.en's error rate.
//! That is a claim about the MODEL, and quoting it against whisper.cpp's
//! tiny.en would be the mismatched-reference error `corpora/references.toml`
//! exists to prevent — it would price the weights, not the implementation.
//!
//! This runs both implementations over the same clips at the same size and
//! scores both through `ffai_bench::metrics`, which normalizes each side
//! identically. Whatever difference remains is ours.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example matched_size -- \
//!     corpora/librispeech-test-clean-v2.toml whisper-small-en .whispercpp/ggml-small.en.bin
//! ```
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use ffai_bench::corpus::Manifest;
use ffai_bench::metrics::{cer, wer};
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::text_decoder::Precision;
use ffai_mercury::asr::WhisperCandle;

/// Length-weighted corpus WER/CER, matching how the bench harness aggregates.
fn score(pairs: &[(String, String)]) -> (f64, f64) {
    let (mut wn, mut wd, mut cn, mut cd) = (0.0, 0.0, 0.0, 0.0);
    for (truth, hyp) in pairs {
        let words = truth.split_whitespace().count() as f64;
        let chars = truth.chars().count() as f64;
        wn += wer(truth, hyp) * words;
        wd += words;
        cn += cer(truth, hyp) * chars;
        cd += chars;
    }
    (wn / wd * 100.0, cn / cd * 100.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let corpus = a.next().unwrap_or("corpora/librispeech-test-clean-v2.toml".into());
    let model = a.next().unwrap_or("whisper-small-en".into());
    let ggml = a.next().unwrap_or(".whispercpp/ggml-small.en.bin".into());
    let limit: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let beam = std::env::var("FFAI_BEAM_SIZE").ok().unwrap_or_else(|| "1".into());

    let manifest = Manifest::load(&PathBuf::from(&corpus))?;
    let clips: Vec<_> = manifest.clips.iter().take(limit).collect();

    let mut truths = Vec::new();
    let mut paths = Vec::new();
    let mut audio_secs = 0.0f64;
    for c in &clips {
        let p = manifest.clip_path(c);
        let (Ok(audio), Ok(Some(truth))) =
            (ffai_media::load_audio(&p), manifest.ground_truth(c))
        else {
            continue;
        };
        audio_secs += audio.duration_secs();
        truths.push(truth);
        paths.push(p);
    }
    println!("corpus {corpus} · {} clips · {audio_secs:.1}s audio · beam {beam}\n", paths.len());

    // ---- Mercury ----
    let engine = WhisperCandle::with_model("models", model.as_str(), Precision::F32);
    let opts = AsrOptions::default();
    engine.transcribe(&ffai_media::load_audio(&paths[0])?, &opts)?; // warm: load + calibrate
    let mut ours = Vec::new();
    let t = Instant::now();
    for p in &paths {
        let audio = ffai_media::load_audio(p)?;
        ours.push(engine.transcribe(&audio, &opts)?.text());
    }
    let our_secs = t.elapsed().as_secs_f64();

    // ---- whisper.cpp, one invocation so the model loads once ----
    let list = std::env::temp_dir().join("matched_size_files.txt");
    let mut f = std::fs::File::create(&list)?;
    for p in &paths {
        writeln!(f, "{}", p.canonicalize()?.display())?;
    }
    drop(f);
    let t = Instant::now();
    let out = Command::new(".venv-bench/Scripts/python.exe")
        .args(["corpora/refs/whisper_cpp_ref.py", "--batch"])
        .arg(&list)
        .args(["--bin", ".whispercpp/whisper-cli.exe", "--model", &ggml])
        .args(["--threads", "24", "--beam-size", &beam])
        .output()?;
    let cpp_secs = t.elapsed().as_secs_f64();
    if !out.status.success() {
        return Err(format!("whisper.cpp failed: {}", String::from_utf8_lossy(&out.stderr)).into());
    }
    // Pair BY PATH, never by index. Matching on position assumes the
    // reference emits results in the order the batch file listed them, and
    // when that assumption broke this harness reported whisper.cpp at
    // 20.79 % WER — a mature reference decoding six times worse than itself,
    // which is the implausible magnitude that gives a broken probe away.
    // Path-matched, the same transcripts score 3.06 %.
    let stem = |p: &std::path::Path| {
        p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
    };
    let mut by_stem: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        let (Some(path), Some(text)) = (v.get("path").and_then(|p| p.as_str()), v.get("text"))
        else {
            continue;
        };
        by_stem.insert(
            stem(std::path::Path::new(path)),
            text.as_str().unwrap_or_default().to_string(),
        );
    }
    let mut theirs: Vec<String> = Vec::new();
    for p in &paths {
        match by_stem.get(&stem(p)) {
            Some(t) => theirs.push(t.clone()),
            // A missing clip means the two arms did not do identical work,
            // so the comparison is void rather than merely incomplete.
            None => {
                return Err(format!(
                    "whisper.cpp returned no transcript for {} — arms are not comparable",
                    p.display()
                )
                .into())
            }
        }
    }

    let ours_pairs: Vec<_> = truths.iter().cloned().zip(ours).collect();
    let theirs_pairs: Vec<_> = truths.iter().cloned().zip(theirs).collect();
    let (ow, oc) = score(&ours_pairs);
    let (tw, tc) = score(&theirs_pairs);

    println!("{:<26} {:>8} {:>8} {:>9}", "IMPLEMENTATION", "WER%", "CER%", "xRT");
    println!("{:<26} {ow:>8.2} {oc:>8.2} {:>9.1}", format!("mercury ({model})"), audio_secs / our_secs);
    println!(
        "{:<26} {tw:>8.2} {tc:>8.2} {:>9.1}",
        "whisper.cpp (same size)",
        audio_secs / cpp_secs
    );
    println!("\nWER delta {:+.2} pp · CER delta {:+.2} pp (negative = Mercury ahead)", ow - tw, oc - tc);

    // Per-clip counts: a corpus delta this small is not a result on its own.
    let (mut better, mut worse) = (0, 0);
    for ((t, o), (_, p)) in ours_pairs.iter().zip(theirs_pairs.iter()) {
        let (a, b) = (wer(t, o), wer(t, p));
        if a < b - 1e-9 {
            better += 1
        } else if a > b + 1e-9 {
            worse += 1
        }
    }
    let moved = better + worse;
    let z = if moved > 0 {
        (better as f64 - moved as f64 / 2.0) / (0.5 * (moved as f64).sqrt())
    } else {
        0.0
    };
    println!(
        "per-clip: mercury better {better} · worse {worse} · tied {} · sign z = {z:+.2}{}",
        ours_pairs.len() - moved,
        if z.abs() > 2.0 { "  SIGNIFICANT" } else { "  (bar |z| > 2)" }
    );
    Ok(())
}
