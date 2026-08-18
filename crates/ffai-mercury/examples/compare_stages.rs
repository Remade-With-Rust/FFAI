//! Function-by-function comparison: Mercury vs whisper.cpp, same clips.
//!
//! Both implementations report a stage breakdown over the same work, so they
//! can be lined up directly. whisper.cpp prints `whisper_print_timings` to
//! stderr; we read our own profiler. Both are totals over an identical clip
//! set at matched decode settings, so the columns are comparable.
//!
//! The output ranks by **absolute gap in milliseconds**, not by ratio: a
//! stage that is 3x slower but costs 6 ms is worth less attention than one
//! that is 1.6x slower and costs 95 ms. Ratios choose what is embarrassing;
//! absolute gaps choose what is worth fixing.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example compare_stages -- --clips 20
//! ```
//!
//! Requires the whisper.cpp reference from docs/benchmarking.md.

use std::path::{Path, PathBuf};
use std::process::Command;

use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;
use ffai_mercury::asr::profile;

/// Pull the `/ N runs` count out of whisper.cpp's timing block, summing.
/// Decode cost is only comparable per token — the two implementations do not
/// necessarily generate the same number of tokens for the same audio.
fn parse_runs(log: &str, key: &str) -> f64 {
    log.lines()
        .filter(|l| l.contains("whisper_print_timings") && l.contains(key))
        .filter_map(|l| {
            let after = l.split('/').nth(1)?;
            after.trim().split_whitespace().next()?.parse::<f64>().ok()
        })
        .sum()
}

/// Pull `key = <number>` out of whisper.cpp's timing block, summing repeats.
fn parse_timing(log: &str, key: &str) -> f64 {
    log.lines()
        .filter(|l| l.contains("whisper_print_timings") && l.contains(key))
        .filter_map(|l| {
            let after = l.split('=').nth(1)?;
            after.trim().split_whitespace().next()?.parse::<f64>().ok()
        })
        .sum()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| default.to_string())
    };
    let count: usize = arg("--clips", "20").parse()?;
    let bin = PathBuf::from(arg("--whisper-bin", ".whispercpp/whisper-cli.exe"));
    let model = PathBuf::from(arg("--whisper-model", ".whispercpp/ggml-tiny.en.bin"));
    let threads = arg("--threads", "24");

    // Same clips for both, in a stable order.
    let dir = Path::new("corpora/clips/librispeech-test-clean/audio");
    let mut clips: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "wav"))
        .collect();
    clips.sort();
    clips.truncate(count);
    if clips.is_empty() {
        return Err("no clips found — run the corpus prep first".into());
    }

    let media: f64 = clips
        .iter()
        .filter_map(|p| ffai_media::load_audio(p).ok())
        .map(|a| a.duration_secs())
        .sum();
    println!(
        "{} clips, {media:.1}s audio, {threads} threads, greedy\n",
        clips.len()
    );

    // ---- whisper.cpp: one invocation, model loaded once ----
    let out = Command::new(&bin)
        // NO -nt. That flag makes whisper.cpp suppress timestamp generation
        // (43 decode runs instead of 50 on a sample clip, 23 % less decode
        // work), so passing it would compare our timestamped decode against
        // its untimestamped one and hand it a free advantage.
        .args([
            "-m",
            &model.to_string_lossy(),
            "-t",
            &threads,
            "-bs",
            "1",
            "-bo",
            "1",
        ])
        .args(clips.iter().map(|p| p.to_string_lossy().into_owned()))
        .output()?;
    let log =
        String::from_utf8_lossy(&out.stderr).into_owned() + &String::from_utf8_lossy(&out.stdout);
    if !out.status.success() {
        return Err(format!(
            "whisper.cpp failed: {}",
            &log[log.len().saturating_sub(500)..]
        )
        .into());
    }
    let cpp = [
        ("mel", parse_timing(&log, "mel time")),
        ("encode", parse_timing(&log, "encode time")),
        (
            "decode",
            parse_timing(&log, "decode time") + parse_timing(&log, "batchd time"),
        ),
        ("sample", parse_timing(&log, "sample time")),
    ];
    let cpp_total: f64 = cpp.iter().map(|(_, v)| v).sum();

    // ---- Mercury: same clips, profiler on ----
    if !profile::is_enabled() {
        return Err("set FFAI_PROFILE=1 so our stage timings are collected".into());
    }
    let engine = WhisperCandle::new();
    // Warm-up: load weights and run the precision calibration outside the
    // measured region, exactly as the benchmark harness does.
    let first = ffai_media::load_audio(&clips[0])?;
    engine.transcribe(&first, &AsrOptions::default())?;
    let before = [
        profile::profile().mel.secs(),
        profile::profile().encoder.secs(),
        profile::profile().decoder.secs(),
        profile::profile().sampling.secs(),
    ];
    for clip in &clips {
        let audio = ffai_media::load_audio(clip)?;
        engine.transcribe(&audio, &AsrOptions::default())?;
    }
    let p = profile::profile();
    let ours = [
        ("mel", (p.mel.secs() - before[0]) * 1e3),
        ("encode", (p.encoder.secs() - before[1]) * 1e3),
        ("decode", (p.decoder.secs() - before[2]) * 1e3),
        ("sample", (p.sampling.secs() - before[3]) * 1e3),
    ];
    let our_total: f64 = ours.iter().map(|(_, v)| v).sum();

    // ---- side by side ----
    println!(
        "{:<10} {:>12} {:>12} {:>10} {:>12}",
        "STAGE", "whisper.cpp", "mercury", "RATIO", "GAP ms"
    );
    let mut gaps = Vec::new();
    for ((name, c), (_, m)) in cpp.iter().zip(ours.iter()) {
        let ratio = if *c > 0.0 { m / c } else { f64::NAN };
        let gap = m - c;
        println!("{name:<10} {c:>12.1} {m:>12.1} {ratio:>9.2}x {gap:>12.1}");
        gaps.push((gap, *name, ratio, *c, *m));
    }
    println!(
        "{:<10} {cpp_total:>12.1} {our_total:>12.1} {:>9.2}x {:>12.1}",
        "TOTAL",
        our_total / cpp_total,
        our_total - cpp_total
    );
    println!(
        "\nxRT: whisper.cpp {:.1}x · mercury {:.1}x",
        media / (cpp_total / 1e3),
        media / (our_total / 1e3)
    );

    // Per-token normalization for decode: the stage totals above are only
    // comparable if both sides generated the same number of tokens.
    let cpp_tokens = parse_runs(&log, "decode time") + parse_runs(&log, "batchd time");
    let our_tokens = p.tokens.load(std::sync::atomic::Ordering::Relaxed) as f64;
    if cpp_tokens > 0.0 && our_tokens > 0.0 {
        let (c, m) = (cpp[2].1 / cpp_tokens, ours[2].1 / our_tokens);
        println!(
            "
decode normalized: whisper.cpp {cpp_tokens:.0} tokens at {c:.2} ms/token ·              mercury {our_tokens:.0} tokens at {m:.2} ms/token  ->  {:.2}x per token",
            m / c
        );
        if (our_tokens - cpp_tokens).abs() / cpp_tokens > 0.1 {
            println!(
                "  NOTE: token counts differ by {:.0}% — the stage total above overstates                  the per-token gap.",
                (our_tokens / cpp_tokens - 1.0) * 100.0
            );
        }
    }

    gaps.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("\nWHERE WE LOSE (ranked by absolute cost, not ratio)");
    for (gap, name, ratio, c, m) in &gaps {
        if *gap <= 0.0 {
            println!(
                "  {name:<10} we WIN by {:.1} ms ({:.2}x faster)",
                -gap,
                c / m
            );
        } else {
            println!(
                "  {name:<10} +{gap:>7.1} ms  ({ratio:.2}x slower)  — {:.0}% of the total gap",
                gap / (our_total - cpp_total) * 100.0
            );
        }
    }
    Ok(())
}
