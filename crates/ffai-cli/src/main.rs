//! The `ffai` binary — a thin shell over the FFai library crates, in the way
//! the `ffmpeg` binary is a thin shell over libav*. All real logic lives in
//! the crates so any application can embed FFai without this CLI.

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ffai_core::engine::{AsrOptions, OcrOptions, Task, TtsOptions, VlmOptions};
use ffai_core::registry::EngineRegistry;

#[derive(Parser)]
#[command(
    name = "ffai",
    version,
    about = "FFai — the AI media toolkit, remade with rust",
    long_about = "FFai — OCR, ASR, TTS, and vision-language understanding in one \
                  pure-Rust toolkit.\n\nComponents: Mercury (voice), Carmenta (OCR), \
                  Argus (vision). Engines are swappable per task, like codecs in ffmpeg."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List registered engines and their status (like `ffmpeg -codecs`)
    Engines {
        /// Filter by task: asr, tts, ocr, vlm
        #[arg(long)]
        task: Option<String>,
    },
    /// List model manifests and their cache status
    Models {
        /// Manifest directory
        #[arg(long, default_value = "models")]
        dir: PathBuf,
        /// Download this model's files into the cache. Do this BEFORE a
        /// measured run — a download inside a timed region is not a
        /// measurement (see docs/benchmarking.md).
        #[arg(long)]
        fetch: Option<String>,
    },
    /// Transcribe speech to text (Mercury)
    Asr {
        /// Input audio file (Phase 0: WAV)
        #[arg(short, long)]
        input: PathBuf,
        /// Output file; `.srt`/`.vtt`/`.json` select the format (default: stdout text)
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long)]
        engine: Option<String>,
        /// Force a language (e.g. en) instead of auto-detecting
        #[arg(long)]
        language: Option<String>,
        /// Word-level timestamps (WhisperX-style alignment)
        #[arg(long)]
        word_timestamps: bool,
        /// Speaker diarization — label who spoke when
        #[arg(long)]
        diarize: bool,
        /// Known number of speakers; overrides the clustering threshold
        #[arg(long)]
        max_speakers: Option<usize>,
        /// Speaker clustering distance threshold, 0..2 (higher = fewer speakers)
        #[arg(long, default_value_t = 0.80)]
        diarize_threshold: f32,
        /// Speech segmentation is on by default; this flag is an accepted no-op
        #[arg(long)]
        vad: bool,
        /// Transcribe the raw fixed 30 s grid instead, without speech segmentation
        #[arg(long, conflicts_with = "vad")]
        no_vad: bool,
        /// VAD speech threshold, 0..1 — higher is stricter
        #[arg(long, default_value_t = 0.5)]
        vad_threshold: f32,
        /// Pack speech into windows of at most this many seconds
        #[arg(long, default_value_t = 30.0)]
        vad_chunk_secs: f32,
    },
    /// Synthesize speech from text (Mercury)
    Tts {
        /// Text to speak
        text: String,
        /// Output WAV path
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        engine: Option<String>,
        #[arg(long)]
        voice: Option<String>,
        /// Playback-rate multiplier (1.0 = normal)
        #[arg(long, default_value_t = 1.0)]
        speed: f32,
        /// Acoustic variation, 0.0 = fully deterministic (default: the voice's own)
        #[arg(long)]
        noise_scale: Option<f32>,
        /// Duration variation, 0.0 = deterministic timing (default: the voice's own)
        #[arg(long)]
        noise_w: Option<f32>,
        /// Noise seed — same seed + same text = byte-identical audio
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Silence between sentences of long-form input, seconds
        #[arg(long, default_value_t = 0.2)]
        sentence_silence: f32,
    },
    /// Recognize text in an image, or stream a frame sequence (Carmenta)
    Ocr {
        /// An image file — or, with --live, a directory of frames (sorted by
        /// filename). Video containers arrive with the rff frame iterator.
        #[arg(short, long)]
        input: PathBuf,
        #[arg(long)]
        engine: Option<String>,
        /// Language hints, repeatable
        #[arg(long)]
        language: Vec<String>,
        /// LIVE mode: treat the input as a frame sequence and emit a timed
        /// text track (mission plan §4.1)
        #[arg(long)]
        live: bool,
        /// Frame rate of the input sequence (LIVE timing)
        #[arg(long, default_value_t = 3.0)]
        fps: f64,
        /// Change-gate: fraction of pixels that must move > 8 grey levels
        /// for a frame to count as changed (below it, the previous result is
        /// reused at zero model cost)
        #[arg(long, default_value_t = 0.0005)]
        change_fraction: f32,
        /// Process every Nth frame
        #[arg(long, default_value_t = 1)]
        sample_every: usize,
        /// Output for LIVE mode; `.srt`/`.vtt` select the format
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Watch the input directory in REAL TIME: process frames as any
        /// capture tool writes them (OBS, `ffmpeg -f gdigrab`, ...); stop
        /// after this many seconds of no new frames. The v1 live source —
        /// rff video ingest slots in behind ffai-media later.
        #[arg(long)]
        watch: Option<f64>,
    },
    /// Caption / describe an image (Argus)
    Caption {
        #[arg(short, long)]
        input: PathBuf,
        /// Instruction prompt (default: plain captioning)
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long)]
        engine: Option<String>,
    },
    /// Benchmark an engine against world standards on a pinned corpus
    Bench {
        /// Task to bench: asr or ocr (tts/vlm follow their engines)
        task: String,
        /// Corpus manifest (corpora/*.toml)
        #[arg(long)]
        corpus: PathBuf,
        /// References file declaring external standards to compare against
        #[arg(long, default_value = "corpora/references.toml")]
        refs: PathBuf,
        #[arg(long)]
        engine: Option<String>,
        /// Compare against only these references (repeatable). Use to keep a
        /// run apples-to-apples, e.g. matching decode strategies.
        #[arg(long = "only")]
        only: Vec<String>,
        /// Baseline the references only (useful before our engine is live)
        #[arg(long)]
        baseline_only: bool,
        /// Timed repetitions per clip (best-of-N)
        #[arg(long, default_value_t = 3)]
        runs: usize,
        /// Claims ledger to append to
        #[arg(long, default_value = "bench/ledger.jsonl")]
        ledger: PathBuf,
    },
}

/// Compose the default registry from every feature crate.
fn build_registry() -> EngineRegistry {
    let mut reg = EngineRegistry::new();
    ffai_mercury::register(&mut reg);
    ffai_carmenta::register(&mut reg);
    ffai_argus::register(&mut reg);
    reg
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let reg = build_registry();

    match cli.cmd {
        Cmd::Engines { task } => {
            let filter = task
                .map(|t| Task::from_str(&t).map_err(anyhow::Error::msg))
                .transpose()?;
            println!("{:<6} {:<16} {:<13} DESCRIPTION", "TASK", "ENGINE", "STATUS");
            for info in reg.list() {
                if filter.is_some_and(|t| t != info.task) {
                    continue;
                }
                println!(
                    "{:<6} {:<16} {:<13} {}",
                    info.task.to_string(),
                    info.name,
                    info.status.to_string(),
                    info.description
                );
            }
        }
        Cmd::Models { dir, fetch } => {
            let manifests = ffai_models::load_dir(&dir)
                .with_context(|| format!("reading manifests from {}", dir.display()))?;
            if let Some(name) = fetch {
                let manifest = manifests
                    .iter()
                    .find(|m| m.name == name)
                    .with_context(|| format!("no model manifest named `{name}` in {}", dir.display()))?;
                println!("fetching {} ({})...", manifest.name, manifest.license);
                let resolved = manifest.fetch()?;
                for (file, path) in &resolved.files {
                    println!("  {file} -> {}", path.display());
                }
                return Ok(());
            }
            println!("{:<6} {:<20} {:<14} {:<7} SOURCE", "TASK", "MODEL", "LICENSE", "CACHED");
            for m in &manifests {
                println!(
                    "{:<6} {:<20} {:<14} {:<7} {}",
                    m.task,
                    m.name,
                    m.license,
                    if m.is_cached() { "yes" } else { "no" },
                    m.hf_repo.as_deref().unwrap_or("-")
                );
            }
            println!("\ncache root: {}", ffai_models::cache_dir().display());
        }
        Cmd::Asr {
            input,
            output,
            engine,
            language,
            word_timestamps,
            diarize,
            max_speakers,
            diarize_threshold,
            vad,
            no_vad,
            vad_threshold,
            vad_chunk_secs,
        } => {
            if let Some(0) = max_speakers {
                anyhow::bail!("--max-speakers must be at least 1");
            }
            if !(0.0..=2.0).contains(&diarize_threshold) {
                anyhow::bail!(
                    "--diarize-threshold is a cosine distance and must be in 0..=2, got \
                     {diarize_threshold}"
                );
            }
            if !(0.0..=1.0).contains(&vad_threshold) {
                anyhow::bail!("--vad-threshold must be in 0..=1, got {vad_threshold}");
            }
            if vad_chunk_secs <= 0.0 || vad_chunk_secs > 30.0 {
                anyhow::bail!(
                    "--vad-chunk-secs must be in (0, 30]; Whisper's context is 30 s and a \
                     longer window cannot be represented (got {vad_chunk_secs})"
                );
            }
            // VAD is on by default for its measured speed (its WER movement
            // is z = 0.00 per clip, NOT a quality mechanism — see
            // AsrOptions::vad), so `--vad` is an accepted no-op and
            // `--no-vad` is the switch that does something. Turning it off
            // while asking for a stage that needs speech boundaries is still
            // a contradiction rather than a silent degradation.
            let _ = vad;
            if no_vad && (word_timestamps || diarize) {
                anyhow::bail!(
                    "--no-vad conflicts with --word-timestamps/--diarize, which need speech \
                     segmentation to work. Drop --no-vad, or drop the stage."
                );
            }
            let vad_on = !no_vad;

            let audio = ffai_media::load_audio(&input)?;
            let opts = AsrOptions {
                language,
                word_timestamps,
                diarize,
                // A one-shot CLI run has no next call to persist into; the
                // registry is for embedders processing a stream.
                persist_speakers: false,
                max_speakers,
                diarize_threshold,
                translate: false,
                vad: vad_on,
                vad_threshold,
                vad_chunk_secs,
                // A one-shot file run starts at the beginning; the absolute
                // window grid only matters to streaming callers.
                stream_offset_secs: 0.0,
            };
            let transcript = reg.asr(engine.as_deref())?.transcribe(&audio, &opts)?;
            match output {
                // Format follows the extension, as it does for `ffmpeg -o`.
                Some(path)
                    if matches!(
                        path.extension().and_then(|e| e.to_str()),
                        Some("srt" | "vtt" | "json")
                    ) =>
                {
                    let body = match path.extension().and_then(|e| e.to_str()) {
                        Some("srt") => transcript.to_srt(),
                        Some("vtt") => transcript.to_vtt(),
                        _ => transcript.to_json(),
                    };
                    std::fs::write(&path, body)?;
                    println!("wrote {}", path.display());
                }
                Some(path) => {
                    std::fs::write(&path, transcript.text())?;
                    println!("wrote {}", path.display());
                }
                None => println!("{}", transcript.text()),
            }
            if ffai_mercury::asr::profile::is_enabled() {
                eprint!("{}", ffai_mercury::asr::profile::profile().report());
            }
            if let Some(audit) = ffai_mercury::asr::vocab_int8::audit_report() {
                eprintln!("
{audit}");
            }
        }
        Cmd::Tts {
            text,
            output,
            engine,
            voice,
            speed,
            noise_scale,
            noise_w,
            seed,
            sentence_silence,
        } => {
            let opts = TtsOptions {
                voice,
                speed,
                noise_scale,
                noise_w,
                seed,
                sentence_silence_s: sentence_silence,
            };
            let audio = reg.tts(engine.as_deref())?.synthesize(&text, &opts)?;
            ffai_media::save_wav(&output, &audio)?;
            println!("wrote {}", output.display());
        }
        Cmd::Ocr { input, engine, language, live, fps, change_fraction, sample_every, output, watch } => {
            let opts = OcrOptions { languages: language, ..Default::default() };
            let eng = reg.ocr(engine.as_deref())?;
            if live {
                if fps <= 0.0 {
                    anyhow::bail!("--fps must be positive");
                }
                let list_frames = |seen: usize| -> Result<Vec<PathBuf>> {
                    let mut frames: Vec<PathBuf> = std::fs::read_dir(&input)
                        .with_context(|| format!("reading frame dir {}", input.display()))?
                        .filter_map(|e| e.ok().map(|e| e.path()))
                        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
                        .collect();
                    frames.sort();
                    Ok(frames.split_off(seen.min(frames.len())))
                };
                let cfg = ffai_carmenta::live::LiveConfig { change_fraction, sample_every, ..Default::default() };
                let mut session = ffai_carmenta::live::LiveSession::new(eng.clone(), opts, cfg);
                let mut n = 0usize;
                let started = std::time::Instant::now();
                let mut last_new = std::time::Instant::now();
                loop {
                    let fresh = list_frames(n)?;
                    if fresh.is_empty() {
                        match watch {
                            // Watch mode: wait for the capture tool, stop
                            // after `idle` seconds without new frames.
                            Some(idle) if last_new.elapsed().as_secs_f64() < idle => {
                                std::thread::sleep(std::time::Duration::from_millis(100));
                                continue;
                            }
                            _ => break,
                        }
                    }
                    for frame in &fresh {
                        let img = ffai_media::load_image(frame)?;
                        // Watch mode timestamps from the wall clock (frames
                        // arrive in real time); batch mode from --fps.
                        let t = if watch.is_some() { started.elapsed().as_secs_f64() } else { n as f64 / fps };
                        session.push_frame(&img, t)?;
                        n += 1;
                    }
                    last_new = std::time::Instant::now();
                    if n > 0 && watch.is_none() && list_frames(n)?.is_empty() {
                        break;
                    }
                }
                if n == 0 {
                    anyhow::bail!("no .png frames in {}", input.display());
                }
                let end_t = if watch.is_some() { started.elapsed().as_secs_f64() } else { n as f64 / fps };
                let (segments, stats) = session.finish(end_t);
                eprintln!(
                    "{n} frames: {} OCR calls, {} change-gated, {} sampled out; \
                     p50 {:.0} ms / p95 {:.0} ms per call",
                    stats.ocr_calls,
                    stats.gated,
                    stats.sampled_out,
                    stats.percentile(0.50).unwrap_or(0.0) * 1000.0,
                    stats.percentile(0.95).unwrap_or(0.0) * 1000.0,
                );
                let body = match output.as_ref().and_then(|p| p.extension()).and_then(|e| e.to_str()) {
                    Some("vtt") => ffai_carmenta::live::to_vtt(&segments),
                    _ => ffai_carmenta::live::to_srt(&segments),
                };
                match output {
                    Some(path) => {
                        std::fs::write(&path, body)?;
                        println!("wrote {}", path.display());
                    }
                    None => print!("{body}"),
                }
            } else {
                let image = ffai_media::load_image(&input)?;
                let out = eng.recognize(&image, &opts)?;
                println!("{}", out.text());
            }
            if ffai_carmenta::profile::is_enabled() {
                eprint!("{}", ffai_carmenta::profile::profile().report());
            }
        }
        Cmd::Caption { input, prompt, engine } => {
            let image = ffai_media::load_image(&input)?;
            let opts = VlmOptions { prompt, max_new_tokens: None };
            let caption = reg.vlm(engine.as_deref())?.describe_image(&image, &opts)?;
            println!("{caption}");
        }
        Cmd::Bench { task, corpus, refs, engine, only, baseline_only, runs, ledger } => {
            let task = Task::from_str(&task).map_err(anyhow::Error::msg)?;
            if !matches!(task, Task::Asr | Task::Ocr | Task::Tts) {
                anyhow::bail!(
                    "`ffai bench {task}` is not wired yet — asr, ocr and tts are the live bench \
                     verticals; vlm follows its engine (see ROADMAP.md)"
                );
            }
            let task_name = task.to_string();
            let references: Vec<_> = if refs.exists() {
                let file = ffai_bench::reference::ReferenceFile::load(&refs)?;
                let mut selected: Vec<_> = file
                    .for_task(&task_name)
                    .filter(|r| only.is_empty() || only.contains(&r.name))
                    .cloned()
                    .collect();
                for name in &only {
                    if !selected.iter().any(|r| &r.name == name) {
                        anyhow::bail!("--only {name}: no such reference in {}", refs.display());
                    }
                }
                // The TTS round-trip judge is harness infrastructure, not a
                // compared implementation — `--only` never filters it out.
                if task == Task::Tts {
                    selected.extend(file.for_task("tts-judge").cloned());
                }
                selected
            } else {
                eprintln!(
                    "note: no references file at {} — running without world-standard baselines",
                    refs.display()
                );
                Vec::new()
            };
            let cfg = ffai_bench::runner::BenchConfig {
                engine,
                skip_engine: baseline_only,
                corpus,
                references,
                runs,
                ledger: ledger.clone(),
            };
            let record = match task {
                Task::Asr => ffai_bench::runner::run_asr(&reg, &cfg)?,
                Task::Ocr => ffai_bench::runner::run_ocr(&reg, &cfg)?,
                Task::Tts => ffai_bench::tts::run_tts(&reg, &cfg)?,
                _ => unreachable!("guarded above"),
            };
            print!("{}", ffai_bench::runner::render(&record));
            println!("appended to {}", ledger.display());
        }
    }
    Ok(())
}
