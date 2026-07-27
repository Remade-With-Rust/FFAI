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
        /// Output file; `.srt` extension selects subtitle format (default: stdout text)
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
        /// Speaker diarization
        #[arg(long)]
        diarize: bool,
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
    },
    /// Recognize text in an image (Carmenta)
    Ocr {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(long)]
        engine: Option<String>,
        /// Language hints, repeatable
        #[arg(long)]
        language: Vec<String>,
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
        /// Task to bench (Phase 0: asr)
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
        Cmd::Asr { input, output, engine, language, word_timestamps, diarize } => {
            let audio = ffai_media::load_audio(&input)?;
            let opts = AsrOptions { language, word_timestamps, diarize, translate: false };
            let transcript = reg.asr(engine.as_deref())?.transcribe(&audio, &opts)?;
            match output {
                Some(path) if path.extension().and_then(|e| e.to_str()) == Some("srt") => {
                    std::fs::write(&path, transcript.to_srt())?;
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
        Cmd::Tts { text, output, engine, voice } => {
            let opts = TtsOptions { voice, ..Default::default() };
            let audio = reg.tts(engine.as_deref())?.synthesize(&text, &opts)?;
            ffai_media::save_wav(&output, &audio)?;
            println!("wrote {}", output.display());
        }
        Cmd::Ocr { input, engine, language } => {
            let image = ffai_media::load_image(&input)?;
            let opts = OcrOptions { languages: language };
            let out = reg.ocr(engine.as_deref())?.recognize(&image, &opts)?;
            println!("{}", out.text());
        }
        Cmd::Caption { input, prompt, engine } => {
            let image = ffai_media::load_image(&input)?;
            let opts = VlmOptions { prompt, max_new_tokens: None };
            let caption = reg.vlm(engine.as_deref())?.describe_image(&image, &opts)?;
            println!("{caption}");
        }
        Cmd::Bench { task, corpus, refs, engine, only, baseline_only, runs, ledger } => {
            if Task::from_str(&task).map_err(anyhow::Error::msg)? != Task::Asr {
                anyhow::bail!(
                    "`ffai bench {task}` is not wired yet — asr is the first bench vertical; \
                     tts/ocr/vlm follow their engines (see ROADMAP.md)"
                );
            }
            let references: Vec<_> = if refs.exists() {
                let selected: Vec<_> = ffai_bench::reference::ReferenceFile::load(&refs)?
                    .for_task("asr")
                    .filter(|r| only.is_empty() || only.contains(&r.name))
                    .cloned()
                    .collect();
                for name in &only {
                    if !selected.iter().any(|r| &r.name == name) {
                        anyhow::bail!("--only {name}: no such reference in {}", refs.display());
                    }
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
            let record = ffai_bench::runner::run_asr(&reg, &cfg)?;
            print!("{}", ffai_bench::runner::render(&record));
            println!("appended to {}", ledger.display());
        }
    }
    Ok(())
}
