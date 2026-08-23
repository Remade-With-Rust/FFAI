//! Run the TTS corpus bench and append a ledger line, WITHOUT going through
//! `ffai-cli`.
//!
//! The CLI currently does not build in this worktree: a concurrent session has
//! uncommitted Diana wiring in `crates/ffai-cli/src/main.rs` (`ffai_diana::
//! register`, `DetectOptions`) while `ffai-diana` is not among ffai-cli's
//! dependencies. That is another session's in-flight work and is not mine to
//! fix or revert, but the bench itself is a library call — `ffai_bench::tts::
//! run_tts` needs only a registry, a corpus, and the references file.
//!
//! Same corpus, same references, same ledger as `ffai bench tts`, so the record
//! it appends is the ordinary one.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example bench_tts
//! ```

use std::path::PathBuf;

use ffai_bench::reference::ReferenceFile;
use ffai_bench::runner::BenchConfig;
use ffai_core::registry::EngineRegistry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reg = EngineRegistry::default();
    ffai_mercury::register(&mut reg);

    // Both the TTS reference AND the round-trip judge: `run_tts` scores our
    // audio by transcribing it with a frozen third-party ASR (declared
    // `task = "tts-judge"`), which is what keeps the quality gate off
    // self-grading. Passing only `task = "tts"` refs makes it refuse to run.
    let refs = ReferenceFile::load(&PathBuf::from("corpora/references.toml"))?;
    let references: Vec<_> = refs
        .for_task("tts")
        .chain(refs.for_task("tts-judge"))
        .cloned()
        .collect();
    println!(
        "references: {:?}",
        references.iter().map(|r| &r.name).collect::<Vec<_>>()
    );

    let cfg = BenchConfig {
        engine: None,
        skip_engine: false,
        // NOT skipped: this example exists to run the TTS reference AND the
        // frozen third-party `tts-judge` ASR that scores our audio. Skipping
        // references here would turn the quality gate into self-grading, which
        // is the one thing the judge is there to prevent.
        skip_references: false,
        corpus: PathBuf::from("corpora/harvard-sentences-v1.toml"),
        references,
        scorers: Vec::new(),
        runs: std::env::var("FFAI_RUNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3),
        ledger: PathBuf::from("bench/ledger.jsonl"),
    };

    let record = ffai_bench::tts::run_tts(&reg, &cfg)?;
    println!("\n{}", serde_json::to_string_pretty(&record)?);
    Ok(())
}
