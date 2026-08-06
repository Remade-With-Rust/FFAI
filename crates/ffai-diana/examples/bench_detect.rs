//! Run `ffai bench detect` for Diana without going through `ffai-cli`.
//!
//! The CLI links every component crate, so a bench run there is hostage to
//! any of them compiling. This example links only `ffai-bench` +
//! `ffai-diana` — the two crates the measurement actually needs — and
//! writes the same ledger line to the same file through the same
//! `runner::run_detect`. It is the detection sibling of
//! `ffai-carmenta/examples/live_bench.rs`, which exists for the same reason.
//!
//! ```sh
//! cargo run --release -p ffai-diana --example bench_detect -- \
//!     corpora/diana-coco-v1.toml [--baseline-only] [--runs N]
//! ```

use std::path::PathBuf;

use ffai_bench::reference::ReferenceFile;
use ffai_bench::runner::{render, run_detect, BenchConfig};
use ffai_core::registry::EngineRegistry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let corpus = PathBuf::from(
        args.next().unwrap_or_else(|| "corpora/diana-coco-v1.toml".to_string()),
    );
    let rest: Vec<String> = args.collect();
    let baseline_only = rest.iter().any(|a| a == "--baseline-only");
    let runs = rest
        .iter()
        .position(|a| a == "--runs")
        .and_then(|i| rest.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(3usize);
    let only: Vec<String> = rest
        .iter()
        .position(|a| a == "--only")
        .and_then(|i| rest.get(i + 1))
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    // Which tier/geometry to bench: `yolo26n` (default), `yolo26s`,
    // `yolo26s-square`, ... The harness derives the comparison key from this
    // name, so it is also what decides which reference row is "matched".
    let engine: Option<String> = rest
        .iter()
        .position(|a| a == "--engine")
        .and_then(|i| rest.get(i + 1))
        .cloned();

    let refs = PathBuf::from("corpora/references.toml");
    let references = if refs.exists() {
        ReferenceFile::load(&refs)?
            .for_task("detect")
            .filter(|r| only.is_empty() || only.contains(&r.name))
            .cloned()
            .collect()
    } else {
        eprintln!("note: no {} — running without references", refs.display());
        Vec::new()
    };

    let mut reg = EngineRegistry::new();
    ffai_diana::register(&mut reg);

    let cfg = BenchConfig {
        engine,
        skip_engine: baseline_only,
        // `--engine-only` is the CLI's flag; this example always wants the
        // references, so it opts out of skipping them explicitly rather than
        // inheriting whatever the field's default becomes later.
        skip_references: false,
        corpus,
        references,
        runs,
        ledger: PathBuf::from("bench/ledger.jsonl"),
    };
    let record = run_detect(&reg, &cfg)?;
    print!("{}", render(&record));
    println!("appended to bench/ledger.jsonl");
    Ok(())
}
