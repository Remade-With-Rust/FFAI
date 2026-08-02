//! `ffai bench ocr`, without `ffai-cli`.
//!
//! Identical code path — same `ffai_bench::runner::run_ocr`, same corpus
//! verification, same four gates, same ledger record — reached through
//! `ffai-carmenta` alone so a sibling campaign mid-edit in the CLI cannot block
//! the audit trail. (It has, three times.) Carmenta is the only registry this
//! needs, which is also why it cannot bench Mercury or Argus; use the CLI for
//! those once it builds.
//!
//! Usage: `bench_ocr <corpus.toml> <engine> [reference ...]`

use ffai_bench::reference::ReferenceFile;
use ffai_bench::runner::{render, run_ocr, BenchConfig};
use ffai_core::registry::EngineRegistry;
use std::path::{Path, PathBuf};

fn main() {
    let mut args = std::env::args().skip(1);
    let corpus = PathBuf::from(args.next().expect("usage: bench_ocr <corpus> <engine> [refs...]"));
    let engine = args.next().expect("usage: bench_ocr <corpus> <engine> [refs...]");
    // `none` means NO references. Omitting the list means ALL of them, which is
    // the harness's contract and a trap in a script: four "engine only" arms
    // silently pulled tesseract, easyocr, paddle, PP-Structure AND Unlimited-OCR
    // — hours of reference time per arm for numbers already measured.
    let only: Vec<String> = args.collect();
    let no_refs = only.len() == 1 && only[0] == "none";

    let mut reg = EngineRegistry::new();
    ffai_carmenta::register(&mut reg);

    let refs_path = Path::new("corpora/references.toml");
    let references = if no_refs {
        Vec::new()
    } else {
        match ReferenceFile::load(refs_path) {
            Ok(f) => f
                .for_task("ocr")
                .filter(|r| only.is_empty() || only.iter().any(|n| *n == r.name))
                .cloned()
                .collect(),
            Err(e) => {
                eprintln!("note: no references at {}: {e}", refs_path.display());
                Vec::new()
            }
        }
    };

    let cfg = BenchConfig {
        engine: Some(engine),
        skip_engine: false,
        corpus,
        references,
        // Best-of-1: CER is deterministic, and the timings this suite produces
        // are already caveated by whatever else the box is doing.
        runs: std::env::var("FFAI_BENCH_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(1),
        ledger: PathBuf::from("bench/ledger.jsonl"),
    };
    match run_ocr(&reg, &cfg) {
        Ok(record) => {
            print!("{}", render(&record));
            println!("appended to bench/ledger.jsonl");
        }
        Err(e) => {
            eprintln!("bench failed: {e}");
            std::process::exit(1);
        }
    }
}
