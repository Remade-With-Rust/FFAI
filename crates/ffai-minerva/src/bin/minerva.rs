//! `minerva` — Minerva's command line. Today: `minerva roofline`.
//!
//! ```text
//! minerva roofline [--gguf PATH] [--mib N] [--reps N] [--threads 1,2,4,8]
//! ```

#![forbid(unsafe_code)]

use std::process::ExitCode;

use ffai_minerva::roofline::{self, ByteCensus};
use rusty_ternary::gguf::Gguf;

const DEFAULT_GGUF: &str = "F:/coding/bitnet-golden/ours-i2_s.gguf";

fn arg<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn mb(b: u64) -> f64 {
    b as f64 / 1e6
}

fn print_census(path: &str, c: &ByteCensus) {
    println!("== bytes per token: {path}");
    println!("  ternary  {:>9.1} MB  ({} I2_S tensors, {} layers)", mb(c.ternary), c.ternary_tensors, c.layers);
    println!(
        "  head     {:>9.1} MB  ({} x {}, type {}, tied={})",
        mb(c.head),
        c.head_shape.0,
        c.head_shape.1,
        c.head_type,
        c.tied
    );
    println!("  norms    {:>9.3} MB   embed row {} B   other {} B", mb(c.norms), c.embed_row, c.other);
    println!("  KV (f16) {:>9.1} KB per cached position", c.kv_per_position(2) as f64 / 1e3);
    let base = c.per_token(c.head, 0, 2);
    println!(
        "  total @pos 0: {:.1} MB   (head is {:.1}% of it)",
        mb(base),
        100.0 * c.head as f64 / base as f64
    );
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("roofline") {
        return Err("usage: minerva roofline [--gguf PATH] [--mib N] [--reps N] [--threads 1,2,4]".into());
    }
    let path = arg(&args, "--gguf").unwrap_or(DEFAULT_GGUF);
    let mib: usize = arg(&args, "--mib").unwrap_or("1024").parse().map_err(|e| format!("--mib: {e}"))?;
    let reps: usize = arg(&args, "--reps").unwrap_or("21").parse().map_err(|e| format!("--reps: {e}"))?;
    let threads: Vec<usize> = arg(&args, "--threads")
        .unwrap_or("1,2,4,6,8,12,16,24")
        .split(',')
        .map(|s| s.trim().parse().map_err(|e| format!("--threads: {e}")))
        .collect::<Result<_, _>>()?;

    let census = {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let g = Gguf::parse(&bytes).map_err(|e| format!("{path}: {e:?}"))?;
        roofline::census(&g)?
    };
    print_census(path, &census);

    let buf = roofline::stream_buffer(mib);
    println!(
        "\n== read bandwidth: {mib} MiB buffer, {reps} passes/arm, read vs NULL arm (same harness, 0 bytes) ABBA, \
         clock = Instant (QPC); GB/s = bytes / (arm - null), 1 GB = 1e9 B"
    );
    println!("  thr   best GB/s  median GB/s  read spread  null/read  clock quantum");
    let rows = roofline::bandwidth(&buf, &threads, reps);
    for r in &rows {
        println!(
            "  {:>3}   {:>9.1}  {:>11.1}  {:>10.3}x  {:>8.4}%  {:>10.1} ns",
            r.threads,
            r.best_gbs(),
            r.median_gbs(),
            r.spread(),
            100.0 * r.null_share(),
            r.quantum() * 1e9
        );
    }

    let best = rows.iter().max_by(|a, b| a.median_gbs().total_cmp(&b.median_gbs())).ok_or("no thread counts")?;
    let bw = best.median_gbs() * 1e9;
    println!("\n== ceiling at the best median ({:.1} GB/s, {} threads): tok/s <= bandwidth / bytes", bw / 1e9, best.threads);
    println!("  head    pos    bytes/token   tok/s ceiling");
    for (label, head) in [("f16 ", census.head), ("int8", census.head_q8())] {
        for pos in [0u64, 64, 512, 1024] {
            let b = census.per_token(head, pos, 2);
            println!("  {label}  {pos:>5}   {:>9.1} MB   {:>10.1}", mb(b), bw / b as f64);
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("minerva: {e}");
            ExitCode::FAILURE
        }
    }
}
