//! `minerva` — Minerva's command line: `roofline`, `info`, `generate`.
//!
//! ```text
//! minerva roofline [--gguf PATH] [--mib N] [--reps N] [--threads 1,2,4,8]
//! ```

#![forbid(unsafe_code)]

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use ffai_minerva::engine::{self, Session, top_k};
use ffai_minerva::model::Model;
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

const USAGE: &str = "usage:
  minerva roofline [--gguf PATH] [--mib N] [--reps N] [--threads 1,2,4]
  minerva info     [--gguf PATH]
  minerva generate --ids FILE --new N [--gguf PATH] [--threads N] [--probs OUT.jsonl]
      prints the generated ids, comma separated, as the last line of stdout";

fn parse_ids(text: &str) -> Result<Vec<u32>, String> {
    text.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().map_err(|e| format!("id `{s}`: {e}")))
        .collect()
}

fn info(args: &[String]) -> Result<(), String> {
    let path = arg(args, "--gguf").unwrap_or(DEFAULT_GGUF);
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let g = Gguf::parse(&bytes).map_err(|e| format!("{path}: {e:?}"))?;
    for (k, v) in &g.metadata {
        match v {
            rusty_ternary::gguf::Value::Array(a) => println!("{k} = [{} values]", a.len()),
            _ => println!("{k} = {v:?}"),
        }
    }
    println!("{} tensors", g.tensors.len());
    for t in g.tensors.iter().filter(|t| !t.name.starts_with("blk.") || t.name.starts_with("blk.0.")) {
        println!("  {} type {} {:?}", t.name, t.ty, t.dims);
    }
    drop(g);
    let m = Model::from_bytes(bytes).map_err(|e| e.to_string())?;
    println!("loads as: {:?}", m.cfg);
    Ok(())
}

fn generate(args: &[String]) -> Result<(), String> {
    let path = arg(args, "--gguf").unwrap_or(DEFAULT_GGUF);
    let ids_path = arg(args, "--ids").ok_or("--ids FILE is required")?;
    let new: usize = arg(args, "--new").ok_or("--new N is required")?.parse().map_err(|e| format!("--new: {e}"))?;
    if let Some(t) = arg(args, "--threads") {
        let n: usize = t.parse().map_err(|e| format!("--threads: {e}"))?;
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global().map_err(|e| e.to_string())?;
    }
    let prompt = parse_ids(&std::fs::read_to_string(ids_path).map_err(|e| format!("{ids_path}: {e}"))?)?;
    let t0 = Instant::now();
    let m = Model::load(Path::new(path)).map_err(|e| format!("{path}: {e}"))?;
    let load = t0.elapsed();
    let mut s = Session::new(&m, prompt.len() + new);
    let mut probs = match arg(args, "--probs") {
        Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p).map_err(|e| format!("{p}: {e}"))?)),
        None => None,
    };
    let t1 = Instant::now();
    let mut first = None;
    let mut io_err = None;
    let out = engine::generate(&mut s, &prompt, new, |i, logits| {
        if i == 0 {
            first = Some(t1.elapsed());
        }
        if let Some(w) = probs.as_mut() {
            let top: Vec<String> = top_k(logits, 8).iter().map(|(t, p)| format!("[{t},{p:.6}]")).collect();
            if let Err(e) = writeln!(w, "{{\"step\":{i},\"top\":[{}]}}", top.join(",")) {
                io_err.get_or_insert(e);
            }
        }
    })
    .map_err(|e| e.to_string())?;
    let total = t1.elapsed();
    if let Some(e) = io_err {
        return Err(format!("--probs: {e}"));
    }
    if let Some(mut w) = probs {
        w.flush().map_err(|e| format!("--probs: {e}"))?;
    }
    let first = first.unwrap_or(total);
    let decode = total - first;
    eprintln!(
        "minerva: load {:.2}s | prefill {} tokens + first: {:.2}s | decode {} tokens: {:.2}s ({:.2} tok/s) | threads {}",
        load.as_secs_f64(),
        prompt.len(),
        first.as_secs_f64(),
        new.saturating_sub(1),
        decode.as_secs_f64(),
        new.saturating_sub(1) as f64 / decode.as_secs_f64().max(1e-9),
        rayon::current_num_threads()
    );
    let line: Vec<String> = out.iter().map(u32::to_string).collect();
    println!("{}", line.join(","));
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("roofline") => {}
        Some("info") => return info(&args),
        Some("generate") => return generate(&args),
        _ => return Err(USAGE.into()),
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
