//! `minerva` — Minerva's command line: `roofline`, `info`, `generate`.
//!
//! ```text
//! minerva roofline [--gguf PATH] [--mib N] [--reps N] [--threads 1,2,4,8]
//! ```

// Two fenced measurement items use `unsafe`: the Windows memory-counter call and the optional
// counting allocator. Nothing else in the binary does.
#![deny(unsafe_code)]

/// Process memory counters at exit: page faults, peak working set, peak private commit.
#[cfg(windows)]
#[allow(unsafe_code)]
mod memstat {
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        page_faults: u32,
        peak_ws: usize,
        ws: usize,
        quota_peak_paged: usize,
        quota_paged: usize,
        quota_peak_nonpaged: usize,
        quota_nonpaged: usize,
        pagefile: usize,
        peak_pagefile: usize,
    }
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut Counters, cb: u32) -> i32;
    }
    /// `(page faults, peak working set, working set, peak private commit)`, bytes.
    pub fn read() -> Option<(u32, usize, usize, usize)> {
        let mut c = Counters {
            cb: u32::try_from(size_of::<Counters>()).ok()?,
            ..Counters::default()
        };
        // SAFETY: a pseudo-handle for this process and a counters struct of the size we pass.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &raw mut c, c.cb) };
        (ok != 0).then_some((c.page_faults, c.peak_ws, c.ws, c.peak_pagefile))
    }
}
#[cfg(not(windows))]
mod memstat {
    pub fn read() -> Option<(u32, usize, usize, usize)> {
        None
    }
}

/// Allocation counts, when built with `--features census-alloc`.
#[cfg(feature = "census-alloc")]
#[allow(unsafe_code)]
mod alloc_census {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static COUNT: AtomicU64 = AtomicU64::new(0);
    static BYTES: AtomicU64 = AtomicU64::new(0);
    /// `MINERVA_ALLOC_TRACE`: print the call site of every allocation while armed.
    static TRACE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    thread_local! { static IN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
    fn trace(size: usize) {
        if TRACE.load(Relaxed) && !IN.with(std::cell::Cell::get) {
            IN.with(|c| c.set(true));
            let bt = std::backtrace::Backtrace::force_capture().to_string();
            let frames: Vec<&str> = bt
                .lines()
                .filter(|l| {
                    !l.contains("alloc_census")
                        && !l.contains("backtrace")
                        && !l.contains("alloc::")
                        && l.trim_start()
                            .chars()
                            .next()
                            .is_some_and(|c| c.is_ascii_digit())
                })
                .take(5)
                .map(str::trim)
                .collect();
            eprintln!("alloc {size} B: {}", frames.join(" <- "));
            IN.with(|c| c.set(false));
        }
    }
    /// Arm or disarm the trace (only if `MINERVA_ALLOC_TRACE` is set).
    pub fn arm(on: bool) {
        if std::env::var_os("MINERVA_ALLOC_TRACE").is_some() {
            TRACE.store(on, Relaxed);
        }
    }
    struct Counting;
    // SAFETY: forwards every call to `System` unchanged; only counts.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            trace(l.size());
            COUNT.fetch_add(1, Relaxed);
            BYTES.fetch_add(l.size() as u64, Relaxed);
            // SAFETY: the caller's contract, passed through.
            unsafe { System.alloc(l) }
        }
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            trace(l.size());
            COUNT.fetch_add(1, Relaxed);
            BYTES.fetch_add(l.size() as u64, Relaxed);
            // SAFETY: as above.
            unsafe { System.alloc_zeroed(l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
            trace(n);
            COUNT.fetch_add(1, Relaxed);
            BYTES.fetch_add(n as u64, Relaxed);
            // SAFETY: as above.
            unsafe { System.realloc(p, l, n) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            // SAFETY: as above.
            unsafe { System.dealloc(p, l) }
        }
    }
    #[global_allocator]
    static A: Counting = Counting;
    /// `(allocations, bytes)` so far.
    pub fn read() -> (u64, u64) {
        (COUNT.load(Relaxed), BYTES.load(Relaxed))
    }
}
#[cfg(not(feature = "census-alloc"))]
mod alloc_census {
    pub fn read() -> (u64, u64) {
        (0, 0)
    }
    pub fn arm(_: bool) {}
}

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use ffai_minerva::engine::{self, HeadMode, Session, top_k, top_k_logprobs};
use ffai_minerva::lora::Adapter;
use ffai_minerva::model::Model;
use ffai_minerva::roofline::{self, ByteCensus};
use ffai_minerva::simd;
use rusty_ternary::gguf::Gguf;

/// The model file: `--gguf PATH`, else the `MINERVA_GGUF` environment variable.
fn gguf(args: &[String]) -> Result<String, String> {
    arg(args, "--gguf")
        .map(str::to_owned)
        .or_else(|| std::env::var("MINERVA_GGUF").ok())
        .ok_or_else(|| "--gguf PATH is required (or set MINERVA_GGUF)".into())
}

fn arg<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn mb(b: u64) -> f64 {
    b as f64 / 1e6
}

fn print_census(path: &str, c: &ByteCensus) {
    println!("== bytes per token: {path}");
    println!(
        "  ternary  {:>9.1} MB  ({} I2_S tensors, {} layers)",
        mb(c.ternary),
        c.ternary_tensors,
        c.layers
    );
    println!(
        "  head     {:>9.1} MB  ({} x {}, type {}, tied={})",
        mb(c.head),
        c.head_shape.0,
        c.head_shape.1,
        c.head_type,
        c.tied
    );
    println!(
        "  norms    {:>9.3} MB   embed row {} B   other {} B",
        mb(c.norms),
        c.embed_row,
        c.other
    );
    println!(
        "  KV (f16) {:>9.1} KB per cached position",
        c.kv_per_position(2) as f64 / 1e3
    );
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
  minerva generate --ids FILE --new N [--gguf PATH] [--threads N] [--head FILE] [--lora FILE [--lora-scale S]]
                   [--probs OUT.jsonl] [--logprobs OUT.jsonl] [--census]
      prints the generated ids, comma separated, as the last line of stdout;
      MINERVA_ISA=scalar|ssse3|avx2|neon|simd128 picks the instruction set (default: the widest);
      --head FILE uses a head file from prepare-head instead of building the screen at the first token;
      --lora FILE adds a LoRA adapter GGUF (llama.cpp's format) at strength S (default 1);
      --logprobs writes each step's token, its log-probability and the top 8 (full vocabulary softmax)
  minerva chat [--gguf PATH] [--system TEXT] [--temp T] [--top-k K] [--top-p P] [--min-p P] [--seed S]
               [-c CTX] [--threads N] [--head FILE] [--lora FILE [--lora-scale S]]
      a conversation in the terminal through the model's chat template, streamed as it is
      generated; an empty line or end of input quits, /reset starts over
  minerva score --ids PROMPT --target IDS [--gguf PATH] [--threads N] [--lora FILE [--lora-scale S]]
      teacher-forced: the log-probability of each target id after the prompt, one per line
      (`index id logprob`), then the total, mean and perplexity
  minerva prepare-head --out FILE [--gguf PATH] [--beside-model]
      builds the greedy head's int8 screen once and writes it as a head file (refuses the model's
      own directory unless --beside-model)";

fn parse_ids(text: &str) -> Result<Vec<u32>, String> {
    text.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().map_err(|e| format!("id `{s}`: {e}")))
        .collect()
}

fn info(args: &[String]) -> Result<(), String> {
    let path = &gguf(args)?;
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let g = Gguf::parse(&bytes).map_err(|e| format!("{path}: {e:?}"))?;
    for (k, v) in &g.metadata {
        match v {
            rusty_ternary::gguf::Value::Array(a) => println!("{k} = [{} values]", a.len()),
            _ => println!("{k} = {v:?}"),
        }
    }
    println!("{} tensors", g.tensors.len());
    for t in g
        .tensors
        .iter()
        .filter(|t| !t.name.starts_with("blk.") || t.name.starts_with("blk.0."))
    {
        println!("  {} type {} {:?}", t.name, t.ty, t.dims);
    }
    drop(g);
    let m = Model::from_bytes(bytes).map_err(|e| e.to_string())?;
    println!("loads as: {:?}", m.cfg);
    Ok(())
}

/// `--threads`, else `RAYON_NUM_THREADS` (the pool reads it), else the fast cores.
fn set_threads(args: &[String]) -> Result<(), String> {
    let threads = match arg(args, "--threads") {
        Some(t) => Some(t.parse().map_err(|e| format!("--threads: {e}"))?),
        None if std::env::var_os("RAYON_NUM_THREADS").is_some() => None,
        None => ffai_minerva::threads::recommended(),
    };
    if let Some(n) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// `--lora FILE [--lora-scale S]`: the adapter, if one is named.
fn adapter(args: &[String], m: &Model) -> Result<Option<Adapter>, String> {
    let Some(f) = arg(args, "--lora") else {
        return Ok(None);
    };
    let scale: f32 = arg(args, "--lora-scale")
        .unwrap_or("1")
        .parse()
        .map_err(|e| format!("--lora-scale: {e}"))?;
    Adapter::load(m, Path::new(f), scale)
        .map(Some)
        .map_err(|e| format!("{f}: {e}"))
}

/// The adapter's note for the output (empty without one).
fn adapter_note(a: Option<&Adapter>, args: &[String]) -> String {
    a.map_or_else(String::new, |a| {
        let (pairs, rank) = a.summary();
        format!(
            " | lora: {} ({pairs} projections, rank {rank}, alpha {}, scale {})",
            arg(args, "--lora").unwrap_or(""),
            a.alpha,
            arg(args, "--lora-scale").unwrap_or("1")
        )
    })
}

/// A session, with the adapter on when there is one.
fn session<'m>(m: &'m Model, ctx: usize, a: Option<&'m Adapter>) -> Result<Session<'m>, String> {
    let s = Session::new(m, ctx);
    match a {
        Some(a) => s.with_adapter(a),
        None => Ok(s),
    }
}

fn generate(args: &[String]) -> Result<(), String> {
    let path = &gguf(args)?;
    let ids_path = arg(args, "--ids").ok_or("--ids FILE is required")?;
    let new: usize = arg(args, "--new")
        .ok_or("--new N is required")?
        .parse()
        .map_err(|e| format!("--new: {e}"))?;
    set_threads(args)?;
    let prompt =
        parse_ids(&std::fs::read_to_string(ids_path).map_err(|e| format!("{ids_path}: {e}"))?)?;
    let a0 = alloc_census::read();
    let t0 = Instant::now();
    let m = Model::load(Path::new(path)).map_err(|e| format!("{path}: {e}"))?;
    if let Some(h) = arg(args, "--head") {
        let h8 =
            ffai_minerva::head::HeadQ8::open(&m, Path::new(h)).map_err(|e| format!("{h}: {e}"))?;
        m.screen.attach(h8);
    }
    let lora = adapter(args, &m)?;
    let load = t0.elapsed();
    let mut s = session(&m, prompt.len() + new, lora.as_ref())?;
    let file = |flag: &str| -> Result<Option<std::io::BufWriter<std::fs::File>>, String> {
        arg(args, flag)
            .map(|p| {
                std::fs::File::create(p)
                    .map(std::io::BufWriter::new)
                    .map_err(|e| format!("{p}: {e}"))
            })
            .transpose()
    };
    let mut probs = file("--probs")?;
    let mut logprobs = file("--logprobs")?;
    simd::census::enable(args.iter().any(|a| a == "--census"));
    simd::census::reset();
    let a1 = alloc_census::read();
    let mut a2 = a1;
    let t1 = Instant::now();
    let mut first = None;
    // `--latency`: each step's end, for the per-token spread (a whole-run rate hides a stutter).
    let mut ends: Option<Vec<std::time::Duration>> = args
        .iter()
        .any(|a| a == "--latency")
        .then(|| Vec::with_capacity(new));
    let mut io_err = None;
    // FNV-1a over every generated step's logit bits: one number that says two runs (two tiers,
    // two devices) computed the same thing, not merely chose the same tokens.
    let mut fnv = 0xcbf2_9ce4_8422_2325_u64;
    // `--probs` and `--logprobs` need every logit, so they run the full head (the same tokens);
    // otherwise MINERVA_HEAD (screen).
    let mode = if probs.is_some() || logprobs.is_some() {
        HeadMode::Full
    } else {
        HeadMode::from_env()
    };
    // The profile at the first token: decode stages are reported from the second step on (the
    // first carries the head's one-time build).
    let mut prof0 = None;
    let out = engine::generate(&mut s, &prompt, new, mode, |i, logits, tok| {
        if let Some(e) = ends.as_mut() {
            e.push(t1.elapsed());
        }
        if i == 0 {
            prof0 = ffai_minerva::prof::report();
            first = Some(t1.elapsed());
            a2 = alloc_census::read();
            alloc_census::arm(true);
        }
        // With logits, the hash covers every logit bit; with the screen, only the token.
        match logits {
            Some(ls) => {
                for l in ls {
                    for b in l.to_bits().to_le_bytes() {
                        fnv = (fnv ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
                    }
                }
            }
            None => {
                for b in tok.to_le_bytes() {
                    fnv = (fnv ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
                }
            }
        }
        if let (Some(w), Some(logits)) = (probs.as_mut(), logits) {
            let top: Vec<String> = top_k(logits, 8)
                .iter()
                .map(|(t, p)| format!("[{t},{p:.6}]"))
                .collect();
            if let Err(e) = writeln!(w, "{{\"step\":{i},\"top\":[{}]}}", top.join(",")) {
                io_err.get_or_insert(e);
            }
        }
        if let (Some(w), Some(logits)) = (logprobs.as_mut(), logits) {
            let norm = engine::log_norm(logits);
            let top: Vec<String> = top_k_logprobs(logits, 8)
                .iter()
                .map(|(t, l)| format!("[{t},{l}]"))
                .collect();
            let lp = engine::logprob(logits, norm, tok);
            if let Err(e) = writeln!(
                w,
                "{{\"step\":{i},\"token\":{tok},\"logprob\":{lp},\"top\":[{}]}}",
                top.join(",")
            ) {
                io_err.get_or_insert(e);
            }
        }
    })
    .map_err(|e| e.to_string())?;
    let total = t1.elapsed();
    if let Some(e) = io_err {
        return Err(format!("--probs / --logprobs: {e}"));
    }
    for mut w in [probs, logprobs].into_iter().flatten() {
        w.flush()
            .map_err(|e| format!("--probs / --logprobs: {e}"))?;
    }
    let first = first.unwrap_or(total);
    let decode = total - first;
    eprintln!(
        "minerva: load {:.2}s | prefill {} tokens + first: {:.2}s | decode {} tokens: {:.2}s ({:.2} tok/s) | threads {} | isa {} | head {:?} | logits-fnv {fnv:016x} | head: {}",
        load.as_secs_f64(),
        prompt.len(),
        first.as_secs_f64(),
        new.saturating_sub(1),
        decode.as_secs_f64(),
        new.saturating_sub(1) as f64 / decode.as_secs_f64().max(1e-9),
        rayon::current_num_threads(),
        s.isa().name(),
        mode,
        m.screen.source()
    );
    if lora.is_some() {
        eprintln!("minerva:{}", adapter_note(lora.as_ref(), args));
    }
    if let Some(e) = ends.filter(|e| e.len() > 1) {
        let ms: Vec<f64> = e
            .windows(2)
            .map(|w| (w[1] - w[0]).as_secs_f64() * 1e3)
            .collect();
        let mut sorted = ms.clone();
        sorted.sort_by(f64::total_cmp);
        let q = |p: usize| sorted[(sorted.len() - 1) * p / 100];
        eprintln!(
            "latency ms: p50 {:.1} p90 {:.1} p99 {:.1} max {:.1} (of {})",
            q(50),
            q(90),
            q(99),
            q(100),
            ms.len()
        );
        let every: Vec<String> = ms.iter().map(|v| format!("{v:.1}")).collect();
        eprintln!("latency series: {}", every.join(","));
    }
    let (steps, rows, most) = s.screen_stats;
    if steps > 0 {
        eprintln!(
            "screen: {steps} steps, {:.2} rows recomputed a step (most {most})",
            rows as f64 / steps as f64
        );
    }
    if cfg!(feature = "census-alloc") {
        let a3 = alloc_census::read();
        eprintln!(
            "allocs: load {} ({:.1} MB) | prefill+first {} ({:.1} MB) | decode {} ({:.1} MB) = {:.2} a token",
            a1.0 - a0.0,
            (a1.1 - a0.1) as f64 / 1e6,
            a2.0 - a1.0,
            (a2.1 - a1.1) as f64 / 1e6,
            a3.0 - a2.0,
            (a3.1 - a2.1) as f64 / 1e6,
            (a3.0 - a2.0) as f64 / new.saturating_sub(1).max(1) as f64
        );
    }
    if let Some((pf, peak, ws, commit)) = memstat::read() {
        eprintln!(
            "memstat: page faults {pf} | peak working set {:.1} MB | working set {:.1} MB | peak private commit {:.1} MB",
            peak as f64 / 1e6,
            ws as f64 / 1e6,
            commit as f64 / 1e6
        );
    }
    if let Some(stages) = ffai_minerva::prof::report() {
        let total = stages[0].1;
        let named: f64 = stages[1..4].iter().map(|s| s.1).sum();
        let parts: Vec<String> = stages[1..4]
            .iter()
            .map(|(n, t)| format!("{n} {t:.3}s ({:.1}%)", 100.0 * t / total.max(1e-12)))
            .collect();
        eprintln!(
            "prof: prefill {total:.3}s = {} + other {:.3}s ({:.1}%)",
            parts.join(" + "),
            total - named,
            100.0 * (total - named) / total.max(1e-12)
        );
    }
    if let Some(stages) = ffai_minerva::prof::report() {
        // Decode: per step, over the steps after the first token (the first carries the head's
        // one-time build).
        let base = prof0.unwrap_or_else(|| stages.iter().map(|s| (s.0, 0.0)).collect());
        let stages: Vec<(&str, f64)> = stages
            .iter()
            .zip(&base)
            .map(|(s, b)| (s.0, s.1 - b.1))
            .collect();
        let steps = new.saturating_sub(1).max(1) as f64;
        let total = stages[4].1;
        let named: f64 = stages[5..8].iter().map(|s| s.1).sum();
        let parts: Vec<String> = stages[5..8]
            .iter()
            .map(|(n, t)| {
                format!(
                    "{n} {:.2}ms ({:.1}%)",
                    1e3 * t / steps,
                    100.0 * t / total.max(1e-12)
                )
            })
            .collect();
        eprintln!(
            "prof: decode {:.2}ms/step over {steps} steps = {} + other {:.2}ms ({:.1}%)",
            1e3 * total / steps,
            parts.join(" + "),
            1e3 * (total - named) / steps,
            100.0 * (total - named) / total.max(1e-12)
        );
    }
    if args.iter().any(|a| a == "--census") {
        let snap = simd::census::snapshot();
        for k in simd::census::NAMES {
            let cells: Vec<_> = snap.iter().filter(|c| c.0 == k).collect();
            let total: u64 = cells.iter().map(|c| c.2).sum();
            let split: Vec<String> = cells
                .iter()
                .map(|c| format!("{} {:.2}%", c.1, 100.0 * c.2 as f64 / total.max(1) as f64))
                .collect();
            eprintln!("census: {k:<13} {total:>15}  {}", split.join(", "));
        }
    }
    let line: Vec<String> = out.iter().map(u32::to_string).collect();
    println!("{}", line.join(","));
    Ok(())
}

/// `chat`: a conversation in the terminal.
fn chat(args: &[String]) -> Result<(), String> {
    use ffai_minerva::sampling::{Rng, Sampling, sample_from};
    use ffai_minerva::tokenizer::{StreamDecoder, Tokenizer};
    use std::io::BufRead;
    let path = &gguf(args)?;
    let num = |flag: &str, d: &str| -> Result<f64, String> {
        arg(args, flag)
            .unwrap_or(d)
            .parse()
            .map_err(|e| format!("{flag}: {e}"))
    };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "small settings"
    )]
    let mut sp = Sampling {
        temperature: num("--temp", "0.7")? as f32,
        top_k: num("--top-k", "40")? as usize,
        top_p: num("--top-p", "0.95")? as f32,
        min_p: num("--min-p", "0.05")? as f32,
        seed: 0,
    };
    sp.seed = match arg(args, "--seed") {
        Some(v) => v.parse().map_err(|e| format!("--seed: {e}"))?,
        None => std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64),
    };
    let ctx: usize = arg(args, "-c")
        .unwrap_or("4096")
        .parse()
        .map_err(|e| format!("-c: {e}"))?;
    set_threads(args)?;
    let m = Model::load(Path::new(path)).map_err(|e| format!("{path}: {e}"))?;
    if let Some(h) = arg(args, "--head") {
        m.screen.attach(
            ffai_minerva::head::HeadQ8::open(&m, Path::new(h)).map_err(|e| format!("{h}: {e}"))?,
        );
    }
    let lora = adapter(args, &m)?;
    let tok = Tokenizer::from_gguf(m.file_bytes()).map_err(|e| format!("{path}: {e}"))?;
    let mut s = session(&m, ctx, lora.as_ref())?;
    let eog: Vec<u32> = ["<|eot_id|>", "<|end_of_text|>"]
        .iter()
        .filter_map(|n| tok.special(n))
        .chain(tok.eos)
        .collect();
    let system = arg(args, "--system").map(str::to_owned);
    let mut history: Vec<(String, String)> = Vec::new();
    let mut cached: Vec<u32> = Vec::new();
    let mut rng = Rng::new(sp.seed);
    eprintln!(
        "minerva chat: {} | ctx {ctx} | temp {} top-k {} top-p {} min-p {} seed {}{} | empty line quits, /reset starts over",
        path,
        sp.temperature,
        sp.top_k,
        sp.top_p,
        sp.min_p,
        sp.seed,
        adapter_note(lora.as_ref(), args)
    );
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    loop {
        eprint!(
            "
> "
        );
        let mut line = String::new();
        if stdin
            .lock()
            .read_line(&mut line)
            .map_err(|e| e.to_string())?
            == 0
            || line.trim().is_empty()
        {
            break;
        }
        if line.trim() == "/reset" {
            history.clear();
            eprintln!("(conversation reset)");
            continue;
        }
        history.push(("user".into(), line.trim().to_owned()));
        let mut msgs: Vec<(String, String)> = system
            .iter()
            .map(|t| ("system".to_owned(), t.clone()))
            .collect();
        msgs.extend(history.iter().cloned());
        let prompt = tok.encode(
            &tok.apply_chat_template(&msgs, true)
                .map_err(|e| e.to_string())?,
            tok.add_bos,
        );
        if prompt.len() + 8 >= ctx {
            history.clear();
            eprintln!("(the conversation no longer fits the context; reset)");
            continue;
        }
        let keep = cached
            .iter()
            .zip(&prompt)
            .take_while(|(a, b)| a == b)
            .count()
            .min(prompt.len() - 1)
            .min(s.pos());
        s.rewind(keep);
        cached.truncate(keep);
        let mut reply = String::new();
        let mut dec = StreamDecoder::new();
        let t0 = Instant::now();
        let mut n = 0usize;
        rayon::scope(|_| -> Result<(), String> {
            s.prefill(&prompt[keep..prompt.len() - 1])
                .map_err(|e| e.to_string())?;
            cached.extend_from_slice(&prompt[keep..prompt.len() - 1]);
            let mut next = prompt[prompt.len() - 1];
            while s.pos() + 1 < ctx {
                let t = if sp.is_greedy() {
                    s.step_argmax(next).map_err(|e| e.to_string())?
                } else if sp.top_k > 0 {
                    let (cands, _) = s.step_top_k(next, sp.top_k).map_err(|e| e.to_string())?;
                    sample_from(&cands, &sp, &mut rng)
                } else {
                    let logits = s
                        .step(next, true)
                        .map_err(|e| e.to_string())?
                        .unwrap_or_default();
                    ffai_minerva::sampling::sample(logits, &sp, &mut rng)
                };
                cached.push(next);
                if eog.contains(&t) {
                    break;
                }
                n += 1;
                let piece = dec.push(&tok, t);
                reply.push_str(&piece);
                let _ = write!(out, "{piece}");
                let _ = out.flush();
                next = t;
            }
            Ok(())
        })?;
        let rest = dec.finish();
        reply.push_str(&rest);
        let _ = writeln!(out, "{rest}");
        #[expect(clippy::cast_precision_loss, reason = "a token count")]
        let rate = n as f64 / t0.elapsed().as_secs_f64().max(1e-9);
        eprintln!("({n} tokens, {rate:.1} tok/s)");
        history.push(("assistant".into(), reply));
    }
    Ok(())
}

/// `topk-census`: greedy decoding through the full head, and at every step the head screen's
/// top `k` checked against the full head's (ids and value bits) for each `--k`, with the rows
/// it recomputed.
fn topk_census(args: &[String]) -> Result<(), String> {
    let path = &gguf(args)?;
    let ids_path = arg(args, "--ids").ok_or("--ids FILE is required")?;
    let new: usize = arg(args, "--new")
        .unwrap_or("32")
        .parse()
        .map_err(|e| format!("--new: {e}"))?;
    let ks: Vec<usize> = arg(args, "--k")
        .unwrap_or("1,40")
        .split(',')
        .map(|k| k.trim().parse().map_err(|e| format!("--k: {e}")))
        .collect::<Result<_, String>>()?;
    set_threads(args)?;
    let prompt =
        parse_ids(&std::fs::read_to_string(ids_path).map_err(|e| format!("{ids_path}: {e}"))?)?;
    let m = Model::load(Path::new(path)).map_err(|e| format!("{path}: {e}"))?;
    if let Some(h) = arg(args, "--head") {
        m.screen.attach(
            ffai_minerva::head::HeadQ8::open(&m, Path::new(h)).map_err(|e| format!("{h}: {e}"))?,
        );
    }
    let mut s = Session::new(&m, prompt.len() + new);
    let (last, head) = prompt.split_last().ok_or("empty prompt")?;
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); ks.len()];
    let mut bad = 0usize;
    rayon::scope(|_| -> Result<(), String> {
        s.prefill(head).map_err(|e| e.to_string())?;
        let mut next = *last;
        for _ in 0..new {
            let t = engine::argmax(
                s.step(next, true)
                    .map_err(|e| e.to_string())?
                    .unwrap_or_default(),
            );
            for (j, &k) in ks.iter().enumerate() {
                if let Some((same, n)) = s.check_top_k(k) {
                    bad += usize::from(!same);
                    rows[j].push(n);
                }
            }
            next = t;
        }
        Ok(())
    })?;
    for (k, r) in ks.iter().zip(&rows) {
        let mut v = r.clone();
        v.sort_unstable();
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let mean = v.iter().sum::<usize>() as f64 / v.len().max(1) as f64;
        println!(
            "topk-census: {} prompt {} | k {k} | {} steps | rows recomputed mean {mean:.1} median {} p90 {} max {} | isa {}",
            ids_path,
            prompt.len(),
            v.len(),
            v.get(v.len() / 2).copied().unwrap_or(0),
            v.get(v.len().saturating_sub(1) * 9 / 10)
                .copied()
                .unwrap_or(0),
            v.last().copied().unwrap_or(0),
            s.isa().name()
        );
    }
    println!(
        "topk-census: screen top-k == full head top-k: {}",
        if bad == 0 {
            "every step, every k".to_owned()
        } else {
            format!("{bad} MISMATCHES")
        }
    );
    if bad > 0 {
        return Err("the screen's top k differed from the full head's".into());
    }
    Ok(())
}

/// `score`: teacher-forced log-probabilities of `--target` after `--ids`.
fn score(args: &[String]) -> Result<(), String> {
    let path = &gguf(args)?;
    let read = |flag: &str| -> Result<Vec<u32>, String> {
        let f = arg(args, flag).ok_or_else(|| format!("{flag} FILE is required"))?;
        parse_ids(&std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?)
    };
    let (prompt, target) = (read("--ids")?, read("--target")?);
    set_threads(args)?;
    let m = Model::load(Path::new(path)).map_err(|e| format!("{path}: {e}"))?;
    let lora = adapter(args, &m)?;
    let mut s = session(&m, prompt.len() + target.len(), lora.as_ref())?;
    let t = Instant::now();
    let lps = engine::score(&mut s, &prompt, &target).map_err(|e| e.to_string())?;
    for (i, (id, lp)) in target.iter().zip(&lps).enumerate() {
        println!("{i} {id} {lp}");
    }
    let total: f64 = lps.iter().sum();
    #[expect(clippy::cast_precision_loss, reason = "a token count")]
    let mean = total / lps.len().max(1) as f64;
    println!(
        "score: {} tokens after {} | logprob total {total} | mean {mean} | perplexity {} | {:.2}s | isa {}{}",
        lps.len(),
        prompt.len(),
        libm::exp(-mean),
        t.elapsed().as_secs_f64(),
        s.isa().name(),
        adapter_note(lora.as_ref(), args)
    );
    Ok(())
}

/// `prepare-head`: build the screen once and write it as a head file. Never beside the model by
/// default (a model's directory may be a checked golden set).
fn prepare_head(args: &[String]) -> Result<(), String> {
    let path = &gguf(args)?;
    let out = arg(args, "--out").ok_or("--out FILE is required")?;
    let dir = |p: &str| {
        std::fs::canonicalize(
            Path::new(p)
                .parent()
                .filter(|d| !d.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
        )
        .ok()
    };
    if !args.iter().any(|a| a == "--beside-model") && dir(path).is_some() && dir(path) == dir(out) {
        return Err(format!(
            "{out}: beside the model; pass --beside-model to write into the model's directory"
        ));
    }
    let m = Model::load(Path::new(path)).map_err(|e| format!("{path}: {e}"))?;
    let t = Instant::now();
    let h8 = ffai_minerva::head::HeadQ8::build(&m);
    let built = t.elapsed();
    h8.write(&m, Path::new(out))
        .map_err(|e| format!("{out}: {e}"))?;
    let t = Instant::now();
    ffai_minerva::head::HeadQ8::open(&m, Path::new(out))
        .map_err(|e| format!("{out}: reopened: {e}"))?;
    eprintln!(
        "prepare-head: {} x {} | built {:.2}s | wrote {:.1} MB to {out} | reopened and checked {:.2}s",
        m.cfg.vocab,
        m.cfg.embd,
        built.as_secs_f64(),
        std::fs::metadata(out).map_or(0, |x| x.len()) as f64 / 1e6,
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("roofline") => {}
        Some("info") => return info(&args),
        Some("generate") => return generate(&args),
        Some("prepare-head") => return prepare_head(&args),
        Some("score") => return score(&args),
        Some("topk-census") => return topk_census(&args),
        Some("chat") => return chat(&args),
        _ => return Err(USAGE.into()),
    }
    let path = &gguf(&args)?;
    let mib: usize = arg(&args, "--mib")
        .unwrap_or("1024")
        .parse()
        .map_err(|e| format!("--mib: {e}"))?;
    let reps: usize = arg(&args, "--reps")
        .unwrap_or("21")
        .parse()
        .map_err(|e| format!("--reps: {e}"))?;
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

    let best = rows
        .iter()
        .max_by(|a, b| a.median_gbs().total_cmp(&b.median_gbs()))
        .ok_or("no thread counts")?;
    let bw = best.median_gbs() * 1e9;
    println!(
        "\n== ceiling at the best median ({:.1} GB/s, {} threads): tok/s <= bandwidth / bytes",
        bw / 1e9,
        best.threads
    );
    println!("  head    pos    bytes/token   tok/s ceiling");
    for (label, head) in [("f16 ", census.head), ("int8", census.head_q8())] {
        for pos in [0u64, 64, 512, 1024] {
            let b = census.per_token(head, pos, 2);
            println!(
                "  {label}  {pos:>5}   {:>9.1} MB   {:>10.1}",
                mb(b),
                bw / b as f64
            );
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
