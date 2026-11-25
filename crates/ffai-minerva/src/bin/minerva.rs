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

use ffai_minerva::engine::{self, HeadMode, Session, top_k};
use ffai_minerva::model::Model;
use ffai_minerva::roofline::{self, ByteCensus};
use ffai_minerva::simd;
use rusty_ternary::gguf::Gguf;

const DEFAULT_GGUF: &str = "F:/coding/bitnet-golden/ours-i2_s.gguf";

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
  minerva generate --ids FILE --new N [--gguf PATH] [--threads N] [--probs OUT.jsonl] [--census]
      prints the generated ids, comma separated, as the last line of stdout;
      MINERVA_ISA=scalar|ssse3|avx2|neon|simd128 picks the instruction set (default: the widest)";

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

fn generate(args: &[String]) -> Result<(), String> {
    let path = arg(args, "--gguf").unwrap_or(DEFAULT_GGUF);
    let ids_path = arg(args, "--ids").ok_or("--ids FILE is required")?;
    let new: usize = arg(args, "--new")
        .ok_or("--new N is required")?
        .parse()
        .map_err(|e| format!("--new: {e}"))?;
    // `--threads`, else `RAYON_NUM_THREADS` (the pool reads it), else the fast cores.
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
    let prompt =
        parse_ids(&std::fs::read_to_string(ids_path).map_err(|e| format!("{ids_path}: {e}"))?)?;
    let a0 = alloc_census::read();
    let t0 = Instant::now();
    let m = Model::load(Path::new(path)).map_err(|e| format!("{path}: {e}"))?;
    let load = t0.elapsed();
    let mut s = Session::new(&m, prompt.len() + new);
    let mut probs = match arg(args, "--probs") {
        Some(p) => Some(std::io::BufWriter::new(
            std::fs::File::create(p).map_err(|e| format!("{p}: {e}"))?,
        )),
        None => None,
    };
    simd::census::enable(args.iter().any(|a| a == "--census"));
    simd::census::reset();
    let a1 = alloc_census::read();
    let mut a2 = a1;
    let t1 = Instant::now();
    let mut first = None;
    let mut io_err = None;
    // FNV-1a over every generated step's logit bits: one number that says two runs (two tiers,
    // two devices) computed the same thing, not merely chose the same tokens.
    let mut fnv = 0xcbf2_9ce4_8422_2325_u64;
    // `--probs` needs every logit, so it runs the full head; otherwise MINERVA_HEAD (screen).
    let mode = if probs.is_some() {
        HeadMode::Full
    } else {
        HeadMode::from_env()
    };
    let out = engine::generate(&mut s, &prompt, new, mode, |i, logits, tok| {
        if i == 0 {
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
        "minerva: load {:.2}s | prefill {} tokens + first: {:.2}s | decode {} tokens: {:.2}s ({:.2} tok/s) | threads {} | isa {} | head {:?} | logits-fnv {fnv:016x}",
        load.as_secs_f64(),
        prompt.len(),
        first.as_secs_f64(),
        new.saturating_sub(1),
        decode.as_secs_f64(),
        new.saturating_sub(1) as f64 / decode.as_secs_f64().max(1e-9),
        rayon::current_num_threads(),
        s.isa().name(),
        mode
    );
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
        let named: f64 = stages[1..].iter().map(|s| s.1).sum();
        let parts: Vec<String> = stages[1..]
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

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("roofline") => {}
        Some("info") => return info(&args),
        Some("generate") => return generate(&args),
        _ => return Err(USAGE.into()),
    }
    let path = arg(&args, "--gguf").unwrap_or(DEFAULT_GGUF);
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
