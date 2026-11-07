//! The identity gate for the instruction-veins campaign
//! (`docs/plans/mercury-instruction-veins.md`).
//!
//! Every brick in that campaign claims to remove work WITHOUT changing the
//! output. This harness is how that claim is checked: it runs the shipped ASR
//! and TTS paths over a fixed clip set and prints every result in a form that
//! can be `diff`ed between two builds — segment boundaries and text for ASR,
//! sample count and a bit-hash for TTS. Run it once on the tree before a
//! brick, once after, and the two files must be byte-identical.
//!
//! Alongside the identity text it prints the deterministic work counters
//! (`ffai_mercury::asr::counters`) and a per-run allocation census from a
//! counting global allocator, so the SAME run that proves the output did not
//! move also proves the work did. Timing is deliberately absent — the
//! campaign's verdicts are counts, and this file is where they are read.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example veins_gate -- --out before.txt
//! # ...land a brick, rebuild...
//! cargo run --release -p ffai-mercury --example veins_gate -- --out after.txt
//! diff before.txt after.txt     # identity section must be empty
//! ```
//!
//! Flags: `--clips N` (test-clean clips, default 12), `--longform N`
//! (longform clips, default 1), `--tts N` (Harvard sentences, default 8),
//! `--no-asr`, `--no-tts`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;
use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::{SynthesisOptions, Vits};

/// Counts every allocation and byte requested through the global allocator.
/// Relaxed atomics: the numbers are sums, and order does not matter.
struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards every call to `System` unchanged; the only addition is two
// relaxed counter increments, which cannot affect the returned memory.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System` via `alloc`/`alloc_zeroed` above.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn alloc_snapshot() -> (u64, u64) {
    (
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

/// FNV-1a over the bit patterns: a stable, dependency-free hash whose only
/// job is to make "did any sample move" a one-line diff.
fn bit_hash(samples: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for s in samples {
        for b in s.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn wavs(dir: &str, n: usize) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "wav"))
                .collect()
        })
        .unwrap_or_default();
    paths.sort();
    paths.truncate(n);
    paths
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
    let has = |name: &str| args.iter().any(|a| a == name);
    // The work counters only count under the profiler's switch; set it before
    // anything reads it (the first `enabled()` call latches the value).
    // SAFETY: single-threaded at this point — no other thread is reading the
    // environment.
    #[allow(unsafe_code)]
    unsafe {
        std::env::set_var("FFAI_PROFILE", "1");
    }
    let out_path = arg("--out", "");
    let n_clips: usize = arg("--clips", "12").parse()?;
    let n_long: usize = arg("--longform", "1").parse()?;
    let n_tts: usize = arg("--tts", "8").parse()?;

    let mut identity = String::new();
    let mut work = String::new();

    // ---------------------------------------------------------------- ASR
    if !has("--no-asr") {
        let mut paths = wavs("corpora/clips/librispeech-test-clean/audio", n_clips);
        paths.extend(wavs("corpora/clips/librispeech-longform/audio", n_long));
        if paths.is_empty() {
            return Err(
                "no ASR clips — run `cargo run -p ffai-bench --example prepare_librispeech`".into(),
            );
        }
        let engine = WhisperCandle::new();
        let opts = AsrOptions::default();
        // Warm outside the census: weight load + any one-time calibration.
        let warm = ffai_media::load_audio(&paths[0])?;
        engine.transcribe(&warm, &opts)?;

        ffai_mercury::asr::profile::reset();
        let (a0, b0) = alloc_snapshot();
        let mut media = 0.0;
        for p in &paths {
            let audio = ffai_media::load_audio(p)?;
            media += audio.duration_secs();
            let tr = engine.transcribe(&audio, &opts)?;
            writeln!(identity, "## {}", p.file_name().unwrap().to_string_lossy())?;
            for seg in &tr.segments {
                writeln!(
                    identity,
                    "[{:8.3} -> {:8.3}] {}",
                    seg.start, seg.end, seg.value
                )?;
            }
        }
        let (a1, b1) = alloc_snapshot();
        writeln!(
            work,
            "## asr work ({} clips, {media:.1} s audio)",
            paths.len()
        )?;
        writeln!(work, "allocations {}   bytes {}", a1 - a0, b1 - b0)?;
        work.push_str(&ffai_mercury::asr::profile::profile().counters_report());
    }

    // ---------------------------------------------------------------- TTS
    if !has("--no-tts") {
        let cache = std::env::var("FFAI_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("ffai"));
        let voice = cache.join("models").join("piper-vits-lessac-medium");
        if voice.exists() {
            let vits = Vits::load(&voice)?;
            let phonemizer = Phonemizer::from_manifest_dir(Path::new("models"))?;
            let manifest =
                ffai_bench::corpus::Manifest::load(Path::new("corpora/harvard-sentences-v1.toml"))?;
            let clips: Vec<_> = manifest.clips.iter().take(n_tts).collect();
            let opts = SynthesisOptions {
                noise_scale: vits.defaults.noise_scale,
                length_scale: vits.defaults.length_scale,
                noise_w: vits.defaults.noise_w,
                seed: 7,
            };
            // Warm once so lazily-built tables sit outside the census.
            if let Some(c) = clips.first() {
                let text = std::fs::read_to_string(manifest.clip_path(c))?;
                let ids = vits
                    .id_map
                    .sentence_to_ids(&phonemizer.phonemize(text.trim())?)
                    .0;
                vits.synthesize_ids(&ids, &opts)?;
            }
            ffai_mercury::asr::profile::reset();
            let (a0, b0) = alloc_snapshot();
            for c in &clips {
                let text = std::fs::read_to_string(manifest.clip_path(c))?;
                let ids = vits
                    .id_map
                    .sentence_to_ids(&phonemizer.phonemize(text.trim())?)
                    .0;
                let audio = vits.synthesize_ids(&ids, &opts)?;
                writeln!(identity, "## tts {}", c.id)?;
                writeln!(
                    identity,
                    "samples {}  hash {:016x}",
                    audio.len(),
                    bit_hash(&audio)
                )?;
            }
            let (a1, b1) = alloc_snapshot();
            writeln!(work, "## tts work ({} sentences)", clips.len())?;
            writeln!(work, "allocations {}   bytes {}", a1 - a0, b1 - b0)?;
            work.push_str(&ffai_mercury::asr::profile::profile().counters_report());
        } else {
            writeln!(identity, "## tts skipped (no voice at {})", voice.display())?;
        }
    }

    let report = format!("# identity\n{identity}\n# work (not part of the identity diff)\n{work}");
    print!("{report}");
    if !out_path.is_empty() {
        std::fs::write(&out_path, report)?;
        eprintln!("wrote {out_path}");
    }
    Ok(())
}
