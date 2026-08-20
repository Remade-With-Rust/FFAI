//! The `ffai` binary — a thin shell over the `FFai` library crates, in the way
//! the `ffmpeg` binary is a thin shell over libav*. All real logic lives in
//! the crates so any application can embed `FFai` without this CLI.

/// The allocator, chosen by measurement.
///
/// The system allocator returns memory to the OS and re-faults it: **58,634
/// page faults per image** at the n tier, roughly 234 MiB of freshly-faulted
/// pages against 293.7 MiB/image of allocation — essentially every byte
/// allocated arriving on a page the process must fault in.
///
/// An allocator that keeps pages mapped removes that. Measured ABBA,
/// min-of-150 per run, six pairs with non-overlapping ranges:
///
/// | | min-of-150 |
/// |---|---|
/// | system | 74.5 · 75.2 · 74.3 · 75.7 · 74.6 · 72.8 ms |
/// | mimalloc | 45.5 · 45.5 · 44.0 · 45.1 · 46.0 · 45.5 ms |
///
/// **1.64x, and the slowest mimalloc run beats the fastest system run by
/// 27 ms.** This is by a wide margin the largest single effect found in the
/// Diana latency campaign, and it was found last because every earlier
/// hypothesis priced work rather than page tables.
///
/// A library cannot set this; downstream users of the crates do not inherit
/// it and must choose for themselves.
///
/// ## Why `rusty_alloc` and not `mimalloc`
///
/// `mimalloc` is a C library, which is the one thing `FFai` exists not to depend
/// on — and it sat here anyway, because 1.64x is not a number you give up for
/// a principle. `rusty_alloc` is the pure-Rust remake of the same design
/// (mimalloc v2.4.5), so the principle no longer costs anything. Measured
/// before switching, ABBA-interleaved with a NULL arm, `examples/alloc_ab.rs`:
///
/// | tier | mimalloc | `rusty_alloc` | peak RSS mi -> ra |
/// |---|---|---|---|
/// | n (N=31) | 38.18 ms | 37.37 ms | 77.5 -> 91.4 MB |
/// | s (N=9) | 104.81 ms | 103.53 ms | 183.6 -> 152.7 MB |
/// | m (N=9) | 247.66 ms | 249.92 ms | 299.3 -> 269.4 MB |
///
/// Pooling all 49 paired rounds `rusty_alloc` wins 33 at **z = +2.43**, against
/// a null arm (mimalloc against itself) that wins 22 at z = -0.71. So: a real
/// but small **~1.5 % faster**, and the honest claim is parity-or-better
/// rather than a speed win.
///
/// Memory INVERTS with scale — ~14 MB more fixed arena, ~30 MB less retained
/// once the model is big enough to matter. Above the n tier that is the better
/// profile, and the n-tier regression is a constant, not growth.
///
/// Both arms produce bit-identical detections. Set `--features mimalloc` to
/// switch back; the C library stays wired up as the oracle, exactly as the
/// scalar twin of a SIMD kernel does.
#[cfg(not(feature = "mimalloc"))]
#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Background page reclaim — **OFF by default, because it measured as doing
/// nothing.** Opt in with `FFAI_TRIM_MS=<ms>`.
///
/// Kept rather than deleted because `rusty_alloc` is under active work on
/// exactly this, and a wired-up toggle makes re-testing one env var instead of
/// one patch. This is a *measured-neutral* revert, not a measured-worse one.
///
/// The reason it is not on: `rusty_alloc` holds more resident memory than
/// mimalloc on Diana's workload — 4 worker threads plus candle's 24, none of
/// which ever exit, so freed pages are retained 28 ways. Calling
/// `rusty_alloc::alloc::collect(false)` on a timer looked like it halved that.
/// It did not. Three arms, ONE binary, trim the only variable, ABBA-rotated,
/// N=5 at 1500 detect reps:
///
/// | arm | RSS median | RSS range | latency |
/// |---|---|---|---|
/// | mimalloc | 111.1 MB | 106.7–134.2 | 31.97 ms |
/// | `rusty_alloc`, no trim | 195.8 MB | 92.2–403.3 | 30.63 ms |
/// | `rusty_alloc`, trim 200 ms | 195.8 MB | 91.2–402.5 | 30.49 ms |
///
/// **0.1 MB reclaimed, 0 %.** The apparent "405.6 → 195.0 MB, halved for free"
/// was one sample against another drawn from a distribution spanning
/// 92–403 MB — a 4.4× run-to-run spread, against mimalloc's 1.26×. That
/// variance, not the median, is the real difference between the two
/// allocators here, and it is why every RSS number in this file carries its
/// range.
#[cfg(not(feature = "mimalloc"))]
fn spawn_page_trimmer() {
    let ms = std::env::var("FFAI_TRIM_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    if ms == 0 {
        return;
    }
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_millis(ms));
            rusty_alloc::alloc::collect(false);
        }
    });
}

/// mimalloc runs its own idle reclaim, so the trimmer has nothing to do.
#[cfg(feature = "mimalloc")]
fn spawn_page_trimmer() {}

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ffai_core::engine::{
    AsrOptions, DepthOptions, DetectEngine, DetectOptions, OcrOptions, Task, TtsOptions, VlmOptions,
};
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
    /// Detect objects in an image (Diana)
    Detect {
        /// An image file — or, with --live, a directory of frames (sorted by
        /// name) fed to the change gate in order. Not required with --serve,
        /// which takes its frames from stdin.
        #[arg(short, long, required_unless_present = "serve")]
        input: Option<PathBuf>,
        /// Stream a frame directory through the change gate: a frame that has
        /// not changed reuses the previous detections at zero model cost.
        ///
        /// Measured at **12.58x** on a fixed-camera sequence (60 frames,
        /// motion every 15th) — and near nothing on a moving camera, because
        /// a ONE-PIXEL shift already changes 63 % of the frame. This is for
        /// surveillance, fixed mounts and screen capture.
        #[arg(long)]
        live: bool,
        /// Follow objects across frames with `ByteTrack`, adding a stable id to
        /// each detection. Tracking runs on the DETECTIONS and never reaches
        /// inside the model, the same separation `LiveSession` keeps.
        #[arg(long)]
        track: bool,
        /// `ByteTrack`: detections at or above this drive the FIRST association
        /// pass; those below it (down to 0.1) drive the second.
        #[arg(long, default_value_t = 0.5)]
        track_thresh: f32,
        /// `ByteTrack`: a NEW identity needs a detection at least this confident.
        /// Deliberately higher than `--track-thresh` — recovering an existing
        /// track from a weak box is cheap, starting a new one from it creates a
        /// false trajectory that costs IDF1 for as long as it lives.
        #[arg(long, default_value_t = 0.7)]
        new_track_thresh: f32,
        /// Read frame paths from stdin, one per line, and write one timed JSON
        /// line per frame to stdout. `--input` is ignored.
        ///
        /// This exists so an external viewer can drive Diana at video rate
        /// without paying the 68 ms model load per frame — a per-frame
        /// subprocess would be measuring process startup, not detection.
        /// Combine with `--live` to put the change gate in the loop; the
        /// emitted `gated` flag says whether the model actually ran.
        #[arg(long)]
        serve: bool,
        /// Fraction of pixels that must move past the per-pixel delta for a
        /// frame to count as changed.
        #[arg(long, default_value_t = ffai_diana::live::DEFAULT_CHANGE_FRACTION)]
        change_fraction: f32,
        /// Process every Nth frame; skipped frames cost nothing at all.
        #[arg(long, default_value_t = 1)]
        sample_every: usize,
        /// Grayscale levels a pixel must move to count as changed.
        ///
        /// The default is set by the compressed-video harvest: a codec
        /// re-quantises every frame, so a "static" scene is not static in
        /// pixels. Lower it only for uncompressed sources.
        #[arg(long, default_value_t = ffai_diana::live::DEFAULT_PIXEL_DELTA)]
        pixel_delta: u8,
        #[arg(long)]
        engine: Option<String>,
        /// Minimum confidence to report
        #[arg(long, default_value_t = 0.25)]
        conf: f32,
        /// Class-wise NMS `IoU`. Omitted by default: the one2one head is
        /// NMS-free by construction, so suppression would only ever drop
        /// legitimately overlapping objects.
        #[arg(long)]
        iou: Option<f32>,
        /// Maximum detections reported
        #[arg(long, default_value_t = 300)]
        max_det: usize,
        /// Restrict to these class ids, repeatable; empty = every class
        #[arg(long)]
        classes: Vec<u32>,
        /// Write to a file; a `.jsonl` extension selects JSON lines
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Estimate per-pixel metric depth from one image (Diana)
    Depth {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(long)]
        engine: Option<String>,
        /// Resize the map to the source image and undo the letterbox.
        /// Off by default: the raw stride-4 map is what the model computed.
        #[arg(long)]
        full_res: bool,
        /// Write the map. `.png` gives a 16-bit grayscale visualisation
        /// (normalised, LOSSY — the metres are gone); `.bin` gives raw f32
        /// metres, row-major, which is the one to use for anything numeric.
        #[arg(short, long)]
        output: Option<PathBuf>,
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
        /// Baseline the references only, skipping OUR engine (useful before
        /// our engine is live). This is not "skip the references" — that is
        /// `--engine-only`, and confusing the two costs an hour per run.
        #[arg(long)]
        baseline_only: bool,
        /// Run OUR engine only, skipping every reference.
        ///
        /// The references are Python subprocesses over the whole corpus and
        /// dominate a run; when the question is about our own engine — an A/B
        /// of a flag, say — they are pure cost. There was no way to say this,
        /// and `--baseline-only` reads like it means this but does the
        /// opposite.
        #[arg(long)]
        engine_only: bool,
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
    ffai_diana::register(&mut reg);
    ffai_argus::register(&mut reg);
    reg
}

/// Match candle's thread pool to Diana's before anything allocates.
///
/// candle 0.11 builds its OWN rayon pool (`candle-core` `utils.rs:368`) sized
/// by `RAYON_NUM_THREADS`, defaulting to every logical core — so the process
/// runs 4 of Diana's workers plus 24 of candle's. mimalloc keeps a heap PER
/// THREAD, and its own stats show the consequence directly: `theaps: 28`,
/// against 8 when candle is told to match.
///
/// Measured, interleaved, six runs each (peak RSS from mimalloc's stats, so
/// no sampling error):
///
/// | | peak RSS | min-of-40 |
/// |---|---|---|
/// | default (28 heaps) | 82 – 142 MiB, median 104 | 36.0 ms |
/// | candle = 4 (8 heaps) | 82 – 124 MiB, median **82** | 35.5 ms |
///
/// 21 % less memory at identical speed, and markedly less variance — four of
/// six runs land on 82 MiB where the default swings by 60.
///
/// Set only when the operator has not chosen a value, and set before the
/// first allocation so candle's lazy pool sees it.
fn match_candle_threads() {
    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return;
    }
    let cores = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
    let n = std::env::var("FFAI_DIANA_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| (cores / 6).clamp(3, 6));
    // SAFETY: called as the first statement of `main`, before any thread is
    // spawned and before any pool is built, so no other thread can be reading
    // the environment concurrently.
    #[allow(unsafe_code)]
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", n.to_string());
    }
}

fn main() -> Result<()> {
    match_candle_threads();
    spawn_page_trimmer();
    let cli = Cli::parse();
    let reg = build_registry();

    match cli.cmd {
        Cmd::Engines { task } => {
            let filter = task
                .map(|t| Task::from_str(&t).map_err(anyhow::Error::msg))
                .transpose()?;
            println!(
                "{:<6} {:<16} {:<13} DESCRIPTION",
                "TASK", "ENGINE", "STATUS"
            );
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
                let manifest = manifests.iter().find(|m| m.name == name).with_context(|| {
                    format!("no model manifest named `{name}` in {}", dir.display())
                })?;
                println!("fetching {} ({})...", manifest.name, manifest.license);
                let resolved = manifest.fetch()?;
                for (file, path) in &resolved.files {
                    println!("  {file} -> {}", path.display());
                }
                return Ok(());
            }
            println!(
                "{:<6} {:<20} {:<14} {:<7} SOURCE",
                "TASK", "MODEL", "LICENSE", "CACHED"
            );
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
                eprintln!(
                    "
{audit}"
                );
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
            // §8.171: the DOCUMENT default is SVTR (+1.521 pp macro on the
            // OmniDocBench text holdout, CI excluding zero), but it costs 1.91x
            // wall time and `--live` runs through this same resolution. LIVE
            // keeps the registry default until it has its own `live_bench` /
            // `live_soak` verdict — a throughput gate document OCR does not
            // have. An explicit `--engine` still wins in both modes.
            let eng = match (engine.as_deref(), live) {
                (Some(name), _) => reg.ocr(Some(name))?,
                (None, true) => reg.ocr(None)?,
                (None, false) => reg.ocr(Some(ffai_carmenta::DOC_DEFAULT))?,
            };
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
                let cfg = ffai_carmenta::live::LiveConfig {
                    change_fraction,
                    sample_every,
                    ..Default::default()
                };
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
                        let t = if watch.is_some() {
                            started.elapsed().as_secs_f64()
                        } else {
                            n as f64 / fps
                        };
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
                let end_t = if watch.is_some() {
                    started.elapsed().as_secs_f64()
                } else {
                    n as f64 / fps
                };
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
                let body = match output
                    .as_ref()
                    .and_then(|p| p.extension())
                    .and_then(|e| e.to_str())
                {
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
        Cmd::Detect {
            input,
            engine,
            conf,
            iou,
            max_det,
            classes,
            output,
            live,
            serve,
            track,
            track_thresh,
            new_track_thresh,
            change_fraction,
            sample_every,
            pixel_delta,
        } => {
            let eng = reg.detect(engine.as_deref())?;
            let opts = DetectOptions {
                confidence: conf,
                max_detections: max_det,
                iou,
                classes,
            };

            if serve {
                return serve_stdin(eng, opts, live, change_fraction, sample_every, pixel_delta);
            }

            let input = input.context("--input is required without --serve")?;

            // A video source streams; an image does not. Dispatch on the
            // extension, matching what Ultralytics' `predict(source=...)` does
            // with the same file.
            let cfg = ffai_diana::track::TrackerConfig {
                track_thresh,
                new_track_thresh,
                ..Default::default()
            };
            if is_video(&input) {
                return detect_video(eng, opts, &input, output.as_deref(), track, cfg);
            }
            // A DIRECTORY of frames tracks too. Scoring against MOT17 ground
            // truth must not go through a re-encode: the GT boxes were drawn on
            // the original JPEGs, and our own measurement put decoded-vs-source
            // pixels at mean |diff| 0.66 — small, but enough to move a box
            // sitting on the confidence threshold and contaminate a sweep.
            if track && input.is_dir() {
                return detect_dir_tracked(eng, opts, &input, output.as_deref(), cfg);
            }

            if live {
                let mut frames: Vec<PathBuf> = std::fs::read_dir(&input)
                    .with_context(|| format!("reading frame dir {}", input.display()))?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| {
                        matches!(
                            p.extension().and_then(|e| e.to_str()),
                            Some("png") | Some("jpg") | Some("jpeg")
                        )
                    })
                    .collect();
                frames.sort();
                if frames.is_empty() {
                    anyhow::bail!("no .png/.jpg frames in {}", input.display());
                }
                let cfg = ffai_diana::live::LiveConfig {
                    change_fraction,
                    sample_every,
                    pixel_delta,
                    ..Default::default()
                };
                let mut session = ffai_diana::live::LiveSession::new(eng.clone(), cfg, opts);
                let names = eng.class_names().to_vec();
                let started = std::time::Instant::now();
                let mut lines = Vec::new();
                for f in &frames {
                    let image = ffai_media::load_image(f)?;
                    let out = session.process(&image)?;
                    let stem = f.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
                    for d in &out.detections {
                        lines.push(format!(
                            "{stem}	{}	{:.3}	{:.0}	{:.0}	{:.0}	{:.0}",
                            names.get(d.class_id as usize).map_or("?", String::as_str),
                            d.confidence,
                            d.x0,
                            d.y0,
                            d.x1,
                            d.y1
                        ));
                    }
                }
                let wall = started.elapsed().as_secs_f64();
                let st = session.stats();
                println!(
                    "{} frames in {:.2}s — {} ran the model, {} gated, {} sampled out, {} forced",
                    st.frames, wall, st.processed, st.gated, st.sampled_out, st.forced
                );
                println!(
                    "skip rate {:.1}%  ({:.1} fps against {:.1} fps if every frame ran)",
                    st.skip_rate() * 100.0,
                    st.frames as f64 / wall,
                    st.processed as f64 / wall
                );
                if let Some(path) = output {
                    std::fs::write(
                        &path,
                        format!(
                            "{}
",
                            lines.join(
                                "
"
                            )
                        ),
                    )?;
                    println!("wrote {} ({} detections)", path.display(), lines.len());
                }
                return Ok(());
            }

            let image = ffai_media::load_image(&input)?;
            let out = eng.detect(&image, &opts)?;
            let names = eng.class_names();
            let label = |id: u32| -> &str { names.get(id as usize).map_or("?", String::as_str) };
            let body = match output
                .as_ref()
                .and_then(|p| p.extension())
                .and_then(|e| e.to_str())
            {
                Some("jsonl") => out
                    .detections
                    .iter()
                    .map(|d| {
                        format!(
                            "{{\"x0\":{:.2},\"y0\":{:.2},\"x1\":{:.2},\"y1\":{:.2},\
                             \"class\":{},\"name\":\"{}\",\"confidence\":{:.5}}}",
                            d.x0,
                            d.y0,
                            d.x1,
                            d.y1,
                            d.class_id,
                            label(d.class_id),
                            d.confidence
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                _ => out
                    .detections
                    .iter()
                    .map(|d| {
                        format!(
                            "{:<16} {:.3}  [{:.1}, {:.1}, {:.1}, {:.1}]",
                            label(d.class_id),
                            d.confidence,
                            d.x0,
                            d.y0,
                            d.x1,
                            d.y1
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            match output {
                Some(path) => {
                    std::fs::write(&path, format!("{body}\n"))?;
                    println!(
                        "wrote {} ({} detections)",
                        path.display(),
                        out.detections.len()
                    );
                }
                None => println!("{body}"),
            }
        }
        Cmd::Depth {
            input,
            engine,
            full_res,
            output,
        } => {
            let image = ffai_media::load_image(&input)?;
            let eng = reg.depth(engine.as_deref())?;
            let out = eng.depth(
                &image,
                &DepthOptions {
                    full_resolution: full_res,
                },
            )?;
            let (lo, hi) = out.range().unwrap_or((0.0, 0.0));
            let finite = out.depth.iter().filter(|v| v.is_finite()).count();
            println!(
                "{} x {}  depth {:.2}-{:.2} m  ({} of {} pixels covered)",
                out.width,
                out.height,
                lo,
                hi,
                finite,
                out.depth.len()
            );
            match output
                .as_ref()
                .and_then(|p| p.extension())
                .and_then(|e| e.to_str())
            {
                Some("png") => {
                    // 16-bit grayscale, near = bright. NORMALISED, so the
                    // metres do not survive — this is for looking at, and the
                    // range is printed above so a reader can put them back.
                    let span = (hi - lo).max(1e-6);
                    let px: Vec<u16> = out
                        .depth
                        .iter()
                        .map(|&d| {
                            if d.is_finite() {
                                (((hi - d) / span).clamp(0.0, 1.0) * 65535.0) as u16
                            } else {
                                0
                            }
                        })
                        .collect();
                    let path = output.as_ref().unwrap();
                    ffai_media::save_gray16_png(path, &px, out.width, out.height)?;
                    println!(
                        "wrote {} (16-bit grayscale, normalised {lo:.2}-{hi:.2} m)",
                        path.display()
                    );
                }
                Some("bin") => {
                    let path = output.as_ref().unwrap();
                    let mut bytes = Vec::with_capacity(out.depth.len() * 4);
                    for v in &out.depth {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    std::fs::write(path, &bytes)?;
                    println!(
                        "wrote {} (raw f32 metres, {} x {}, row-major)",
                        path.display(),
                        out.width,
                        out.height
                    );
                }
                Some(other) => anyhow::bail!(
                    "unknown depth output extension `.{other}` — use .png (visualisation)                      or .bin (raw f32 metres)"
                ),
                None => {}
            }
        }
        Cmd::Caption {
            input,
            prompt,
            engine,
        } => {
            let image = ffai_media::load_image(&input)?;
            let opts = VlmOptions {
                prompt,
                max_new_tokens: None,
            };
            let caption = reg.vlm(engine.as_deref())?.describe_image(&image, &opts)?;
            println!("{caption}");
        }
        Cmd::Bench {
            task,
            corpus,
            refs,
            engine,
            only,
            baseline_only,
            engine_only,
            runs,
            ledger,
        } => {
            let task = Task::from_str(&task).map_err(anyhow::Error::msg)?;
            if !matches!(task, Task::Asr | Task::Ocr | Task::Tts | Task::Detect) {
                anyhow::bail!(
                    "`ffai bench {task}` is not wired yet — asr, ocr, tts and detect are the \
                     live bench verticals; vlm follows its engine (see ROADMAP.md)"
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
                skip_references: engine_only,
                corpus,
                references,
                runs,
                ledger: ledger.clone(),
            };
            let record = match task {
                Task::Asr => ffai_bench::runner::run_asr(&reg, &cfg)?,
                Task::Ocr => ffai_bench::runner::run_ocr(&reg, &cfg)?,
                Task::Tts => ffai_bench::tts::run_tts(&reg, &cfg)?,
                Task::Detect => ffai_bench::runner::run_detect(&reg, &cfg)?,
                _ => unreachable!("guarded above"),
            };
            print!("{}", ffai_bench::runner::render(&record));
            println!("appended to {}", ledger.display());
        }
    }
    Ok(())
}

/// Containers the streaming path accepts.
///
/// MUST stay in step with `stream_frames`' own dispatch table. It did not once:
/// the library gained mkv/avi/ts while this still said mp4-only, so those files
/// fell through to the IMAGE loader and were rejected with a message about PNG.
/// A mismatch here is a wrong error, not a missing feature.
fn is_video(p: &std::path::Path) -> bool {
    matches!(
        p.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("mp4" | "mov" | "m4v" | "mkv" | "webm" | "mka" | "avi" | "ts" | "m2ts" | "mts")
    )
}

/// Detect over a video, one frame at a time, at constant memory.
///
/// # Why the output looks like Ultralytics'
///
/// Anyone evaluating this is running `yolo predict source=clip.mp4` in the
/// other terminal. Matching the line format means the two can be diffed
/// directly instead of eyeballed, and it is the same reason the letterbox
/// reproduces their `auto=True` rule rather than inventing one. Their line:
///
/// ```text
/// video 1/1 (frame 1/164) /path/clip.mp4: 544x640 1 person, 1 tv, 110.7ms
/// ```
///
/// The trailing summary matches too, so a reader comparing the two sees the
/// same shape of number in the same place.
///
/// The frame TOTAL is only printed when the container declares one. Ultralytics
/// gets its total from `OpenCV`'s `CAP_PROP_FRAME_COUNT`; where we do not have it
/// we print the index alone rather than a guess, because a wrong total in a
/// progress line is worse than no total.
fn detect_video(
    eng: std::sync::Arc<dyn DetectEngine>,
    opts: DetectOptions,
    path: &std::path::Path,
    output: Option<&std::path::Path>,
    track: bool,
    cfg: ffai_diana::track::TrackerConfig,
) -> Result<()> {
    use std::io::Write;

    let names = eng.class_names().to_vec();
    let mut tracker = track.then(|| ffai_diana::track::ByteTrack::new(cfg));
    let stream = ffai_media::stream_frames(path, 0.0)?;
    let total = stream.frame_count_hint();

    let mut lines: Vec<String> = Vec::new();
    let mut n_frames = 0usize;
    let mut infer_total = 0.0f64;
    let mut shape = (0usize, 0usize);

    for (i, frame) in stream.enumerate() {
        let frame = frame?;
        let t = std::time::Instant::now();
        let found = eng.detect(&frame.image, &opts)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        infer_total += ms;
        n_frames += 1;

        // Class tallies, printed the way Ultralytics prints them: count, name,
        // pluralised, ordered by class id.
        let mut tally: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
        for d in &found.detections {
            *tally.entry(d.class_id).or_insert(0) += 1;
        }
        let summary = if tally.is_empty() {
            "(no detections)".to_string()
        } else {
            tally
                .iter()
                .map(|(c, n)| {
                    let base = names.get(*c as usize).map_or("?", String::as_str);
                    if *n == 1 {
                        format!("1 {base}")
                    } else {
                        format!("{n} {base}s")
                    }
                })
                .collect::<Vec<_>>()
                .join(", ")
        };

        // Ultralytics prints the LETTERBOX shape, height first.
        let (lh, lw) = ffai_diana::image::letterbox_shape(
            frame.image.width as usize,
            frame.image.height as usize,
            640,
            ffai_diana::image::Geometry::Rect,
        );
        shape = (lh, lw);

        match total {
            Some(t) => println!(
                "video 1/1 (frame {}/{}) {}: {}x{} {}, {:.1}ms",
                i + 1,
                t,
                path.display(),
                lh,
                lw,
                summary,
                ms
            ),
            None => println!(
                "video 1/1 (frame {}) {}: {}x{} {}, {:.1}ms",
                i + 1,
                path.display(),
                lh,
                lw,
                summary,
                ms
            ),
        }

        // MOT-challenge column order (frame, id, x, y, w, h, conf, -1, -1, -1)
        // so the output feeds a scorer directly rather than needing a converter.
        if let Some(tk) = tracker.as_mut() {
            let bx: Vec<[f32; 4]> = found
                .detections
                .iter()
                .map(|d| [d.x0, d.y0, d.x1, d.y1])
                .collect();
            let sc: Vec<f32> = found.detections.iter().map(|d| d.confidence).collect();
            let cl: Vec<u32> = found.detections.iter().map(|d| d.class_id).collect();
            for t in tk.update(&bx, &sc, &cl) {
                let bb = t.xyxy();
                lines.push(format!(
                    "{},{},{:.1},{:.1},{:.1},{:.1},{:.3},-1,-1,-1",
                    i + 1,
                    t.id,
                    bb[0],
                    bb[1],
                    bb[2] - bb[0],
                    bb[3] - bb[1],
                    t.score
                ));
            }
        } else if output.is_some() {
            for d in &found.detections {
                lines.push(format!(
                    "{}\t{}\t{:.3}\t{:.0}\t{:.0}\t{:.0}\t{:.0}",
                    i,
                    names.get(d.class_id as usize).map_or("?", String::as_str),
                    d.confidence,
                    d.x0,
                    d.y0,
                    d.x1,
                    d.y1
                ));
            }
        }
    }

    if n_frames == 0 {
        anyhow::bail!("{}: no frames decoded", path.display());
    }
    println!(
        "Speed: {:.1}ms inference per image at shape (1, 3, {}, {})",
        infer_total / n_frames as f64,
        shape.0,
        shape.1
    );
    if let Some(p) = output {
        let mut f = std::io::BufWriter::new(std::fs::File::create(p)?);
        for l in &lines {
            writeln!(f, "{l}")?;
        }
        println!("wrote {} ({} detections)", p.display(), lines.len());
    }
    Ok(())
}

/// Track over a directory of frames, sorted by name.
///
/// Exists so a MOT17 sweep can score the ORIGINAL JPEGs. Routing a benchmark
/// through an encode/decode round trip changes pixels by a small amount —
/// measured at mean |diff| 0.66 of 255 against `OpenCV` — which is enough to flip
/// a detection sitting on the threshold, and a threshold sweep is exactly the
/// experiment that cannot afford that.
fn detect_dir_tracked(
    eng: std::sync::Arc<dyn DetectEngine>,
    opts: DetectOptions,
    dir: &std::path::Path,
    output: Option<&std::path::Path>,
    cfg: ffai_diana::track::TrackerConfig,
) -> Result<()> {
    use std::io::Write;

    let mut frames: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading frame dir {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("png" | "jpg" | "jpeg")
            )
        })
        .collect();
    frames.sort();
    if frames.is_empty() {
        anyhow::bail!("no .png/.jpg frames in {}", dir.display());
    }

    let mut tk = ffai_diana::track::ByteTrack::new(cfg);
    let mut lines = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let image = ffai_media::load_image(f)?;
        let found = eng.detect(&image, &opts)?;
        let bx: Vec<[f32; 4]> = found
            .detections
            .iter()
            .map(|d| [d.x0, d.y0, d.x1, d.y1])
            .collect();
        let sc: Vec<f32> = found.detections.iter().map(|d| d.confidence).collect();
        let cl: Vec<u32> = found.detections.iter().map(|d| d.class_id).collect();
        for t in tk.update(&bx, &sc, &cl) {
            let bb = t.xyxy();
            lines.push(format!(
                "{},{},{:.1},{:.1},{:.1},{:.1},{:.3},-1,-1,-1",
                i + 1,
                t.id,
                bb[0],
                bb[1],
                bb[2] - bb[0],
                bb[3] - bb[1],
                t.score
            ));
        }
    }
    match output {
        Some(p) => {
            let mut w = std::io::BufWriter::new(std::fs::File::create(p)?);
            for l in &lines {
                writeln!(w, "{l}")?;
            }
            println!(
                "{} frames, {} tracked boxes -> {}",
                frames.len(),
                lines.len(),
                p.display()
            );
        }
        None => {
            for l in &lines {
                println!("{l}");
            }
        }
    }
    Ok(())
}

/// Detect frames named on stdin, one JSON line of results per frame.
///
/// # Why this is not just a loop around `ffai detect`
///
/// Model load is 68 ms against a ~32 ms detect. A viewer that spawned one
/// process per frame would spend two thirds of its wall clock in startup and
/// would report a frame rate that says nothing about the engine. Holding the
/// engine across frames is the only way the number on screen means what it
/// says.
///
/// The protocol is deliberately dumb — a path in, a line out, flushed every
/// frame — so the driver can be any language and the timing stays on this side
/// of the pipe, where the detect call actually happens. `ready` is emitted
/// after the engine is built so the driver starts its clock after load rather
/// than including it.
fn serve_stdin(
    eng: std::sync::Arc<dyn DetectEngine>,
    opts: DetectOptions,
    live: bool,
    change_fraction: f32,
    sample_every: usize,
    pixel_delta: u8,
) -> Result<()> {
    use std::io::{BufRead, Write};

    let names = eng.class_names().to_vec();
    let mut session = live.then(|| {
        let cfg = ffai_diana::live::LiveConfig {
            change_fraction,
            sample_every,
            pixel_delta,
            ..Default::default()
        };
        ffai_diana::live::LiveSession::new(eng.clone(), cfg, opts.clone())
    });

    let mut out = std::io::stdout().lock();
    writeln!(out, "{{\"ready\":true,\"live\":{live}}}")?;
    out.flush()?;

    let stdin = std::io::stdin();
    let mut n_frames: u64 = 0;
    for line in stdin.lock().lines() {
        let line = line?;
        let path = line.trim();
        if path.is_empty() {
            continue;
        }
        n_frames += 1;
        // A frame that will not load is reported, not fatal: a viewer driving
        // a directory someone is still writing into should skip and continue,
        // not die on a half-written file.
        //
        // DECODE IS INSIDE THE TIMED REGION, and it has to be. The reference
        // this gets compared against is `predict(path)`, which opens the file
        // itself, so a `ms` that excluded our decode would be pricing our
        // engine against their engine PLUS their decoder. That defect shipped
        // here once: it read as parity at 1080p, where the JPEG is 2.1 MP and
        // the decode is the single largest thing the comparison was hiding.
        // Reported split so neither half can hide again.
        let t_dec = std::time::Instant::now();
        let image = match ffai_media::load_image(std::path::Path::new(path)) {
            Ok(i) => i,
            Err(e) => {
                writeln!(out, "{{\"error\":\"{}\"}}", e.to_string().replace('"', "'"))?;
                out.flush()?;
                continue;
            }
        };
        let decode_ms = t_dec.elapsed().as_secs_f64() * 1e3;

        let t = std::time::Instant::now();
        let (found, gated) = match session.as_mut() {
            Some(s) => {
                let before = s.stats().processed;
                let r = s.process(&image)?;
                let ran = s.stats().processed != before;
                (r, !ran)
            }
            None => (eng.detect(&image, &opts)?, false),
        };
        let ms = t.elapsed().as_secs_f64() * 1e3;

        let dets = found
            .detections
            .iter()
            .map(|d| {
                format!(
                    "{{\"x0\":{:.1},\"y0\":{:.1},\"x1\":{:.1},\"y1\":{:.1},\
                     \"class\":{},\"name\":\"{}\",\"conf\":{:.3}}}",
                    d.x0,
                    d.y0,
                    d.x1,
                    d.y1,
                    d.class_id,
                    names.get(d.class_id as usize).map_or("?", String::as_str),
                    d.confidence
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        writeln!(
            out,
            "{{\"ms\":{:.2},\"detect_ms\":{ms:.2},\"decode_ms\":{decode_ms:.2},\
             \"gated\":{gated},\"n\":{},\"detections\":[{dets}]}}",
            ms + decode_ms,
            found.detections.len()
        )?;
        out.flush()?;
    }

    // The stage report is the point of FFAI_PROFILE, and until now nothing
    // called it —  had zero callers in the tree, so the
    // instrument existed and had never once been read. stderr, so a driver
    // parsing the JSONL on stdout is unaffected.
    if ffai_diana::profile::is_enabled() {
        eprintln!("{}", ffai_diana::profile::profile().report());
    }
    if ffai_diana::profile::roofline_enabled() {
        eprintln!("{}", ffai_diana::profile::roofline_report(n_frames));
        eprintln!("{}", ffai_diana::profile::sliceop_report(n_frames));
        eprintln!("{}", ffai_diana::profile::denorm_report(n_frames));
        eprintln!("{}", ffai_diana::profile::plumb_report(n_frames));
    }
    Ok(())
}
