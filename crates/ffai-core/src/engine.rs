//! One trait per task, many engines per trait.
//!
//! An *engine* is a named, swappable implementation of a task — exactly the
//! role a codec plays in ffmpeg. Engines are registered in an
//! [`crate::registry::EngineRegistry`] and selected by name.

use std::fmt;
use std::str::FromStr;

use crate::error::Result;
use crate::types::{
    AudioBuffer, DetectOutput, ImageBuffer, OcrOutput, TimedSegment, Transcript, VideoFrame,
};

/// The tasks FFai knows about (the "stream types" of the toolkit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Task {
    Asr,
    Tts,
    Ocr,
    Vlm,
    /// Object detection (Diana). The task exists ahead of its first engine so
    /// `ffai bench detect` can baseline the world references (M-D0); the
    /// `DetectEngine` trait and `DetectOutput` type land with the first
    /// engine at M-D1.
    Detect,
    /// Monocular depth estimation (Diana). Shares Diana's backbone and neck
    /// with [`Task::Detect`] — only the final head differs — but the output
    /// is a dense metric map rather than boxes, so it is its own task.
    Depth,
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Task::Asr => "asr",
            Task::Tts => "tts",
            Task::Ocr => "ocr",
            Task::Vlm => "vlm",
            Task::Detect => "detect",
            Task::Depth => "depth",
        })
    }
}

impl FromStr for Task {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "asr" => Ok(Task::Asr),
            "tts" => Ok(Task::Tts),
            "ocr" => Ok(Task::Ocr),
            "vlm" => Ok(Task::Vlm),
            "detect" => Ok(Task::Detect),
            other => {
                Err(format!("unknown task `{other}` (expected asr, tts, ocr, vlm, or detect)"))
            }
        }
    }
}

/// Honesty marker shown in `ffai engines` — stubs are visible, not hidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineStatus {
    /// Registered and selectable, returns `Error::NotImplemented`.
    Stub,
    /// Works, not yet oracle-gated against a reference implementation.
    Experimental,
    /// Oracle-gated against a reference implementation.
    Stable,
}

impl fmt::Display for EngineStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            EngineStatus::Stub => "stub",
            EngineStatus::Experimental => "experimental",
            EngineStatus::Stable => "stable",
        })
    }
}

/// Metadata every engine exposes for discovery (`ffai engines`).
#[derive(Debug, Clone)]
pub struct EngineInfo {
    pub name: String,
    pub task: Task,
    pub status: EngineStatus,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct AsrOptions {
    /// Force a language instead of auto-detecting.
    pub language: Option<String>,
    /// Word-level timestamps (WhisperX-style forced alignment).
    pub word_timestamps: bool,
    /// Speaker diarization (WhisperX-style).
    pub diarize: bool,
    /// Keep speaker identities across calls, so a voice heard in one chunk
    /// keeps its label in the next.
    ///
    /// Off by default, and the default is the batch behaviour every
    /// diarization system has: labels are arbitrary names for clusters within
    /// ONE call, and `SPEAKER_00` in two separate calls need not be the same
    /// person. That is fine for a file and useless for a stream.
    ///
    /// With this set, the engine keeps a speaker registry between calls. Call
    /// `WhisperCandle::reset_speakers()` when a new recording begins — a new
    /// session is a new set of people, and carrying identities across is
    /// worse than starting fresh.
    ///
    /// Matching is deliberately stricter than in-call clustering: a registry
    /// merge is permanent, and two people who share a centroid stay merged for
    /// the rest of the session.
    pub persist_speakers: bool,
    /// Known speaker count, when the caller has one ("this is an interview,
    /// two people"). Overrides the clustering threshold.
    ///
    /// Measured caution: with the threshold tuned this is **not** the safer
    /// choice. Blind clustering scores 4.21 % DER against 5.00 % with the
    /// true count supplied, because forcing a count forces a merge, and a bad
    /// merge attributes one speaker's words to another. Set it when the count
    /// is certain, not as insurance.
    pub max_speakers: Option<usize>,
    /// Cosine-distance threshold for merging speaker clusters.
    ///
    /// Swept against DER on a 6-conversation corpus: the minimum sits at
    /// 0.85 (2.71 %), and 0.80 (4.21 %) ships instead because over-merging
    /// fails catastrophically (44.7 % at 0.95) while over-splitting fails
    /// gently. See `ffai_mercury::asr::diarize::DEFAULT_THRESHOLD`.
    pub diarize_threshold: f32,
    /// Translate to English instead of transcribing.
    pub translate: bool,
    /// Segment on speech before transcribing, so silence never reaches the
    /// model.
    ///
    /// **On by default, for measured speed — not for quality.**
    ///
    /// - Audio with trailing silence: 2.2–4.2× faster, transcript byte-identical.
    /// - Silent input: empty transcript, with no encoder pass at all.
    /// - A live sliding window stops spending five encoder passes to produce
    ///   nothing.
    ///
    /// Corpus WER *does* move with this on (test-clean 7.99 → 6.79,
    /// test-other 16.79 → 16.43), and that is **not** a quality improvement —
    /// do not cite it as one. Per-clip decomposition over 400 clips gives 38
    /// improved and 38 worsened, a sign test of z = 0.00. VAD shifts where
    /// speech sits inside Whisper's fixed 30 s context by ~0.2 s, which
    /// re-rolls the decode on about a fifth of clips, half each way; the
    /// aggregate moved because WER is dominated by a few high-delta clips.
    /// Full descent: `docs/whys/vad-quality.md`.
    ///
    /// Set `false` for the unsegmented fixed-30 s-grid behaviour.
    pub vad: bool,
    /// Speech threshold, 0..1, higher being stricter. Only read when
    /// [`Self::vad`] is set.
    pub vad_threshold: f32,
    /// Pack speech regions into windows of at most this many seconds.
    pub vad_chunk_secs: f32,
    /// Where this buffer starts in the wider stream, in seconds.
    ///
    /// Only meaningful for a streaming caller that re-sends a sliding window
    /// (a live transcriber sending the trailing N seconds every tick). It
    /// costs nothing to leave at `0.0`.
    ///
    /// **What it buys.** Diarization sub-segments each speech region into
    /// 1.5 s windows and embeds each one — the dominant cost, ~172 ms apiece.
    /// Those windows are placed relative to the region, and a region clipped
    /// by the buffer's leading edge is anchored to the *buffer*, which moves.
    /// So consecutive ticks re-cut the same audio at shifted offsets and every
    /// embedding is recomputed. Measured on a 10 s window at a 1 s tick: the
    /// window grids realign only every 3 s (`lcm(1.0, 0.75)`), and the cache
    /// hit rate sat at ~24 %.
    ///
    /// Given this, windows are placed on an ABSOLUTE grid, so the same audio
    /// yields the same window bounds no matter where the buffer happens to
    /// start — which is what makes the embedding cache actually hit.
    pub stream_offset_secs: f64,
}

impl Default for AsrOptions {
    fn default() -> Self {
        // Written out rather than derived: `vad_threshold` and
        // `vad_chunk_secs` have meaningful defaults, and `#[derive(Default)]`
        // would silently make them 0.0 — a threshold that calls everything
        // speech and a window width that holds nothing.
        AsrOptions {
            language: None,
            word_timestamps: false,
            diarize: false,
            persist_speakers: false,
            max_speakers: None,
            diarize_threshold: 0.80,
            translate: false,
            vad: true,
            vad_threshold: 0.5,
            vad_chunk_secs: 30.0,
            stream_offset_secs: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TtsOptions {
    pub voice: Option<String>,
    /// Playback-rate multiplier, 1.0 = normal.
    pub speed: f32,
    /// Acoustic variation (VITS prior noise); `None` = the voice's own
    /// default. 0.0 is fully deterministic audio.
    pub noise_scale: Option<f32>,
    /// Duration variation (stochastic duration predictor noise); `None` =
    /// voice default, 0.0 = deterministic timing.
    pub noise_w: Option<f32>,
    /// Seed for all sampled noise. Mercury synthesis is byte-stable per
    /// (input, options, seed) — a capability the references do not offer.
    pub seed: u64,
    /// Silence inserted between sentences of long-form input, in seconds.
    pub sentence_silence_s: f32,
}

impl Default for TtsOptions {
    fn default() -> Self {
        TtsOptions {
            voice: None,
            speed: 1.0,
            noise_scale: None,
            noise_w: None,
            seed: 0,
            sentence_silence_s: 0.2,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct OcrOptions {
    /// Language hints (engine-specific tags); empty = engine default.
    pub languages: Vec<String>,
    /// The image IS one text line: skip detection, recognize the whole
    /// frame as a single line (tesseract's `--psm 7`). LIVE's dirty-band
    /// path sets this when a band's known geometry is a single line —
    /// detection becomes async maintenance, recognition the only
    /// synchronous work.
    pub single_line: bool,
}

/// Detection options (Diana).
///
/// `confidence` defaults to 0.25 — the threshold a person looking at boxes
/// wants. Benchmarks that need the low-confidence tail for mAP set it to
/// ~0.001 explicitly, the way `corpora/refs/*_ref.py` do; the default is
/// not tuned for the scorer, and the scorer does not inherit it silently.
#[derive(Debug, Clone)]
pub struct DetectOptions {
    /// Minimum confidence to report.
    pub confidence: f32,
    /// Maximum detections returned, highest confidence first.
    pub max_detections: usize,
    /// Class-wise NMS IoU. `None` for the NMS-free one-to-one path, which
    /// is YOLO26's default and needs no suppression.
    pub iou: Option<f32>,
    /// Restrict to these class ids; empty = every class.
    pub classes: Vec<u32>,
}

impl Default for DetectOptions {
    fn default() -> Self {
        DetectOptions {
            confidence: 0.25,
            max_detections: 300,
            iou: None,
            classes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct VlmOptions {
    /// Instruction for the model; `None` = plain captioning.
    pub prompt: Option<String>,
    pub max_new_tokens: Option<usize>,
}

/// Speech → text (Mercury).
pub trait AsrEngine: Send + Sync {
    fn info(&self) -> EngineInfo;
    fn transcribe(&self, audio: &AudioBuffer, opts: &AsrOptions) -> Result<Transcript>;
}

/// Text → speech (Mercury).
pub trait TtsEngine: Send + Sync {
    fn info(&self) -> EngineInfo;
    fn synthesize(&self, text: &str, opts: &TtsOptions) -> Result<AudioBuffer>;
}

/// Image → text (Carmenta).
pub trait OcrEngine: Send + Sync {
    fn info(&self) -> EngineInfo;
    fn recognize(&self, image: &ImageBuffer, opts: &OcrOptions) -> Result<OcrOutput>;
}

/// Image → objects (Diana).
/// What a depth engine returns: a dense map plus what is needed to place it
/// back on the source image.
#[derive(Debug, Clone)]
pub struct DepthOutput {
    /// Row-major depth, `height * width` values, in **metres**.
    pub depth: Vec<f32>,
    pub width: usize,
    pub height: usize,
    /// The letterbox that produced this map, when one was applied — the same
    /// role it plays in [`DetectOutput`], and needed for the same reason: the
    /// map is in letterboxed space and means nothing without it.
    pub letterbox: Option<crate::types::Letterbox>,
}

impl DepthOutput {
    /// Depth at a pixel of the map. `None` when out of range.
    pub fn at(&self, x: usize, y: usize) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.depth.get(y * self.width + x).copied()
    }

    /// `(min, max)` over the map, for callers normalising it for display.
    /// Returns `None` for an empty map rather than a nonsense range.
    pub fn range(&self) -> Option<(f32, f32)> {
        if self.depth.is_empty() {
            return None;
        }
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for &v in &self.depth {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        (lo <= hi).then_some((lo, hi))
    }
}

/// Options for a depth run.
#[derive(Debug, Clone, Default)]
pub struct DepthOptions {
    /// Resize the map to the SOURCE image's resolution and undo the
    /// letterbox, instead of returning the raw network output at stride 4.
    ///
    /// Off by default: the raw map is what the model computed, and resizing
    /// is a lossy convenience the caller may want to do differently.
    pub full_resolution: bool,
}

/// Monocular depth estimation.
pub trait DepthEngine: Send + Sync {
    fn info(&self) -> EngineInfo;
    fn depth(&self, image: &ImageBuffer, opts: &DepthOptions) -> Result<DepthOutput>;
}

pub trait DetectEngine: Send + Sync {
    fn info(&self) -> EngineInfo;
    fn detect(&self, image: &ImageBuffer, opts: &DetectOptions) -> Result<DetectOutput>;

    /// Detect over many images, using the whole machine.
    ///
    /// **This is a first-class path, not a convenience wrapper**, because it
    /// is where a detector's throughput actually lives. Measured on Diana:
    /// running images concurrently is worth **3.6-5.3x** over calling
    /// [`Self::detect`] in a loop, with byte-identical output and no change
    /// whatever to the per-image path — because intra-image parallelism is
    /// nearly exhausted (24 cores buy one image only 1.42x) while the images
    /// themselves are independent.
    ///
    /// The trait is `Send + Sync`, so one loaded model serves every thread.
    /// That is the structural advantage over the Python reference: measured
    /// on the same machine, PyTorch gets *slower* under threading (0.68-0.72x)
    /// because of the GIL, and its escape hatch — multiprocessing — pays a
    /// full model copy per worker.
    ///
    /// The default implementation is sequential so existing engines keep
    /// working; an engine that can do better should override it.
    fn detect_batch(
        &self,
        images: &[ImageBuffer],
        opts: &DetectOptions,
    ) -> Result<Vec<DetectOutput>> {
        images.iter().map(|i| self.detect(i, opts)).collect()
    }
    /// Class names for the ids in [`DetectOutput`], in id order. They come
    /// from the weight manifest, so they belong to the engine rather than
    /// to every output it produces.
    fn class_names(&self) -> &[String];
}

/// Image/video → description (Argus).
pub trait VlmEngine: Send + Sync {
    fn info(&self) -> EngineInfo;
    fn describe_image(&self, image: &ImageBuffer, opts: &VlmOptions) -> Result<String>;
    /// Video understanding over sampled frames → a timed caption track.
    fn describe_video(
        &self,
        frames: &[VideoFrame],
        opts: &VlmOptions,
    ) -> Result<Vec<TimedSegment<String>>>;
}
