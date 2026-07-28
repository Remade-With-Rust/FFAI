//! One trait per task, many engines per trait.
//!
//! An *engine* is a named, swappable implementation of a task — exactly the
//! role a codec plays in ffmpeg. Engines are registered in an
//! [`crate::registry::EngineRegistry`] and selected by name.

use std::fmt;
use std::str::FromStr;

use crate::error::Result;
use crate::types::{AudioBuffer, ImageBuffer, OcrOutput, TimedSegment, Transcript, VideoFrame};

/// The tasks FFai knows about (the "stream types" of the toolkit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Task {
    Asr,
    Tts,
    Ocr,
    Vlm,
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Task::Asr => "asr",
            Task::Tts => "tts",
            Task::Ocr => "ocr",
            Task::Vlm => "vlm",
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
            other => Err(format!("unknown task `{other}` (expected asr, tts, ocr, or vlm)")),
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
            max_speakers: None,
            diarize_threshold: 0.80,
            translate: false,
            vad: true,
            vad_threshold: 0.5,
            vad_chunk_secs: 30.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TtsOptions {
    pub voice: Option<String>,
    /// Playback-rate multiplier, 1.0 = normal.
    pub speed: f32,
}

impl Default for TtsOptions {
    fn default() -> Self {
        TtsOptions { voice: None, speed: 1.0 }
    }
}

#[derive(Debug, Clone, Default)]
pub struct OcrOptions {
    /// Language hints (engine-specific tags); empty = engine default.
    pub languages: Vec<String>,
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
