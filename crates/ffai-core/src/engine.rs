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

#[derive(Debug, Clone, Default)]
pub struct AsrOptions {
    /// Force a language instead of auto-detecting.
    pub language: Option<String>,
    /// Word-level timestamps (WhisperX-style forced alignment).
    pub word_timestamps: bool,
    /// Speaker diarization (WhisperX-style).
    pub diarize: bool,
    /// Translate to English instead of transcribing.
    pub translate: bool,
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
