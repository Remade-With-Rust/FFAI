//! Mercury ASR engines (speech → text).

use ffai_core::engine::{AsrEngine, AsrOptions, EngineInfo, EngineStatus, Task};
use ffai_core::error::{Error, Result};
use ffai_core::types::{AudioBuffer, Transcript};

/// OpenAI Whisper running on candle-transformers.
///
/// The reference ASR engine — first to go live (Phase 1), oracle-gated
/// against openai-whisper/whisper.cpp output on LibriSpeech WER.
pub struct WhisperCandle;

impl AsrEngine for WhisperCandle {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "whisper-candle".into(),
            task: Task::Asr,
            status: EngineStatus::Stub,
            description: "OpenAI Whisper on candle (reference engine; Phase 1)".into(),
        }
    }

    fn transcribe(&self, _audio: &AudioBuffer, _opts: &AsrOptions) -> Result<Transcript> {
        Err(Error::NotImplemented { task: Task::Asr, engine: "whisper-candle".into() })
    }
}

/// cool-japan's pure-Rust GGUF Whisper (zero C/C++, quantized kernels).
///
/// Tracked as the CPU-lean alternative engine; adopted once upstream
/// stabilizes (it is early-stage as of mid-2026).
pub struct OxiWhisper;

impl AsrEngine for OxiWhisper {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "oxiwhisper".into(),
            task: Task::Asr,
            status: EngineStatus::Stub,
            description: "pure-Rust GGUF Whisper (cool-japan/oxiwhisper; watchlist)".into(),
        }
    }

    fn transcribe(&self, _audio: &AudioBuffer, _opts: &AsrOptions) -> Result<Transcript> {
        Err(Error::NotImplemented { task: Task::Asr, engine: "oxiwhisper".into() })
    }
}
