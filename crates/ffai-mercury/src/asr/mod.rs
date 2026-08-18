//! Mercury ASR: speech → text.
//!
//! Decomposed into independently testable stages (mission plan §2) rather
//! than one opaque `transcribe()`. Each stage below owns a contract and its
//! own oracle test, so a WER regression is attributable to one box:
//!
//! | Stage | Contract |
//! |---|---|
//! | [`mel`] | PCM → log-mel spectrogram |
//! | [`tokenizer`] | token ids ↔ text, and the special-token grammar |
//! | [`model`] | weights → a loaded Whisper on a candle device |
//! | [`decoder`] | mel → token ids (greedy, with the timestamp grammar) |

pub mod adaptive;
pub mod align;
pub mod aligner;
pub mod audio_encoder;
pub mod decoder;
pub mod diarize;
pub mod diarizer;
pub mod f16_gemv;
pub mod fbank;
pub mod flash_attn;
pub mod knobs;
pub mod mel;
pub mod model;
pub mod profile;
pub mod registry;
pub mod speaker;
pub mod text_decoder;
pub mod tokenizer;
pub mod vad;
pub mod vocab_int8;
pub mod wav2vec2;

pub mod whisper_candle;

pub use whisper_candle::WhisperCandle;

use ffai_core::engine::{AsrEngine, AsrOptions, EngineInfo, EngineStatus, Task};
use ffai_core::error::{Error, Result};
use ffai_core::types::{AudioBuffer, Transcript};

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
        Err(Error::NotImplemented {
            task: Task::Asr,
            engine: "oxiwhisper".into(),
        })
    }
}
