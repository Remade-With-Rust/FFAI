//! Mercury TTS: text → speech.
//!
//! Decomposed into independently testable stages (mission plan §2), the same
//! law the ASR side follows. Live today:
//!
//! | Stage | Contract |
//! |---|---|
//! | [`normalize`] | raw text → speakable words (numbers, case) |
//! | [`lexicon`] | `CMUdict`: word → ARPABET pronunciations |
//! | [`phonemize`] | sentence → espeak-compatible IPA phoneme string |
//!
//! The synthesis stages (`vits`, `voice`, the `piper-candle` engine) land in
//! M-T2; the engines below are honest stubs until then.

pub mod chunk;
pub mod decoder_kernels;
pub mod lexicon;
pub mod normalize;
pub mod onnx;
pub mod phoneme_ids;
pub mod phonemize;
pub mod piper_candle;
pub mod vits;

pub use piper_candle::PiperCandle;

use ffai_core::engine::{EngineInfo, EngineStatus, Task, TtsEngine, TtsOptions};
use ffai_core::error::{Error, Result};
use ffai_core::types::AudioBuffer;

/// any-tts: one trait-based candle API over Kokoro-82M, `OmniVoice`, Qwen3-TTS,
/// `VibeVoice`, and Voxtral. Architecturally a sibling of `FFai`'s own engine
/// registry — planned as the first live TTS engine (Phase 2), likely as a
/// direct dependency with contributions upstream.
///
/// Weight-license caveat: several supported voices (e.g. some `VibeVoice`
/// checkpoints) are CC BY-NC — surfaced per-model via `ffai-models`
/// manifests, never silently bundled.
pub struct AnyTts;

impl TtsEngine for AnyTts {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "any-tts".into(),
            task: Task::Tts,
            status: EngineStatus::Stub,
            description: "candle TTS multiplexer: Kokoro/Qwen3-TTS/VibeVoice (Phase 2)".into(),
        }
    }

    fn synthesize(&self, _text: &str, _opts: &TtsOptions) -> Result<AudioBuffer> {
        Err(Error::NotImplemented {
            task: Task::Tts,
            engine: "any-tts".into(),
        })
    }
}

/// `VoiRS`: full pure-Rust G2P → acoustic (VITS/FastSpeech2) → vocoder
/// (HiFi-GAN/DiffWave) pipeline. The heavier second engine, valuable for its
/// training support and voice breadth.
pub struct Voirs;

impl TtsEngine for Voirs {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "voirs".into(),
            task: Task::Tts,
            status: EngineStatus::Stub,
            description: "VoiRS G2P→acoustic→vocoder framework (Phase 2 alternative)".into(),
        }
    }

    fn synthesize(&self, _text: &str, _opts: &TtsOptions) -> Result<AudioBuffer> {
        Err(Error::NotImplemented {
            task: Task::Tts,
            engine: "voirs".into(),
        })
    }
}
