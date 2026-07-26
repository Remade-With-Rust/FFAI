//! # Mercury — FFai's voice component
//!
//! Named for the Roman god of language and messages (Greek: Hermes).
//! Mercury owns both directions of speech: **ASR** (speech → text) and
//! **TTS** (text → speech).
//!
//! Engine lineup (Phase 0 = registered stubs, see ROADMAP.md for landing
//! order):
//!
//! | Engine | Task | Plan |
//! |---|---|---|
//! | `whisper-candle` | asr | OpenAI Whisper on candle-transformers — the reference engine, first to go live |
//! | `oxiwhisper` | asr | cool-japan's pure-Rust GGUF Whisper — quantized/CPU-lean alternative |
//! | `any-tts` | tts | trait-based candle TTS (Kokoro, Qwen3-TTS, VibeVoice, …) |
//! | `voirs` | tts | full G2P→acoustic→vocoder framework (VITS/FastSpeech2 + HiFi-GAN) |
//!
//! The WhisperX layer (silero VAD, wav2vec2 forced alignment for word
//! timestamps, diarization) is exposed through [`ffai_core::engine::AsrOptions`]
//! — `word_timestamps` and `diarize` — and lands as Phase 1.5 on top of any
//! ASR engine.

mod asr;
mod tts;

pub use asr::{OxiWhisper, WhisperCandle};
pub use tts::{AnyTts, Voirs};

use std::sync::Arc;

use ffai_core::registry::EngineRegistry;

/// Install every Mercury engine into a registry.
pub fn register(reg: &mut EngineRegistry) {
    reg.register_asr(Arc::new(WhisperCandle));
    reg.register_asr(Arc::new(OxiWhisper));
    reg.register_tts(Arc::new(AnyTts));
    reg.register_tts(Arc::new(Voirs));
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffai_core::engine::Task;

    #[test]
    fn registers_asr_and_tts_engines() {
        let mut reg = EngineRegistry::new();
        register(&mut reg);
        let infos = reg.list();
        assert_eq!(infos.iter().filter(|i| i.task == Task::Asr).count(), 2);
        assert_eq!(infos.iter().filter(|i| i.task == Task::Tts).count(), 2);
        // default = first registered = the reference engine
        assert_eq!(reg.asr(None).unwrap().info().name, "whisper-candle");
        assert!(reg.asr(Some("whisper-candle")).is_ok());
        assert!(reg.asr(Some("nonexistent")).is_err());
    }

    #[test]
    fn stub_reports_not_implemented() {
        let mut reg = EngineRegistry::new();
        register(&mut reg);
        let audio = ffai_core::types::AudioBuffer {
            samples: vec![0.0; 160],
            sample_rate: 16_000,
            channels: 1,
        };
        let err = reg
            .asr(Some("whisper-candle"))
            .unwrap()
            .transcribe(&audio, &Default::default())
            .unwrap_err();
        assert!(err.to_string().contains("stub"));
    }
}
