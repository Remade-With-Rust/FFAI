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

pub mod asr;
pub mod tts;

pub use asr::{OxiWhisper, WhisperCandle};
pub use tts::{AnyTts, Voirs};

use std::sync::Arc;

use ffai_core::registry::EngineRegistry;

/// Install every Mercury engine into a registry.
pub fn register(reg: &mut EngineRegistry) {
    // Named engines are named *configurations*, the way ffmpeg exposes codec
    // presets: same implementation, different model size.
    reg.register_asr(Arc::new(WhisperCandle::new()));
    reg.register_asr(Arc::new(WhisperCandle::with_model(
        "models",
        "whisper-tiny-en",
        asr::text_decoder::Precision::Q8_0,
    )));
    reg.register_asr(Arc::new(WhisperCandle::with_model(
        "models",
        "whisper-base-en",
        asr::text_decoder::Precision::F32,
    )));
    reg.register_asr(Arc::new(WhisperCandle::with_model(
        "models",
        "whisper-base-en",
        asr::text_decoder::Precision::Q8_0,
    )));
    reg.register_asr(Arc::new(OxiWhisper));
    // piper-candle first: the first registered TTS engine is the default,
    // and it is the only live one (M-T2); the stubs stay visible after it.
    reg.register_tts(Arc::new(tts::PiperCandle::new()));
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
        // tiny f32/q8_0, base f32/q8_0, oxiwhisper
        assert_eq!(infos.iter().filter(|i| i.task == Task::Asr).count(), 5);
        // piper-candle (live, default) + the any-tts/voirs stubs
        assert_eq!(infos.iter().filter(|i| i.task == Task::Tts).count(), 3);
        // default = first registered = the reference engine
        assert_eq!(reg.asr(None).unwrap().info().name, "whisper-candle");
        assert_eq!(reg.tts(None).unwrap().info().name, "piper-candle");
        assert!(reg.asr(Some("whisper-candle")).is_ok());
        assert!(reg.asr(Some("nonexistent")).is_err());
    }

    #[test]
    fn remaining_stubs_say_so_instead_of_failing_obscurely() {
        let mut reg = EngineRegistry::new();
        register(&mut reg);
        let audio = ffai_core::types::AudioBuffer {
            samples: vec![0.0; 160],
            sample_rate: 16_000,
            channels: 1,
        };
        let err = reg
            .asr(Some("oxiwhisper"))
            .unwrap()
            .transcribe(&audio, &Default::default())
            .unwrap_err();
        assert!(err.to_string().contains("stub"), "got: {err}");
    }

    #[test]
    fn whisper_candle_is_no_longer_a_stub() {
        let mut reg = EngineRegistry::new();
        register(&mut reg);
        let info = reg.asr(Some("whisper-candle")).unwrap().info();
        assert_eq!(info.status, ffai_core::engine::EngineStatus::Experimental);
    }

    #[test]
    fn diarization_no_longer_refuses_and_reaches_the_model() {
        // Both WhisperX stages used to be refused here. Word timestamps
        // landed first, then diarization; what is left to assert is that the
        // flag is HONOURED rather than silently ignored — the failure mode
        // that would be worse than an error.
        let mut reg = EngineRegistry::new();
        register(&mut reg);
        let audio = ffai_core::types::AudioBuffer {
            samples: vec![0.0; 160],
            sample_rate: 16_000,
            channels: 1,
        };
        let opts = ffai_core::engine::AsrOptions { diarize: true, ..Default::default() };
        let err = reg
            .asr(Some("whisper-candle"))
            .unwrap()
            .transcribe(&audio, &opts)
            .unwrap_err();
        // 160 samples of silence: too short for the speaker model to embed,
        // so this exercises the wiring rather than the network. Whatever it
        // reports, it must not be the old "not built yet" refusal.
        let msg = err.to_string();
        assert!(!msg.contains("not built yet"), "stale refusal still present: {msg}");
        assert!(!msg.contains("phase D"), "stale refusal still present: {msg}");
    }
}
