//! # Mercury — FFai's voice component
//!
//! Named for the Roman god of language and messages (Greek: Hermes).
//! Mercury owns both directions of speech: **ASR** (speech → text) and
//! **TTS** (text → speech).
//!
//! Engine lineup:
//!
//! | Engine | Task | Status |
//! |---|---|---|
//! | `whisper-candle` | asr | **live** — OpenAI Whisper on candle: our own mel front-end, tokenizer grammar, decode loop, audio encoder, AVX2 kernels. Sizes and q8_0 variants are named configurations of the same engine |
//! | `piper-candle` | tts | **live** — the full VITS stack on candle, running piper's own voice files; deterministic, oracle-exact against piper's runtime |
//! | `oxiwhisper` | asr | stub — cool-japan's pure-Rust GGUF Whisper, on the watchlist |
//! | `any-tts` | tts | stub — trait-based candle TTS (Kokoro, Qwen3-TTS, VibeVoice, …) |
//! | `voirs` | tts | stub — alternative G2P→acoustic→vocoder lineage |
//!
//! The WhisperX layer (energy VAD, wav2vec2 forced alignment for word
//! timestamps, ECAPA-TDNN diarization with cross-call speaker persistence) is
//! exposed through [`ffai_core::engine::AsrOptions`] — `word_timestamps`,
//! `diarize`, `persist_speakers` — as flags on any ASR engine, not a fork.
//!
//! TTS is decomposed the same way ([`tts`]): [`tts::phonemize`] is a
//! clean-room pure-Rust G2P over CMUdict, so nothing GPL is linked (espeak-ng
//! — the reason piper itself is GPL — serves only as an out-of-process test
//! oracle). Synthesis knobs live on [`ffai_core::engine::TtsOptions`], and
//! `seed` makes output byte-identical run over run.

pub mod asr;
pub mod tts;

pub use asr::{OxiWhisper, WhisperCandle};
pub use tts::{AnyTts, PiperCandle, Voirs};

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
    // The accuracy tiers. Registered late so the default stays tiny.en —
    // the first registered engine is the default, and a 769M model is not a
    // default anyone asked for. Above medium the family is multilingual-only,
    // which needs language detection we do not have (docs/mercury-asr-todo.md
    // §1), so medium.en is the largest size that is honestly usable today.
    reg.register_asr(Arc::new(WhisperCandle::with_model(
        "models",
        "whisper-small-en",
        asr::text_decoder::Precision::F32,
    )));
    reg.register_asr(Arc::new(WhisperCandle::with_model(
        "models",
        "whisper-medium-en",
        asr::text_decoder::Precision::F32,
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
        // tiny f32/q8_0, base f32/q8_0, small.en, medium.en, oxiwhisper
        assert_eq!(infos.iter().filter(|i| i.task == Task::Asr).count(), 7);
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
