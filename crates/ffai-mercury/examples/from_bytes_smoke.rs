//! Does `WhisperCandle::from_bytes` produce the same transcript as the model
//! the manifest path builds — and the same one a wasm module produces?
//!
//! `from_bytes` is the constructor `ffai-mercury-wasm` uses, because a browser
//! has no filesystem and no mmap. A constructor that is only exercised on wasm
//! is a constructor nobody can debug, so this runs it NATIVELY against the same
//! three artefacts and the same audio the browser gets, and prints the
//! transcript for byte comparison against the wasm module's.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example from_bytes_smoke -- \
//!     <dir-with-the-three-files> <clip.wav>
//! ```
//!
//! The WAV must be 16 kHz mono PCM16 — Whisper's own format, which is what the
//! LibriSpeech clips in `corpora/` already are.

use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_core::types::AudioBuffer;
use ffai_mercury::asr::model::WhisperBytes;
use ffai_mercury::asr::text_decoder::Precision;
use ffai_mercury::asr::whisper_candle::WhisperCandle;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: from_bytes_smoke <model-dir> <clip.wav>");
    let wav = args.next().expect("usage: from_bytes_smoke <model-dir> <clip.wav>");

    let read = |name: &str| {
        std::fs::read(std::path::Path::new(&dir).join(name))
            .unwrap_or_else(|e| panic!("{dir}/{name}: {e}"))
    };
    let engine = WhisperCandle::from_bytes(
        WhisperBytes {
            weights: read("model.safetensors"),
            config: String::from_utf8(read("config.json")).expect("config.json is not UTF-8"),
            tokenizer: read("tokenizer.json"),
            name: "whisper-tiny-en".into(),
        },
        Precision::F32,
    )
    .expect("from_bytes");

    // A PCM16 mono WAV, walked chunk by chunk rather than assuming a 44-byte
    // header: a `LIST` chunk before `data` is common and would shift it.
    let raw = std::fs::read(&wav).unwrap_or_else(|e| panic!("{wav}: {e}"));
    let (mut off, mut rate, mut channels, mut data) = (12usize, 0u32, 1u16, &raw[0..0]);
    while off + 8 <= raw.len() {
        let id = &raw[off..off + 4];
        let sz = u32::from_le_bytes(raw[off + 4..off + 8].try_into().unwrap()) as usize;
        if id == b"fmt " {
            channels = u16::from_le_bytes(raw[off + 10..off + 12].try_into().unwrap());
            rate = u32::from_le_bytes(raw[off + 12..off + 16].try_into().unwrap());
        }
        if id == b"data" {
            data = &raw[off + 8..(off + 8 + sz).min(raw.len())];
            break;
        }
        off += 8 + sz + (sz & 1);
    }
    assert!(rate > 0 && !data.is_empty(), "{wav}: no fmt/data chunk found");
    let stride = 2 * channels as usize;
    let samples: Vec<f32> = data
        .chunks_exact(stride)
        .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
        .collect();
    println!(
        "audio: {rate} Hz, {channels}ch, {:.1}s",
        samples.len() as f64 / f64::from(rate)
    );

    let audio = AudioBuffer {
        samples,
        sample_rate: rate,
        channels: 1,
    };
    let out = engine
        .transcribe(&audio, &AsrOptions::default())
        .expect("transcribe");
    let text = out
        .segments
        .iter()
        .map(|s| s.value.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    println!("segments: {}", out.segments.len());
    println!("NATIVE: {text}");
}
