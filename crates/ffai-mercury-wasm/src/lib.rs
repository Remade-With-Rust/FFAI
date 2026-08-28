//! Mercury in the browser: pure-Rust Whisper speech recognition on WebAssembly.
//!
//! The mel front end, the encoder, the decoder and the tokenizer all run inside
//! the wasm module. There is no whisper.cpp, no ONNX Runtime Web, and no
//! JavaScript inference engine underneath — the same Rust that runs natively is
//! the Rust that runs here.
//!
//! ## What is different on this target, and what is not
//!
//! **Serial, because there is nothing to parallelise onto.**
//! `wasm32-unknown-unknown` has no threads, so `ffai_mercury::par` supplies the
//! identical serial iterators and `rayon` is not a dependency of this build at
//! all. Mercury reaches rayon on the hot path — `flash_attn` over attention
//! heads, `text_decoder` over logit chunks, `vocab_int8` over vocabulary blocks
//! — so without the shim a browser build panics inside the first forward pass
//! rather than at load.
//!
//! **No clock, and one decision had to stop depending on one.**
//! `Instant::now()` panics here, and `adaptive::matmul_dtype` was *timing* two
//! candidate matmuls to choose between f32 and f16. On wasm it now answers
//! `F32` directly — the same verdict its own "cannot be timed" fallback
//! documents, reached without running probe matmuls to learn nothing.
//!
//! **Weights are supplied by the caller, never bundled.** There is no
//! filesystem to read and no mmap to take, so [`Recognizer::whisper`] takes the
//! three artefacts as bytes. `WhisperCandle::from_bytes` is gated against the
//! manifest constructor for identical output, so this is not a more lenient
//! door into the engine.
//!
//! ## 16 kHz mono, and why this crate does not resample for you
//!
//! Whisper is a 16 kHz mono model and the engine refuses anything else rather
//! than resampling silently. A browser already has a correct, fast resampler
//! that a wasm module has no business duplicating:
//!
//! ```js
//! const off = new OfflineAudioContext(1, Math.ceil(buf.duration * 16000), 16000);
//! const src = off.createBufferSource();
//! src.buffer = buf;
//! src.connect(off.destination);
//! src.start();
//! const mono16k = (await off.startRendering()).getChannelData(0);
//! ```
//!
//! ```js
//! import init, { Recognizer } from './ffai_mercury_wasm.js';
//! await init();
//! const r = Recognizer.whisper(weights, configJson, tokenizerJson, 'whisper-tiny-en');
//! console.log(r.text(mono16k, 16000));
//! ```

use wasm_bindgen::prelude::*;

use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_core::types::AudioBuffer;
use ffai_mercury::asr::model::WhisperBytes;
use ffai_mercury::asr::whisper_candle::WhisperCandle;

/// The cdylib is the binary, so it is the thing entitled to pick an allocator.
///
/// See the note in `Cargo.toml`: `ffai-carmenta-wasm`'s A/B found a real
/// segment-recycling defect in `rusty_alloc` <= 1.1.4 on this target, fixed
/// upstream in 1.1.5 and 1.1.6. Mercury's allocation pattern is different
/// enough — one long encoder/decoder chain rather than many small crops — that
/// the answer does not carry over, and `--no-default-features` is what makes it
/// answerable.
#[cfg(feature = "rusty-alloc")]
#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

/// One transcript segment, flattened for the JS boundary.
#[wasm_bindgen(getter_with_clone)]
pub struct Segment {
    pub text: String,
    /// Start time in seconds, in the SOURCE audio's timeline.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
}

/// A loaded speech-recognition pipeline.
#[wasm_bindgen]
pub struct Recognizer {
    engine: WhisperCandle,
}

// `Self` is not usable in these return positions: `#[wasm_bindgen]` builds its
// glue from the written type and does not resolve `Self`, so `use_self` is
// wrong here rather than merely stylistic.
#[allow(clippy::use_self)]
#[wasm_bindgen]
impl Recognizer {
    /// Load a Whisper checkpoint from its three artefacts.
    ///
    /// `weights` is `model.safetensors`, `config` the text of `config.json`,
    /// and `tokenizer` the bytes of `tokenizer.json` — the same three files the
    /// manifest resolves natively, so a browser and a server parse identical
    /// inputs. `name` is what the model calls itself in `info()` and in errors.
    ///
    /// **Pick the smallest checkpoint you can live with.** These are 32-bit
    /// address spaces: `whisper-tiny-en` is ~75 MB of f32 weights and
    /// comfortable, `base` ~145 MB, and the large checkpoints are not a browser
    /// proposition at all. Loading two pipelines into one module instance is
    /// also not one — `ffai-carmenta-wasm` measured ~118 MB of weights in a
    /// single linear memory failing to complete.
    #[wasm_bindgen]
    pub fn whisper(
        weights: Vec<u8>,
        config: String,
        tokenizer: Vec<u8>,
        name: String,
    ) -> Result<Recognizer, JsValue> {
        console_error_panic_hook();
        let engine = WhisperCandle::from_bytes(
            WhisperBytes {
                weights,
                config,
                tokenizer,
                name,
            },
            ffai_mercury::asr::text_decoder::Precision::F32,
        )
        .map_err(|e| JsValue::from_str(&format!("model load failed: {e}")))?;
        Ok(Self { engine })
    }

    /// Transcribe 16 kHz mono `f32` samples — the layout an
    /// `OfflineAudioContext` render hands out (see the module docs).
    ///
    /// `sample_rate` is taken rather than assumed so a mismatch is an error
    /// naming both rates, instead of a transcript that is quietly wrong because
    /// the audio played back at the wrong speed to the model.
    pub fn transcribe(&self, samples: Vec<f32>, sample_rate: u32) -> Result<Vec<Segment>, JsValue> {
        let out = self.run(samples, sample_rate)?;
        Ok(out
            .segments
            .iter()
            .map(|s| Segment {
                text: s.value.clone(),
                start: s.start,
                end: s.end,
            })
            .collect())
    }

    /// The whole utterance as one string, segments joined with spaces.
    pub fn text(&self, samples: Vec<f32>, sample_rate: u32) -> Result<String, JsValue> {
        let out = self.run(samples, sample_rate)?;
        Ok(out
            .segments
            .iter()
            .map(|s| s.value.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" "))
    }

    /// Detected language tag, once something has been transcribed.
    #[wasm_bindgen(js_name = detectLanguage)]
    pub fn detect_language(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> Result<Option<String>, JsValue> {
        Ok(self.run(samples, sample_rate)?.language)
    }

    fn run(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> Result<ffai_core::types::Transcript, JsValue> {
        if samples.is_empty() {
            return Err(JsValue::from_str("no audio: samples is empty"));
        }
        let audio = AudioBuffer {
            samples,
            sample_rate,
            channels: 1,
        };
        // Word timestamps and diarization both load a SECOND model on first
        // use, which a 32-bit address space that already holds Whisper cannot
        // afford. They stay off here rather than failing halfway through a
        // transcript with an allocation error.
        let opts = AsrOptions::default();
        self.engine
            .transcribe(&audio, &opts)
            .map_err(|e| JsValue::from_str(&format!("transcribe failed: {e}")))
    }
}

/// Current size of the wasm linear memory, in bytes.
///
/// The 4 GB cap is a cap on THIS number, and linear memory only ever grows — so
/// a growth curve across repeated calls is the only honest memory instrument on
/// this target. `performance.memory` measures the JS heap and says nothing
/// about ours.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = linearMemoryBytes)]
#[must_use]
pub fn linear_memory_bytes() -> f64 {
    (core::arch::wasm32::memory_size::<0>() * 65536) as f64
}

/// Which allocator this module was built with, so a browser A/B can report it.
#[wasm_bindgen]
#[must_use]
pub fn allocator() -> String {
    #[cfg(feature = "rusty-alloc")]
    {
        format!("rusty_alloc {}", rusty_alloc_api::VERSION)
    }
    // No `#[global_allocator]`, so this is whatever the target ships —
    // dlmalloc on wasm32-unknown-unknown.
    #[cfg(not(feature = "rusty-alloc"))]
    {
        "dlmalloc (target default)".to_string()
    }
}

/// Route Rust panics to `console.error` instead of an opaque `unreachable`.
fn console_error_panic_hook() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            web_error(&format!("{info}"));
        }));
    });
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn web_error(s: &str);
}
