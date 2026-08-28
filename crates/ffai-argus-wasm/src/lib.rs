//! Argus in the browser: pure-Rust `SmolVLM` image captioning on WebAssembly.
//!
//! Preprocessing, the `SigLIP` vision tower, the pixel-shuffle connector and the
//! Llama text decoder all run inside the wasm module. There is no ONNX Runtime
//! Web, no transformers.js, and no Python anywhere — the same Rust that runs
//! natively is the Rust that runs here.
//!
//! ## What is different on this target, and what is not
//!
//! **Serial, because there is nothing to parallelise onto.**
//! `wasm32-unknown-unknown` has no threads, so `ffai_argus::par` supplies the
//! identical serial iterators and `rayon` is not a dependency of this build at
//! all. Argus reaches rayon at roughly a dozen sites across the vision tower,
//! so without the shim a browser build panics inside the first tile rather
//! than at load.
//!
//! **No clock.** `Instant::now()` panics here, and Argus called it 51 times.
//! Every one of those feeds a millisecond REPORT rather than a decision — that
//! was checked, not assumed — so on wasm the profile tables read all-zero
//! instead of lying about a number the target cannot take.
//!
//! **Weights are supplied by the caller, never bundled.** There is no
//! filesystem to read and no mmap to take, so [`Captioner::smolvlm`] takes the
//! three artefacts as bytes. `SmolVlm::from_bytes` funnels through the same
//! `build` the manifest path uses, so this is not a more lenient door into the
//! engine.
//!
//! **One `VarBuilder`, cloned.** The native loader mapped the checkpoint twice
//! — once renamed for candle's tower, once raw for ours. That is free with an
//! mmap and is a second full COPY of the weights in a 32-bit address space, so
//! the loader now clones a builder instead. Native gets the same saving.
//!
//! ## Size is the binding constraint
//!
//! `SmolVLM-256M` at f32 is roughly 1 GB of weights against a 4 GB linear
//! memory that only ever grows. It fits, with room for the vision tower's
//! activations and a KV cache, and nothing larger is a browser proposition.
//! Tiling is what makes an image expensive: a still is 17 tiles / 1088 image
//! tokens, so prefer [`Captioner::describe`]'s default over a large image.
//!
//! ```js
//! import init, { Captioner } from './ffai_argus_wasm.js';
//! await init();
//! const c = Captioner.smolvlm(weights, configJson, tokenizerJson);
//! console.log(c.describe(rgbaFromCanvas, width, height, 'What is in this image?'));
//! ```

use wasm_bindgen::prelude::*;

use ffai_argus::engine::{ArgusBytes, SmolVlm};
use ffai_core::engine::{VlmEngine, VlmOptions};
use ffai_core::types::{ImageBuffer, PixelFormat};

/// The cdylib is the binary, so it is the thing entitled to pick an allocator.
///
/// See the note in `Cargo.toml`: `ffai-carmenta-wasm`'s A/B found a real
/// segment-recycling defect in `rusty_alloc` <= 1.1.4 on this target, fixed
/// upstream in 1.1.5 and 1.1.6. Argus's allocation pattern is different enough
/// — a vision tower over tiles feeding a decoder with a growing KV cache — that
/// the answer does not carry over, and `--no-default-features` is what makes it
/// answerable.
#[cfg(feature = "rusty-alloc")]
#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

/// A loaded vision-language pipeline.
#[wasm_bindgen]
pub struct Captioner {
    engine: SmolVlm,
}

// `Self` is not usable in these return positions: `#[wasm_bindgen]` builds its
// glue from the written type and does not resolve `Self`, so `use_self` is
// wrong here rather than merely stylistic.
#[allow(clippy::use_self)]
#[wasm_bindgen]
impl Captioner {
    /// Load `SmolVLM-256M-Instruct` from its three artefacts.
    ///
    /// `weights` is `model.safetensors`, `config` the text of `config.json`,
    /// and `tokenizer` the bytes of `tokenizer.json` — the same three files the
    /// manifest resolves natively, so a browser and a server parse identical
    /// inputs.
    ///
    /// The geometry, the `<image>` token id and the end-of-turn token set are
    /// all read from the checkpoint's own config here, never assumed, so a
    /// different `SmolVLM` size loads correctly rather than silently mis-tiling.
    #[wasm_bindgen]
    pub fn smolvlm(
        weights: Vec<u8>,
        config: String,
        tokenizer: Vec<u8>,
    ) -> Result<Captioner, JsValue> {
        console_error_panic_hook();
        let engine = SmolVlm::from_bytes(ArgusBytes {
            weights,
            config,
            tokenizer,
        })
        .map_err(|e| JsValue::from_str(&format!("model load failed: {e}")))?;
        Ok(Self { engine })
    }

    /// Caption an RGBA buffer — the layout `getImageData` hands out.
    ///
    /// `prompt` is the instruction; pass an empty string for the model's own
    /// default. Alpha is ignored.
    pub fn describe(
        &self,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        prompt: String,
    ) -> Result<String, JsValue> {
        let want = (width as usize) * (height as usize) * 4;
        if rgba.len() != want {
            return Err(JsValue::from_str(&format!(
                "expected {want} bytes of RGBA for {width}x{height}, got {}",
                rgba.len()
            )));
        }
        // `Vec<u8>` rather than `&[u8]`, and that is a memory decision rather
        // than a style one: wasm-bindgen has ALREADY copied the caller's bytes
        // into linear memory to form this argument, so borrowing them would
        // make `ImageBuffer` copy a second time.
        let image = ImageBuffer {
            data: rgba,
            width,
            height,
            format: PixelFormat::Rgba8,
        };
        let opts = VlmOptions {
            prompt: (!prompt.trim().is_empty()).then_some(prompt),
            ..VlmOptions::default()
        };
        self.engine
            .describe_image(&image, &opts)
            .map_err(|e| JsValue::from_str(&format!("describe failed: {e}")))
    }

    /// As [`Self::describe`], with an explicit token budget.
    ///
    /// Decode is autoregressive and the KV cache grows with every token, so
    /// this is the knob that bounds both time and memory on a long answer.
    #[wasm_bindgen(js_name = describeWithLimit)]
    pub fn describe_with_limit(
        &self,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        prompt: String,
        max_new_tokens: usize,
    ) -> Result<String, JsValue> {
        let want = (width as usize) * (height as usize) * 4;
        if rgba.len() != want {
            return Err(JsValue::from_str(&format!(
                "expected {want} bytes of RGBA for {width}x{height}, got {}",
                rgba.len()
            )));
        }
        let image = ImageBuffer {
            data: rgba,
            width,
            height,
            format: PixelFormat::Rgba8,
        };
        let opts = VlmOptions {
            prompt: (!prompt.trim().is_empty()).then_some(prompt),
            max_new_tokens: Some(max_new_tokens),
            ..VlmOptions::default()
        };
        self.engine
            .describe_image(&image, &opts)
            .map_err(|e| JsValue::from_str(&format!("describe failed: {e}")))
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
