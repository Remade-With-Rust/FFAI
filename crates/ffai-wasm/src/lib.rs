//! Diana in the browser: YOLO26 object detection compiled to WebAssembly.
//!
//! The whole graph runs in wasm — backbone, neck, the NMS-free one2one head,
//! the letterbox and the decode. There is no ONNX Runtime Web underneath, no
//! TensorFlow.js, and no JavaScript inference engine: the same Rust that runs
//! natively is the Rust that runs here.
//!
//! ## What is different on this target, and what is not
//!
//! **Serial, and that is the arm that was already winning on CPU.**
//! `wasm32-unknown-unknown` has no threads to spawn, so `crate::par` supplies
//! the identical serial iterators and `rayon` is not a dependency at all.
//! Diana measured the fan-out at **363 ms of CPU per image at one thread
//! against 844 ms at twenty-four** — the work is 363 ms and the pool spends
//! 844 ms doing it. Wasm loses wall-clock parallelism; it does not lose the
//! efficient arm.
//!
//! **Weights are supplied by the caller, never bundled.** YOLO26 checkpoints
//! are AGPL-3.0 and this crate ships none. `Detector::new` takes the
//! safetensors bytes and the manifest JSON, which is also the only thing that
//! works in a browser — there is no filesystem to read from.
//!
//! ```js
//! import init, { Detector } from './ffai_wasm.js';
//! await init();
//! const det = new Detector(safetensorsBytes, manifestJson, 'n');
//! const found = det.detect(rgbaFromCanvas, width, height, 0.25);
//! ```
use wasm_bindgen::prelude::*;

use ffai_core::engine::{DetectEngine, DetectOptions};
use ffai_core::types::{ImageBuffer, PixelFormat};
use ffai_diana::engine::Yolo26;
use ffai_diana::image::Geometry;

/// The cdylib is the binary, so it is the thing entitled to pick an allocator.
///
/// wasm32's Rust default is `dlmalloc`. `rusty_alloc` is the pure-Rust remake
/// of mimalloc that replaced the C allocator in our native binary at parity,
/// and it already carries a wasm backend plus a measured wasm-specific
/// decision (no arena pre-reservation, because `memory.grow` is never returned
/// to the host). Which of the two is faster HERE is an open measurement, and
/// having this crate is what makes it answerable.
#[cfg(feature = "rusty-alloc")]
#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

/// One detection, flattened for the JS boundary.
#[wasm_bindgen(getter_with_clone)]
pub struct Detection {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub class_id: u32,
    pub name: String,
    pub confidence: f32,
}

/// A loaded YOLO26 model.
#[wasm_bindgen]
pub struct Detector {
    engine: Yolo26,
    names: Vec<String>,
}

#[wasm_bindgen]
impl Detector {
    /// Build from checkpoint bytes the caller supplies.
    ///
    /// `tier` is one of `n`, `s`, `m`, `l`, `x`. Rectangular geometry is the
    /// default for the same reason it is natively: a 1920x1080 frame
    /// letterboxes to 640x384 rather than 640x640, which is 30 % less model
    /// work for identical accuracy.
    #[wasm_bindgen(constructor)]
    pub fn new(safetensors: &[u8], manifest_json: &str, tier: &str) -> Result<Detector, JsValue> {
        console_error_panic_hook();
        let engine = Yolo26::from_bytes(tier, Geometry::Rect, safetensors.to_vec(), manifest_json)
            .map_err(|e| JsValue::from_str(&format!("model load failed: {e}")))?;
        let names = engine.class_names().to_vec();
        Ok(Detector { engine, names })
    }

    /// Detect in an RGBA buffer, the layout `CanvasRenderingContext2D.getImageData`
    /// hands out. Alpha is ignored; the letterbox reads the first three channels.
    pub fn detect(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        confidence: f32,
    ) -> Result<Vec<Detection>, JsValue> {
        let want = (width as usize) * (height as usize) * 4;
        if rgba.len() != want {
            return Err(JsValue::from_str(&format!(
                "expected {want} bytes of RGBA for {width}x{height}, got {}",
                rgba.len()
            )));
        }
        let image = ImageBuffer {
            data: rgba.to_vec(),
            width,
            height,
            format: PixelFormat::Rgba8,
        };
        let opts = DetectOptions { confidence, ..Default::default() };
        let found = self
            .engine
            .detect(&image, &opts)
            .map_err(|e| JsValue::from_str(&format!("detect failed: {e}")))?;
        Ok(found
            .detections
            .iter()
            .map(|d| Detection {
                x0: d.x0,
                y0: d.y0,
                x1: d.x1,
                y1: d.y1,
                class_id: d.class_id,
                name: self.names.get(d.class_id as usize).cloned().unwrap_or_default(),
                confidence: d.confidence,
            })
            .collect())
    }

    /// Class names in class-id order.
    #[wasm_bindgen(js_name = classNames)]
    pub fn class_names(&self) -> Vec<String> {
        self.names.clone()
    }
}

/// Which allocator this module was built with, so a browser A/B can report it.
#[wasm_bindgen]
pub fn allocator() -> String {
    #[cfg(feature = "rusty-alloc")]
    {
        format!("rusty_alloc {}", rusty_alloc_api::VERSION)
    }
    // No `#[global_allocator]`, so this is whatever the target ships — dlmalloc
    // on wasm32-unknown-unknown.
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
