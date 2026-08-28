//! Carmenta in the browser: pure-Rust OCR compiled to WebAssembly.
//!
//! Detection, recognition and reading order all run inside the wasm module.
//! There is no Tesseract.js, no ONNX Runtime Web, and no JavaScript inference
//! engine underneath — the same Rust that runs natively is the Rust that runs
//! here.
//!
//! ## What is different on this target, and what is not
//!
//! **Serial, because there is nothing to parallelise onto.**
//! `wasm32-unknown-unknown` has no threads, so `ffai_carmenta::par` supplies
//! the identical serial iterators and `rayon` is not a dependency of this
//! build at all. Natively that costs less than it sounds: three rayon levels
//! nest there, and a one-line strip measured **177 ms/line under `par_iter`
//! against 82 ms serial** (§8.100).
//!
//! **candle's convolution, not ours.** `ffai_carmenta::conv3x3` beats candle
//! by 1.65x *when vectorised*; every scalar form of it measured below candle,
//! and wasm has no AVX2. Measured interleaved on a 620x200 capture: ours
//! scalar 5282 ms, candle 2771 ms. The kernel is `cfg`-ed off here, which is
//! the largest single win in this build and cost one line.
//!
//! **Weights are supplied by the caller, never bundled.** There is no
//! filesystem to read and no mmap to take, so [`Reader::new`] takes the
//! safetensors bytes and the charset text. `CraftCrnn::from_bytes` is gated
//! against the filesystem constructor for identical output, so this is not a
//! more lenient door into the engine.
//!
//! ## Which pair to ship
//!
//! `mobiledet-svtr` — `PP-OCRv5` mobile det + mobile rec. The detector is
//! **4.7 MB against CRAFT's 83 MB**, which matters more in a browser than
//! anywhere else, and it measured **1.71x faster end to end** than
//! `craft-crnn` on a browser-sized capture. `craft-crnn` remains reachable for
//! callers who want the `EasyOCR` lineage and can pay for the download.
//!
//! ```js
//! import init, { Reader } from './ffai_carmenta_wasm.js';
//! await init();
//! const r = Reader.mobiledetSvtr(detBytes, recBytes, charsetText);
//! const lines = r.read(rgbaFromCanvas, width, height);
//! ```

use wasm_bindgen::prelude::*;

use ffai_carmenta::engine::{CraftCrnn, DetStage, RecStage, WeightBytes};
use ffai_core::engine::{OcrEngine, OcrOptions};
use ffai_core::types::{ImageBuffer, PixelFormat};

/// The cdylib is the binary, so it is the thing entitled to pick an allocator.
///
/// See the note in `Cargo.toml`: `ffai-wasm`'s A/B put `rusty_alloc` at parity
/// with `dlmalloc` on Diana's workload, and Carmenta's allocation pattern —
/// many small line crops rather than one long tensor chain — is different
/// enough that the answer does not carry over. Having the cdylib is what makes
/// it answerable.
#[cfg(feature = "rusty-alloc")]
#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

/// One recognized line, flattened for the JS boundary.
#[wasm_bindgen(getter_with_clone)]
pub struct Line {
    pub text: String,
    /// Box in ORIGINAL image pixels — the same x/y/width/height shape
    /// `ffai_core` carries, so a caller can pass it straight to
    /// `ctx.strokeRect` without knowing anything about the internal scale.
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
}

/// A loaded OCR pipeline.
///
#[wasm_bindgen]
pub struct Reader {
    engine: CraftCrnn,
}

// `Self` is not usable in these return positions: `#[wasm_bindgen]` builds its
// glue from the written type and does not resolve `Self`, so `use_self` is
// wrong here rather than merely stylistic.
#[allow(clippy::use_self)]
#[wasm_bindgen]
impl Reader {
    /// `PP-OCRv5` mobile det + mobile rec (SVTR) — **the pair to ship**.
    ///
    /// 4.7 MB of detector against CRAFT's 83 MB, and 1.71x faster end to end
    /// on browser-sized input. `charset` is the text of the model's
    /// `charset.txt`, one entry per line: its length plus one blank must equal
    /// the recognition head's class count, and the engine asserts that at load
    /// rather than trusting it — an off-by-one there shifts every decoded
    /// character and reads as a bad model instead of a bad loader.
    #[wasm_bindgen(js_name = mobiledetSvtr)]
    pub fn mobiledet_svtr(det: &[u8], rec: &[u8], charset: &str) -> Result<Reader, JsValue> {
        console_error_panic_hook();
        Self::build(
            DetStage::MobileDet,
            RecStage::Svtr,
            WeightBytes {
                mobiledet: Some(det.to_vec()),
                svtr: Some((rec.to_vec(), charset.to_string())),
                ..WeightBytes::default()
            },
        )
    }

    /// PP-OCRv5 mobile det + `english_g2` CRNN — **the pair to ship in a
    /// browser**, and not the pair either component's native ranking predicts.
    ///
    /// Measured in Node, 200x40 crop, `readLine` (detection skipped), so this
    /// is purely the recognizer:
    ///
    /// | recognizer | |
    /// |---|---:|
    /// | CRNN | **206 ms** |
    /// | SVTR | 8905 ms |
    ///
    /// **43x.** SVTR is a transformer and reaches candle's `gemm` for
    /// everything, and `gemm` has no SIMD on wasm (see the module docs); CRNN
    /// is convolutions and an LSTM. Natively SVTR is about 1.9x slower than
    /// CRNN and worth it for accuracy — the crossing to wasm turns that into
    /// two orders of magnitude, which is the kind of thing a native benchmark
    /// cannot tell you.
    ///
    /// Pairing that recognizer with the 4.7 MB mobile DETECTOR — rather than
    /// CRAFT's 83 MB, which measured 1.71x slower end to end natively as well
    /// — gives the smallest download and the fastest read of the three
    /// combinations offered here.
    #[wasm_bindgen(js_name = mobiledetCrnn)]
    pub fn mobiledet_crnn(det: &[u8], crnn: &[u8]) -> Result<Reader, JsValue> {
        console_error_panic_hook();
        Self::build(
            DetStage::MobileDet,
            RecStage::Crnn,
            WeightBytes {
                mobiledet: Some(det.to_vec()),
                crnn: Some(crnn.to_vec()),
                ..WeightBytes::default()
            },
        )
    }

    /// CRAFT + `english_g2` CRNN — the `EasyOCR` lineage.
    ///
    /// 83 MB of detector weights, which is a real cost over a network. Offered
    /// because it is the pair the accuracy record is written against, not
    /// because it is the right default here.
    #[wasm_bindgen(js_name = craftCrnn)]
    pub fn craft_crnn(craft: &[u8], crnn: &[u8]) -> Result<Reader, JsValue> {
        console_error_panic_hook();
        Self::build(
            DetStage::Craft,
            RecStage::Crnn,
            WeightBytes {
                craft: Some(craft.to_vec()),
                crnn: Some(crnn.to_vec()),
                ..WeightBytes::default()
            },
        )
    }

    fn build(det: DetStage, rec: RecStage, w: WeightBytes) -> Result<Reader, JsValue> {
        let engine = CraftCrnn::from_bytes(det, rec, w)
            .map_err(|e| JsValue::from_str(&format!("model load failed: {e}")))?;
        Ok(Self { engine })
    }

    /// Read an RGBA buffer — the layout `CanvasRenderingContext2D.getImageData`
    /// hands out. Alpha is ignored.
    ///
    /// `Vec<u8>` rather than `&[u8]`, and that is a memory fix rather than a
    /// style choice: wasm-bindgen has ALREADY copied the caller's bytes into
    /// linear memory to form the argument. Borrowing them meant `ImageBuffer`
    /// had to copy a second time, so a 1240x1754 page held two 8.7 MB copies of
    /// the same pixels at once. Taking ownership hands that first copy straight
    /// through — same JS signature, one buffer instead of two.
    pub fn read(&self, rgba: Vec<u8>, width: u32, height: u32) -> Result<Vec<Line>, JsValue> {
        self.read_with(rgba, width, height, false)
    }

    /// As [`Self::read`], but treating the whole image as ONE text line —
    /// detection skipped entirely.
    ///
    /// This is the fast path and, on a browser-sized crop, by far the most
    /// useful one: detection is **86-89 % of the time** on a small capture, so
    /// a caller who already knows where the text is (a selection rectangle, a
    /// receipt field, a subtitle bar) skips almost all of the cost.
    #[wasm_bindgen(js_name = readLine)]
    pub fn read_line(&self, rgba: Vec<u8>, width: u32, height: u32) -> Result<Vec<Line>, JsValue> {
        self.read_with(rgba, width, height, true)
    }

    fn read_with(
        &self,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        single_line: bool,
    ) -> Result<Vec<Line>, JsValue> {
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
        let opts = OcrOptions {
            single_line,
            ..OcrOptions::default()
        };
        let out = self
            .engine
            .recognize(&image, &opts)
            .map_err(|e| JsValue::from_str(&format!("recognize failed: {e}")))?;
        Ok(out
            .blocks
            .iter()
            .flat_map(|b| b.lines.iter())
            .map(|l| {
                // `bbox` and `confidence` are `Option`: absent means the stage
                // did not produce one, which a JS caller cannot represent. Zero
                // for an absent box is honest here because every consumer draws
                // it — a NaN or a whole-image box would both read as a claim.
                let b = l.bbox.unwrap_or(ffai_core::types::BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 0.0,
                    height: 0.0,
                });
                Line {
                    text: l.text.clone(),
                    x: b.x,
                    y: b.y,
                    width: b.width,
                    height: b.height,
                    confidence: l.confidence.unwrap_or(0.0),
                }
            })
            .collect())
    }

    /// The whole page as text, lines joined with newlines and blocks separated
    /// by a blank line — the reading order the engine computed, not the order
    /// the boxes were found in.
    pub fn text(&self, rgba: Vec<u8>, width: u32, height: u32) -> Result<String, JsValue> {
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
        let out = self
            .engine
            .recognize(&image, &OcrOptions::default())
            .map_err(|e| JsValue::from_str(&format!("recognize failed: {e}")))?;
        Ok(out.text())
    }
}

/// Cap the DETECTOR's short side at `px` pixels.
///
/// `0` restores the gated default (1280); `0xFFFF_FFFF` disables the ceiling
/// and restores the 0.9.x policy, in which nothing bounded a large image except
/// the 4000-pixel cap on the LONG side.
///
/// `mobiledet_input` floors the short side at 736 and caps only the LONG side,
/// at 4000 — so a 3000x4000 phone photo reaches the detector at 12 MP, every
/// pixel above the resolution the network works at. Natively that is
/// `FFAI_DET_MAX_SHORT`; a browser has no environment, so it gets this.
///
/// Detection is **86-89 % of the time** on a browser-sized capture and its cost
/// is in the input area, so this is the largest speed knob a caller has — and
/// on wasm32 it is a memory knob too, because the detector input tensor and
/// every activation above it scale with the same area.
#[wasm_bindgen(js_name = setDetMaxShort)]
pub fn set_det_max_short(px: u32) {
    ffai_carmenta::image::set_det_max_short(px as usize);
}

/// Current size of the wasm linear memory, in bytes.
///
/// The 4 GiB cap is a cap on THIS number, and linear memory only ever grows —
/// so a growth curve across repeated reads is the only honest memory
/// instrument on this target. `performance.memory` measures the JS heap and
/// says nothing about ours.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = linearMemoryBytes)]
#[must_use]
pub fn linear_memory_bytes() -> f64 {
    (core::arch::wasm32::memory_size::<0>() * 65536) as f64
}

/// Reserve a rusty_alloc ARENA of `mib` mebibytes, up front.
///
/// On wasm this is not an optimisation, it is the recycling layer. `prim::free`
/// is a no-op (linear memory cannot shrink), and the arena is the ONLY other
/// place a freed 32 MiB segment can go: `segment_free` and `huge_free` both try
/// `arena::chunk_free*` first and fall through to `os::free` — i.e. to nothing —
/// when the address belongs to no arena. rusty_alloc disables its own default
/// arena on wasm (`DEFAULT_ARENA_PAYS = false`), so out of the box every freed
/// segment is lost and linear memory grows without bound.
///
/// **Kept as an instrument; no longer needed as a workaround.** It existed
/// because rusty_alloc 1.1.4 leaked a whole segment per >16 MiB alloc/free
/// cycle on wasm, and an arena was the only free list that could catch them.
/// It was never a good workaround even then: an arena only bounds growth while
/// it has chunks left, and `reserveArena(256)` followed by one A4 read still
/// went 352 MiB -> 2162 MiB, because Carmenta's peak exceeds 256 MiB and the
/// overflow fell straight back to the leaking path. 1.1.5 fixed the leak at
/// the source and 1.1.6 the rounding above it, so the default allocator now
/// recycles without help.
///
/// Returns whether the reservation succeeded.
#[cfg(all(target_arch = "wasm32", feature = "rusty-alloc"))]
#[wasm_bindgen(js_name = reserveArena)]
pub fn reserve_arena(mib: u32) -> bool {
    rusty_alloc::arena::reserve_os_memory_ex(mib as usize * 1024 * 1024, true, false, false).is_ok()
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
