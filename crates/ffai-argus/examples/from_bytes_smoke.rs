//! Does `SmolVlm::from_bytes` produce the same caption as the model the
//! manifest path builds — and the same one a wasm module produces?
//!
//! `from_bytes` is the constructor `ffai-argus-wasm` uses, because a browser
//! has no filesystem and no mmap. A constructor exercised only on wasm is one
//! nobody can debug, so this runs it NATIVELY against the same three artefacts
//! and the same pixels the browser gets, and prints the caption for comparison
//! against the wasm module's.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example from_bytes_smoke -- \
//!     <dir-with-the-three-files> <image.rgba> <width> <height> [prompt]
//! ```
//!
//! `.rgba` is raw `width * height * 4` bytes — what `getImageData` hands out,
//! so both sides genuinely receive identical input.

use ffai_argus::engine::{ArgusBytes, SmolVlm};
use ffai_core::engine::{VlmEngine, VlmOptions};
use ffai_core::types::{ImageBuffer, PixelFormat};

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: from_bytes_smoke <model-dir> <image.rgba> <w> <h> [prompt]";
    let dir = args.next().expect(usage);
    let rgba_path = args.next().expect(usage);
    let width: u32 = args.next().expect(usage).parse().expect("width");
    let height: u32 = args.next().expect(usage).parse().expect("height");
    let prompt = args.next().unwrap_or_default();

    let read = |name: &str| {
        std::fs::read(std::path::Path::new(&dir).join(name))
            .unwrap_or_else(|e| panic!("{dir}/{name}: {e}"))
    };
    let engine = SmolVlm::from_bytes(ArgusBytes {
        weights: read("model.safetensors"),
        config: String::from_utf8(read("config.json")).expect("config.json is not UTF-8"),
        tokenizer: read("tokenizer.json"),
    })
    .expect("from_bytes");

    let data = std::fs::read(&rgba_path).unwrap_or_else(|e| panic!("{rgba_path}: {e}"));
    let want = width as usize * height as usize * 4;
    assert_eq!(data.len(), want, "{rgba_path}: expected {want} bytes of RGBA");
    let image = ImageBuffer {
        data,
        width,
        height,
        format: PixelFormat::Rgba8,
    };
    let opts = VlmOptions {
        prompt: (!prompt.trim().is_empty()).then_some(prompt),
        max_new_tokens: Some(40),
        ..VlmOptions::default()
    };
    let caption = engine.describe_image(&image, &opts).expect("describe");
    println!("image: {width}x{height}");
    println!("NATIVE: {}", caption.trim());
}
