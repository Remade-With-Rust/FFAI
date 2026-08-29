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
    println!("image: {width}x{height}");
    // Warm: first call pays lazy init and first-touch, which is not what a
    // per-image cost means.
    let _ = engine.describe_image_unsplit(&image, &opts);

    let t = std::time::Instant::now();
    let fast = engine.describe_image_unsplit(&image, &opts).expect("unsplit");
    let t_fast = t.elapsed().as_secs_f64();
    println!("NATIVE describe_image_unsplit (1 tile):  {t_fast:8.3} s");
    println!("   {}", fast.trim());

    // The COST MODEL. Deterministic — same input, same number, on any machine
    // under any load — which is what makes it admissible on a box that swings
    // +/-12% (codec-measurement 14b, written on this very crate). A win here is
    // a counter that went DOWN; the clock is confirmation, never the verdict.
    // NOTE: `start()` returns the PREVIOUS state, so `if start()` skips the
    // first probe entirely — which is how the unsplit row silently went missing
    // the first time this was run.
    ffai_core::cost::start();
    let _ = engine.describe_image_unsplit(&image, &opts);
    println!("{}", ffai_core::cost::stop().report("unsplit(1 tile)"));

    // Per-op inside the tower (FFAI_VIS_PROFILE=1). The tower is ~96% of the
    // whole caption, so this is the only ranking that matters.
    let ops = ffai_argus::siglip::prof::take();
    if !ops.is_empty() {
        let tot: f64 = ops.iter().map(|(_, ms)| ms).sum();
        println!("
VISION TOWER PER-OP (unsplit, 1 tile) — total {tot:.0} ms");
        for (name, ms) in ops.iter().take(14) {
            println!("  {name:<28} {ms:9.1} ms  {:5.1} %", 100.0 * ms / tot);
        }
    }

    ffai_core::cost::start();
    let _ = engine.describe_image(&image, &opts);
    println!("{}", ffai_core::cost::stop().report("split(17 tiles)"));

    // Where the time goes inside the tower, single-threaded so the shares
    // match what a browser sees.
    let (_, tr) = engine.describe_image_traced(&image, &opts).expect("traced");
    let tower: f64 = tr.tower_per_tile_ms.iter().sum();
    let steps: f64 = tr.step_ms.iter().sum();
    let total = tr.preprocess_ms + tower + tr.assemble_ms + tr.prefill_ms + steps + tr.detokenize_ms;
    println!("
STAGE SPLIT (17-tile path, {} tiles):", tr.tiles);
    for (name, ms) in [
        ("preprocess", tr.preprocess_ms),
        ("vision tower", tower),
        ("assemble", tr.assemble_ms),
        ("prefill", tr.prefill_ms),
        ("decode steps", steps),
        ("detokenize", tr.detokenize_ms),
    ] {
        println!("  {name:14} {ms:9.1} ms  {:5.1} %", 100.0 * ms / total);
    }
    println!("  {:14} {total:9.1} ms", "TOTAL");
    if !tr.tower_per_tile_ms.is_empty() {
        let per = tower / tr.tower_per_tile_ms.len() as f64;
        println!("  per tile      {per:9.1} ms");
    }
}
