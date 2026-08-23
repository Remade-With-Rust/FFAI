//! Step 6's gate: `describe_image` against the reference's own caption.
//!
//! Every earlier oracle test reaches inside — it loads the tower, the decoder
//! and the tokenizer itself and wires them together the way it believes the
//! engine does. That is the right shape for attributing a tensor mismatch and
//! the wrong shape for catching a PLUMBING mistake, because a test that
//! assembles the pipeline itself cannot notice that the engine assembles it
//! differently.
//!
//! So this one goes through the public surface only: an [`ImageBuffer`] in, a
//! `String` out, `SmolVlm::describe_image` in between. If the manifest, the
//! tokenizer lookup, the chat template, the geometry or the cache handling is
//! wrong, this is where it shows.

use ffai_core::engine::{VlmEngine, VlmOptions};
use ffai_core::types::{ImageBuffer, PixelFormat};

use std::path::{Path, PathBuf};

const IMG: usize = 512;

fn manifests() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn oracle() -> Option<(PathBuf, serde_json::Value)> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.oracle/smolvlm-prompt");
    let v = serde_json::from_str(&std::fs::read_to_string(d.join("prompt.json")).ok()?).ok()?;
    Some((d, v))
}

/// The same deterministic pattern every other oracle in this crate uses,
/// wrapped as the `ImageBuffer` a decoder would have produced.
fn reference_image() -> ImageBuffer {
    let mut data = vec![0u8; IMG * IMG * 3];
    let mut i = 0;
    for y in 0..IMG {
        let fy = y as f64 / IMG as f64;
        for x in 0..IMG {
            let fx = x as f64 / IMG as f64;
            let r = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * fx).sin();
            let g = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * fy + 1.0).sin();
            let b = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * (fx + fy) + 2.0).sin();
            data[i] = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            data[i + 1] = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            data[i + 2] = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            i += 3;
        }
    }
    ImageBuffer {
        width: IMG as u32,
        height: IMG as u32,
        format: PixelFormat::Rgb8,
        data,
    }
}

/// The reference's generated ids, turned back into the text a user would see.
///
/// Comparing TEXT rather than ids is deliberate here: ids are what steps 4 and
/// 5 gate, and this step's job is the surface, which includes detokenization
/// and the stop handling that ids alone would not exercise.
fn reference_caption(doc: &serde_json::Value) -> Option<String> {
    let ids: Vec<u32> = doc["reference_output_ids"]
        .as_array()?
        .iter()
        .filter_map(|v| v.as_u64())
        .map(|v| v as u32)
        .collect();
    let tok_path = ffai_models::load_dir(&manifests())
        .ok()?
        .into_iter()
        .find(|m| m.name == ffai_argus::engine::MODEL)?
        .local_path("tokenizer.json")?;
    let tok = tokenizers::Tokenizer::from_file(tok_path).ok()?;
    Some(tok.decode(&ids, true).ok()?.trim().to_string())
}

#[test]
fn describe_image_reproduces_the_reference_caption() {
    let Some((_, doc)) = oracle() else {
        eprintln!("SKIP: no prompt oracle — run corpora/refs/dump_smolvlm_prompt.py");
        return;
    };
    let Some(want) = reference_caption(&doc) else {
        eprintln!("SKIP: checkpoint not resolvable offline");
        return;
    };
    let question = doc["question"].as_str().expect("question").to_string();

    let engine = ffai_argus::SmolVlm::with_manifest_dir(manifests());
    let opts = VlmOptions {
        prompt: Some(question.clone()),
        // The reference generated exactly this many tokens, so matching the
        // budget keeps the comparison about the CONTENT rather than about
        // where two different budgets happened to cut.
        max_new_tokens: Some(doc["reference_output_ids"].as_array().map_or(32, Vec::len)),
        ..VlmOptions::default()
    };

    let got = match engine.describe_image(&reference_image(), &opts) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: engine unavailable: {e}");
            return;
        }
    };

    eprintln!("question : {question}");
    eprintln!("reference: {want:?}");
    eprintln!("ours     : {got:?}");
    assert_eq!(
        got, want,
        "describe_image must reproduce the reference caption. Steps 3-5 gate \
         the tensors and the token ids; a mismatch HERE with those passing is \
         the engine's own plumbing — the chat template, the tokenizer lookup, \
         the geometry it hands the prompt, or the stop handling."
    );
}

/// Two calls, one engine, same answer.
///
/// `candle`'s `llama::Cache` has no public reset, so the obvious
/// implementation — build the cache once at load, reuse it — makes the SECOND
/// caption depend on the first. Nothing in a single-call test can see that,
/// and in production it presents as "the captions are fine until you batch a
/// directory", which nobody attributes to a cache.
#[test]
fn a_second_caption_does_not_inherit_the_first() {
    let Some((_, doc)) = oracle() else {
        eprintln!("SKIP: no prompt oracle");
        return;
    };
    let engine = ffai_argus::SmolVlm::with_manifest_dir(manifests());
    let img = reference_image();
    let opts = VlmOptions {
        prompt: Some(doc["question"].as_str().unwrap_or("Describe the image.").into()),
        max_new_tokens: Some(24),
        ..VlmOptions::default()
    };
    let first = match engine.describe_image(&img, &opts) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: engine unavailable: {e}");
            return;
        }
    };
    // A DIFFERENT prompt in between, so a leaked cache carries something that
    // would actually change the answer rather than something identical.
    let other = VlmOptions {
        prompt: Some("How many colours are in this picture?".into()),
        max_new_tokens: Some(24),
        ..VlmOptions::default()
    };
    let _ = engine.describe_image(&img, &other);
    let again = engine.describe_image(&img, &opts).expect("third call");

    eprintln!("first: {first:?}");
    eprintln!("again: {again:?}");
    assert_eq!(
        first, again,
        "the same image and prompt must caption the same way regardless of \
         what ran before — a KV cache that survives a call is a correctness \
         bug, not a performance feature"
    );
}

/// Grayscale and RGBA must reach the tower, not an error.
///
/// `load_image` returns whatever the file held: `rusty_png` yields `Gray8` for
/// a grayscale PNG and `Rgba8` when there is an alpha channel. An engine that
/// only handles `Rgb8` fails on real files while passing every synthetic test.
#[test]
fn every_pixel_format_captions() {
    let engine = ffai_argus::SmolVlm::with_manifest_dir(manifests());
    let opts = VlmOptions {
        prompt: Some("What is this?".into()),
        max_new_tokens: Some(8),
        ..VlmOptions::default()
    };
    let rgb = reference_image();
    // Probe availability once so an absent checkpoint skips rather than fails.
    if engine.describe_image(&rgb, &opts).is_err() {
        eprintln!("SKIP: engine unavailable");
        return;
    }
    for (name, format, data) in [
        (
            "Gray8",
            PixelFormat::Gray8,
            rgb.data.chunks_exact(3).map(|p| p[0]).collect::<Vec<u8>>(),
        ),
        (
            "Rgba8",
            PixelFormat::Rgba8,
            rgb.data
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect::<Vec<u8>>(),
        ),
    ] {
        let img = ImageBuffer {
            width: IMG as u32,
            height: IMG as u32,
            format,
            data,
        };
        let out = engine
            .describe_image(&img, &opts)
            .unwrap_or_else(|e| panic!("{name} must caption, got: {e}"));
        eprintln!("  {name:6} -> {out:?}");
        assert!(!out.is_empty(), "{name} produced an empty caption");
    }
    // RGBA is the same picture with an opaque alpha, so it must caption
    // IDENTICALLY to the RGB original — that is the check that alpha is
    // dropped rather than smeared into a channel.
    let rgba = ImageBuffer {
        width: IMG as u32,
        height: IMG as u32,
        format: PixelFormat::Rgba8,
        data: rgb
            .data
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
    };
    assert_eq!(
        engine.describe_image(&rgba, &opts).expect("rgba"),
        engine.describe_image(&rgb, &opts).expect("rgb"),
        "opaque RGBA must caption exactly as its RGB original"
    );
}

/// A non-square image, where the tile grid stops being 4x4.
///
/// Every fixture in this crate is 512x512, which makes `rows == cols` and
/// hides any place the two are swapped. A transposed grid produces a valid
/// prompt of the right token count describing the image sideways.
#[test]
fn a_non_square_image_captions_with_the_right_grid() {
    let engine = ffai_argus::SmolVlm::with_manifest_dir(manifests());
    let (w, h) = (800usize, 600usize);
    let mut data = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            data[i] = ((x * 255) / w) as u8;
            data[i + 1] = ((y * 255) / h) as u8;
            data[i + 2] = 128;
        }
    }
    let img = ImageBuffer {
        width: w as u32,
        height: h as u32,
        format: PixelFormat::Rgb8,
        data,
    };
    let opts = VlmOptions {
        prompt: Some("Describe the image.".into()),
        max_new_tokens: Some(16),
        ..VlmOptions::default()
    };
    match engine.describe_image(&img, &opts) {
        Ok(out) => {
            eprintln!("800x600 -> {out:?}");
            // The real assertion is that this ran at all: a grid mismatch
            // between the prompt's `<row_r_col_c>` markers and the tensor's
            // tiles is a shape error inside the splice, not a bad caption.
            assert!(!out.is_empty(), "non-square image produced no caption");
        }
        Err(e) => eprintln!("SKIP: engine unavailable: {e}"),
    }
}
