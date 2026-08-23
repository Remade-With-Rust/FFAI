//! Step 5's gate: greedy decode matches reference greedy decode.
//!
//! ```text
//! .venv-argus/Scripts/python.exe corpora/refs/dump_smolvlm_prompt.py \
//!     --out .oracle/smolvlm-prompt --embeds
//! ```
//!
//! Two tests, and the split is the point:
//!
//! * **The decoder alone**, fed the REFERENCE's `inputs_embeds`. If this fails,
//!   the loop, the KV cache, the `RoPE` offset or the argmax is wrong — and
//!   nothing upstream can be blamed.
//! * **The whole pipeline**, our tower + connector + assembly + decoder. This
//!   is the one that answers "would Argus produce this caption", and it can
//!   only be interpreted once the first passes.
//!
//! Running only the composed test would be the mistake `codec-bringup-decoder`
//! warns about: one failure, four candidate causes.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};

fn oracle() -> Option<(PathBuf, serde_json::Value)> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.oracle/smolvlm-prompt");
    let v = serde_json::from_str(&std::fs::read_to_string(d.join("prompt.json")).ok()?).ok()?;
    Some((d, v))
}

fn checkpoint() -> Option<(PathBuf, String)> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()?;
    let snaps = Path::new(&home)
        .join(".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots");
    let snap = std::fs::read_dir(snaps).ok()?.flatten().next()?.path();
    let w = snap.join("model.safetensors");
    let c = std::fs::read_to_string(snap.join("config.json")).ok()?;
    w.exists().then_some((w, c))
}

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect()
}

fn shape_of(doc: &serde_json::Value, name: &str) -> Vec<usize> {
    doc["embeds"][name]["shape"]
        .as_array()
        .expect("shape")
        .iter()
        .map(|x| x.as_u64().expect("dim") as usize)
        .collect()
}

fn want_tokens(doc: &serde_json::Value) -> Vec<u32> {
    doc["reference_output_ids"]
        .as_array()
        .expect("reference_output_ids")
        .iter()
        .map(|x| x.as_u64().expect("id") as u32)
        .collect()
}

/// Compare, and say WHERE rather than merely that.
fn compare_tokens(got: &[u32], want: &[u32], what: &str) {
    if got == want {
        eprintln!("  {what}: {}/{} tokens identical", got.len(), want.len());
        return;
    }
    let first = got
        .iter()
        .zip(want)
        .position(|(a, b)| a != b)
        .unwrap_or(got.len().min(want.len()));
    let same = got.iter().zip(want).filter(|(a, b)| a == b).count();
    panic!(
        "{what}: {same}/{} tokens match, first divergence at step {first}\n  \
         ours      : {:?}\n  reference : {:?}\n\n  \
         A divergence at step 0 is structural — the prefill, the RoPE offset or \
         the argmax. A LATE divergence is greedy decode amplifying float error, \
         which is a tolerance question rather than a correctness one.",
        want.len(),
        &got[first.saturating_sub(2)..(first + 4).min(got.len())],
        &want[first.saturating_sub(2)..(first + 4).min(want.len())],
    );
}

/// The decoder in isolation: reference embeddings in, reference tokens out.
#[test]
fn our_decode_loop_matches_the_reference_on_its_own_embeddings() {
    let Some((dir, doc)) = oracle() else {
        eprintln!("SKIP: no prompt oracle");
        return;
    };
    if !dir.join("inputs_embeds.f32").exists() {
        eprintln!("SKIP: dump lacks --embeds");
        return;
    }
    let Some((weights, cfg)) = checkpoint() else {
        eprintln!("SKIP: checkpoint not in the HF cache");
        return;
    };

    let device = Device::Cpu;
    let mut dec = ffai_argus::decode::TextDecoder::load(&weights, &cfg, &device)
        .expect("text decoder should load");

    let s = shape_of(&doc, "inputs_embeds");
    let embeds = Tensor::from_vec(read_f32(&dir.join("inputs_embeds.f32")), (s[0], s[1], s[2]), &device)
        .expect("embeds")
        .to_dtype(DType::F32)
        .expect("f32");

    let want = want_tokens(&doc);
    // No stop ids: the reference ran to its token budget, so ours must too.
    // A stop firing here would itself be a difference worth failing on.
    let got = dec
        .generate_greedy(&embeds, want.len(), &[])
        .expect("greedy decode");

    eprintln!("decoder-only (reference embeddings in):");
    compare_tokens(&got, &want, "decode");
}

/// The whole pipeline: our tower, our connector, our assembly, our decoder.
#[test]
fn our_whole_pipeline_reproduces_the_reference_tokens() {
    let Some((dir, doc)) = oracle() else {
        eprintln!("SKIP: no prompt oracle");
        return;
    };
    if !dir.join("pixel_values.f32").exists() {
        eprintln!("SKIP: dump lacks pixel_values");
        return;
    }
    let Some((weights, cfg)) = checkpoint() else {
        eprintln!("SKIP: checkpoint not in the HF cache");
        return;
    };

    let device = Device::Cpu;
    let vision = ffai_argus::vision::load(&weights, &cfg, &device).expect("vision");
    let mut dec =
        ffai_argus::decode::TextDecoder::load(&weights, &cfg, &device).expect("decoder");

    // Preprocessing stays isolated out: tiles come from the reference, so a
    // failure here cannot be a resize or a normalisation.
    let pv = shape_of(&doc, "pixel_values");
    let (tiles, c, h, w) = (pv[1], pv[2], pv[3], pv[4]);
    let pixels = Tensor::from_vec(read_f32(&dir.join("pixel_values.f32")), (tiles, c, h, w), &device)
        .expect("pixel values");

    let mut blocks = Vec::with_capacity(tiles);
    for t in 0..tiles {
        let tile = pixels
            .get(t)
            .expect("tile")
            .unsqueeze(0)
            .expect("batch dim");
        blocks.push(vision.forward(&tile).expect("tower+connector").squeeze(0).expect("squeeze"));
    }
    let image_hidden = Tensor::stack(&blocks, 0).expect("stack");

    // Text embeddings from OUR tower's own table — the last piece that was
    // previously taken from the dump.
    let ids: Vec<i64> = std::fs::read(dir.join("input_ids.i64"))
        .expect("ids")
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")))
        .collect();
    let id_u32: Vec<u32> = ids.iter().map(|&t| t as u32).collect();
    let id_tensor = Tensor::from_vec(id_u32, (1, ids.len()), &device).expect("id tensor");
    let text_embeds = dec.embed(&id_tensor).expect("embed");

    let image_token_id = doc["image_token"]["id"].as_i64().expect("image token id");
    let merged =
        ffai_argus::prompt::merge_image_embeddings(&text_embeds, &image_hidden, &ids, image_token_id)
            .expect("splice");

    let want = want_tokens(&doc);
    let got = dec.generate_greedy(&merged, want.len(), &[]).expect("greedy");

    eprintln!("whole pipeline (our tower + connector + assembly + decoder):");
    compare_tokens(&got, &want, "pipeline");
}

/// **The full content path**: formula image -> our preprocessing -> our tower
/// -> our assembly -> our decoder -> tokens.
///
/// Every earlier gate fed the reference's `pixel_values` in, isolating
/// preprocessing out so failures could be attributed. This one closes the loop:
/// nothing of the reference's is used except the answer to compare against.
///
/// # What it took to get here
///
/// The first version of this test produced **8/32**, and the cause was the
/// resampler: a floating-point Lanczos that matched PIL to one quantisation
/// level everywhere. That residual — 7.8e-3, about 40x the 2.06e-4 the vision
/// tower carries — flipped a token at step 5. `preprocess.rs` now implements
/// PIL's FIXED-POINT path (i32 coefficients at `1<<22`, integer accumulation,
/// a `u8` intermediate between the two passes) and is bit-identical to PIL in
/// both directions; `resize_oracle.rs` gates exactly that.
///
/// The reference's own `pixel_values` still differ from PIL's by ~20 pixels
/// per border tile out of 786,432 — ULP-boundary ties between two independent
/// fixed-point implementations, since `AutoProcessor` defaults to the
/// torchvision-backed fast processor. That is a difference between two
/// REFERENCES, not a defect in ours, and this test is what decides whether it
/// matters: token equality, not tensor distance.
#[test]
fn the_whole_content_path_reproduces_the_reference_tokens() {
    let Some((dir, doc)) = oracle() else {
        eprintln!("SKIP: no prompt oracle");
        return;
    };
    let Some((weights, cfg)) = checkpoint() else {
        eprintln!("SKIP: checkpoint not in the HF cache");
        return;
    };

    // The same formula image the dumper used — generated here, not loaded, so
    // a drift between the two fails loudly instead of comparing an image
    // against itself.
    const N: usize = 512;
    let mut px = vec![0u8; N * N * 3];
    let mut i = 0;
    for y in 0..N {
        let fy = y as f64 / N as f64;
        for x in 0..N {
            let fx = x as f64 / N as f64;
            let r = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * fx).sin();
            let g = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * fy + 1.0).sin();
            let b = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * (fx + fy) + 2.0).sin();
            px[i] = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            px[i + 1] = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            px[i + 2] = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            i += 3;
        }
    }

    // OUR preprocessing — the same call `describe_image` makes.
    let pre = ffai_argus::preprocess::preprocess_rgb8(&px, N, N);
    let per = 3 * pre.tile * pre.tile;

    let device = Device::Cpu;
    let vision = ffai_argus::vision::load(&weights, &cfg, &device).expect("vision");
    let mut dec = ffai_argus::decode::TextDecoder::load(&weights, &cfg, &device).expect("decoder");

    let mut blocks = Vec::with_capacity(pre.tiles);
    for t in 0..pre.tiles {
        let px = pre.pixel_values[t * per..(t + 1) * per].to_vec();
        let tensor = Tensor::from_vec(px, (1, 3, pre.tile, pre.tile), &device).expect("tile");
        blocks.push(vision.forward(&tensor).expect("tower").squeeze(0).expect("sq"));
    }
    let image_hidden = Tensor::stack(&blocks, 0).expect("stack");
    assert_eq!(image_hidden.dims()[0], pre.rows * pre.cols + 1, "17 blocks");

    let ids: Vec<i64> = std::fs::read(dir.join("input_ids.i64"))
        .expect("ids")
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")))
        .collect();
    let id_u32: Vec<u32> = ids.iter().map(|&t| t as u32).collect();
    let id_tensor = Tensor::from_vec(id_u32, (1, ids.len()), &device).expect("ids");
    let text_embeds = dec.embed(&id_tensor).expect("embed");
    let image_token_id = doc["image_token"]["id"].as_i64().expect("image token id");
    let merged =
        ffai_argus::prompt::merge_image_embeddings(&text_embeds, &image_hidden, &ids, image_token_id)
            .expect("splice");

    let want = want_tokens(&doc);
    let got = dec.generate_greedy(&merged, want.len(), &[]).expect("greedy");
    eprintln!("full content path (our preprocessing included):");
    compare_tokens(&got, &want, "content-path");
}
