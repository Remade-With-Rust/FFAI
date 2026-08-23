//! Build `inputs_embeds` end-to-end from OUR half, for step 4's output-token
//! gate.
//!
//! ```text
//! cargo run --release -p ffai-argus --example build_inputs_embeds
//! .venv-argus/Scripts/python.exe corpora/refs/check_smolvlm_tokens.py
//! ```
//!
//! # What this composes, and why it is the real gate
//!
//! The splice test already proves our merge is bit-exact *given the
//! reference's own tensors*. That is a necessary check and a trivial one to
//! pass end-to-end, because zero error in equals zero error out.
//!
//! The question step 4 actually asks is different: our vision tower carries
//! ~1e-4 of float error against the reference (step 3, 104.8 dB). **Greedy
//! decoding is an argmax.** Small differences can flip a token, and a flipped
//! token changes the answer — which is exactly the "plausible but degraded"
//! failure the plan warns about, arriving through accumulated numerics rather
//! than through a structural mistake.
//!
//! So this runs OUR tower over all 17 tiles, OUR connector, and OUR assembly,
//! and hands the result to the reference's own decoder. If the output tokens
//! match, our half of the pipeline is good enough that the decoder cannot tell
//! the difference — which is the only standard that matters.
//!
//! Preprocessing is still isolated out: the tiles come from the reference's
//! `pixel_values`, so a mismatch here cannot be a resize or a normalisation.

use candle_core::{DType, Device, IndexOp, Tensor};

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect()
}

fn read_i64(path: &std::path::Path) -> Vec<i64> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = root.join(".oracle/smolvlm-prompt");
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("prompt.json"))?)?;

    // The checkpoint, from the HF cache.
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))?;
    let snaps = std::path::Path::new(&home)
        .join(".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots");
    let snap = std::fs::read_dir(&snaps)?
        .flatten()
        .next()
        .ok_or("no snapshot in the HF cache")?
        .path();
    let device = Device::Cpu;
    let model = ffai_argus::vision::load(
        &snap.join("model.safetensors"),
        &std::fs::read_to_string(snap.join("config.json"))?,
        &device,
    )?;

    // The reference's own pixel_values: (1, tiles, 3, H, W).
    let pv_shape: Vec<usize> = doc["embeds"]["pixel_values"]["shape"]
        .as_array()
        .ok_or("pixel_values shape")?
        .iter()
        .map(|x| x.as_u64().expect("dim") as usize)
        .collect();
    let (tiles, c, h, w) = (pv_shape[1], pv_shape[2], pv_shape[3], pv_shape[4]);
    let pv = Tensor::from_vec(
        read_f32(&dir.join("pixel_values.f32")),
        (tiles, c, h, w),
        &device,
    )?;
    eprintln!("running our tower over {tiles} tiles of {c}x{h}x{w} ...");

    // Tile by tile, exactly as the reference does — the tower has no batch
    // interaction, and one-at-a-time keeps peak memory to a single tile.
    let mut blocks = Vec::with_capacity(tiles);
    for t in 0..tiles {
        let tile = pv.i(t)?.unsqueeze(0)?;
        blocks.push(model.forward(&tile)?.squeeze(0)?);
    }
    let image_hidden = Tensor::stack(&blocks, 0)?;
    eprintln!("  image_hidden {:?}", image_hidden.dims());

    // Text embeddings come from the reference dump: this gate is about the
    // VISION half and the assembly. The token embedding table is a lookup with
    // no arithmetic, so porting it would add nothing to test.
    let ts: Vec<usize> = doc["embeds"]["text_embeds"]["shape"]
        .as_array()
        .ok_or("text_embeds shape")?
        .iter()
        .map(|x| x.as_u64().expect("dim") as usize)
        .collect();
    let text = Tensor::from_vec(
        read_f32(&dir.join("text_embeds.f32")),
        (ts[0], ts[1], ts[2]),
        &device,
    )?;

    let ids = read_i64(&dir.join("input_ids.i64"));
    let image_token_id = doc["image_token"]["id"].as_i64().ok_or("image token id")?;
    let merged =
        ffai_argus::prompt::merge_image_embeddings(&text, &image_hidden, &ids, image_token_id)?;
    eprintln!("  inputs_embeds {:?}", merged.dims());

    let flat = merged.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let out = dir.join("ours_inputs_embeds.f32");
    let mut bytes = Vec::with_capacity(flat.len() * 4);
    for v in &flat {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&out, bytes)?;
    eprintln!("wrote {}", out.display());

    // Report the drift against the reference's merged tensor, so the decode
    // comparison that follows has context for whatever it finds.
    let want = read_f32(&dir.join("inputs_embeds.f32"));
    let max_abs = flat
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max_abs vs reference inputs_embeds: {max_abs:.3e}");
    Ok(())
}
