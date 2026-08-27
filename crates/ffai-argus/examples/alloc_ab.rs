//! Does the ALLOCATOR account for prefill's unexplained 15 %?
//!
//! `text_inline_prof` attributes 85 % of prefill to named ops and leaves ~191 ms
//! unaccounted. A prefill allocates a **47 MB** score tensor per layer, thirty
//! times, plus every other intermediate — and 47 MB is above the size where
//! most allocators stop pooling and hand the request to the OS, making each
//! layer a map/unmap pair.
//!
//! It also fixes a measurement/production mismatch: `ffai` declares
//! `rusty_alloc`, while `examples/` binaries declared nothing and therefore ran
//! on the system allocator. Every number this crate has published from an
//! example was taken under an allocator production does not use.
//!
//! Run it twice to compare:
//! ```sh
//! cargo run --release -p ffai-argus --example alloc_ab              # rusty_alloc
//! FFAI_SYSTEM_ALLOC=1 cargo run --release -p ffai-argus --example alloc_ab
//! ```
//! The binary cannot switch allocator at run time — that is a compile-time
//! choice — so the env var only labels the run; build the two arms separately
//! if you need both in one sitting.
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::time::Instant;

#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))?;
    let snap = std::fs::read_dir(
        std::path::Path::new(&home)
            .join(".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots"),
    )?
    .flatten()
    .next()
    .ok_or("no snapshot")?
    .path();
    let weights = snap.join("model.safetensors");
    let cj = std::fs::read_to_string(snap.join("config.json"))?;
    let d = Device::Cpu;
    let v: serde_json::Value = serde_json::from_str(&cj)?;
    let tc = v.get("text_config").unwrap_or(&v);
    let g = |k: &str, dv: u64| tc.get(k).and_then(serde_json::Value::as_u64).unwrap_or(dv);
    let gf = |k: &str, dv: f64| tc.get(k).and_then(serde_json::Value::as_f64).unwrap_or(dv);
    let heads = g("num_attention_heads", 9) as usize;
    let hidden = g("hidden_size", 576) as usize;
    let cfg = ffai_argus::text::Cfg {
        layers: g("num_hidden_layers", 30) as usize,
        hidden,
        heads,
        kv_heads: g("num_key_value_heads", 3) as usize,
        head_dim: hidden / heads,
        inter: g("intermediate_size", 1536) as usize,
        eps: gf("rms_norm_eps", 1e-5),
        rope_theta: gf("rope_theta", 100_000.0) as f32,
        max_pos: g("max_position_embeddings", 8192) as usize,
    };
    // SAFETY: mapped file owned by the model cache, not mutated here.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, &d)?
    };
    let mut tower = ffai_argus::text::TextTower::load(&vb, cfg, &d)?;

    const SEQ: usize = 1142;
    let x = Tensor::rand(-1.0f32, 1.0, (1, SEQ, hidden), &d)?;
    for _ in 0..2 {
        tower.reset();
        let _ = tower.forward(&x, 0)?;
    }
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        tower.reset();
        let t = Instant::now();
        std::hint::black_box(tower.forward(&x, 0)?.dims());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "allocator: {}\n  prefill seq {SEQ}, best of 5: {best:.1} ms",
        if std::env::var("FFAI_SYSTEM_ALLOC").is_ok() { "(label) system" } else { "rusty_alloc" }
    );
    Ok(())
}
