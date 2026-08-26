//! Our text tower against candle's: same weights, same input, both timed.
//!
//! Correctness first — a faster arm computing something else is not an arm.
//! The gate is the logits, because that is what the decode loop argmaxes; a
//! tensor that is "close" can still flip a token, which is the lesson the
//! one-quantisation-level resampler taught this crate (§16).
//!
//! ```sh
//! cargo run --release -p ffai-argus --example text_ab
//! ```
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))?;
    let snaps = std::path::Path::new(&home)
        .join(".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots");
    let snap = std::fs::read_dir(&snaps)?
        .flatten()
        .next()
        .ok_or("no snapshot in the HF cache")?
        .path();
    let weights = snap.join("model.safetensors");
    let config_json = std::fs::read_to_string(snap.join("config.json"))?;
    let device = Device::Cpu;

    let v: serde_json::Value = serde_json::from_str(&config_json)?;
    let t = v.get("text_config").unwrap_or(&v);
    let g = |k: &str, d: u64| t.get(k).and_then(serde_json::Value::as_u64).unwrap_or(d);
    let gf = |k: &str, d: f64| t.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d);
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
        rope_theta: gf("rope_theta", 10000.0) as f32,
        max_pos: 8192,
    };
    println!("{cfg:?}\n");

    // SAFETY: the mapped file is owned by the model cache and not mutated.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, &device)?
    };
    let mut ours = ffai_argus::text::TextTower::load(&vb, cfg, &device)?;
    // The ORACLE arm — candle's tower, forced. `TextDecoder::load` now routes
    // through ours by default, so plain `load` here would compare ours against
    // ours (it did, and reported a max logit delta of exactly 0.000e0).
    let mut theirs =
        ffai_argus::decode::TextDecoder::load_reference(&weights, &config_json, &device)?;

    // Page the 540 MB of mmapped weights in BEFORE timing anything. Without
    // this the first row measures major page faults, which is why an earlier
    // run read candle as SLOWER at seq 64 (2087 ms) than at seq 512 (2811 ms)
    // — an ordering that cannot be true of compute.
    {
        let warm = Tensor::zeros((1, 8, hidden), DType::F32, &device)?;
        for _ in 0..2 {
            ours.reset();
            let _ = ours.forward(&warm, 0)?;
            theirs.reset();
            let _ = theirs.forward_embeds(&warm, 0)?;
        }
    }

    for &seq in &[1usize, 64, 512, 1142] {
        // Deterministic input: a fixed ramp, not randn — a flaky arm is not an
        // arm either.
        let vals: Vec<f32> = (0..seq * hidden)
            .map(|i| ((i % 97) as f32 - 48.0) / 64.0)
            .collect();
        let embeds = Tensor::from_vec(vals, (1, seq, hidden), &device)?;

        ours.reset();
        theirs.reset();
        let a = ours.forward(&embeds, 0)?;
        let b = theirs.forward_embeds(&embeds, 0)?;
        let (av, bv) = (
            a.flatten_all()?.to_vec1::<f32>()?,
            b.flatten_all()?.to_vec1::<f32>()?,
        );
        let worst = av
            .iter()
            .zip(&bv)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let am = |v: &[f32]| {
            (0..v.len())
                .max_by(|&i, &j| v[i].partial_cmp(&v[j]).expect("cmp"))
                .expect("nonempty")
        };
        let same_argmax = am(&av) == am(&bv);

        let mut t_ours = f64::INFINITY;
        let mut t_theirs = f64::INFINITY;
        for _ in 0..3 {
            ours.reset();
            let s = Instant::now();
            std::hint::black_box(ours.forward(&embeds, 0)?.dims());
            t_ours = t_ours.min(s.elapsed().as_secs_f64());

            theirs.reset();
            let s = Instant::now();
            std::hint::black_box(theirs.forward_embeds(&embeds, 0)?.dims());
            t_theirs = t_theirs.min(s.elapsed().as_secs_f64());
        }
        println!(
            "  seq {seq:>5}   candle {:>8.1} ms   ours {:>8.1} ms   {:>5.2}x   \
             max|dlogit| {worst:.3e}   argmax {}",
            t_theirs * 1e3,
            t_ours * 1e3,
            t_theirs / t_ours,
            if same_argmax { "SAME" } else { "**DIFFERENT**" }
        );
    }
    println!("\n  The logit delta is the gate. A speedup with a different argmax is a");
    println!("  regression wearing a win's clothes.");
    Ok(())
}
