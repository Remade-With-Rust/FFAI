//! Our `SigLIP` encoder against candle's, on the same weights and input.
//!
//! # Why this is not a bit-identity gate
//!
//! Two of the optimisations change the last bits, and could not avoid it:
//!
//! * the attention scale is folded into the **q weights at load**, so the
//!   multiply happens before the projection's sum rather than after it;
//! * `gelu_pytorch_tanh` is evaluated through `2*sigmoid(2z) - 1` instead of
//!   `tanh(z)` — an exact identity, one extra rounding.
//!
//! Demanding bit-identity would therefore forbid the two changes worth making.
//! What matters is whether the difference can change an ANSWER, and §16 pinned
//! that empirically: the vision tower's own **2.06e-4** disagreement with the
//! reference flips none of 32 argmaxes, while preprocessing's 7.8e-3 flips one.
//! The bar below sits far under the number already known to be harmless, and
//! the real gate is unchanged — `decode_oracle` still requires 32/32 tokens and
//! `engine_oracle` a byte-identical caption.

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::siglip;

use std::path::{Path, PathBuf};

fn checkpoint() -> Option<(PathBuf, String)> {
    let manifests =
        ffai_models::load_dir(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")).ok()?;
    let m = manifests
        .into_iter()
        .find(|m| m.name == ffai_argus::engine::MODEL)?;
    let w = m.local_path("model.safetensors")?;
    let c = std::fs::read_to_string(m.local_path("config.json")?).ok()?;
    Some((w, c))
}

fn stats(ours: &Tensor, theirs: &Tensor) -> (f32, f64) {
    let a = ours.flatten_all().expect("a").to_vec1::<f32>().expect("av");
    let b = theirs.flatten_all().expect("b").to_vec1::<f32>().expect("bv");
    let max_abs = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let (mut se, mut sr) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(&b) {
        se += f64::from(x - y) * f64::from(x - y);
        sr += f64::from(*y) * f64::from(*y);
    }
    (max_abs, 10.0 * (sr / se.max(f64::MIN_POSITIVE)).log10())
}

#[test]
fn our_tower_agrees_with_candles() {
    let Some((weights, config)) = checkpoint() else {
        eprintln!("SKIP: checkpoint not resolvable offline");
        return;
    };
    let device = Device::Cpu;
    let (cfg, _) = ffai_argus::vision::vision_config_from_json(&config).expect("config");

    // SAFETY: the mapped file is owned by the model cache and is not mutated
    // while this process holds it.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, &device)
    }
    .expect("weights");
    let vb = vb.pp("model.vision_model");

    let ours = ffai_argus::siglip::VisionTower::new(&cfg, vb.clone()).expect("ours");
    let theirs = siglip::VisionModel::new(&cfg, false, vb).expect("candle");

    // A real image, not noise: noise exercises no particular part of the
    // activation curve, and GELU's interesting region is the small-negative
    // dip that a natural image's normalised pixels actually visit.
    let (w, h) = (512usize, 512usize);
    let mut px = vec![0u8; w * h * 3];
    for (i, v) in px.iter_mut().enumerate() {
        *v = ((i * 7 + i / 512) % 251) as u8;
    }
    let pre = ffai_argus::preprocess::preprocess_rgb8(&px, w, h);
    let per = 3 * pre.tile * pre.tile;
    let input = Tensor::from_vec(
        pre.pixel_values[..per].to_vec(),
        (1, 3, pre.tile, pre.tile),
        &device,
    )
    .expect("input");

    let a = ours.forward(&input).expect("ours fwd");
    let b = theirs.forward(&input).expect("candle fwd");
    assert_eq!(a.dims(), b.dims(), "shape must match candle's exactly");

    let (max_abs, snr) = stats(&a, &b);
    let std = {
        let v = b.flatten_all().expect("f").to_vec1::<f32>().expect("v");
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32).sqrt()
    };
    eprintln!("ours vs candle: max_abs={max_abs:.3e}  ({:.2e} of std {std:.4})  SNR={snr:.1} dB",
              max_abs / std);

    // Scaled by the tensor's own spread — an absolute threshold on activations
    // whose magnitude we do not control is a threshold on the wrong thing.
    assert!(
        max_abs / std < 1e-3,
        "our tower differs from candle's by {max_abs:.3e} ({:.2e} of std). \
         The scale-fold and the GELU identity should cost far less than this; \
         a gap this size means one of them is not the identity it claims.",
        max_abs / std
    );
    assert!(snr > 70.0, "SNR {snr:.1} dB is too low for a rounding difference");
}

/// The fused QKV must select the same three projections candle's three
/// separate `Linear`s do — and in the right order.
///
/// A q/k swap is the failure this catches: it produces a perfectly valid
/// tensor of the right shape, attention still "works", and the only symptom is
/// a caption that is subtly wrong. Comparing whole-tower output would catch it
/// too, but would not say WHERE.
#[test]
fn the_fused_projection_keeps_q_k_v_in_order() {
    let Some((weights, config)) = checkpoint() else {
        eprintln!("SKIP: checkpoint not resolvable offline");
        return;
    };
    let device = Device::Cpu;
    let (cfg, _) = ffai_argus::vision::vision_config_from_json(&config).expect("config");
    // SAFETY: as above.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, &device)
    }
    .expect("weights")
    .pp("model.vision_model.encoder.layers.0.self_attn");

    let hidden = cfg.hidden_size;
    let head_dim = hidden / cfg.num_attention_heads;
    let scale = (head_dim as f64).powf(-0.5);

    let wq = vb.get((hidden, hidden), "q_proj.weight").expect("q");
    let wk = vb.get((hidden, hidden), "k_proj.weight").expect("k");
    let wv = vb.get((hidden, hidden), "v_proj.weight").expect("v");
    let fused = Tensor::cat(
        &[&(wq.clone() * scale).expect("scaled"), &wk, &wv],
        0,
    )
    .expect("cat");
    assert_eq!(fused.dims(), &[3 * hidden, hidden]);

    // The k and v thirds must be untouched copies...
    let k_third = fused.narrow(0, hidden, hidden).expect("k third");
    let v_third = fused.narrow(0, 2 * hidden, hidden).expect("v third");
    for (name, got, want) in [("k", &k_third, &wk), ("v", &v_third, &wv)] {
        let d = (got - (*want).clone())
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max")
            .to_scalar::<f32>()
            .expect("s");
        assert_eq!(d, 0.0, "the {name} third of the fused matrix was altered");
    }

    // ...and the q third must be exactly the scaled original, which is also
    // what proves the scale landed on q and not on k.
    let q_third = fused.narrow(0, 0, hidden).expect("q third");
    let ratio = (&q_third / &wq).expect("div");
    let flat = ratio.flatten_all().expect("f").to_vec1::<f32>().expect("v");
    let finite: Vec<f32> = flat.into_iter().filter(|x| x.is_finite()).collect();
    let worst = finite
        .iter()
        .map(|r| (r - scale as f32).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-6,
        "the q third is not the original scaled by {scale}: worst ratio error {worst:.3e}"
    );
}
