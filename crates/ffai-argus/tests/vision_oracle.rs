//! Step 3's gate: the candle vision tower must match the reference runtime's
//! tensors, stage by stage, before anything generates a token.
//!
//! Run the reference dump first (it is gitignored and regenerable):
//!
//! ```text
//! .venv-argus/Scripts/python.exe corpora/refs/dump_smolvlm_vision.py \
//!     --out .oracle/smolvlm-vision
//! ```
//!
//! Without it these tests SKIP rather than fail — a machine that has never run
//! the dumper has not discovered a defect, and a test that fails for a missing
//! fixture teaches people to ignore it. The plan's own words on why this gate
//! exists at all: *"A mismatched tower cannot be debugged later through
//! generated text."*

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};

fn oracle_dir() -> Option<PathBuf> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.oracle/smolvlm-vision");
    d.join("summary.json").exists().then_some(d)
}

/// Read a raw little-endian f32 dump written by the reference dumper.
fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(
        bytes.len().is_multiple_of(4),
        "{} is not a whole number of f32",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Compare two tensors the way a float oracle must be compared.
///
/// # Why this does NOT gate on max relative error
///
/// The first version of this test did, and it failed the tower at
/// `max_rel = 2.69` — on an element whose reference value was **9.0e-7**.
/// Absolute error there was 2.6e-6. The tower was right; the gate was wrong,
/// in exactly the way the previous version of this very comment warned about:
/// *relative error explodes on values legitimately near zero*, and a 1024x768
/// activation has thousands of them.
///
/// So the gate is the pair `codec-optimize` prescribes for float paths:
///
/// * **max absolute error, scaled by the tensor's own spread** — a tolerance
///   of "2e-4" means nothing until you know whether the activation's std is
///   0.19 or 15.2 (both occur in this tower). Judging against the reference's
///   own std makes one threshold meaningful at every stage.
/// * **SNR in dB** — the summary statistic that no single outlier can move,
///   and the one this project already uses for decoder matches.
///
/// Max relative error is still REPORTED, restricted to elements above a
/// magnitude floor, because it is informative. It is not the verdict.
fn compare(name: &str, got: &[f32], want: &[f32], max_rel_of_std: f32, min_snr_db: f64) {
    assert_eq!(
        got.len(),
        want.len(),
        "{name}: element count differs — {} vs {}",
        got.len(),
        want.len()
    );
    let n = want.len() as f64;
    let mean = f64::from(want.iter().sum::<f32>()) / n;
    let var = want
        .iter()
        .map(|&w| (f64::from(w) - mean) * (f64::from(w) - mean))
        .sum::<f64>()
        / n;
    let std = var.sqrt().max(f64::MIN_POSITIVE);
    // Only elements meaningfully above the noise get a relative reading.
    let floor = (std * 1e-2) as f32;

    let (mut max_abs, mut max_rel, mut at) = (0.0f32, 0.0f32, 0usize);
    let (mut sum_sq_err, mut sum_sq_ref) = (0.0f64, 0.0f64);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let abs = (g - w).abs();
        sum_sq_err += f64::from(abs) * f64::from(abs);
        sum_sq_ref += f64::from(w) * f64::from(w);
        if abs > max_abs {
            max_abs = abs;
            at = i;
        }
        if w.abs() > floor {
            let rel = abs / w.abs();
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    let snr = 10.0 * (sum_sq_ref / sum_sq_err.max(f64::MIN_POSITIVE)).log10();
    let abs_of_std = f64::from(max_abs) / std;
    eprintln!(
        "  {name:12} n={:<7} std={std:.4}  max_abs={max_abs:.3e} ({abs_of_std:.2e} of std)           max_rel|>1%std|={max_rel:.3e}  SNR={snr:.1} dB",
        got.len()
    );
    assert!(
        (abs_of_std as f32) <= max_rel_of_std,
        "{name}: worst absolute error is {abs_of_std:.3e} of the tensor's own std          (limit {max_rel_of_std:.1e}); at index {at} got {} want {}",
        got[at],
        want[at]
    );
    assert!(
        snr >= min_snr_db,
        "{name}: SNR {snr:.1} dB is below the {min_snr_db:.0} dB floor — that is a          structural difference, not float reassociation"
    );
}

/// Locate the checkpoint in the `HuggingFace` cache, if it has been fetched.
fn checkpoint() -> Option<(PathBuf, String)> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()?;
    let root = Path::new(&home)
        .join(".cache/huggingface/hub")
        .join("models--HuggingFaceTB--SmolVLM-256M-Instruct")
        .join("snapshots");
    let snap = std::fs::read_dir(root).ok()?.flatten().next()?.path();
    let weights = snap.join("model.safetensors");
    let cfg = std::fs::read_to_string(snap.join("config.json")).ok()?;
    weights.exists().then_some((weights, cfg))
}

#[test]
fn the_candle_tower_matches_the_reference_tensors() {
    let Some(dir) = oracle_dir() else {
        eprintln!("SKIP: no oracle dump — run corpora/refs/dump_smolvlm_vision.py");
        return;
    };
    let Some((weights, config_json)) = checkpoint() else {
        eprintln!("SKIP: SmolVLM-256M-Instruct not in the HF cache");
        return;
    };

    let device = Device::Cpu;
    let model = ffai_argus::vision::load(&weights, &config_json, &device)
        .expect("the vision tower should load from the checkpoint");

    // Feed the REFERENCE's own pixel_values, not our own preprocessing.
    // Step 3 gates the tower; mixing in a resize/normalise would test two
    // bricks at once and a mismatch could not be localised to either.
    let pixels = read_f32(&dir.join("pixel_values_tile0.f32"));
    let input = Tensor::from_vec(pixels, (1, 3, 512, 512), &device)
        .expect("input tensor")
        .to_dtype(DType::F32)
        .expect("f32");

    eprintln!("stage comparison (candle vs transformers, same input):");

    // The tower. Both sides are f32 but differ in matmul order and in whether
    // attention is fused, so bit equality is not available — the float gate in
    // `codec-optimize` applies. The thresholds are deliberately loose against
    // reassociation and brutally tight against structure: a transposed axis, a
    // dropped position embedding or a wrong activation moves SNR by tens of
    // dB, not by fractions.
    let got = model.tower(&input).expect("tower forward");
    let want = read_f32(&dir.join("vision_out.f32"));
    let flat = got
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("to_vec1");
    compare("vision_out", &flat, &want, 1e-3, 80.0);

    // The connector: pixel-shuffle + one matmul. Checked separately because a
    // correct tower feeding a wrong shuffle is exactly the bug that would
    // otherwise surface as "the captions are a bit odd".
    let got_c = model.connect(&got).expect("connector forward");
    let want_c = read_f32(&dir.join("connector.f32"));
    let flat_c = got_c
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("to_vec1");
    compare("connector", &flat_c, &want_c, 1e-3, 80.0);
}

/// The shapes are a cheaper, stronger-typed check than the values, and they
/// run even when the full dump is absent — `summary.json` is committed.
#[test]
fn the_oracle_summary_describes_the_shapes_we_expect() {
    let Some(dir) = oracle_dir() else {
        eprintln!("SKIP: no oracle dump");
        return;
    };
    let text = std::fs::read_to_string(dir.join("summary.json")).expect("summary");
    let v: serde_json::Value = serde_json::from_str(&text).expect("json");

    // 17 tiles for one image is the fact that sizes the whole port: the tower
    // runs 17 times per image, so it — not the decoder — is where the speed
    // gate is decided.
    assert_eq!(v["tiles"].as_u64(), Some(17), "SmolVLM tiles one image into 17");

    let stages = &v["stages"];
    assert_eq!(
        stages["vision_out"]["shape"].as_array().map(Vec::len),
        Some(3)
    );
    let vo: Vec<u64> = stages["vision_out"]["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap())
        .collect();
    assert_eq!(vo, vec![1, 1024, 768], "32x32 patches of 768");

    let co: Vec<u64> = stages["connector"]["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap())
        .collect();
    // 1024 / 4^2 = 64 tokens, projected to the text hidden size 576.
    assert_eq!(vo[1] / 16, co[1], "scale_factor 4 gives a 16x token reduction");
    assert_eq!(co, vec![1, 64, 576]);

    // `last_hidden_state` IS the post-LN output — one fewer stage for the port
    // to get wrong than the stage list suggests.
    assert_eq!(
        stages["post_layernorm"]["sha256_f32le"], stages["vision_out"]["sha256_f32le"],
        "post_layernorm and vision_out should be the same tensor"
    );
}
