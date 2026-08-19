//! Per-layer oracles: our candle backbone against the official model.
//!
//! The reference values come from `corpora/refs/fixtures/yolo26{tier}_oracle_digest.json`,
//! which is **tracked** — 256 exact values per tensor at fixed deterministic
//! indices, plus shape and mean/std/min/max, plus the complete `[300, 6]`
//! selection. That is a real gate rather than a placeholder: a wiring error,
//! a transposed axis, a wrong hidden width or a missing activation moves all
//! 256 values, so the test fails loudly. The full activation dumps stay out
//! of the repo (see `.gitignore`), and this design is the resolution of the
//! two Carmenta fixture lessons pulling in opposite directions — fixtures
//! must ship, and checkpoints must not.
//!
//! **One tolerance here is not a numeric bound but a statement about what is
//! defined.** The final top-k's ROW ORDER is only meaningful where scores
//! are separated by more than f32 reassociation noise; below that the
//! reference's own ordering is arbitrary and asserting on it tests luck.
//! See [`FINAL_ORDER_FLOOR`], which is measured rather than picked.
//!
//! Every tier runs independently — the scale derivation has three
//! tier-dependent branches and n and s exercise none of them.
//!
//! Regenerate the digest with `tools/diana_oracle_dump.py --model yolo26X`;
//! produce the weights this test loads with `tools/diana_convert.py`.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use ffai_diana::backbone::Backbone;
use ffai_diana::head::Head;
use ffai_diana::neck::Neck;

const IMGSZ: usize = 640;

/// Absolute floor, in activation units. The reference is f32 and we
/// accumulate in f32 in a different order, so exact equality is not the
/// right ask; 1e-4 on activations of order 1–10 is tight enough that any
/// structural error blows through it by orders of magnitude.
const TOL_ABS: f32 = 1e-4;

/// Relative band, applied against the tensor's own magnitude.
///
/// **Both bounds exist because the tensors are not all in the same units.**
/// Layer activations sit around 1–10, where an absolute 1e-4 is a tight
/// test. `decoded` carries box coordinates in PIXELS — up to ~640 — where
/// the same absolute number silently becomes a 1.5e-7 relative demand that
/// f32 reassociation alone cannot meet (measured: 1.98e-4 absolute, i.e.
/// 3e-7 relative, on an otherwise exact decode). A pure relative band has
/// the mirror problem near zero. This is the same "relative band OR
/// absolute floor" shape `ffai-bench`'s quality gate settled on, for the
/// same reason.
const TOL_REL: f32 = 1e-5;

fn repo_root() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR is crates/ffai-diana.
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/ffai-diana has two ancestors")
        .to_path_buf()
}

/// The fixture input, from the same closed form `tools/diana_oracle_dump.py`
/// uses. Computed in f64 then narrowed to f32, matching numpy's default
/// promotion in the reference, so both sides start from identical bytes.
fn fixture_input(device: &Device) -> Tensor {
    let n = IMGSZ;
    let mut data = vec![0f32; 3 * n * n];
    for row in 0..n {
        let y = row as f64 / n as f64;
        for col in 0..n {
            let x = col as f64 / n as f64;
            let r = (3.0 * x).sin() * (2.0 * y).cos();
            let r = r.abs();
            let g = (((x * 7.0) % 1.0) + ((y * 11.0) % 1.0)) / 2.0;
            let checker = ((x * 8.0).floor() + (y * 8.0).floor()) % 2.0;
            let b = if checker == 0.0 { 1.0 } else { 0.25 };
            let px = row * n + col;
            data[px] = r as f32;
            data[n * n + px] = g as f32;
            data[2 * n * n + px] = b as f32;
        }
    }
    Tensor::from_vec(data, (1, 3, n, n), device).expect("fixture tensor")
}

/// The tracked letterboxed canvas, as the network's input tensor.
///
/// `/255` and HWC->CHW, matching `tools/diana_oracle_dump.py` exactly. No
/// resize and no padding happen here — the canvas is already 640x640, which
/// is the entire point of shipping it rather than the source JPEG.
fn photo_tensor(path: &std::path::Path, device: &Device) -> Tensor {
    let img = ffai_media::load_image(path).unwrap_or_else(|e| {
        panic!(
            "failed to read the tracked photo fixture {}: {e}. It is TRACKED — regenerate \
             with tools/diana_oracle_dump.py if it is somehow absent",
            path.display()
        )
    });
    let (w, h) = (img.width as usize, img.height as usize);
    assert_eq!((w, h), (IMGSZ, IMGSZ), "photo fixture is a 640x640 canvas");
    let mut data = vec![0f32; 3 * w * h];
    for (px, chunk) in img.data.chunks_exact(3).enumerate() {
        for c in 0..3 {
            data[c * w * h + px] = f32::from(chunk[c]) / 255.0;
        }
    }
    Tensor::from_vec(data, (1, 3, h, w), device).expect("photo tensor")
}

struct Digest(serde_json::Value);

impl Digest {
    fn load(root: &std::path::Path, tier: &str) -> Option<Self> {
        let p = root.join(format!("corpora/refs/fixtures/yolo26{tier}_oracle_digest.json"));
        let text = std::fs::read_to_string(p).ok()?;
        serde_json::from_str(&text).ok().map(Digest)
    }

    /// The complete `[300, 6]` selection, flattened — not a sample of it.
    fn final_full(&self, fixture: &str) -> Option<Vec<f32>> {
        Some(
            self.0
                .get("fixtures")?
                .get(fixture)?
                .get("final_full")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_f64())
                .map(|v| v as f32)
                .collect(),
        )
    }

    fn tensor(&self, fixture: &str, name: &str) -> Option<DigestTensor> {
        let t = self.0.get("fixtures")?.get(fixture)?.get(name)?;
        let shape = t.get("shape")?.as_array()?.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect();
        let idx = t.get("sample_idx")?.as_array()?.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect();
        let val = t.get("sample_val")?.as_array()?.iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect();
        // The tensor's own magnitude, from the reference's recorded extremes
        // — the scale the relative band is measured against.
        let lo = t.get("min")?.as_f64()? as f32;
        let hi = t.get("max")?.as_f64()? as f32;
        let scale = lo.abs().max(hi.abs()).max(1.0);
        Some(DigestTensor { shape, idx, val, scale })
    }
}

struct DigestTensor {
    shape: Vec<usize>,
    idx: Vec<usize>,
    val: Vec<f32>,
    scale: f32,
}

/// Confidence below which a selected row's POSITION carries no information.
///
/// **This floor is measured, not chosen for comfort.** The two-stage top-k
/// ranks 8400x80 candidates down to 300. Perturbing the reference's own
/// tensors by exactly the f32 divergence we measure against it (1.1e-4 in
/// logit space, 4.8e-5 on box coordinates) and re-running its own selection
/// moves the result by up to 5.8e2 pixels and flips classes on 10 rows —
/// on the n tier, which "passed" the old positional check. It passed by
/// luck. Above this floor the same perturbation moves nothing: box delta
/// 4.8e-5, zero class mismatches, on every tier and both fixtures.
///
/// So the row order below the floor is genuinely undefined, and a test that
/// asserts on it is a coin flip that will eventually land badly on a
/// rebuild that changes accumulation order. What IS defined everywhere is
/// the sorted confidence SEQUENCE — swapping two near-tied rows leaves it
/// unchanged — so that is checked for all 300 rows and the positional
/// box/class identity only above the floor.
const FINAL_ORDER_FLOOR: f32 = 0.05;

/// Compare our `[300, 6]` selection against the reference's, splitting the
/// tie-robust claim from the order-dependent one. See [`FINAL_ORDER_FLOOR`].
fn check_final(digest: &Digest, fixture: &str, rows: &[ffai_diana::head::DecodedBox]) {
    let want = digest
        .final_full(fixture)
        .unwrap_or_else(|| panic!("digest has no final_full for {fixture} — regenerate with \
                                   tools/diana_oracle_dump.py"));
    assert_eq!(want.len(), 300 * 6, "final_full is [300, 6]");

    // 1. The confidence sequence: every row, tie-robust.
    let mut conf_worst = 0f32;
    for (i, d) in rows.iter().enumerate() {
        let w = want[i * 6 + 4];
        conf_worst = conf_worst.max((d.confidence - w).abs());
    }
    assert!(
        conf_worst <= TOL_ABS,
        "{fixture} final: confidence sequence diverges by {conf_worst:.3e} (> {TOL_ABS:.0e}). \
         This is the tie-ROBUST half of the top-k check, so a failure here is a real \
         selection bug — wrong k, wrong stage order, or a wrong flatten — not a reordering \
         of near-equal scores"
    );

    // 2. Box and class identity, only where the ordering is determined.
    let mut determinate = 0usize;
    let mut box_worst = 0f32;
    let mut class_mismatches = 0usize;
    for (i, d) in rows.iter().enumerate() {
        if want[i * 6 + 4] < FINAL_ORDER_FLOOR {
            continue;
        }
        determinate += 1;
        let ours = [d.x0, d.y0, d.x1, d.y1];
        for (k, &o) in ours.iter().enumerate() {
            box_worst = box_worst.max((o - want[i * 6 + k]).abs());
        }
        if d.class_id as f32 != want[i * 6 + 5] {
            class_mismatches += 1;
        }
    }
    // Box coordinates are in PIXELS (up to 640), so the relative band is
    // what applies here — the same reasoning as TOL_REL's doc comment.
    let allowed = TOL_ABS.max(TOL_REL * 640.0);
    assert_eq!(
        class_mismatches, 0,
        "{fixture} final: {class_mismatches} of {determinate} unambiguous rows chose a \
         different class than the reference"
    );
    assert!(
        box_worst <= allowed,
        "{fixture} final: box delta {box_worst:.3e} over {determinate} unambiguous rows \
         exceeds {allowed:.3e} — the two-stage top-k ORDER differs from the reference"
    );
    if determinate == 0 {
        // Not a pass. The fixture simply contains nothing to order, and
        // saying so is the difference between a gate and a green tick.
        eprintln!(
            "final ({fixture}): confidence sequence matches to {conf_worst:.3e}; \
             0 of 300 rows clear conf {FINAL_ORDER_FLOOR} — this fixture cannot test \
             detection ORDER, only the selection's score sequence"
        );
    } else {
        eprintln!(
            "final ({fixture}): conf seq {conf_worst:.3e} · {determinate} unambiguous rows, \
             box max {box_worst:.3e}, classes exact — two-stage top-k order reproduced"
        );
    }
}

/// Every tier the port claims to support, oracled independently.
///
/// **The tiers are not redundant runs of one test.** Three things in the
/// scale derivation are tier-dependent — the 512-channel cap applied before
/// the width scale, the depth-scaled repeat count, and the C3k promotion on
/// m/l/x — and each is exercised by a different subset of these. n and s
/// together cover none of the three, which is exactly how the C3k promotion
/// survived a fully green n-and-s board.
#[test]
fn full_graph_matches_the_reference_on_every_converted_tier() {
    let mut ran = Vec::new();
    for tier in ffai_diana::engine::TIERS {
        if oracle_for_tier(tier) {
            ran.push(tier);
        }
    }
    assert!(
        !ran.is_empty(),
        "no tier could be oracled — convert at least one checkpoint with \
         tools/diana_convert.py before trusting this suite"
    );
    eprintln!("oracled tiers: {ran:?}");
}

/// Returns false when this tier's weights or digest are absent — an
/// expected state, since the checkpoints are AGPL-3.0 and user-converted.
fn oracle_for_tier(tier: &str) -> bool {
    let root = repo_root();
    let weights = root.join(format!("corpora/cache/yolo26{tier}-diana.safetensors"));
    let Some(digest) = Digest::load(&root, tier) else {
        eprintln!(
            "SKIP {tier} oracle: no digest fixture — regenerate with \
             tools/diana_oracle_dump.py --model yolo26{tier}"
        );
        return false;
    };
    if !weights.exists() {
        // The weights are AGPL-3.0 and are never vendored (mission plan §7),
        // so a fresh clone has no safetensors until the user converts their
        // own checkpoint. Say exactly how, rather than failing opaquely.
        eprintln!(
            "SKIP {tier} oracle: {} absent.\n  \
             .venv-diana/Scripts/python.exe tools/diana_convert.py --model yolo26{tier}",
            weights.display()
        );
        return false;
    }
    eprintln!("---- oracle: yolo26{tier} ----");

    let device = Device::Cpu;
    // One mmap shared by all three components: `VarBuilder` is a cheap
    // handle, and `pp` scoping is what separates them. SAFETY: the mapped
    // file is not mutated for the lifetime of the test.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, &device)
            .expect("load safetensors")
    };
    let dims = ffai_diana::config::Dims::for_scale(tier).expect("dims for tier");
    let backbone = Backbone::new(vb.clone(), dims).expect("build backbone");
    let neck = Neck::new(vb.clone(), dims).expect("build neck");
    let head = Head::new(vb, dims, 80, 1, vec![8.0, 16.0, 32.0]).expect("build head");
    let x = fixture_input(&device);
    let traced = backbone.forward_traced(&x).expect("backbone forward");

    // The input itself is the first thing to check: if the fixture formula
    // has drifted from the reference, every downstream delta is noise about
    // the wrong question.
    let inp = digest.tensor("synth", "input").expect("digest has input");
    assert_eq!(inp.shape, vec![1, 3, IMGSZ, IMGSZ], "fixture input shape");
    let got = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for (k, (&i, &w)) in inp.idx.iter().zip(inp.val.iter()).enumerate() {
        assert!(
            (got[i] - w).abs() <= TOL_ABS,
            "fixture input diverges at sample {k} (flat index {i}): ours {} vs reference {w} — \
             the formula in this test and in tools/diana_oracle_dump.py have drifted apart",
            got[i]
        );
    }

    let mut worst = 0f32;
    let mut check = |name: &str, tensor: &Tensor| {
        let Some(want) = digest.tensor("synth", name) else {
            panic!("digest has no {name}");
        };
        assert_eq!(
            tensor.dims().to_vec(),
            want.shape,
            "{name}: shape mismatch — ours {:?} vs reference {:?}",
            tensor.dims(),
            want.shape
        );
        let got = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut layer_worst = 0f32;
        for (&i, &w) in want.idx.iter().zip(want.val.iter()) {
            let d = (got[i] - w).abs();
            if d > layer_worst {
                layer_worst = d;
            }
        }
        let allowed = TOL_ABS.max(TOL_REL * want.scale);
        assert!(
            layer_worst <= allowed,
            "{name}: max delta {layer_worst:.3e} over {} sampled values exceeds {allowed:.3e} \
             (abs floor {TOL_ABS:.0e}, rel {TOL_REL:.0e} x magnitude {:.3e})",
            want.idx.len(),
            want.scale
        );
        let rel = layer_worst / want.scale;
        if rel > worst {
            worst = rel;
        }
        eprintln!("{name}: {:?} max delta {layer_worst:.3e} (rel {rel:.3e})", tensor.dims());
    };

    for (layer, tensor) in traced.iter().enumerate() {
        check(&format!("layer_{layer:02}"), tensor);
    }

    // ---- neck: layers 11..=22 -----------------------------------------
    let b_out = backbone.forward(&x).expect("backbone forward");
    let trace = neck.forward_traced(&b_out).expect("neck forward");
    for (name, tensor) in trace.layers() {
        check(name, tensor);
    }

    // ---- head + decode -------------------------------------------------
    let neck_out = neck.forward(&b_out).expect("neck forward");
    let per_level = head.forward(&neck_out).expect("head forward");
    let (boxes, scores) = head.concat_levels(&per_level).expect("concat levels");
    check("head_boxes", &boxes);
    check("head_scores", &scores);

    let anchors = head.anchors(&per_level).expect("anchors");
    let decoded = head.decoded_tensor(&boxes, &scores, &anchors).expect("decode tensor");
    check("decoded", &decoded);

    // ---- the two-stage top-k, against the reference's own final tensor --
    let decoded_rows = head.decode(&boxes, &scores, &anchors, 300).expect("decode");
    assert_eq!(decoded_rows.len(), 300, "max_det rows expected");
    check_final(&digest, "synth", &decoded_rows);

    // ---- the photo fixture: the only one that can test ORDER ------------
    //
    // The synth input produces no detection above 0.008, so its 300 rows are
    // a ranking of noise and `check_final` correctly declines to assert on
    // their order. The photo fixture has real detections up to 0.88, which
    // is where the two-stage top-k is actually exercised.
    //
    // Its input cannot be recomputed from a formula — it is a corpus JPEG
    // through PIL's BILINEAR resize, which our letterbox does not reproduce
    // bit-for-bit (nor should it; that is a resampler comparison, not a
    // detector one). So the LETTERBOXED CANVAS ships as a tracked PNG. It
    // is uint8, so the round-trip is exact (verified: max |png - reference
    // input| = 0.0), tier-independent, and 640 KB — which buys an order
    // gate that runs on a fresh clone rather than one that needs a 127 MB
    // regenerable dump.
    let photo_png = root.join("corpora/refs/fixtures/diana_photo_input.png");
    let photo = photo_tensor(&photo_png, &device);
    let b = backbone.forward(&photo).expect("backbone forward (photo)");
    let n_out = neck.forward(&b).expect("neck forward (photo)");
    let lv = head.forward(&n_out).expect("head forward (photo)");
    let (bx, sc) = head.concat_levels(&lv).expect("concat levels (photo)");
    let an = head.anchors(&lv).expect("anchors (photo)");
    let rows = head.decode(&bx, &sc, &an, 300).expect("decode (photo)");
    check_final(&digest, "photo", &rows);

    eprintln!("yolo26{tier} full-graph oracle PASS — worst sampled RELATIVE delta {worst:.3e}");
    true
}
