//! Step 4's gate: our assembled prompt must tokenize to the reference's exact
//! ids.
//!
//! ```text
//! .venv-argus/Scripts/python.exe corpora/refs/dump_smolvlm_prompt.py \
//!     --out .oracle/smolvlm-prompt
//! ```
//!
//! # Why this gate is equality and not a tolerance
//!
//! Step 3 compared floats and had to argue about thresholds. Token ids are
//! integers. They match or they do not, and "close" is meaningless — a
//! sequence that is right except for one misplaced `<fake_token_around_image>`
//! is not 99.9 % right, it is a different prompt.
//!
//! That matters here more than anywhere else in the build. §2.2 calls the chat
//! template "the highest-risk silent failure in the whole build", and §7
//! measured it: **43 of 50 answers changed on identical weights** from prompt
//! formatting alone, with nothing raising an error. A wrong assembly does not
//! crash. It produces fluent, plausible, differently-scored text.

use std::path::{Path, PathBuf};

use ffai_argus::prompt::{PromptLayout, expected_fake_tokens, expected_image_tokens};

fn oracle() -> Option<(PathBuf, serde_json::Value)> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.oracle/smolvlm-prompt");
    let text = std::fs::read_to_string(d.join("prompt.json")).ok()?;
    let v = serde_json::from_str(&text).ok()?;
    Some((d, v))
}

fn read_i64(path: &Path) -> Vec<i64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")))
        .collect()
}

/// Locate the tokenizer beside the checkpoint in the HF cache.
fn tokenizer_path() -> Option<PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()?;
    let root = Path::new(&home)
        .join(".cache/huggingface/hub")
        .join("models--HuggingFaceTB--SmolVLM-256M-Instruct")
        .join("snapshots");
    let snap = std::fs::read_dir(root).ok()?.flatten().next()?.path();
    let t = snap.join("tokenizer.json");
    t.exists().then_some(t)
}

/// The gate: assemble, tokenize, compare id-for-id.
#[test]
fn our_assembled_prompt_tokenizes_to_the_reference_ids() {
    let Some((dir, doc)) = oracle() else {
        eprintln!("SKIP: no prompt oracle — run corpora/refs/dump_smolvlm_prompt.py");
        return;
    };
    let Some(tok_path) = tokenizer_path() else {
        eprintln!("SKIP: tokenizer.json not in the HF cache");
        return;
    };
    let tok = tokenizers::Tokenizer::from_file(&tok_path).expect("load tokenizer");

    let question = doc["question"].as_str().expect("question");
    // 17 spans = a 4x4 tile grid plus the global thumbnail.
    let spans = doc["image_spans"].as_array().expect("spans").len();
    let (rows, cols) = (4, 4);
    assert_eq!(spans, rows * cols + 1, "the oracle should have 17 image spans");

    let layout = PromptLayout::default().with_geometry(512, 16, 4);
    let text = layout.user_turn(question, rows, cols);

    // Structural checks first: they localise a failure that the id comparison
    // would only report as "sequences differ at index N".
    assert_eq!(
        text.matches("<image>").count(),
        expected_image_tokens(&layout, rows, cols)
    );
    assert_eq!(
        text.matches("<fake_token_around_image>").count(),
        expected_fake_tokens(rows, cols)
    );

    let enc = tok.encode(text.as_str(), true).expect("encode");
    let got: Vec<i64> = enc.get_ids().iter().map(|&t| i64::from(t)).collect();
    let want = read_i64(&dir.join("input_ids.i64"));

    eprintln!("assembled {} tokens, reference has {}", got.len(), want.len());

    // Report the FIRST divergence with its neighbourhood. "sequences differ"
    // is not a diagnosis; the surrounding tokens say whether a separator was
    // dropped, a block was mis-sized, or the order is wrong.
    if got != want {
        let at = got
            .iter()
            .zip(&want)
            .position(|(a, b)| a != b)
            .unwrap_or(got.len().min(want.len()));
        let lo = at.saturating_sub(6);
        let hi_g = (at + 6).min(got.len());
        let hi_w = (at + 6).min(want.len());
        panic!(
            "prompt assembly differs from the reference.\n  \
             lengths: ours {} vs reference {}\n  \
             first difference at index {at}\n  \
             ours      [{lo}..]: {:?}\n  \
             reference [{lo}..]: {:?}",
            got.len(),
            want.len(),
            &got[lo..hi_g],
            &want[lo..hi_w],
        );
    }

    assert_eq!(
        got.len(),
        doc["n_tokens"].as_u64().expect("n_tokens") as usize
    );
}

/// The image-token count and span layout are the load-bearing arithmetic; they
/// can be checked from the committed summary alone.
#[test]
fn the_span_layout_matches_the_arithmetic() {
    let Some((_, doc)) = oracle() else {
        eprintln!("SKIP: no prompt oracle");
        return;
    };
    let layout = PromptLayout::default().with_geometry(512, 16, 4);
    let spans = doc["image_spans"].as_array().expect("spans");

    assert_eq!(
        doc["n_image_tokens"].as_u64().expect("n_image_tokens") as usize,
        expected_image_tokens(&layout, 4, 4),
        "17 blocks of 64 image tokens"
    );

    // Every block is the same size, and consecutive blocks are separated by
    // exactly two tokens — the fake token and the row/col (or global) marker.
    for w in spans.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        assert_eq!(a["len"].as_u64(), Some(layout.tokens_per_tile as u64));
        let gap = b["start"].as_u64().expect("start") - a["start"].as_u64().expect("start")
            - a["len"].as_u64().expect("len");
        assert!(
            gap == 2 || gap == 3,
            "expected a 2-token separator (3 before the thumbnail, which adds \\n\\n), got {gap}"
        );
    }
}

/// Step 4's second gate: the embedding-level splice.
///
/// Fed the reference's OWN `text_embeds` and its OWN `image_hidden`, our merge
/// must reproduce its `inputs_embeds`. Using our tower's output instead would
/// test the tower and the splice together, and a mismatch could be blamed on
/// either — `codec-bringup-decoder`'s per-stage isolation law.
///
/// The comparison is near-exact on purpose. A `masked_scatter` COPIES values;
/// it does no arithmetic, so unlike step 3 there is no reassociation to
/// tolerate. Anything above float noise means vectors landed in the wrong
/// places, which is the off-by-one that produces fluent, degraded output.
#[test]
fn our_splice_reproduces_the_reference_inputs_embeds() {
    let Some((dir, doc)) = oracle() else {
        eprintln!("SKIP: no prompt oracle");
        return;
    };
    if !dir.join("inputs_embeds.f32").exists() {
        eprintln!("SKIP: no embeds in the dump — re-run the dumper with --embeds");
        return;
    }

    let read_f32 = |name: &str| -> Vec<f32> {
        std::fs::read(dir.join(name))
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect()
    };
    let shape = |name: &str| -> Vec<usize> {
        doc["embeds"][name]["shape"]
            .as_array()
            .expect("shape")
            .iter()
            .map(|x| x.as_u64().expect("dim") as usize)
            .collect()
    };

    let device = candle_core::Device::Cpu;
    let ts = shape("text_embeds");
    let is = shape("image_hidden");
    let text = candle_core::Tensor::from_vec(read_f32("text_embeds.f32"), (ts[0], ts[1], ts[2]), &device)
        .expect("text embeds");
    let img = candle_core::Tensor::from_vec(read_f32("image_hidden.f32"), (is[0], is[1], is[2]), &device)
        .expect("image hidden");

    let ids = read_i64(&dir.join("input_ids.i64"));
    let image_token_id = doc["image_token"]["id"].as_i64().expect("image token id");

    let merged = ffai_argus::prompt::merge_image_embeddings(&text, &img, &ids, image_token_id)
        .expect("splice");
    let got = merged
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("to_vec1");
    let want = read_f32("inputs_embeds.f32");

    assert_eq!(got.len(), want.len(), "merged sequence length");
    let mut max_abs = 0.0f32;
    let mut at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
            at = i;
        }
    }
    eprintln!(
        "  splice  n={}  max_abs={max_abs:.3e}  (row {} of {})",
        got.len(),
        at / ts[2],
        ts[1]
    );
    assert!(
        max_abs == 0.0,
        "the splice only COPIES vectors, so it must be bit-exact; worst \
         difference {max_abs:.3e} at flat index {at} (row {}) means a vector \
         landed in the wrong position",
        at / ts[2]
    );
}

/// A count mismatch must be an ERROR, never a silent truncation — truncating
/// is precisely how every block after the mismatch would be misaligned.
#[test]
fn a_wrong_image_vector_count_is_refused() {
    let device = candle_core::Device::Cpu;
    let text = candle_core::Tensor::zeros((1, 5, 4), candle_core::DType::F32, &device).expect("t");
    // Three image positions, but only two vectors supplied.
    let img = candle_core::Tensor::zeros((2, 4), candle_core::DType::F32, &device).expect("i");
    let ids = vec![7i64, 99, 99, 99, 7];
    let err = ffai_argus::prompt::merge_image_embeddings(&text, &img, &ids, 99)
        .expect_err("a count mismatch must be refused");
    assert!(
        err.to_string().contains("image positions"),
        "the error must name the mismatch: {err}"
    );
}
