//! Form-value bench over `carmenta-form-v1` (docs/plans/commercial-gaps.md,
//! gap 12): how often a typed value on a ruled line is read EXACTLY, and how
//! often a WRONG value still passes a caller's confidence gate.
//!
//! ```text
//! cargo run --release -p ffai-carmenta --example form_bench -- \
//!     [--engine mobiledet-crnn|mobiledet-svtr] [--split train|holdout|all] \
//!     [--gate 0.85] [--remove-rules] [--charset-by-kind]
//! ```
//!
//! Scoring is by VALUE, because a value is what a caller stores:
//! * `exact`: the value string appears, whitespace-collapsed, in the output.
//! * `CER`: edit distance from the value to the output line nearest to it,
//!   over the value's length.
//! * `wrong>gate`: the value was not read exactly AND the line carrying it
//!   passed the confidence gate. This is the count that embarrassed a caller.
//!
//! Every rule condition is paired with its `none` control on the same value,
//! font and size, so a difference between rows is the rule's doing.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ffai_carmenta::engine::{CraftCrnn, DetStage, RecStage};
use ffai_core::engine::{OcrEngine, OcrOptions};

/// Smallest edit distance from `value` to any substring of `text` (the
/// value's own span inside a line that also carries the label).
fn substring_distance(value: &[char], text: &[char]) -> usize {
    let mut prev = vec![0usize; text.len() + 1];
    for (i, cv) in value.iter().enumerate() {
        let mut cur = vec![i + 1; text.len() + 1];
        for (j, ct) in text.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(cv != ct))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev.into_iter().min().unwrap_or(value.len())
}

fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Default)]
struct Row {
    n: usize,
    exact: usize,
    edits: usize,
    chars: usize,
    wrong_passed: usize,
    conf_sum: f32,
    /// `forms::pair_fields` returned this clip's label with its exact value.
    paired: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let has = |k: &str| args.iter().any(|a| a == k);
    let engine = get("--engine").unwrap_or_else(|| "mobiledet-crnn".into());
    let split = get("--split").unwrap_or_else(|| "all".into());
    let gate: f32 = get("--gate").and_then(|g| g.parse().ok()).unwrap_or(0.85);
    let remove_rules = has("--remove-rules");
    let charset_by_kind = has("--charset-by-kind");
    let misses = has("--misses");

    let rec = match engine.as_str() {
        "mobiledet-svtr" => RecStage::Svtr,
        "mobiledet-crnn" => RecStage::Crnn,
        other => panic!("unsupported engine {other}"),
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let e = CraftCrnn::variant_in(rec, DetStage::MobileDet, &root.join("models"));

    let clips = root.join("corpora/clips/carmenta-form");
    let mut metas: Vec<PathBuf> = std::fs::read_dir(&clips)
        .expect("corpus: run tools/carmenta_form_corpus.py")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    metas.sort();

    // (confidence of the value's line, read exactly?) for every clip: the
    // calibration question is how well the first predicts the second.
    let mut scored: Vec<(f32, bool)> = Vec::new();
    let mut rows: BTreeMap<(String, String), Row> = BTreeMap::new();
    let mut kinds: BTreeMap<(String, String), Row> = BTreeMap::new();
    for meta_path in &metas {
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(meta_path).unwrap()).unwrap();
        let idx = meta["index"].as_u64().unwrap();
        let in_split = match split.as_str() {
            "train" => idx % 2 == 0,
            "holdout" => idx % 2 == 1,
            _ => true,
        };
        if !in_split {
            continue;
        }
        let value = meta["value"].as_str().unwrap().to_string();
        let kind = meta["kind"].as_str().unwrap().to_string();
        let rule = meta["rule"].as_str().unwrap().to_string();
        let ink = meta["ink"].as_str().unwrap().to_string();
        let img = ffai_media::load_image(&meta_path.with_extension("png")).unwrap();

        // The per-kind charset a form reader would pass for this field.
        let charset = charset_by_kind.then(|| match kind.as_str() {
            "amount" => "0123456789$,.- to".to_string(),
            _ => String::new(),
        });
        let opts = OcrOptions {
            charset: charset.filter(|c| !c.is_empty()),
            remove_rules,
            ..OcrOptions::default()
        };
        let out = e
            .recognize(&img, &opts)
            .unwrap_or_else(|err| panic!("{}: {err}", meta_path.display()));

        let v: Vec<char> = value.chars().collect();
        let lines: Vec<(String, Option<f32>)> = out
            .blocks
            .iter()
            .flat_map(|b| &b.lines)
            .map(|l| (collapse(&l.text), l.confidence))
            .collect();
        let best = lines
            .iter()
            .map(|(t, c)| (substring_distance(&v, &t.chars().collect::<Vec<_>>()), *c))
            .min_by_key(|(d, _)| *d);
        let (dist, conf) = best.unwrap_or((v.len(), None));
        let exact = collapse(
            &lines
                .iter()
                .map(|(t, _)| t.as_str())
                .collect::<Vec<_>>()
                .join(" "),
        )
        .contains(&collapse(&value));
        scored.push((conf.unwrap_or(0.0), exact));
        // Field pairing: the clip's label (without its colon) must come back
        // with the clip's exact value.
        let label = meta["label"].as_str().unwrap().trim_end_matches(':').to_string();
        let paired = ffai_carmenta::forms::pair_fields(&out)
            .iter()
            .any(|f| f.label == label && collapse(&f.value) == collapse(&value));
        if misses && !exact {
            let got = lines
                .iter()
                .min_by_key(|(t, _)| substring_distance(&v, &t.chars().collect::<Vec<_>>()))
                .map_or(String::new(), |(t, c)| {
                    format!("{t:?} conf {:.3}", c.unwrap_or(0.0))
                });
            eprintln!("MISS {rule}/{ink} {value:?} -> {got}");
        }
        for row in [
            rows.entry((rule.clone(), ink.clone())).or_default(),
            kinds.entry((rule.clone(), kind.clone())).or_default(),
        ] {
            row.n += 1;
            row.exact += usize::from(exact);
            row.edits += dist;
            row.chars += v.len();
            row.conf_sum += conf.unwrap_or(0.0);
            row.wrong_passed += usize::from(!exact && conf.is_some_and(|c| c >= gate));
            row.paired += usize::from(paired);
        }
    }

    println!(
        "engine {engine} | split {split} | gate {gate} | remove_rules {remove_rules} | charset_by_kind {charset_by_kind}"
    );
    let print = |title: &str, table: &BTreeMap<(String, String), Row>| {
        println!(
            "\n{title:<16} {:>4} {:>8} {:>8} {:>10} {:>9} {:>8}",
            "n", "exact", "CER", "wrong>gate", "mean conf", "paired"
        );
        for ((a, b), r) in table {
            println!(
                "{:<16} {:>4} {:>7.1}% {:>7.2}% {:>10} {:>9.3} {:>7.1}%",
                format!("{a}/{b}"),
                r.n,
                100.0 * r.exact as f32 / r.n.max(1) as f32,
                100.0 * r.edits as f32 / r.chars.max(1) as f32,
                r.wrong_passed,
                r.conf_sum / r.n.max(1) as f32,
                100.0 * r.paired as f32 / r.n.max(1) as f32
            );
        }
    };
    print("rule/ink", &rows);
    print("rule/kind", &kinds);

    // Calibration: AUROC = P(a right value's confidence > a wrong value's),
    // ties counted half. 1.0 separates perfectly; 0.5 is a coin.
    let right: Vec<f32> = scored.iter().filter(|s| s.1).map(|s| s.0).collect();
    let wrong: Vec<f32> = scored.iter().filter(|s| !s.1).map(|s| s.0).collect();
    let mut wins = 0f64;
    for r in &right {
        for w in &wrong {
            wins += if r > w {
                1.0
            } else if (r - w).abs() < f32::EPSILON {
                0.5
            } else {
                0.0
            };
        }
    }
    let auroc = wins / (right.len() * wrong.len()).max(1) as f64;
    let kept_right = right.iter().filter(|&&c| c >= gate).count();
    let kept_wrong = wrong.iter().filter(|&&c| c >= gate).count();
    println!(
        "\ncalibration: AUROC {auroc:.3} | at gate {gate}: kept {kept_right}/{} right, {kept_wrong}/{} wrong",
        right.len(),
        wrong.len()
    );
}
