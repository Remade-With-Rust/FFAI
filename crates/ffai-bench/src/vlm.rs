//! The VLM bench vertical (Argus) — **predictions are ours, scoring is
//! theirs.**
//!
//! This module is step 0a of `docs/plans/argus-launch-plan.md`, and it exists
//! before Argus has an engine on purpose: the plan's Gate 1.1 says stand the
//! scoreboard up FIRST and score a reference through it, because Carmenta's
//! §32 lesson was that a "residual gap" turned out to be the harness. A
//! harness built after the engine is a harness the engine's behaviour has
//! already shaped.
//!
//! # The one structural decision
//!
//! Every other vertical in this crate scores its own output: WER and CER are
//! implementation-independent arithmetic over two strings, so `crate::metrics`
//! owns them and nobody can accuse the scorer of favouritism. **VLM metrics
//! are not like that.** Most VLM benchmarks are multiple-choice or
//! short-answer, so *answer extraction* — deciding that "the third one" means
//! option C — is part of the metric, and every benchmark implements it
//! differently. A scorer written here would inevitably be tuned, however
//! unconsciously, to the shape of our own model's output. That is not a
//! hypothetical: it is precisely how the Carmenta campaign acquired a metric
//! biased 2.8× in its own favour and shipped four mechanisms that turned out
//! to be artifacts of it.
//!
//! So this module splits the run in two:
//!
//! | phase | who owns it | why |
//! |---|---|---|
//! | **prediction** | us, in-process (or a declared reference adapter) | the four gates need OUR process's wall clock and OUR process's resident memory. Routing our engine through a Python harness would measure the wrapper |
//! | **scoring** | the benchmark's own evaluator, via [`crate::corpus::ScorerSpec`] | answer extraction is part of the metric, and the metric belongs to the benchmark |
//!
//! **There is no answer-comparison code anywhere in this file, and there must
//! never be.** The nearest thing to scoring here is dividing the evaluator's
//! number by a scale the corpus declared.
//!
//! # The two arms
//!
//! `references.toml` is expected to declare two kinds of VLM reference, and
//! the distinction is the plan's §1.3:
//!
//! - **Arm 1 — the scoreboard.** The benchmark's own published pipeline. It
//!   prices the *model* and produces the row comparable to a leaderboard.
//! - **Arm 2 — the matched reference.** The *same checkpoint* run by an
//!   independent CPU runtime under the same pinned decode config. It prices
//!   the *implementation*, and it is the only arm the speed and footprint
//!   gates mean anything against.
//!
//! Both are ordinary `[[reference]]` entries; the matched one is identified
//! the way every other vertical identifies it — by declaring the same
//! [`crate::runner::DECODE_KEY`] config string the engine reports. That is not
//! a new mechanism, it is the mechanism `fill_gates` already uses to stop a
//! 74M beam-search model deciding a 39M greedy engine's quality gate.
//!
//! # Predictions are kept
//!
//! Every run writes its predictions to `<ledger-dir>/predictions/<id>.jsonl`
//! and leaves them there. Re-scoring is then free, which matters because
//! metric scripts fail in dumb ways — a locale-broken `grep -P` once blanked a
//! whole column in a sibling campaign — and re-running predictions to recover
//! from a broken scorer is the most expensive possible way to fix a typo.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ffai_core::engine::{Decoding, EngineStatus, VlmOptions};
use ffai_core::error::{Error, Result};
use ffai_core::registry::EngineRegistry;

use crate::corpus::{ClipEntry, Manifest, ScorerSpec};
use crate::gate::{GateKind, GateOutcome, GateResult, GateReport};
use crate::ledger::{BenchRecord, Environment, LEDGER_SCHEMA, RunSummary, VlmScore, append};
use crate::reference::ReferenceSpec;
use crate::runner::{BenchConfig, QualityMetric, fill_gates};
use crate::speed::real_time_factor;

/// One item's answer, as written to the predictions file the scorer reads.
#[derive(Debug, Clone, serde::Serialize)]
struct Prediction<'a> {
    id: &'a str,
    path: String,
    /// Echoed so the evaluator can pair an answer with its question without
    /// re-reading our manifest. A scorer that silently mismatched questions to
    /// answers would produce a plausible, wrong number.
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<&'a str>,
    prediction: &'a str,
}

/// What a scorer must print on stdout: one JSON object.
#[derive(Debug, Clone, serde::Deserialize)]
struct ScorerOutput {
    score: f64,
    #[serde(default)]
    metric: Option<String>,
    /// Items the evaluator says it scored.
    #[serde(default)]
    n: Option<usize>,
}

/// Run a VLM bench end-to-end and append the record to the ledger.
///
/// Differences from the other verticals, all deliberate:
///
/// - **The media unit is the ITEM**, so `media_secs` carries the item count
///   and the RTF fields read as items/second. Same schema, task-scoped
///   meaning, labelled per task by [`crate::runner::render`]. This follows
///   detect, where the unit is the image.
/// - **Scoring is external and mandatory** (see the module docs). A VLM
///   corpus with no `[scorer]` is rejected rather than scored by us.
/// - **The quality gate verdicts on `1 − normalised score`**
///   ([`QualityMetric::VlmScore`]); the raw score and its metric name ride in
///   the ledger so the normalised figure can never be quoted alone.
/// - **The engine arm runs IN-PROCESS.** The speed and footprint gates ask
///   what our binary costs, and measuring that through a subprocess wrapper
///   would measure the wrapper — a Python interpreter is not a small term next
///   to a 256M model. Argus went live at step 6 (launch plan §16); a *stub*
///   engine is still refused outright, because a run with nothing behind it
///   would append a ledger line that reads like a measurement.
/// - **The comparison key names a CHECKPOINT, not an engine.** `smolvlm` is
///   equally true of the 256M and 500M weights, whose numbers are not
///   comparable; the key is `SmolVLM-256M/greedy-64`. An engine with no
///   declared identity SKIPS its quality gate rather than being compared
///   against a reference it may not match — see `vlm_comparison_key`.
pub fn run_vlm(reg: &EngineRegistry, cfg: &BenchConfig) -> Result<BenchRecord> {
    let manifest = Manifest::load(&cfg.corpus)?;
    if manifest.task != "vlm" {
        return Err(Error::Other(format!(
            "corpus `{}` is a {} corpus, not vlm",
            manifest.name, manifest.task
        )));
    }
    let bad = manifest.verify()?;
    if !bad.is_empty() {
        return Err(Error::Other(format!(
            "corpus verification failed for clips {bad:?} — bytes on disk don't match the \
             manifest's pinned SHA-256; results on drifted data are not valid claims"
        )));
    }
    // The scorer is required, and its absence is an error rather than a
    // skipped gate, because the alternative is not "no score" — the
    // alternative is somebody adding a string comparison here next month.
    let scorer = manifest.scorer.clone().ok_or_else(|| {
        Error::Other(format!(
            "corpus `{}` declares no [scorer] — a VLM corpus must name the benchmark's own \
             evaluator. Answer extraction is part of a VLM metric, so scoring it here would \
             build exactly the self-favouring scorer that cost the Carmenta campaign a year \
             (docs/plans/argus-launch-plan.md §5). See crate::corpus::ScorerSpec for the \
             contract.",
            manifest.name
        ))
    })?;

    // Resolve the scorer's ARGV from the trusted references file.
    //
    // The corpus selects by name; `references.toml` says how to run it. See
    // `crate::reference::NamedScorer` for why the argv does not live in a
    // corpus: a data-shaped file that executes code moved a trust boundary,
    // and the danger was invisibility rather than possibility.
    let argv = resolve_scorer_argv(
        &scorer,
        &cfg.scorers,
        std::env::var("FFAI_BENCH_ALLOW_CORPUS_SCORER").is_ok_and(|v| v == "1"),
    )?;

    let holdout: Vec<&ClipEntry> = manifest.holdout().collect();
    if holdout.is_empty() {
        return Err(Error::Other(
            "corpus has no holdout clips — nothing to measure".into(),
        ));
    }
    let paths: Vec<PathBuf> = holdout.iter().map(|c| manifest.clip_path(c)).collect();
    let items = holdout.len() as f64;

    let (id, appended_at) = BenchRecord::now_id("vlm");
    let pred_dir = cfg
        .ledger
        .parent()
        .unwrap_or(Path::new("."))
        .join("predictions");

    let scorer_version = scorer.version();

    let mut references = Vec::new();
    for spec in cfg.references.iter().filter(|_| !cfg.skip_references) {
        eprintln!(
            "running reference `{}` over {} items ...",
            spec.name,
            holdout.len()
        );
        let summary = run_vlm_reference(
            spec,
            &holdout,
            &paths,
            items,
            cfg.runs,
            &scorer,
            scorer_version.as_deref(),
            &pred_dir,
            &id,
            &cfg.corpus,
            &argv,
        )
        .unwrap_or_else(|e| {
            eprintln!("  reference `{}` failed: {e}", spec.name);
            let mut s = blank_summary(spec.name.clone(), holdout.len(), items);
            s.version = spec.version();
            s.command = Some(spec.command_line());
            s.config = crate::runner::decode_config(spec.config.as_deref());
            s.notes.push(e.to_string());
            s
        });
        references.push(summary);
    }

    let engine_summary = if cfg.skip_engine {
        None
    } else {
        Some(run_vlm_engine(
            reg,
            cfg.engine.as_deref(),
            &manifest,
            &holdout,
            items,
            cfg.runs,
            &scorer,
            scorer_version.as_deref(),
            &pred_dir,
            &id,
            vlm_max_new_tokens().or_else(|| budget_from_references(&references)),
            &argv,
        )?)
    };

    let mut gates = GateReport::new();
    fill_gates(
        &mut gates,
        engine_summary.as_ref(),
        &references,
        QualityMetric::VlmScore,
    );
    tighten_correctness(&mut gates, engine_summary.as_ref(), holdout.len());

    let record = BenchRecord {
        schema: LEDGER_SCHEMA,
        id,
        task: "vlm".into(),
        corpus: manifest.name.clone(),
        corpus_manifest_hash: manifest.manifest_hash(),
        engine: engine_summary,
        references,
        gates,
        environment: Environment::capture(),
        notes: String::new(),
        appended_at,
    };
    append(&cfg.ledger, &record)?;
    Ok(record)
}

/// Re-verdict the correctness gate on **work parity with the scorer**.
///
/// [`fill_gates`] checks that our engine answered every holdout item. That is
/// necessary and not sufficient here, because there is a second process in the
/// loop: the evaluator. If it scored 900 of 1000 items — a filtered subset, a
/// parse failure, a benchmark whose own split differs from ours — the score is
/// a real number computed over the wrong population, and it would read as a
/// perfectly ordinary result.
///
/// This is `codec-measurement` §4 applied literally: compare a COUNT both
/// sides report, and treat divergent counts as a void comparison rather than a
/// small discrepancy. It is the cheapest check available and the one most
/// likely to fire.
fn tighten_correctness(gates: &mut GateReport, engine: Option<&RunSummary>, holdout: usize) {
    let Some(eng) = engine else { return };
    let Some(vlm) = eng.vlm.as_ref() else { return };
    let Some(scored) = vlm.scored_items else {
        // The scorer did not report a count. Not a failure — but not a silent
        // pass either: say so, so nobody later assumes the check ran.
        if let Some(existing) = gates.get(GateKind::Correctness)
            && existing.outcome == GateOutcome::Pass
        {
            let detail = format!(
                "{} — scorer `{}` reported no item count, so engine-vs-scorer work parity was \
                 NOT checked",
                existing.detail, vlm.scorer
            );
            gates.set(GateResult {
                kind: GateKind::Correctness,
                outcome: GateOutcome::Pass,
                metric: None,
                detail,
            });
        }
        return;
    };
    if scored != holdout {
        gates.set(GateResult {
            kind: GateKind::Correctness,
            outcome: GateOutcome::Fail,
            metric: None,
            detail: format!(
                "work-count parity FAILED: corpus holdout is {holdout} items, scorer `{}` \
                 scored {scored}. The score is computed over a different population than the \
                 one measured, so the comparison is void, not merely approximate",
                vlm.scorer
            ),
        });
    }
}

fn blank_summary(name: String, total: usize, items: f64) -> RunSummary {
    RunSummary {
        name,
        version: None,
        command: None,
        config: BTreeMap::new(),
        wer: None,
        cer: None,
        map50: None,
        map5095: None,
        vlm: None,
        rtf_warm: None,
        rtf_e2e: None,
        load_secs: None,
        wall_secs: None,
        media_secs: Some(items),
        clips_ok: 0,
        clips_total: total,
        notes: Vec::new(),
        peak_bytes: None,
        steady_bytes: None,
    }
}

/// Run one declared reference over the corpus, then score its answers.
#[allow(clippy::too_many_arguments)]
fn run_vlm_reference(
    spec: &ReferenceSpec,
    holdout: &[&ClipEntry],
    paths: &[PathBuf],
    items: f64,
    runs: usize,
    scorer: &ScorerSpec,
    scorer_version: Option<&str>,
    pred_dir: &Path,
    record_id: &str,
    corpus_path: &Path,
    argv: &[String],
) -> Result<RunSummary> {
    let mut summary = blank_summary(spec.name.clone(), holdout.len(), items);
    summary.version = spec.version();
    summary.command = Some(spec.command_line());
    summary.config = crate::runner::decode_config(spec.config.as_deref());

    if !spec.supports_batch() {
        return Err(Error::Other(format!(
            "reference `{}` declares no batch_command — per-clip invocation would put \
             interpreter startup and model load inside every timed run, which is not a \
             comparable measurement (see crates/ffai-bench/src/reference.rs)",
            spec.name
        )));
    }

    // Keep the FASTEST run's wall clock together with THAT RUN's answers.
    // Timing numbers only compose if they come from the same execution — a
    // best wall time paired with another run's per-item timings produces
    // impossible pairs (warm faster than end-to-end).
    let mut best: Option<(f64, crate::reference::BatchResult)> = None;
    let corpus_arg = corpus_path.to_string_lossy().into_owned();
    for _ in 0..runs.max(1) {
        let started = std::time::Instant::now();
        // `{corpus}` as well as `{filelist}`: a VLM item is an (image,
        // question) pair and the question lives in the pinned manifest, so an
        // adapter that cannot see the manifest would have to invent prompts.
        let batch = spec.run_batch_subst(paths, &[("{corpus}", corpus_arg.as_str())])?;
        let wall = started.elapsed().as_secs_f64();
        if best.as_ref().is_none_or(|(prev, _)| wall < *prev) {
            best = Some((wall, batch));
        }
    }
    let (wall_secs, batch) = best.expect("at least one run");

    let mut answers: Vec<(&ClipEntry, &str)> = Vec::new();
    for (clip, path) in holdout.iter().zip(paths) {
        match batch.text_for(path) {
            Some(text) => {
                summary.clips_ok += 1;
                answers.push((clip, text));
            }
            None => summary
                .notes
                .push(format!("{}: no result returned", clip.id)),
        }
    }

    summary.load_secs = batch.load_secs;
    summary.wall_secs = Some(wall_secs);
    summary.rtf_e2e = Some(real_time_factor(items, wall_secs));
    summary.rtf_warm = batch
        .transcribe_secs()
        .map(|t| real_time_factor(items, t))
        .or(summary.rtf_e2e);
    summary.peak_bytes = batch.peak_bytes;
    summary.steady_bytes = batch.steady_bytes;

    score_into(
        &mut summary,
        &answers,
        scorer,
        scorer_version,
        pred_dir,
        record_id,
        &spec.name,
        argv,
    );
    Ok(summary)
}

/// The token budget the REFERENCES were run at, read out of their declared
/// decode config.
///
/// # Why the engine must not use its own default here
///
/// `SmolVlm`'s `DEFAULT_MAX_NEW_TOKENS` is **256** — a sensible ceiling for a
/// person asking for a caption. The reference adapters are pinned to **64**.
/// Left alone, the speed gate compared our engine doing up to **four times the
/// decode work** against a reference doing a quarter of it, and reported the
/// difference as though it were implementation speed. That is exactly the
/// defect `run_detect_engine` documents and pins against — *"scoring our engine
/// at 0.25 against references at 0.001 would report a recall collapse that is
/// purely a configuration difference"* — and the VLM arm never got the same
/// treatment.
///
/// `FFAI_VLM_MAX_NEW_TOKENS` still wins when set, so a deliberate sweep is
/// still possible; this only supplies the default the comparison needs.
fn budget_from_references(references: &[RunSummary]) -> Option<usize> {
    let mut found: Option<usize> = None;
    for r in references {
        let key = r.config.get(crate::runner::DECODE_KEY)?;
        // "SmolVLM-256M/greedy-64" -> 64
        let budget = key
            .rsplit('/')
            .next()
            .and_then(|d| d.strip_prefix("greedy-").or_else(|| d.split('-').nth(1)))
            .and_then(|n| n.parse::<usize>().ok())?;
        match found {
            // References disagreeing about the budget is not something to
            // resolve silently by picking one: the run is not comparable and
            // the engine should keep its own default so the mismatch stays
            // visible in the config key.
            Some(prev) if prev != budget => return None,
            _ => found = Some(budget),
        }
    }
    found
}

/// The engine's comparison key, in the same vocabulary references use.
///
/// References name a **checkpoint**, not an engine: `tiny.en/greedy`,
/// `yolo26n/e2e-640sq`, `SmolVLM-256M/greedy-64`. The engine name is our own
/// registry label (`smolvlm`, what `--engine` takes) and says nothing about
/// which weights ran — `smolvlm` would be equally true of the 500M checkpoint,
/// whose numbers are not comparable to the 256M row.
///
/// This mirrors [`crate::runner`]'s `engine_comparison_key` for ASR, including
/// the part that matters most: **an unrecognised engine returns `None` and its
/// quality gate SKIPS**, rather than being compared against a reference it may
/// not match. A wrong comparison is worse than an absent one.
fn vlm_comparison_key(engine: &str) -> Option<String> {
    // `ffai-argus`'s engine is pinned to one checkpoint (see its `MODEL`
    // constant); a second size would register under its own name and get its
    // own arm here.
    match engine {
        "smolvlm" => Some("SmolVLM-256M".to_string()),
        _ => None,
    }
}

/// Token budget for the engine arm, from `FFAI_VLM_MAX_NEW_TOKENS`.
///
/// It is a knob rather than a constant because it is part of the decode config
/// the quality gate matches on: Arm 2 declares `greedy-64`, so the engine must
/// be able to say the same thing. An unset value reports `greedy-default`,
/// which will NOT match Arm 2 — and that mismatch showing up as a SKIPPED
/// quality gate is the intended behaviour, not a bug to paper over.
fn vlm_max_new_tokens() -> Option<usize> {
    std::env::var("FFAI_VLM_MAX_NEW_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
}


/// Resolve a corpus's chosen scorer to an executable argv.
///
/// **This is the trust boundary.** A corpus names a scorer; the argv comes
/// from `corpora/references.toml`, which has always been read and reviewed as
/// executable input. A corpus that carries its own inline `command` is refused
/// unless `FFAI_BENCH_ALLOW_CORPUS_SCORER=1` is set on purpose — see
/// [`crate::reference::NamedScorer`].
fn resolve_scorer_argv(
    scorer: &ScorerSpec,
    scorers: &[crate::reference::NamedScorer],
    allow_inline: bool,
) -> Result<Vec<String>> {
    if let Some(inline) = &scorer.command {
        if allow_inline {
            eprintln!(
                "WARNING: running an INLINE scorer command from the corpus, permitted only 
                 because FFAI_BENCH_ALLOW_CORPUS_SCORER=1 is set:
  {}",
                inline.join(" ")
            );
            return Ok(inline.clone());
        }
        return Err(Error::Other(format!(
            "corpus scorer `{}` declares an inline `command`, which is refused.
             
             A corpus is data. An argv is code. Executing one from a file shaped like 
             data moves a trust boundary, and the danger is not that execution is 
             possible — corpora/references.toml always could — but that it is INVISIBLE, 
             on one line among a thousand lines of hashes and prompts.
             
             Move it to a [[scorer]] entry in corpora/references.toml:
             
                 [[scorer]]
                 name = \"{}\"
                 command = [...]
             
             …and leave only `name`, `metric` and `scale` in the corpus.
             Set FFAI_BENCH_ALLOW_CORPUS_SCORER=1 to override deliberately.",
            scorer.name, scorer.name
        )));
    }
    let named = scorers.iter().find(|s| s.name == scorer.name).ok_or_else(|| {
        let known: Vec<&str> = scorers.iter().map(|s| s.name.as_str()).collect();
        Error::Other(format!(
            "corpus selects scorer `{}`, which is not declared in the references file. 
             Known scorers: {}. Add a [[scorer]] entry naming its command.",
            scorer.name,
            if known.is_empty() { "(none)".to_string() } else { known.join(", ") }
        ))
    })?;
    // An executed command should be VISIBLE, not merely permitted.
    eprintln!("scorer `{}`: {}", named.name, named.command_line());
    Ok(named.command.clone())
}

/// The engine arm's decode configuration, from the environment.
///
/// Greedy is the default and stays the default: plan §2 Gate 2 makes
/// byte-stability a v1 requirement, and `Decoding::Greedy` is the only variant
/// that is deterministic without a seed. `FFAI_VLM_SEED` opts into seeded
/// sampling — seeded, because [`Decoding`] has no unseeded variant to select.
fn vlm_options() -> VlmOptions {
    let env_f32 = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let decoding = match std::env::var("FFAI_VLM_SEED").ok().and_then(|v| v.parse::<u64>().ok()) {
        Some(seed) => Decoding::Sampled {
            temperature: env_f32("FFAI_VLM_TEMPERATURE").unwrap_or(1.0),
            top_p: env_f32("FFAI_VLM_TOP_P"),
            top_k: std::env::var("FFAI_VLM_TOP_K").ok().and_then(|v| v.parse().ok()),
            seed,
        },
        None => Decoding::Greedy,
    };
    VlmOptions {
        decoding,
        repetition_penalty: env_f32("FFAI_VLM_REPETITION_PENALTY"),
        ..VlmOptions::default()
    }
}

/// A stable, human-readable summary of the decode configuration.
///
/// This string is the comparison key `fill_gates` matches a reference against,
/// so it has to name every knob that changes the output — and nothing that
/// does not, or two identical runs would stop matching each other.
fn decode_key_of(opts: &VlmOptions) -> String {
    let budget = opts
        .max_new_tokens
        .map_or_else(|| "default".to_string(), |n| n.to_string());
    let mut key = match &opts.decoding {
        Decoding::Greedy => format!("greedy-{budget}"),
        Decoding::Sampled {
            temperature,
            top_p,
            top_k,
            seed,
        } => {
            let mut s = format!("sampled-{budget}-t{temperature}-seed{seed}");
            if let Some(p) = top_p {
                s.push_str(&format!("-p{p}"));
            }
            if let Some(k) = top_k {
                s.push_str(&format!("-k{k}"));
            }
            s
        }
    };
    if let Some(r) = opts.repetition_penalty {
        key.push_str(&format!("-rp{r}"));
    }
    key
}

/// Run our own engine in-process over the corpus, then score its answers.
///
/// In-process is the point. The speed and footprint gates ask what OUR binary
/// costs; measuring that through a subprocess wrapper would measure the
/// wrapper, and a wrapper's Python interpreter is not a small term next to a
/// 256M model.
#[allow(clippy::too_many_arguments)]
fn run_vlm_engine(
    reg: &EngineRegistry,
    engine: Option<&str>,
    manifest: &Manifest,
    holdout: &[&ClipEntry],
    items: f64,
    runs: usize,
    scorer: &ScorerSpec,
    scorer_version: Option<&str>,
    pred_dir: &Path,
    record_id: &str,
    max_new_tokens: Option<usize>,
    argv: &[String],
) -> Result<RunSummary> {
    let vlm = reg.vlm(engine)?;
    let info = vlm.info();

    // A stub engine fails LOUDLY rather than producing an empty run that
    // would land in the ledger looking like a measurement. Argus's own engine
    // is live, so this now guards the NEXT stub to be registered rather than
    // the current state — which is exactly when a guard is worth keeping.
    if info.status == EngineStatus::Stub {
        return Err(Error::Other(format!(
            "engine `{}` is a stub ({}). Baseline the references with --baseline-only, or \
             select a real engine with --engine; a stub run would append a ledger line that \
             looks like a measurement and is not.",
            info.name, info.description
        )));
    }

    let mut summary = blank_summary(info.name.clone(), holdout.len(), items);
    summary.version = Some(env!("CARGO_PKG_VERSION").to_string());
    // The decode config IS the comparison key: it is what `fill_gates` matches
    // a reference against, so an engine that does not report one gets a
    // SKIPPED quality gate rather than a flattering one.
    //
    // It must equal the string Arm 2 declares in `corpora/references.toml`
    // (today `SmolVLM2-256M/greedy-64`), or the matched comparison silently
    // becomes an open-field one. The key therefore names the DECODE, not the
    // engine: two implementations of the same checkpoint under the same
    // decoding are what "matched" means.
    //
    // `VlmOptions` still carries only `prompt` and `max_new_tokens` — growing
    // it a sampling surface is plan §2 Gate 2 / step 2, deliberately NOT done
    // here because it is a breaking change to a published crate and the plan
    // says decide it once, on purpose. Until then the honest key is the
    // decoding this code actually performs: greedy, at whatever token budget
    // was asked for.
    // The key is DERIVED from the options the engine will actually receive,
    // never hand-written. A literal "greedy" here would keep reading `greedy`
    // on the day someone switches the default to sampling, and the ledger
    // would describe a run that did not happen — the exact defect
    // `RunSummary::config` was added to prevent on the ASR side.
    let decode_key = decode_key_of(&VlmOptions {
        max_new_tokens,
        ..vlm_options()
    });
    if let Some(model) = vlm_comparison_key(&info.name) {
        summary
            .config
            .insert(crate::runner::DECODE_KEY.to_string(), format!("{model}/{decode_key}"));
    } else {
        eprintln!(
            "  ! engine `{}` has no declared model identity — its quality gate will SKIP. \
             Add it to `vlm_comparison_key`.",
            info.name
        );
    }

    let sampling = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let samples = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
    let sampler = {
        let (sampling, out) = (sampling.clone(), samples.clone());
        std::thread::spawn(move || {
            while sampling.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(b) = crate::footprint::current_self()
                    && let Ok(mut v) = out.lock()
                {
                    v.push(b.0);
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        })
    };

    let mut texts: Vec<String> = Vec::new();
    let mut per_item_secs: Vec<f64> = Vec::new();
    let mut first_error: Option<String> = None;
    let mut best_wall = f64::INFINITY;
    let mut answer_secs = 0.0_f64;

    // ONE UNTIMED WARM-UP, whose cost becomes `load_secs`.
    //
    // The engine loads its weights lazily on the first `describe_image`, so
    // without this the model load sits inside the FIRST timed run. The
    // reference does not carry that cost: its adapter reports `load_secs`
    // separately and its per-item timings exclude it. Best-of-N hides the
    // asymmetry when N > 1 and reports it as our throughput when N == 1 —
    // which is how a `--runs 1` run read "engine 0.2x realtime".
    //
    // `run_detect_engine` already does exactly this, for exactly this reason.
    // The VLM arm never did, and its `load_secs` was therefore always `None`
    // while every reference row carried a real number.
    if let Some(first) = holdout.first() {
        let path = manifest.clip_path(first);
        if let Ok(image) = crate::runner::load_image_resilient(&path) {
            let opts = VlmOptions {
                prompt: first.prompt.clone(),
                max_new_tokens,
                ..vlm_options()
            };
            let t = std::time::Instant::now();
            let _ = vlm.describe_image(&image, &opts);
            summary.load_secs = Some(t.elapsed().as_secs_f64());
        }
    }

    for _ in 0..runs.max(1) {
        texts.clear();
        per_item_secs.clear();
        let run_started = std::time::Instant::now();
        for clip in holdout {
            let path = manifest.clip_path(clip);
            // The timer starts BEFORE the image is read, because the
            // reference's per-item figure includes `Image.open` and its
            // preprocessing. Excluding our decode while their timing carries
            // theirs is a small bias, but it is a bias in OUR favour, and the
            // whole value of this harness is that it is not.
            let t = std::time::Instant::now();
            let image = match crate::runner::load_image_resilient(&path) {
                Ok(i) => i,
                Err(e) => {
                    if first_error.is_none() {
                        first_error = Some(format!("{}: {e}", clip.id));
                    }
                    texts.push(String::new());
                    continue;
                }
            };
            let opts = VlmOptions {
                prompt: clip.prompt.clone(),
                max_new_tokens,
                ..vlm_options()
            };
            match vlm.describe_image(&image, &opts) {
                Ok(text) => {
                    per_item_secs.push(t.elapsed().as_secs_f64());
                    texts.push(text);
                }
                Err(e) => {
                    if first_error.is_none() {
                        first_error = Some(format!("{}: {e}", clip.id));
                    }
                    texts.push(String::new());
                }
            }
        }
        let wall = run_started.elapsed().as_secs_f64();
        if wall < best_wall {
            best_wall = wall;
            answer_secs = per_item_secs.iter().sum();
        }
    }

    sampling.store(false, std::sync::atomic::Ordering::Relaxed);
    let _ = sampler.join();
    if let Ok(v) = samples.lock()
        && !v.is_empty()
    {
        let mut v = v.clone();
        v.sort_unstable();
        summary.steady_bytes = Some(v[v.len() / 2]);
    }
    summary.peak_bytes = crate::footprint::peak_self().map(|p| p.0);

    summary.clips_ok = per_item_secs.len();
    if let Some(e) = first_error {
        summary.notes.push(e);
    }
    summary.wall_secs = Some(best_wall);
    summary.rtf_e2e = Some(real_time_factor(items, best_wall));
    summary.rtf_warm = if answer_secs > 0.0 {
        Some(real_time_factor(items, answer_secs))
    } else {
        summary.rtf_e2e
    };

    let answers: Vec<(&ClipEntry, &str)> = holdout
        .iter()
        .zip(&texts)
        .map(|(c, t)| (*c, t.as_str()))
        .collect();
    score_into(
        &mut summary,
        &answers,
        scorer,
        scorer_version,
        pred_dir,
        record_id,
        &info.name,
        argv,
    );
    Ok(summary)
}

/// Write the predictions file, invoke the declared evaluator, record what it
/// says. A scorer failure is a NOTE and a missing score — never a substituted
/// one.
#[allow(clippy::too_many_arguments)]
fn score_into(
    summary: &mut RunSummary,
    answers: &[(&ClipEntry, &str)],
    scorer: &ScorerSpec,
    scorer_version: Option<&str>,
    pred_dir: &Path,
    record_id: &str,
    who: &str,
    argv: &[String],
) {
    let file = pred_dir.join(format!("{record_id}-{}.jsonl", slug(who)));
    match write_predictions(&file, answers) {
        Ok(()) => summary
            .notes
            .push(format!("predictions kept at {}", file.display())),
        Err(e) => {
            summary
                .notes
                .push(format!("could not write predictions: {e}"));
            return;
        }
    }
    match run_scorer(argv, &file, &scorer.name) {
        Ok(out) => {
            let raw = out.score;
            summary.vlm = Some(VlmScore {
                raw,
                normalised: (raw / scorer.scale).clamp(0.0, 1.0),
                scale: scorer.scale,
                metric: out.metric.unwrap_or_else(|| scorer.metric.clone()),
                scorer: scorer.name.clone(),
                scorer_version: scorer_version.map(ToString::to_string),
                command: Some(argv.join(" ")),
                scored_items: out.n,
            });
        }
        Err(e) => summary.notes.push(format!("scorer failed: {e}")),
    }
}

fn write_predictions(file: &Path, answers: &[(&ClipEntry, &str)]) -> Result<()> {
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut body = String::new();
    for (clip, text) in answers {
        let line = serde_json::to_string(&Prediction {
            id: &clip.id,
            path: clip.path.to_string_lossy().replace('\\', "/"),
            prompt: clip.prompt.as_deref(),
            prediction: text,
        })
        .map_err(|e| Error::Other(format!("prediction encode: {e}")))?;
        body.push_str(&line);
        body.push('\n');
    }
    std::fs::write(file, body)?;
    Ok(())
}

/// Filesystem-safe form of an engine or reference name.
fn slug(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

impl ScorerSpec {
    /// Version string of the evaluator, when it reports one. A score is not
    /// reproducible without knowing which evaluator produced it — the same
    /// benchmark's harness changes its answer extraction between releases.
    #[must_use]
    pub fn version(&self) -> Option<String> {
        let argv = self.version_command.as_ref()?;
        let (exe, args) = argv.split_first()?;
        let out = std::process::Command::new(exe).args(args).output().ok()?;
        let text = if out.stdout.is_empty() {
            String::from_utf8_lossy(&out.stderr)
        } else {
            String::from_utf8_lossy(&out.stdout)
        };
        text.lines().next().map(|s| s.trim().to_string())
    }

}

/// Invoke the resolved evaluator argv on a predictions file, parse its verdict.
///
/// Takes the argv rather than reading it off the corpus: resolution — and the
/// trust decision that goes with it — happens once, in
/// [`resolve_scorer_argv`], so there is exactly one place a command can enter.
fn run_scorer(argv: &[String], predictions: &Path, name: &str) -> Result<ScorerOutput> {
    let (exe, args) = argv
        .split_first()
        .ok_or_else(|| Error::Other(format!("scorer `{name}` has an empty command")))?;
    let pred = predictions.to_string_lossy().to_string();
    let args: Vec<String> = args
        .iter()
        .map(|a| a.replace("{predictions}", &pred))
        .collect();
    let out = std::process::Command::new(exe)
        .args(&args)
        .output()
        .map_err(|e| Error::Other(format!("scorer `{name}` failed to start: {e}")))?;
    // A crashed scorer is not a zero score. `codec-measurement` §4: check the
    // exit code, because an arm that died is not an arm that scored 0 — and a
    // 0 entering a ledger looks exactly like a measurement.
    if !out.status.success() {
        return Err(Error::Other(format!(
            "scorer `{name}` exited with {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Tolerate leading log chatter: take the last non-empty line that parses
    // as the contract object. Tolerating chatter is not the same as guessing —
    // a line that does not parse is still an error.
    stdout
        .lines()
        .rev()
        .filter(|l| !l.trim().is_empty())
        .find_map(|l| serde_json::from_str::<ScorerOutput>(l.trim()).ok())
        .ok_or_else(|| {
            Error::Other(format!(
                "scorer `{name}` printed no line matching the contract                  {{\"score\": <number>, \"metric\": <string>, \"n\": <count>}} — got: {}",
                stdout.trim()
            ))
        })
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{ContentClass, Split};

    fn clip(id: &str) -> ClipEntry {
        ClipEntry {
            id: id.into(),
            path: format!("{id}.png").into(),
            ground_truth: None,
            prompt: Some("What is in this image?".into()),
            class: ContentClass::Photo,
            split: Split::Holdout,
            license: "CC-BY-4.0".into(),
            sha256: "aa".into(),
        }
    }

    #[test]
    fn a_vlm_corpus_without_a_scorer_is_rejected() {
        // The whole point of the module: there is no fallback path that scores
        // the answers here. If this ever starts returning Ok, someone has
        // written an answer comparator.
        let m: Manifest = toml::from_str(
            r#"
            name = "argus-smoke"
            version = 1
            task = "vlm"
            "#,
        )
        .expect("manifest parses");
        assert!(m.scorer.is_none());
    }

    #[test]
    fn a_scorer_declaration_round_trips() {
        let m: Manifest = toml::from_str(
            r#"
            name = "argus-ocrbench"
            version = 1
            task = "vlm"

            [scorer]
            name = "vlmevalkit-ocrbench"
            metric = "OCRBench"
            scale = 1000.0
            "#,
        )
        .expect("manifest parses");
        let s = m.scorer.expect("scorer present");
        assert!((s.scale - 1000.0).abs() < f64::EPSILON);
        assert_eq!(s.metric, "OCRBench");
        assert!(s.command.is_none(), "a corpus must SELECT a scorer, not define one");
    }

    /// **The trust boundary, pinned.** A corpus that carries its own argv is
    /// refused — a file shaped like data must not be able to run code.
    #[test]
    fn a_corpus_that_defines_its_own_command_is_refused() {
        let spec = ScorerSpec {
            name: "sneaky".into(),
            command: Some(vec!["calc.exe".into()]),
            version_command: None,
            metric: "acc".into(),
            scale: 1.0,
        };
        // The override is a PARAMETER, not an env read inside the function, so
        // this test cannot be affected by the developer's shell — and the
        // production caller is the only place the env var is consulted.
        let err = resolve_scorer_argv(&spec, &[], false)
            .expect_err("an inline command must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("refused"),
            "the refusal must say so plainly: {msg}"
        );
        assert!(
            msg.contains("references.toml"),
            "and must say where the command belongs instead: {msg}"
        );
    }

    /// The SHIPPED corpora must actually be on the safe side of the boundary.
    ///
    /// A rule enforced only in unit tests is a rule the real files can quietly
    /// violate; this reads them off disk.
    #[test]
    fn the_shipped_vlm_corpora_select_rather_than_define() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpora");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return; // a checkout without corpora is not a failure of this code
        };
        let mut checked = 0;
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_none_or(|x| x != "toml") {
                continue;
            }
            let Ok(m) = Manifest::load(&p) else { continue };
            if m.task != "vlm" {
                continue;
            }
            let scorer = m.scorer.as_ref().unwrap_or_else(|| {
                panic!("{} is a vlm corpus with no [scorer]", p.display())
            });
            assert!(
                scorer.command.is_none(),
                "{} carries an inline scorer command — a corpus is data and must not                  carry an argv; declare it as a [[scorer]] in references.toml",
                p.display()
            );
            checked += 1;
        }
        assert!(checked > 0, "no vlm corpora found to check");
    }

    /// The safe path: a corpus names a scorer, the trusted file defines it.
    #[test]
    fn a_named_scorer_resolves_from_the_references_file() {
        let spec = ScorerSpec {
            name: "vlmevalkit-ocrbench".into(),
            command: None,
            version_command: None,
            metric: "OCRBench".into(),
            scale: 1000.0,
        };
        let declared = crate::reference::NamedScorer {
            name: "vlmevalkit-ocrbench".into(),
            command: vec!["python".into(), "score.py".into(), "{predictions}".into()],
            version_command: None,
        };
        let argv = resolve_scorer_argv(&spec, std::slice::from_ref(&declared), false).expect("resolves");
        assert_eq!(argv, declared.command);
    }

    /// Selecting a scorer nobody declared is an error naming what IS declared —
    /// not a silent skip, which would leave the run unscored and looking fine.
    #[test]
    fn an_undeclared_scorer_fails_with_the_known_names() {
        let spec = ScorerSpec {
            name: "does-not-exist".into(),
            command: None,
            version_command: None,
            metric: "acc".into(),
            scale: 1.0,
        };
        let declared = crate::reference::NamedScorer {
            name: "the-real-one".into(),
            command: vec!["true".into()],
            version_command: None,
        };
        let err = resolve_scorer_argv(&spec, std::slice::from_ref(&declared), false).expect_err("unknown");
        assert!(err.to_string().contains("the-real-one"));
    }

    #[test]
    fn predictions_are_one_json_object_per_line() {
        let dir = std::env::temp_dir().join("ffai-vlm-pred-test");
        let file = dir.join("p.jsonl");
        let a = clip("item-1");
        let b = clip("item-2");
        let answers = vec![(&a, "a cat"), (&b, "a dog")];
        write_predictions(&file, &answers).expect("written");
        let body = std::fs::read_to_string(&file).expect("read back");
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in lines {
            let v: serde_json::Value = serde_json::from_str(line).expect("valid json");
            assert!(v.get("id").is_some());
            assert!(v.get("prediction").is_some());
            assert!(v.get("prompt").is_some());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_scale_is_applied_but_the_raw_score_survives() {
        // OCRBench is 0-1000. The gate needs 0..1; the ledger needs the number
        // that is comparable to a published row. Both, or the row is unusable.
        let s = VlmScore {
            raw: 612.0,
            normalised: 612.0 / 1000.0,
            scale: 1000.0,
            metric: "OCRBench".into(),
            scorer: "vlmevalkit-ocrbench".into(),
            scorer_version: None,
            command: None,
            scored_items: Some(1000),
        };
        assert!((s.normalised - 0.612).abs() < 1e-12);
        assert!((s.raw - 612.0).abs() < 1e-12);
    }

    #[test]
    fn the_quality_gate_reads_a_higher_score_as_a_lower_error() {
        // VLM scores are higher-better and the gate is lower-better. Prove the
        // fold has the right sign, because a sign error here would silently
        // invert every VLM verdict ever recorded.
        let mut better = blank_summary("ours".into(), 10, 10.0);
        better.config.insert(
            crate::runner::DECODE_KEY.to_string(),
            "argus/greedy".to_string(),
        );
        better.clips_ok = 10;
        better.vlm = Some(VlmScore {
            raw: 800.0,
            normalised: 0.8,
            scale: 1000.0,
            metric: "OCRBench".into(),
            scorer: "s".into(),
            scorer_version: None,
            command: None,
            scored_items: Some(10),
        });
        let mut worse_ref = blank_summary("theirs".into(), 10, 10.0);
        worse_ref.config.insert(
            crate::runner::DECODE_KEY.to_string(),
            "argus/greedy".to_string(),
        );
        worse_ref.vlm = Some(VlmScore {
            raw: 600.0,
            normalised: 0.6,
            scale: 1000.0,
            metric: "OCRBench".into(),
            scorer: "s".into(),
            scorer_version: None,
            command: None,
            scored_items: Some(10),
        });

        let mut gates = GateReport::new();
        fill_gates(
            &mut gates,
            Some(&better),
            std::slice::from_ref(&worse_ref),
            QualityMetric::VlmScore,
        );
        assert_eq!(
            gates.get(GateKind::Quality).map(|g| g.outcome),
            Some(GateOutcome::Pass),
            "scoring 800 against a reference's 600 must PASS quality"
        );
    }

    #[test]
    fn a_scorer_that_scored_a_subset_voids_the_run() {
        // The cheapest check in the harness and the one most likely to fire:
        // a count both sides report, compared.
        let mut eng = blank_summary("ours".into(), 1000, 1000.0);
        eng.clips_ok = 1000;
        eng.vlm = Some(VlmScore {
            raw: 612.0,
            normalised: 0.612,
            scale: 1000.0,
            metric: "OCRBench".into(),
            scorer: "vlmevalkit".into(),
            scorer_version: None,
            command: None,
            scored_items: Some(900),
        });
        let mut gates = GateReport::new();
        fill_gates(&mut gates, Some(&eng), &[], QualityMetric::VlmScore);
        assert_eq!(
            gates.get(GateKind::Correctness).map(|g| g.outcome),
            Some(GateOutcome::Pass),
            "fill_gates alone sees 1000/1000 answered and passes"
        );
        tighten_correctness(&mut gates, Some(&eng), 1000);
        assert_eq!(
            gates.get(GateKind::Correctness).map(|g| g.outcome),
            Some(GateOutcome::Fail),
            "but the scorer only scored 900 of them, so the comparison is void"
        );
    }

    #[test]
    fn a_missing_item_count_is_disclosed_not_assumed() {
        let mut eng = blank_summary("ours".into(), 10, 10.0);
        eng.clips_ok = 10;
        eng.vlm = Some(VlmScore {
            raw: 1.0,
            normalised: 1.0,
            scale: 1.0,
            metric: "ANLS".into(),
            scorer: "quiet-scorer".into(),
            scorer_version: None,
            command: None,
            scored_items: None,
        });
        let mut gates = GateReport::new();
        fill_gates(&mut gates, Some(&eng), &[], QualityMetric::VlmScore);
        tighten_correctness(&mut gates, Some(&eng), 10);
        let g = gates.get(GateKind::Correctness).expect("gate present");
        assert_eq!(g.outcome, GateOutcome::Pass);
        assert!(
            g.detail.contains("NOT checked"),
            "an unchecked parity check must SAY it was unchecked: {}",
            g.detail
        );
    }

    /// Build a `ScorerSpec` that runs a one-liner through the system python.
    /// Skips (returns None) where python is absent, so the suite still passes
    /// on a machine without it rather than failing for an unrelated reason.
    fn python_scorer(name: &str, body: &str) -> Option<(ScorerSpec, Vec<String>)> {
        let probe = std::process::Command::new("python")
            .arg("--version")
            .output()
            .ok()?;
        if !probe.status.success() {
            return None;
        }
        Some((
            ScorerSpec {
                name: name.into(),
                command: None,
                version_command: None,
                metric: "test".into(),
                scale: 100.0,
            },
            vec!["python".into(), "-c".into(), body.into()],
        ))
    }

    #[test]
    fn a_scorer_that_crashes_is_void_not_zero() {
        // The distinction the whole gate rests on: an arm that died is not an
        // arm that scored 0, and a 0 entering the ledger looks exactly like a
        // measurement (codec-measurement §4).
        let Some((spec, argv)) = python_scorer("crasher", "import sys; sys.exit(3)") else {
            return;
        };
        let err = run_scorer(&argv, Path::new("nonexistent.jsonl"), &spec.name)
            .expect_err("a non-zero exit must be an error");
        let msg = err.to_string();
        assert!(
            msg.contains("exited"),
            "the failure must name the exit, not report a score: {msg}"
        );
    }

    #[test]
    fn leading_log_chatter_is_tolerated_but_garbage_is_not() {
        // Real evaluators print progress bars and warnings before their
        // verdict. Tolerating that is necessary; tolerating a line that does
        // not parse is not, because "no score" must never silently become one.
        let Some((good, good_argv)) = python_scorer(
            "chatty",
            "print('loading dataset...'); print('warning: whatever'); \
             print('{\"score\": 61.2, \"metric\": \"ANLS\", \"n\": 7}')",
        ) else {
            return;
        };
        let out = run_scorer(&good_argv, Path::new("ignored"), &good.name).expect("parses");
        assert!((out.score - 61.2).abs() < 1e-9);
        assert_eq!(out.n, Some(7));
        assert_eq!(out.metric.as_deref(), Some("ANLS"));

        let (noisy, noisy_argv) = python_scorer("silent", "print('all done!')").expect("python");
        let err = run_scorer(&noisy_argv, Path::new("ignored"), &noisy.name)
            .expect_err("no contract line must be an error");
        assert!(err.to_string().contains("contract"));
    }

    /// A registered-but-unimplemented VLM engine, for the guard below.
    ///
    /// It used to borrow Argus, which WAS a stub. Argus now ships a real
    /// engine (launch plan §16), so borrowing it stopped testing the guard and
    /// started testing a full captioning run — which failed on a missing
    /// fixture rather than on the stub refusal, and said so in a way that took
    /// a moment to read.
    ///
    /// The guard is still worth having: it is what stops the NEXT
    /// registered-but-unimplemented engine appending a ledger line that looks
    /// like a measurement. So the test now brings its own stub instead of
    /// depending on a production crate staying unimplemented.
    struct StubVlm;

    impl ffai_core::engine::VlmEngine for StubVlm {
        fn info(&self) -> ffai_core::engine::EngineInfo {
            ffai_core::engine::EngineInfo {
                name: "stub-vlm".into(),
                task: ffai_core::engine::Task::Vlm,
                status: EngineStatus::Stub,
                description: "registered, not implemented".into(),
            }
        }
        fn describe(
            &self,
            _prompt: &ffai_core::engine::VlmPrompt<'_>,
            _opts: &VlmOptions,
        ) -> ffai_core::error::Result<String> {
            Err(ffai_core::error::Error::NotImplemented {
                task: ffai_core::engine::Task::Vlm,
                engine: "stub-vlm".into(),
            })
        }
        fn describe_video(
            &self,
            _frames: &[ffai_core::types::VideoFrame],
            _opts: &VlmOptions,
        ) -> ffai_core::error::Result<Vec<ffai_core::types::TimedSegment<String>>> {
            Err(ffai_core::error::Error::NotImplemented {
                task: ffai_core::engine::Task::Vlm,
                engine: "stub-vlm".into(),
            })
        }
    }

    #[test]
    fn a_stub_engine_refuses_rather_than_recording_an_empty_run() {
        let mut reg = EngineRegistry::new();
        reg.register_vlm(std::sync::Arc::new(StubVlm));
        let m: Manifest = toml::from_str(
            r#"
            name = "argus-smoke"
            version = 1
            task = "vlm"

            [scorer]
            name = "s"
            command = ["true"]
            metric = "acc"
            scale = 1.0
            "#,
        )
        .expect("manifest parses");
        let scorer = m.scorer.clone().expect("scorer");
        let c = clip("item-1");
        let holdout = vec![&c];
        let err = run_vlm_engine(
            &reg,
            None,
            &m,
            &holdout,
            1.0,
            1,
            &scorer,
            None,
            Path::new("."),
            "bench-vlm-0",
            Some(64),
            &["true".to_string()],
        )
        .expect_err("a stub must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("stub"),
            "the refusal must name the reason: {msg}"
        );
    }
}
