//! The claims ledger, ported from Prometheus (`prom-ledger`).
//!
//! Append-only JSONL: one record per bench run, never edited or deleted.
//! Every public performance/quality claim `FFai` makes should be traceable to
//! a ledger line that pins the corpus (manifest hash), the environment
//! (os/arch/rustc/cpu), the reference versions, and the full gate report —
//! reproducible from the line alone. Losses stay in the ledger too; a pruned
//! result is knowledge.

use crate::gate::GateReport;
use ffai_core::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Schema 2 adds `RunSummary::config` — the in-process engine's effective
/// decode configuration. Schema-1 lines remain readable (`serde(default)`),
/// but they carry no config, so a line with `schema: 1` cannot be assumed to
/// share defaults with a current run. That is the point of the bump: it makes
/// the older lines' ambiguity explicit instead of silent.
pub const LEDGER_SCHEMA: u32 = 2;

/// Summary of one implementation's run over the holdout split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    /// Engine or reference name ("whisper-candle", "whisper-cpp", ...).
    pub name: String,
    /// Version string when known (references) or crate version (engines).
    pub version: Option<String>,
    /// The exact argv this implementation was invoked with, so the decode
    /// configuration (beam size, compute type, model) is part of the record
    /// rather than an assumption. `None` for in-process engines.
    #[serde(default)]
    pub command: Option<String>,
    /// Effective decode configuration, as sorted `key -> value` pairs.
    ///
    /// **The in-process counterpart of `command`, and it exists because its
    /// absence produced a real defect.** References record their argv, so a
    /// change in beam size or compute type is visible in the ledger. An
    /// in-process engine had no equivalent, so when speech segmentation
    /// became a default on 2026-07-28 the lines before and after described
    /// materially different engines under one name — `whisper-candle`, WER
    /// 7.99 then 6.79, with nothing in the record to say why. Two published
    /// READMEs and a crates.io release were cut from those numbers.
    ///
    /// A ledger line has to be sufficient on its own to reproduce the run.
    /// Anything that changes the output belongs here, not in the reader's
    /// memory of what the defaults were that week.
    #[serde(default)]
    pub config: std::collections::BTreeMap<String, String>,
    /// Mean word error rate over scored clips (ASR/OCR-word), if computed.
    pub wer: Option<f64>,
    /// Mean character error rate, if computed.
    pub cer: Option<f64>,
    /// Detection mAP@0.5 (proxy scorer, maxDets=100 — see
    /// `crate::detect`), if computed. Detect-task runs only; serde-defaulted
    /// and omitted when absent so non-detect lines are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub map50: Option<f64>,
    /// Detection mAP@0.5:0.95 (same proxy scorer), recorded beside `map50`
    /// so neither can be quoted alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub map5095: Option<f64>,
    /// **Warm** real-time factor: media seconds per second of processing-only
    /// time, model already loaded. The steady-state number implementations
    /// publish. Higher is faster.
    pub rtf_warm: Option<f64>,
    /// **End-to-end** real-time factor: media seconds per wall second for the
    /// whole corpus run, including one model load amortized over it. What a
    /// CLI user experiences. Recorded alongside `rtf_warm` so neither can be
    /// quoted alone.
    pub rtf_e2e: Option<f64>,
    /// Model load seconds, when the implementation reports it separately.
    pub load_secs: Option<f64>,
    /// Total best-of-N wall seconds over the holdout split.
    pub wall_secs: Option<f64>,
    /// Total media duration scored, for reproducing the RTF arithmetic.
    pub media_secs: Option<f64>,
    /// Clips successfully processed / clips attempted.
    pub clips_ok: usize,
    pub clips_total: usize,
    /// Peak resident memory in bytes over the run — the fourth gate's metric.
    ///
    /// For our engine this is our own process; for a reference it is that
    /// subprocess. `None` where the platform cannot measure it, which makes
    /// the gate skip rather than guess. Serde-defaulted so ledger lines
    /// written before this existed still parse.
    #[serde(default)]
    pub peak_bytes: Option<u64>,
    /// Median resident memory over the run — what the implementation SITS at
    /// while working, as opposed to its worst instant (usually model load).
    ///
    /// Both are recorded because they answer different questions and quoting
    /// either alone misleads: peak decides whether it fits, steady decides
    /// what it costs to keep running.
    #[serde(default)]
    pub steady_bytes: Option<u64>,
    /// Per-clip failures or notes, kept short.
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Reproduction context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Environment {
    pub os: String,
    pub arch: String,
    pub rustc: Option<String>,
    pub cpu: Option<String>,
}

impl Environment {
    #[must_use]
    pub fn capture() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            rustc: rustc_version(),
            cpu: std::env::var("PROCESSOR_IDENTIFIER").ok(),
        }
    }
}

fn rustc_version() -> Option<String> {
    let out = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .map(|s| s.trim().to_string())
}

/// One bench run: our engine vs the world standards on a pinned corpus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchRecord {
    pub schema: u32,
    /// Record id, e.g. "bench-asr-1753500000".
    pub id: String,
    /// Task: "asr", "tts", "ocr", "vlm".
    pub task: String,
    /// Fingerprint of the corpus manifest the run was measured on.
    pub corpus: String,
    pub corpus_manifest_hash: String,
    /// Our engine's run (None when only baselining references).
    pub engine: Option<RunSummary>,
    /// Reference runs.
    #[serde(default)]
    pub references: Vec<RunSummary>,
    pub gates: GateReport,
    pub environment: Environment,
    pub notes: String,
    /// Unix seconds at append time.
    pub appended_at: u64,
}

impl BenchRecord {
    #[must_use]
    pub fn now_id(task: &str) -> (String, u64) {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        (format!("bench-{task}-{secs}"), secs)
    }
}

/// Append one record as a single JSONL line (atomic single write).
pub fn append(path: &Path, record: &BenchRecord) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut line =
        serde_json::to_string(record).map_err(|e| Error::Other(format!("ledger encode: {e}")))?;
    line.push('\n');
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(line.as_bytes())?;
    Ok(())
}

/// Read every record. A missing file is an empty ledger, not an error.
pub fn read_all(path: &Path) -> Result<Vec<BenchRecord>> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| Error::Other(format!("ledger decode: {e}"))))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_through_jsonl() {
        let (id, secs) = BenchRecord::now_id("asr");
        let rec = BenchRecord {
            schema: LEDGER_SCHEMA,
            id,
            task: "asr".into(),
            corpus: "asr-smoke".into(),
            corpus_manifest_hash: "abc".into(),
            engine: None,
            references: vec![RunSummary {
                name: "whisper-cpp".into(),
                version: Some("v1.7".into()),
                command: Some("whisper-cli --batch {filelist}".into()),
                config: Default::default(),
                wer: Some(0.042),
                cer: None,
                map50: None,
                map5095: None,
                rtf_warm: Some(31.0),
                rtf_e2e: Some(18.0),
                load_secs: Some(1.9),
                wall_secs: Some(12.5),
                media_secs: Some(225.0),
                clips_ok: 10,
                clips_total: 10,
                notes: vec![],
                peak_bytes: Some(412 * 1024 * 1024),
                steady_bytes: Some(180 * 1024 * 1024),
            }],
            gates: GateReport::new(),
            environment: Environment::capture(),
            notes: "baseline only".into(),
            appended_at: secs,
        };
        let path = std::env::temp_dir().join("ffai_bench_ledger_test.jsonl");
        std::fs::remove_file(&path).ok();
        append(&path, &rec).unwrap();
        append(&path, &rec).unwrap();
        let all = read_all(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0], rec);
    }

    #[test]
    fn missing_ledger_is_empty() {
        let all = read_all(Path::new("definitely_missing_ledger.jsonl")).unwrap();
        assert!(all.is_empty());
    }
}
