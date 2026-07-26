//! The four-gate verdict, ported from Prometheus (`prom-core::gate`).
//! Nothing is claimed unless all four pass; a skipped gate is never a pass.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Which of the four gates a [`GateResult`] belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateKind {
    /// Engine completed every holdout clip with well-formed output.
    Correctness,
    /// Task metric vs reference on the holdout split (e.g. WER within the
    /// declared parity band of the best reference).
    Quality,
    /// Best-of-N wall-clock (real-time factor) vs reference.
    Speed,
    /// Peak memory / binary-size budget. (Instrumented in Phase 1 — until
    /// then always `Skipped`, which by design blocks `all_passed`.)
    Footprint,
}

impl GateKind {
    pub const ALL: [GateKind; 4] = [
        GateKind::Correctness,
        GateKind::Quality,
        GateKind::Speed,
        GateKind::Footprint,
    ];

    pub fn label(self) -> &'static str {
        match self {
            GateKind::Correctness => "correctness",
            GateKind::Quality => "quality",
            GateKind::Speed => "speed",
            GateKind::Footprint => "footprint",
        }
    }
}

/// Pass / fail / not-run. `Skipped` carries a reason and is never a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateOutcome {
    Pass,
    Fail,
    Skipped,
}

impl fmt::Display for GateOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            GateOutcome::Pass => "PASS",
            GateOutcome::Fail => "FAIL",
            GateOutcome::Skipped => "SKIP",
        })
    }
}

/// One gate's result with a human-readable detail line and optional headline
/// metric (WER %, RTF ×, MB).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateResult {
    pub kind: GateKind,
    pub outcome: GateOutcome,
    pub metric: Option<f64>,
    pub detail: String,
}

impl GateResult {
    pub fn skipped(kind: GateKind, reason: impl Into<String>) -> Self {
        GateResult { kind, outcome: GateOutcome::Skipped, metric: None, detail: reason.into() }
    }
}

/// The full four-gate verdict for one bench run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateReport {
    pub results: Vec<GateResult>,
}

impl GateReport {
    /// Fresh report: all four gates `Skipped("not run")`.
    pub fn new() -> Self {
        GateReport {
            results: GateKind::ALL
                .iter()
                .map(|&k| GateResult::skipped(k, "not run"))
                .collect(),
        }
    }

    pub fn set(&mut self, result: GateResult) {
        if let Some(slot) = self.results.iter_mut().find(|r| r.kind == result.kind) {
            *slot = result;
        } else {
            self.results.push(result);
        }
    }

    pub fn get(&self, kind: GateKind) -> Option<&GateResult> {
        self.results.iter().find(|r| r.kind == kind)
    }

    /// True only if **every** gate is `Pass` — a skipped gate has not cleared
    /// the bar. This is the rule that keeps public claims honest.
    pub fn all_passed(&self) -> bool {
        GateKind::ALL.iter().all(|&k| {
            self.get(k).map(|r| r.outcome == GateOutcome::Pass).unwrap_or(false)
        })
    }
}

impl Default for GateReport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skipped_is_not_passed() {
        assert!(!GateReport::new().all_passed());
    }

    #[test]
    fn all_four_required() {
        let mut r = GateReport::new();
        for k in GateKind::ALL {
            assert!(!r.all_passed());
            r.set(GateResult { kind: k, outcome: GateOutcome::Pass, metric: None, detail: "ok".into() });
        }
        assert!(r.all_passed());
        r.set(GateResult {
            kind: GateKind::Speed,
            outcome: GateOutcome::Fail,
            metric: Some(0.4),
            detail: "0.4x of reference".into(),
        });
        assert!(!r.all_passed());
    }
}
