//! Great Gate feature tap — W1 of `docs/plan/benching-history-made.md` §11.
//!
//! Writes one CSV row per page of the signals the ordering decision actually
//! sees, so `_greatgate/gate-calculator` can search rules offline against them.
//! The economics are the whole point: an engine pass costs 236 pages x ~9 s
//! plus the official harness on top, while rule search over a CSV is free. So
//! this emits a WIDE set once rather than a narrow set per lever — the
//! calculator discovers features from the header and cannot search for a signal
//! that is not in the file, and every omitted column costs a full engine pass
//! to add later.
//!
//! ## Inert unless asked
//!
//! `FFAI_GATE_HARVEST=<path>` enables it. Unset, the cost is one cached bool
//! load and no allocation — `harvest_inert_when_unset` asserts the output is
//! byte-identical either way. This ships in the default build because a tap
//! that must be compiled in specially is a tap nobody runs.
//!
//! ## It emits for EVERY page, including the ones that decide nothing
//!
//! `probe_apply` returns early on three guards (too few lines, too few columns,
//! too little coverage) and again if the probe order is not a permutation.
//! Tapping only at the decision point would harvest the population that reached
//! it — a silently narrowed denominator, which §3 rule 10 exists to prevent, and
//! which would make every rate in the CSV a rate over the wrong set. So the row
//! is emitted on ALL paths and carries `reached_probe` / `probe_valid` flags;
//! a rule that only applies to deciding pages filters on those columns HERE,
//! where the filtering is visible, rather than in the harvest.
//!
//! ## One page per process
//!
//! `ocr_text.exe` handles a single image, so rows append one per invocation and
//! the harness sets `FFAI_GATE_PAGE` to the page's stem. The header is written
//! only when the file is empty. This is safe because the harness runs pages
//! sequentially; it is NOT safe under concurrent writers, and the builder in
//! `.tools-bench/gg_harvest.py` keeps it that way.
// Both `Write` traits are needed and they collide by name: `write!` into a
// String resolves through `fmt::Write`, `write_all` into a File through
// `io::Write`. Importing one anonymously keeps both in scope.
use std::fmt::Write as _;
use std::io::Write;
use std::sync::OnceLock;

/// The harvest path, resolved once. `None` = disabled.
fn path() -> Option<&'static str> {
    static P: OnceLock<Option<String>> = OnceLock::new();
    P.get_or_init(|| std::env::var("FFAI_GATE_HARVEST").ok().filter(|s| !s.is_empty()))
        .as_deref()
}

/// True when the tap is armed. One cached load on the shipped path.
#[inline]
pub fn enabled() -> bool {
    path().is_some()
}

/// One page's feature row. Column ORDER is insertion order and must be stable
/// across pages within a run, which it is because the tap fills the same fields
/// in the same sequence on every path.
pub struct Row {
    cols: Vec<(&'static str, f64)>,
}

impl Row {
    pub fn new() -> Self {
        Row { cols: Vec::with_capacity(32) }
    }

    /// Record a feature. NaN and infinities are written as empty cells rather
    /// than as `NaN`, which the calculator's parser would read as a feature
    /// value of zero and silently fit a rule to a missing measurement.
    pub fn put(&mut self, name: &'static str, v: f64) {
        self.cols.push((name, v));
    }

    pub fn put_bool(&mut self, name: &'static str, v: bool) {
        self.put(name, if v { 1.0 } else { 0.0 });
    }

    /// Column count, for the row-width invariant test.
    pub fn len(&self) -> usize {
        self.cols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cols.is_empty()
    }

    /// Emit `names` as MISSING (empty cells).
    ///
    /// Every row must carry every column. A page that returns early from the
    /// ordering decision never reaches the columns after its exit, and without
    /// this it would emit a SHORT row — ragged CSV, and worse: the header is
    /// written from whichever row lands first, so one early-returning page at
    /// the head of a run silently misaligns every full row after it, and the
    /// calculator would then fit rules to columns shifted under their names.
    /// Empty is also the honest value: the signal was not computed, which is
    /// different from computed-as-zero.
    pub fn missing(&mut self, names: &[&'static str]) {
        for n in names {
            self.put(n, f64::NAN);
        }
    }

    /// Append the row, writing the header first if the file is empty.
    pub fn emit(self) {
        let Some(p) = path() else { return };
        let page = std::env::var("FFAI_GATE_PAGE").unwrap_or_else(|_| "unknown".into());
        let mut line = String::with_capacity(256);
        line.push_str(&page.replace(',', "_"));
        for (_, v) in &self.cols {
            line.push(',');
            if v.is_finite() {
                let _ = write!(line, "{v:.6}");
            }
        }
        line.push('\n');

        let need_header = std::fs::metadata(p).map(|m| m.len() == 0).unwrap_or(true);
        let mut f = match std::fs::OpenOptions::new().create(true).append(true).open(p) {
            Ok(f) => f,
            // A tap that panics would take the engine down over telemetry.
            Err(_) => return,
        };
        if need_header {
            let mut h = String::from("page");
            for (n, _) in &self.cols {
                h.push(',');
                h.push_str(n);
            }
            h.push('\n');
            let _ = f.write_all(h.as_bytes());
        }
        let _ = f.write_all(line.as_bytes());
    }
}

impl Default for Row {
    fn default() -> Self {
        Self::new()
    }
}
