//! Stage timing for the detection path — M-D2.0's spine.
//!
//! Same shape as Carmenta's `profile` and Mercury's `asr::profile`: enable
//! with `FFAI_PROFILE=1`, disabled cost is one relaxed atomic load per stage,
//! so it stays compiled into release builds.
//!
//! Two additions the analyzer discipline requires and the OCR version does
//! not have:
//!
//! - **A `total` scope wrapping the whole detect**, so the RESIDUE
//!   (`total − Σ stages`) is computable. The residue is the most important
//!   line in the report: it is where frame management, allocation and
//!   framework glue hide, and "the residue is irreducible" almost always
//!   means "I stopped scoping too early."
//! - **An overhead estimate**, because a residue that equals
//!   `Σ calls × timer cost` is the profiler measuring itself rather than
//!   hidden work. The report computes it and says so, so nobody optimizes a
//!   ghost.
//!
//! Stages are deliberately coarse at first. Decompose only where the numbers
//! point — a fine scope at a high call count inflates its own bucket.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("FFAI_PROFILE").is_some())
}

#[derive(Debug, Default)]
pub struct Stage {
    nanos: AtomicU64,
    calls: AtomicU64,
}

impl Stage {
    const fn new() -> Self {
        Stage { nanos: AtomicU64::new(0), calls: AtomicU64::new(0) }
    }

    fn add(&self, nanos: u64) {
        self.nanos.fetch_add(nanos, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    pub fn secs(&self) -> f64 {
        self.nanos.load(Ordering::Relaxed) as f64 / 1e9
    }

    pub fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.nanos.store(0, Ordering::Relaxed);
        self.calls.store(0, Ordering::Relaxed);
    }
}

/// The stages of one image's detection.
pub struct Profile {
    /// Letterbox + normalize + tensor build.
    pub pre: Stage,
    /// Layers 0–10.
    pub backbone: Stage,
    /// Layers 11–22.
    pub neck: Stage,
    /// The one2one box + class branches.
    pub head: Stage,
    /// Anchors, dist2bbox, sigmoid, two-stage top-k.
    pub decode: Stage,
    /// The whole `detect()` — the denominator, and the source of the residue.
    pub total: Stage,

    // ---- second tier: INFO ONLY, excluded from every sum ----
    // These nest inside the stages above, so counting them in the total
    // would double-count. They exist to answer "what inside the backbone?"
    // and should be removed once read — at high call counts a scope's own
    // overhead swamps what it measures.
    /// Every fused convolution (info tier).
    pub conv: Stage,
    /// The 1x1 subset of `conv` — a matmul wearing a convolution's costume.
    pub conv1x1: Stage,
    /// The 3x3 dense stride-1 subset of `conv`.
    pub conv3x3: Stage,
    /// The 3x3 dense STRIDE-2 downsamples, still on candle's path.
    pub conv_s2: Stage,
    /// Depthwise, on our own kernel.
    pub conv_dw: Stage,
    /// The SiLU activation applied after most convolutions.
    pub act: Stage,
    /// im2col materialization inside our 3x3 kernels.
    pub im2col: Stage,
    /// The GEMM inside our 3x3 kernels.
    pub gemm: Stage,
    /// Every attention block (info tier).
    pub attn: Stage,
}

static PROFILE: Profile = Profile {
    pre: Stage::new(),
    backbone: Stage::new(),
    neck: Stage::new(),
    head: Stage::new(),
    decode: Stage::new(),
    total: Stage::new(),
    conv: Stage::new(),
    conv1x1: Stage::new(),
    conv3x3: Stage::new(),
    conv_s2: Stage::new(),
    conv_dw: Stage::new(),
    act: Stage::new(),
    im2col: Stage::new(),
    gemm: Stage::new(),
    attn: Stage::new(),
};

pub fn profile() -> &'static Profile {
    &PROFILE
}

pub fn is_enabled() -> bool {
    enabled()
}

pub fn reset() {
    for s in [
        &PROFILE.pre,
        &PROFILE.backbone,
        &PROFILE.neck,
        &PROFILE.head,
        &PROFILE.decode,
        &PROFILE.total,
        &PROFILE.conv,
        &PROFILE.conv1x1,
        &PROFILE.conv3x3,
        &PROFILE.conv_s2,
        &PROFILE.conv_dw,
        &PROFILE.act,
        &PROFILE.im2col,
        &PROFILE.gemm,
        &PROFILE.attn,
    ] {
        s.reset();
    }
}

/// Time `f` into `stage` when profiling is on; run it untimed otherwise.
pub fn timed<T>(stage: fn(&Profile) -> &Stage, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let t0 = Instant::now();
    let out = f();
    stage(&PROFILE).add(t0.elapsed().as_nanos() as u64);
    out
}

/// Measured cost of one `timed` scope, for the overhead line in the report.
fn scope_nanos() -> f64 {
    static COST: OnceLock<f64> = OnceLock::new();
    *COST.get_or_init(|| {
        let n = 200_000;
        let t0 = Instant::now();
        for _ in 0..n {
            let inner = Instant::now();
            std::hint::black_box(inner.elapsed());
        }
        t0.elapsed().as_nanos() as f64 / n as f64
    })
}

impl Profile {
    pub fn report(&self) -> String {
        let rows: [(&str, &Stage); 5] = [
            ("pre", &self.pre),
            ("backbone", &self.backbone),
            ("neck", &self.neck),
            ("head", &self.head),
            ("decode", &self.decode),
        ];
        let named: f64 = rows.iter().map(|(_, s)| s.secs()).sum();
        let total = self.total.secs();
        let images = self.total.calls().max(1);

        let mut out = String::from("\ndetect stage profile (FFAI_PROFILE=1)\n");
        out.push_str(&format!(
            "{:<10} {:>9} {:>7} {:>9} {:>10}\n",
            "STAGE", "SECONDS", "SHARE", "CALLS", "MS/IMAGE"
        ));
        for (name, s) in rows {
            let secs = s.secs();
            out.push_str(&format!(
                "{:<10} {:>9.3} {:>6.1}% {:>9} {:>10.2}\n",
                name,
                secs,
                if total > 0.0 { secs / total * 100.0 } else { 0.0 },
                s.calls(),
                secs * 1000.0 / images as f64,
            ));
        }

        // The residue, and whether it is real work or the instrument.
        let residue = total - named;
        let scope_calls: u64 = rows.iter().map(|(_, s)| s.calls()).sum::<u64>() + self.total.calls();
        let overhead = scope_calls as f64 * scope_nanos() / 1e9;
        out.push_str(&format!(
            "{:<10} {:>9.3} {:>6.1}% {:>9} {:>10.2}\n",
            "residue",
            residue,
            if total > 0.0 { residue / total * 100.0 } else { 0.0 },
            "",
            residue * 1000.0 / images as f64,
        ));
        out.push_str(&format!("{:<10} {total:>9.3} {:>6.1}%\n", "TOTAL", 100.0));
        out.push_str(&format!(
            "\ntimer overhead ~{:.3} s ({} scope entries x {:.0} ns)\n",
            overhead,
            scope_calls,
            scope_nanos()
        ));
        if residue > 0.0 && overhead > residue * 0.5 {
            out.push_str(
                "  NOTE: overhead is a large share of the residue — the residue is the\n  \
                 instrument measuring itself, not hidden work. Stop decomposing.\n",
            );
        }

        let info: [(&str, &Stage); 9] = [
            ("conv", &self.conv),
            ("  1x1", &self.conv1x1),
            ("  3x3 s1", &self.conv3x3),
            ("  3x3 s2", &self.conv_s2),
            ("  depthw", &self.conv_dw),
            ("  silu", &self.act),
            ("   >im2col", &self.im2col),
            ("   >gemm", &self.gemm),
            ("attn", &self.attn),
        ];
        if info.iter().any(|(_, s)| s.calls() > 0) {
            out.push_str("\ninfo tier (NESTED — excluded from the sums above)\n");
            for (name, s) in info {
                let secs = s.secs();
                out.push_str(&format!(
                    "{:<10} {:>9.3} {:>6.1}% {:>9} {:>10.2}\n",
                    name,
                    secs,
                    if total > 0.0 { secs / total * 100.0 } else { 0.0 },
                    s.calls(),
                    if s.calls() > 0 { secs * 1e6 / s.calls() as f64 } else { 0.0 },
                ));
            }
            out.push_str("  (last column is us/call for the info tier)\n");
        }
        out
    }
}
