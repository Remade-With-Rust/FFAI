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
    /// The WRAPPER around a convolution: bias broadcast-add and the final
    /// reshape. Split out because `1x1` and `3x3` are PARENT buckets that
    /// contain im2col + gemm, so everything else inside them was invisible —
    /// and the arithmetic said ~50 ms/image lived there unaccounted. A
    /// residue you cannot name is not a residue, it is a hiding place.
    pub conv_wrap: Stage,
    /// The `SliceOp`/`CustomOp1` round trip: candle tensor in, Vec out.
    pub sliceop: Stage,
    /// Every attention block (info tier).
    pub attn: Stage,
    /// The two `transpose(...).contiguous()` calls inside attention.
    /// Bucketed because candle's generic transpose measured 4.3x slower
    /// than a blocked loop, and attention is the ONE op in the profile that
    /// does not parallelise at all (1.02x from 1 to 24 threads).
    pub attn_t: Stage,
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
    attn_t: Stage::new(),
    conv_wrap: Stage::new(),
    sliceop: Stage::new(),
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
        &PROFILE.attn_t,
        &PROFILE.conv_wrap,
        &PROFILE.sliceop,
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

        let info: [(&str, &Stage); 12] = [
            ("conv", &self.conv),
            ("  1x1", &self.conv1x1),
            ("  3x3 s1", &self.conv3x3),
            ("  3x3 s2", &self.conv_s2),
            ("  depthw", &self.conv_dw),
            ("  silu", &self.act),
            ("   >im2col", &self.im2col),
            ("   >attn_t", &self.attn_t),
            ("   >wrap", &self.conv_wrap),
            ("   >sliceop", &self.sliceop),
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

// ---------------------------------------------------------------------------
// Per-layer roofline — `FFAI_DIANA_ROOFLINE=1`
//
// The stage buckets say "3x3 stride-2 is 22.3 % of detect in 7 calls per
// image". They cannot say whether that is a lot, because a call's cost is only
// interpretable against the work it does. 22.3 % of time for 22 % of the FLOPs
// is proportional and uninteresting; 22.3 % for 10 % is a target.
//
// So this keys on the actual SHAPE — in and out channels, in and out spatial,
// kernel, stride — accumulates calls and nanos per distinct shape, and divides
// by the arithmetic that shape implies. What comes out is effective GFLOP/s per
// layer, which is the number that ranks candidates rather than describing them.
//
// The FLOP count is the honest 2*K*K*Cin*Cout*Hout*Wout for dense convolution
// and 2*K*K*Cin*Hout*Wout for depthwise — multiply-add counted as two, no
// credit for anything an implementation might skip. It is a lower bound on
// useful work, so effective GFLOP/s is an UNDER-estimate of efficiency and a
// layer that looks bad here is at worst as bad as it looks.
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::Mutex;

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct ConvShape {
    pub kind: &'static str,
    pub cin: usize,
    pub cout: usize,
    pub hin: usize,
    pub win: usize,
    pub hout: usize,
    pub wout: usize,
    pub k: usize,
    pub depthwise: bool,
}

impl ConvShape {
    /// Multiply-adds counted as two flops. Depthwise does one filter per
    /// channel rather than Cin x Cout, which is exactly why it is cheap and
    /// why counting it like a dense conv would make it look 64x more
    /// efficient than it is.
    fn flops(&self) -> f64 {
        let spatial = (self.hout * self.wout) as f64;
        let taps = (self.k * self.k) as f64;
        if self.depthwise {
            2.0 * taps * self.cin as f64 * spatial
        } else {
            2.0 * taps * self.cin as f64 * self.cout as f64 * spatial
        }
    }

    /// Bytes that must move at minimum: read the input, write the output,
    /// read the weights once. Used for arithmetic intensity, which is what
    /// says whether a slow layer is starved or just badly scheduled.
    fn bytes(&self) -> f64 {
        let f32b = 4.0;
        let w = if self.depthwise {
            (self.k * self.k * self.cin) as f64
        } else {
            (self.k * self.k * self.cin * self.cout) as f64
        };
        f32b * ((self.cin * self.hin * self.win) as f64
            + (self.cout * self.hout * self.wout) as f64
            + w)
    }
}

static ROOFLINE: Mutex<Option<HashMap<ConvShape, (u64, u64)>>> = Mutex::new(None);

/// Convolution shapes in EXECUTION ORDER for the first image.
///
/// The per-shape roofline can say which layout each shape prefers. It cannot
/// say what that costs, because a layout choice is only free when the
/// PREVIOUS layer left the activation in that layout. Order is the missing
/// half, and no aggregate keyed by shape can recover it.
static ORDER: Mutex<Option<Vec<ConvShape>>> = Mutex::new(None);

pub fn record_order(shape: ConvShape) {
    if !roofline_enabled() {
        return;
    }
    let mut g = ORDER.lock().unwrap();
    let v = g.get_or_insert_with(Vec::new);
    // First image only: the sequence repeats, and recording every image would
    // just concatenate copies of it.
    if v.len() < 512 {
        v.push(shape);
    }
}

/// One line per convolution, in order, so a driver can solve the layout
/// assignment over the real sequence instead of assuming one.
pub fn order_report() -> String {
    let g = ORDER.lock().unwrap();
    let Some(v) = g.as_ref() else {
        return String::new();
    };
    let mut out = String::from("
conv execution order (first image)
idx kind cin cout hin win hout wout
");
    for (i, s) in v.iter().enumerate() {
        out.push_str(&format!(
            "{i} {} {} {} {} {} {} {}
",
            s.kind.replace(' ', "_"),
            s.cin, s.cout, s.hin, s.win, s.hout, s.wout
        ));
    }
    out
}

pub fn roofline_enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var_os("FFAI_DIANA_ROOFLINE").is_some())
}

pub fn record_conv(shape: ConvShape, nanos: u64) {
    if !roofline_enabled() {
        return;
    }
    let mut g = ROOFLINE.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    let e = m.entry(shape).or_insert((0, 0));
    e.0 += 1;
    e.1 += nanos;
}

/// Ranked by total time, because that is the order in which work is worth
/// doing. `GFLOP/s` is the column that says whether a row is a target.
pub fn roofline_report(images: u64) -> String {
    let g = ROOFLINE.lock().unwrap();
    let Some(m) = g.as_ref() else {
        return "roofline: no data (set FFAI_DIANA_ROOFLINE=1)".into();
    };
    let mut rows: Vec<(&ConvShape, u64, u64)> = m.iter().map(|(s, (c, n))| (s, *c, *n)).collect();
    rows.sort_by_key(|(_, _, n)| std::cmp::Reverse(*n));

    let total_ns: u64 = rows.iter().map(|(_, _, n)| *n).sum();
    let total_flops: f64 = rows.iter().map(|(s, c, _)| s.flops() * *c as f64).sum();

    let mut out = String::from(
        "\nper-layer roofline (FFAI_DIANA_ROOFLINE=1)\n\
         kind       cin->cout   in HxW     out HxW  calls   ms/img  share  GFLOP/s  AI(f/B)\n",
    );
    // The frame count comes from the CALLER. Deriving it from the max call
    // count was wrong by exactly the number of times the busiest shape repeats
    // within one image (12x here), and it announced itself as conv totalling
    // 1.69 ms/img while im2col NESTED INSIDE IT read 2.774 — a parent cheaper
    // than its child. Shares and GFLOP/s were unaffected; they never used it.
    let imgs = images.max(1);
    for (s, calls, ns) in &rows {
        let secs = *ns as f64 / 1e9;
        let gflops = (s.flops() * *calls as f64 / 1e9) / secs.max(1e-12);
        let ai = s.flops() / s.bytes();
        out.push_str(&format!(
            "{:<9} {:>4}->{:<5} {:>5}x{:<5} {:>5}x{:<5} {:>5} {:>8.2} {:>5.1}% {:>8.1} {:>8.1}\n",
            s.kind,
            s.cin,
            s.cout,
            s.hin,
            s.win,
            s.hout,
            s.wout,
            calls,
            secs * 1000.0 / (imgs as f64),
            100.0 * *ns as f64 / total_ns as f64,
            gflops,
            ai,
        ));
    }
    out.push_str(&format!(
        "\ntotal {:.2} ms/img over {} distinct shapes; {:.2} GFLOP/img; aggregate {:.1} GFLOP/s\n",
        total_ns as f64 / 1e6 / imgs as f64,
        rows.len(),
        total_flops / 1e9 / imgs as f64,
        (total_flops / 1e9) / (total_ns as f64 / 1e9),
    ));
    out
}

// ---------------------------------------------------------------------------
// SliceOp accounting — WRAPPER vs WORK, per op name
//
// `p.sliceop` times `apply_op1_no_bwd`, which CONTAINS the closure it is
// dispatching to. So its 15.8 % nests im2col's 11.4 % and cannot be read as
// "15.8 % of detect is plumbing" — that would double-count the arithmetic and
// send someone to optimise a bucket that is mostly real work.
//
// This splits it. `total` is the wrapper as seen from `run()`; `work` is the
// closure alone, timed inside `cpu_fwd`. The difference is what candle's
// custom-op dispatch costs us per call: allocation of the output storage,
// the Shape, the dyn dispatch, and whatever the tensor layer does around it.
//
// That difference is the only part that could be a win, because the closure is
// the arithmetic and is not going anywhere.
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_WORK_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub fn set_last_work_ns(n: u64) {
    LAST_WORK_NS.with(|c| c.set(n));
}

pub fn take_last_work_ns() -> u64 {
    LAST_WORK_NS.with(|c| c.replace(0))
}

/// name -> (calls, total nanos incl. wrapper, closure nanos)
static SLICEOPS: Mutex<Option<HashMap<&'static str, (u64, u64, u64)>>> = Mutex::new(None);

pub fn record_sliceop(name: &'static str, total_ns: u64, work_ns: u64) {
    if !roofline_enabled() {
        return;
    }
    let mut g = SLICEOPS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    let e = m.entry(name).or_insert((0, 0, 0));
    e.0 += 1;
    e.1 += total_ns;
    e.2 += work_ns;
}

pub fn sliceop_report(images: u64) -> String {
    let g = SLICEOPS.lock().unwrap();
    let Some(m) = g.as_ref() else {
        return "sliceop: no data (set FFAI_DIANA_ROOFLINE=1)".into();
    };
    let mut rows: Vec<(&&str, u64, u64, u64)> =
        m.iter().map(|(n, (c, t, w))| (n, *c, *t, *w)).collect();
    rows.sort_by_key(|(_, _, t, _)| std::cmp::Reverse(*t));
    let imgs = images.max(1) as f64;

    let mut out = String::from(
        "\nSliceOp: wrapper vs work (FFAI_DIANA_ROOFLINE=1)\n\
         op                        calls/img  total ms/img   work ms/img  WRAPPER ms/img  wrapper%\n",
    );
    let (mut tt, mut tw) = (0u64, 0u64);
    for (name, calls, total, work) in &rows {
        tt += *total;
        tw += *work;
        let over = total.saturating_sub(*work);
        out.push_str(&format!(
            "{:<26} {:>9.1} {:>13.3} {:>13.3} {:>15.3} {:>9.1}%\n",
            name,
            *calls as f64 / imgs,
            *total as f64 / 1e6 / imgs,
            *work as f64 / 1e6 / imgs,
            over as f64 / 1e6 / imgs,
            100.0 * over as f64 / (*total as f64).max(1.0),
        ));
    }
    let over = tt.saturating_sub(tw);
    out.push_str(&format!(
        "\nTOTAL {:.3} ms/img, of which work {:.3} and WRAPPER {:.3} ({:.1}%)\n\
         the wrapper is the only part that could be removed; the closure is the arithmetic\n",
        tt as f64 / 1e6 / imgs,
        tw as f64 / 1e6 / imgs,
        over as f64 / 1e6 / imgs,
        100.0 * over as f64 / (tt as f64).max(1.0),
    ));
    out
}

// ---------------------------------------------------------------------------
// Denormal census — `FFAI_DIANA_ROOFLINE=1`
//
// `silu.rs` carries a HYPOTHESIS that has never been measured: replacing SiLU
// with the identity made the pipeline SLOWER (49.7 ms against 45.1), and the
// explanation offered was that unbounded activations land in denormals, which
// are orders of magnitude slower on x86. That is plausible and it is the exact
// shape of a story that is never checked because the toggle was deleted.
//
// Denormals are also the one genuinely CONTENT-dependent effect available in a
// convolution graph: shapes are static, arithmetic is dense, and the only thing
// an image can change is the VALUES. So if a content-adaptive dispatch exists
// anywhere in this engine, this is where.
//
// A count settles it in one run and needs no quiet box: how many activation
// values are subnormal, per image. Zero kills the hypothesis outright; a large
// and image-varying count makes it a dispatch, and a large and constant count
// makes it a global flush-to-zero flag.
// ---------------------------------------------------------------------------

static DENORM: (AtomicU64, AtomicU64, AtomicU64) =
    (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));

/// Census one activation buffer: total values, subnormals, exact zeros.
///
/// Exact zeros are counted separately because they are FAST — they are not
/// denormals and a flush-to-zero flag would not change them. Lumping them in
/// would inflate the apparent prize by whatever fraction of a post-SiLU
/// activation is hard zero, which is large.
pub fn census(xs: &[f32]) {
    if !roofline_enabled() {
        return;
    }
    let mut sub = 0u64;
    let mut zero = 0u64;
    for &v in xs {
        if v == 0.0 {
            zero += 1;
        } else if v.abs() < f32::MIN_POSITIVE {
            sub += 1;
        }
    }
    DENORM.0.fetch_add(xs.len() as u64, Ordering::Relaxed);
    DENORM.1.fetch_add(sub, Ordering::Relaxed);
    DENORM.2.fetch_add(zero, Ordering::Relaxed);
}

pub fn denorm_report(images: u64) -> String {
    let (t, s, z) = (
        DENORM.0.load(Ordering::Relaxed),
        DENORM.1.load(Ordering::Relaxed),
        DENORM.2.load(Ordering::Relaxed),
    );
    if t == 0 {
        return String::new();
    }
    let imgs = images.max(1);
    format!(
        "\ndenormal census over {} images\n  \
         values {:>14}  ({:.1}M/image)\n  \
         SUBNORMAL {:>11}  ({:.6}%)  <- the only ones a flush-to-zero flag changes\n  \
         exact zero {:>10}  ({:.2}%)  <- fast already, not denormals\n",
        imgs,
        t,
        t as f64 / imgs as f64 / 1e6,
        s,
        100.0 * s as f64 / t as f64,
        z,
        100.0 * z as f64 / t as f64,
    )
}
