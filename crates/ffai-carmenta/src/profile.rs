//! Stage timing for the OCR path — the same discipline as Mercury's
//! `asr::profile`: measure before touching anything. The speed gate reads
//! 15-25× behind Tesseract at bring-up, and that gap has several plausible
//! causes (full-resolution CRAFT forward, per-line CRNN calls, the CPU
//! upsample, crop preprocessing); optimizing the wrong one is wasted effort
//! that still adds complexity.
//!
//! Enable with `FFAI_PROFILE=1`. Disabled overhead is one relaxed atomic
//! load per stage, so it stays compiled into release builds.

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

    pub(crate) fn add(&self, nanos: u64) {
        self.nanos.fetch_add(nanos, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    pub fn secs(&self) -> f64 {
        self.nanos.load(Ordering::Relaxed) as f64 / 1e9
    }

    pub fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

/// The stages of one page/frame recognition.
pub struct Profile {
    /// Grayscale conversion + CRAFT canvas preparation.
    pub det_pre: Stage,
    /// The CRAFT forward pass.
    pub det_fwd: Stage,
    /// Map thresholding, connected components, line grouping.
    pub boxes: Stage,
    /// Line crop + bicubic resize + normalize (per line).
    pub rec_pre: Stage,
    /// The CRNN forward pass (per line).
    pub rec_fwd: Stage,
    /// CTC decode (per line).
    pub decode: Stage,
    /// Inside `rec_fwd`: the CRNN's 7-conv backbone (per line).
    pub rec_cnn: Stage,
    /// Inside `rec_fwd`: the two BiLSTMs — candle's `LSTM::seq`, which walks
    /// timesteps SEQUENTIALLY at batch 1, so every gate matmul takes the
    /// m=1 vector path. Split out to test that hypothesis (§8.100).
    pub rec_rnn: Stage,
    /// Inside `rec_fwd`: the final vocabulary projection (per line).
    pub rec_head: Stage,
}

static PROFILE: Profile = Profile {
    det_pre: Stage::new(),
    det_fwd: Stage::new(),
    boxes: Stage::new(),
    rec_pre: Stage::new(),
    rec_fwd: Stage::new(),
    decode: Stage::new(),
    rec_cnn: Stage::new(),
    rec_rnn: Stage::new(),
    rec_head: Stage::new(),
};

pub fn profile() -> &'static Profile {
    &PROFILE
}

pub fn is_enabled() -> bool {
    enabled()
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

impl Profile {
    pub fn report(&self) -> String {
        let rows: [(&str, &Stage); 9] = [
            ("det_pre", &self.det_pre),
            ("det_fwd", &self.det_fwd),
            ("boxes", &self.boxes),
            ("rec_pre", &self.rec_pre),
            ("rec_fwd", &self.rec_fwd),
            ("decode", &self.decode),
            ("  .rec_cnn", &self.rec_cnn),
            ("  .rec_rnn", &self.rec_rnn),
            ("  .rec_head", &self.rec_head),
        ];
        // The `.rec_*` rows decompose rec_fwd and are already counted in it;
        // summing them into the total would double-count the same nanoseconds.
        let total: f64 = rows
            .iter()
            .filter(|(n, _)| !n.starts_with("  ."))
            .map(|(_, s)| s.secs())
            .sum();
        let mut out = String::from("\nOCR stage profile (FFAI_PROFILE=1)\n");
        out.push_str(&format!(
            "{:<10} {:>9} {:>7} {:>8} {:>10}\n",
            "STAGE", "SECONDS", "SHARE", "CALLS", "MS/CALL"
        ));
        for (name, s) in rows {
            let secs = s.secs();
            let calls = s.calls();
            out.push_str(&format!(
                "{:<10} {:>9.3} {:>6.1}% {:>8} {:>10.2}\n",
                name,
                secs,
                if total > 0.0 { secs / total * 100.0 } else { 0.0 },
                calls,
                if calls > 0 { secs * 1000.0 / calls as f64 } else { 0.0 },
            ));
        }
        out.push_str(&format!("{:<10} {total:>9.3}\n", "total"));
        out
    }
}
