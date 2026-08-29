//! Stage timing for the ASR path.
//!
//! M2 is a performance milestone, and the first rule of performance work is
//! that you measure before you touch anything (`codec-optimize`: profile
//! first, always). A 4× gap has several plausible causes — f32 vs int8, an
//! O(n²) decode loop, unfused matmuls, single-threaded execution — and
//! optimizing the wrong one is wasted effort that still adds complexity.
//!
//! Enable with `FFAI_PROFILE=1`. Overhead when disabled is one relaxed atomic
//! load per stage, so this can stay compiled into release builds.

//! Cast policy (gate H-15): `cast_possible_truncation`, `cast_sign_loss` and
//! `cast_possible_wrap` are allowed in this module. Every value converted here
//! is a MODEL-INTERNAL dimension, index or accumulator - bounded by weights the
//! loader has already validated - not a number read from caller input. The lint
//! stays DENIED in the untrusted-surface modules (`mel`, `fbank`, `onnx`,
//! `normalize`, `lexicon`, `chunk`, `phonemize`, `phoneme_ids`), which is where
//! this audit's arithmetic defects were actually found.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use crate::clock::Instant;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("FFAI_PROFILE").is_some())
}

/// One stage's accumulated cost.
#[derive(Debug, Default)]
pub struct Stage {
    nanos: AtomicU64,
    calls: AtomicU64,
}

impl Stage {
    const fn new() -> Self {
        Self {
            nanos: AtomicU64::new(0),
            calls: AtomicU64::new(0),
        }
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
}

/// The stages of one transcription.
pub struct Profile {
    /// PCM → log-mel spectrogram.
    pub mel: Stage,
    /// The audio encoder forward pass — once per 30 s window.
    pub encoder: Stage,
    /// The decoder forward pass — once per generated token.
    pub decoder: Stage,
    /// Logit filtering + argmax, per token.
    pub sampling: Stage,

    // ---- decoder internals (op-level, Mercury's own decoder only) ----
    /// Token + positional embedding lookup.
    pub dec_embed: Stage,
    /// Self-attention across all layers, including cache append.
    pub dec_self_attn: Stage,
    /// Cross-attention across all layers.
    pub dec_cross_attn: Stage,
    /// Feed-forward across all layers.
    pub dec_mlp: Stage,
    /// The final projection to the ~51k vocabulary.
    pub dec_final: Stage,

    // ---- encoder internals (op-level, Mercury's own encoder only) ----
    /// The two convolutional front-end layers.
    pub enc_conv: Stage,
    /// Bidirectional self-attention across all encoder layers.
    pub enc_attn: Stage,
    /// Feed-forward across all encoder layers.
    pub enc_mlp: Stage,

    // ---- cross-attention internals (in-context, not microbenched) ----
    pub xa_qproj: Stage,
    /// Encoder-attention sub-ops. The stage had a 16.6 ms residue (20 % of
    /// itself) that standalone probes could only guess at; these name it in
    /// context, which is the measurement that wins when the two disagree.
    pub em_ln: Stage,
    pub em_fc1: Stage,
    pub em_gelu: Stage,
    pub em_fc2: Stage,
    pub ea_proj: Stage,
    pub ea_prep: Stage,
    pub ea_kernel: Stage,
    pub ea_merge: Stage,
    pub xa_prep: Stage,
    pub xa_qk: Stage,
    pub xa_softmax: Stage,
    pub xa_wv: Stage,
    pub xa_merge: Stage,
    pub xa_out: Stage,

    /// Tokens generated, for per-token cost.
    pub tokens: AtomicU64,
}

impl Profile {
    const fn new() -> Self {
        Self {
            mel: Stage::new(),
            encoder: Stage::new(),
            decoder: Stage::new(),
            sampling: Stage::new(),
            dec_embed: Stage::new(),
            dec_self_attn: Stage::new(),
            dec_cross_attn: Stage::new(),
            dec_mlp: Stage::new(),
            dec_final: Stage::new(),
            enc_conv: Stage::new(),
            enc_attn: Stage::new(),
            enc_mlp: Stage::new(),
            xa_qproj: Stage::new(),
            em_ln: Stage::new(),
            em_fc1: Stage::new(),
            em_gelu: Stage::new(),
            em_fc2: Stage::new(),
            ea_proj: Stage::new(),
            ea_prep: Stage::new(),
            ea_kernel: Stage::new(),
            ea_merge: Stage::new(),
            xa_prep: Stage::new(),
            xa_qk: Stage::new(),
            xa_softmax: Stage::new(),
            xa_wv: Stage::new(),
            xa_merge: Stage::new(),
            xa_out: Stage::new(),
            tokens: AtomicU64::new(0),
        }
    }

    pub fn count_tokens(&self, n: usize) {
        if enabled() {
            self.tokens.fetch_add(n as u64, Ordering::Relaxed);
        }
    }

    /// Human-readable breakdown, shares of total.
    pub fn report(&self) -> String {
        let stages: [(&str, &Stage); 4] = [
            ("mel", &self.mel),
            ("encoder", &self.encoder),
            ("decoder", &self.decoder),
            ("sampling", &self.sampling),
        ];
        let total: f64 = stages.iter().map(|(_, s)| s.secs()).sum();
        let tokens = self.tokens.load(Ordering::Relaxed);
        let mut out = format!(
            "\n{:<10} {:>10} {:>8} {:>9} {:>12}\n",
            "STAGE", "SECONDS", "SHARE", "CALLS", "MS/CALL"
        );
        for (name, stage) in stages {
            let secs = stage.secs();
            let calls = stage.calls().max(1);
            out.push_str(&format!(
                "{name:<10} {secs:>10.3} {:>7.1}% {:>9} {:>12.3}\n",
                if total > 0.0 {
                    secs / total * 100.0
                } else {
                    0.0
                },
                stage.calls(),
                secs * 1000.0 / calls as f64,
            ));
        }
        out.push_str(&format!("{:<10} {total:>10.3}\n", "total"));

        // Decoder internals, shown as shares of the decoder stage above.
        let inner: [(&str, &Stage); 5] = [
            ("  embed", &self.dec_embed),
            ("  self-attn", &self.dec_self_attn),
            ("  cross-attn", &self.dec_cross_attn),
            ("  mlp", &self.dec_mlp),
            ("  final-proj", &self.dec_final),
        ];
        let enc: [(&str, &Stage); 3] = [
            ("  conv", &self.enc_conv),
            ("  attn", &self.enc_attn),
            ("  mlp", &self.enc_mlp),
        ];
        let enc_total: f64 = enc.iter().map(|(_, s)| s.secs()).sum();
        if enc_total > 0.0 {
            out.push_str(&format!(
                "
{:<14} {:>10} {:>8} {:>9}
",
                "ENCODER OP", "SECONDS", "SHARE", "CALLS"
            ));
            for (name, stage) in enc {
                out.push_str(&format!(
                    "{name:<14} {:>10.3} {:>7.1}% {:>9}
",
                    stage.secs(),
                    stage.secs() / enc_total * 100.0,
                    stage.calls(),
                ));
            }
        }

        let xa: [(&str, &Stage); 15] = [
            ("  em ln", &self.em_ln),
            ("  em fc1", &self.em_fc1),
            ("  em gelu", &self.em_gelu),
            ("  em fc2", &self.em_fc2),
            ("  ea proj", &self.ea_proj),
            ("  ea prep", &self.ea_prep),
            ("  ea kernel", &self.ea_kernel),
            ("  ea merge", &self.ea_merge),
            ("  q proj", &self.xa_qproj),
            ("  q prep", &self.xa_prep),
            ("  q@k", &self.xa_qk),
            ("  softmax", &self.xa_softmax),
            ("  w@v", &self.xa_wv),
            ("  merge", &self.xa_merge),
            ("  out proj", &self.xa_out),
        ];
        let xa_total: f64 = xa.iter().map(|(_, s)| s.secs()).sum();
        if xa_total > 0.0 {
            out.push_str(&format!(
                "
{:<14} {:>10} {:>8} {:>9}
",
                "CROSS-ATTN OP", "SECONDS", "SHARE", "CALLS"
            ));
            for (name, stage) in xa {
                out.push_str(&format!(
                    "{name:<14} {:>10.3} {:>7.1}% {:>9}
",
                    stage.secs(),
                    stage.secs() / xa_total * 100.0,
                    stage.calls(),
                ));
            }
            out.push_str(&format!(
                "{:<14} {:>10.3}   (stage total {:.3})
",
                "  SUM",
                xa_total,
                self.dec_cross_attn.secs()
            ));
        }

        let inner_total: f64 = inner.iter().map(|(_, s)| s.secs()).sum();
        if inner_total > 0.0 {
            out.push_str(&format!(
                "\n{:<14} {:>10} {:>8} {:>9}\n",
                "DECODER OP", "SECONDS", "SHARE", "CALLS"
            ));
            for (name, stage) in inner {
                out.push_str(&format!(
                    "{name:<14} {:>10.3} {:>7.1}% {:>9}\n",
                    stage.secs(),
                    stage.secs() / inner_total * 100.0,
                    stage.calls(),
                ));
            }
        }
        if tokens > 0 {
            out.push_str(&format!(
                "\n{tokens} tokens generated · {:.3} ms/token through the decoder\n",
                self.decoder.secs() * 1000.0 / tokens as f64
            ));
        }
        out
    }
}

/// The process-wide profile. One transcription run is the unit of interest,
/// so the CLI prints and this is not reset between clips.
#[must_use]
pub fn profile() -> &'static Profile {
    static PROFILE: Profile = Profile::new();
    &PROFILE
}

/// True when `FFAI_PROFILE` is set — the CLI checks this to decide whether to
/// print a report.
#[must_use]
pub fn is_enabled() -> bool {
    enabled()
}

/// Time `f` into `stage`, unless profiling is off.
pub fn timed<T>(stage: &Stage, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let started = Instant::now();
    let out = f();
    stage.add(started.elapsed().as_nanos() as u64);
    out
}

/// Zero every bucket, so a harness can warm up (model load, precision
/// calibration) outside the measured region and then start clean.
pub fn reset() {
    let p = profile();
    let stages = [
        &p.mel,
        &p.encoder,
        &p.decoder,
        &p.sampling,
        &p.dec_embed,
        &p.dec_self_attn,
        &p.dec_cross_attn,
        &p.dec_mlp,
        &p.dec_final,
        &p.enc_conv,
        &p.enc_attn,
        &p.enc_mlp,
        &p.xa_qproj,
        &p.em_ln,
        &p.em_fc1,
        &p.em_gelu,
        &p.em_fc2,
        &p.ea_proj,
        &p.ea_prep,
        &p.ea_kernel,
        &p.ea_merge,
        &p.xa_prep,
        &p.xa_qk,
        &p.xa_softmax,
        &p.xa_wv,
        &p.xa_merge,
        &p.xa_out,
    ];
    for s in stages {
        s.nanos.store(0, Ordering::Relaxed);
        s.calls.store(0, Ordering::Relaxed);
    }
    p.tokens.store(0, Ordering::Relaxed);
}

/// Marker so `AtomicBool` stays available if a per-run reset lands later.
#[allow(dead_code)]
type Unused = AtomicBool;
