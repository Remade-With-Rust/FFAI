//! # ffai-bench — FFai's analyzer
//!
//! One call to answer: **how does our engine compare to the world standard,
//! on pinned data, reproducibly?** `ffai bench asr --corpus corpora/x.toml`
//! runs our engine and any configured reference implementations over the same
//! holdout clips, computes task metrics (WER/CER, real-time factor), and
//! appends an audit-grade record to the claims ledger.
//!
//! ## Lineage
//!
//! The measurement spine is ported from **Prometheus**, the private refinery
//! built for `remade_ffmpeg_rs`: the four-gate verdict, best-of-N wall-clock
//! timing, hashed corpus manifests with clip-level train/holdout splits, and
//! the append-only JSONL ledger where a skipped gate is never a pass and
//! losses are recorded as knowledge. The symbolic-discovery half of Prometheus
//! (symreg → E-graph simplify → SMT prove → codegen) stays private and
//! codec-focused; it does not apply to learned-model engines.
//!
//! This crate is public on purpose: performance/quality claims FFai makes are
//! only worth making if anyone can re-run them from a ledger line alone.
//!
//! ## The four gates (adapted from Prometheus for model engines)
//!
//! | Gate | Prometheus meaning | FFai meaning |
//! |---|---|---|
//! | correctness | bit-exact / conformance | engine completes every holdout clip with well-formed output |
//! | quality | corpus BD-rate / PEAQ | task metric vs reference (WER/CER parity band) on holdout |
//! | speed | best-of-N vs cycle budget | best-of-N real-time factor vs reference |
//! | footprint | SMT safety proof | peak memory / binary size budget (instrumented in Phase 1) |

pub mod corpus;
pub mod der;
pub mod footprint;
pub mod gate;
pub mod ledger;
pub mod metrics;
pub mod normalize;
pub mod reference;
pub mod resample;
pub mod runner;
pub mod speed;
pub mod tts;
