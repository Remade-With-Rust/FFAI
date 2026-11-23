//! # Minerva
//!
//! FFai's CPU/WASM inference engine for ternary (`BitNet` b1.58) language models: a
//! rusty_sloth export (an `I2_S` `GGUF` base, an adapter `GGUF` beside it) run on x86, ARM and
//! in the browser, with the same tokens on every device.
//!
//! Built one measured brick at a time (plan: rusty_sloth `docs/plans/bitnet.md`, Phase 4):
//!
//! - [`roofline`]: bytes a token streams against this machine's read bandwidth — the ceiling
//!   every kernel is judged against, computed before any kernel exists.
//!
//! The format and the scalar integer kernels come from `rusty_ternary`; those kernels are the
//! permanent oracle every faster one is `assert_eq!`-gated against.

#![forbid(unsafe_code)]

pub mod roofline;
