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
//! - [`model`], [`engine`]: the scalar engine — `GGUF` load, the `BitNet` b1.58 blocks, a KV
//!   cache, greedy decoding — gated on bitnet.cpp's tokens.
//! - [`head`]: the greedy head's `int8` screen with a proven per-row bound: half the head's
//!   bytes, the full head's token.
//! - [`numerics`]: the float arithmetic in one canonical order, so a file gives the same tokens
//!   on every device; these scalar functions are the oracles.
//! - [`lora`]: LoRA adapters (llama.cpp's adapter `GGUF`) on top of the ternary base.
//! - [`simd`]: a bit-identical SIMD twin of every scalar kernel (SSSE3, AVX2, NEON, WASM-SIMD128),
//!   with a kernel-reach census.
//!
//! The format and the scalar integer kernels come from `rusty_ternary`; those kernels are the
//! permanent oracle every faster one is `assert_eq!`-gated against.

// `unsafe` lives in one place: the SIMD twins (`simd`), each behind run-time detection and
// bounds its safe entry point asserts. Everything else is safe code.
#![deny(unsafe_code)]

pub mod engine;
pub mod head;
pub mod kernel;
pub mod lora;
pub mod model;
pub mod numerics;
pub mod prof;
pub mod roofline;
#[allow(unsafe_code)]
pub mod simd;
pub mod threads;
