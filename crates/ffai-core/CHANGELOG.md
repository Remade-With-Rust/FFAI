# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-core-v0.7.2...ffai-core-v0.8.0) - 2026-09-25

### Changed

- **BREAKING:** `OcrOptions` gains `charset`, `remove_rules` and `auto_orient`, which breaks construction by struct literal

## [0.7.2](https://github.com/Remade-With-Rust/FFAI/compare/ffai-core-v0.7.1...ffai-core-v0.7.2) - 2026-09-20

### Other

- Merge remote-tracking branch 'origin/master' into docs/reconcile-claims
- *(core)* an exact ties-away round, and why it did NOT help vocab_int8
- *(core,carmenta,diana)* take floor out of five per-pixel loops

## [0.7.1](https://github.com/Remade-With-Rust/FFAI/compare/ffai-core-v0.7.0...ffai-core-v0.7.1) - 2026-08-29

### Added

- *(core)* NEON twins so aarch64 stops taking the scalar oracle

### Other

- *(ci)* clippy green — the gate is REQUIRED, so this is the merge blocker
- *(ci)* make the new kernels clippy-clean, and unblock the supply-chain gate
- ten wasm wins: the ISA asymmetry was the whole gap
