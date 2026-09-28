# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-carmenta-v0.10.2...ffai-carmenta-v0.11.0) - 2026-09-25

### Added

- offline weights: manifests ship in the crate; `offline()` and `preload()`
- charset-constrained CTC decoding
- quarter-turn orientation (`auto_orient`)
- opt-in form-rule removal
- full-width to ASCII fold on non-CJK lines
- lexicon matching
- checkbox detection
- label-to-value pairing

## [0.10.2](https://github.com/Remade-With-Rust/FFAI/compare/ffai-carmenta-v0.10.1...ffai-carmenta-v0.10.2) - 2026-09-20

### Other

- Merge remote-tracking branch 'origin/master' into docs/reconcile-claims
- bring every Remade-With-Rust crate to its latest published version
- *(carmenta)* measure it — 1.30x on the kernel, nothing end to end
- hoist the bounds checks out of six per-pixel loops, safely
- *(core,carmenta,diana)* take floor out of five per-pixel loops

## [0.10.1](https://github.com/Remade-With-Rust/FFAI/compare/ffai-carmenta-v0.10.0...ffai-carmenta-v0.10.1) - 2026-08-29

### Fixed

- versioned dev-dependencies on sibling crates deadlocked the release

### Other

- *(carmenta,diana)* the third blocking clippy command, which I had missed
- an In the wild block above the headline ([#31](https://github.com/Remade-With-Rust/FFAI/pull/31))
- release

## [0.10.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-carmenta-v0.9.1...ffai-carmenta-v0.10.0) - 2026-08-28

### Other

- carmenta 0.10.0 + wasm 0.2.0: the 4 GB wall was the allocator, and the detector had no ceiling
- vision round 4: the 15% is not available, with the arithmetic that bounds it
