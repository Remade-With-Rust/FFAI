# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-carmenta-v0.12.0...ffai-carmenta-v0.13.0) - 2026-10-31

### Added

- `OcrOptions::deskew` for the mobile detector: the angle comes from the detector's own text regions, and a page at least 0.3 degrees off is rotated once and read
- Word boxes for `mobiledet-svtr`, from the CTC emission frames snapped to ink
- `auto_orient` reports the turn it applied in `OcrOutput::turns`
- Ruled table bands are read row-major (`boxes::grid_order`)
- Hosted weights: the ppocrv5 manifests name `remade-with-rust/carmenta-ppocrv5-mobile`, so `fetch` downloads them sha256-pinned

### Changed

- **BREAKING:** `deskew::rotated_corners` takes the page's `turns`; requires `ffai-core` 0.9
- 4.6-6.0x faster than 0.12 on a clean 300 dpi page (measured independently by PdfCraft): a direct depthwise conv, fused epilogues, and 1x1 and transposed convs as matmuls, with output bit-identical to candle where gated
- `auto_orient` reads the page first and decides orientation only when the read is unconvincing, and keeps the turned read only when it is more confident
- The detector's floor upscale is capped at 4x
- Drops the unused `candle-transformers` and `tokenizers` dependencies

### Fixed

- An inline formula spliced into a line keeps the word boxes it does not touch

## [0.12.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-carmenta-v0.11.0...ffai-carmenta-v0.12.0) - 2026-10-08

### Changed

- rusty_alloc and rusty_alloc-api `=2.2.0` -> `=2.2.5`: builds on MSVC with LTO and unwinding, frees model memory stranded by exited loader threads, purges freed huge blocks, and fails cleanly when Windows refuses a re-commit. The pin is exact, so a consumer that pins rusty_alloc itself must move to 2.2.5 too

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
