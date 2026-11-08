# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.1](https://github.com/Remade-With-Rust/FFAI/compare/ffai-mercury-v1.2.0...ffai-mercury-v1.2.1) - 2026-11-08

### Fixed

- TTS (and any candle matmul) gave one of two exact outputs per process on hybrid Intel CPUs, the second in about 3–5 % of runs. gemm detects cache sizes once per process from whichever core it starts on, and P-cores and E-cores disagree, which changes its k-blocking. Model loaders now run that detection on a thread pinned to the lowest allowed logical CPU. New public `pin_gemm_blocking()` for applications that run other candle code first.

## [1.1.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-mercury-v1.0.5...ffai-mercury-v1.1.0) - 2026-10-31

### Changed

- Requires `ffai-core` 0.9: `OcrOptions` gained `deskew` and `OcrOutput` gained `skew_degrees` and `turns`, so the core types this crate exposes changed.

## [1.0.5](https://github.com/Remade-With-Rust/FFAI/compare/ffai-mercury-v1.0.4...ffai-mercury-v1.0.5) - 2026-09-25

### Other

- dependency bump only: ffai-core 0.8 / ffai-carmenta 0.11 / ffai-models 0.6.6 (see the workspace CHANGELOG)

## [1.0.4](https://github.com/Remade-With-Rust/FFAI/compare/ffai-mercury-v1.0.3...ffai-mercury-v1.0.4) - 2026-09-20

### Other

- Merge remote-tracking branch 'origin/master' into docs/reconcile-claims
- hoist the bounds checks out of six per-pixel loops, safely

## [1.0.3](https://github.com/Remade-With-Rust/FFAI/compare/ffai-mercury-v1.0.2...ffai-mercury-v1.0.3) - 2026-08-29

### Other

- *(ci)* the last three clippy findings, two of them Linux-only
- Merge remote-tracking branch 'origin/master' into wasm-simd128
- *(mercury)* restore rustfmt-clean, which this branch had broken
- ten wasm wins: the ISA asymmetry was the whole gap

## [1.0.2](https://github.com/Remade-With-Rust/FFAI/compare/ffai-mercury-v1.0.1...ffai-mercury-v1.0.2) - 2026-08-28

### Other

- restore the two crates the Format gate blocks on
