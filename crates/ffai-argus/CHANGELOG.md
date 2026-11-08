# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.10.1](https://github.com/Remade-With-Rust/FFAI/compare/ffai-argus-v0.10.0...ffai-argus-v0.10.1) - 2026-11-08

### Changed

- 45 instruction-count wins across 15 veins (callgrind Ir, one core, each change counted alone). Against 0.10.0: prefill (1142 tokens) −24.2 %, decode −15.3 %, a whole 16-token caption −8.7 %, preprocessing −85.1 %, prompt assembly −58.7 %, one vision tile −0.8 % (94 % of a tile is the GEMM microkernel). The caption, the decode tokens and the preprocessed bytes match 0.10.0 exactly, and `decode_oracle` still gives 32/32 tokens. Wall time is not re-measured.
- The largest changes are: prefill attention skips the masked half of each causal block, causal softmax runs on vector kernels, the Lanczos resampler was rewritten as row AXPYs, SwiGLU is a single-pass kernel, and GEMM outputs are no longer zero-filled before gemm overwrites them (`mm`, calling the `gemm` 0.19 crate candle already links).
- Requires `ffai-core` 0.9.1 for the new `fastmath` kernels.

### Added

- `TextDecoder::step`, `decode::argmax`, `prompt::merge_compact`, `Engine::assemble_for_probe`, and the `mm` module.
- `FFAI_ARGUS_MATMUL_CANDLE=1` routes every product back through candle's matmul.

## [0.10.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-argus-v0.9.0...ffai-argus-v0.10.0) - 2026-11-07

### Changed

- rusty_alloc and rusty_alloc-api `=2.2.5` -> `=2.4.0`. 2.3.0 shortens the allocation paths (up to 4.2 % fewer instructions across whole Rust programs in its own benchmarks). 2.3.1 and 2.4.0 commit memory as it is used, so a process no longer charges 1 GiB of commit on Windows for its arena, nor 32 MiB per thread. 2.4.0 turns delayed purging on by default (`purge_delay` 10 ms): freed spans of 512 KiB or more go back to the OS after 10 ms, so RSS falls back after a peak; `RUSTY_ALLOC_PURGE_DELAY=-1` restores 2.2.5's retain-everything behaviour. The pin is exact, so a consumer that pins rusty_alloc itself must move to 2.4.0 too.

## [0.9.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-argus-v0.8.0...ffai-argus-v0.9.0) - 2026-10-31

### Changed

- Requires `ffai-core` 0.9: `OcrOptions` gained `deskew` and `OcrOutput` gained `skew_degrees` and `turns`, so the core types this crate exposes changed.

## [0.8.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-argus-v0.7.5...ffai-argus-v0.8.0) - 2026-10-08

### Changed

- rusty_alloc and rusty_alloc-api `=2.2.0` -> `=2.2.5`: builds on MSVC with LTO and unwinding, frees model memory stranded by exited loader threads, purges freed huge blocks, and fails cleanly when Windows refuses a re-commit. The pin is exact, so a consumer that pins rusty_alloc itself must move to 2.2.5 too

## [0.7.5](https://github.com/Remade-With-Rust/FFAI/compare/ffai-argus-v0.7.4...ffai-argus-v0.7.5) - 2026-09-25

### Other

- dependency bump only: ffai-core 0.8 / ffai-carmenta 0.11 / ffai-models 0.6.6 (see the workspace CHANGELOG)

## [0.7.4](https://github.com/Remade-With-Rust/FFAI/compare/ffai-argus-v0.7.3...ffai-argus-v0.7.4) - 2026-09-20

### Other

- Merge remote-tracking branch 'origin/master' into docs/reconcile-claims
- bring every Remade-With-Rust crate to its latest published version

## [0.7.3](https://github.com/Remade-With-Rust/FFAI/compare/ffai-argus-v0.7.2...ffai-argus-v0.7.3) - 2026-08-29

### Fixed

- *(argus)* unbreak the lib test, and clear the crate's hidden clippy debt
- *(argus)* drop the `D` import my softmax change orphaned

### Other

- *(ci)* the last three clippy findings, two of them Linux-only
- Merge remote-tracking branch 'origin/master' into wasm-simd128
- ten wasm wins: the ISA asymmetry was the whole gap
- measure the ceiling honestly — threads are NOT the lever, the ISA is
- expose the unsplit path — 544s -> 37.6s, and the caption holds up

## [0.7.2](https://github.com/Remade-With-Rust/FFAI/compare/ffai-argus-v0.7.1...ffai-argus-v0.7.2) - 2026-08-28

### Other

- carmenta 0.10.0 + wasm 0.2.0: the 4 GB wall was the allocator, and the detector had no ceiling
