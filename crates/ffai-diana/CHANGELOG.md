# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.1](https://github.com/Remade-With-Rust/FFAI/compare/ffai-diana-v0.11.0...ffai-diana-v0.11.1) - 2026-11-08

### Fixed

- *(diana)* clear the 8 clippy -D warnings failing the harden gate

## [0.11.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-diana-v0.10.0...ffai-diana-v0.11.0) - 2026-11-07

### Changed

- rusty_alloc and rusty_alloc-api `=2.2.5` -> `=2.4.0`. 2.3.0 shortens the allocation paths (up to 4.2 % fewer instructions across whole Rust programs in its own benchmarks). 2.3.1 and 2.4.0 commit memory as it is used, so a process no longer charges 1 GiB of commit on Windows for its arena, nor 32 MiB per thread. 2.4.0 turns delayed purging on by default (`purge_delay` 10 ms): freed spans of 512 KiB or more go back to the OS after 10 ms, so RSS falls back after a peak; `RUSTY_ALLOC_PURGE_DELAY=-1` restores 2.2.5's retain-everything behaviour. The pin is exact, so a consumer that pins rusty_alloc itself must move to 2.4.0 too.

## [0.10.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-diana-v0.9.0...ffai-diana-v0.10.0) - 2026-11-07

### Changed

- **Breaking:** `dwconv::depthwise3x3` takes an `act: bool` and fuses bias and SiLU into its own pass.
- Default arithmetic changed in two places, both behind knobs: Winograd F(2x2,3x3) for 3x3 convolutions with at least 16 channels (`FFAI_DIANA_NO_WINOGRAD=1` restores the GEMM), and a polynomial `exp` in attention's softmax (`FFAI_DIANA_SOFTMAX_LIBM=1` restores libm). Both pass the full-graph and depth oracles on all five tiers.

### Performance

- 34.6 % fewer instructions per detect (1,075 M to 703 M, callgrind, yolo26n at 640) and 24.3 % fewer per tracker frame. Forty-five changes, each counted alone and byte-identical; ledger in `docs/plans/diana-ir-veins.md`. Wall time not re-measured.

### Added

- `maxpool`, `softmax`, `stem`, `upcat` and `winograd` modules; `cpuop::matmul2`, `matmul2_batched`, `matmul2_into`, `write_block`, `add_assign`, `cat_channels`, `cat_last`, `split_qkv`, `SliceOp2`; `conv3x3::conv3x3_residual`; `epilogue::apply_residual`; flat cost-matrix API in `track::assign` (`CostMat`, `assign_flat`, `assign_gated_flat`).
- Direct dependency on `gemm` 0.19 (already in the tree through candle).

## [0.9.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-diana-v0.8.0...ffai-diana-v0.9.0) - 2026-10-31

### Changed

- Requires `ffai-core` 0.9: `OcrOptions` gained `deskew` and `OcrOutput` gained `skew_degrees` and `turns`, so the core types this crate exposes changed.

## [0.8.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-diana-v0.7.8...ffai-diana-v0.8.0) - 2026-10-08

### Changed

- rusty_alloc and rusty_alloc-api `=2.2.0` -> `=2.2.5`: builds on MSVC with LTO and unwinding, frees model memory stranded by exited loader threads, purges freed huge blocks, and fails cleanly when Windows refuses a re-commit. The pin is exact, so a consumer that pins rusty_alloc itself must move to 2.2.5 too

## [0.7.8](https://github.com/Remade-With-Rust/FFAI/compare/ffai-diana-v0.7.7...ffai-diana-v0.7.8) - 2026-09-25

### Other

- dependency bump only: ffai-core 0.8 / ffai-carmenta 0.11 / ffai-models 0.6.6 (see the workspace CHANGELOG)

## [0.7.7](https://github.com/Remade-With-Rust/FFAI/compare/ffai-diana-v0.7.6...ffai-diana-v0.7.7) - 2026-09-20

### Other

- Merge remote-tracking branch 'origin/master' into docs/reconcile-claims
- bring every Remade-With-Rust crate to its latest published version
- hoist the bounds checks out of six per-pixel loops, safely
- *(core,carmenta,diana)* take floor out of five per-pixel loops
- reconcile the claim surface, and open the rounding census

## [0.7.6](https://github.com/Remade-With-Rust/FFAI/compare/ffai-diana-v0.7.5...ffai-diana-v0.7.6) - 2026-08-29

### Other

- *(carmenta,diana)* the third blocking clippy command, which I had missed

## [0.7.5](https://github.com/Remade-With-Rust/FFAI/compare/ffai-diana-v0.7.4...ffai-diana-v0.7.5) - 2026-08-28

### Other

- carmenta 0.10.0 + wasm 0.2.0: the 4 GB wall was the allocator, and the detector had no ceiling
- vision round 4: the 15% is not available, with the arithmetic that bounds it
