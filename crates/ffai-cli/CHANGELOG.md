# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.2](https://github.com/Remade-With-Rust/FFAI/compare/ffai-cli-v0.11.1...ffai-cli-v0.11.2) - 2026-11-10

### Other

- update Cargo.lock dependencies

## [0.11.1](https://github.com/Remade-With-Rust/FFAI/compare/ffai-cli-v0.11.0...ffai-cli-v0.11.1) - 2026-11-08

### Other

- update Cargo.lock dependencies

## [0.11.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-cli-v0.10.0...ffai-cli-v0.11.0) - 2026-11-07

### Changed

- rusty_alloc and rusty_alloc-api `=2.2.5` -> `=2.4.0`. 2.3.0 shortens the allocation paths (up to 4.2 % fewer instructions across whole Rust programs in its own benchmarks). 2.3.1 and 2.4.0 commit memory as it is used, so a process no longer charges 1 GiB of commit on Windows for its arena, nor 32 MiB per thread. 2.4.0 turns delayed purging on by default (`purge_delay` 10 ms): freed spans of 512 KiB or more go back to the OS after 10 ms, so RSS falls back after a peak; `RUSTY_ALLOC_PURGE_DELAY=-1` restores 2.2.5's retain-everything behaviour. The pin is exact, so a consumer that pins rusty_alloc itself must move to 2.4.0 too.
- Requires `ffai-diana` 0.11, `ffai-argus` 0.10 and `ffai-carmenta` 0.14.

## [0.10.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-cli-v0.9.0...ffai-cli-v0.10.0) - 2026-11-07

### Changed

- Requires `ffai-diana` 0.10: about a third fewer instructions per detection, and Winograd 3x3 convolutions plus a polynomial softmax `exp` by default (see ffai-diana's CHANGELOG for the knobs that restore the old arithmetic).

## [0.9.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-cli-v0.8.0...ffai-cli-v0.9.0) - 2026-10-31

### Changed

- Requires `ffai-core` 0.9: `OcrOptions` gained `deskew` and `OcrOutput` gained `skew_degrees` and `turns`, so the core types this crate exposes changed.

## [0.8.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-cli-v0.7.0...ffai-cli-v0.8.0) - 2026-10-08

### Changed

- rusty_alloc and rusty_alloc-api `=2.2.0` -> `=2.2.5`: builds on MSVC with LTO and unwinding, frees model memory stranded by exited loader threads, purges freed huge blocks, and fails cleanly when Windows refuses a re-commit. The pin is exact, so a consumer that pins rusty_alloc itself must move to 2.2.5 too

## [0.7.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-cli-v0.6.9...ffai-cli-v0.7.0) - 2026-09-25

### Added

- `ffai ocr` reads PDFs, compositing redactions and withholding risky text layers
- `--charset`, `--remove-rules`, `--auto-orient` and `--checkboxes`
- `ffai models --verify`

## [0.6.9](https://github.com/Remade-With-Rust/FFAI/compare/ffai-cli-v0.6.8...ffai-cli-v0.6.9) - 2026-09-20

### Other

- Merge remote-tracking branch 'origin/master' into docs/reconcile-claims
- bring every Remade-With-Rust crate to its latest published version

## [0.6.8](https://github.com/Remade-With-Rust/FFAI/compare/ffai-cli-v0.6.7...ffai-cli-v0.6.8) - 2026-08-29

### Other

- *(ci)* clippy green — the gate is REQUIRED, so this is the merge blocker

## [0.6.7](https://github.com/Remade-With-Rust/FFAI/compare/ffai-cli-v0.6.6...ffai-cli-v0.6.7) - 2026-08-28

### Other

- carmenta 0.10.0 + wasm 0.2.0: the 4 GB wall was the allocator, and the detector had no ceiling
