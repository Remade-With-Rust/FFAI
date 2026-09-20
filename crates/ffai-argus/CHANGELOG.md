# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
