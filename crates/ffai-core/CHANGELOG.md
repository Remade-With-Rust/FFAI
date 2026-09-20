# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.1](https://github.com/Remade-With-Rust/FFAI/compare/ffai-core-v0.7.0...ffai-core-v0.7.1) - 2026-08-29

### Added

- *(core)* NEON twins so aarch64 stops taking the scalar oracle

### Other

- *(ci)* clippy green — the gate is REQUIRED, so this is the merge blocker
- *(ci)* make the new kernels clippy-clean, and unblock the supply-chain gate
- ten wasm wins: the ISA asymmetry was the whole gap
