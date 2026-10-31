# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0](https://github.com/Remade-With-Rust/FFAI/compare/ffai-models-v0.6.6...ffai-models-v0.7.0) - 2026-10-31

### Added

- `ModelManifest::fetch_only`: resolve and download only the named files

### Changed

- Requires `ffai-core` 0.9: `OcrOptions` gained `deskew` and `OcrOutput` gained `skew_degrees` and `turns`, so the core types this crate exposes changed.
- Drops `dirs` (and with it the MPL `option-ext`)

### Fixed

- An engine fetching from the hub downloads only the files it opens, as offline resolution already did

## [0.6.6](https://github.com/Remade-With-Rust/FFAI/compare/ffai-models-v0.6.5...ffai-models-v0.6.6) - 2026-09-25

### Added

- `resolve_only_in` and `verify`
- `load_dir` names the missing directory

## [0.6.5](https://github.com/Remade-With-Rust/FFAI/compare/ffai-models-v0.6.4...ffai-models-v0.6.5) - 2026-09-20

### Other

- updated the following local packages: ffai-core

## [0.6.4](https://github.com/Remade-With-Rust/FFAI/compare/ffai-models-v0.6.3...ffai-models-v0.6.4) - 2026-08-29

### Other

- updated the following local packages: ffai-core
