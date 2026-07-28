# ffai-bench

The analyzer for [FFai](https://github.com/Remade-With-Rust/FFAI): one call compares an engine against world-standard implementations on a pinned, hash-verified corpus, and appends an audit-grade record.

```text
ffai bench asr --corpus corpora/librispeech-test-clean-v2.toml
```

## What it does that a timing loop does not

**Four gates, and a skipped gate is never a pass.** Correctness, quality, speed, footprint. All four must pass before a claim is claimable; a gate with nothing to compare against reports `SKIP`, never `PASS`.

**Like-for-like quality.** The quality gate compares only against references declaring the engine's own configuration (`config = "tiny.en/greedy"` in `references.toml`). Comparing a 39M greedy engine against a 74M beam-search one measures model size, not implementation quality — and reports it under the same label. The best-of-all-references number stays in the record as context.

**The configuration is in the record.** `RunSummary::config` captures the options an in-process engine actually ran with, because a reference records its argv and an engine had no equivalent. When speech segmentation became a default, the same engine name produced 7.99 % one day and 6.79 % the next with nothing to distinguish the two runs. That is what this field prevents.

**Corpora are hashed.** A clip whose bytes drift from the manifest fails the run rather than quietly changing the result.

## Metrics

- `metrics` — WER and CER, both sides through the same normalizer
- `der` — Diarization Error Rate under the *optimal* label mapping, with collared and full variants, an RTTM parser, and no clamping at 100 %
- `speed` — best-of-N, warm and end-to-end reported separately
- `footprint` — steady and peak resident memory, sampled the same way for every implementation

## Ledger

Every run appends one JSON line: corpus fingerprint, reference versions, engine configuration, environment, per-implementation results, and the four-gate verdict. Every public claim FFai makes should trace to one.

## License

MIT OR Apache-2.0.
