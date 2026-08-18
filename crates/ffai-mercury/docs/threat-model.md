# ffai-mercury — threat model

**Scope**: the `ffai-mercury` crate — Whisper/WhisperX-class ASR and VITS/Piper-class TTS.
**Method**: STRIDE per trust boundary, plus an attack tree on the highest-value path.
**Reviewed**: 2026-08-15 · **Next review**: 2026-11-15, or on any change to the model
loading path, the diarizer, or the crate's dependency set.
**Related**: [`plans/use-protection-please.md`](plans/use-protection-please.md) (gate
status), [`data-inventory.md`](data-inventory.md) (what personal data is processed).

## 1. What we are protecting

| Asset | Why it matters | Worst case |
|---|---|---|
| Speech audio supplied by the caller | Personal data; often confidential in substance (meetings, medical, legal) | Disclosure to a third party or a log sink |
| Transcripts | Same content, easier to index and search | Disclosure; silent corruption relied on as truth |
| **Speaker embeddings (voiceprints)** | **GDPR Art 9 special-category biometric data** when used to identify | Disclosure enables re-identification across recordings |
| Integrity of recognition output | Callers act on transcripts | Wrong text presented as correct — no error, no signal |
| Availability of the engine | Batch and interactive pipelines depend on it | Panic or unbounded work turns one bad input into an outage |
| Host process memory | The crate contains ~30 `unsafe` sites | Memory corruption escalating to code execution |

## 2. Trust boundaries

```
   caller's audio  ─────────────►┐
   caller's text   ─────────────►│   B1: the public API
                                 │
   model cache on disk ─────────►┤   B2: the model cache  ◄── THE WEAK ONE
   (weights, ONNX, JSON,         │
    lexicons, tokenizers)        │
                                 ▼
                        ffai-mercury process memory
                                 │
                                 ├──► transcripts / audio returned to caller
                                 └──► log sinks (B3)
```

- **B1 — the public API.** Audio and text arrive from the caller. Assumed hostile in
  *shape* (any length, any values, NaNs, empty), not in *intent*.
- **B2 — the model cache.** Weights, ONNX graphs, JSON configs, and binary lexicons are
  read from disk and parsed. **This crate does not verify them.** Hash verification lives
  upstream in `ffai-models`; mercury trusts whatever is on disk.
- **B3 — log sinks.** Anything logged leaves the process and lands somewhere with
  different retention and access rules.

There is **no network boundary**: this crate makes no outbound connections. Weight
downloading belongs to `ffai-models`.

## 3. STRIDE

| Threat | Concrete case here | Mitigation | Gate |
|---|---|---|---|
| **S**poofing | A "voice pack" or weight file substituted in the cache, impersonating a legitimate model | Upstream hash verification in `ffai-models`; **not asserted in this crate** | C-01, H-19 |
| **T**ampering | Crafted ONNX graph, `vits-graph.json`, `voice-config.json`, or lexicon triggering a parse defect; safetensors mutated *while mmapped* (immediate UB) | Bounds-checked parsing; no fuzz targets yet; mmap lifetime unaudited | H-19, H-26, H-27 |
| **R**epudiation | No record of which model version produced a transcript | Out of scope for a library; the consuming service owns audit logging | C-07 (N/A here) |
| **I**nformation disclosure | Transcript or audio content reaching a log sink; voiceprints persisted or returned without the caller understanding they are biometric | Log hygiene unreviewed; data inventory now written | H-20, C-08, C-01 |
| **D**enial of service | Panic on malformed input (126 `unwrap`/`expect` sites); unbounded allocation from an attacker-chosen length; pathological audio driving quadratic work | `overflow-checks` now on in release; per-callsite triage outstanding | H-05, H-18 |
| **E**levation of privilege | Memory corruption via the AVX2 kernels or the `SendPtr` raw-pointer sharing across rayon tasks | SAFETY comments + `UNSAFE.md` inventory; no Miri, no TSan yet | H-16, H-21, H-23, H-24 |

## 4. The highest-value attack path

**Write to the model cache → memory corruption in the host process.**

1. Attacker obtains write access to the model cache directory — a shared machine, a
   compromised download step, a malicious "voice pack" a user was told to install, or a
   world-writable cache path.
2. They replace a safetensors file. `asr/model.rs` maps it with
   `VarBuilder::from_mmaped_safetensors`, which is `unsafe` precisely because **mutating
   the file while it is mapped is undefined behaviour** — the attacker does not even need
   a parser bug, only the ability to write during the mapping's lifetime.
3. Alternatively they supply a crafted ONNX graph or lexicon and attack the parser, which
   currently has no fuzz coverage.

**Why it is credible**: the cache is an ordinary directory with ordinary permissions, and
this crate performs zero verification of what it finds there.

**What closes it**: assert hash verification at mercury's own boundary rather than
inheriting it silently (C-01/H-19); fuzz the ONNX/JSON/lexicon loaders (H-26); document
that the cache is a trust boundary the deployer must protect (this file).

## 5. Adversaries

| Adversary | Capability | In scope |
|---|---|---|
| Malicious input supplier | Chooses audio and text through the public API | Yes |
| Local attacker with cache write access | Replaces or mutates model files | Yes — the primary path |
| Supply-chain actor | Compromises `candle`, `tokenizers`, `rustfft`, or their transitive deps | Yes — H-08/H-09/H-10 |
| Remote network attacker | — | **No**: this crate opens no sockets |
| Caller of the API | Already inside the trust boundary; sees their own data | No |

## 6. Accepted, for now

- **The model cache is trusted.** Documented rather than fixed. Deployers must treat the
  cache directory as security-relevant and restrict write access to it. Tracked as R-001.
- **No formal proof of the `unsafe` kernels.** They carry SAFETY arguments and an
  inventory; Miri cannot execute most of the x86 intrinsics involved, so a scalar-fallback
  test path is required before that gate can close. Tracked as R-002.
