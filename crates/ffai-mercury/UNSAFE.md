# ffai-mercury — `unsafe` inventory

Satisfies gate **H-16**. Every `unsafe` site in this crate, what it is for, and what makes
it sound. **Last audited: 2026-08-15** (survey depth — no Miri, no TSan; see H-23/H-24).

> **How this list was built, and why the obvious method undercounts.** The workspace lint
> `unsafe_code = "warn"` enumerates 25 sites in this crate. It reports **nothing** for the
> 3 sites carrying `#[allow(unsafe_code)]` — an attribute that silences the lint also
> silences the inventory. Both sources were used. If you add `#[allow(unsafe_code)]`
> anywhere, add the site here in the same commit, because nothing else will catch it.

**Totals**: 28 sites — 12 blocks, 11 `unsafe fn`, 2 `unsafe impl`, 3 allow-suppressed.
**SAFETY comments**: 27, covering all 28 sites — the adjacent `unsafe impl Send`/`Sync`
pair at `decoder_kernels.rs:194-195` shares one comment. Each states the invariant *and*
what upholds it: for Class A the dominating `have_avx2()`/`have_f16c()` check, for Class B
the index arithmetic that makes the regions disjoint, for Class C the fact that the
invariant is **not** enforceable from inside this crate.

## Class A — SIMD kernels (`#[target_feature]`)

23 sites. `unsafe` here is a *compile-time capability* claim, not pointer arithmetic: a
`#[target_feature(enable = "avx2")]` function is `unsafe` because calling it on a CPU
without AVX2 is UB. Soundness therefore rests entirely on the runtime dispatch check.

| File | Sites | Guard |
|---|---|---|
| [`asr/flash_attn.rs`](src/asr/flash_attn.rs) | 15 — 7 blocks, 8 `fn` | `have_avx2()` / `have_f16c()`; non-x86 builds get an `unreachable!` stub |
| [`tts/decoder_kernels.rs`](src/tts/decoder_kernels.rs) | 4 — 212, 241, 1341 (blocks); 732 (`fn`) | same dispatch pattern |
| [`asr/f16_gemv.rs`](src/asr/f16_gemv.rs) | 2 — 141 (block); 153 (`fn`) | `have_f16c()` |
| [`asr/vocab_int8.rs`](src/asr/vocab_int8.rs) | 2 — 202 (block); 237 (`fn`) | `have_avx2()` |

**Invariant**: every `#[target_feature]` function is called only from a path that has
already tested for that feature at runtime. Adding a call site without the check is UB on
older CPUs and will not fail any current test — the CI machine has AVX2.

Two of these blocks carry a *second* obligation beyond the feature check: `flash_attn`'s
f16 K/V views reinterpret candle's `F16` storage as `u16`, which is sound only because
`half::f16` is `repr(transparent)` over `u16` and the tensors were checked contiguous.
Both say so at the site.

## Class B — raw-pointer sharing across rayon tasks

| Site | What it does | Argument | Status |
|---|---|---|---|
| [`tts/decoder_kernels.rs:194-195`](src/tts/decoder_kernels.rs#L194) | `unsafe impl Send`/`Sync` for `SendPtr(*mut f32)` | Chunk co-ranges partition `[0, c_out)` and block t-ranges partition `[0, l_out)`, so no two tasks address the same element | Documented at the site, **still unproven by tooling** — no TSan |
| `tts/decoder_kernels.rs:212, 241` | `from_raw_parts_mut` into the output buffer per task | Disjointness as above | SAFETY comment present |
| `asr/flash_attn.rs:186` | `from_raw_parts_mut` per attention head band | Disjoint column bands | SAFETY comment present |

**Invariant**: the index arithmetic that computes each task's region must be *provably*
non-overlapping. This is the crate's highest-risk `unsafe`: a wrong bound is a data race,
not a panic — silently wrong audio, or memory corruption, with no test failing.

**Open**: no TSan run (H-24), no Miri (H-23). The disjointness argument is now written at
the site rather than only in this file, but a comment is not a proof. TSan on the TTS
decode path remains the single highest-value item here.

## Class C — memory-mapped model weights (allow-suppressed)

| Site | What it does |
|---|---|
| [`asr/model.rs:110`](src/asr/model.rs#L110) | `VarBuilder::from_mmaped_safetensors` |
| [`asr/model.rs:118`](src/asr/model.rs#L118) | same, second mapping of the same blob |
| [`asr/aligner.rs:86`](src/asr/aligner.rs#L86) | same, for the alignment model |

**Invariant**: the mapped file must not change for the lifetime of the mapping. This is
**not enforceable from inside this crate** — any process with write access to the model
cache can violate it, and the result is immediate UB rather than a parse error.

This is R-001 in the audit and the primary path in the
[threat model](docs/threat-model.md#4-the-highest-value-attack-path). Deployers must treat
the model cache directory as security-relevant. The code-side mitigation is to assert the
upstream hash verification at this boundary instead of assuming it.

## Rules for changing anything here

1. Every new `unsafe` gets a `// SAFETY:` comment naming the invariant *and* what upholds
   it. "Coverage is 28/28" is a gate (H-16) — a new site without a comment breaks it.
2. Every new site is added to this file in the same commit, including `#[allow]`-ed ones.
3. Class A: prove the runtime feature check dominates the call site.
4. Class B: state the disjointness arithmetic explicitly; TSan before merge.
5. Class C: do not add another mapping without revisiting R-001.
