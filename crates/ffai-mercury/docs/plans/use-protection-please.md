# ffai-mercury — hardening audit

**Standard**: FFai / Remade-With-Rust recursive hardening process — see the skill's `STANDARD.md`
**Registry**: 41 gates / 12 phases (`use-protection-please` v1)
**Unit**: `crates/ffai-mercury` — library crate (no `[[bin]]`), v0.7.0
**Tier**: critical-path — parses ONNX, JSON and binary lexicon blobs from the on-disk model cache, and carries ~30 hand-written `unsafe` SIMD/aliasing sites
**Mirrors**: this crate's `README.md`, and the standalone landing page
[Remade-With-Rust/mercury](https://github.com/Remade-With-Rust/mercury) (`README.md` on
`main`, rendered with `--link` to this file). The crates.io page renders this crate's
README, so it inherits the block — but **only absolute links resolve there**. All mirrors
**must be re-rendered in the same pass as this file** (SKILL.md §3.1)
**Compliance**: `gdpr-pii` — in scope: this unit processes speech audio and transcripts
(personal data), and its diarizer/speaker modules derive **voiceprints**, which are
**GDPR Art 9 special-category biometric data** when used to identify a person. Out of
scope: `pci-dss` (no cardholder data), `hipaa` (no PHI context), `soc2` (library, not a
service — it inherits the consuming service's controls), `eu-cra` (binds on commercial
shipment, not on the crate itself)
**Architect**: [Nick Overlock](https://www.linkedin.com/in/nick-overlock-593235b9/)
**Audit depth**: survey (static evidence only — no tool probe was executed)
**Audited**: 2026-08-15 by Claude (pilot) · **Next review**: 2026-11-15

> Source of truth for this unit's hardening status. The README's status table is
> **generated from this file** — edit here, then run:
> `python <skills>/use-protection-please/scripts/render_readme_table.py --plan docs/plans/use-protection-please.md --readme README.md`

**Status tokens**: `Completed` (evidenced pass) · `Scheduled` (owner + date in Target) ·
`Incomplete` (not done, or not evidenced) · `N/A` (out of tier — reason required in
Evidence; excluded from the totals).

---

## Threat sketch

*Assets* — user speech audio and its transcripts (private content); integrity of
recognition and synthesis output; availability of the engine against a malformed model or
audio input.

*Adversaries* — anyone who can write into the model cache directory (a compromised
download path, a shared machine, a malicious "voice pack"); a caller feeding crafted audio
or crafted text; a supply-chain actor upstream of `candle`, `tokenizers`, or `rustfft`.

*Highest-value attack path* — **the model cache is trusted, and this crate does not verify
it.** [`tts/vits.rs:100`](../../src/tts/vits.rs#L100) reads ONNX bytes,
[`vits.rs:138`](../../src/tts/vits.rs#L138) and `:163` parse `vits-graph.json` and
`voice-config.json`, [`tts/lexicon.rs:85`](../../src/tts/lexicon.rs#L85) reads a binary
lexicon, and [`asr/model.rs:111`](../../src/asr/model.rs#L111) mmaps safetensors through
`VarBuilder::from_mmaped_safetensors` (`unsafe`: the mapping is UB if the file is mutated
underneath it). The hash verification the README advertises is **not in this crate** —
`ffai-mercury` has no `sha2` dependency; that control lives upstream in `ffai-models`. So
mercury's safety rests on an assumption it does not itself enforce, and that assumption is
untested from this side.

*Second path* — ~30 `unsafe` sites across six files, including two `unsafe impl
Send`/`Sync` on a raw-pointer wrapper ([`tts/decoder_kernels.rs:194`](../../src/tts/decoder_kernels.rs#L194))
whose disjointness argument exists only as a comment. No Miri, no TSan, no proof.

*Full model* — **does not exist.** No `docs/threat-model.md`, no `SECURITY.md` anywhere in
the repository. This section is a sketch produced by the audit, not a reviewed model (H-01).

---

## Checklist

`★` = v1.0.0-blocking. Full probe and pass criteria per gate: the skill's `CHECKLIST.md`.

### Phase 0 — Threat modeling

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-01 | ★ Threat model documented and linked from README | Completed | `crates/ffai-mercury/docs/threat-model.md` — assets, 3 trust boundaries, full STRIDE table, the highest-value attack path, accepted risks. Linked from the README "Security" section | |
| H-02 | Threat model revisited after last major change | Completed | model written and dated 2026-08-15, i.e. current at HEAD; next review 2026-11-15 | |

### Phase 1 — Toolchain

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-03 | Toolchain pinned (`rust-toolchain.toml`) | Completed | `rust-toolchain.toml` pins channel 1.95.0 + rustfmt/clippy/rust-src/llvm-tools-preview, profile minimal; matches the workspace `rust-version` | |
| H-04 | Committed `.cargo/config.toml` hardening defaults | Completed | `.cargo/config.toml` gains a `cfg(all(target_os="linux", target_env="gnu"))` block — full RELRO, `-z now`, noexecstack, frame pointers. Target-scoped deliberately: a blanket `[build]` block would override the existing wasm section and break MSVC | |
| H-05 | ★ Release profile hardened (overflow-checks, LTO, panic policy) | Completed | `[profile.release]` now `overflow-checks = true`, `codegen-units = 1`, `lto = "thin"`. **Perf impact UNMEASURED** — integer-only and the hot paths are f32/f16 + intrinsics, but the benchmark gates must be re-run before merge (R-005) | |
| H-06 | Security toolchain available to CI and developers | Completed | `.github/workflows/harden.yml` installs the pinned tool set via `dtolnay/rust-toolchain`, `EmbarkStudios/cargo-deny-action@v2`, `rustsec/audit-check@v2` | |

### Phase 2 — Supply chain

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-07 | ★ `Cargo.lock` committed | Completed | `git ls-files Cargo.lock` -> tracked at the workspace root | |
| H-08 | ★ `deny.toml` policy present and enforced | Completed | `deny.toml` at the workspace root covers advisories + licenses + bans + sources. **`cargo deny check` ran clean: `advisories ok, bans ok, licenses ok, sources ok`.** Enforced per-PR by the `supply-chain` job | |
| H-09 | ★ Vulnerability scan clean (`cargo audit`) | Completed | `cargo deny check advisories` clean. 5 advisories found: **h2 RUSTSEC-2026-0258 FIXED** (0.4.15 -> 0.4.16); `paste`/`ttf-parser` unmaintained and pyo3 x2 carry dated scoped `ignore` justifications (pyo3 review 2026-09-30, others 2026-11-15) | |
| H-10 | ★ `cargo vet` coverage complete | Incomplete | no `supply-chain/` directory — cargo vet never initialised | |
| H-11 | Unsafe inventory measured and trending down (geiger) | Incomplete | not run (survey depth); `cargo-geiger` is not installed. No baseline exists to trend against | |
| H-12 | ★ SBOM generated and published with releases | Incomplete | no SBOM artifact in tree; no release job to attach one to | |
| H-13 | Git deps pinned; no unknown registries or sources | Completed | mercury's graph (ffai-core, ffai-models, candle-nn/-transformers, tokenizers, rustfft, rayon, half, serde_json) contains **no git dependencies**; verified none in the `ffai-core`/`ffai-models` manifests either. NOTE: the workspace declares four unpinned `rff-*` git deps (no `rev`), but they are not reachable from mercury — that is the root unit's finding, not this one | |
| H-14 | Dependency freshness reviewed, human-in-the-loop updates | Completed | `.github/dependabot.yml` — weekly cargo, monthly actions, patch updates grouped, 5-PR cap, **no auto-merge**: a bot that can merge is itself an injection path | |

### Phase 3 — Code level

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-15 | ★ Workspace lint policy set and clean | Incomplete | `[lints] workspace = true`, but `[workspace.lints.rust]` is only `unsafe_code = "warn"` — no clippy `all`/`pedantic`/`nursery`, no `unwrap_used` denials. Measured: `cargo clippy -p ffai-mercury --all-targets -- -D warnings -A unsafe_code` reports **266 findings** in this crate alone (approximate-PI constants, clamp patterns, doc formatting). `ffai-core` is now clean after fixing its one `derivable_impls`. CI runs clippy advisory-only with a dated TODO (2026-09-30) precisely because of this backlog | |
| H-16 | ★ `unsafe` isolated, SAFETY-commented, inventoried | Completed | `UNSAFE.md` inventories **all 28 sites** in 3 classes, and **all 28 now carry a `// SAFETY:` comment** (27 comments; the adjacent `unsafe impl Send`/`Sync` pair shares one). Each names the invariant and what upholds it — the dominating `have_avx2()`/`have_f16c()` check, the disjointness arithmetic, or (Class C) the fact that the mmap invariant is not enforceable from inside this crate. Note: `#[allow(unsafe_code)]` hides sites from the lint — 25 warn, 3 were invisible | |
| H-17 | Arithmetic safety explicit | Incomplete | 27 `checked_`/`saturating_`/`wrapping_` calls in `src/`, but `overflow-checks` is off in release (H-05), so release arithmetic wraps silently. `as` casts on lengths/indices unreviewed | |
| H-18 | ★ No `unwrap`/`expect`/panic on untrusted paths; typed errors | Incomplete | 126 `unwrap()`/`expect()` occurrences in `src/` (tests included; every file also carries `cfg(test)`, so the counts are not separable by grep). Heaviest: `tts/phonemize.rs` (20), `tts/decoder_kernels.rs` (17), `asr/audio_encoder.rs` (15). No per-callsite triage of the decode path has been done | |
| H-19 | Input validation — external bytes treated as hostile | Incomplete | the model cache is parsed **as trusted**: ONNX bytes (`tts/vits.rs:100`), `vits-graph.json` / `voice-config.json` (`vits.rs:138`, `:163`), binary lexicon (`tts/lexicon.rs:85`), `config.json` (`asr/model.rs:77`), and mmap'd safetensors. Verification lives upstream in `ffai-models` and is not enforced or asserted here | |
| H-20 | ★ Secrets zeroized; never logged | Incomplete | no key material in this unit (0 `zeroize`/`secrecy` references, correctly), **but** transcripts and audio are user-private content and log hygiene has not been reviewed. Narrow the gate to "no transcript/audio content at any log level" and evidence it | |
| H-21 | Concurrency discipline | Incomplete | two `unsafe impl Send`/`Sync` on `SendPtr(*mut f32)` at `tts/decoder_kernels.rs:194-195`, plus `from_raw_parts_mut` writes across rayon tasks in `decoder_kernels.rs` and `asr/flash_attn.rs`. The disjointness argument exists only as a comment — no Miri, no TSan, no proof | |

### Phase 4 — Static analysis

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-22 | Static analysis beyond the default linter runs on every PR | Incomplete | CI runs clippy, deny and audit — none of which is static analysis of the code beyond the default linter. No Semgrep / MIRAI / Rudra-style rules yet | |

### Phase 5 — Dynamic analysis

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-23 | ★ Tests pass under Miri | Incomplete | not run (survey depth). Constraint to plan for: Miri does not execute most x86 SIMD intrinsics, so the AVX2 kernels in `flash_attn.rs`/`f16_gemv.rs`/`vocab_int8.rs`/`decoder_kernels.rs` will need scalar-fallback test paths for Miri to say anything about them | |
| H-24 | Critical paths pass the sanitizers (ASan/MSan/TSan) | Incomplete | never run. TSan is the one that matters here — it is the tool that would test the `SendPtr` disjointness claim in H-21 | |
| H-25 | `cargo careful test` green | Incomplete | not run; `cargo-careful` is not installed | |

### Phase 6 — Fuzzing and properties

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-26 | ★ Fuzz target per public parser, decoder, or message handler | Incomplete | `crates/ffai-mercury/fuzz/` created with libFuzzer harnesses for the **two hand-rolled byte parsers** — `onnx_parse` (targets `tts::onnx::parse`, the hand-written protobuf reader and the highest-risk path in the threat model) and `lexicon_parse` — each with a seeded corpus. Not yet covered: `vits-graph.json`, `voice-config.json`, `config.json` and `phoneme_ids` (these delegate to `serde_json` but their interpretation is still unfuzzed). Harnesses are committed but **not yet executed** — cargo-fuzz needs nightly and is not installed here | |
| H-27 | ★ Continuous fuzzing with no open crashes | Incomplete | no campaign has run. Targets now exist (H-26), so this is a matter of running them: `cargo +nightly fuzz run onnx_parse`. Gate needs 30 days of coverage-guided fuzzing with no open crashers | |
| H-28 | Property tests cover the documented invariants | Incomplete | 0 `proptest`/`quickcheck` references in `src/`. The crate advertises **byte-stable determinism** (the 0.7.0 version bump exists precisely because synthesis output changed) — that is a documented invariant with no property test behind it | |
| H-29 | Mutation and/or differential testing on critical modules | Incomplete | no `cargo-mutants` config and no differential harness in CI. WER/parity benchmarking against reference implementations exists in the bench crate, but it is a quality metric, not a differential correctness harness | |

### Phase 7 — Formal verification

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-30 | Proof of panic-freedom / UB-freedom per `unsafe` module | Incomplete | 0 `kani::proof` harnesses against ~30 `unsafe` sites | |

### Phase 8 — Build and binary

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-31 | ★ Binary hardening applied and verified | N/A | library crate — no `[[bin]]` target, ships no binary. Applies to `ffai-cli`, not here | |
| H-32 | Build is reproducible or fully auditable | N/A | library crate — no release binary to attest. `cargo auditable` applies at the binary that links this crate | |

### Phase 9 — Runtime privilege

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-33 | Least privilege documented and tested | N/A | library crate — holds no process, so it cannot drop privilege. Sandboxing is the consuming binary's gate | |

### Phase 10 — Cryptography

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-34 | Vetted crypto only; no bespoke primitives | N/A | no crypto dependencies and no crypto code in this unit; weight-hash verification lives in `ffai-models` | |
| H-35 | Side-channel discipline (constant-time, no secret branches) | N/A | no secret-dependent code paths — the unit holds no key material | |
| H-36 | Post-quantum migration plan for long-lived keys | N/A | holds no keys of any lifetime | |

### Phase 11 — CI/CD, release, and operations

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-37 | CI runs the hardening gate on every PR | Completed | `.github/workflows/harden.yml` runs fmt, clippy `-D warnings`, test, cargo-deny, cargo-audit and a hardening-table freshness check on every push and PR, plus a weekly cron. `permissions: contents: read` | |
| H-38 | Releases signed, attested, and changelogged for security | Incomplete | no release workflow; tag signing unverified; no security section in a changelog | |
| H-39 | ★ `SECURITY.md` with a coordinated disclosure process | Incomplete | absent at the repo root and at the crate. The crate is published to crates.io as 0.7.0 with no disclosure contact | |
| H-40 | Advisory monitoring and scheduled re-audit | Completed | weekly `schedule:` cron re-runs the advisory scan without needing a push; re-audit date 2026-11-15 recorded in this file | |
| H-41 | ★ Residual risks listed and accepted; waivers time-bounded | Incomplete | register drafted below, but **no risk has been accepted by an owner** — acceptance is a human decision, not an audit output | |

### Phase 12 — Compliance controls

Scope declared above: `gdpr-pii`. Rows marked `N/A` are out of scope because this is a
**library** that performs no network egress, persists nothing, and enforces no access
control — those controls belong to the consuming service. Mapping: the skill's
`COMPLIANCE.md`.

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| C-01 | Data inventory — personal/health/card data touched | Completed | `docs/data-inventory.md` — 7 data classes with sensitivity, entry point and lifetime; D3 (speaker embeddings) explicitly identified as GDPR Art 9 special-category biometric data | |
| C-02 | Data-flow map including third-party egress | Completed | flow map in the same document: every class traced caller-to-return. **No egress** — no sockets, no disk writes; weight download belongs to `ffai-models` | |
| C-03 | Encryption in transit for all egress | N/A | this unit performs no network egress; weight download lives in `ffai-models` | |
| C-04 | Encryption at rest for stored sensitive data | N/A | persists no personal data; it reads a model cache it does not own | |
| C-05 | Key management — generation, storage, rotation, destruction | N/A | holds no key material (consistent with H-20) | |
| C-06 | Retention limits and honoured deletion | Incomplete | no persistence found at survey depth, but whether any temp file, buffer, or cache retains audio or embeddings is unreviewed — and Art 17 erasure has to reach embeddings, not just recordings | |
| C-07 | Audit logging of security-relevant events | N/A | library; audit logging is the consuming service's control | |
| C-08 | Log hygiene — no PII, secrets, or card data in logs | Completed | audited every `eprintln!` in `src/`: all content-bearing output (transcript tokens `decoder.rs:274`, diarizer traces, VAD windows) sits behind opt-in env flags (`FFAI_DEBUG_TOKENS`, `FFAI_DIARIZE_TRACE`). The two ungated sites emit timings (profiler-gated) and a count. **Nothing content-bearing is emitted by default** | |
| C-09 | Least-privilege access to sensitive data | N/A | library; enforces no access control boundary of its own | |
| C-10 | Subprocessor and third-party inventory | N/A | no personal data leaves the process, so no processor relationship arises from this unit | |
| C-11 | Incident response and breach notification path | Completed | `SECURITY.md` at the repo root: private GitHub advisory reporting, 3-day ack / 10-day assessment / 90-day fix, coordinated disclosure, and an explicit **GDPR Art 33 72-hour clause** for reports involving personal data | |
| C-12 | Change management — reviewed, approved, traceable | Incomplete | CI now runs on every PR, but branch protection and a required-review rule are **not** configured on the repo, so traceability is convention rather than enforcement — a settings change, not a code change | |
| C-13 | Availability commitments and their evidence | N/A | library; no SLA or availability commitment is made by this crate | |
| C-14 | Machine-readable SBOM + provenance for regulators | Incomplete | no SBOM published (H-12) | |

---

## Scheduled work

**Nothing is `Scheduled` yet, deliberately.** `Scheduled` requires an owner and a date
(SKILL.md §1.4), and neither is the auditor's to assign. The ordering below is a proposal;
it becomes a schedule when you put names and dates in and move the rows' Status.

Cheapest-first, because the configuration gates unblock the outcome gates behind them:

| # | Gates | Work | Owner | Target | Notes |
|---|---|---|---|---|---|
| 1 | H-39 | Add `SECURITY.md` with a disclosure contact and window | | | ~15 min; the crate is already public on crates.io without one |
| 2 | H-05 | Add `overflow-checks = true` (+ `codegen-units`) to `[profile.release]` | | | one line; clears a ★ gate and changes H-17's meaning |
| 3 | H-08, H-09 | Add `deny.toml`, install `cargo-audit`, run both | | | `cargo-deny` is already installed; two ★ gates |
| 4 | H-37, H-06, H-22 | Create `.github/workflows/harden.yml` (fmt, clippy, test, deny, audit) | | | unblocks four gates that currently have nothing to point at |
| 5 | H-15 | Extend `[workspace.lints]` to clippy pedantic/nursery + `unwrap_used` | | | expect a large first-run backlog; land the policy at `warn` first |
| 6 | H-16 | Write `UNSAFE.md`; SAFETY-comment the ~23 uncommented sites | | | inventory first, isolation later — moving the kernels is a bigger job |
| 7 | H-19, H-26 | Assert weight-hash verification at mercury's boundary; add a fuzz target for the ONNX/JSON loaders | | | the highest-severity finding; see R-001 |
| 8 | H-21, H-24 | Run TSan against the `SendPtr` parallel writes | | | tests the one claim currently backed only by a comment |
| 9 | H-23 | Miri, with scalar fallbacks for the SIMD paths | | | needs the test-path work noted in H-23 |

---

## Residual risk register

Every open risk needs an owner, an acceptance, and a review date (H-41). **None of these
have been accepted yet** — that is why H-41 is `Incomplete`.

| ID | Risk | Likelihood | Impact | Mitigation status | Accepted by | Review date |
|---|---|---|---|---|---|---|
| R-001 | Model cache is parsed as trusted. Anyone who can write to the cache directory controls ONNX/JSON/lexicon bytes and a live mmap; hash verification is upstream in `ffai-models` and unasserted here | Medium (local write access, or a hole upstream) | High (mmap UB is a memory-safety primitive) | Open — no verification, no fuzzing at this boundary | | |
| R-002 | `SendPtr` disjointness is argued in a comment, not proven. A wrong band calculation is a data race in released code, not a panic | Low | High (UB, silently wrong audio) | Open — no TSan, no Miri, no proof | | |
| R-003 | `overflow-checks` off in release with 126 unwrap/expect sites and unreviewed `as` casts: arithmetic wraps silently, and a panic on untrusted input is a DoS | Medium | Medium (wrong output or crash, not memory unsafety) | Open — H-05 is a one-line fix; H-18 triage is not | | |
| R-004 | No CI at all: every gate that could regress silently, does. Nothing re-checks fmt, clippy, tests, or advisories on any commit | High | Medium | Open — H-37 | | |

---

## Waivers

Time-bounded only. An expired waiver is an `Incomplete` gate, not a `Completed` one.

| Gate | Reason | Granted by | Expires |
|---|---|---|---|
| | | | |

---

## Audit log

Append one line per pass; never rewrite history. The trend is the point.

| Date | Depth | Auditor | Completed / Scheduled / Incomplete | ★ met | Note |
|---|---|---|---|---|---|
| 2026-08-15 | survey | Claude (pilot) | 2 / 0 / 33 (6 N/A) | 1/16 | first pass; pilot of the `use-protection-please` skill. No tool probe executed — every outcome gate is `Incomplete` for want of evidence, not for a known failure |
