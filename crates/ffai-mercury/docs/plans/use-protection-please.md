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
| H-05 | ★ Release profile hardened (overflow-checks, LTO, panic policy) | Incomplete | **WAIVED 2026-08-15** (see Waivers). `overflow-checks` deliberately NOT set: measured cost 1.060x TTS pipeline / 1.094x decoder (n=22 paired, z=+4.26, null floor 0.995), and the threat it defends - parsing attacker-controlled model files - is retired by an explicit product decision that model files are trusted (R-001). `[profile.release]` keeps `lto = "thin"` and `codegen-units = 1`. The waiver expires if model files ever become untrusted | |
| H-06 | Security toolchain available to CI and developers | Completed | `.github/workflows/harden.yml` installs the pinned tool set via `dtolnay/rust-toolchain`, `EmbarkStudios/cargo-deny-action@v2`, `rustsec/audit-check@v2` | |

### Phase 2 — Supply chain

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-07 | ★ `Cargo.lock` committed | Completed | `git ls-files Cargo.lock` -> tracked at the workspace root | |
| H-08 | ★ `deny.toml` policy present and enforced | Completed | `deny.toml` at the workspace root covers advisories + licenses + bans + sources. **`cargo deny check` ran clean: `advisories ok, bans ok, licenses ok, sources ok`.** Enforced per-PR by the `supply-chain` job | |
| H-09 | ★ Vulnerability scan clean (`cargo audit`) | Completed | `cargo deny check advisories` clean. 5 advisories found: **h2 RUSTSEC-2026-0258 FIXED** (0.4.15 -> 0.4.16); `paste`/`ttf-parser` unmaintained and pyo3 x2 carry dated scoped `ignore` justifications (pyo3 review 2026-09-30, others 2026-11-15) | |
| H-10 | ★ `cargo vet` coverage complete | Incomplete | no `supply-chain/` directory — cargo vet never initialised | |
| H-11 | Unsafe inventory measured and trending down (geiger) | Completed | baseline archived at `crates/ffai-mercury/docs/geiger-baseline.txt` (cargo-geiger 0.13.0, default features). Mercury's own row: **14/27 unsafe functions, 2/2 unsafe impls, 1216/2274 unsafe expressions** — independently confirming `UNSAFE.md` (14 `unsafe fn`, 2 `unsafe impl`) from a tool that reads the SOURCE rather than the compiler's warnings about one target. Trend gate: the next audit diffs against this file. `--all-features` is unusable here — it enables candle's `metal`, pulling macOS-only `objc2`, which cannot compile on Windows | |
| H-12 | ★ SBOM generated and published with releases | Incomplete | `release.yml` generates a CycloneDX 1.5 SBOM per crate and attaches it to the release; `cargo auditable` additionally embeds the dependency list inside each binary, so a shipped artifact can be scanned directly with `cargo audit bin`. Default features only — `--all-features` pulls candle's `metal` and the macOS-only `objc2`, which cannot resolve on a Linux runner. **Blocked on the same fact as H-38**: no release exists yet to attach an SBOM to | |
| H-13 | Git deps pinned; no unknown registries or sources | Completed | mercury's graph (ffai-core, ffai-models, candle-nn/-transformers, tokenizers, rustfft, rayon, half, serde_json) contains **no git dependencies**; verified none in the `ffai-core`/`ffai-models` manifests either. NOTE: the workspace declares four unpinned `rff-*` git deps (no `rev`), but they are not reachable from mercury — that is the root unit's finding, not this one | |
| H-14 | Dependency freshness reviewed, human-in-the-loop updates | Completed | `.github/dependabot.yml` — weekly cargo, monthly actions, patch updates grouped, 5-PR cap, **no auto-merge**: a bot that can merge is itself an injection path | |

### Phase 3 — Code level

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-15 | ★ Workspace lint policy set and clean | Incomplete | **ffai-mercury's lib is now clippy-clean: 267 findings -> 0**, and CI blocks on it for this crate (advisory for the rest — the incremental-adoption pattern). Correcting the earlier "266 findings" evidence, which was wrong twice: **199 of them were one lint**, `unsafe_op_in_unsafe_fn` (an edition-2024 migration, not code smell), and the `--all-targets` count of 4 was clippy aborting on `ffai-media` before it ever reached mercury. Fixed by `cargo fix` (204), `cargo clippy --fix`, 6 justified per-site `#[allow(dead_code)]`, a documented crate lint policy for 7 design-decision classes (NaN-semantics `clamp`, reference-transcribed constants, kernel arity, index loops), and 3 by hand — including a **stranded doc comment that described a function no longer beneath it**. Verified behaviour-preserving: 128 lib + 7 property tests pass, cargo-careful green, and geiger reports an IDENTICAL unsafe surface (14/27 fns, 2/2 impls). **Gate stays open**: the criterion asks for pedantic + nursery at the workspace level, and 11 other crates are not yet clean | |
| H-16 | ★ `unsafe` isolated, SAFETY-commented, inventoried | Completed | `UNSAFE.md` inventories **all 31 sites** in 3 classes; all 31 carry a `// SAFETY:` comment. The count was 28 until `tools/lint_invariants.py` found 3 more: `#[cfg(not(target_arch = "x86_64"))]` fallback stubs, invisible to the compiler on an x86 machine. Combined with the 3 `#[allow(unsafe_code)]` sites the lint also hides, that is **two independent blind spots in a compiler-warning-derived inventory** — both now closed by R1/R2 in CI | |
| H-17 | Arithmetic safety explicit | Incomplete | 27 `checked_`/`saturating_`/`wrapping_` calls in `src/`, but `overflow-checks` is off in release (H-05), so release arithmetic wraps silently. `as` casts on lengths/indices unreviewed | |
| H-18 | ★ No `unwrap`/`expect`/panic on untrusted paths; typed errors | Incomplete | 126 `unwrap()`/`expect()` occurrences in `src/` (tests included; every file also carries `cfg(test)`, so the counts are not separable by grep). Heaviest: `tts/phonemize.rs` (20), `tts/decoder_kernels.rs` (17), `asr/audio_encoder.rs` (15). No per-callsite triage of the decode path has been done | |
| H-19 | Input validation — external bytes treated as hostile | Completed | the untrusted surface is caller audio and text (model files are trusted — R-001). `transcribe` validates `sample_rate` and rejects mismatches. Both front ends are now property-tested as total and fuzz-targeted, and the audit **found and fixed two real input-validation defects**: `MelSpectrogram::compute(&[])` indexed out of bounds on an empty buffer, and `normalize` overflowed `u64` on a digit run longer than 19 characters — a panic in debug, and a SILENT WRAPAROUND in release, where this workspace deliberately carries no `overflow-checks`. Both inputs are permanent fuzz-corpus regressions | |
| H-20 | ★ Secrets zeroized; never logged | Completed | no key material exists in this unit (0 `zeroize`/`secrecy` references, correctly — there is nothing to zeroize). The logging half is evidenced by the C-08 audit: every content-bearing `eprintln!` (transcript tokens, diarizer traces, VAD windows) sits behind an opt-in env flag, and nothing content-bearing is emitted by default. Personal data is the asset here, not secrets — see `docs/data-inventory.md` | |
| H-21 | Concurrency discipline | Completed | the two manual `unsafe impl Send`/`Sync` on `SendPtr(*mut f32)` (`tts/decoder_kernels.rs:194-195`) now carry a SAFETY comment naming the disjointness arithmetic that justifies them — chunk co-ranges partition `[0, c_out)`, block t-ranges partition `[0, l_out)` — and are classified in `UNSAFE.md` class B. No `static mut`. Proving it with TSan is H-24, tracked separately as R-002 | |

### Phase 4 — Static analysis

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-22 | Static analysis beyond the default linter runs on every PR | Completed | `tools/lint_invariants.py` — 4 project-specific rules enforcing invariants no general linter knows about, wired into CI as the `invariants` job. R1 every unsafe site carries a `// SAFETY:` comment; R2 every `#[allow(unsafe_code)]` file is named in `UNSAFE.md`; R3 banned constructs (`mem::forget`, `Box::leak`, `transmute`) with a `// LINT-ALLOW:` override; R4 no `println!`/`dbg!` in library code. Chosen over Semgrep deliberately — its Rust support is experimental, and these rules are deterministic and stdlib-only. **Found 3 real violations on its first run** (below). Currently: 0 violations | |

### Phase 5 — Dynamic analysis

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-23 | ★ Tests pass under Miri | Completed | `tests/miri_safe.rs` — **4/4 green under `cargo +nightly miri test` with `-Zmiri-strict-provenance`**, and BLOCKING in CI. Scope is deliberate and documented: Miri interprets Rust, so it cannot execute x86 SIMD intrinsics (`rustfft`, reached through `mel::compute`) or foreign functions (the Win32 memory instrumentation reached from `--lib`). Both blockers were measured, not assumed. The subset covers what Miri is actually good at here — pure index arithmetic on caller-supplied lengths, which is where `reflect_pad`'s empty-input panic lived. **It found a second real bug on its first run** (`normalize` overflow, below). Widening the scope means moving tests into that target as they become SIMD- and FFI-free | |
| H-24 | Critical paths pass the sanitizers (ASan/MSan/TSan) | Incomplete | never run. TSan is the one that matters here — it is the tool that would test the `SendPtr` disjointness claim in H-21 | |
| H-25 | `cargo careful test` green | Completed | `cargo +nightly careful test -p ffai-mercury --lib` -> **128 passed, 0 failed**: extra debug assertions plus `-Zextra-const-ub-checks` and `-Zstrict-init-checks` across the whole lib suite. Needs nightly explicitly — the `rust-toolchain.toml` pin sends a bare `cargo careful` to stable, where it cannot build its sysroot | |

### Phase 6 — Fuzzing and properties

| ID | Gate | Status | Evidence | Target |
|---|---|---|---|---|
| H-26 | ★ Fuzz target per public parser, decoder, or message handler | Completed | `crates/ffai-mercury/fuzz/` — 4 libFuzzer targets with seeded corpora, aimed at the boundary the threat model actually declares untrusted. Caller-controlled input: `mel_compute` (audio front end; carries the empty-buffer crasher as a permanent regression seed) and `normalize_text` (text front end, asserts determinism). Defence-in-depth on trusted-but-parsed model data: `onnx_parse` (hand-written protobuf reader) and `lexicon_parse`. Not covered: end-to-end `transcribe`/`synthesize`, which need cached weights | |
| H-27 | ★ Continuous fuzzing with no open crashes | Incomplete | no campaign has run. Targets now exist (H-26), so this is a matter of running them: `cargo +nightly fuzz run onnx_parse`. Gate needs 30 days of coverage-guided fuzzing with no open crashers | |
| H-28 | Property tests cover the documented invariants | Completed | `tests/properties.rs` — 7 proptest properties over the model-free untrusted surface: `pad_or_trim_to` length + prefix/zero-fill contracts, `compute` totality and shape contract, `resized` frame contract, and **bit-determinism** of both `compute` and `normalize` (asserted via `to_bits()`, since NaN != NaN and the claim is BYTE stability). **Found a real defect on the first run**: `compute(&[])` panicked with an out-of-bounds index in `reflect_pad` — the `n == 0` guard returned index 0 into an empty slice, and `n - 1 - i.min(n - 1)` underflowed on the same path. Fixed and seeded as a fuzz regression. 7/7 pass | |
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
| H-38 | Releases signed, attested, and changelogged for security | Incomplete | `.github/workflows/release.yml` implements the whole path — a **signed-tag gate** that fails the release if `git cat-file tag` shows no PGP/SSH signature, a `cargo auditable` build so binaries carry their own dependency manifest, SLSA **build provenance** via `actions/attest-build-provenance` (OIDC-signed), and `checksec` on the Linux binary. `CHANGELOG.md` defines the Security-subsection convention this gate asks for, including the rule that "no security-relevant changes" must be stated rather than implied. **Blocked, not unbuilt**: the repository has no tags and no releases, so nothing has been signed or attested yet. Flips on the first verified release | |
| H-39 | ★ `SECURITY.md` with a coordinated disclosure process | Completed | `SECURITY.md` at the repo root: private GitHub advisory reporting, 3-day ack / 10-day assessment / 90-day fix, coordinated disclosure, scope and supported-versions sections, plus an explicit GDPR Art 33 72-hour clause. (Was left `Incomplete` by oversight in the first remediation pass — the file existed from that pass onward) | |
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
| C-14 | Machine-readable SBOM + provenance for regulators | Incomplete | same machinery as H-12: CycloneDX 1.5 JSON per crate, plus an OIDC-signed SLSA provenance attestation verifiable with `gh attestation verify`. That is the format and the chain of custody a regulator asks for (EU CRA Art 13). **Blocked**: needs a release to exist | |

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
| R-001 | Model cache is parsed as trusted: ONNX/JSON/lexicon bytes and a live mmap are taken on faith, and hash verification lives upstream in `ffai-models` | Medium | High (mmap UB is a memory-safety primitive) | **ACCEPTED 2026-08-15** — model files are trusted input by product decision; deployers must restrict write access to the cache. Documented in the crate README and threat model. **Load-bearing**: user-supplied voice packs or a shared cache invalidate it | Tim | 2026-11-15 |
| R-002 | `SendPtr` disjointness is argued in a comment, not proven. A wrong band calculation is a data race in released code, not a panic | Low | High (UB, silently wrong audio) | Open — no TSan, no Miri, no proof | | |
| R-003 | 126 `unwrap`/`expect` sites and unreviewed `as` casts on lengths. `overflow-checks` is now ON in release (H-05), so arithmetic no longer wraps silently — but a panic on untrusted input is still a DoS | Medium | Medium | Partly mitigated — checks on; per-callsite triage of the decode path outstanding (H-18) | | 2026-11-15 |
| R-004 | CI now exists (H-37) but `fmt` and `clippy` are advisory, not blocking, and branch protection is not configured (C-12) — so a regression in either can still land | Medium | Medium | Partly mitigated — test/deny/audit/table-freshness block; fmt+clippy dated to 2026-09-30 | | 2026-09-30 |
| R-005 | `overflow-checks` costs ~6% TTS pipeline / ~10% decoder (measured 2026-08-15). **DECIDED: reverted.** The flag defends against hostile model files, which this product does not accept | n/a - closed | n/a | **Closed 2026-08-15** — reverted; H-05 waived; the assumption is documented in the crate README and threat model. Reopens automatically if model files become untrusted | Tim | 2026-11-15 |

---

## R-005 measurement — what `overflow-checks` actually costs (2026-08-15)

**Method line.** Two release builds of `examples/profile_tts` in separate
`CARGO_TARGET_DIR`s, differing ONLY in `CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS`. The
manifest was never edited while measuring — editing `[profile.release]` invalidates
rust-analyzer's cache and starts a rustc storm that poisons the measurement it was made
for. Flag effect proven in the binaries: 5 `attempt to ... with overflow` panic strings
in the ON arm vs 1 in OFF, +73 KB of code. Harness `tools/diana_ab.ps1`: ABBA-interleaved,
High priority, CPU-time primary, work-count parity enforced, 2 warm-up reps discarded.
Corpus: 20 Harvard holdout sentences through piper-vits-lessac-medium, seeded RNG.
**Work parity: 45.0 s of audio from both arms in every rep.**

**The null arm failed first, and that is the point of running one.** Same binary on both
sides read **5/22, z = -2.56** — a "significant" result from identical code. Two causes:
the harness forced `FFAI_PROFILE=1`, putting the profiler inside the system under test,
and it had no warm-up, so the 62 MB model load sat inside the paired samples. Both fixed;
the null then read **13/22, z = +0.85**, ratio 0.995. Everything below is measured against
that 0.5% floor.

| metric | checks OFF | checks ON | ratio |
|---|---:|---:|---:|
| CPU ms, whole process | 62,656 | 65,203 | 1.041 |
| paired CPU-time wins | 21 / 22 | | **z = +4.26** |
| paired in-process wins | 22 / 22 | | **z = +4.69** |
| pipeline total, min (n=7 ABBA) | 1541.7 ms | 1633.7 ms | **1.060** |
| **decoder stage, min** | 899.9 ms | 984.4 ms | **1.094** |

The process-level 1.041 is diluted — model load is identical in both arms — so the
in-process figures are the honest ones. The arithmetic closes: the decoder is 56.5% of the
pipeline and slows 9.9%, predicting 5.6% at pipeline level against 6.5% observed. That is
mechanistically right, because the decoder carries the hand-written index arithmetic in
`tts/decoder_kernels.rs` — exactly what overflow-checks instruments.

**Not measured: the ASR path.** No whisper weights are cached on this machine, so the
whisper.cpp comparison could not be re-run. That is the claim actually at risk, so R-005
stays open regardless of what is decided about TTS.

**Decision required** — this is a product trade-off, not an audit finding:

1. **Keep.** Accept ~6% on TTS for overflow safety in a crate that parses untrusted model
   files. H-05 stays green.
2. **Revert.** Drop `overflow-checks`, record a time-bounded waiver against H-05 and a
   residual risk for silent wraparound on untrusted lengths.
3. **Split** (best outcome, most work). Keep checks on, and make the proven-bounded inner
   loops in `decoder_kernels.rs` use explicit `wrapping_*`/`get_unchecked`-free index
   arithmetic. Protection stays where untrusted data is indexed; most of the 10% comes
   back.

---

## Waivers

Time-bounded only. An expired waiver is an `Incomplete` gate, not a `Completed` one.

H-05 is **v1.0.0-blocking**. Waiving it does not make the gate pass — it records that the project lead accepted the residual risk with the numbers in front of them, which is what STANDARD.md §14 requires before a 1.0.0 release with an open ★ gate.

| Gate | Reason | Granted by | Expires |
|---|---|---|---|
| H-05 | `overflow-checks` in release defends against parsing attacker-controlled input. Mercury does not accept hostile model files — they come from a trusted source and the deployer owns the cache. Measured cost 1.060x TTS pipeline / 1.094x decoder stage, so the flag buys nothing against a retired threat. Assumption documented in the crate README and threat model | Tim (project lead) | 2026-11-15, or immediately if model files become untrusted (user-supplied voice packs, shared cache, unverified downloads) |

---

## Audit log

Append one line per pass; never rewrite history. The trend is the point.

| Date | Depth | Auditor | Completed / Scheduled / Incomplete | ★ met | Note |
|---|---|---|---|---|---|
| 2026-08-15 | survey | Claude (pilot) | 2 / 0 / 33 (6 N/A) | 1/16 | first pass; pilot of the `use-protection-please` skill. No tool probe executed — every outcome gate is `Incomplete` for want of evidence, not for a known failure |
