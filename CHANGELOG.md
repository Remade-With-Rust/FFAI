# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: each crate versions independently — `ffai-mercury` is ahead of the
workspace on purpose (see its manifest).

## Every release states its security position

Hardening gate **H-38** requires that security-relevant changes are called out
rather than buried in a feature list. A release entry therefore carries a
**Security** subsection whenever any of the following changed, and says "no
security-relevant changes" when none did — silence is ambiguous, and a reader
cannot tell "nothing changed" from "nobody looked":

- a CVE fixed, a dependency advisory resolved, or an advisory newly waived;
- a change to `unsafe` code, or to the invariants in a crate's `UNSAFE.md`;
- a change to what is trusted — the model-cache trust decision is the live
  example (see `crates/ffai-mercury/docs/threat-model.md`);
- a change to what personal data is processed (`docs/data-inventory.md`);
- a hardening gate moving to or from `Completed`, or a waiver granted/expiring.

Each release also ships an SBOM and a signed build provenance attestation; the
mechanics are in `.github/workflows/release.yml`.

---

## [Unreleased]

Nothing yet.

---

## [1.0.0] — 2026-08-19 — `ffai-mercury`

First stable release. Mercury has been through a full 41-gate hardening audit
(`crates/ffai-mercury/docs/plans/use-protection-please.md`): **93%, 39 gates
Completed, 14 of 16 v1.0.0-blocking gates met**, and the two that are not —
H-05 and H-10 — are formally **waived with named owners and expiry dates**, not
quietly passed.

**What 1.0.0 does and does not promise.** The public API — `WhisperCandle`,
`PiperCandle`, `AnyTts`, `OxiWhisper` — is stable under semver from here. It is
expressed in `ffai-core` types (`AsrEngine`, `TtsEngine`, `AudioBuffer`,
`Transcript`), and **`ffai-core` remains 0.6.x**: a breaking change there would
reach this crate's surface without a major-version signal from cargo. That is a
known limitation of shipping Mercury alone, and it is stated rather than
discovered. `ffai-core` was not audited; only Mercury was, and only an audited
crate gets a 1.0 from this project.

### Security

- **Fixed** — an **out-of-bounds read inside `unsafe`** in `asr::vocab_int8`'s
  `dot_i8_blocked`. The inner loop tested `o < blk` but read 32 bytes per step,
  so a `blk` not a multiple of 32 read past the block and, on the final block,
  past the allocation. The guard checked `d % blk == 0` and not `blk % 32 == 0`,
  directly beneath a comment stating the kernel steps 32 lanes at a time.
- **Fixed** — **`cargo audit` had never actually executed in CI.** The
  `rustsec/audit-check` action ran inside a musl container that could not resolve
  this repo's pinned toolchain and failed before scanning. Running it directly
  immediately surfaced two live advisories: **RUSTSEC-2025-0020** (buffer-overflow
  risk in pyo3's `PyString::from_object`) and **RUSTSEC-2026-0177** (missing
  `Sync` bound on `PyCFunction::new_closure`). pyo3 0.23.5 → 0.29.2.
- **Fixed** — **CI was running no integration tests at all.** The test job read
  `cargo test --workspace --lib` under a comment claiming "Lib + integration
  tests". Now `--lib --tests`: **369 tests across 22 binaries**.
- **Fixed** — `cargo fuzz cmin` **silently deleted all twelve named regression
  seeds**, including the empty-input seed for the `reflect_pad` panic. Restored,
  and a test now fails if any goes missing.
- **Fixed** — `ffai-mercury` **did not compile on any non-x86_64 target**.
  `flash_head_strided` and `xattn_head_f16` had no non-x86 stubs while their call
  sites were unconditional. Every developer box and CI job here is x86_64; the
  first aarch64 build in project history found it.
- **Fixed** — the release pipeline's `verify-tag` gate **rejected a correctly
  signed tag**, because `actions/checkout` leaves a detached HEAD and
  `git cat-file tag <name>` resolved nothing. A gate that fails valid input is as
  corrosive as one that passes invalid input.
- **Verified** — **602,793,111 fuzz executions, zero crashes** across four
  targets (H-27, met via the criterion's equivalent-compute clause: 1,320
  target-minutes against a 30-night budget of 1,200). Corpus 17 → 1,407 inputs,
  replayed on every PR.
- **Verified** — **13/13 Kani bounded proofs** execute and verify, blocking in CI
  (H-30). TSan and ASan clean over 136 tests (H-24).
- **Supply chain** — `cargo vet` exemptions **473 → 304**; all nine registry
  peers imported, trusted publishers 128 → 167, nine crates certified after
  reading their complete diffs. **H-10 is NOT met**: 304 dependencies remain
  exempted rather than certified, waived to 2026-11-15 with compensating controls
  (`cargo audit` per PR, `deny.toml`, committed `Cargo.lock`, `cargo vet check`
  passing). Certifying the rest requires a person to read the code and sign for
  it; those attestations were not manufactured.

### Security — carried forward, open

- **R-006** — `ffai-carmenta` and `ffai-diana` allow three narrowing-cast lints
  wholesale across 187 unreviewed sites, on image and document parsers. The
  identical review found four real defects in Mercury. Deferred by the project
  lead to a planned repo-wide FFai audit. **This is why only Mercury is 1.0.0.**

### Security — from the preceding audit rounds

- **Fixed** — `MelSpectrogram::compute(&[])` panicked with an out-of-bounds
  index in `reflect_pad`. Caller-supplied audio reaches this path, so an empty
  buffer was a denial of service in the embedding process. Found by the new
  property tests; the input is now a permanent fuzz regression seed.
- **Fixed** — the diarizer's embedding cache had no working erasure path.
  `clear_embed_cache_counters()` was documented as "Drop every cached embedding"
  and only reset two atomic counters. The cache holds up to 512 **speaker
  embeddings** — voiceprints, GDPR Art 9 special-category data. Added
  `Diarizer::clear_embed_cache()` and `embed_cache_len()`; both regression-tested.
  Integrators honouring an Art 17 erasure must clear it: deleting a recording does
  not delete a voiceprint derived from it.
- **Fixed** — `tts::onnx::parse` accepted tensors whose declared shape did not
  describe their data. `dims.iter().product()` was an unchecked multiply over
  `i64 as usize`: `[1<<32, 1<<32]` wrapped to exactly 0 and matched an empty
  payload, and a negative dim (ONNX's `-1`) became colossal. Debug panicked;
  release — which carries no `overflow-checks` — accepted the tensor, and
  downstream code indexes `data` through `dims`.
- **Fixed** — protobuf length `u64 as usize` truncated on 32-bit targets, and
  `ffai-wasm` makes wasm32 real: a declared length of 2^32+5 truncated to 5 and
  passed the bounds check.
- **Fixed** — `normalize` overflowed `u64` on digit runs longer than 19
  characters (panic in debug, silent wraparound in release) and dropped leading
  zeros on the digit-by-digit path.
- **Fixed** — `h2` `RUSTSEC-2026-0258` (unbounded empty DATA frames), reached
  through `hf-hub → hyper` on the weight-download path. Updated 0.4.15 → 0.4.16.
- **Changed** — model files are now explicitly **trusted input**. Release builds
  therefore carry no `overflow-checks` (measured cost: 1.060x on the TTS
  pipeline, 1.094x on the decoder stage). Gate H-05 is waived, time-bounded, and
  the waiver expires the moment model files stop being trusted. Deployers must
  restrict write access to the model cache — see the crate README.
- **Added** — `SECURITY.md` with a coordinated-disclosure process and an explicit
  GDPR Art 33 72-hour clause for reports involving personal data.
- **Added** — supply-chain policy (`deny.toml`); `cargo deny check` is clean and
  runs per-PR. Four advisories carry dated, scoped waivers.
- **Added** — a threat model, a data inventory recording that diarization derives
  **GDPR Art 9 special-category voiceprints**, and an `UNSAFE.md` covering all 31
  unsafe sites in `ffai-mercury`.
- **Added** — CI hardening gate, project invariant lint, property tests, and
  fuzz targets aimed at the caller-controlled boundary.

### Changed

- `ffai-mercury` gains a default-on `fetch` feature. Building with
  `--no-default-features` drops **166 of 320 dependencies** and removes
  `aws-lc-sys` — a C crypto library — from the graph entirely, by removing the
  weight downloader (`hf-hub → reqwest → rustls → aws-lc-rs`) rather than the
  crypto. The default is unchanged, so inference performance is untouched.

- `ffai-mercury`'s lib is clippy-clean and rustfmt-clean; CI blocks on both for
  that crate. Behaviour verified unchanged: 128 lib + 7 property tests pass,
  `cargo-careful` green, and the `cargo-geiger` unsafe surface is identical.
