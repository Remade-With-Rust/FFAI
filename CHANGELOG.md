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

### Security

- **Fixed** — `MelSpectrogram::compute(&[])` panicked with an out-of-bounds
  index in `reflect_pad`. Caller-supplied audio reaches this path, so an empty
  buffer was a denial of service in the embedding process. Found by the new
  property tests; the input is now a permanent fuzz regression seed.
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

- `ffai-mercury`'s lib is clippy-clean and rustfmt-clean; CI blocks on both for
  that crate. Behaviour verified unchanged: 128 lib + 7 property tests pass,
  `cargo-careful` green, and the `cargo-geiger` unsafe surface is identical.
