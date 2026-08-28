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

### `ffai-carmenta` 0.10.0 — the detector's short side gains a ceiling

- **Changed** — `mobiledet_input` floored the short side at 736 and capped only
  the LONG side, at 4000. Nothing brought a large image DOWN, so a 3000x4000
  phone photo reached the detector at **12 megapixels**, every pixel of it above
  the resolution the network works at. `max_short` (`FFAI_DET_MAX_SHORT`, or
  `image::set_det_max_short` where there is no environment) is the missing
  ceiling, default **1280**; the short side is now bounded on both ends.

  This changes output on any image whose short side exceeds 1280, which this
  crate versions as a compatibility break. `FFAI_DET_MAX_SHORT=4294967295`
  restores 0.9.x.

  **1280 and not the 736 a single page suggested.** One document page measured
  word-for-word identical at 736 and ~16 % faster, which opened the question;
  `examples/det_scale_sweep.rs` then gated the two corpora where the ceiling
  actually fires. On HOLDOUT, mean CER delta against uncapped, worst single-clip
  regression beside it:

  | cap | cord | worst clip | doc | worst clip |
  |---|---:|---:|---:|---:|
  | 736 | +0.0033 | +0.1928 | +0.0395 | +0.2920 |
  | 960 | -0.0157 | +0.0377 | -0.0011 | +0.2628 |
  | **1280** | **-0.0065** | **+0.0189** | **-0.0195** | **+0.0869** |

  736 loses on both holdouts and takes a clip with it — `cord-039` is 2304x4096
  and lands at 736x1308, close to the 544x960 that once merged a whole receipt
  into one 1903x1781 blob. 1280 is the only arm that improves both corpora with
  no badly-regressing clip. `carmenta-capture` is unaffected by construction:
  all 108 clips are 620x200, so the floor fires and the ceiling cannot.

  The ceiling is safe because it changes only the resolution at which **box
  geometry** is computed — boxes are mapped back to original image pixels and
  crops are cut from the full-resolution plane, so the recognizer always reads
  full-resolution pixels.

- **Added** — `examples/det_scale_sweep.rs`, the gate above, re-runnable.
- **Added** — `image::resize_bilinear_u8`, which samples one channel of an
  interleaved `u8` image directly into the resize.
- **Changed** — `mobiledet_input`, `craft_input_color`, `doclayout` and `table`
  each built a **source-resolution `f32` plane per channel** that held nothing
  but a `u8 -> f32` cast — `w*h*4` bytes, three times per page. They now sample
  the source. Output is bit-identical and `resize_u8_matches_plane_path` asserts
  it across all three pixel formats and up/down/identity scaling.

### `ffai-carmenta-wasm` 0.2.0 — the 4 GB wall

- **Fixed** — a single `Reader` could not read six A4 pages. Peak linear memory
  grew ~750 MB per page with no plateau and aborted with
  `RuntimeError: unreachable` — Rust's allocation-failure abort against wasm32's
  4 GB ceiling. The cause was **not** in Carmenta: `rusty_alloc` 1.1.4 never
  reused a freed segment on wasm32, because `prim::free` is correctly a no-op on
  a memory that cannot shrink and the arena that would have caught the segment
  was disabled on that target. Fixed upstream in 1.1.5 (the leak) and 1.1.6 (the
  segment-size rounding above it); this crate pins `=1.1.6`.

  Peak linear memory, byte-identical OCR output in every cell:

  | | dlmalloc | 1.1.4 | 1.1.5 | 1.1.6 |
  |---|---:|---:|---:|---:|
  | model load | 54 MiB | 128 MB | 128 MiB | **98 MiB** |
  | A4 page, steady state | 507 MiB | **trap on read 6** | 1280 MiB | **733 MiB** |
  | 12 MP photo, one read | 593 MiB | — | 1280 MiB | **701 MiB** |

- **Changed** *(breaking, Rust only)* — `read`, `readLine` and `text` take the
  RGBA buffer **by value** (`Vec<u8>`) rather than by slice. wasm-bindgen has
  already copied the caller's bytes into linear memory to form the argument, so
  borrowing them forced `ImageBuffer` to copy a second time — two 8.7 MB buffers
  for one A4 page. **The JS signature is unchanged**; it is still a
  `Uint8Array`, and no browser caller needs to change.
- **Added** — `setDetMaxShort(px)`, reaching the ceiling above from a target
  with no environment. On wasm this is a memory bound, not just a speed knob:
  a 12 MP photo goes from 2487 MiB of peak to 593 MiB.
- **Added** — `linearMemoryBytes()`, the only honest memory instrument on a
  target whose memory never shrinks.

### Security

Two resource-exhaustion issues on the untrusted-input path, both now bounded.

- **Unbounded allocation as a function of image dimensions.** Detector memory
  scaled with the full input area, so a caller-supplied 12 MP image drove ~2.5 GB
  of wasm linear memory in a single call — within reach of the 4 GB ceiling on
  one image, and reachable on smaller images by repetition. The short-side
  ceiling bounds detector input area regardless of what the caller submits.
- **Unbounded growth across calls** (`ffai-carmenta-wasm` only) — the allocator
  defect above, fixed by the `=1.1.6` pin. Peak is now flat across repeated
  reads instead of monotonically rising to an abort.

No change to `unsafe` code or to any crate's `UNSAFE.md` invariants; no change to
what is trusted or to what personal data is processed; no advisory resolved or
waived; no hardening gate moved.


### `ffai-carmenta` — one dependency dropped

- **Removed** — `candle-transformers`. It was declared and **never imported**:
  Carmenta's CRAFT detector and its CRNN/PARSeq recognizers are built directly
  on `candle-core` and `candle-nn`, because none of them is an architecture
  that library carries. Found while answering "which crates use candle
  transformers"; verified by building and running the full suite without it
  (44 tests, unchanged). No API change — it shrinks the dependency tree of a
  published crate and nothing else.

### `ffai-media` — `fps` sampling was broken, silently, for every clip

- **Fixed** — `stream_frames(path, fps)` returned **exactly one frame** from
  every video, at every rate. It computed its source frame rate as
  `time_base.den / time_base.num`, but a time base is a clock TICK rate, not a
  frame rate; MP4 commonly uses 1/12800, so asking for 1 fps computed a stride
  of 12800. Decimation is now by **timestamp deadline**, which needs no frame
  rate at all and stays correct on variable-frame-rate sources.

  It failed quietly in the worst way — one frame is a perfectly good frame, so
  callers got a plausible result rather than an error. **And the test guarding
  it was green throughout**: it asserted only `some.len() < all.len()`, which
  `1 < 48` satisfies. The test now asserts the RATE — spacing at least one
  interval, count within ±50% of `duration x fps`, and oversampling keeping
  every frame.

### `ffai-argus` — video: windowed captions and a timed track

- **Added** — `describe_video` groups sampled frames into windows and captions
  each window as one multi-image prompt, returning a `TimedSegment` track.
  Windows tile the timeline with no gaps or overlap; the remainder window is
  never dropped.
- **Added** — `preprocess_rgb8_opts(.., split)`. Video turns tile splitting
  **off**: a still is 17 tiles / 1088 image tokens and the tower holds 8192, so
  a split window caps at seven frames. Unsplit is one tile / 64 tokens and the
  same budget holds a hundred. The unsplit tile is *bit-identical* to the split
  path's global thumbnail, so video inherits that path's oracle gate.
- **Added** — `tile_geometry`, a pixel-free tile-count predictor, so an
  oversized window is refused in milliseconds instead of after running the
  vision tower over two hundred frames. Gated against real preprocessing at
  five sizes and both split modes.

### `ffai-cli` — `ffai caption` takes video

- **Added** — `ffai caption -i clip.mp4 --fps --window --max-frames --output`,
  writing `.srt` / `.vtt` / `.json` through `Transcript`'s own renderers, or
  timestamped text to stdout.
- The CLI drives the engine **one window at a time**, so peak memory is a
  function of `--window` rather than of the length of the clip.

### `ffai-argus` — the engine, and the resampler that decided it

Argus stops being a stub. `SmolVlm` — `SigLIP` tower, pixel-shuffle connector,
Llama decoder on candle — reports `EngineStatus::Stable`, which in this tree
means *oracle-gated against a reference implementation*.

- **Added** — `SmolVlm`, a real `VlmEngine`. From a raw `ImageBuffer` it
  reproduces the reference implementation's caption **byte-identically**
  (`docs/plans/argus-launch-plan.md` §16). `ffai caption` and `ffai bench vlm`
  both reach it; weights resolve through the `ffai-models` manifest seam
  (`models/smolvlm-256m-instruct.toml`), never a hardcoded cache path.
- **Added** — `preprocess`: the content path (Lanczos-3, `AnyRes` tiling,
  rescale/normalize), implementing **PIL's fixed-point resampler** rather than
  approximating it — `i32` coefficients at `1<<22`, integer accumulation, and
  a `u8` intermediate between the two passes. Gated bit-identical to PIL in
  both directions by `tests/resize_oracle.rs`.

  This is the entry worth reading twice. The previous float implementation
  matched PIL to **one quantisation level** — `7.843e-3`, exactly `1/255`,
  visually indistinguishable, ~50 dB SNR — and produced **8 of 32** correct
  tokens. The vision tower's own `2.06e-4` flips nothing across the same 32
  argmaxes; preprocessing's `7.8e-3` flips one at step 5. **No tensor
  tolerance could have told us which side of that line we were on** — only
  token equality could.
- **Added** — `decode`: greedy and sampled generation with a `KV` cache.
  `TextDecoder` keeps a pristine cache and clones it back at the top of
  `generate`, so a second caption cannot inherit the first's keys and values.
  Without that, the failure presents as *"the captions are fine until you
  batch a directory"*, which nobody attributes to a cache.
- **Added** — `vision`, `prompt`: the tower/connector and the Idefics3 prompt
  assembly, each gated in isolation (104.8 / 113.2 dB; 1142/1142 token ids).
- **Changed** — `tokenizers` and `ffai-models` are now real dependencies, not
  dev-dependencies: turning the assembled prompt into ids and the generated
  ids back into text is the engine's job.

### `ffai-cli` — caption gains the Gate 2 decoding surface

- **Added** — `ffai caption --max-new-tokens --temperature --top-p --top-k
  --seed --repetition-penalty --stop`. The `Decoding` surface existed since
  `ffai-core` 0.7.0 and nothing could reach it.
- **Added** — sampling **requires** `--seed`. Passing `--temperature` alone is
  refused rather than quietly returning a caption that cannot be reproduced.
- **Deliberately NOT added** — `--system-prompt`. `VlmOptions` carries the
  field because the trait needs it, but the contract is that an engine with no
  system slot ignores it, and SmolVLM is such an engine. A flag the only
  available engine silently drops is worse than no flag.
- **Fixed** — the candle thread cap is no longer process-wide. It was measured
  on Diana (a 36 ms detector: 21% less peak RSS at identical speed) and then
  applied to everything. On Argus — seventeen 512x512 tiles and a 1142-token
  prefill — it cost **23% of throughput** (min-of-4, ABBA-interleaved: 35639 ms
  at 4 threads vs 27497 ms at 24) to save 22 MiB of a ~1400 MiB working set.
  It now applies only to `detect`/`depth`, the commands it was measured on.

### `ffai-bench` — the VLM comparison key names a checkpoint

- **Fixed** — the engine arm's config string was built from the *engine* name
  (`smolvlm`), while every other task names a *checkpoint* (`tiny.en/greedy`,
  `yolo26n/e2e-640sq`). `smolvlm` is equally true of the 500M weights, whose
  numbers are not comparable to the 256M row. Now `SmolVLM-256M/greedy-64`, via
  a `vlm_comparison_key` that mirrors the ASR precedent — including the part
  that matters: an unrecognised engine SKIPS its quality gate rather than being
  compared against a reference it may not match.

### Security

No security-relevant changes: no `unsafe` added or altered in `ffai-argus`
(the crate keeps `unsafe_code = "warn"`, and its two `unsafe` blocks are
`VarBuilder::from_mmaped_safetensors` calls carrying `SAFETY` notes), no
change to what is trusted, and no change to what personal data is processed.
The scorer-argv trust boundary settled in step 0c is unchanged: a corpus
selects a scorer **by name**, and `corpora/references.toml` defines the argv.

### `ffai-core` 0.6.5 → 0.7.0 — BREAKING, deliberately

The Argus VLM trait surface, settled in one pass before an engine exists
(`docs/plans/argus-launch-plan.md` §2 Gate 2, §10). Every field is cheap now
and a semver break once Argus has users; the blast radius today was **two
struct literals**, both in-repo.

- **Added** — `Decoding` enum (`Greedy` | `Sampled { temperature, top_p, top_k,
  seed }`). `seed` is **not** an `Option`: there is no way to express
  stochastic decoding without a seed, so the determinism requirement is
  enforced by the type rather than by a doc comment. `Greedy` is `#[default]`.
- **Added** — `VlmPart` / `VlmPrompt`: ordered, interleaved multimodal prompts
  (`text <img> text <img>`), borrowed rather than owned.
- **Added** — `VlmOptions::{system_prompt, decoding, stop, repetition_penalty}`.
- **Added** — `VlmOptions::frames_per_window`, the video knob. It decides what a
  caption can be ABOUT: a split frame is 1088 image tokens and the tower holds
  8192, so it trades detail within a frame against context across frames. A
  caller's option rather than a constant, because the right value depends on
  the question.
- **Changed** — `VlmEngine::describe` is now the required method and
  `describe_image` is a provided convenience that delegates to it. An engine
  implementing only the single-image case would otherwise compile against a
  multi-image prompt and silently answer with one of them.
- **Not added, as decisions rather than oversights** — streaming, grounding,
  structured/JSON output, logprobs, conversation history. Reasons are recorded
  on `VlmOptions` itself.

**Consequence for `ffai-mercury` 1.0.0, stated rather than discovered.** That
release's entry warned: *"`ffai-core` remains 0.6.x: a breaking change there
would reach this crate's surface without a major-version signal from cargo."*
This is that change. In practice Mercury's surface is untouched — it is
expressed in `AsrEngine`/`TtsEngine`/`AudioBuffer`/`Transcript`, none of which
moved, and nothing in Mercury references the VLM types. The break is a cargo
version signal, not an API one, and a Mercury release against `ffai-core` 0.7
should say exactly that.

### `ffai-bench` — the VLM vertical (Argus steps 0a–0c)

- **Added** — `ffai bench vlm`, `Task::Vlm` un-guarded in the CLI, `run_vlm`,
  and `VlmScore` in the ledger (raw + normalised + metric + scorer + item
  count, because VLM scores are not commensurable across benchmarks).
- **Added** — `ScorerSpec`: a VLM corpus declares the benchmark's **own**
  evaluator. This crate contains no VLM answer-comparison code and must not
  grow any — answer extraction is part of a VLM metric, and a home-grown
  extractor is the 2.8×-biased scorer that cost the Carmenta campaign a year.
- **Added** — `ClipEntry::prompt`, and both the prompt and the `[scorer]` block
  are now inside `Manifest::manifest_hash`. Hashed **only when present**, so
  every pre-VLM corpus keeps its existing fingerprint — verified by
  `a_shipped_corpus_still_hashes_to_its_ledger_value`, which recomputes
  `librispeech-test-clean-v2` and asserts it still equals the value already in
  `bench/ledger.jsonl`.
- **Fixed** — two runs of the same clips under different scorer scales produced
  normalised scores **20× apart under an identical corpus hash**, because the
  scorer sat outside the fingerprint. A ledger line has to be sufficient on its
  own to reproduce a run.

### Security

Two changes are security-relevant, and one of them widens what an untrusted
file can do.

- **Changed — a corpus manifest can now name a command that `ffai-bench`
  executes.** `[scorer]` in `corpora/*.toml` carries an argv that `run_vlm`
  spawns. Previously only `corpora/references.toml` could cause execution;
  a `corpora/*.toml` was pure data. **Anyone who can write a corpus file can
  now run code as the bench user.**

  Mitigating, and stated so the reader can judge rather than take it on trust:
  corpora are repo-controlled and reviewed, this is the same trust level
  `references.toml` has always had, and the clip *bytes* remain hash-pinned so
  the data half is unchanged. But the scorer command is **not** hash-pinned,
  and a corpus TOML should from now on be reviewed as executable input, not as
  data. It is listed here rather than buried because the trust boundary moved.
- **Unchanged** — no new `unsafe` in any crate; no dependency advisories
  resolved or waived; no change to what personal data is processed. The new
  `ffai-argus` dev-dependency in `ffai-bench` is test-only (one test needs a
  registered stub VLM engine) and does not enter the library graph.

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
