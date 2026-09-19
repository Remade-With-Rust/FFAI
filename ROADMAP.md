# FFai Roadmap

> **Reconciled against the README and `bench/ledger.jsonl` on 2026-09-19.**
> Every ✅ below traces to a claim the README evidences; every ⬜ is open
> work. If this line is old, distrust the boxes — the README is the source of
> truth for what is measured.

Sequencing logic: ASR first because it exercises the hardest infrastructure
(streaming audio, timestamps, large models); plugins last because a plugin
ABI frozen before the traits have survived four verticals is a liability.

## Phase 0 — Skeleton ✅ (this repo, July 2026)

Workspace, `ffai-core` types + engine registry, candle wired as the tensor
spine, WAV I/O, model manifests + cache layout, CLI, and honest stubs for
every planned engine.

Also live: **`ffai-bench`**, the analyzer (`ffai bench asr --corpus …`) —
four-gate verdicts, best-of-N timing, WER/CER metrics, hash-pinned corpora,
external reference adapters, and the append-only claims ledger
(`bench/ledger.jsonl`). Ported from the Prometheus measurement spine (the
private remade_ffmpeg_rs refinery); the symbolic-discovery half of Prometheus
stays private and codec-side. Per-phase bench work: Phase 1 adds Whisper-
normalizer parity + memory footprint instrumentation (the 4th gate) + the
`bench tts/ocr/vlm` verticals as each engine lands, with corpus manifests for
LibriSpeech (ASR), public OCR ground-truth sets (Carmenta), and caption
corpora (Argus).

> Mercury (Phases 1–2) now has a dedicated mission plan with per-milestone
> exit gates: [docs/finished/mercury-mission-plan.md](docs/finished/mercury-mission-plan.md).

## Phase 1 — Mercury ASR goes live

- ✅ `whisper-candle` engine: our own mel front-end (STFT + Slaney
  filterbank), tokenizer grammar, and greedy decode loop, composed over
  candle-transformers' Whisper blocks. Transcribes today.
- ✅ `ffai-models` fetcher: Hugging Face hub download into the shared cache,
  checksum-verified, license surfaced.
- ✅ Stage oracles: mel matches openai-whisper to < 1e-3; tokenizer
  round-trips.
- ✅ Beam search (`FFAI_BEAM_SIZE=5`; greedy stays the default), temperature
  fallback, long-audio seek, and the tiny/base/small `.en` tiers — `small.en`
  more than halves the error rate (3.05 % WER against tiny.en's 6.39 %).
- ✅ `ffai-media`: **remade_ffmpeg_rs** (`rff-format`/`rff-codec`) is the
  backend — any container in = 16 kHz mono out.
- ✅ whisper.cpp baseline — built, and **all four gates PASS on both
  holdouts** (correctness, quality, speed, footprint; ledger
  `bench-asr-1785387940`, `-1785388172`). Speed came from adaptive encoder
  context, not from a kernel — [docs/whys/adaptive-context.md](docs/whys/adaptive-context.md).
- ⬜ **Quantization.** int8 across the decoder cost 8.39 % WER and was pruned.
  A finer scheme (per-head scales, or int8 keys with f16 values) needs corpus
  WER per iteration, because the argmax-flip instrument is blind to this class
  — [docs/whys/OPEN.md](docs/whys/OPEN.md) § 4.
- ⬜ **A stable streaming API.** The demo's **Listen** tab already runs Mercury
  on a live microphone, so the path exists; it is not yet a published surface
  on `AsrEngine`.

## Phase 1.5 — The WhisperX layer ✅ LIVE

The whole layer is in pure Rust and composes over ANY registered ASR engine
rather than being welded to one.

- ✅ VAD → better segmentation on long audio.
- ✅ Forced alignment (CTC) → word-level timestamps, `--word-timestamps`.
- ✅ Speaker diarization → `--diarize --max-speakers N`.
- ⬜ Non-English alignment heads; the shipped aligner is en-only.

## Phase 2 — Mercury TTS ✅ LIVE

**Shipped as `piper-candle`, not as the `any-tts`/Kokoro plan below** — the
full VITS stack on candle, running piper's own ungated voice files through our
own pure-Rust ONNX reader, with a clean-room CMUdict G2P. espeak-ng (GPL)
participates only as an out-of-process test oracle; nothing GPL ships.

- ✅ Oracle-exact against piper's onnxruntime (text encoder 4e-6, durations
  integer-exact, waveform 3e-5), quality parity through a frozen judge
  (5.49 % vs 5.27 % WER), **1.58× faster wall-clock at 5 % less CPU**, and
  byte-identical output per seed.
- ✅ Weight-license surfacing in `ffai models`.
- ⬜ Languages beyond en-US — the honest cost of the clean-room G2P.
- ⬜ A second engine (`voirs`, `any-tts`, Kokoro-82M tiers) — still wanted, so
  that the trait has more than one implementation behind it.

## Phase 3 — Carmenta OCR

> Carmenta now has a dedicated mission plan with per-milestone exit gates:
> [docs/Carmenta-mission-plan.md](docs/Carmenta-mission-plan.md). Four
> functions over one det+rec core — LIVE, DOCUMENT, LONG, FORMULA — with
> LIVE first, each benchmarked against its non-Rust world standard
> (Tesseract as the C++ bar).

- ✅ Detection → recognition pipeline on candle, shipped as user-selectable
  engines under the names that survived M-C1: `mobiledet-svtr` (the document
  default), `craft-crnn` and `craft-parseq` (scene text), plus the
  `mobiledet-crnn` / `mobiledet-parseq` / `composed-*` cross-pairings.
- ✅ LIVE streaming mode (`--live --watch N`) — change-gated, **zero churn
  across 156 unchanged frames** where stateless Tesseract churns 24 times.
- ✅ Oracle gate on public ground truth: **OmniDocBench v1.6 scored by their
  own evaluator**, all 1 651 pages — Text^Edit 0.1157, ReadOrder^Edit 0.2039
  ([docs/plans/restarting-carmenta-doc.md](docs/finished/restarting-carmenta-doc.md)).
- ⬜ Photo accuracy still trails PaddleOCR; causes are diagnosed, not fixed.
- ⬜ Evaluate pure-Rust `ocrs` as a zero-setup baseline engine.
- ⬜ rff image decoders (PNG/JPEG/WebP).

## Phase 4 — Argus VLM

**Image captioning / VQA is LIVE**: `SmolVLM-256M-Instruct` on candle —
`SigLIP` tower, pixel-shuffle connector, Llama decoder — reproducing the
reference implementation's caption byte-identically from a raw image
(`docs/plans/argus-launch-plan.md` §16). `ffai caption -i image.png`.

**Video understanding is LIVE**: `ffai caption -i clip.mp4 --fps 2 --window 8`
streams frames, captions them in windows, and writes a timed track
(`--output x.srt` / `.vtt` / `.json`). Splitting is off for video — one tile
per frame at 64 tokens, so a window fits the tower's 8192 positions — and the
unsplit tile is bit-identical to the still path's already-gated thumbnail.
Memory is one window deep regardless of clip length.

Remaining:

- **A video quality claim.** None is made today, deliberately:
  `SmolVLM-256M-Instruct` is an image model with no temporal training and no
  published Video-MME/MVBench row, so a number here would be one we invented.
  It needs a video-capable checkpoint first.
- **Speed.** Now **1.20x** off the PyTorch reference end to end (10 918 vs
  9 106 ms, same image, idle box), down from 2.4x across five optimization
  rounds; quality is an exact tie and footprint is 0.71x. The deficit is
  concentrated in the vision tower, which is 75 % of a caption and 77 % matmul
  — the elementwise phase is spent. Blocked attention has been refuted five
  times, the last with candle's own GEMM.
- **Re-run the `ffai bench vlm` gate.** Its recorded row still reads
  `speed FAIL` at 2.4x and is stale. Re-running is blocked by a harness defect,
  not an engine one: the engine arm segfaults on the second `describe_image` in
  one process, and the crash reproduces with the optimizations reverted.
- `mistralrs` behind the reserved `mistralrs-backend` feature, for the serving
  concerns it owns: quantized weights and grammar-constrained JSON decoding.
  Blocked on a crates.io release that can load `SmolVLM` — the working version
  is a git revision, and `cargo publish` refuses a git dependency.
- Larger checkpoints (`SmolVLM-500M`, Qwen-VL) as user-selectable engines, each
  registering under its own name so the bench's checkpoint-level comparison key
  keeps working.

## Phase 5 — The graph and plugins

- Filtergraph execution: `ffai -i clip.mp4 -g "asr=diarize=1 [subs]; ..."` —
  typed segments/buffers flowing through a node DAG (the types already exist
  in `ffai-core`).
- WASM plugin ABI: a plugin is an engine registered at runtime; the registry
  was designed for this from day one.

## Beyond — future components (the wider pantheon)

| Codename | Namesake | Capability |
|---|---|---|
| **Echo** | the nymph who returns voices | voice cloning / conversion (under Mercury) |
| **Janus** | god of gateways and two faces | translation — text and speech-to-speech (Mercury × Janus) |
| **Moneta** | Juno Moneta, "the reminder" | embeddings, semantic search, vector memory over media |
| **Minerva** | goddess of wisdom | text intelligence: summarization, chaptering, structured extraction |
| **Apollo** | god of music | music & audio generation |
| **Vulcan** | god of the forge | image / video generation |

## Watchlist — removing C/C++ from the low level

Policy: pure Rust by default; every remaining C/C++ touchpoint has a named
pure-Rust replacement we adopt when it matures.

| Dependency risk | Pure-Rust replacement tracked |
|---|---|
| CUDA toolchain via candle's `cuda` feature | [cool-japan/oxicuda](https://github.com/cool-japan) (as used by TrustformeRS) |
| Any transformer-stack gaps | [cool-japan/trustformers](https://github.com/cool-japan/trustformers) (watch, don't depend yet — v0.2 alpha) |
| Whisper inference alternatives | [cool-japan/oxiwhisper](https://github.com/cool-japan/oxiwhisper) (early; GGUF, SIMD, zero C/C++) |
| TTS stack | [cool-japan/voirs](https://github.com/cool-japan/voirs), [any-tts](https://github.com/Rheosoph/any-tts) |
| All containers/codecs | [remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs) — ours, default backend |

Review cadence: check [cool-japan](https://github.com/cool-japan?tab=repositories)
and the deps above at each phase boundary.
