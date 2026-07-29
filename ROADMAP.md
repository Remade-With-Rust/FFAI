# FFai Roadmap

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
- ⬜ Beam search + temperature fallback, long-audio seek, all model sizes,
  quantization, streaming API (M2).
- ⬜ `ffai-media`: wire **remade_ffmpeg_rs** (`rff-format`/`rff-codec`) so any
  container in = 16 kHz mono out.
- ⬜ whisper.cpp baseline — the native no-Python comparison (M2 blocker).

## Phase 1.5 — The WhisperX layer

- VAD (silero-class) → better segmentation on long audio.
- Forced alignment (wav2vec2-class) → word-level timestamps.
- Speaker diarization → `--diarize`.
- All composable over ANY registered ASR engine, not welded to one.

## Phase 2 — Mercury TTS

- `any-tts` engine (Kokoro-82M first, then Qwen3-TTS/VibeVoice tiers).
- `voirs` as the second engine (VITS/FastSpeech2 + HiFi-GAN lineage).
- Weight-license surfacing in `ffai models` (some voices are CC BY-NC).

## Phase 3 — Carmenta OCR

- Detection → recognition pipeline on candle.
- `unlimited-ocr` (document/layout tier) and `easy-ocr` (CRAFT+CRNN
  scene-text tier) as user-selectable engines.
- Evaluate pure-Rust `ocrs` as a zero-setup baseline engine.
- rff image decoders (PNG/JPEG/WebP) land here.
- Oracle gate: CER/WER on public ground-truth sets vs EasyOCR/Tesseract.

## Phase 4 — Argus VLM

- `mistralrs` engine: Qwen-VL / LLaVA-class captioning + VQA, quantized.
- Video understanding: rff keyframe sampling → windowed captions → timed
  track (`ffai caption -i video.mp4` → chaptered description).

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
