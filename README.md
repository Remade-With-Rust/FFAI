# FFAI

**The AI media toolkit, remade with Rust.** Speech recognition, text-to-speech,
OCR, object detection and tracking, and vision-language captioning in one
toolkit. It is built the way ffmpeg was built: libraries first, one binary on
top, every engine swappable by name. No Python runtime, no ONNX runtime, no
gated weights, and the same engines compile to WebAssembly and run in a browser
tab.

Part of [Remade With Rust](https://github.com/Remade-With-Rust).

## In The Wild with 580,129 Active Installs

> [RAG Converter](https://ragconverter.com) runs FFAI's engines as WebAssembly in
> the browser to make personal and business files AI-readable: emails, PDFs,
> scans, images, audio and video, converted to RAG data in real time with
> nothing uploaded and nothing installed. An API serves the same pipeline to
> businesses.

> [DLDeploy.com](https://dldeploy.com) delivers browser-based AI built from the
> Remade With Rust portfolio, with FFAI as the AI layer.

> [MATA Network](https://mata.network) builds Disco Party, the sovereign
> distributed cloud, on the Remade With Rust stack. FFAI is that stack's AI
> layer, so the apps deployed onto Disco transcribe, speak, read and see
> without sending data to a third-party AI service.

> Build on it with our
> [AI skills for Rust](https://drive.google.com/drive/folders/1VjvP1zSJWH1DmkTqL1qEPt3OM8Ub8jPX?usp=drive_link).

```text
ffai asr -i talk.wav -o talk.srt                      # speech to subtitles
ffai asr -i talk.wav -o talk.json --word-timestamps   # per-word times
ffai asr -i meeting.wav --diarize --max-speakers 3    # who spoke when
ffai tts "Hello from FFai." -o hello.wav               # piper voices, pure Rust
ffai ocr -i page.png                                  # documents
ffai ocr -i filing.pdf --auto-orient                  # PDFs, scans, forms
ffai ocr --live --watch 5 -i captures/ -o screen.srt  # live screen reading
ffai detect -i street.jpg --engine yolo26n            # YOLO26 detection
ffai detect --track -i frames/ -o tracks.txt          # multi-object tracking
ffai caption -i frame.png --prompt "what is happening here?"
ffai engines        # every engine and its status, like `ffmpeg -codecs`
ffai models         # model manifests, licenses, cache status
```

## Speed

Each engine is measured against the implementation people already deploy, with
the same weights, the same inputs and the same decoding settings, on CPU.

| Engine | Task | Against | Speed | Memory | Quality |
|---|---|---|---|---|---|
| **Mercury** | ASR | whisper.cpp | **1.07× faster on clean speech, 1.70× on noisy** | **0.84–0.92×** | WER 7.27 % vs 7.58 % |
| **Mercury** | TTS | piper | **1.58× faster** at 5 % less CPU, **10× faster load** | parity | WER 5.49 % vs 5.27 % |
| **Carmenta** | OCR recognition | PaddleOCR mobile | **2.6× lower latency** | — | CER **1.5 %** vs 3.0 % |
| **Carmenta** | Document OCR | Unlimited-OCR (3B, GPU) | **~9× throughput, on CPU** | MB of weights vs 6.4 GB | edit distance 0.108 vs 0.041 |
| **Carmenta** | Live screen OCR | Tesseract per frame | **p95 257 ms vs 343 ms** | flat over 30 min | **0** churn vs 24 |
| **Diana** | Detection | Ultralytics | **3.71× less CPU per frame**, 1.5–2.4× throughput | **1.6–5.6× leaner** | mAP50 within 0.08 pp |
| **Diana** | Tracking | Ultralytics ByteTrack | — | no ReID model | **218 ID switches vs 800** |
| **Argus** | Captioning | `transformers` | preprocessing **2.6× faster**; whole caption 1.20× slower | — | caption **byte-identical** |

**The latest releases do the same work with less of it.** These are
deterministic counts (callgrind instructions on one core, or bytes written),
not timings, and the output stays byte-identical. Wall time on the new
releases has not been re-measured, so the speed column above comes from the
previous release of each engine.

| Release | Workload | Work counted |
|---|---|---:|
| `ffai-diana` 0.10 | detection, yolo26n at 640 | **−34.6 %** (1,075 M → 703 M) |
| `ffai-diana` 0.10 | tracker frame | **−24.3 %** |
| `ffai-argus` 0.10.1 | prefill / decode token / whole caption | **−24.2 % / −15.3 % / −8.7 %** |
| `ffai-argus` 0.10.1 | preprocessing / prompt assembly | **−85.1 % / −58.7 %** |
| `ffai-mercury` 1.2 | ASR zero-fill memset | **−99.4 %** (573 MB → 3.3 MB) |
| `ffai-mercury` 1.2 | TTS zero-fill memset | **−93.1 %** (503 MB → 34 MB) |

The decode path is ours too: `rusty_jpeg` decodes 1080p in 7.05 ms, within
14 % of libjpeg-turbo; `rusty_png` is byte-identical to upstream `png`; and
`rusty_h264` decodes x264's default profile at 22.7 ms per 1080p frame.

Every figure traces to a line in [`bench/ledger.jsonl`](bench/ledger.jsonl),
produced by `ffai bench` (below).

## Components

| Component | Crate | Task | Namesake |
|---|---|---|---|
| **Mercury** | [`ffai-mercury`](crates/ffai-mercury) | ASR + TTS | Roman god of language and messages |
| **Carmenta** | [`ffai-carmenta`](crates/ffai-carmenta) | OCR | Roman goddess who adapted the Greek alphabet into Latin letters |
| **Diana** | [`ffai-diana`](crates/ffai-diana) | Detection + tracking | Roman goddess of the hunt |
| **Argus** | [`ffai-argus`](crates/ffai-argus) | VLM captioning | Argus Panoptes, the all-seeing watchman |

Browser builds: [`ffai-wasm`](https://crates.io/crates/ffai-wasm) (Diana),
`ffai-carmenta-wasm`, `ffai-mercury-wasm` and `ffai-argus-wasm`.

Infrastructure: `ffai-core` (engine traits, registry and shared fast math, on
candle), `ffai-media` (ingest and egress via
[remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs)),
`ffai-models` (weight manifests and cache), `ffai-bench` (the analyzer),
`ffai-cli` (the `ffai` binary), and `ffai-demo`, a three-tab demo: **Listen**
runs Mercury beside whisper.cpp on one microphone, **Speak** shows phonemes and
a same-seed determinism check, **Read** runs OCR on an image you drop in
(`cargo run --release -p ffai-demo`).

### Mercury: speech recognition and synthesis

**ASR** runs OpenAI Whisper, tiny to medium, on candle with our own mel
front-end, tokenizer grammar, decode loop and AVX2 kernels. It passes all four
gates (correctness, quality, speed, footprint) against whisper.cpp on both
LibriSpeech holdouts, 134 clips each, tiny.en, greedy:

| Corpus | Implementation | WER % | ×realtime | steady MiB |
|---|---|---:|---:|---:|
| test-clean | **Mercury** | **7.27** | **27.8** | **179** |
| test-clean | whisper.cpp | 7.58 | 25.9 | 195 |
| test-other | **Mercury** | 16.89 | **33.3** | **163** |
| test-other | whisper.cpp | 16.82 | 19.6 | 194 |

Most of the speed comes from **adaptive encoder context**: Whisper pads every
window to 30 s, so each window is encoded at a context sized to the audio it
actually holds, with guards that fall back to the full context. Stage by stage
it wins every stage against whisper.cpp with flash attention on: encode about
2.0×, decode 1.1–1.2×, mel 1.4×.

- **Accuracy tiers:** `small.en` reaches **3.05 % WER**, ahead of whisper.cpp
  `small.en` (3.38 %) at 4.2× against 3.7× realtime. Beam search is available
  with `FFAI_BEAM_SIZE=5`.
- **The WhisperX layer, without Python or gated weights:** speech segmentation
  on by default (2.2–4.2× faster on audio with trailing silence, identical
  transcript); `--word-timestamps` by CTC forced alignment (100 % containment);
  `--diarize` with ECAPA-TDNN (**DER 4.21 %**); and speaker labels that persist
  across calls for live streams (streaming DER 5.68 %).

**TTS** runs the full VITS stack on candle, reading
[piper](https://github.com/OHF-Voice/piper1-gpl)'s own voice files straight
from ONNX, with a clean-room pure-Rust phonemizer. Nothing GPL ships.

- **Oracle-exact** against piper's runtime: waveform to 3e-5, durations
  integer-exact.
- **1.58× faster wall-clock at 5 % less CPU**, model load 0.64 s against
  6.69 s, memory at parity.
- **Deterministic:** the same text and seed give one exact WAV per machine
  (1.2.1 pins the GEMM cache detection that hybrid P-core/E-core CPUs used to
  split). Piper samples noise inside its graph and cannot offer this.
- **1.2** cut 45 pieces of redundant work across ASR and TTS with byte-identical
  output, including 99.4 % of ASR's and 93.1 % of TTS's buffer zero-filling.
- English (en-US) voices.

### Carmenta: OCR for documents, forms, PDFs and live screens

Five engines from two detector lineages and three recognizers, each
oracle-matched against its reference. `mobiledet-svtr` is the document
default and `craft-crnn` the live default.

- **Recognition beats PaddleOCR's mobile recognizer** on 400 real-photo word
  crops: **1.5 % vs 3.0 % CER**, 93 % vs 88 % exact match, at **2.6× lower
  latency**.
- **Documents,** scored by
  [OmniDocBench](https://github.com/opendatalab/OmniDocBench)'s official
  evaluator on all 1,651 pages: text edit distance **0.116**, reading order
  **0.204**, inside the published range (0.033–0.157). On CPU it runs at about
  **9× the throughput** of the 3B record holder Unlimited-OCR, which scores
  higher (0.041 vs 0.108).
- **Live mode** emits text only when it changes: **zero churn** across 156
  unchanged frames where per-frame engines churned 24 times, 24/24 changes
  caught, p95 257 ms, memory flat over a 30-minute soak.
- **Forms and PDFs:** auto-orientation (144/144 quarter turns undone),
  checkboxes (100 % found and read on 960 boxes), label-to-value pairing
  (220/221), and PDF pages read as a viewer shows them: redactions are never
  read through.

### Diana: YOLO26 detection and tracking

The real YOLO26 graph on candle, all five tiers, loaded from the official
Ultralytics `.pt`. No ONNX, no Python at inference. Weights are AGPL-3.0 and
are never in this repository.

Graded on a 450-image COCO holdout against Ultralytics at the same
configuration:

| Tier | mAP50 Δ (pp) | Diana MiB | Ultralytics MiB | leaner |
|---|---:|---:|---:|---:|
| n | −0.01 | **71** | 403 | 5.6× |
| s | −0.08 | **110** | 447 | 4.1× |
| m | +0.06 | **178** | 544 | 3.1× |
| l | +0.06 | **197** | 535 | 2.7× |
| x | −0.07 | **346** | 678 | 2.0× |

At four tiers the detections are identical in count, class and order, within
0.3 px. On all seven MOT17 training sequences (5,316 frames) AP50 matches
Ultralytics to a mean 0.029 pp.

- **Speed:** **3.71× less CPU per frame** (84.6 ms against 314.2). On a
  dedicated machine with each engine pinned to its own cores, wall-clock is
  **1.30×** faster; under 16 competing threads it is **1.92×**; batch throughput
  is **1.5–2.4×** at every tier.
- **0.10:** **34.6 % fewer instructions per detection**, from removed copies,
  fused passes, Winograd 3×3 convolutions and a vectorized softmax. All five
  tiers still pass the full-graph oracle against the official model.
- **Tracking:** ByteTrack with no appearance model, so it adds no download and
  no license. **218 ID switches against Ultralytics' 800** on MOT17, IDF1
  35.93 % vs 36.92 %.
- **In the browser:** [`ffai-wasm`](https://crates.io/crates/ffai-wasm) runs
  the whole graph as a 1.82 MB module with an 18 ms model load. Every box and
  class matches native. `crates/ffai-wasm/demo.html` is a working page.
- **Lean to embed:** 138 transitive crates natively, 95 on wasm32, and no
  dependency on the other engines.
- Standalone page: [Remade-With-Rust/diana](https://github.com/Remade-With-Rust/diana).

### Argus: vision-language captioning

`SmolVLM-256M-Instruct` on candle, with our own SigLIP encoder and SmolLM2 text
tower. From a raw image file, the caption matches the Python reference **byte
for byte**, including preprocessing, which implements PIL's fixed-point
resampler exactly.

- **0.10.1** executes **24.2 % fewer instructions per prefill, 15.3 % fewer per
  decoded token, 8.7 % fewer per caption**, 85.1 % fewer in preprocessing and
  58.7 % fewer in prompt assembly. The caption, the decode tokens and the
  preprocessed bytes are unchanged. Wall time has not been re-measured since
  0.10.0.
- On 0.10.0, preprocessing is **2.6× faster** than `transformers`, and a whole
  caption is 1.20× slower, with the gap in the vision tower.
- Video captions stream to `.srt`, `.vtt` or `.json` one window at a time, so
  memory is set by `--window`, not by clip length.

```sh
ffai caption -i street.jpg --prompt "What is written in this image?"
ffai caption -i clip.mp4  --fps 2 --window 8 --output captions.srt
```

## The analyzer: `ffai bench`

One command compares any FFAI engine against world-standard implementations on
a pinned, hash-verified corpus:

```text
ffai bench asr --corpus corpora/librispeech-test-clean-v2.toml
ffai bench asr --corpus corpora/librispeech-test-other-v1.toml --baseline-only
```

It runs our engine and every reference in `corpora/references.toml` over the
same clips, scores both with the same metric code, and appends an audit record
to `bench/ledger.jsonl`: corpus fingerprint, reference versions, environment,
and a four-gate verdict (correctness, quality, speed, footprint) in which **a
skipped gate is never a pass**. Methodology:
[docs/benchmarking.md](docs/benchmarking.md).

## Architecture

**One trait per task, many engines per trait, selected by name.**
`AsrEngine`, `TtsEngine`, `OcrEngine` and `VlmEngine` are the `AVCodec` of
FFAI, and `--engine whisper-candle` is `-c:v libx264`. New models are new
engines, not rewrites.

1. **Pure Rust.** GPU and accelerator backends sit behind feature flags.
2. **Library-first.** The CLI contains no logic; every capability is a crate
   you can embed.
3. **Candle is the tensor spine.** One `Tensor` and `Device` across all
   engines.
4. **Weights are data, not code.** Never vendored; fetched into a cache from
   TOML manifests that surface each model's own license. Every model is
   fetchable without an account.
5. **Streaming-first.** Engines process chunks; whole-file is the degenerate
   case.
6. **Oracle-gated.** An engine is `stable` only when validated against its
   reference implementation on public corpora.
7. **Codecs come from home.** Media routes through
   [remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs);
   images decode through [`rusty_jpeg`](https://crates.io/crates/rusty_jpeg)
   and [`rusty_png`](https://crates.io/crates/rusty_png).
8. **Pure-Rust allocator.** The `ffai` binary and the wasm crates use
   [`rusty_alloc`](https://crates.io/crates/rusty_alloc) 2.4.0. It replaced
   mimalloc with byte-identical output and speed parity, and the engines'
   byte-identity oracles pass on every release.

**Supply chain.** Workspace builds patch
[`tokenizers`](https://github.com/Remade-With-Rust/tokenizers) onto
[`rusty_expressions`](https://github.com/Remade-With-Rust/rusty_expressions),
our pure-Rust Oniguruma, so no C regex engine is compiled. Two C components
remain: crates taken from crates.io still pull `onig_sys` through candle until
the patch lands upstream, and `aws-lc-sys` sits behind the optional
Hugging Face downloader (`ffai-models/fetch`). The wasm32 builds compile no C.

## Install

```sh
cargo install ffai-cli      # the `ffai` binary
cargo add ffai-mercury      # or embed any component as a library
```

Weights are fetched into a local cache on first use, or ahead of time with
`ffai models --fetch <name>`.

## Build from source

```sh
cargo build --release
cargo test --workspace
target/release/ffai engines
```

## Development history

The full campaign record behind every number, including refuted experiments
and corrections, stays in the repository:
[Mercury](docs/finished/mercury-mission-plan.md),
[Mercury TTS](docs/finished/mercury-tts-mission.md),
[Carmenta](docs/Carmenta-mission-plan.md),
[Diana](docs/finished/diana-mission-plan.md),
[Argus](docs/finished/argus-launch-plan.md), the instruction-count ledgers in
[docs/plans/](docs/plans/), and the root-cause write-ups in
[docs/whys/](docs/whys/). See [ROADMAP.md](ROADMAP.md) for what is next.

## License

MIT OR Apache-2.0 (code). Model weights carry their own licenses; check
`ffai models`.
