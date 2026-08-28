> **In the wild** — [RAG Converter](https://ragconverter.com) uses `ffai-carmenta` for OCR, reading scans and screenshots in the browser.
> It makes personal and work files AI-readable without them leaving the machine:
> the whole conversion runs as WebAssembly in the browser tab, with nothing
> uploaded and nothing to install.

# ffai-carmenta

[![crates.io](https://img.shields.io/crates/v/ffai-carmenta?logo=rust)](https://crates.io/crates/ffai-carmenta)
[![docs.rs](https://img.shields.io/docsrs/ffai-carmenta?logo=docsdotrs)](https://docs.rs/ffai-carmenta)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](https://github.com/Remade-With-Rust/FFAI)
[![Remade With Rust](https://img.shields.io/badge/Remade%20With-Rust-000?logo=rust&logoColor=fff)](https://github.com/Remade-With-Rust)
[![By Mata Network](https://img.shields.io/badge/by-Mata%20Network-5b2be0)](https://www.mata.network/)

> **OCR in pure Rust — documents, screens, and a LIVE streaming mode no
> mainstream OCR tool ships.** Text detection, mixed-case recognition, computed
> reading order, and a change-gated frame reader that never re-rolls on noise.
> No Python, no gated weights, nothing GPL — the OCR component of the
> [FFai](https://github.com/Remade-With-Rust/FFAI) toolkit.

**Most users want the whole toolkit — [FFai](https://github.com/Remade-With-Rust/FFAI)
and its `ffai` binary.** Depend on this crate directly when OCR is the only
component you need: it brings the detection and recognition engines, the
computed reading order and `LiveSession`, and nothing else.

Carmenta is named for the Roman goddess who adapted the Greek alphabet into
Latin letters. Standalone landing page:
[Remade-With-Rust/carmenta](https://github.com/Remade-With-Rust/carmenta).

Part of **[Remade With Rust](https://github.com/Remade-With-Rust)** by
**[Mata Network](https://www.mata.network/)**.

---

## Install

```sh
cargo add ffai-carmenta ffai-core ffai-media
```

```toml
[dependencies]
ffai-carmenta = "0.9"
ffai-core = "0.6"
ffai-media = "0.6"
```

## Recognize

```rust
use ffai_core::engine::{OcrEngine, OcrOptions};
use ffai_carmenta::engine::{CraftCrnn, RecStage};

let engine = CraftCrnn::new_mobiledet(RecStage::Svtr);   // mobiledet-svtr
let image  = ffai_media::load_image("page.png")?;
let out = engine.recognize(&image, &OcrOptions::default())?;
println!("{}", out.text());

for line in out.lines() {                     // blocks -> lines -> words,
    println!("{:?} {}", line.bbox, line.text);// reading order = Vec order
}
```

Weights download once, into a local cache, from hash-verified manifests. Every
model is reimplemented on candle and oracle-matched against its reference
runtime — CRAFT detection to <5e-3, CRNN recognition to the exact CTC argmax,
PP-OCRv5 mobile detection against paddle's own exported program.

Engines are selected by lineage name, like codecs in ffmpeg:

| Engine | Detector | Recognition | Where it wins (measured) |
|---|---|---|---|
| `mobiledet-svtr` | PP-OCRv5 mobile (4.7 MB) | SVTR CTC (CJK+Latin) | **documents** — the strongest on real pages, English and Chinese |
| `mobiledet-crnn` | PP-OCRv5 mobile | line CTC | lighter documents fallback |
| `craft-crnn` | CRAFT VGG16-BN | line CTC | clean UI/screen text, LIVE |
| `craft-parseq` | CRAFT | word AR (PARSeq-tiny) | photographs — **1.5 % CER on real-photo crops where PaddleOCR's own recognizer reads 3.0 %**, 2.6x faster |

## Documents

The document engine is `mobiledet-svtr`: DBNet detection (4.7 MB) plus
PP-OCRv5 SVTR recognition (16 MB, an 18 385-class CJK+Latin head). Reading
order is **computed, not learned**: several candidate orderings are built per
page from the boxes the detector already produced and the most self-coherent
is kept — no layout model required. Tables and isolated formulas can
additionally be routed to dedicated structure models (`FFAI_ROUTE=1`,
experimental: pure-Rust ports of PP-DocLayout-S, SLANet-plus and
PP-FormulaNet-S, each matched against its reference runtime).

Measured by [OmniDocBench](https://github.com/opendatalab/OmniDocBench)'s
**own evaluator** — the benchmark PaddleOCR-VL and Baidu's Unlimited-OCR state
their records on — over all 1 651 pages of v1.6, English and Chinese, zero
page failures:

| all 1 651 pages, official metric (lower is better) | Carmenta, CPU | published range |
|---|---:|---|
| Text edit distance | **0.116** | 0.033 (PaddleOCR-VL) – 0.157 (Marker) |
| Reading-order edit distance | **0.204** | 0.116 – 0.243 |

That row is this crate's defaults **plus `FFAI_ROUTE=1`**, the opt-in
table/formula stage — it needs model files the weight cache doesn't ship yet,
which is the one thing between a default install and the number above.
Everything else in it is on by default as of 0.9.0.

Ahead of published pipelines on reading order's worst rows, well behind the
VLM leaders on text — and the gap is *characterised*, not guessed. On the
236-page English holdout with no tables or formulas (content the pipeline
fully represents) Carmenta reads **0.041**, between PaddleOCR-VL's 0.033
(a 3B-class model on GPU) and PP-StructureV3's 0.079 (the full Paddle
pipeline). Where the rest of the loss lives is measured block by block from
the evaluator's own match records, and it reconstructs the headline exactly:

| remaining loss | share |
|---|---:|
| inline-math blocks (LaTeX we don't yet emit) | 36 % |
| text present but assembled/matched wrong | 17 % |
| text genuinely unread (mostly vertical & classical CJK) | 17 % |
| recognition substitutions | 20 % |
| detection misses · sequence errors · harness | 10 % |

So this is a **coverage and character** problem, not a layout one — the
reading-order and detection stages this crate spent its campaign on are down
to a combined 4 % of remaining error.

All of this on CPU at ~9–17 s/page — roughly 9× the reference VLM's measured
throughput on the same machine, from megabytes of weights against gigabytes.

## LIVE: point it at a screen

`LiveSession` wraps **any** `OcrEngine` and adds what a stream needs: a change
gate (a frame whose pixels didn't move reuses the last result at ~zero cost, so
an unchanged frame *cannot* flicker its text), ROI band tracking, and timed
SRT/VTT output.

```rust
use ffai_carmenta::live::{LiveConfig, LiveSession};

let mut live = LiveSession::new(engine, OcrOptions::default(), LiveConfig::default());
for (i, frame) in frames.iter().enumerate() {
    let out = live.push_frame(frame, i as f64 / fps)?;
}
let (track, stats) = live.finish(duration);   // TimedSegment<String> spans
```

```sh
ffai ocr --live --watch 5 -i captures/ -o screen.srt
```

Measured on a pinned 180-frame screencast: **24 OCR calls for exactly 24 text
changes, and zero churn across 156 unchanged frames** — stateless Tesseract
churns 24 times on the same frames. CER on change frames **1.21 %**, better than
the same engine in batch mode (1.74 %).

`LiveSession` takes `ImageBuffer` and is decoder-agnostic. Frame *sources* are
not: a directory of stills goes through `rusty_png`/`rusty_jpeg`, video ingest
is not built yet, and camera/MJPEG input is untested — so "point it at a screen"
today means frames on disk.

## Things measurement taught us, which you may want

**Content decides the strategy, repeatedly.** Carmenta has measured several
per-content sign-flips where the better choice on rendered text is the worse
choice on photographs:

| Decision | Rendered/screen | Photographs |
|---|---|---|
| engine | `craft-crnn` — 1.602 % vs 5.034 % | `craft-parseq` — 21.70 % vs 27.42 % |
| word segmentation | ink-gap projection — 0.149 % vs 0.673 % | CRAFT boxes direct — 21.70 % vs 34.16 % |

Segmentation dispatches automatically on a per-image signal (the fraction of
exactly-equal adjacent pixels: 0.88–0.99 on rendered content, 0.10–0.51 on camera
captures — a 0.37-wide empty band between the classes). Override with
`FFAI_CONTENT=rendered|photo`; pick the engine yourself like a codec.

**Detection input is scale-normalized adaptively** — small photos magnified above
CRAFT's ~8 px glyph floor, camera-resolution images capped
(`FFAI_DET_TARGET`, default 1536). Worth 2x detection latency and multi-GB peak
memory on phone photos.

**Big fonts break line detection, and the fix has to be gated.** On slides and
exam papers a text detector trained on body copy returns one box per *word*
("Facts / about / our / students"), which then gets sequenced as word soup.
Merging word fragments back into lines *before* recognition — so the
recognizer reads the whole line in context — is worth **+0.010 text and
+0.004 order** on the full benchmark. Applying it everywhere instead costs
**−0.021**: a wrong merge is irrecoverable, and pages that were already
correctly line-per-box pay for the minority that weren't. So it dispatches on
a page signal (the fraction of boxes sitting in mergeable chains: 0.25+ on
fragmented pages, 0.02 on clean ones). On by default; `FFAI_WORD_MERGE=0`
turns it off.

**Real photographs remain the open front.** On photographed receipts the full
pipeline still trails PaddleOCR (20.9 % vs 15.6 % CER) even though our
recognition stage beats theirs on identical crops — the diagnosed cause is
tilt-sensitive line grouping, with deskew as the named fix.

## Status: `experimental`, honestly

Documents are the strongest surface: 1651/1651 pages processed with zero
engine failures on OmniDocBench v1.6, scored by the benchmark's own evaluator,
and a gap to the VLM leaders that is characterised rather than guessed. The
table/formula routing stage is opt-in and experimental — its model weights are
not yet wired into the `ffai-models` cache. Inline-formula splicing is built
but **off by default and not recommended yet**: it improves the text column
and perturbs block matching enough to cost reading order, which is a
non-regression gate here, so it ships as a knob with the measurement written
down rather than as a default. LIVE holds zero-churn and beats its own batch
mode on accuracy; its batch-parity check currently reports breaks that trace
to the change gate's tolerance rather than to lost accuracy, and that gate
definition is being fixed rather than explained away. Photographs trail
PaddleOCR.

Every number traces to the campaign log at
[`docs/plans/benching-history-made.md`](https://github.com/Remade-With-Rust/FFAI/blob/master/docs/plans/benching-history-made.md),
which records the losses and refuted hypotheses beside the wins — including
the ordering variants that failed, the instrument errors that produced
retracted findings, and the scorer bias that forced this page's earlier
numbers to be re-measured on the official evaluator.

## Where this sits

| Crate | Role |
|---|---|
| [`ffai-cli`](https://crates.io/crates/ffai-cli) | the `ffai` binary — every component behind one command |
| [`ffai-core`](https://crates.io/crates/ffai-core) | engine traits, shared types, the registry; candle is the tensor spine |
| [`ffai-media`](https://crates.io/crates/ffai-media) | ingest and egress — images, audio, video, subtitle formats |
| [`ffai-models`](https://crates.io/crates/ffai-models) | hash-verified weight manifests and the local cache |
| **[`ffai-carmenta`](https://crates.io/crates/ffai-carmenta)** | **← you are here** — OCR: detection, recognition, reading order, LIVE |
| [`ffai-mercury`](https://crates.io/crates/ffai-mercury) | speech — ASR (Whisper/WhisperX-class) and TTS (VITS/piper-class) |
| [`ffai-diana`](https://crates.io/crates/ffai-diana) | object detection, depth and tracking (YOLO26) |
| [`ffai-argus`](https://crates.io/crates/ffai-argus) | vision-language captioning — pending build |
| [`ffai-bench`](https://crates.io/crates/ffai-bench) | the measurement harness every number on this page comes from |

Engines are selected by lineage name, like codecs in ffmpeg; `ffai engines`
lists them all with status.

## The Remade With Rust ecosystem

<!-- ORG BOILERPLATE — keep identical across repos -->

**Remade With Rust** is an initiative by **[Mata Network](https://www.mata.network/)**
to rebuild essential C and C++ tools in Rust — for the memory safety, the
predictable performance, and the freedom of a permissive license. Each project
is a reimplementation, not a fork: same wire protocols and file formats, new
code you can actually depend on. No copyleft. No surprises.

| Project | What it is |
|---|---|
| 🎬 **[remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs)** | **Our FFmpeg alternative.** Drop-in `ffmpeg` and `ffprobe` binaries — demux → decode → filter → encode → mux, rebuilt as composable Rust crates with **zero GPL/LGPL**. Apache-2.0. `rusty_h264` is its H.264 codec. |
| 🧠 **[FFAI](https://github.com/Remade-With-Rust/FFAI)** | **Our sister project: media *for* AI.** "The AI media toolkit, remade with rust." Embedded ASR + TTS (**Mercury**), OCR (**Carmenta**) and vision-language captioning (**Argus**) behind an ffmpeg-style, swap-by-name architecture — no Python, no CUDA. MIT OR Apache-2.0. |
| 🌐 **[Mata Network](https://www.mata.network/)** | **The home page.** *"Stop sacrificing your privacy for convenience."* Sovereign, self-hostable privacy infrastructure — wallet & identity, password manager, contact manager, and a browser extension that stops information leaking as you browse. Remade With Rust is its open-source arm. |

→ All projects: **[github.com/Remade-With-Rust](https://github.com/Remade-With-Rust)**

<!-- /ORG BOILERPLATE -->

## License

MIT OR Apache-2.0, at your option. **Model weights are not covered by it** —
each carries its own license, surfaced at selection time and in
`ffai models`.
