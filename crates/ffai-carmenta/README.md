# ffai-carmenta

**OCR in pure Rust — documents, screens, and a LIVE streaming mode no mainstream OCR tool ships.** Text detection, mixed-case recognition, computed reading order, and a change-gated frame reader that never re-rolls on noise. No Python, no C/C++ by default, no gated weights, nothing GPL.

Carmenta is [FFai](https://github.com/Remade-With-Rust/FFAI)'s OCR component, named for the Roman goddess who adapted the Greek alphabet into Latin letters. Standalone landing page: [Remade-With-Rust/carmenta](https://github.com/Remade-With-Rust/carmenta).

```toml
[dependencies]
ffai-carmenta = "0.6"
ffai-core = "0.6"
ffai-media = "0.6"
```

## Recognize

```rust
use ffai_core::engine::{OcrEngine, OcrOptions};
use ffai_carmenta::engine::CraftCrnn;

let engine = CraftCrnn::new_mobiledet(RecStage::Crnn);   // mobiledet-crnn
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
| `mobiledet-crnn` | PP-OCRv5 mobile (4.7 MB) | line CTC | **documents** — the strongest on real pages |
| `craft-crnn` | CRAFT VGG16-BN | line CTC | clean UI/screen text, LIVE |
| `craft-parseq` | CRAFT | word AR (PARSeq-tiny) | photographs — **1.5 % CER on real-photo crops where PaddleOCR's own recognizer reads 3.0 %**, 2.6x faster |

## Documents

Reading order is **computed, not learned**: a recursive XY-cut over the boxes the
detector already produced, routed per-node between a column grid and a
projection cut. No layout model, no extra weights.

Measured on the [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
English holdout (236 pages, Apache-2.0 — the benchmark Baidu's Unlimited-OCR
states its record on):

| 236 real document pages | CER | correctness |
|---|---:|---|
| `mobiledet-crnn` | **20.27 %** micro · 31.41 % macro | 236/236 PASS |

Reading order is worth **4.5 points of CER** on this corpus and the recursive cut
is worth **3.4x** against raster ordering. An oracle layout model would be worth
a further ~12 points — measured, and the ceiling every ordering idea competes
for.

On a 43-page subset where all three engines ran through the same harness and the
same metric:

| 43 pages | CER | pages/s | memory |
|---|---:|---:|---:|
| Unlimited-OCR (Baidu, 3B MoE, **GPU**) | 15.51 % | 0.01 | 8745 MiB peak |
| PP-StructureV3 | 19.14 % | 0.02 | 1481 MiB steady |
| **`mobiledet-crnn` (ours, CPU)** | 25.91 % | **0.17** | **425 MiB steady** |

Behind on quality, **17x the throughput on a machine with no GPU, from 4.7 MB of
detector weights against 6.4 GB**. Where the remaining gap lives is measured:
**89 % of it is sequence, not characters** — order-free CER sits 1.40 pp from
PP-StructureV3 (69 pages, z = +2.29). We read the characters about as well; we
assemble them worse.

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

**Real photographs remain the open front.** On photographed receipts the full
pipeline still trails PaddleOCR (20.9 % vs 15.6 % CER) even though our
recognition stage beats theirs on identical crops — the diagnosed cause is
tilt-sensitive line grouping, with deskew as the named fix.

## Status: `experimental`, honestly

Documents are the strongest surface: 236/236 correctness on OmniDocBench, and a
gap to the 3B GPU reference that is now characterised rather than guessed. LIVE
holds zero-churn and beats its own batch mode on accuracy; its batch-parity
check currently reports breaks that trace to the change gate's tolerance rather
than to lost accuracy, and that gate definition is being fixed rather than
explained away. Photographs trail PaddleOCR.

Run-to-run CER varies by ~0.5 pp on identical inputs (non-deterministic parallel
reduction), so single-run differences below that are not results.

Every number traces to a line in
[`bench/ledger.jsonl`](https://github.com/Remade-With-Rust/FFAI/blob/master/bench/ledger.jsonl);
the [mission plan](https://github.com/Remade-With-Rust/FFAI/blob/master/docs/Carmenta-mission-plan.md)
records the losses and refuted hypotheses beside the wins — including five
ordering variants that failed and the instrument errors that produced two
retracted findings.

## License

MIT OR Apache-2.0. Model weights carry their own licenses, surfaced at selection
time.
