# ffai-carmenta

**OCR in pure Rust, with a LIVE streaming mode no mainstream OCR tool ships** — text detection, mixed-case recognition, and a change-gated screen/frame reader that never re-rolls on noise. No Python, no C/C++ by default, no gated weights.

Carmenta is [FFai](https://github.com/Remade-With-Rust/FFAI)'s OCR component, named for the Roman goddess who adapted the Greek alphabet into Latin letters.

```toml
[dependencies]
ffai-carmenta = "0.4"
ffai-core = "0.4"
ffai-media = "0.4"
```

## Recognize

```rust
use ffai_core::engine::{OcrEngine, OcrOptions};
use ffai_carmenta::engine::CraftCrnn;

let engine = CraftCrnn::new();                       // craft-crnn, the default
let image  = ffai_media::load_image("page.png")?;
let out = engine.recognize(&image, &OcrOptions::default())?;
println!("{}", out.text());

for line in out.lines() {                            // blocks -> lines -> words,
    println!("{:?} {}", line.bbox, line.text);       // reading order = Vec order
}
```

Weights download once, into a local cache, from hash-verified manifests — CRAFT detection (MIT) and the english_g2 CRNN (Apache-2.0), the EasyOCR model stack reimplemented on candle and oracle-matched against PyTorch (detection to <5e-3, recognition to the exact CTC argmax).

Two engines, selected by lineage name like codecs in ffmpeg:

| Engine | Recognition | Where it wins (measured) |
|---|---|---|
| `craft-crnn` *(default)* | line-level CTC | clean UI/screen text, LIVE |
| `craft-parseq` | word-level AR + refinement (PARSeq-tiny, Apache-2.0) | photographs — reads real-photo crops at **1.5 % CER where PaddleOCR's own recognizer reads 3.0 %**, 2.6× faster |

## LIVE: point it at a screen

`LiveSession` wraps **any** `OcrEngine` and adds what a stream needs: a change gate (a frame whose pixels didn't move reuses the last result at ~zero cost — which also means an unchanged frame *cannot* flicker its text), auto-ROI band tracking with background full-frame sweeps, and timed SRT/VTT output.

```rust
use ffai_carmenta::live::{LiveConfig, LiveSession};

let mut live = LiveSession::new(engine, OcrOptions::default(), LiveConfig::default());
for (i, frame) in frames.iter().enumerate() {
    let out = live.push_frame(frame, i as f64 / fps)?;
}
let (track, stats) = live.finish(duration);          // TimedSegment<String> spans
```

Or from the CLI, interoperating with any capture tool that writes frames:

```sh
ffai ocr --live --watch 5 -i captures/ -o screen.srt
```

Measured on a pinned 180-frame screencast: **24 OCR calls for exactly 24 text changes**, zero churn across 156 unchanged frames (stateless engines churn 24 times on the same frames), steady p95 **230 ms/frame vs 377 ms** for per-frame Tesseract, flat memory over a 30-minute soak.

## Things measurement taught us, which you may want

**Recognition lineages are content-dependent, with a sign flip.** The CRNN beats PARSeq on crisp UI text; PARSeq beats the CRNN 7× on photographs (and beats PaddleOCR's recognizer there too). Pick per content; averaging the two loses both wins.

**Detection input is scale-normalized adaptively** — small photos are magnified above CRAFT's measured ~8 px glyph floor, camera-resolution images are capped (`FFAI_DET_TARGET`, default 1536). This is a 2× detection-latency and multi-GB peak-memory difference on phone photos.

**Real photographs are the open front.** On photographed receipts the full pipeline still trails PaddleOCR (27 % vs 16 % CER) even though our recognition stage beats theirs on identical crops — the diagnosed gap is tilt-sensitive line grouping, with deskew as the named fix. The per-stage instruments that localized this ship in `examples/`.

## Status: `experimental`, honestly

LIVE holds all four of its exit gates against the C++ per-frame bar (latency, zero-churn, accuracy parity, flat footprint). Batch quality sits between EasyOCR (beaten ~5× on its own model stack) and PaddleOCR mobile (behind on photos, ahead at the recognition stage). Every number traces to a line in [`bench/ledger.jsonl`](https://github.com/Remade-With-Rust/FFAI/blob/master/bench/ledger.jsonl); the mission plan records losses and refuted hypotheses beside the wins.

## License

MIT OR Apache-2.0. Model weights carry their own licenses, surfaced at selection time.
