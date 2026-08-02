<!--
Landing README for a standalone `Remade-With-Rust/carmenta` repository, mirroring
what `Remade-With-Rust/mercury` does for Mercury. It is kept here so it versions
with the code whose numbers it quotes; copy it to that repo's root as README.md.

Rule inherited from Mercury's page: every figure is a ledger line or it does not
appear. Where a claim is qualified, the qualifier ships with it.
-->

# Carmenta

**Optical character recognition in pure Rust — documents, screens, and live frames.**

Carmenta reads text out of images: detection, mixed-case recognition, and
reading order computed from geometry rather than a layout model. It runs on
[candle](https://github.com/huggingface/candle), downloads its weights from
hash-verified manifests, and has no Python runtime, no C/C++ by default, no
gated models and nothing GPL.

It is the OCR component of [FFai](https://github.com/Remade-With-Rust/FFAI),
named for the Roman goddess who adapted the Greek alphabet into Latin letters.

```toml
[dependencies]
ffai-carmenta = "0.6"
ffai-core = "0.6"
ffai-media = "0.6"
```

---

## What it does well, measured

### Documents

On the [OmniDocBench](https://github.com/opendatalab/OmniDocBench) English
holdout — 236 real pages, Apache-2.0, the benchmark Baidu's Unlimited-OCR states
its record on:

| | CER | correctness |
|---|---:|---|
| `mobiledet-crnn` | **20.27 %** micro · 31.41 % macro | **236/236** |

Reading order is computed by a recursive XY-cut over detector boxes, routed
per-node between a column grid and a projection cut. **No layout model, no extra
weights.** It is worth 4.5 points of CER, and 3.4x against reading top-to-bottom.

### Against the state of the art

43 pages, all three engines through the same harness and the same metric:

| | CER | pages/s | memory |
|---|---:|---:|---:|
| Unlimited-OCR (Baidu, 3B MoE, **GPU**) | 15.51 % | 0.01 | 8745 MiB peak |
| PP-StructureV3 | 19.14 % | 0.02 | 1481 MiB steady |
| **Carmenta (CPU)** | 25.91 % | **0.17** | **425 MiB steady** |

Behind on quality. **17x the throughput with no GPU, on 4.7 MB of detector
weights against 6.4 GB** — a deployment class neither reference can enter.

And the gap is characterised, not guessed: **89 % of it is sequence rather than
characters.** Strip ordering from both outputs and Carmenta sits **1.40 pp** from
PP-StructureV3 (69 pages, z = +2.29). It reads the characters about as well; it
assembles them worse. An oracle layout model is worth ~12 further points — the
measured ceiling any ordering work competes for.

### LIVE — a streaming mode nothing mainstream ships

Point it at a sequence of frames and it behaves like a stream instead of a
loop over images. A frame whose pixels have not moved reuses the previous
result, so **text cannot flicker on an unchanged frame** — the failure that makes
naive per-frame OCR unusable for screen reading.

On a pinned 180-frame screencast:

| | Carmenta LIVE | per-frame Tesseract |
|---|---|---|
| OCR calls for 24 text changes | **24** | 180 |
| churn on 156 unchanged frames | **0** | 24 |
| CER on change frames | **1.21 %** | — |

LIVE also beats the *same engine* in batch mode on the same frames
(1.21 % vs 1.74 %).

### Photographs — the open front

On photographed receipts the full pipeline trails PaddleOCR (**20.9 % vs
15.6 %**) even though the recognition stage beats theirs on identical crops
(**1.5 % vs 3.0 %**). The cause is diagnosed — tilt-sensitive line grouping —
and deskew is the named fix. It is stated here because a page that only lists
wins is not a measurement.

---

## Use it

```rust
use ffai_core::engine::{OcrEngine, OcrOptions};
use ffai_carmenta::engine::{CraftCrnn, RecStage};

let engine = CraftCrnn::new_mobiledet(RecStage::Crnn);
let out = engine.recognize(&ffai_media::load_image("page.png")?,
                           &OcrOptions::default())?;
println!("{}", out.text());
```

Engines are chosen by lineage name, like codecs in ffmpeg — `mobiledet-crnn` for
documents, `craft-crnn` for screens, `craft-parseq` for photographs. The choice
stays yours; only the per-image segmentation strategy auto-dispatches, on a
measured signal with an empty band between the content classes.

```sh
ffai ocr -i page.png --engine mobiledet-crnn
ffai ocr --live --watch 5 -i captures/ -o screen.srt
```

---

## Honest scope

* **Frame sources.** `LiveSession` takes pixels and is decoder-agnostic. A
  directory of stills decodes through `rusty_png`/`rusty_jpeg`; video ingest is
  not built; camera/MJPEG input is untested. "Point it at a screen" today means
  frames on disk.
* **Determinism.** Run-to-run CER varies ~0.5 pp on identical inputs
  (non-deterministic parallel reduction). Differences below that are not
  results.
* **Batch parity in LIVE.** The harness reports parity breaks against batch
  mode; they trace to the change gate's tolerance rather than to lost accuracy —
  the quality gate has LIVE *ahead* of batch. The gate definition is being fixed
  rather than explained away.
* **Status is `experimental`.** Documents are the strongest surface; photographs
  trail PaddleOCR.

Every figure above is a line in
[`bench/ledger.jsonl`](https://github.com/Remade-With-Rust/FFAI/blob/master/bench/ledger.jsonl).
The [mission plan](https://github.com/Remade-With-Rust/FFAI/blob/master/docs/Carmenta-mission-plan.md)
records the losses beside the wins — five refuted ordering variants, two
retracted findings, and the instrument errors that produced them.

## License

MIT OR Apache-2.0. Model weights carry their own licenses, surfaced at selection
time.
