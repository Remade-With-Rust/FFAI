# Diana Mission Plan

**Component:** Diana — FFai's detection component (`ffai-diana`)
**Tasks:** vision detection across five heads — **detect**, **segment**,
**pose**, **OBB**, **classify** — plus a streaming/LIVE loop. DETECT first,
edge-first
**Status:** Phase 0 planning · this plan takes the component to `stable`
**Prime directive:** pure Rust end-to-end on the candle spine, loading
official Ultralytics `.pt` checkpoints through an audited, reproducible,
offline conversion to safetensors + manifest — never a runtime Python or
libtorch dependency. Measured against the non-Rust world standards — above
all the official Ultralytics Python path and the ONNX Runtime CPU export —
by `ffai bench detect` at every milestone. No claim without a ledger line.

---

## 1. Mission

Ship the first production-grade, pure-Rust YOLO spine:

1. **The architecture, reimplemented** — YOLO26 (and YOLO11/YOLOv8 blocks
   where the delta is small) faithfully enough that official weights load
   strict and produce numerically matching outputs, oracle-gated per block.
2. **`.pt` deployment without Python** — a developer runs the audited
   conversion once, ships a pure-Rust binary, and gets latency competitive
   with or better than the common ONNX Runtime CPU path on the same
   edge-class hardware.
3. **The task surface as compositions** — detect → segment → pose → OBB →
   classify over one backbone+neck core, plus a LIVE streaming path with
   the change-gate discipline Carmenta proved.

Every milestone exits through the analyzer: `ffai bench detect` compares
our Rust code to the **non-Rust world standards** — Ultralytics Python,
ONNX Runtime CPU, and the vendor runtimes where relevant — on pinned
corpora, and the result, win or loss, is appended to `bench/ledger.jsonl`.

**Success =** numerical parity with the official model on pinned holdouts
(per-block activation max-abs/cosine within recorded tolerances; final
boxes and scores within a tight relative band on identical pixels); p95
warm latency competitive with or better than ONNX Runtime CPU for the n/s
tiers on the same hardware; edge claims (frame latency, steady memory,
cold start, long-stream flatness) ledger-backed. Every public number maps
to a ledger line id.

**What Mercury and Carmenta proved that this plan assumes:** candle on CPU
can stand toe-to-toe with hand-tuned native code when the architecture,
the oracles, and the measurement harness are honest — and the harness is
how you find out where it can't. Diana therefore does **not** carry ONNX
Runtime or libtorch as the spine — the spine is candle, and any
accelerator backend must arrive as a feature flag with a measured reason
(Principles 1 and 3).

**Non-goals (v1), stated now rather than discovered later:**

- Training or fine-tuning in Rust (stays Python-side; we consume
  checkpoints).
- Open-vocabulary / YOLOE-class prompt heads (explicit later milestone,
  §9 watchlist).
- Shipping or redistributing Ultralytics weights (AGPL-3.0 — conversion
  tools + manifests only, §7).
- Replacing every export target (TensorRT, CoreML, …) on day one — those
  are references to measure against, not features to match.

---

## 2. Design rule: independent stages, composed functions

Diana is a **toolbox of independently callable stages**, each with its own
contract and oracle test. The task heads and the live loop are
*compositions* over one core — flags and loops, not forks. This is the
WhisperX lesson, applied from day one exactly as Carmenta applied it.

```
ffai-diana
├── preprocess/       letterbox, normalize, color convert (independent,
│                     optional, inverse transform recorded)
├── blocks/           Conv+BN+SiLU, Bottleneck, C3k2 (c3k/attn flags),
│                     SPPF (+ residual), C2PSA/PSABlock, DWConv
├── backbone/         stem + C3k2 stages + SPPF + C2PSA (candle)
├── neck/             FPN top-down + PAN bottom-up (upsample/concat/C3k2)
├── head/             dual Detect head (one-to-one NMS-free default +
│                     one-to-many), then Seg / Pose / OBB / Classify heads
├── decode/           box decoding, score filtering, optional NMS,
│                     letterbox-inverse coordinate mapping
├── live/             streaming loop: frame sampler, change gate, optional
│                     track association, timed output
├── convert/          .pt → safetensors + manifest: key/shape map,
│                     strict load, intermediate-activation oracles
└── engine.rs         composes the above → DetectOutput / task outputs
```

Contracts that keep the stages independent:

- Every stage consumes/produces `ffai-core` types (`ImageBuffer`,
  `DetectOutput`, `TimedSegment`) or plain candle tensors — no stage
  reaches into another's internals.
- Each major block **and** the full forward pass has **its own oracle
  test** against a deterministic fixture (§6.2): pinned reference dumps
  from the official model on synthetic images, per-block max-abs/cosine,
  final boxes/scores in a recorded band.
- The live/decode/track stages operate on *any* engine's output — they
  wrap results, they do not reach into a model's internals. When a better
  backbone replaces the first one, LIVE survives unchanged.
- Conversion maps are versioned (§7). A YOLO26n map does not silently
  load a future architecture revision.

**Functions are flags, engines are lineages** — and engines are named by
lineage, the rule Carmenta's stub-rename converged on. `--engine yolo26n`
selects the weight + architecture lineage (the codec); `ffai detect`,
`--segment`, `--pose`, `--obb`, `--live` select the composition. Nothing
that has not earned a default gets one.

### 2.1 The output type is the AVFrame of this mission

`DetectOutput` does not exist in `ffai-core` today; neither does a
`DetectEngine` trait. Both are designed at M-D1 and reviewed against all
five task heads before anything ships, because every engine, every
function, and the bench harness will depend on them:

- image → list of detections;
- geometry: xyxy/xywh box, optional polygon, keypoints, oriented box
  (angle + long-edge convention pinned in the type, not the docs);
- per-detection confidence + class id + optional track id;
- masks/coefficients as typed payloads when the segment milestone lands,
  not speculatively;
- reversible mapping back to original image coordinates (the letterbox
  inverse travels with the output);
- `TimedSegment<DetectOutput>` wrapping for LIVE (same converters Mercury
  and Carmenta use), and conversion helpers for JSONL and downstream
  consumers — including Carmenta region proposals (§10).

Open-vocabulary and dense-caption structures stay **out** of the v1 type.

---

## 3. The detection spine (the substrate)

Not one of the task functions — the thing all of them stand on, and the
first code built (M-D1).

| Capability | Contract |
|---|---|
| Backbone | YOLO26-style stem + repeated C3k2 stages (c3k/attn flags as in the official YAML), SPPF with residual when channels match, C2PSA |
| Neck | FPN top-down + PAN bottom-up, C3k2 fusion blocks |
| Head | dual Detect: one-to-one (default, NMS-free, fixed top-k output) + one-to-many; **DFL removed**, direct box regression per the YOLO26 formulation |
| Model tiers | n / s / m / l / x (+ task variants), selected via `ffai-models` manifests; n and s first — they are the edge tiers |
| Input | letterboxed, default 640; resolution tiers configurable |
| Quantization | fp16 / int8 variants measured per stage — Mercury's lesson: quantization pays by kernel shape, not by faith (encoder 2.4× regression vs decoder 3× win) |
| Weight loading | `.pt` → audited conversion (§7) → safetensors + manifest; **strict load, fail closed** on missing, unexpected, or shape-mismatched keys |

**Lineage selection is an M-D0 audit output, not a preference.** Primary
target is **YOLO26**; YOLO11/YOLOv8 are compatibility targets only where
blocks are shared. Candidates are scored on: weight license *and*
redistribution terms (§7), cleanliness of conversion to candle, size-tier
fit for edge, and existence of a reference implementation to oracle
against.

**Architecture fidelity rules (non-negotiable for weight loading):**

- Exact channel counts, kernel sizes, strides, and residual connections.
- C3k2 internal variants (Bottleneck vs C3k vs attention) follow the YAML
  flags, never a guess.
- SPPF residual behaviour matches the official shortcut when `c1 == c2`.
- Head output shapes and decode logic match the chosen path (end2end vs
  NMS) exactly.
- No silent Conv+BN fusion at load time unless the conversion artifact
  explicitly records fused weights (§7 step 6).

---

## 4. The functions

### 4.1 DETECT — first, by mission order

Core object detection. The degenerate case (one image) is `ffai detect`;
LIVE is the loop around it.

```
ffai detect -i image.jpg --engine yolo26n -o out.json
ffai detect -i video.mp4 --live -o tracks.jsonl
ffai detect -i captures/ --live --watch 5 --conf 0.25
```

| Piece | Contract |
|---|---|
| Preprocess | letterbox + normalize; inverse transform recorded for box mapping |
| Forward | backbone → neck → dual head |
| Decode | one-to-one path default (NMS-free); one-to-many + NMS opt-in |
| Output | boxes, scores, classes in original image coordinates |

**DETECT metrics** (all ledger-recorded, best-of-N, warm/cold separated
exactly as [benchmarking.md](benchmarking.md) requires):

- **parity:** per-block activation max-abs/cosine vs the official dump;
  final box IoU / score deltas within recorded bands on identical pixels;
- box quality (AP-proxy) on synthetic + audited public holdouts;
- p50/p95 per-image latency, warm; time-to-first-detection cold, with
  model load recorded separately;
- steady MiB, peak beside it.

### 4.2 SEGMENT / POSE / OBB / CLASSIFY

Task-specific heads on the shared backbone + neck, each landing after the
detect core is stable, each through the same conversion + oracle
discipline:

- **Segment** — proto module + per-detection mask coefficients;
- **Pose** — keypoint regression matching the official head's
  formulation;
- **OBB** — oriented boxes, long-edge/angle convention pinned in
  `DetectOutput`;
- **Classify** — global classification head.

Priority order inside M-D4 is decided by measurement and downstream
demand — Carmenta region proposals favour detect + light segment early —
not by the order Ultralytics lists them.

### 4.3 LIVE / STREAMING

Streaming detection over frame sources: video, camera, screen via the
watch-directory interop LIVE OCR shipped, rff ingest when it publishes.

| Piece | Contract |
|---|---|
| Frame sampler | every-N / keyframe / change-triggered; a skipped frame costs zero model work |
| Change gate | an unchanged frame **must** produce identical output at ~zero cost (byte-identical, or a recorded tolerance) |
| Association | optional lightweight IoU / ByteTrack-class track ids — measured, not assumed |
| Smoothing | temporal smoothing only where it reduces churn without breaking the latency gate |
| Timed output | `TimedSegment<DetectOutput>` → JSONL/SRT, same converters as Mercury/Carmenta |

**LIVE is the edge showcase.** Carmenta's change-gate discipline transfers
whole: churn on identical frames is a defect, measured as such, and the
gate is calibrated on a synthetic corpus whose ground truth includes
*when* content changed (§6.2).

**LIVE metrics:**

- p50/p95 warm frame latency vs the ONNX Runtime path on the same frames;
- time-to-first-detection (cold);
- churn rate on identical consecutive frames (gate: zero, or a documented
  tolerance);
- box-stability metrics on the pinned change/no-change corpus;
- detection parity with our own single-image mode on the same frames
  (LIVE must not silently trade accuracy for latency);
- footprint flat over a long synthetic stream (30-min soak, the Carmenta
  gate).

---

## 5. World-standard references (non-Rust)

Declared in `corpora/references.toml`; versions recorded per ledger line.

| Reference | Stack | Status | Why it's the bar |
|---|---|---|---|
| **Ultralytics (official)** | Python + PyTorch | ✅ measured, n + s | source of truth for architecture, weights, and numerical behaviour — the parity oracle |
| **Ultralytics ONNX export + ONNX Runtime CPU** | ORT (C++) | ✅ measured, n + s | the practical deployment bar most edge users actually compare against — the whisper.cpp of this mission, and at M-D0 the leader on speed *and* memory (§8.1) |
| Rust + ORT wrappers | Rust over ORT (C++) | ⬜ M-D2 | the existing "YOLO in Rust" answer; Diana must justify its existence against it, and the pure-Rust distinction is measured, not asserted. Deferred because it wraps the same ORT already on the board — it prices the binding, not the inference, and that question only becomes live once Diana has something to compare |
| TensorRT / CoreML / OpenVINO | vendor runtimes | ⬜ M-D5 | secondary speed references once accelerator features land |
| Pure-Rust / candle peers | candle, custom | ⬜ M-D2 | measured for standing, not the bar |

Same discipline as Mercury's and Carmenta's tables: pin every knob that
changes the work, record exact argv in the ledger, and add
matched-configuration variants when defaults differ.

**This mission's `beam_size` turned out to be `--rect`, not the end2end
flag.** YOLO26 is natively end-to-end, so there is no NMS knob to pin — but
Ultralytics' `predict()` defaults to rectangular letterboxing while the
ONNX export is fixed square, and left unpinned that made two rows of the
same tier disagree by 1.5–1.8 pp. It fired at M-D0 and is recorded in §8.1
and the risk register. The pinned set is `--imgsz 640 --conf 0.001
--max-dets 100 --rect off` for the matched comparison, with the
rectangular default carried as its own declared config.

---

## 6. Analyzer integration (`ffai bench detect`)

The ASR and OCR verticals stand; Diana work extends the harness.
benchmarking.md gains its detection reproduce-from-source section at
M-D0.

### 6.1 Metrics, per function

| Function | Quality | Speed | Footprint |
|---|---|---|---|
| core / detect | AP-proxy + parity deltas vs official dumps | ms/image warm + e2e | steady MiB, peak beside it |
| LIVE | quality on change frames; churn on static | p50/p95 frame latency, time-to-first-detection | steady MiB **flat over a 30-min stream** |
| segment / pose / obb / classify | mask AP / OKS / OBB mAP / top-1 | frames/s | steady MiB |

### 6.2 Corpora — two kinds, deliberately

**Synthetic, deterministic, license-free (the mel-fixture trick).**
Rendered scenes with exactly known boxes — and later masks and keypoints
— from pinned scripts and open assets: a corpus with *exact* ground truth
that anyone can regenerate, no license, no download gate. A scripted HUD
/ moving-object sequence with timed changes is the LIVE change-gate
oracle, exactly Carmenta's screencast corpus. These are the smoke corpora
and per-stage oracles. The **parity fixtures** are their sibling: pinned
input images + the official model's dumped activations, regenerated by a
recorded script.

**Public ground-truth sets (the claims corpora).** Synthetic corpora
cannot support a public claim about real photographs. Candidates (COCO
val2017 first; VOC, LVIS, DOTA for OBB) are **audited at M-D0** for
license *and* fetchable-without-an-account — the pyannote lesson applies
to corpora too, and COCO needs the audit read carefully: the
*annotations* are CC-BY-4.0 but the images carry individual Flickr
licenses, so what a claims corpus may contain is an audit output, not an
assumption. Chosen sets get hash-pinned manifests with holdout splits;
claims are measured on holdout only.

### 6.3 Harness work items

- ✅ image corpus support shared with Carmenta — the manifest schema took
  detection unchanged (`task = "detect"`, ground truth a JSON boxes file
  per image instead of a text file);
- ✅ detection scoring beside the existing CER/WER code —
  `crates/ffai-bench/src/detect.rs`, cross-validated against pycocotools
  (§8.1);
- ✅ latency modes: per-image p50/p95 from adapter timings, warm/e2e as
  two numbers, steady/peak memory — all inherited from the shared
  `run_reference` contract;
- ✅ reference adapters under `corpora/refs/` (Ultralytics Python and ORT
  via the existing batch JSONL contract, boxes carried in the `text`
  field);
- ⬜ intermediate-tensor dump + max-abs/cosine parity scoring against
  pinned official dumps — **M-D1**, it needs the module names the port
  defines;
- ⬜ stage-level timing (preprocess / backbone / neck / head / decode) so
  every speed campaign starts from a profile — **M-D1**, same reason.

---

## 7. Weights: license, conversion, and gating (Principle 4)

Ultralytics weights are **AGPL-3.0**. Diana's code stays under the
workspace's permissive license; the weights are never vendored, never
redistributed, and the license surface is documented by `ffai models`,
not hidden. What ships is the conversion tool and the manifest — the user
fetches the checkpoint themselves, and the obligations that follow from
AGPL weights are stated plainly in the docs.

**The conversion pipeline is a first-class product artifact (M-D1), not a
footnote:**

1. Load the official `.pt` in a controlled, pinned environment.
2. Extract the state dict; map keys to Diana module names; verify every
   tensor's shape and dtype against the expected architecture.
3. Emit safetensors + a machine-readable manifest: architecture id,
   strides, nc, task, imgsz, end2end flag, conversion-map version, sha256
   of source and output.
4. Strict-load in candle; **fail closed** on missing, unexpected, or
   shape-mismatched keys — the `strict=True` lesson from Carmenta's CRAFT
   port, where the repo's own constructor calls didn't match the shipped
   checkpoint.
5. Oracle the full forward **and** selected intermediate activations
   against the source model on the pinned fixture.
6. Record the fused-vs-unfused decision explicitly: if the conversion
   fuses Conv+BN, the artifact says so and the oracle covers it.

Conversion maps are versioned: a map for `yolo26n-detect@<release>` does
not silently accept a later architecture revision. Tools live in
`tools/`, re-runnable from a fresh clone (the oracle-fixtures-ship
lesson).

**Audit questions (M-D0 deliverable, answered against live sources, not
reputation):**

| Question | Why it matters |
|---|---|
| exact license + redistribution terms per checkpoint family | AGPL obligations differ from "can't ship" — state precisely what a Diana user takes on |
| fetchable without an account? | the pyannote rule; verified per URL on a recorded date |
| YOLOE / open-vocabulary extra constraints | before the watchlist item is ever promoted |
| which size tiers are realistic for edge | n/s first is the hypothesis; the audit prices m/l/x honestly |
| any custom op that blocks a clean candle port | discovered at audit time for the price of a script, not at port time — the TrOCR lesson |

### 7.1 Audit results (M-D0, verified 2026-07-30)

Every row verified against a live URL or a local artifact on the date
above, not answered from reputation.

**Weights — cleared, with the license stated rather than softened:**

| Item | Verdict |
|---|---|
| YOLO26 checkpoints (n/s/m/l/x) | **AGPL-3.0**, ungated. `ultralytics` fetches them from `github.com/ultralytics/assets/releases/download/v8.4.0/` — no account, no click-through. Verified by downloading **yolo26n.pt (5,544,453 B)** and **yolo26s.pt (20,422,725 B)**, both reporting `task: detect`. |
| Task variants | detect / segment / semantic-seg / depth / classify / pose / OBB, all five tiers — every head §4.2 plans has released weights, so M-D4's order is a measurement decision, not an availability one |
| Redistribution | Diana **never vendors or redistributes** them. The conversion tool + manifest ship; the user fetches the checkpoint. What AGPL obligations a Diana user takes on is documented, not buried (§7). |
| Custom ops | none found blocking a candle port — the ONNX export contains no custom-domain operators |

**The end-to-end contract, verified rather than assumed.** The docs say
YOLO26 is natively NMS-free; the export proves it. `yolo export
format=onnx imgsz=640` emits a single output of shape **`[1, 300, 6]`** —
final detections (`x0,y0,x1,y1,conf,cls`), fixed top-300, no NMS
subgraph — for both tiers. That shape is now a **checked precondition**:
`tools/diana_export_onnx.py` deletes the file and fails if it ever changes,
and `corpora/refs/yolo_ort_ref.py` refuses to run against anything else
rather than supplying NMS glue that belongs to the engine under test.

Published reference points for the n and s tiers (Ultralytics' own COCO
val numbers, quoted as *their* claim, to be answered on our own corpus):
**yolo26n 40.9 mAP / 38.9 ms CPU-ONNX**, **yolo26s 48.6 mAP / 87.2 ms**.

**Corpora — cleared, with one real constraint the reputation would have
missed:** COCO val2017 annotations are CC-BY-4.0 and the images download
ungated from `images.cocodataset.org` (HEAD-verified 200 on both the
241 MB annotation zip and a sample image). **But COCO images carry
individual Flickr licenses**, ids 1–8 in the annotation file's own
`licenses` table, several of which are NonCommercial or NoDerivs. A
"COCO is CC-BY" assumption would have put NC-licensed images in a claims
corpus. `tools/diana_coco_corpus.py` therefore admits only license ids
{4, 5, 7, 8} (Attribution, Attribution-ShareAlike, no-known-copyright,
US-Government-Work) and records the license *name* per clip. Of 5000
val2017 images, **1166 are eligible** after that filter plus the two
below, which is ample.

**Two corpus-design decisions recorded as decisions:**

1. **Crowd exclusion.** Images containing any `iscrowd=1` annotation are
   ineligible. pycocotools treats crowd regions as ignore-zones during
   matching; the proxy scorer implements no ignore logic, so the corpus
   excludes the cases where that would matter instead of silently
   mis-scoring them.
2. **Deterministic selection.** Eligible images sorted by id, fixed-stride
   walk to 60 clips, `i % 4 == 3` → train (45 holdout / 15 train). No RNG
   anywhere; the manifest pins every image by SHA-256.
3. **Stored as PNG (corpus v2).** COCO ships JPEG. The corpus decodes once
   and re-encodes losslessly to PNG so **every implementation reads
   identical pixels** — left as JPEG, ours and Ultralytics' and ORT's
   decoders would each produce a slightly different array and the mAP
   delta would be a decoder comparison wearing a detector's clothes. It
   also matches FFai's PNG-only ingest until the rff decoders land. The
   reference rows re-measured **identical** to their v1 (JPEG) values,
   which is the check that the re-encode really was lossless.

**Deferred, flagged now rather than at ship time:** YOLOE / open-vocabulary
checkpoints were **not** audited — they are a §9 watchlist item behind
core detect, and their extra constraints get their own audit before that
milestone is ever promoted.

---

## 8. Milestones and exit gates

Every milestone exits through all four gates (correctness / quality /
speed / footprint) on holdout — a skipped gate blocks exit. Losses are
recorded.

| # | Deliverable | Exit gate (ledger-recorded) |
|---|---|---|
| **M-D0** ✅ | Baselines: `ffai bench detect` vertical (§6.3), audited public corpus pinned, `--baseline-only` runs of Ultralytics Python + ORT CPU at two tiers; the §7 audit | reference latency / mAP / memory on the board ✅; corpus hashes pinned ✅; audit published ✅ (§7.1); benchmarking.md detection section merged ✅ — **see §8.1** |
| **M-D1** ✅ | Core spine on candle: blocks + backbone + neck + one2one Detect head; conversion pipeline; strict load of official YOLO26n; per-layer + full-forward oracles; `DetectOutput` + `DetectEngine` in `ffai-core` | activation parity within recorded tolerances ✅; box/score parity on holdout ✅; all four gates run and recorded ✅ (3 pass, speed fails honestly) — **see §8.2, §8.3** |
| **M-D2** | **The edge latency campaign**: instruments first (pinned/CPU-time harness, stage profiler, deterministic counters, shape-correct roofline), then profiler-routed speed bricks — byte-identical order, precision changes last | p95 warm latency competitive with ORT CPU for n/s on the same machine; **per-layer parity held through every brick**; memory profile recorded — **plan in §8.4** |
| **M-D3** | **LIVE**: sampler + change gate, timed output, watch-dir interop / rff integration point, optional association | p95 vs the ORT path per-frame; zero churn on identical frames; detection parity with single-image mode; footprint flat over the 30-min soak |
| **M-D4** | Task heads (segment → pose → OBB → classify, priority by measurement) | task metrics vs the official model per head; detect core byte-unchanged (the heads compose, they do not fork) |
| **M-D5** | Accelerator backends (Metal / CUDA / …) as measured opt-in features; further quantization tiers | clear ledger win vs CPU candle on the claimed device; no correctness regression; feature-flagged per §9 |
| **M-D6** | Diana `stable`: docs, library examples, claims page generated FROM the ledger, FFAI integration polish (Carmenta region proposals) | every public claim maps to a ledger line id |

**Sequencing:** M-D1 is the critical path — nothing exists until official
weights load and match. M-D3 (LIVE) is the edge showcase and follows as
soon as DETECT is honest. M-D4's heads fan out after the substrate is
solid and can run in parallel with M-D5 — two tracks, one analyzer,
exactly the Mercury M3/M4 and Carmenta M-C4/M-C5 pattern. Accelerator
work is explicitly post-CPU-parity: a backend that hides an unoptimized
spine would un-earn the pure-Rust claim.

### 8.1 M-D0 result — baselines on the board (CLOSED 2026-07-30)

Ledger `bench-detect-1785466255`, best-of-3, CPU only, Windows x86_64.
Corpus `diana-coco` v1 (§7.1): 60 COCO val2017 images, license-filtered and
crowd-free, SHA-pinned (`38aaeee9cc62`), **45 holdout**. All six reference
configurations processed every holdout image (45/45).

| Implementation | Stack | mAP50 | mAP50-95 | img/s warm | p50/p95 per image | steady MiB |
|---|---|---:|---:|---:|---|---:|
| ultralytics yolo26n | Python/PyTorch | 68.65 | 52.59 | 14.68 | 51 / 64 ms | 297 |
| **ort yolo26n** | **ORT (C++)** | **68.65** | **52.59** | **23.62** | **27 / 106 ms** | **164** |
| ultralytics yolo26s | Python/PyTorch | 78.06 | 63.12 | 5.74 | 124 / 394 ms | 384 |
| **ort yolo26s** | **ORT (C++)** | **78.06** | **63.12** | **9.39** | **73 / 306 ms** | **272** |
| ultralytics yolo26n-rect | Python/PyTorch | 70.14 | 53.77 | 13.09 | 58 / 150 ms | 318 |
| ultralytics yolo26s-rect | Python/PyTorch | 76.26 | 61.71 | 7.69 | 100 / 227 ms | 401 |

Read the table four ways:

1. **Two independent runtimes agree to 4×10⁻⁵ — the strongest correctness
   evidence this vertical has.** At matched geometry, PyTorch and ONNX
   Runtime score **0.686499 vs 0.686462** (n) and **0.780563 vs 0.780598**
   (s). Same weights, same decode, completely different inference stacks
   and preprocessing implementations — the residual is PIL's bilinear
   against OpenCV's, nothing more. This is the tiny.en agreement from
   Mercury M0 reproduced here, and it is what says the harness measures
   the implementation rather than an accident of configuration.
2. **That agreement did not exist before the geometry was pinned, and
   finding out why is the milestone's real work.** The first M-D0 run had
   the `.pt` and ORT rows of one tier disagreeing by 1.5–1.8 pp **in
   opposite directions by tier** — n reading better under PyTorch, s worse.
   The cause: Ultralytics' `predict()` defaults `rect=True`, letterboxing
   each image to the smallest multiple-of-32 *rectangle* (a 586×640 image
   is fed as 640×608), while the ONNX export is fixed 640×640 square. The
   two rows were never doing the same work. This is precisely the M0
   defect — unpinned decode configuration making a reference look faster
   or better than it is — and it was caught by the same tell: two things
   that should agree, didn't. `--rect` is now a **required** adapter
   argument, pinned `off` for the matched comparison, with the official
   rectangular default kept as its own declared variant.
3. **The geometry sign-flips by tier, so it could never have been
   averaged away.** Isolated (rect vs square, same weights, same run):
   **n gains +1.49 pp from rectangular, s loses −1.79 pp.** A single
   "which is better" answer does not exist, which is exactly why the
   configuration is declared per row rather than chosen once. Diana will
   implement the square path — it is what the edge deployment route does —
   and the rect rows stay on the board as context.
4. **ONNX Runtime is the bar on every axis, and it is not a soft target.**
   Against the PyTorch path it is 1.6× faster warm at both tiers, at
   0.55–0.71× the steady memory, with 6× faster model load — at *identical*
   mAP. The plan called ORT "the whisper.cpp of this mission" and the
   framing survives contact: M-D2's speed gate is **23.62 img/s warm /
   27 ms p50** for the n tier, and the footprint gate is **164 MiB
   steady**. Those are the numbers a pure-Rust candle engine has to answer.

**What M-D0 leaves for M-D1/M-D2.** The det core must land within the
quality band of the 68.65 mAP50 (n) row at bring-up, and the latency and
footprint gates now have concrete targets. Two instruments were built and
proved before any of it: the mAP scorer (below) and the geometry pin (2).

**The scorer was cross-validated before it was trusted.**
`crates/ffai-bench/src/detect.rs` is a from-scratch COCO-style mAP
implementation — a new scorer, on a new corpus, written by the same hand
that will eventually be judged by it. `tools/diana_validate_scorer.py`
scores identical detections through pycocotools and through the Rust
scorer:

| metric | pycocotools | ffai-bench | delta |
|---|---:|---:|---:|
| mAP@0.5 | 0.7014 | 0.7014 | −0.0000 |
| mAP@0.5:0.95 | 0.5377 | 0.5377 | +0.0000 |

Agreement to four decimals. This is the Carmenta instrument-defect lesson
applied *in advance* rather than after a contradiction: a scorer is
cross-checked against a known-good implementation before its numbers go on
the board. One difference is recorded rather than glossed — this scorer
truncates maxDets per image across all classes where pycocotools truncates
per (image, category); the two agree here only because the adapters are
themselves pinned to `--max-dets 100`, so neither truncation binds.

**Open, and deliberately not built at M-D0:** the synthetic detection
corpus of §6.2. Its jobs are per-stage oracles (M-D1) and the LIVE
change-gate oracle (M-D3), and neither exists yet to oracle against — the
M-D1 parity fixture is pinned *official activations*, not rendered shapes.
Building it now would be a corpus with nothing to fail. Recorded as an
M-D1 entry item rather than an M-D0 omission.

Exit gate: reference mAP / images-per-second / per-image latency / memory
on the board across two stacks and two tiers ✅ · corpus hashes pinned ✅ ·
§7.1 audit published ✅ · benchmarking.md detection section merged ✅ ·
scorer cross-validated ✅. **M-D0 CLOSED.**

### 8.2 M-D1 — how the port was staged to mechanical

**The strategy, stated because it is the whole point.** Carmenta lost a
session to a port built from a repo's own constructor calls that did not
match its shipped checkpoint, and PARSeq's checkpoint carried
`dec_heads=6 / eos=0 / bos=95` that no diagram mentioned. So no Rust was
written until the architecture had been read off the artifact, the
reference activations were on disk, and a **functional reimplementation
built only from the converted weights had reproduced them**. The Rust port
is then a transliteration of known-correct code rather than an
investigation.

**What the checkpoint actually says** (`tools/diana_probe_arch.py` →
`corpora/refs/fixtures/yolo26n_arch.json`, tracked): 708 tensors,
2,591,962 params, 24 layers, `nc=80`, strides 8/16/32.

| Fact | Value | Why it matters |
|---|---|---|
| `reg_max` | **1** | DFL is genuinely gone — `self.dfl = DFL(reg_max) if reg_max > 1 else Identity`, and `no = 80 + 4×1 = 84`. The 4 box channels are `(l,t,r,b)` distances, not distribution bins |
| `end2end` | **True** | inference decodes `preds["one2one"]`; `xywh and not end2end` ⇒ boxes come out **xyxy** |
| `max_det` | 300 | fixed top-k, no NMS anywhere |
| head branches | `cv2`/`cv3` **and** `one2one_cv2`/`one2one_cv3` | see below — this is the port's highest-stakes fact |
| activations | 93 SiLU, 9 Identity | see below — the module tree lies about this |

**Three traps found before they could cost anything:**

1. **The one2many head is training-only, and porting it would be silently
   wrong.** `Detect.forward` runs `self._inference(preds["one2one"] if
   self.end2end else preds)`. The `cv2`/`cv3` branch — **120 of the head's
   240 tensors** — contributes nothing at inference. It has the same
   shapes and would produce plausible detections, which is precisely the
   failure mode Mercury hit when hidden states were argmaxed as logits:
   "still decodes to plausible text, the most dangerous kind of wrong."
   The conversion drops it explicitly and counts it in the manifest.
2. **`named_modules()` hides most activations.** Ultralytics sets
   `Conv.default_act = nn.SiLU()` as a **class** attribute, so every Conv
   taking the default shares one instance and PyTorch's module walk —
   which dedupes by object identity — lists it once and omits it
   everywhere else. `model.9.cv2` therefore *appears* to have no
   activation when it has SiLU, while `model.9.cv1` genuinely has none.
   Same shape as the torchvision inplace-ReLU trap on CRAFT: the module
   tree does not describe the forward pass. The probe now records every
   Conv's activation by direct access (93 SiLU / 9 Identity).
3. **The end-to-end top-k is two-stage.** `postprocess` takes the top-300
   *anchors* by best class score, then takes the top-300 over the
   flattened (300 × 80) class scores — so **one anchor can emit several
   detections under different classes**. A natural "top-300 anchors, take
   each one's argmax class" port is wrong in a way no shape check catches.
   The photo oracle fixture was deliberately switched to the corpus's
   busiest image (23 objects, 6 classes) so this path is actually
   exercised; the formula fixture produces **zero** confident detections
   and cannot fail it.

**The conversion is verified, not asserted** (`tools/diana_convert.py` →
`tools/diana_verify_convert.py`). 708 → **204 tensors**: 120 one2many
dropped, 96 conv+bn folds (exact in eval mode — the manifest records
`fused: true` per §7 step 6), 12 passthrough. A complete YOLO26n forward
written only against the safetensors and `torch.nn.functional` — no
Ultralytics classes anywhere — then reproduces the reference:

| fixture | worst layer (relative) | confident dets | box delta | classes |
|---|---:|---:|---:|---:|
| synth (formula) | 2.4e-05 | 0 | — | — |
| photo (coco-032) | 7.5e-05 | 13 | 2.7e-04 px | **13/13 identical** |

Every one of the 24 layers plus the head tensors, anchors, decode and
final output land within **1e-4 relative**. That result certifies three
things at once: the key map is complete, the BN fold and the one2many drop
are safe, and the architecture *as understood* is the architecture the
checkpoint implements. Everything the Rust port needs is now on disk.

**Fixtures ship, within reason.** The full activation dumps are ~79 MB
across the two fixtures and are gitignored like `mobiledet_stages.npz`.
What is **tracked** is `yolo26n_arch.json` plus
`yolo26n_oracle_digest.json` (681 KB: per-tensor shape and mean/std/min/max
plus 256 exact values at fixed deterministic indices). That is a real gate
on a fresh clone — a wiring, ordering or channel error moves all 256 — and
the Rust oracle upgrades to full-tensor max-abs when the npz files are
present. Both halves of the Carmenta fixture lesson, honoured.

**Landed in `ffai-core`** (§2.1's deliverable): `Detection` (xyxy in
original pixels, class id, confidence, optional track id), `DetectOutput`
(confidence-ordered, carrying its `Letterbox`), `Letterbox` with the
inverse transform travelling *with* the output rather than living in the
caller's head, `DetectOptions`, and the `DetectEngine` trait + registry
slot. Class names live on the engine, from the manifest, not on every
output.

**The crate exists, and the backbone is oracle-clean.** `ffai-diana` ships
`blocks` (ConvAct, Bottleneck, C3k, C3k2, Attention, PSABlock, SPPF,
C2PSA), `backbone` (layers 0–10), and `config` — the fail-closed manifest
reader, which refuses an unknown conversion-map version, a DFL head, a
`one2many` inference branch, or unfused weights rather than loading
whatever fits. Against the tracked digest, on the formula fixture:

| layer | shape | max delta | | layer | shape | max delta |
|---|---|---:|---|---|---|---:|
| 00 Conv | 1×16×320×320 | 3.8e-06 | | 06 C3k2 | 1×128×40×40 | 1.9e-06 |
| 01 Conv | 1×32×160×160 | **2.3e-05** | | 07 Conv | 1×256×20×20 | 5.6e-06 |
| 02 C3k2 | 1×64×160×160 | 6.7e-06 | | 08 C3k2 | 1×256×20×20 | 2.3e-06 |
| 03 Conv | 1×64×80×80 | 6.0e-06 | | 09 SPPF | 1×256×20×20 | 4.8e-06 |
| 04 C3k2 | 1×128×80×80 | 3.0e-06 | | 10 C2PSA | 1×256×20×20 | 2.8e-06 |
| 05 Conv | 1×128×40×40 | 5.0e-06 | | | | |

**Fail-closed caught a real bug on the first run**, which is the argument
for the policy in one line. `C3k`'s inner bottlenecks are constructed with
`e = 1.0` while `C3k2`'s own are `e = 0.5`; taking the reference's
*default* built `model.6.m.0.m.0.cv1` as 32→16 where the checkpoint has
32→32. A permissive loader would have zero-filled or silently reshaped;
the strict load named the tensor and the expected shape. The expansion is
now a caller's argument (`Bottleneck::half` / `::full`), not a default.

Two smaller decisions worth recording, both places where "close enough"
was available and rejected:

- **SPPF's max-pool uses replication padding, and that is exact rather
  than approximate.** candle has no padded max-pool; PyTorch maxes over
  the valid window only, and every replicated value is a copy of an edge
  element already inside that window, so the maximum is unchanged. Zero
  padding would *not* be equivalent — these activations are signed, and a
  zero wins wherever the true window maximum is negative.
- **Channel widths are tabulated from the probe, not derived** from the
  YAML `scales` row. The derivation is easy to write and easy to get
  subtly wrong, and a wrong `C3k2` hidden width (0.25 vs 0.5 expansion)
  changes results *without* changing any output shape — the one class of
  error the strict load cannot catch.

**The whole graph now reproduces the reference.** Neck (layers 11–22), the
one2one head and the decode landed on the same discipline; every tensor in
the 24-layer graph plus the head branches, the decode and the final
selection are inside the oracle band, worst **relative** delta 1.8e-06:

| stage | tensors checked | worst relative |
|---|---|---:|
| backbone 00–10 | 11 layers | 1.8e-06 |
| neck 11–22 | 12 layers | 5.9e-07 |
| head (one2one) | `head_boxes`, `head_scores` | 1.8e-06 |
| decode | `decoded` `[1,84,8400]` | 2.5e-07 |
| two-stage top-k | `final` `[1,300,6]` | 1.2e-06, **classes exact** |

**The top-k check is the one that earns its place.** Taking the top-k
anchors and each one's argmax class yields a different 300-row result with
*identical dimensions* — no shape check can see it. The oracle rebuilds
the reference's `[300, 6]` layout row for row and compares at the digest's
sampled indices, so it tests the selection ORDER, with class ids required
to match exactly rather than approximately. It reproduces the reference on
the formula fixture, where all 300 rows are near-tied junk detections —
about the most order-sensitive input available.

**A tolerance bug in the test, found and fixed rather than tuned around.**
`decoded` initially failed at 1.98e-04 against a flat 1e-04 absolute
bound. The port was correct: `decoded` carries box coordinates in *pixels*
(up to ~640) while every other tensor holds activations of order 1–10, so
one absolute bound was silently a 1.5e-07 relative demand there — tighter
than f32 reassociation can meet. The gate is now `relative band OR
absolute floor`, the same shape `ffai-bench`'s quality gate settled on for
the same reason, and it reports relative deltas so the units stop
mattering. Loosening the number without diagnosing it would have hidden
whatever the next real error turns out to be.

**The engine runs.** `ffai-diana` composes letterbox → backbone → neck →
head → decode behind `DetectEngine`, registers as `yolo26n`, and ships
`ffai detect` on the CLI (`--conf`, `--iou`, `--max-det`, `--classes`,
`.jsonl` out). On the corpus's busiest image it returns 13 detections —
people, cars, a backpack — deterministic across calls, 275 ms warm.

Three decisions inside the engine that are recorded rather than defaulted:

- **Square letterbox only.** The rectangular geometry is not implemented,
  because M-D0 measured the two disagreeing by 1.5–1.8 pp mAP in opposite
  directions by tier (§8.1). Which one this engine implements is a pinned
  decision. The pad arithmetic reproduces Ultralytics'
  `round((size - n) / 2 - 0.1)` half-pixel bias rather than rounding the
  obvious way — a one-pixel offset shifts every box by one pixel, small
  enough to pass a smoke test and large enough to cost mAP.
- **NMS is off by default.** The one2one head is NMS-free by construction,
  so `DetectOptions::iou` is `None` unless asked for; running suppression
  anyway would silently drop legitimately overlapping objects.
- **The bench pins confidence to 0.001, not the engine's 0.25 default.**
  mAP needs the low-confidence tail and the reference adapters run at
  `--conf 0.001 --max-dets 100`. Scoring our engine at 0.25 against
  references at 0.001 would report a recall collapse that is purely a
  configuration difference — the M0 unpinned-decode defect in a detection
  costume.

**A blocker worth recording, because the fix improved the corpus.**
`ffai_media::load_image` is PNG-only (the rff image decoders are ROADMAP
Phase 3) and the corpus was COCO's JPEG. Rather than add a decoder, the
corpus moved to **v2 with losslessly re-encoded PNG** — which turns out to
be the better answer regardless: as JPEG, our decoder and Ultralytics' and
ORT's would each produce slightly different arrays, so any mAP delta would
have been a decoder comparison wearing a detector's clothes. The reference
rows re-measured identical to their v1 values, which is what proves the
re-encode lossless.

**Also recorded: the bench ran outside the CLI.** `ffai-cli` links every
component crate, and `ffai-mercury` was mid-edit in this shared worktree
(another session's diarization signature change), so a CLI build was
hostage to code unrelated to this measurement.
`ffai-diana/examples/bench_detect.rs` links only `ffai-bench` +
`ffai-diana` and writes the same ledger line through the same
`runner::run_detect` — the same escape hatch `ffai-carmenta`'s
`live_bench.rs` uses, and the reason that pattern exists.

### 8.3 M-D1 result — parity reached, speed open (CLOSED 2026-07-30)

Ledger `bench-detect-1785475484`, corpus `diana-coco` v2 (`fef55735a225`),
45 holdout images, best-of-3, CPU. All six reference configurations and the
engine processed every image.

| Implementation | Stack | mAP50 | mAP50-95 | img/s warm | p50/p95 | steady MiB |
|---|---|---:|---:|---:|---|---:|
| **yolo26n (ours)** | **pure Rust / candle** | **68.65** | **52.59** | 4.19 | 282 / 304 ms | **112** |
| ultralytics-yolo26n | Python/PyTorch | 68.65 | 52.59 | 17.08 | 58 / 73 ms | 297 |
| ort-yolo26n | ORT (C++) | 68.65 | 52.59 | **31.39** | **31 / 37 ms** | 161 |

**Quality: PASS, at exact parity — three independent implementations agree
to 5.6e-05.** At full precision: ours **0.686518**, PyTorch 0.686499, ORT
0.686462. Read that as *identical*, not as a lead — the spread is
resize-and-decode noise between three different preprocessing paths, and
nothing in it favours anyone. It is the same evidence Mercury took from
whisper.cpp matching its WER to the digit, and it is the strongest
correctness statement this component can make: a from-scratch candle
reimplementation, official weights, same answer.

**Footprint: PASS — 112 MiB steady against ORT's 161 MiB (0.70x)**, and
37 MiB of ours is pre-decoded images the harness holds for the comparison,
so the engine itself sits near 75 MiB.

**Speed: FAIL, honestly, by 7.5x** — 4.19 img/s warm against ORT's 31.39,
p50 282 ms against 31 ms. That is the expected shape of a bring-up: this
is unoptimized f32 candle with no profiling done, exactly where Mercury's
M1 sat at 4x behind and Carmenta's M-C1 sat before its speed campaign.
The milestone's own exit criterion anticipated it ("speed may fail
honestly at bring-up — it did for Mercury and Carmenta"), and M-D2 is the
campaign that answers it. **No speed lever has been pulled yet, so no
speed lever has been ruled out** — the profiler runs first, per §8.4.

**Corpus v2 validated itself.** Every reference row re-measured *identical*
to its v1 (JPEG) value — 68.65/52.59 and 78.06/63.12 across both tiers —
which is the check that the PNG re-encode was genuinely lossless rather
than assumed to be.

Exit gate: per-layer activation parity within recorded tolerances ✅
(§8.2, worst 1.8e-06 relative) · box/score parity on holdout ✅ (exact to
5.6e-05 mAP) · conversion pipeline + strict load ✅ · `DetectOutput` /
`DetectEngine` in `ffai-core` ✅ · all four gates run and recorded ✅
(3 pass, speed fails honestly). **M-D1 CLOSED.**

**What M-D1 leaves for M-D2:** a named target rather than a vague one —
**31.39 img/s warm / 31 ms p50** for the n tier, from a starting point of
4.19 / 282 ms, with parity held at every step. The parity oracles are the
standing gate that makes a speed campaign safe: any brick that moves a
layer outside 1e-4 relative is a wrong brick, whatever the stopwatch says.

### 8.4 M-D2 campaign plan — the speed gap, instrument first

M-D1 closed with quality at parity and speed **7.5× behind ORT** (4.19 vs
31.39 img/s warm; 282 vs 31 ms p50). This section is the plan for closing it,
written to the `codec-optimize` discipline: profile before touching anything,
byte-identical-first technique order, revert anything that isn't faster, one
brick per commit. The parity oracles of §8.2 are the standing gate — **any
brick that moves a layer outside 1e-4 relative is a wrong brick, whatever the
stopwatch says.**

**Depth 6 was run before any profiling** (`examples/depth6.rs`), per the
six-whys rule that instrument checks cost minutes and invalidate days. Three
results, all of which change the plan:

| probe | result | consequence |
|---|---|---|
| noise floor, unpinned | spread **1.14×–1.82×** across sessions | **no A/B is trustworthy yet.** At 1.82× the resolution is 74% — wider than the entire prize of most bricks |
| thread scaling | 1 thread **415 ms** → 24 threads **293 ms** = **1.42×** | the bottleneck is **not parallelizable compute** |
| letterbox share | **6.1 ms = 2.4%** of a 259 ms detect | preprocessing is **ruled out**; ~97% is the network forward + decode |

**The thread-scaling number is the most important thing measured so far, and
it disqualifies the obvious plan.** Twenty-four cores buying 1.42× is the
signature Mercury hit on its encoder — *"a kernel that does not scale with
cores is not compute-bound, and hand-vectorizing a compute-bound kernel that
is not compute-bound is the classic wasted week."* So M-D2 does **not** open
with SIMD, and the reflex to hand-write convolution kernels is explicitly
deferred until a profile earns it.

**The leading hypothesis, stated as a hypothesis.** The six-whys law "look for
hot loops that are secretly matmuls" applies directly: a convolution is a GEMM
after im2col, and a tuned GEMM beats a direct loop by orders of magnitude. ORT
runs im2col/Winograd into tuned kernels. Whether candle's CPU `conv2d` takes a
GEMM path *at these shapes* — many small channel counts, 3×3 and 1×1, 320×320
down to 20×20 — is unknown and is the first thing depth 3 must answer. It is
plausible, unproven, and exactly the kind of clean depth-3 story the skill
warns is "the most dangerous artifact in the process."

**M-D2.0 — the instruments, before any optimization brick.**

1. **A pinned / CPU-time harness.** `ProcessorAffinity` + `High` priority, and
   prefer `TotalProcessorTime` over elapsed wall, which measured 5× tighter
   under foreign load. Report the noise floor in every run. Nothing is
   measured until this exists — the current 74% resolution makes every A/B a
   coin flip wearing a table.
2. **A stage profiler** (`FFAI_PROFILE`-gated, zero-cost when off), bucketing
   preprocess / backbone / neck / head / decode, with **call counts**, and the
   residue decomposed until every line is named. Ranked by **absolute ms** —
   the worst ratio is rarely the best target.
3. **A deterministic counter** — convolution calls, MACs, bytes moved. Immune
   to every timing artifact, and the cheapest instrument that can rank work.
4. **A roofline measured at OUR shapes**, not a 2048³ square matmul. A ceiling
   from the wrong shape was wrong by 2.5× in one campaign and inverted its
   narrative.

**M-D2.1 — descend, then route.** Classify each fat stage as compute-bound
kernel / memory-bound glue / redundant work / framework overhead, then route
to the technique skill the classification names — not the one that sounds
exciting. The byte-identical order is not negotiable: eliminate-redundancy →
memory-copies → cache-tiles (only if a size sweep says cache-bound) →
vectorize → asm.

**Arithmetic prunes to run BEFORE building anything** (`expected pipeline gain
= stage share × speedup`): a brick on a 10% stage cannot return more than
1.11× overall, which is inside today's noise floor. Any candidate whose
ceiling lands under the measured floor is skipped on paper, not in code.

**Levers already visible, ranked by expected value, all unproven:**

| candidate | why it might pay | why it might not |
|---|---|---|
| conv path (im2col+GEMM vs direct) | the "secretly a matmul" law; would explain a 7.5× gap by itself | candle may already do this; then the gap is elsewhere entirely |
| f16 / int8 weights | 2–4× traffic cut; Mercury's int8 GEMV won 4.34× | Mercury also measured int8 on the *encoder* as a 2.4× REGRESSION — quantization pays by kernel shape, not by faith |
| the 8400×80 decode scan | pure scalar Rust over 672k floats | measured at ~0 so far; likely <1%, prune on arithmetic first |
| batching / fused conv+SiLU | fewer framework round-trips | candle's op dispatch may already be thin |

**The gate that must not move.** Speed work here is byte-identical work: the
digest oracle runs on every brick, and a brick that changes a layer is
reverted regardless of its stopwatch. The one exception is a precision change
(f16/int8), which is *not* byte-identical by construction — that one gets the
corpus mAP gate of §8.3 instead, and must hold 68.65 within the parity band.

#### M-D2 progress — the profile, and the first brick

**The profile** (`examples/profile_detect.rs`, `FFAI_PROFILE=1`, 10 runs,
pinned). Residue **0.6%** and timer overhead negligible, so nothing is
hidden and the ranking is trustworthy:

| stage | share | ms/image |
|---|---:|---:|
| backbone | 42.4% | 112.2 |
| head | 27.6% | 73.1 |
| neck | 25.4% | 67.1 |
| pre | 2.3% | 6.2 |
| decode | 1.7% | 4.5 |
| residue | 0.6% | 1.5 |

The info tier answers it outright: **convolution is 84.6% of detect** across
96 calls/image; attention is 4.9%. Preprocessing and decode together are 4% —
both are pruned as targets on arithmetic alone.

**Depth 4: candle's conv is 8.2× off the GEMM ceiling at our shapes**
(`examples/conv_roofline.rs`) — 1–39 GFLOP/s against 94–499 for the
equivalent matmul, measured per shape rather than borrowed from a big square
matmul. But the "secretly a matmul" hypothesis is **partly refuted**: candle
already defaults to `TiledIm2Col`, so it *is* GEMM-backed. The gap is
elsewhere, and naming it is open work.

**Depth 5 found one mechanism outright.** candle has **no grouped-convolution
kernel**: `Tensor::conv2d` with `groups > 1` does `chunk(groups, 1)` → one
`conv2d_single_group` per group → `cat`. A 256-channel depthwise therefore
runs 256 separate single-channel convolutions. Measured, it ran at
0.44–1.04 GFLOP/s — **72.9× slower per FLOP than the dense convolution beside
it**, which does 64× more arithmetic in less time.

**The brick: a depthwise 3×3 kernel** (`src/dwconv.rs`) — one pass, parallel
over channels, no im2col (a 9-tap stencil does not want a 9×-larger matrix).
Gated against candle's grouped path as oracle across six shapes, and the
full-graph parity oracle still passes at **1.714e-06**.

**Its verdict, stated in both states because they differ:**

| arms | head ms/image | detect median | process CPU |
|---|---|---|---|
| 24 threads | 71.19 vs 70.87 | 252.5 vs 252.2 | **12,234 vs 13,688 (−10.6%)** |
| 1 thread | **162.5 vs 167.4** | **553.6 vs 562.7** | 4,797 vs 4,750 |

Single-threaded it is a real if small win — 2.9% on the head stage, 1.6%
end-to-end, 5/6 paired. **At 24 threads it is flat on wall while cutting CPU
work 10.6%.** KEPT on that basis — strictly less work, provably correct,
behind `FFAI_DIANA_NO_DWCONV` — but recorded as **not a latency win at
thread count**, not as a speedup.

**★ The methodology error, which is worth more than the brick.** The
standalone probe priced these six convolutions at **34.2 ms/image (12.9% of
detect)** and projected 1.14× pipeline. In context they are worth ~5 ms. The
isolated measurement over-stated the cost **~7×** — exactly the
"isolation misleads in BOTH directions" law, and it would have justified far
more work than the brick deserved. **Every remaining M-D2 prize must be
priced in-context** (by A/B against the stage median with the arm behind a
knob), never by a standalone microbenchmark.

**Bricks 2 and 3 — route the convolutions through the GEMM candle already
ships.** `examples/conv_scaling.rs` measured conv against `Tensor::matmul` on
identical arithmetic at both 1 and 24 threads. candle's conv is **3.6–20×
slower than its own matmul**, at both thread counts — so it is not a
parallelism problem, and the "secretly a matmul" law applies even though
candle nominally im2col's already.

- **1×1 → matmul** (`src/blocks.rs::pointwise_matmul`). A 1×1 convolution
  needs no im2col at all, only a reshape: `W (Co,Ci) @ x (Ci,HW)`. It moves
  strictly less data than any convolution path can. **10/10 paired,
  z = 3.16**, detect 234.6 → 202.6 ms.
- **3×3 stride-1 → explicit im2col + matmul** (`src/conv3x3.rs`), written
  **channel-major** `(Ci·9, HW)` so each im2col row is a contiguous
  `copy_from_slice` with a horizontal offset rather than a strided gather —
  the layout choice is the whole performance argument. **20/22 paired,
  z = 3.84**, detect 261.7 → 242.8 ms.

**★ Brick 4, and the biggest single lever, was not a convolution at all.**
Decomposing the `conv` bucket left 26.4% unaccounted, and the only thing
left inside that scope was the activation. **SiLU was 25.8% of detect**
across 870 calls — Mercury's GELU result reproduced on a different model:
an elementwise function nobody would nominate, costing as much as the
arithmetic it decorates. Two mechanisms, both removable: `x·sigmoid(x)` is
two tensor passes and two allocations, and `exp` is a scalar libm call that
blocks vectorization of the whole loop. `src/silu.rs` fuses it into one pass
with `exp(x) = 2^(x·log2e)` — integer part written into the f32 exponent
field, fraction by a degree-5 polynomial — so no libm call and no branches.
**22/22 paired, z = 4.69**, detect 209.7 → 192.3 ms.

Every brick is gated by the §8.2 parity oracle, which still passes
(1.71e-06 → 3.24e-06 as float accumulation reassociates, well inside the
1e-5 band, classes exact and top-k order reproduced), and each sits behind
its own `FFAI_DIANA_NO_*` knob so the A/B arm and the shipped fallback are
the same switch.

**★ A methodology confirmation worth keeping.** The 3×3 brick read
**7/10 paired, z = 1.26 — inconclusive** — and then **20/22, z = 3.84** on
the *same change* at larger N. The six-whys rule ("run the paired estimator
at N ≥ 20 before judging it") is not a formality; at N = 10 this brick would
have been wrongly pruned.

**Where M-D2 stands** (ledger `bench-detect-1785511208`, same corpus and
harness as the M-D1 line):

| | M-D1 | now | |
|---|---:|---:|---|
| mAP50 | 0.686518 | **0.686518** | bit-identical — quality PASS held through four kernels |
| img/s warm | 4.19 | **5.33** | 1.27× |
| p50 latency | 282 ms | **188 ms** | 1.50× |
| steady MiB | 112 | **110** | footprint PASS, 0.69× ORT |
| gap to ORT | 7.5× | **4.7×** | speed still FAIL, honestly |
| steady MiB | 112 | **94** | 0.58× ORT |

The paired A/B ratios compound to **1.43×** (1.158 × 1.078 × 1.090 × 1.047).
**The ledger lines cannot confirm that, and should not be asked to** — in the
final run ORT itself fell from 28.50 to 23.72 img/s and Ultralytics from
14.46 to 12.71, so every absolute number on that board moved together. This
is Mercury's rule exactly: *never headline a ratio whose denominator drifts
more than your improvement.* **The paired, same-binary, ABBA-interleaved
numbers are the evidence; the ledger line is the standing** — which is why
each brick was measured behind its own knob rather than by differencing
ledger lines.

**★ The finding that redirects the campaign.** Two independent measurements
now say the wall is **not gated by CPU throughput**: 24 cores buy only
**1.42×**, and removing 10.6% of CPU work moved the wall **0%**. That is
why the four bricks that followed were all *structural* — routing work to a
better-shaped primitive — and none was a SIMD kernel.

**The board after four bricks** (`FFAI_PROFILE=1`, all knobs on). The
`conv` bucket now decomposes fully, which is the point of decomposing it:

| conv sub-bucket | calls/image | share |
|---|---:|---:|
| SiLU | 87 | 25.8% → **ours** |
| 3×3 **stride-2** | 7 | **18.0%** — still candle's path |
| 3×3 stride-1 | 32 | 16.8% → ours |
| 1×1 | 49 | 13.8% → ours |
| depthwise | 8 | 1.9% → ours |

**Brick 5 — stride-2 downsamples** (`conv3x3_strided`). The last convolution
family on candle's path: 7 calls/image at 6.9 ms. Extending the kernel costs
the contiguous-copy property — at stride 2 the horizontal tap becomes a
constant-stride gather — so it was a real experiment, not a transcription.
**22/22 paired, z = 4.69**, detect 202.1 → 193.0 ms.

Its unit test earned its place immediately: the first implementation was
**wrong on `640x640`, the shape the stem actually runs** (max rel 3.6e-01),
from an off-by-one in the in-bounds column bound at `kx=2`. The oracle named
it before it could reach a measurement, and the odd-size cases in that test
(`7x9`, `5x5`, `1x8`) exist precisely because the range arithmetic differs
there.

**★ Where the remaining time actually is — and it is not a convolution.**
With every family routed, our own 3×3 kernels are 36.0% of detect. Splitting
them:

| inside our 3x3 kernels | share of detect |
|---|---:|
| marshalling (`to_vec1`, zeroed alloc, `from_vec`, bias) | **~14.4%** |
| im2col materialization | 12.5% |
| the GEMM itself | 9.1% |

**The framework glue is larger than the im2col, and both dwarf the
arithmetic.** Every convolution copies its input tensor out to a `Vec`,
allocates and *zero-fills* a multi-megabyte im2col buffer, and copies the
result back into a tensor — so each intermediate round-trips
Tensor → Vec → Tensor between consecutive layers. This is the
"delivery is not a detail" law: marshalling that costs more than the kernel
it delivers.

**Next brick, named by that table:** take candle's zero-copy storage hook
(`CustomOp1`, as Mercury did — it recovered 90 → 78 ms with no change to the
arithmetic) so the input is read in place and the output written once, and
stop zero-filling a buffer whose interior is immediately overwritten. That
targets ~14.4% with no numerical change at all, which makes it a
byte-identical brick rather than a tolerance one.

Then, in order: im2col traffic (12.5%), attention (5.6%), the scalar
bilinear letterbox (3.9%). The GEMM at 9.1% is the arithmetic and is the
floor — nothing to win there.

#### Parity with the PyTorch reference — what it can mean, and what we hold

**Bit-identical output to PyTorch is not an achievable target, and saying so
is part of the result.** Float addition is not associative; every GEMM
reorders accumulation; PyTorch's own oneDNN kernels change with shape,
thread count and ISA, so PyTorch does not reproduce *its own* bits across
`OMP_NUM_THREADS`. Bit-exactness would require replicating the reference's
exact scalar op order, non-FMA and GEMM-free — roughly a 20× speed cost and
a revert of every M-D2 brick. Recorded so the question is settled rather
than re-asked.

Two parity claims *are* meaningful, and both now hold as standing gates.

**1. Detection parity vs PyTorch.** `examples/detect_parity.rs` compares
detection-to-detection against the Ultralytics dump over all 45 holdout
images — because **mAP is an aggregate and could hide compensating errors**
(a box drifting one way on one image and the other way elsewhere nets out;
"a metric can be blind to the change you are actively making").

| conf ≥ | detections | count | classes | max box delta |
|---:|---:|---|---|---:|
| 0.25 | 133 | EXACT every image | 133/133 | 0.1443 px |
| 0.10 | 242 | EXACT every image | 242/242 | 0.5611 px |
| 0.05 | 361 | EXACT every image | 361/361 | 0.5612 px |
| 0.01 | 1161 | EXACT every image | 1161/1161 | **11.83 px** |

The report separates two claims deliberately. **Structural parity — count,
class and order — is EXACT at every threshold down to 0.01, across 1161
detections.** That is discrete, has no tolerance to hide behind, and any
mismatch would be a real behavioural divergence. **Geometric parity** is a
float tolerance: sub-pixel (≤0.56 px) down to conf 0.05, and it **fails at
conf 0.01, reported rather than omitted**. The mechanism is understood and
benign — the box head regresses `(l,t,r,b)` distances that are multiplied
by the level stride (up to 32×), so in the junk tail, where a 1%-confidence
box has no meaningful geometry anyway, a small float difference in the
regression becomes a large coordinate difference. It is confined to
detections no caller would use, and the structural agreement is unaffected
there.

**2. Self-determinism — this one IS byte-exact.**
`tests/determinism.rs`: byte-identical detections across 6 consecutive
calls, and the digest is **identical at 1, 4 and 24 threads**
(`0ca4b3b8f40a52d8`). PyTorch cannot offer this — its kernels reassociate
per schedule. Three properties make it hold, and the test exists to catch a
future brick that breaks any of them: our kernels partition by rayon over
**disjoint output ranges** (no shared accumulator), the GEMM sees fixed
shapes, and the two-stage top-k sorts by `total_cmp` — a **total order**, so
ties resolve by position rather than by whichever thread finished first. A
parallel reduction into a shared accumulator — the usual way to speed up a
reduction — would break this silently, which is exactly why the gate is
standing rather than a one-off check.

#### Full standing vs Ultralytics — quality and performance (2026-07-31)

Ledger `bench-detect-1785520547`, all seven configurations over the same 45
holdout images, best-of-3. Caveat recorded rather than buried: another
session's `ffai-demo` was resident throughout, so the absolutes are soft;
every arm shares that load, so the within-run ratios are the usable part.

| Implementation | Stack | mAP50 | mAP50-95 | img/s warm | img/s e2e | p50 | steady MiB | load s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **yolo26n (ours)** | **pure Rust** | **68.65** | **52.59** | 6.06 | 5.89 | 168 ms | **95** | **0.21** |
| ultralytics-yolo26n | PyTorch | 68.65 | 52.59 | 15.30 | 8.09 | 67 ms | 310 | 2.13 |
| ort-yolo26n | ORT C++ | 68.65 | 52.59 | **33.24** | **25.14** | **30 ms** | 162 | 0.31 |
| ultralytics-yolo26n-rect | PyTorch | 70.14 | 53.77 | 18.26 | 9.16 | 54 ms | 344 | 1.93 |
| ultralytics-yolo26s | PyTorch | 78.06 | 63.12 | 7.49 | 5.23 | 139 ms | 383 | 2.11 |
| ultralytics-yolo26s-rect | PyTorch | 76.26 | 61.71 | 9.65 | 6.17 | 100 ms | 411 | 2.11 |
| ort-yolo26s | ORT C++ | 78.06 | 63.12 | 17.96 | 14.81 | 55 ms | 271 | 0.41 |

**Against PyTorch at matched weights and matched geometry — the only
comparison that prices the implementation rather than the model:**

- **Quality: identical.** 68.65 / 52.59 on both, and structurally identical
  detection-for-detection (previous section). There is nothing left to win
  or lose here.
- **Steady memory: 3.3× leaner** — 95 MiB against 310.
- **Model load: 10.1× faster** — 0.21 s against 2.13 s, which is the
  Python-runtime tax made measurable rather than asserted.
- **Warm throughput: 2.5× behind** — 6.06 against 15.30 img/s.

**The load/throughput split has a crossover, and it is worth stating
because it changes the deployment answer.** Per image we cost ~0.165 s to
their ~0.065 s, but we start 1.92 s sooner:

```
ours   = 0.21 + 0.165 n        theirs = 2.13 + 0.065 n
equal at n ~= 19 images
```

So **on a single image we finish ~5.9× sooner than the PyTorch path, at ten
images ~1.6× sooner, and we fall behind past roughly twenty.** For a CLI
invocation, a serverless call, or an edge device waking to classify one
frame, we are already the faster answer; for a batch job we are not. That is
a real characterisation, not a consolation — and it is falsifiable from the
two numbers above.

**Against ORT, the deployment bar, we remain 5.5× behind warm** (6.06 vs
33.24) and 1.7× leaner. That gap is the M-D2 campaign's remaining target and
the profile already names where it lives (marshalling 14.4%, im2col 12.5%).

**One comparison deliberately NOT made:** our n-tier engine's e2e
throughput (5.89) exceeds PyTorch's **s**-tier (5.23). That is real but it
prices the *model*, not the implementation — their s tier scores 78.06 mAP50
to our 68.65 — and quoting it would be exactly the error the matched-tier
reference table exists to prevent. Recorded here as context, never as a
claim.

#### Investigating the two speed numbers separately — and they are not one number

`warm throughput 2.5x behind` and `p50 latency 2.5x behind` read as one
finding. Treating them as two questions produced a different answer to each,
and reversed one of them.

**Depth 6 first, and it corrected the gap AGAINST us — twice.**

1. **The spans were not matched.** Our harness pre-decodes images outside
   the timed region; the Ultralytics adapter passes a *path*, and
   Ultralytics decodes inside `predict()`. Measured: **60.6 ms path-in vs
   52.4 ms array-in — 8.2 ms (15.7%) of PNG decode we never paid.** The
   honest per-image gap is **3.20x, not 2.5x**. Reported because it moves
   the number the wrong way.
2. **We were using three times the hardware to lose.** `torch.get_num_threads()`
   is **8**; rayon defaults to **24**. PyTorch reaches its number on a third
   of the machine.

**The two metrics separate cleanly once measured as separate functions:**

| RAYON_NUM_THREADS | p50 latency | sequential img/s | concurrent img/s |
|---:|---:|---:|---:|
| 4 | 167.6 ms | 5.99 | 11.85 |
| 8 | 157.7 ms | 6.34 | 18.12 |
| 24 | 159.6 ms | 6.25 | **22.73** |

**p50 is flat against thread count** — intra-image parallelism is
exhausted, confirming the earlier 1.42x scaling result. **Throughput is
not**: running images concurrently, with no change whatever to the
per-image path, is worth **3.6-5.3x**, and the concurrent output is
asserted byte-identical to the sequential output.

**★ The reversal: PyTorch cannot take that win, and we can.** Measured on
the same machine, same images, pre-decoded arrays both sides:

| PyTorch arm | img/s | vs its own best |
|---|---:|---:|
| sequential, 8 intra-op (its optimum) | **17.29** | — |
| + Python threads x4 / x8 / x24 | 12.30 / 12.50 / 11.77 | 0.71 / 0.72 / 0.68x |
| sequential, 24 intra-op | 10.46 | 0.60x |

Every attempt to spend more of the machine makes PyTorch **slower**. So
17.29 img/s is not its measured value, it is its **ceiling** — and ours is
22.73. **On best-achievable throughput we are 1.31x AHEAD of PyTorch**,
having been "2.5x behind" on the same axis an hour earlier. The mechanism
is structural rather than clever: `DetectEngine` is `Send + Sync`, so N
concurrent detections share **one 95 MiB model in one process**, while
Python's GIL pushes PyTorch to multiprocessing at ~310 MiB per worker.

**Honesty on ORT: it takes the same win** (55.9 -> 81.9 img/s, 1.47x), so
concurrency is not uniquely ours and ORT stays ahead. Its probe also
excludes letterboxing while ours and PyTorch's include it, so only the
PyTorch comparison above is span-matched; a matched-span ORT concurrent
number is **not yet measured** and is not estimated here.

**★ The latency lever, which the reference table had been showing all
along: square inference is DOMINATED at this tier.** We letterbox to
640x640, so a 428x640 photo spends **33% of its compute on grey padding**.
Ultralytics' own default is rectangular, and M-D0's own board says rect is
better on *both* axes at the n tier — **70.14 mAP50 vs 68.65, and 18.26 vs
15.30 img/s**. Measured on our forward:

| input | fwd ms | pixels vs square | speedup |
|---|---:|---:|---:|
| 640x640 (ours today) | 156.4 | 100% | 1.00x |
| 640x448 (a 428x640 image) | 116.1 | 70% | **1.35x** |
| 384x640 (a 640x359 image) | 105.5 | 60% | **1.48x** |

Our kernels are already shape-generic — `conv3x3`, `dwconv` and the
pointwise path all take arbitrary H, W — so **only `image::letterbox`
hard-codes the square.** This is a preprocessing decision, not a port.

The error worth naming: pinning square was correct for the **parity gate**
(M-D0 measured that an unpinned geometry made the .pt and ORT rows disagree
by 1.5-1.8 pp), but pinning the gate and choosing the **default** are two
decisions, and treating them as one cost us ~1.35x latency *and* ~1.5 pp
mAP. The fix is to keep square as the parity configuration and ship rect as
the default, exactly as the reference does.

**Also identified, unmeasured, cheap:** `Head::anchors` rebuilds the full
8400-entry anchor grid on **every call**. It depends only on the feature-map
sizes, which are fixed for a fixed input geometry — the textbook
"does this value depend on the input?" redundancy. Hoisting it is free and
byte-identical.

#### The four follow-up bricks — all landed, all gated

Ordered by the codec discipline: byte-identical first, cheapest first.

**1. Anchor grid — byte-identical, and the fix was not caching.**
`Head::anchors` built a `Vec<(f32,f32,f32)>` of all 8400 grid positions on
every call — a 100 KB allocation whose contents depend only on the
feature-map sizes. The obvious fix is a cache; the better one is **not
building it**: decode needs an anchor for the `max_det` positions it
actually selects (300), not all 8400, and position → `(x, y, stride)` is
three integer ops. `AnchorGrid` is a `Copy` 40-byte descriptor with an
`at(i)`, so the allocation *and* 96% of the arithmetic are gone. Oracle
unchanged at 3.869e-06.

**2. Zero-copy marshalling — 22/22 paired, z = 4.69, 1.136x.**
`crate::cpuop::SliceOp` wraps candle's `CustomOp1`, so kernels read the
activation in place and their output `Vec` *becomes* the result tensor's
storage. Detect median **174.1 → 153.3 ms**, CPU −9.5% — almost exactly the
14.4% the decomposition predicted. Both arms live behind
`FFAI_DIANA_NO_ZEROCOPY`, running identical arithmetic and differing only
in delivery, which is what makes it a clean A/B on a drifting machine. The
helper refuses a non-contiguous layout rather than silently misreading it.

**3. Rectangular inference — 22/22 paired, z = 4.69, 1.324x AND +1.49 pp
mAP50.** Detect median **231.5 → 174.9 ms**, CPU −23%. And the quality half
landed exactly as the reference board predicted: ledger
`bench-detect-1785530596` reads **70.14 mAP50 / 53.77 mAP50-95, identical
to `ultralytics-yolo26n-rect`**, up from 68.65 / 52.59 on square.

Both engines are now registered, because **the geometry is a measurement
configuration, not a knob**: `yolo26n` (rectangular, the default, matching
Ultralytics' own) and `yolo26n-square` (what the parity oracle pins). The
bench derives its comparison key from the engine name rather than
hardcoding it — hardcoding one key while the engine ran the other geometry
would be the M-D0 defect exactly: two rows that look matched and are not.

**4. `detect_batch` — a first-class trait method, not a convenience.**
It is where a detector's throughput lives: **3.6-5.3x** over calling
`detect` in a loop, byte-identical output, no change to the per-image path.
The default implementation is sequential so other engines keep working;
Diana overrides it with a rayon fan-out and forces weight loading *before*
the fan-out so N threads do not race the lazy init. The doc comment states
the structural reason it is ours to take — one `Send + Sync` model serves
every thread, while PyTorch gets **slower** under threading (0.68-0.72x)
and its escape hatch pays a full model copy per worker.

**Compounded, bricks 2 and 3 are 1.50x on latency**, on top of the 1.43x
from the first five.

**A test defect found on the way, worth more than it cost.**
`matches_candle_silu` used unseeded `Tensor::randn`, so at `n = 1` its
relative error divided by a single random draw — it had been passing by
luck and failed once the surrounding code changed. Replaced with a
deterministic sweep. **A test whose verdict depends on the draw is not a
gate**, and this one had been sitting in the suite calling itself one.

#### ★ The measurement upgrade — and the law it exposed

Every A/B above was headlined on **pinned wall clock**, with the noise
apologised for. That was the wrong instrument: pinning a process
**restricts** it to a core, it does not **reserve** one, so a foreign load
still lands there and elapsed wall counts time spent descheduled. The rule
is to change the metric — **CPU time does not accrue off-core** — and
`tools/diana_ab.ps1` now does that, with the `$p.Handle`-before-`WaitForExit`
trap encoded (without it `TotalProcessorTime` reads empty and silently
injects zeros into the median).

**Re-running the bricks on CPU time immediately contradicted one of them**,
and chasing that contradiction produced a law worth more than the brick:

| brick | what it removes | 24-thread CPU | 1-thread CPU |
|---|---|---|---|
| zero-copy marshalling | a **serial** memcpy | **z = −1.28** (no verdict) | **z = +3.41**, ratio 1.113 |
| rectangular inference | **parallel** compute | **z = +4.69**, ratio 1.232 | — |

**CPU time answers "how much total work", not "how long did it take"** — it
sums across threads, so a brick that removes SERIAL work is diluted by
roughly the thread count while one that removes PARALLEL work shows in
full. The zero-copy brick is real (19/22 at one thread); it reads as a
non-result at 24 threads purely because of the summing.

Both readings were needed, and the mechanism predicted the second before it
was measured. The rule now lives in the harness header: **parallel-work
bricks measure on CPU time at any thread count; serial-work bricks measure
at `RAYON_NUM_THREADS=1`, where CPU ≡ wall.** Getting it backwards prunes a
good brick on a confident-looking z-score — the expensive direction of the
refutation asymmetry.

**Caveat on the ledger line, stated rather than buried:** that bench ran
under heavy foreign load — every arm reads roughly half its earlier speed
(ORT 33.24 → 18.07, Ultralytics 15.30 → 7.89). The **mAP figures are
unaffected and are the claim**; the speed evidence remains the paired,
same-binary A/Bs above, not this line's throughput column.

#### Model tiers — n/s/m/l/x from one graph

The port was n-only: every channel width was a hand-tabulated constant read
off the probe. That was correct and **unextendable**, and few people deploy
the nano tier, so it capped who could adopt Diana at all.

**The scaling rule was validated against the checkpoints before a line of
Rust was written**, per the law that cost Carmenta a session. Ultralytics'
`parse_model` computes
`c = make_divisible(min(yaml_c, max_channels) * width, 8)` and
`n = max(round(yaml_n * depth), 1)`; a probe compared that against what
`yolo26n.pt` and `yolo26s.pt` actually contain, layer by layer:
**ALL MATCH**, every width and every repeat count, on both tiers.

`config::Dims` now carries the three numbers per scale and the graph derives
everything from them, so `backbone`/`neck`/`head` build any tier from one
code path. The scales differ in ways a width table would have got wrong:
**m/l/x cap channels at 512 BEFORE scaling** (so they are not simply wider
than s at the deep layers), and **only l/x deepen the blocks** —
`rep(2) = 2` there against 1 for n/s/m.

**★ The second tier immediately exposed a bug the first had hidden.** The
class branch's internal width is `max(ch[0], min(nc, 100))`, and my head
used `nc`. At the n tier those coincide — `max(64, 80) = 80 = nc` — so it
was invisible and every gate passed. At s the correct width is 128 and `nc`
is still 80. **A constant that is right only because two unrelated numbers
collide at one configuration is exactly what a second configuration
finds**, and no amount of testing the n tier harder would have found it.

Verified end to end, same binary, same source:

| tier | strict load | warm ms | detections | top |
|---|---|---:|---:|---|
| n | 204 tensors ✓ | 204.5 | 13 | person 0.891 |
| **s** | **204 tensors ✓** | **340.0** | **19** | **person 0.924** |

The strict load is the real evidence: every one of the 204 shapes had to
match the derived widths, so a wrong derivation anywhere would have failed
closed rather than produced plausible numbers. m/l/x report a missing
manifest — their weights are AGPL and user-converted, so absence is the
expected state, and `examples/tiers.rs` prints the command that produces
them rather than an opaque error.

The bench now derives **both** the tier and the geometry from the engine's
name, so `yolo26s` is judged against the s references and a rectangular
engine against rectangular ones. Hardcoding either while the engine ran
something else is the M-D0 defect in miniature.

**The s tier then passed the quality gate at exact parity in both
geometries** (45-image COCO holdout, ledger `bench-detect-1785540868`,
`-1785542187`):

| engine | mAP50 | mAP50-95 | matched reference | reference mAP50 / 50-95 |
|---|---:|---:|---|---:|
| `yolo26s` (rect) | **76.26** | **61.71** | `ultralytics-yolo26s-rect` | 76.26 / 61.71 |
| `yolo26s-square` | **78.06** | **63.11** | `ultralytics-yolo26s` | 78.06 / 63.12 |

Three of the four figures are identical and the fourth differs by 0.01 pp.
That is the derivation's proof: a wrong width anywhere in a 204-tensor graph
does not produce a 0.01 pp miss, it produces garbage. Footprint held too —
134 MiB rect / 152 MiB square against 356–374 MiB, a 0.38–0.43× ratio, so
the *larger* tier improved on n's memory ratio rather than eroding it. Speed
stays FAIL and neither run's speed column is quotable: the box was loaded
enough that Ultralytics' own load time read 21.4 s against its usual ~2 s.

**★ The geometry sign-flip is now confirmed on two tiers, in opposite
directions.**

| tier | rect mAP50 | square mAP50 | winner |
|---|---:|---:|---|
| n | **70.14** | 68.65 | rect, by 1.49 pp |
| s | 76.26 | **78.06** | square, by 1.80 pp |

M-D0 saw this in the *references* and used it to justify pinning geometry;
Diana reproduces both signs itself, which upgrades it from an artifact of
their harness to a property of the models. This is the codec
`content-adaptive-dispatch` shape — **the sign-flip IS the dispatch
trigger** — and it looked like the default geometry should be a per-tier
decision.

**It was deliberately not wired, and 450 images say that was right.**

#### ★ The sign-flip survives and the EFFECT does not — settled on v3

Two tiers is two data points, and a 1.5-1.8 pp effect measured on **45
images** is not distinguishable from sampling noise. The whole reason
`diana-coco-v3` exists is that the proposed follow-up — convert m/l/x to get
more tiers — would have produced four noisy points instead of two and
answered nothing. Re-measured on **450 holdout images**, all five tiers,
both geometries (ledger `bench-detect-1785550365`):

| tier | rect mAP50 | square mAP50 | delta | the same delta on 45 images |
|---|---:|---:|---:|---:|
| n | **61.36** | 61.01 | **+0.35** rect | +1.49 rect |
| s | 68.16 | **68.84** | -0.68 square | -1.80 square |
| m | 73.07 | **73.23** | -0.16 square | not measured |
| l | 74.02 | **74.16** | -0.14 square | not measured |
| x | 77.38 | **77.67** | -0.29 square | not measured |

The **sign** flip is real and holds — n prefers rectangular, every larger
tier prefers square. The **magnitude** collapses by roughly 4x, to
0.14-0.68 pp, which is at or under the harness's own quality band (+0.25 pp
absolute or +5 % relative). n's preference, the one that motivated making
rectangular the default, is **0.35 pp**.

**DECISION: no tier-dependent geometry dispatch.** A dispatch that moves
0.14-0.68 pp is machinery whose benefit sits inside the band its own gate
uses to decide whether two numbers differ. Geometry stays a **required,
named argument** — that part was never about the size of the effect, it is
about not comparing a rectangular engine against a square baseline, which
is the M-D0 defect and costs 1.5 pp of pure confusion regardless of which
letterbox is better.

The four-fold shrinkage is the finding worth keeping: it is a measured
instance of the thing that makes small corpora dangerous, in a repo whose
own rule is "no claim without a number". The number was there at 45 images.
It was the wrong number.

#### All five tiers — and the third branch neither n nor s could reach

Converting m/l/x closed the tier work and immediately paid for itself. The
tensor counts were the tell before any test ran: n and s emit 204, **m emits
224 and l emits 332.** l's extra was predicted — `depth = 1.00` makes every
`rep(2)` resolve to 2. **m's twenty were not.** m has n's depth.

A module-tree diff against the checkpoint found exactly two differences:

```text
model.2.m.0   Bottleneck -> C3k
model.4.m.0   Bottleneck -> C3k
```

Those are the only two `C3k2` layers whose YAML argument is `False`, and
`parse_model` overrides it:

```python
if m is C3k2:
    if scale in "mlx":
        args[3] = True
```

**The YAML flag is not the answer; the YAML flag OR the scale is.** This is
the third tier-dependent branch, after the 512-channel cap applied before
scaling and the depth-scaled repeat count — and n and s together exercise
**none of the three**. The board was fully green on both tiers, the oracle
was at 3.869e-06, and the rule was still wrong. It is the same lesson the s
tier taught with the head's class-branch width, one level up: a constant
that is right only because the configurations you have tested happen to
agree is found by configuration N+1, never by testing configuration 1
harder.

Fixed in `Dims::c3k`, and both halves of the graph now resolve the flag
through one function so the backbone and neck cannot drift apart.

| tier | tensors | params | strict load | warm ms | dets |
|---|---:|---:|---|---:|---:|
| n | 204 | 2.4 M | ✓ | 257.9 | 13 |
| s | 204 | 9.5 M | ✓ | 282.9 | 19 |
| **m** | **224** | **20.4 M** | ✓ | 573.4 | 20 |
| **l** | **332** | **24.8 M** | ✓ | 769.6 | 22 |
| **x** | **332** | **55.7 M** | ✓ | — | — |

#### ★ The final-selection gate was measuring luck

The l tier failed the oracle at one assertion — the `[300, 6]` two-stage
top-k — while every layer through `decoded` matched at 4.3e-06 relative.
The obvious read is a decode bug. The obvious read was wrong, and the way
that was established matters more than the fix.

**Refutation by simulation on the reference's own data.** Take the
reference's `decoded` tensor, perturb it by exactly the divergence we
measure against it (1.1e-4 in logit space, 4.8e-5 on box coordinates),
re-run *its own* selection, and compare to *its own* unperturbed result:

| tier | fixture | max delta in `final` | class flips |
|---|---|---:|---:|
| n | synth | **581.4** | **6** |
| n | photo | 140.0 | 2 |
| m | synth | 4.8e-05 | 0 |
| m | photo | 93.5 | 4 |
| l | synth | 556.8 | 10 |
| l | photo | 4.8e-05 | 0 |

The reference cannot reproduce its own ordering under noise it is entitled
to have — including on **n/synth, the exact configuration that was
passing**. n passed by luck. The test would have flipped to failing on any
rebuild that changed accumulation order, and it would have been read as a
regression.

The cause is structural: the two-stage top-k ranks 8400x80 candidates down
to 300, and on an input with no detections the rows near the cut are
separated by 2.6e-08 to 8e-07 — below f32 reassociation noise. Their order
is genuinely undefined. **The synth fixture has zero rows above confidence
0.05 on every tier**, so it never had a detection to order in the first
place.

Two measurements said what a valid gate looks like. The **confidence
sequence** is tie-robust — swapping two near-equal rows leaves the sorted
scores unchanged — and moves only 7.5e-07 to 2.7e-05 under the same
perturbation. **Rows above 0.05** are fully determined: box delta 4.8e-05,
zero class flips, every tier and both fixtures. So the gate now asserts the
confidence sequence for all 300 rows and positional box/class identity only
above the floor, and when zero rows qualify it **says so** instead of
reporting a pass.

That left detection order untested, since synth qualifies nowhere. The
photo fixture has 35–57 determinate rows, but its input is a corpus JPEG
through PIL's BILINEAR, which our letterbox does not reproduce bit-for-bit
— and should not, since that would be a resampler comparison wearing a
detector's clothes. The letterboxed **canvas** is uint8, so it round-trips
through PNG exactly (verified: max delta 0.0) and now ships tracked at
640 KB. The order gate runs on a fresh clone.

All five tiers, green, with the order gate live:

| tier | worst relative delta | determinate rows | box max | classes |
|---|---:|---:|---:|---|
| n | 3.869e-06 | 35 | 2.7e-04 | exact |
| s | 3.809e-06 | 45 | 5.8e-04 | exact |
| m | 8.703e-06 | 57 | 6.1e-05 | exact |
| l | 4.327e-06 | 53 | 6.1e-05 | exact |
| x | 6.076e-06 | 44 | 1.7e-04 | exact |

The s tier had never been oracled at all — the digest was n-only, so `s`
had shipped on a strict load and a corpus mAP without a single per-layer
check. It has one now.

#### Two enumerations that expired

Both were correct when written and silently stopped covering anything the
moment a second instance existed. Recorded together because the shape is
the same and it is not a coding error, it is a class of them.

**The npz ignore rule.** `.gitignore` listed the oracle dumps by literal
filename, `yolo26n_oracle_synth.npz` and `_photo.npz`. Dumping m/l/x
produced ~700 MB in the same directory, none of it ignored, in a tracked
path. Now a glob: `corpora/refs/fixtures/yolo26*_oracle_*.npz`.

**The corpus source cache.** `tools/diana_coco_corpus.py` cached each
downloaded JPEG as `coco-NNN.src.jpg` — but `NNN` is a position in a
stride walk whose stride is `len(eligible) // COUNT`. Raising COUNT
remaps every index to a different COCO image while the cached files stay
put, so the rebuilt corpus would pair image N's pixels with image M's
ground truth. Nothing errors; mAP is just quietly wrong. The cache is now
keyed by COCO's own file name in `corpora/cache/coco-val2017/`, shared
across corpus versions, and written through a `.part` temp so an
interrupted download cannot be mistaken for a complete one.

#### diana-coco v3 — 600 clips, because 45 could not answer the question

The geometry sign-flip was measured on 45 images. A 1.5–1.8 pp difference
on 45 images is not obviously distinguishable from sampling noise, and the
proposed follow-up — converting more tiers to get more points — would have
produced four noisy points instead of two rather than answering anything.
v3 is 600 clips / 450 holdout, built by the same audited rules (license
filter, crowd exclusion, deterministic stride, PNG storage), under its own
manifest and its own clips directory so v2 and every ledger row naming its
hash stay reproducible.

#### ★ DECISION — what Diana may claim while the speed gate fails

The question was raised as a launch blocker and it deserves an answer in the
tree rather than in a conversation: Mercury did not claim parity until all
four gates passed. Diana's speed gate fails. Does Diana launch, and under
what standard?

**The standard does not change. It gets applied.**
`docs/benchmarking.md` §4 says exactly two things, and they are not the same
thing: *"a skipped gate is never a pass"*, and *"`verdict: claimable`
requires all four to pass"*. Neither says a component may not ship with a
measured loss — §5 says the opposite, that losses are recorded and stay in
the file. **Carmenta is the standing precedent**: it ships today with
"photo accuracy still trails PaddleOCR, causes diagnosed" in the README's
component line, having passed speed and footprint.

So:

| claim | status |
|---|---|
| `verdict: claimable` (the aggregate) | **NO.** Withheld until speed passes. |
| Quality — mAP identical to PyTorch at matched configuration | **YES**, gated, per tier, per geometry |
| Footprint — 0.38-0.43x the reference's steady memory | **YES**, gated |
| Correctness — five-tier oracle, byte-determinism | **YES**, gated |
| Speed — per-image latency | **NO — published as FAIL with its number** |

Diana therefore launches as *"YOLO26 detection inference in pure Rust,
bit-for-bit parity with PyTorch, ahead on memory / load / determinism /
batch throughput, behind on single-image latency"* — never as "parity" or
"a YOLO replacement" unqualified. The losing row goes in the headline
table, not a footnote, because a project that hides a losing row gets found
out and deserves to be.

**What makes this defensible rather than an excuse is that the loss is now
DIAGNOSED, not merely admitted.** `docs/whys/diana-latency.md` carries the
descent; the three findings are:

1. **The crate has no `target-cpu`.** Every hand-written kernel here — 34 %
   of the profile — compiles for the x86-64 baseline: SSE2, no AVX2, no
   FMA. A build flag alone moves the SiLU kernel 1.39x. It is not simply
   *set*, because baking the build machine's ISA into a published crate is
   a portability bug; the fix is runtime feature dispatch, and that is
   scoped work, not a config line.
2. **The activation was 30.9 % of a detection** — larger than any
   convolution shape. Fixed this session (magic-number rounding, 1.94x,
   bit-identical), and the fix's own ceiling says another 1.45x remains in
   it.
3. **Single-image fan-out costs 2.32x the CPU of the work it performs**, on
   this hybrid CPU, across ~120 barriers per image.

None of the three is a claim that YOLO is beatable on latency tomorrow.
They are the reason the gate reads FAIL, stated so the next session starts
from a diagnosis instead of a symptom.

### 8.5 Design principles carried from Mercury and Carmenta

- **Profile first.** Only touch stages the profiler indicts (`FFAI_PROFILE`
  from day one, §6.3 stage timing).
- **Parity before speed.** A fast wrong model is not a milestone exit;
  every speed brick re-runs the parity oracle.
- **Dispatch when one configuration cannot win everywhere** (tier, end2end
  vs NMS, resolution) — the sign-flip is the trigger, never averaged away.
- **Never delete a working path on one contradictory number.** Verify the
  binary and the exact config first — the stale-binary hour (Carmenta
  §8.12) and its rebuild guard apply verbatim here.
- **Conversion and oracles are product**, not scaffolding: re-runnable
  from a fresh clone, fixtures shipped.
- **Claims are ledger lines or they do not exist.**

### 8.6 Risk register (explicit)

| Risk | Mitigation |
|---|---|
| Ultralytics architecture drift | versioned conversion maps; fail closed on key/shape mismatch; track upstream YAML + module code per release |
| AGPL weight terms | crate permissive; weights never vendored; user obligations documented plainly (§7) |
| Numerical drift from fusion / op differences | intermediate-activation oracles; explicit fused/unfused policy in the manifest. **Retired at M-D1**: the fold is verified to 1e-4 relative across all 24 layers on two fixtures (§8.2) |
| **Porting a training-only branch** — **FIRED at M-D1** | the one2many `cv2`/`cv3` head is dropped at conversion, counted in the manifest, and the oracle dumps the `one2one` branch explicitly so a port cannot compare against the wrong tensors (§8.2) |
| Edge memory pressure | n/s tiers first; steady-memory soak gates; int8 only by measured kernel shape |
| LIVE change-gate false negatives/positives | synthetic corpus with known change times; the fraction-of-pixels-moving gate Carmenta measured into existence, recalibrated for this content |
| Over-claiming vs ORT / vendor runtimes | matched hardware, matched config, exact argv in the ledger; single-run gap ratios never quoted (Mercury's rule) |
| **Unpinned preprocessing geometry** — **FIRED at M-D0** | `--rect` is a required adapter argument, pinned per row, with the two geometries as separate declared configs (§8.1). The general form of this risk: any reference default that changes the work done is a `beam_size`, and the tell is two implementations that should agree and don't |

---

## 9. Pure-Rust boundary and watchlist

**Decisions, recorded:**

- **No ONNX Runtime, no libtorch, no MNN in the default build.** Candle
  is the spine (Principle 3); accelerator backends are feature flags that
  must earn their place with ledger numbers, or they do not exist.
- **Conversion is offline and reproducible.** Runtime loads only
  safetensors + manifest. No Python at inference time, ever.
- **Frames in, structure out.** Camera capture, screen capture, and video
  demux are input plumbing outside the detection crate; `ffai-media` /
  rff / watch directories / caller buffers own ingest — the same boundary
  as Carmenta.

**Watchlist** (re-checked each milestone):

| Item | Status |
|---|---|
| Official YOLO26 (and successor) architecture changes | track; conversion maps versioned per release |
| Pure-Rust / candle YOLO peers | measure for standing; adopt ideas only if they clear the same gates |
| rff video ingest (frame iteration for LIVE) | needed by M-D3; watch-dir interop is the standing fallback, as it was for LIVE OCR |
| Edge NPU / mobile toolchains | candidate backends after CPU and Metal/CUDA (M-D5+) |
| YOLOE / open-vocabulary heads | future milestone after core detect is stable; extra license constraints audited before promotion |
| Training in pure Rust | explicit non-goal for v1 |

---

## 10. Fit inside FFAI and Remade-With-Rust

Diana is a sibling to Mercury and Carmenta, not a loner:

- lives in the workspace as `ffai-diana`, registered like every engine
  family, visible in `ffai engines`;
- consumes frames from the pure-Rust media path, emits `DetectOutput`
  that downstream tools consume — **Carmenta region proposals** are the
  first planned consumer (a detector that hands OCR its text regions);
- shares the bench ledger, the model-manifest system, and the
  no-claim-without-a-number culture.

The name follows the house convention: Diana, Roman goddess of the hunt —
fast, precise detection — beside Mercury (language) and Carmenta (the
alphabet).

---

## 11. What "FFmpeg-grade" means here, restated

Everything optional and composable; explicit measured trade-offs; rich
structured output by default; engines swappable by name; official weights
usable without a Python runtime at inference time; the same mental model
as Mercury and Carmenta — and every number on this page eventually
replaced by a ledger line or deleted.

**The headline edge criterion:** a developer takes an official
`yolo26n.pt`, runs the audited conversion once, ships a pure-Rust binary,
and gets competitive or better latency than the common ONNX Runtime CPU
path on the same edge-class hardware — with numerical behaviour that
matches the official model on the pinned holdouts.
