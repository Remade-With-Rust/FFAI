# Carmenta Mission Plan

**Component:** Carmenta — FFai's OCR component (`ffai-carmenta`)
**Tasks:** OCR across four functions — **live**, **document**, **long**, **formula** — LIVE first
**Status:** Phase 0 stubs registered · this plan takes the component to `stable`
**Prime directive:** pure Rust end-to-end on the candle spine, measured against
the non-Rust world standards — above all the C++ competitor, Tesseract — by
`ffai bench ocr` at every milestone. No claim without a ledger line.

---

## 1. Mission

Ship the first production-grade, pure-Rust OCR stack, delivered as four
functions over one detection → recognition core:

1. **LIVE** — streaming OCR over frames (screen, camera, video): low-latency,
   stable output, and the auto-region system (§4.1) that no mainstream OCR
   tool ships.
2. **DOCUMENT** — layout-aware page parsing: regions, reading order,
   structured Markdown/JSON out.
3. **LONG** — multi-page coherent parsing: state across pages, bounded
   memory, one coherent document out instead of N disconnected pages.
4. **FORMULA** — math regions → LaTeX.

Every milestone exits through the analyzer: `ffai bench ocr` compares our
Rust code to the **non-Rust world standards** — Tesseract (C++), PaddleOCR,
EasyOCR, docTR, pix2tex — on pinned corpora, and the result, win or loss, is
appended to `bench/ledger.jsonl`.

**Success =** CER parity (within the 5 % relative band) with Tesseract on
printed-document holdouts at ≥ its warm throughput, in safe Rust; LIVE
holding its latency and stability gates on a pinned frame corpus; each
remaining function gated against its own world standard. Claims traceable to
ledger lines, reproducible by anyone from the public repo.

**What Mercury proved that this plan assumes:** candle on CPU can stand
toe-to-toe with a hand-tuned C++ implementation, and the measurement harness
is how you find out where it can't. Carmenta therefore does **not** carry
ONNX Runtime or MNN backends — the spine is candle, and any exception must
arrive as a feature flag with a measured reason (Principles 1 and 3).

---

## 2. Design rule: independent stages, composed functions

Carmenta is a **toolbox of independently callable stages**, each with its own
contract and oracle test. The four functions are *compositions* over one
core — flags and loops, not forks. This is the WhisperX lesson applied from
day one instead of retrofitted.

```
ffai-carmenta
├── image/            preprocess: grayscale, binarize, deskew, resize (each independent, each optional)
├── detect/           text detection → line/word polygons (DBNet/CRAFT-class, candle)
├── recognize/        polygon crops → text + confidence (CRNN/SVTR/TrOCR-class, candle)
├── live/             the streaming loop: frame sampling, change detection,
│                     ROI memory + auto-region (§4.1), output stabilization
├── layout/           region classification + reading order (document function)
├── doc/              multi-page state: cross-page merge, header/footer
│                     suppression, hyphenation repair (long function)
├── formula/          math region → LaTeX head (formula function)
└── engine.rs         composes the above → OcrOutput
```

Contracts that keep the stages independent:

- Every stage consumes/produces `ffai-core` types (`ImageBuffer`,
  `OcrOutput`, `TimedSegment`) or plain candle tensors — no stage knows its
  neighbors.
- Each stage has **its own oracle test** against a deterministic fixture
  (§6.2): detection polygons vs the reference implementation's own output on
  the shared synthetic corpus, recognition round-trip on rendered text,
  reading order vs ground truth.
- The live/layout/doc/formula stages operate on *any* `OcrEngine`'s output —
  they wrap results, they do not reach into a model's internals. When a
  better detector replaces the first one, LIVE and DOCUMENT survive
  unchanged.

**Functions are flags, engines are lineages.** `--engine` selects the
det+rec lineage (the codec); `ffai ocr --live`, `--layout`, `--pages`,
`--formula` select the composition. Nothing that has not earned a default
gets one. The Phase 0 stub names (`unlimited-ocr`, `easy-ocr`) are
reconciled at M-C1 when the first real lineage lands — a stub that never
shipped can be renamed freely.

### 2.1 The output type is the AVFrame of this mission

`OcrOutput` today is a flat `Vec<TextSpan>`. Every engine, every function,
and the bench harness will depend on its replacement, so it is designed at
M-C1 and reviewed against all four functions before anything ships:
hierarchy (page → block → line → word), geometry (bbox + polygon), per-level
confidence, reading order as an explicit sequence, and room for
`TimedSegment` wrapping (LIVE emits timed OCR — SRT/VTT of on-screen text).
Tables/formula/chart structures stay **out** of the v1 type; they arrive as
typed block payloads when their milestones land, not speculatively.

---

## 3. The detection → recognition core (the substrate)

Not one of the four functions — the thing all four stand on, and the first
code built (M-C1).

| Capability | Contract |
|---|---|
| Detection | text-line/word polygons, DBNet/CRAFT-class model on candle, resolution-tiered |
| Recognition | per-line transcription + per-char confidence, CRNN/SVTR/TrOCR-class on candle |
| Model tiers | small (live-latency) and accurate (document), selected via `ffai-models` manifests |
| Languages | Latin scripts first; the recognition head is swappable per script family |
| Preprocess | grayscale/binarize/deskew as independent, benchmarked, opt-in stages |
| Quantization | int8 variants measured per stage — Mercury's lesson: quantization pays by kernel shape, not by faith (encoder 2.4× regression vs decoder 3× win) |

**Lineage selection is an M-C0 audit output, not a preference.** Candidates
are scored on: weight license *and* gatedness (§7), feasibility of conversion
to safetensors/candle, size tier fit for LIVE, and existence of a reference
implementation to oracle against. `candle-transformers` 0.11 already ships
TrOCR — a working recognition bring-up shortcut worth weighing against the
smaller classic stacks.

---

## 4. The four functions

### 4.1 LIVE — first, by mission order

Streaming OCR over a frame source: screen capture, camera, video file. The
degenerate case (one frame) is just `ffai ocr`; the function is the loop.

```
ffai ocr -i screen.mp4 --live -o captions.srt      # timed on-screen text
ffai ocr -i lecture.mp4 --live --sample keyframes  # frame strategy
ffai ocr -i cam.mp4 --live --roi 100:100:800:600   # fixed region
ffai ocr -i ui.mp4 --live --auto-roi               # region memory (below)
```

| Piece | Contract |
|---|---|
| Frame sampler | every-N / keyframe / change-triggered; a skipped frame costs zero model work |
| Change detection | frame-diff gate — an unchanged frame **must** produce byte-identical text at ~zero cost |
| Output stabilizer | text churn between visually identical frames is a defect, measured as such |
| Timed output | `TimedSegment<OcrOutput>` → SRT/VTT/JSON, same converters Mercury uses |
| ROI | fixed crop/polygon from the caller |
| Auto-ROI | the region-memory system below |

**Auto-ROI (the distinctive piece).** Full-frame detection continuously is
wasted work when text lives in stable regions (HUDs, tickers, UI panels).
The design, kept from the original notes:

1. **Calibration:** run full-frame detection for the first ~30–120 frames
   (or in the background at low frequency while provisional ROIs serve).
2. **Cluster:** accumulate detection boxes over time; cluster spatially;
   keep regions that fire repeatedly with high-confidence text.
3. **Promote:** stable clusters become active ROIs — loose boxes first,
   shrunk gradually under metric supervision.
4. **Monitor and adapt:** per-region confidence, detection count, and
   new-vs-repeated text rate; a region going quiet widens or re-triggers
   full-frame detection; periodic low-frequency full-frame sweeps catch new
   elements; scene change resets.

**Auto-ROI is gated like a search-skip gate:** an observe-only harvest on
the pinned frame corpus first (what would ROI mode have skipped, what did
skipping cost), the ceiling sweep recorded in the ledger, and the feature
lands opt-in. If the harvest says full-frame detection is already cheap
enough, the honest outcome is "not built" — the sweep is the discovery.

**LIVE metrics** (all ledger-recorded, best-of-N, warm/cold separated
exactly as [benchmarking.md](benchmarking.md) requires):

- time-to-first-text (cold, includes model load — recorded separately);
- p50/p95 per-frame latency, warm;
- ×realtime over the pinned video corpus;
- **stability:** churn rate on identical consecutive frames (gate: zero) and
  on a static-with-noise segment (gate: within band);
- CER against per-frame ground truth — and parity with our own batch mode on
  the same frames (live must not silently trade accuracy for latency; when
  `--auto-roi` trades it explicitly, the trade is printed).

**The C++ competitor for LIVE is Tesseract invoked per frame.** No
established live-OCR reference exists — that is the opportunity — so the
speed/latency bar is the C++ engine on the same frames, and the quality bar
is external ground truth, never our own batch path (no self-grading).

### 4.2 DOCUMENT

Layout-aware single-page parsing: region classes (text, title, table,
figure, caption, header/footer), reading-order recovery, structured output.

```
ffai ocr -i scan.png --layout -o page.md           # structure preserved
ffai ocr -i scan.png --layout -o page.json         # full hierarchy + geometry
```

- Layout model: PP-Structure/LayoutParser-class region detector on candle —
  same audit criteria as §3, licenses first (several popular layout models
  are AGPL or non-commercial; those are disqualified at the audit, not
  discovered at ship time).
- Reading order: rule-based baseline first (column detection + XY-cut),
  measured; a learned model only if the baseline measurably fails.
- Tables → structure (TEDS-scored) are **in** the document function's remit
  but a separate milestone gate — the plain layout path must not wait for
  table structure to mature.

### 4.3 LONG

Multi-page coherent parsing — the document function with state:

- streaming page-by-page over pre-rendered page images, memory bounded and
  **flat in page count** (footprint gate on a 500-page run);
- cross-page repair: hyphenation merge, running header/footer suppression,
  paragraph continuation across page breaks;
- one coherent document out: single reading order, stable section hierarchy;
- caching of per-page intermediates so re-runs with different output formats
  don't re-OCR.

**Sequencing note (deliberate deviation from the goal's listed order):**
DOCUMENT builds before LONG because LONG consumes layout and reading order —
built the other way round, cross-page coherence would be rebuilt after
layout lands. LIVE stays first as directed; it composes over the bare
det+rec core and shares nothing with layout.

**PDF is explicitly out of scope for the raster milestones.** Carmenta's
inputs are raster images (PNG/JPEG/WebP via the rff decoders the ROADMAP
assigns to Phase 3) and pre-rendered page images. The pure-Rust ecosystem
has no production PDF rasterizer; writing one is a real future
Remade-With-Rust project in its own right (§9), not a Carmenta side quest.

### 4.4 FORMULA

Math region → LaTeX, as a specialized recognition head:

```
ffai ocr -i paper.png --layout --formula -o paper.md   # inline $...$ in output
ffai ocr -i equation.png --formula                     # single-region mode
```

- Model: pix2tex/UniMERNet-class encoder-decoder on candle, audit-selected.
- Metrics: normalized LaTeX edit distance on an im2latex-class holdout;
  expression-level exact-match rate reported beside it (edit distance
  flatters near-misses; ExpRate punishes them — report both, claim neither
  alone).
- **Honest reference note:** formula OCR has no C++ world standard; the bar
  is pix2tex (Python/PyTorch). Recorded as such in the references table
  rather than quietly substituting an easier comparison — the Mercury M0
  pattern for the missing whisper.cpp baseline.

---

## 5. World-standard references (non-Rust)

Declared in `corpora/references.toml`; versions recorded per ledger line.
Tesseract runs in the single-file `command` mode benchmarking.md already
specifies for it.

| Reference | Stack | Why it's the bar |
|---|---|---|
| **Tesseract** | **C++** | the native-CPU document OCR bar — the whisper.cpp of this mission; per-function competitor for LIVE (per-frame), DOCUMENT, and LONG |
| PaddleOCR (PP-OCR det+rec, PP-Structure) | Python + Paddle (C++ core) | the scene+document accuracy definition most products actually use |
| EasyOCR | Python/PyTorch | the scene-text breadth bar; the lineage `easy-ocr` answers to |
| docTR | Python/PyTorch | the modern document-pipeline bar (detection+recognition, orientation) |
| pix2tex (LaTeX-OCR) | Python/PyTorch | the formula bar (no C++ standard exists — recorded, not hidden) |
| ocrs | Rust | the pure-Rust peer — measured for standing, not the bar; also tracked as a possible zero-download baseline engine |

Same discipline as Mercury's table: pin every configuration knob that
changes the work (model tier, resolution, page-segmentation mode —
Tesseract's `--psm` is this mission's `beam_size`), record exact argv in the
ledger, and add matched-configuration variants when defaults differ.

---

## 6. Analyzer integration (`ffai bench ocr`)

The ASR vertical stands; Carmenta work extends the harness. benchmarking.md
gains its OCR reproduce-from-source section at M-C0.

### 6.1 Metrics, per function

| Function | Quality | Speed | Footprint |
|---|---|---|---|
| core | CER/WER on transcription (normalizer parity per corpus class), detection precision/recall/hmean @ IoU 0.5 | ×RT warm + e2e, pages/s | steady MiB, peak beside it |
| LIVE | CER on ground-truthed frames; churn rate | p50/p95 frame latency, time-to-first-text, ×RT | steady MiB **flat over a 30-min stream** |
| DOCUMENT | end-to-end CER + reading-order accuracy; TEDS when tables land | pages/s | steady MiB |
| LONG | full-document CER (coherence must not cost accuracy vs per-page) | pages/s over 500 pages | **flat in page count** |
| FORMULA | normalized edit distance + ExpRate | regions/s | steady MiB |

### 6.2 Corpora — two kinds, deliberately

**Synthetic, deterministic, license-free (the mel-fixture trick).** Text
rendered to pixels with pinned open-licensed fonts is a corpus with *exact*
ground truth that anyone can regenerate from a formula — no license, no
download gate, byte-reproducible. One per function: rendered pages (core),
a scripted synthetic screencast with timed text changes (LIVE — also the
stability oracle, since ground truth includes *when* text changed), rendered
multi-page books from public-domain Gutenberg text (LONG), rendered LaTeX
(FORMULA). These are the smoke corpora and per-stage oracles.

**Public ground-truth sets (the claims corpora).** Real photographs, scans,
and handwriting — synthetic corpora cannot support a public claim about real
documents. Candidates (FUNSD, IIIT5K, SVT, Total-Text, PubLayNet/DocBank,
im2latex-100k) are **audited at M-C0** for license *and* for
fetchable-without-an-account (several ICDAR sets sit behind registration —
the pyannote lesson applies to corpora too). Chosen sets get hash-pinned
manifests with holdout/train splits; claims are measured on holdout only.

#### The DOCUMENT holdout, pinned (2026-07-30)

`carmenta-doclaynet-v1` — 60 pages from **DocLayNet-v1.1**, hash-pinned,
stratified 10 per document category across all six (financial reports,
government tenders, laws and regulations, manuals, patents, scientific
articles), split 12 train / 48 holdout *within* each category. Per page:
the image, plain text from `pdf_cells` for the CER gate, and a JSON of
layout regions with their classes for the reading-order gate. M-C3 had no
corpus that could fail it before this; now it does.

Licence **CDLA-Permissive-1.0**, stated on the dataset card. Clips are not
committed — `/corpora/clips/` is ignored and the manifest's hashes are what
travel, so a fresh checkout regenerates the corpus and verifies it byte for
byte.

Two candidates were rejected on audit, and the reasons are worth keeping:

| corpus | verdict |
|---|---|
| **OmniDocBench** | Purpose-built for document parsing and the better fit on paper — reading order, formulas, tables. **Rejected 2026-07-30, and the rejection was WRONG — see the correction below.** |
| **PubLayNet** | Layout only, no text ground truth, and the HF mirrors return 401. |

#### Correction (2026-07-31): the OmniDocBench rejection was wrong, twice

The audit above rejected OmniDocBench for having no stated licence, and gave
"we redistribute clips in-repo" as why that mattered. Both halves are false.

1. **The licence is stated: Apache-2.0**, in the project's own repository. The
   check only read the HuggingFace card metadata and the mirror's README, and
   neither surfaces it. *Check the project, not the mirror.*
2. **We do not redistribute clips at all.** `/corpora/clips/` is gitignored;
   manifests and prepare scripts travel, pixels are refetched and hash-verified.
   The rationale given for why an unstated licence disqualified it therefore
   did not apply to this repo in the first place — and I had already corrected
   exactly that sentence for DocLayNet in the same section while leaving this
   rejection standing on it.

The cost of the error is not small. OmniDocBench is 1,651 pages with layout,
reading order, text, table (LaTeX and HTML) and formula annotations — it is a
better fit for M-C3 than anything else audited, it carries M-C5's formula
ground truth too, and it is **the board on which Baidu's Unlimited-OCR states
its headline result** (93.23 % v1.5, 93.92 % v1.6). Rejecting it kept every
Carmenta document number self-referential, which is fine for engineering and
worthless for a comparable claim.

Recorded rather than quietly reversed: a corpus rejected on a bad audit is
indistinguishable from a corpus that was never considered, and the next person
reading the table would have inherited the conclusion without the defect.

A first attempt at this pinned 39 pages covering four of six categories, with
every financial report in train and every government tender in holdout —
DocLayNet is stored grouped by category, so a prefix of one shard is not a
sample of the dataset. Stratifying *within* category fixed it. Recorded
because an unrepresentative holdout fails silently: it produces a number.

DocLayNet also seeds M-C4 (LONG) at no extra cost — `original_filename` and
`num_pages` group pages back into documents, which is exactly what the
multi-page holdout needs.

### 6.3 Harness work items

- image/frame corpus support in `ffai-bench` (manifests currently assume
  audio clips);
- detection-metric scoring (polygon IoU matching) beside the existing
  CER/WER code;
- frame-latency timing mode for LIVE (the warm/e2e two-number discipline
  maps directly: warm = per-frame, e2e = includes load);
- OCR reference adapters under `corpora/refs/` (tesseract via `command`
  mode; Python references via the existing batch JSONL contract);
- the quality gate learns **matched-tier comparison** — judged against the
  reference at the *matched* model tier, with the best-reference number
  reported beside it. This is the gate-split the README already records as
  open work for ASR; OCR lands it from the start rather than inheriting the
  conflation.

---

## 7. Weights: license and gating audit (Principle 4)

Every model Carmenta fetches must be fetchable without an account, its
license surfaced by `ffai models`. The audit table is an M-C0 deliverable —
filled with checked facts, not reputations. Candidates going in:

| Stage | Candidate lineages | Audit questions |
|---|---|---|
| detect | DBNet (PaddleOCR det), CRAFT | license of *weights* (not just code); Paddle→safetensors conversion cost |
| recognize | CRNN, SVTR, PARSeq, TrOCR | HF-hosted ungated? size tier for LIVE? candle port exists (TrOCR) or is real work? |
| layout | PP-Structure-class | several popular layout models are AGPL or research-only — disqualifying |
| formula | pix2tex, UniMERNet-class | weight license; decoder vocabulary licensing |

**Paddle-format weights convert to safetensors as milestone work, not a
footnote** — budgeted in M-C1, verified by per-stage oracle against the
reference implementation's own output.

### 7.1 Audit results (M-C0, verified 2026-07-29)

Every row verified against a live URL on the date above, not answered from
reputation. Full row-by-row detail (hosts, formats, sizes, verified-at URLs)
in the M-C0 ledger notes; the decisions:

**Weights — cleared:**

| Stage | Lineage | Verdict |
|---|---|---|
| detect | **PP-OCRv5 det** (DBNet-class) | Apache-2.0, ungated HF (`PaddlePaddle/*`) **plus** direct bcebos tar — two independent no-auth hosts; mobile 4.7 MB / server 84 MB, the size tiers §3 wants for LIVE vs DOCUMENT |
| detect (fallback) | CRAFT | MIT, but official weights sit behind Google Drive — use the ungated re-host in EasyOCR's GitHub releases, never the Drive links |
| recognize | **PP-OCRv5 rec** (SVTR-class) | Apache-2.0, same dual hosting, mobile ~16.5 MB |
| recognize (alt) | PARSeq | Apache-2.0, GitHub-release `.pt` (includes an Apache CRNN baseline); TrOCR is ungated on HF but carries **no license tag** (MIT only by inference from microsoft/unilm) — secondary |
| layout | **PP-DocLayout** | Apache-2.0, ungated HF |

**Weights — disqualified:** DocLayout-YOLO (AGPL-3.0); **pix2tex weights
(CC BY-NC-SA — non-commercial)**. Consequence for M-C5: pix2tex remains the
*reference implementation* we benchmark against (we never redistribute its
weights), but Carmenta's own formula lineage must come from the
UniMERNet-class pool or be trained/converted from clean sources — an M-C5
entry decision, flagged now rather than at ship time.

**Corpora — cleared:** im2latex-100k (**CC0**, direct Zenodo — the formula
claims corpus), CORD-v2 (**CC-BY-4.0**, ungated HF — receipts), DocBank
(Apache-2.0, ungated HF — layout; repo asks no-redistribution, so benchmark
against it, never vendor it), PubLayNet (CDLA-Permissive annotations; IBM's
official CDN is dead, HF mirror only — acceptable with provenance noted).

**Corpora — disqualified:** every RRC-portal set (ICDAR 2013/2015,
Text-in-Video, SROIE — registration verified required), FUNSD
(non-commercial license), Total-Text (commercial-permission clause + Google
Drive), IIIT5K (official host unreachable, no stated license). SVT is
downloadable ungated but carries no explicit license — usable only with a
provenance flag, not as a claims corpus.

The pyannote lesson held in both directions: the most-cited academic OCR
corpora are exactly the gated ones, and the synthetic tier (§6.2) plus
CORD/im2latex/DocBank covers the claims path without a single account.

---

## 8. Milestones and exit gates

Every milestone exits through all four gates (correctness / quality / speed /
footprint) on holdout — a skipped gate blocks exit. Losses are recorded.

| # | Deliverable | Exit gate (ledger-recorded) |
|---|---|---|
| **M-C0** | Baselines: `ffai bench ocr` vertical (§6.3), synthetic + audited public corpora pinned, `--baseline-only` runs of Tesseract/PaddleOCR/EasyOCR/docTR/ocrs; the §7 audit table | reference CER/hmean/pages-per-second on the board; corpus hashes pinned; audit published; benchmarking.md OCR section merged |
| **M-C1** | Det+rec core on candle: lineage selected by audit, `OcrOutput` v2 hierarchy in `ffai-core`, per-stage oracles vs references on the synthetic corpus, stub names reconciled | CER within the 5 % relative band of **PaddleOCR-mobile** on the printed holdout (see the note below); all four gates run and recorded (speed may fail honestly at bring-up — it did for Mercury) |
| **M-C2** | **LIVE**: streaming loop, frame sampler + change gate, stabilizer, timed SRT/VTT output, `--roi`; auto-ROI observe-only harvest + ceiling sweep, then opt-in `--auto-roi` if the sweep pays | p95 warm frame latency ≤ Tesseract per-frame on the same frames; zero churn on identical frames; CER parity with own batch mode on the frame holdout; footprint flat over a 30-min synthetic stream; auto-ROI sweep in the ledger win or lose |
| **M-C3** | **DOCUMENT**: layout stage, reading order, `--layout`, Markdown/JSON structured output | reading-order accuracy + end-to-end CER vs Tesseract and PP-Structure on the document holdout; structured output round-trips losslessly to JSON and back |
| **M-C4** | **LONG**: multi-page state, cross-page repair, bounded-memory streaming, intermediate caching | full-document CER ≤ the same engine's per-page score (coherence costs nothing); footprint flat over 500 pages; vs Tesseract/docTR on the multi-page holdout |
| **M-C5** | **FORMULA**: LaTeX head, `--formula` routing from layout regions | edit distance + ExpRate vs pix2tex on the pinned holdout; composes with `--layout` (inline `$...$` in Markdown out) |
| **M-C6** | Carmenta `stable`: docs, library examples, claims page generated FROM the ledger; tables/TEDS if M-C3's table work matured | every public claim maps to a ledger line id |

**Gate correction (2026-07-31): M-C1's quality bar named Tesseract and was
unsatisfiable.** §8.1 measured Tesseract at **0.00 % CER** on the printed
holdout — the ceiling effect that section already recorded — and "within 5 %
relative of 0.00 %" cannot be cleared by any engine, PaddleOCR included, which
reads 0.02 % there. A gate no implementation can pass is not a gate; it is a
blocker with a number on it. The bar is now PaddleOCR-mobile, which is also the
reference this campaign was directed to compete against, and the printed corpus
keeps the role §8.1 assigned it: a smoke/oracle tier that answers "is the core
correct", not "is it better". Recorded rather than quietly amended, because
moving a bar you are failing is exactly the move that needs a paper trail.

Sequencing notes: **M-C2 (LIVE) is the priority milestone** and follows
M-C1 directly — it needs only the core plus composition. M-C5 (FORMULA)
shares no code with M-C4 (LONG) and can run in parallel with it — two
tracks, one analyzer, exactly the Mercury M3/M4 pattern. DOCUMENT precedes
LONG for the dependency reason in §4.3.

### 8.1 M-C0 result — baselines on the board (CLOSED 2026-07-29)

Ledger `bench-ocr-1785332842` (render) and `bench-ocr-1785333423` (frames),
best-of-3, CPU only, Windows x86_64. Corpora §6.2: synthetic, deterministic,
SHA-pinned (`2f7151b9e150`, `fd0b2cab9585`). All three named competitors ran
every holdout clip; scoring is `Mode::Ocr` (whitespace-collapsed,
case/punctuation-preserving), identical for all.

**carmenta-render-v1** — 18 holdout printed pages:

| Implementation | Stack | CER % | WER % | pages/s warm | p50/p95 per page | peak MiB |
|---|---|---:|---:|---:|---|---:|
| tesseract 5.5.3 | **C++** | **0.00** | **0.00** | **2.96** | **342 / 397 ms** | **56** |
| paddleocr-mobile (PP-OCRv5) | Python+Paddle | 0.02 | 0.16 | 0.09 | 12.3 / 14.1 s | 878 |
| easyocr 1.7.2 | Python/PyTorch | 3.65 | 9.03 | 0.31 | 3.3 / 3.8 s | 1227 |

**carmenta-frames-v1** — 23 holdout HUD-style frames (the LIVE view):

| Implementation | Stack | CER % | WER % | pages/s warm | p50/p95 per frame | peak MiB |
|---|---|---:|---:|---:|---|---:|
| tesseract 5.5.3 | **C++** | **0.44** | **1.51** | **4.75** | **205 / 312 ms** | **56** |
| paddleocr-mobile (PP-OCRv5) | Python+Paddle | 1.31 | 4.90 | 0.25 | 3.9 / 6.2 s | 891 |
| easyocr 1.7.2 | Python/PyTorch | 4.33 | 16.93 | 0.50 | 2.0 / 2.5 s | 1295 |

Read the tables four ways:

1. **Tesseract is the bar on every axis of this corpus class — quality,
   speed, latency, and footprint at once.** 0.00 % CER on clean rendered
   print, ~10–60× lower per-frame latency than the neural references, at
   1/15th–1/20th their memory. The "whisper.cpp of this mission" framing
   survives contact: the C++ engine is not a soft target, and M-C1's parity
   band is a real gate. Worth restating what the bar is: 40 years of
   hand-tuned C++ against unoptimized-for-CPU Python inference stacks — the
   gap is real but it is also exactly the gap a pure-Rust candle engine is
   supposed to close from the other side.
2. **The render corpus has a ceiling effect, recorded now rather than
   discovered later.** Two of three references sit at ≈0 % CER on clean
   synthetic print, so this corpus cannot *rank* quality among strong
   engines — it is the smoke/oracle tier (§6.2), and M-C1's quality gate on
   it answers "is the core correct", not "is it better". The frames corpus
   separates the field (0.44 / 1.31 / 4.33) and real-photograph claims wait
   on the audited public corpora.
3. **Both stated handicaps are visible in the numbers.** PaddleOCR runs
   `--mkldnn off` (paddlepaddle 3.3.1's oneDNN executor crashes on this box
   — §7.1 adapter note), which is much of why the *mobile* tier reads 12 s a
   page; its quality (0.02 % CER) is unaffected and is the accuracy bar
   among the neural stacks. Tesseract's per-frame numbers include its
   ~10–30 ms process-spawn tax. Neither caveat is in a footnote nobody will
   read — both ride in the ledger notes of every run.
4. **EasyOCR's WER≫CER split (9.03 vs 3.65) is a word-segmentation
   signature, not a reading failure** — characters are mostly right, spaces
   are not. `Mode::Ocr` scores it as-is; the same behaviour will apply to
   our engine, which is why the quality gate verdicts on CER with WER
   recorded beside it.

**What M-C0 leaves for M-C1/M-C2:** the det+rec core must land within the
5 % CER band of *these* tesseract rows at bring-up quality gate, and LIVE's
p95-vs-tesseract gate now has concrete numbers to beat: **397 ms (pages) /
312 ms (frames)** — including, honestly, deciding how our in-process p95
compares against a number that carries their spawn tax.

Exit gate: reference CER / pages-per-second / per-page latency on the board
across both corpora ✅ · corpus hashes pinned ✅ · §7.1 audit published ✅ ·
benchmarking.md OCR section merged ✅. **M-C0 CLOSED.**

### 8.2 M-C1 result — the core is real; quality gate open with a named cause

Ledger `bench-ocr-1785341085` (render) / `bench-ocr-1785342064` (frames),
full four-gate runs, engine + all three competitors, best-of-3, CPU.

**The engine is `craft-crnn`** — CRAFT detection + english_g2 CRNN
recognition, the EasyOCR model stack in pure Rust on candle. Not the planned
lineage: recognition was TrOCR's seat until the oracle fixture measured
trocr-small-printed reading mixed-case text as ALL CAPS (SROIE-trained) —
fatal under a case-scoring CER gate, discovered for the price of one Python
script instead of a port. Stub names reconciled: engines are named by
lineage, `easy-ocr`/`unlimited-ocr` retired.

**Per-stage oracles (the §2 contract), both PASS:** CRAFT region/affinity
maps match PyTorch to <5e-3 max-abs on the pinned fixture; CRNN matches to
EXACT per-timestep argmax. Two porting traps are now recorded in the code
where they bit: torchvision's inplace ReLUs make three of CRAFT's four skip
taps effectively post-ReLU (a 0.96 max-abs failure until instrumented), and
the upconv widths in the repo's own constructor calls don't match the
shipped checkpoint — every conversion loads `strict=True` for exactly this
reason.

| carmenta-render (18 pages) | CER % | WER % | pages/s | steady MiB |
|---|---:|---:|---:|---:|
| tesseract | **0.00** | **0.00** | **1.15–2.96** | **54** |
| paddleocr-mobile | 0.02 | 0.16 | 0.08 | 929 |
| **craft-crnn (ours)** | 0.73 | 4.86 | 0.14 | 760 |
| easyocr | 3.65 | 9.03 | 0.28 | 1226 |

Frames corpus: ours 1.74 % CER vs tesseract 0.44 / paddle 1.31 / easyocr 4.33.

**Gates: correctness PASS (41/41), quality FAIL, speed FAIL, footprint FAIL
— all honestly.** The gate table gained an absolute floor (+0.25 pp beside
the 5 % relative band): a relative band against a 0.00 % reference demands
perfection to the character, which is degenerate, and the change is in code
and ledger, not in anyone's memory.

Three findings worth more than the verdicts:

1. **We beat our own lineage's reference implementation 5× on CER** (0.73 vs
   easyocr's 3.65) — line-level crops dodge its word-segmentation errors.
2. **The residual quality gap is the MODEL's ceiling, not the port's:**
   the dominant error class (sentence-final `.` → `:` or dropped) is
   reproduced *identically* by EasyOCR's own torch model on the same pixels.
   No amount of preprocessing tuning closes it — measured: pad sweeps and a
   bicubic crop resize moved ≤0.1 pp. The successor lineage is staged:
   **parseq-tiny weights are converted, sha-pinned, and shape-mapped**
   (`models/parseq-tiny.toml`, `tools/carmenta_parseq_prepare.py`); the
   candle port + forward oracle is the open quality-gate brick.
3. **The profiler (`FFAI_PROFILE=1`, §8.3) localized the speed gap on its
   first run** — this is Mercury's M1→M2 position: unoptimized bring-up,
   levers named, campaign open.

### 8.3 Speed campaign opening + M-C2 LIVE first light

**Profile first, always.** Per-stage split on first measurement: CRAFT's
forward is **61.9 % of a page and 89.3 % of a frame**; recognition is the
rest; box extraction and preprocessing are noise (<1 %). Nothing but the
two model forwards is worth touching.

**Lever 1, measured and landed: detection scale.** Recognition crops from
the ORIGINAL image, so detection resolution trades only detection recall —
`FFAI_DET_SCALE=0.5` cuts a 720p frame **5.5 s → 2.07 s (2.7×)** at +0.09 pp
train CER. Recorded as an env knob, default 1.0 (reference behaviour).

**LIVE exists** (`live.rs` + `ffai ocr --live` over frame sequences; rff
video ingest slots in behind `ffai-media::sample_frames` when published) and
its bench ran on the new 180-frame `carmenta-screencast-v1` corpus (six HUD
slots, staggered change periods, ±1-level per-frame noise so byte-equality
can't fake a change gate):

| LIVE gate (ledger, screencast corpus) | Result |
|---|---|
| correctness: churn + batch parity | **PASS** — 0 churn / 156 unchanged pairs, 0 parity breaks; stateless tesseract churns 24/156 on the same frames |
| quality: CER on change frames | **PASS** — 1.80 % vs batch-mode 1.74 % (+0.25 pp band) |
| speed: p95 vs tesseract per-frame | **FAIL** — 2468 ms vs 414 ms |
| footprint: flat over 30 min | **PASS** — 6033 frames, 313 → 326 MiB window medians, ratio 1.041 (gate ≤ 1.10); soak steady ~330 MiB vs 1073 MiB batch-harness steady — the change gate is also a memory feature |

The change gate needed one measured revision on the way: mean-abs-diff at
threshold 2.0 swallowed real single-slot changes (9 calls for 24 change
events; stale text read as 12 % CER). The shipped gate counts the FRACTION
of pixels moving >8 levels — noise crosses it never, one changed line
crosses it by orders of magnitude — and the rerun hit **24 OCR calls for 24
change events exactly**, zero missed, zero spurious.

**Auto-ROI: the observe-only harvest says BUILD.** Calibration bands from
the first 30 frames cover **100 % of all later text boxes at 38.8 % of frame
area** — a 61 % detection-pixel ceiling on top of lever 1. With detection at
~70 % of the post-lever-1 frame, that projects roughly 2468 → ~1200 ms p95;
the rest of the gap to tesseract's 414 ms is the det/rec forward work
itself (int8-by-shape, batched line crops — unmeasured, so unclaimed).

**Open bricks, in order:** parseq word-crop defect (below) · auto-ROI
implementation against the recorded harvest (speed gate) · 30-minute
footprint soak · det/rec forward optimization.

### 8.5 LIVE speed campaign, round 2 — dirty bands + async sweeps

The dirty-band gate is the lever the harvest predicted: per-band change
tracking re-reads ONLY the band whose pixels moved (plus an outside-band
escape that forces an immediate sweep when text appears where no band is).
Combined with sweeps moved OFF the serving path (background thread, and
the hard-won rule that a landed sweep refreshes GEOMETRY ONLY — caching its
text resurrected stale content one frame late, measured as CER 1.85 → 8.74 %
and a swallowed segment before the fix) and calibration reported as the
LOAD_S of the loop (the warm/e2e precedent):

| Steady-state band call (screencast, det scale 0.5) | p50 | p95 |
|---|---:|---:|
| before this round (all-bands ROI) | 2096 ms | 2468 ms |
| dirty bands + async sweeps | **~350–450 ms** | **~700–850 ms** |
| tesseract per-frame (same runs) | 240–270 ms | 330–425 ms |

Quality held throughout: churn 0/156, CER 1.85 % (PASS), zero missed
change events. **Steady p50 is at parity with the C++ bar; steady p95
reads ~1.8× behind** (multi-slot frames re-read two bands; outer band
parallelism was measured a REGRESSION — rayon nesting with candle's pools —
and reverted; line-level parallelism is now conditional on ≥3 lines for the
same reason). Run-to-run machine noise on these numbers is ±20 % (tesseract
itself read 332–424 ms p95 across the round) — per Mercury's rule, read the
throughput trend, not single ratios.

Also measured and closed this round: detection scale is DONE as a lever —
0.5 is the optimum (0.4 costs +2 pp frames-CER, 0.35 collapses to 17 %:
22 px HUD glyphs at 0.35 are below CRAFT's floor). **The speed gate's
remaining distance is the detector lineage swap (PP-OCRv5 mobile det,
audit-cleared), which shrinks BOTH the strip reads and the sweeps.**

**Round 3 — the gate closed by removing detection from the serving path
entirely.** A dirty band whose known geometry is a single line does not
need detection at all: `OcrOptions::single_line` (tesseract's `--psm 7`
analog, a public engine option, not a LIVE backdoor) recognizes the band
strip as one line — CRAFT runs only in calibration and the async sweeps,
which is what "region memory with periodic full-image verification" meant
all along. Steady band calls collapsed to recognition cost:

| Run (same corpus, same machine) | engine p50/p95 | tesseract p50/p95 | speed gate |
|---|---|---|---|
| 1 | 269 / **316 ms** | 250 / 342 ms | **PASS** |
| 2 | 328 / 371 ms | 273 / 360 ms | FAIL (by 3 %) |
| 3 | 330 / 373 ms | 294 / 410 ms | **PASS** |
| 4 (quiet machine) | 318 / 354 ms | 280 / 385 ms | **PASS** |

Under the harness's own best-of-N rule (benchmarking.md: the minimum is
the run least perturbed, applied to BOTH sides): **316 vs 342 ms — PASS.**
Read it the Mercury way: statistical parity with the C++ bar, margin ~8 %
and inside machine noise, so the honest claim is "at the bar", not "past
it" — the mobile-det lineage remains the margin-builder, no longer the
gate-closer. Quality held through every round: CER 1.91 %, churn 0/156,
24/24 change events caught. Batch parity in ROI mode is reported
informational per §4.1 (the explicit opt-in trade); it remains a hard
gate with auto-ROI off.

**M-C2 exit criteria: latency PASS (above) · churn zero PASS · CER parity
PASS · footprint flat PASS (§8.3 soak, ratio 1.041). All four green.**

**Frame source, decided and shipped:** LIVE v1 interops with any capture
tool via `ffai ocr --live --watch <idle-secs>` — a watched directory
processed in real time as OBS / `ffmpeg -f gdigrab` / any screen-capture
writes frames, wall-clock timestamps, SRT/VTT out. rff video ingest slots
in behind `ffai-media::sample_frames` when it publishes; containers are
plumbing, not a Carmenta blocker. The tool now points at a screen.

### 8.6 Target change: Paddle is the matched bar; function-vs-function benching

**Decision (user directive, 2026-07-30):** tesseract leaves the matched
quality comparison. Its 0.00% render CER is real and five-times reproduced
— the SYNTHETIC corpus's ceiling effect, recorded since M-C0 — but a
zero-error reference degenerates the parity band, and the accuracy bar a
neural stack must answer to is PaddleOCR. Effective immediately:
`paddleocr-mobile` (render 0.019% / frames 1.31% CER) is the matched
reference; tesseract scores open-field only, and stays in LIVE benches as
the per-frame C++ latency line.

**The program shifts from corpus-level CER/WER to FUNCTION-vs-FUNCTION
against Paddle's same stage:**

| Our stage | Paddle counterpart | Metrics |
|---|---|---|
| detection (CRAFT today, mobile-det port next) | PP-OCRv5_mobile_det | box recall/precision vs their polys (IoU 0.5), ms/frame |
| recognition (craft-parseq / crnn) | PP-OCRv5_mobile_rec | per-crop CER on IDENTICAL crops, ms/crop |
| LIVE loop | paddle per-frame (stateless) | churn, p50/p95, CER on change frames |

Harness work: a stage-dump adapter (paddle det polys + rec text per given
crop, mkldnn-off path) and a `stage_bench` example scoring stage-for-stage
on the pinned corpora. This is the WhisperX-layer measurement philosophy
applied inside the OCR pipeline: optimize the stage that measurement
indicts, not the pipeline average.

### 8.7 The CORD det campaign — three refutations and a visual verdict

Function-vs-function on real receipts, chasing the 27%-vs-15.6% pipeline
gap after the rec stage was ACQUITTED (parseq 1.5% beats paddle's rec 3.0%
on identical crops, 2.6x faster):

- thresholds: REFUTED (coverage bit-identical across 3 configs);
- color input: REFUTED (63.1 -> 63.6%; receipts are monochrome ink;
  kept for reference fidelity);
- adaptive target-scale: LANDED for latency+footprint (det 10.45 -> 5.58s
  at target 1280, camera monsters capped at 0.375x, small photos magnified
  above CRAFT's measured ~8px floor) but coverage moved only 63 -> 68%.

Then the box-overlay instrument settled it visually (det_compare_055):
1. the coverage METRIC under-credits us — our word-fragment boxes vs
   paddle's line-group polys means wide-poly centers land in inter-word
   gaps (granularity artifact, metric to be fixed bidirectionally);
2. CORD images are PRIVACY-BLURRED; if exported GT includes blurred lines,
   every engine's CER is inflated (paddle's own 15.6% says yes) — corpus
   GT needs a blur filter;
3. **the real spike mechanism is CAMERA TILT**: line grouping assumes
   axis-aligned rows, tilted receipts split physical lines into phantom
   bands and scramble reading order AFTER correct detection. Deskew (the
   plan 3 preprocess that waited for a corpus that could fail it) and/or
   slope-tolerant grouping is the indicted fix.

Open bricks from this campaign, in order: blur-filtered CORD GT (corpus
honesty first) -> bidirectional coverage metric -> deskew/slope-tolerant
grouping gated per-clip -> parseq-for-photos dispatch -> pipeline re-bench
vs paddle.

### 8.8 The three diagnostics — two indictments overturned, one fix pruned

Run before buying any expensive fix, exactly as the campaign rule requires.
Every headline below replaced a belief this plan previously held.

**1. Detection recall was never the problem.** The "63 % coverage" that
drove three sessions was a METRIC ARTIFACT: centre-in-rect scoring punished
our word-level boxes against paddle's line-level polys. Scored
bidirectionally with any-overlap (granularity-fair):

| | |
|---|---:|
| paddle regions found by ours (recall) | **93.3 %** (4/45 clips < 80 %) |
| our regions found by paddle (precision) | **97.0 %** (1/45 clips < 80 %) |

Consequence: the mobile-det port is a SPEED/margin play, not a recall fix.

**2. Reading order is not the loss either.** Sequential vs bag-of-words CER
(sort the tokens on both sides — if order were scrambled, bag collapses):

| | seq CER | bag CER | order tax |
|---|---:|---:|---:|
| ours (craft-crnn) | 27.3 % | 33.9 % | **−6.7 pp** |
| paddle | 15.6 % | 14.8 % | +0.8 pp |

Our order tax is NEGATIVE — sorting makes us worse. The text comes out in
sequence; something else is wrong with it.

**3. The blur-filtered-GT test was invalid by construction** and is recorded
as such rather than as a finding: removing GT words while the engine still
emits them converts each one into an insertion, so both sides "worsened"
(ours 27.4 → 48.5 %, paddle 15.4 → 38.3 %). A valid version needs spatial
alignment of hypothesis text to GT regions.

**PRUNED: slope-tolerant grouping + rotated-rectangle crops.** Built,
oracle-clean, unit-tested — and null. The mechanism is genuinely present
(measured on the box field: **26/45 clips ≥ 1° skew, up to 4°**, estimator
fires on 33/45), the correction applies, and CORD CER moved **27.4 → 27.3 %**.
Tilt exists, gets corrected, and is not the loss. The §8.7 visual diagnosis
was wrong; reverted rather than kept as dead weight.

**Two instrument defects caught, both of which would have produced false
claims:**

- the CORD holdout is files `cord-015…059`; mapping it to `cord-000…044`
  scored every receipt against a DIFFERENT receipt's GT and produced 104 %
  and 1263 % CER. Cross-validating a new scorer against the known ledger
  number (27.15 %) is what caught it — do that first, always.
- a "+0.53 pp frames regression" was a BASELINE CONFIG MISMATCH: the
  1.602 % baseline is measured at `FFAI_DET_SCALE=0.5` (what LIVE uses),
  the gate ran at the adaptive default. At matched settings the build
  reproduces 1.602 % exactly. A gate must pin the same knobs its baseline
  was measured with.
- and the CRAFT oracle failed correctly: adaptive scaling had silently
  rescaled its 640×640 fixture to 2×, so the oracle had stopped testing the
  network. It now pins its scale explicitly (`craft_input_scaled`) — a stage
  oracle tests the NET, never the surrounding policy.

**The surviving hypothesis, with evidence.** Our error is
INSERTION-DOMINATED: sorting scatters junk tokens among correct ones (hence
bag > seq for us and the reverse for paddle), while detection recall is high
and precision is high but our box COUNT runs ~2× paddle's on the same image
(23 rects vs 11 polys on cord-055). That is over-segmentation producing
fragment reads. The next brick is line-assembly/merge policy — measured as
token-count and insertion/deletion ratios against paddle, not by eye.

### 8.9 Error composition — the over-segmentation hypothesis, refuted

§8.8 closed by naming "insertion-dominated error from over-segmentation" as
the next brick. Measured by Levenshtein BACKTRACE (word-level ins/del/sub
against the same GT), it is wrong, and the reasoning behind it was wrong in
three separate ways worth recording so they are not repeated:

| | ins | del | sub | hits | ins % | sub % |
|---|---:|---:|---:|---:|---:|---:|
| ours (craft-crnn) | 123 | **103** | 374 | 550 | 20 % | **62 %** |
| paddle | 46 | **103** | 192 | 732 | 13 % | 56 % |

- **Substitutions dominate (62 %)**: we MISREAD text we detected and
  cropped correctly. Insertions are a minor term.
- **Word counts: ours 1.02x GT, paddle 0.94x.** We emit the right AMOUNT of
  text — there is no junk bloat.
- **Deletions are identical, 103 vs 103.** Both engines miss the same text
  (CORD's blurred regions), which independently confirms §8.8's coverage
  result and closes the recall question.
- box-count ratio vs word-count ratio correlates **+0.17** — over-
  segmentation does not predict output bloat; the splitter recovers.

**Why the hypothesis was wrong.** (1) "bag > seq implies insertions" is
BACKWARDS — substitutions that change a word's sort position inflate bag
CER, while pure insertions cost the same either way. (2) The "2x box count"
cited as evidence is the word-vs-line GRANULARITY artifact this same plan
had already debunked in §8.8. (3) The "90 % of error is not recognition"
framing compared PARSeq's stage quality (1.5 %) against craft-crnn's
pipeline number (27.3 %) — two different engines.

**Where the gap actually is.** Normalizing each pipeline against its OWN
recognizer's true-crop score: ours degrades 10.9 -> 27.3 % (2.5x), paddle
3.0 -> 15.6 % (5.2x). Our detection and cropping are proportionally BETTER
than theirs; the whole pipeline gap (1.75x) is smaller than the raw
recognizer gap (3.6x). **The CRNN is the deficit, not segmentation.**

**Next brick, evidence-backed:** feed CRAFT's word boxes to PARSeq
DIRECTLY. We already own a recognizer that beats paddle's (1.5 % vs 3.0 %
on identical crops); it reads 33.4 % in-pipeline only because the ink-gap
splitter re-derives word boxes badly on photographs — and the census says
that re-derivation is unnecessary: **1230 CRAFT boxes against 1027 GT
words**. Deleting the line-merge-then-resplit round trip is simpler than
what exists AND the only route that passes paddle rather than approaching
it.

### 8.10 The direct CRAFT-box feed — 12.5 points, and a content sign-flip

§8.9's prediction, tested: stop re-deriving word boundaries from pixels and
hand PARSeq the boxes CRAFT already produced.

| Config | CORD (real photos) | render (synthetic) |
|---|---:|---:|
| **parseq + CRAFT boxes DIRECT** | **21.70 %** | 0.673 % |
| parseq + ink-gap split (old) | 34.16 % | **0.149 %** |
| craft-crnn (standing default) | 27.42 % | 0.710 % |
| paddleocr-mobile | 15.6 % | 0.019 % |

**The direct feed cuts real-photo CER by 12.5 points (36 % relative).** The
census predicted exactly this: 1230 CRAFT boxes against 1027 GT words means
the components already ARE words, and re-deriving them by ink projection
could only lose information.

**CORRECTION TO THE CORRECTION (2026-07-30).** This section briefly claimed
the render sign-flip "does not reproduce" and the splitter was deleted on
that basis. **Both the claim and the deletion were wrong**, caused by a
STALE BINARY (§8.12): `cargo build` had failed with "Access is denied" while
a background run held `ffai.exe` open, and every measurement for the next
hour ran an hour-old binary. Ground truth on a verified-fresh build:

| corpus | ink-split | CRAFT direct |
|---|---:|---:|
| render (synthetic print) | **0.149 %** | 0.673 % |
| CORD (real photographs) | 34.16 % | **21.70 %** |

The flip is genuinely TWO-SIDED — 4.5x one way, 1.6x the other — so the
dispatcher is earned, the splitter is restored, and the deletion is reverted.

A separate, REAL finding stands from the same period: PARSeq is 3.3x worse
than the CRNN on HUD/screen text (frames 5.339 % vs 1.602 %) and slower, so
`craft-crnn` KEEPS the default seat and `craft-parseq` is the documented
photograph engine. No engine wins everywhere; the toolkit answer is content
dispatch, not a ranking.

Standing gap to paddle on real photos: **21.70 % vs 15.6 %**, down from
27.4 %. The remaining deficit is recognition on our own crops, and the
next lever is the crop GEOMETRY those boxes get padded with — the pads
were tuned on synthetic render and have never been swept on photographs.

### 8.11 Content dispatch — three sign-flips, one signal, and a pad sweep that found nothing

**The crop pads were exonerated, not tuned.** They had never been swept on
photographs, so they were the standing suspect. Nine configurations on
CORD's TRAIN split (holdout untouched, so the 21.7 % claim stays clean):

| pad_x \ pad_y | 0.04 | 0.12 | 0.28 |
|---|---:|---:|---:|
| 0.02 | 17.53 | 17.66 | 21.27 |
| **0.10** | **17.15** | **17.15** *(shipping value)* | 17.47 |
| 0.20 | 19.43 | 17.59 | 18.54 |

The optimum is a TIE with the value already shipping, and both extremes
degrade (too tight AND too loose cost 2–4 points), so the surface has a real
interior optimum that we were already sitting on. The synthetic-tuned pads
transfer to photographs unchanged. Recorded as a negative result; no change
made. Pads are NOT a dispatch axis.

**Dispatch, however, is now earned three times over.** Every strategy choice
in this component has flipped sign by content:

| Decision | Rendered/screen | Photographs |
|---|---|---|
| engine | `craft-crnn` — frames 1.60 % vs 5.34 % | `craft-parseq` — CORD 21.7 % vs 27.4 % |
| word segmentation | ink-gap projection — 0.15 % vs 0.67 % | CRAFT boxes direct — 21.7 % vs 34.2 % |
| recognition (identical crops) | CRNN | PARSeq 1.5 %, beating paddle's 3.0 % |

**The signal ([`content.rs`]): adjacent-pixel exact-equality.** Measured
across all four corpora:

| corpus | flatness |
|---|---|
| render (synthetic print) | 0.881 – 0.943 |
| frames (synthetic HUD) | 0.974 – 0.995 |
| capture (GDI ClearType) | 0.974 |
| CORD (real photographs) | 0.103 – 0.507 |

A **0.37-wide EMPTY band** separates the classes, so the 0.70 threshold is
not fitted — it is the middle of a gap. The mechanism is physical rather
than statistical: shot noise perturbs every photosite independently, so a
sensor essentially never emits large exactly-flat regions and a renderer
emits little else. That is why this dispatch is expected to hold on content
neither corpus contains.

Word segmentation dispatches on it per image; `FFAI_CONTENT=rendered|photo`
overrides. The ENGINE choice stays user-facing (`--engine craft-parseq`),
matching the ffmpeg model this toolkit is built on: lineages are codecs you
select, strategies within a lineage are the engine's business.

### 8.12 The stale-binary hour — an instrument failure and the guard that ends it

**What happened.** `cargo build` failed with `Access is denied` because a
background measurement still held `ffai.exe` open. The failure scrolled past a
`grep -E "^error"` filter, and for roughly an hour EVERY measurement ran a
binary built at 09:29 while the source said otherwise. It produced four
mutually contradictory numbers, one wrong refutation, and one deletion of a
working feature.

**The tell, and why I missed it.** CER is DETERMINISTIC — same binary, same
input, same number. When 0.149 % became 0.673 % on unchanged code, that was
proof the binary differed, not evidence the earlier number was an artifact.
I reasoned about the code for three exchanges before testing the instrument.
The six-whys rule exists for exactly this: *if a number contradicts one you
just took, suspect the artifact before the code* — and *run depth 6 first*.

**What it cost, and what recovered it.** I declared my own committed
measurement an artifact and deleted the ink-gap splitter. Ground truth on a
verified-fresh build:

| corpus | ink-split | CRAFT direct |
|---|---:|---:|
| render | **0.149 %** | 0.673 % |
| CORD | 34.16 % | **21.70 %** |

A genuine two-sided flip. The splitter is restored and the dispatcher earned.
Recovery came only from re-running a refutation — the skill's asymmetry rule
made concrete: a wrong KEEP faces the next gate, a wrong REFUTE is permanent.

**A second defect the same recovery surfaced:** `parseq_pads()` survived in
`lib.rs` but lost its call site in `engine.rs` during the stash/checkout
dance — a dead public function, and silently dead sweep knobs. Re-wired; the
pad sweep then reproduced its original numbers exactly (17.66 / 17.15 /
17.59), confirming both the knob and the earlier conclusion that the
synthetic-tuned pads are already optimal on photographs.

**The guard ([`tools/rebuild.sh`]).** Kills stale processes, deletes the
binary, builds, and FAILS if any `.rs` is newer than the artifact. No
measurement without a proven-fresh binary. It is tracked in-repo — it first
lived in a gitignored directory, which would have let the same failure recur
on a fresh clone.

**Full re-verification after the guard**, every contaminated number re-run:

| measurement | verified | prior |
|---|---:|---:|
| render, crnn | 0.710 % | 0.710 % |
| render, parseq (auto-dispatch) | 0.149 % | 0.149 % |
| frames, crnn | 1.602 % | 1.602 % |
| frames, parseq (auto-dispatch) | **5.034 %** | 5.339 % (pre-dispatch) |
| CORD holdout, crnn | 27.42 % | 27.42 % |
| CORD holdout, parseq (auto) | 21.70 % | 21.70 % |

Dispatch is confirmed on BOTH sides — rendered content routes to the
splitter, photographs to the direct feed — and the frames improvement is the
dispatcher working, not a discrepancy.

### 8.13 Crop geometry — the 16-point cause, isolated and partly banked

The last standing hypothesis, tested by isolation rather than argument: cut
the SAME 400 CORD words two ways — from the corpus's ground-truth quads and
from our detector's boxes — and read both with the same recognizer. Any
difference is geometry with every other variable held fixed.

| crops from | parseq CER | exact | crnn CER |
|---|---:|---:|---:|
| CORD ground-truth quads | **1.30 %** | 94 % | 10.95 % |
| our detector's boxes | **17.41 %** | 68 % | 35.26 % |

**A 16-point penalty from box geometry alone**, and it hits both recognizers.
That is the bulk of the remaining pipeline gap to paddle. Matched against the
quads our boxes measure **0.69x their height, 0.82x their width**, shifted
+0.17/+0.18 line-heights, median IoU **0.537**.

**What did NOT work, and why it is worth knowing.** Correcting the boxes by
that measured MEDIAN bias recovered only 1.4 of the 16 points (17.41 ->
15.97 %). The bias is not systematic: an IoU of 0.537 for the SAME word means
boxes are individually wrong — fragmenting and merging — not uniformly
shifted. No constant pad can fix that, which is also why the earlier pad
sweep found wider padding worse.

**PRUNED: reference dilation inside `extract_boxes`.** Porting clovaai's
component dilation regressed everything — CORD 21.70 -> 32.5 %, render
0.710 -> 2.260 %, frames 1.602 -> 5.187 %. Two causes, both instructive:
(1) `cv2.dilate` with a `(1+niter)²` kernel grows by niter/2 per side, not
niter — a 2x over-expansion; (2) more fundamentally, **detection boxes do TWO
jobs** — they define line GROUPING and they define recognition CROPS. Widening
them for the crops broke the grouping, and that cascade swamped the crop win.
Test the chain, not the link.

**KEPT: per-word crop geometry (CORD 21.70 -> 20.9 %, render unchanged).**
The expansion moved to crop time only, so `extract_boxes` and therefore
grouping are untouched, and each word is cut from its OWN vertical extent
rather than the shared line band — a line's tallest word no longer dictates
every crop in it.

**The residual ~15 points is box QUALITY, not padding**, and it has a named
fix that removes the problem rather than tuning it: the staged PP-OCRv5
mobile detector emits word/line polygons directly, so no component
reconstruction is needed at all. Weights, shape map and behavioural oracle
are already on disk (§7.1, `docs/whys/mobiledet-port-notes.md`).

### 8.4 PARSeq-tiny port: oracle PASS; engine variant open with a localized defect

The port landed and is PROVEN at the stage level: ViT-tiny encoder +
one-layer PARSeq decoder on candle, AR-greedy (refine_iters=0 both sides,
recorded), matching the PyTorch reference EXACTLY on the pinned fixture —
including through the ENGINE's own preprocessing path (grayscale + our
bicubic reads the fixture identically, probe in examples/parseq_probe.rs).
Checkpoint facts that would have sunk a paper-figure port: dec_heads=6,
eos=0/bos=95/pad=96, id-1 charset indexing — all pinned by the oracle dump.

`craft-parseq` is REGISTERED but NOT default: on full pages it misreads
with a first-letter-doubling signature (train CER ~20% vs craft-crnn's
0.69%). Localized so far: NOT the AR loop (oracle), NOT preprocessing
(probe), NOT line-level squeeze (word splitter on max(region,affinity)
now cuts correctly — region-only splitting cut words mid-glyph, measured
and fixed), NOT crop pads (swept 0.35/0.25 and 0.12/0.08, both ~20%).
**Refinement pass PORTED (oracle-exact): craft-parseq 20.2% -> 8.9% CER,
the doubling class eliminated.** Residual gap vs craft-crnn (0.69%) is the
projection splitter's word boundaries — the variant's next problem.
**Mobile-det staged to mechanical:** inference weights (op-indexed names,
port-blocking — recorded), TRAINING checkpoint fetched ungated with
structural names (905 tensors, both shape-mapped, manifests pinned), and a
behavioral oracle fixture (polys+texts, mkldnn-off reference). The candle
port (LCNetV3 rep-fusion + RSEFPN + DBHead) is a CRAFT-scale session with
every input now on disk.

RESOLVED by the crop-dump instrument (FFAI_DUMP_CROPS) + reference
replay: the dumped crops are PRISTINE (verified visually), and the
reference PyTorch model reproduces the same error class on the identical
crops ('boats' -> 'ooats') under AR-greedy refine_iters=0. The defect is
AR-greedy fragility on this crisp-UI-font rendering — a distribution the
scene-text model wasn't trained on — and production PARSeq masks exactly
this slip class with its iterative refinement pass. **The fix is porting
refine_iters=1** (the cloze-mask refinement step), the named next brick
for craft-parseq; craft-crnn stays the default engine until the variant
beats it on the corpus.

---

### 8.14 The mobile-det port — exact to paddle, and two instrument failures on the way

The named fix for §8.13's residual: DBNet emits text regions directly, so
nothing has to reassemble character components into boxes. `mobiledet.rs`
now loads PP-OCRv5 mobile-det (4.7 MB against CRAFT's VGG16) and reproduces
paddle's own exported program:

| check | result |
|---|---|
| probability map vs paddle, pinned 256x256 page crop | max abs 8.3e-05 |
| binarised agreement at threshold 0.3 | **0 / 65 536 pixels disagree** |
| DB postprocess boxes vs paddle's `DBPostProcess` | 4 / 4, IoU > 0.99 |
| box scores, axis-aligned quads | exact, identical pixel counts |

Two things nearly shipped wrong, and neither was a coding error.

**The fusion's self-check could not fail.** `carmenta_mobiledet_fuse.py`
verified its branch sum against its fused conv and reported 4.88e-04 — while
silently dropping all **19 identity BatchNorm branches**, because a check
assembled from the parts it collected cannot detect a part it never
collected. The number looked like evidence and was a tautology. Verification
now runs the whole fused model against paddle's exported program and refuses
to write on disagreement. Restated for the log: *a self-consistency check is
not evidence.*

**Two undocumented constants, resolved by search rather than by reading.**
The det variant of PP-LCNetV3 is not vendored on this box, and the two
sources that describe its parts disagree. So the numpy reference searched the
space against the oracle:

| choice | measured | the plausible reading | cost of the wrong one |
|---|---|---|---|
| backbone SE hardsigmoid slope | **1/6** | 0.2 | 0.55 logit |
| neck SE hardsigmoid slope | **0.2** | 0.2 | 0.52 logit |
| RSELayer residual shortcut | **on** | — | 13.3 logit |

They genuinely differ between backbone and neck. "0.2 everywhere" — the
reading the PaddleOCR source supports — would have produced a detector that
loads, runs, and degrades every box.

A third trap is now unrepresentable rather than commented: `LearnableRepLayer`
builds its post-activation unconditionally but applies it only when
stride != 2, so the checkpoint carries parameters for stride-2 layers that
must never be used. The fused file omits them. (The same asymmetry is the
stride oracle: an identity branch exists iff `in == out && stride == 1`, and a
depthwise conv always has `in == out`, so the 19 identity branches pin every
depthwise stride exactly. Nothing about the architecture was guessed.)

### 8.15 The 17x input — a depth-6 finding that inverted the first result

First end-to-end run of `mobiledet-crnn` on a CORD receipt returned the
labels and **not one number**. The boxes explained it: one 1903x1781 blob
swallowing the whole receipt.

The instinct was to suspect the port. The measurement said otherwise —
paddle's OWN probability map at that input size contains the same merged
component, and paddle's own `DBPostProcess` returns the same giant box. The
detector was never wrong. **The input was 17x too small.**

`inference.yml` says `resize_long: 960`, which reads as "scale the long side
to 960". It is not what the reference does. The effective policy is a
**minimum** short side with a 4000-px cap: images larger than the floor pass
through untouched. A 2376x4224 receipt reaches the reference detector at
**2240x4000**, not 544x960. The proof is in the reference's own log —
`Resized image size (2376x4224) exceeds max_side_limit of 4000` — which can
only print if the ratio was still 1.0 when the cap applied.

With the policy corrected, the same receipt yields 11 boxes and every number
appears. Two consequences worth carrying forward:

- **The speed comparison was never like-for-like.** PaddleOCR reads a page in
  ~12 s partly because it runs detection at 4000 px. Any future speed claim
  has to state the detector input size on both sides, exactly as the `-nt`
  finding forced token counts into the Mercury comparison.
- Detector input resolution is now the dominant quality/speed knob
  (`FFAI_DET_MIN_SIDE`, `FFAI_DET_MAX_SIDE`), and it is a sweep, not a
  constant.

This is the fourth consecutive campaign descent to terminate at depth 6 on a
configuration difference rather than a defect.

### 8.16 Mobile-det measured: speed and footprint won decisively, quality LOST

The port was predicted to "close both open fronts at once". On CORD it closed
one and reopened the other, and the prediction was wrong.

Ledger `bench ocr --corpus carmenta-cord-v1 --engine mobiledet-crnn`:

| CORD holdout, 45 clips | mobiledet-crnn | craft-crnn | paddleocr-mobile |
|---|---:|---:|---:|
| CER | **37.30 %** | 27.27 % | 15.62 % |
| pages/s warm | **0.33** | 0.11 | 0.04 |
| steady MiB | **649** | 2055 | 733 |
| gates | quality FAIL, **speed PASS, footprint PASS** | all FAIL | — |

**3x faster than CRAFT and 3.2x leaner**, and the first Carmenta configuration
to pass the speed and footprint gates on real photographs at all. Also the
first to run leaner than PaddleOCR (0.89x its steady memory). That part of the
thesis — a 4.7 MB detector against a VGG16 — is confirmed.

Quality went the other way, and consistently. Scored on one ad-hoc scorer so
the arms are comparable (it reads ~1 pp below the ledger; used for A/B only,
never quoted as a claim):

| engine | CER, same scorer |
|---|---:|
| craft-parseq | **21.96 %** |
| craft-crnn | ~26-27 % |
| mobiledet-crnn | 36.15 % |
| mobiledet-parseq | 38.13 % |

**One cause found and banked.** DBNet's boxes arrive already unclipped, and the
crop-time pad expanded them a second time. DB's offset is
`area * unclip / perimeter`, which on a wide line is huge *vertically*: a
1900x90 line grows 64 px on every side — 72 % of its own height — before
`PAD_Y = 0.35` adds another 35 %. Setting the mobile-det crop pads to zero
(`FFAI_MDET_PAD_X/_Y`, default 0) took **36.15 % -> 33.08 %**. Real, 3 points,
and not nearly enough.

**Why the rest, and why it is not a defect.** §8.13 diagnosed the CORD gap as
box *quality* and named this port as the fix. That diagnosis was half right:
mobile-det gives better LINE boxes and no word boxes at all. CORD receipts put
a label and its amount in two widely separated columns, and DBNet merges them
into one region — so the line recognizer reads across the gap, and PARSeq must
recover word boundaries with the ink-gap splitter, the strategy §8.11 already
measured at **34.16 %** on photographs against 21.70 % for CRAFT's own boxes.
mobiledet-parseq's 38.13 % is that same number reappearing.

So CRAFT's character-level components are not merely a clumsy route to word
boxes — on two-column receipts they are carrying real structural information
that a line detector discards. The §8.13 residual is **not** closed, and the
hypothesis that named this port as its fix is refuted for the CORD class.

**What this changes.** Mobile-det is not a replacement; it is a second lineage
with an opposite trade, and the engine registry already expresses that — four
engines, two detectors x two recognizers, chosen per corpus class rather than
ranked. The open questions it leaves, in order:

1. Does mobile-det WIN on documents and screens, where text is single-column
   and the line IS the unit? `carmenta-doclaynet-v1` (§6.2) and the frames
   corpus can answer that, and neither has been run.
2. Would DBNet's boxes plus a word split from CRAFT's affinity map beat both?
   That composes the two lineages rather than choosing between them.
3. The unclip ratio is pinned at the reference's 1.5 and has never been swept
   for OUR crop geometry, which is not the reference's.

Recorded as a loss, in the shape the ledger requires: the claim that this brick
closes the quality front is withdrawn, and the speed/footprint claim it does
support is now measured.

### 8.17 The detector sign-flip, and a corpus that cannot gate what it was pinned for

§8.16 left one question: does mobile-det win where the LINE is the unit? Run
across every engine and corpus on one fixed scorer (it reads ~1 pp below the
ledger; used for A/B only, never quoted as a claim):

| CER % | frames (HUD) | CORD (receipts) | DocLayNet (documents) |
|---|---:|---:|---:|
| craft-crnn | 1.80 | **~27** | 47.70 |
| **mobiledet-crnn** | **1.71** | 37.30 | 48.70 |
| craft-parseq | 2.84 | **21.96** | 55.94 |
| mobiledet-parseq | 10.91 | 38.13 | 81.52 |

**The sign-flip is real and it is the detector's.** Mobile-det wins the frames
class outright — and does it while running 3x faster — then loses receipts by
ten points. Single-column screen text is exactly the case where a line region
IS the unit, and two-column receipts are exactly the case where it is not.
That is a third measured dispatch axis, alongside §8.11's content flip.

**mobiledet-parseq is dominated everywhere and should not be promoted.** Its
losses track the ink-gap splitter, not the detector: PARSeq needs word boxes,
DBNet supplies none, and the fallback splitter was already measured at 34 % on
photographs. The pairing is a mistake the registry currently lets you make.

#### The document corpus cannot gate absolute quality — measured, not assumed

DocLayNet came back at 47.70 % / 48.70 %, and two detectors that disagree
about everything else landing within one point is a tell. Three probes:

1. **Reading order — REFUTED as the cause.** Scoring each page twice, once
   as-is and once with both sides' lines sorted (which destroys order
   information symmetrically), moves 44.19 % to 40.99 %. Order is worth
   **3.21 pp** of a 44-point hole.
2. **The reference fails too.** PaddleOCR on the same 12 pages: **53.94 %**.
   When every engine lands at 50 %, the instrument is the suspect. (Our 56.83 %
   on that same split is the honest comparison — an earlier read of "we beat
   paddle by 6" was our *holdout* against paddle's *train*, and did not
   survive matching the splits.)
3. **The pixels explain it.** DocLayNet ships full pages at 1025x1025, so a
   body-text line is 9-12 px tall — below what any OCR reads reliably.

**Upscaling refuted as a fix.** Sweeping the detector input across 3.5x
(`FFAI_DET_MIN_SIDE` 736 -> 1536 -> 2048 -> 2560) gives 57.22 / 57.68 / 57.53 /
59.12 %. Flat, then worse. Resampling does not create information a 9-px glyph
never had.

So the corpus pinned in §6.2 is sound for the gate DocLayNet was *built* for —
layout regions and reading order, whose ground truth is exact at any raster
resolution — and **cannot support M-C3's end-to-end CER gate**. It remains
valid as a *relative* comparison on matched pixels, where paddle currently
leads by ~3 points. M-C3's CER gate needs a higher-resolution document source;
DocLayNet's own release does not ship one, so that is open work.

Recorded rather than quietly re-scoped: pinning a corpus is not the same as
validating it, and the validation here cost four measurements and changed the
milestone's plan.

### 8.18 LIVE with mobile-det — 18 % lower p95, and a default left alone

§8.17's frames win is worth something only if it survives into LIVE, where the
speed gate is the one that has always been tightest. Same harness, same
screencast, best-of-3, each detector at its own best setting:

| LIVE, per band call | CRAFT | **mobile-det** |
|---|---:|---:|
| p50 | 156 ms | **114 ms** |
| p95 | 241 ms | **198 ms** |
| best-of-3 p95 range | [241, 247, 266] | **[198, 203, 206]** |
| CER on change frames | **1.74 %** | 1.83 % |
| churn | 0 / 156 | 0 / 156 |
| four gates | PASS | PASS |

**The p95 ranges do not overlap**, so the 18 % is a result and not a coin flip
— the best-of-N rule this campaign runs on. Against per-frame Tesseract the
margin widens from 1.33x to 1.6x.

**The default stays CRAFT anyway.** LIVE's own quality metric moved the wrong
way (+0.09 pp), and it passes only because the band is +0.25 pp. Flipping a
default on a metric that regressed — even inside its band — is how a campaign
accumulates quiet losses; the frames *batch* holdout says mobile-det is the
better reader there (1.71 % vs 1.80 %), and the disagreement between the two
measurements is itself unexplained. `FFAI_LIVE_DET=mobiledet` exposes it, and
the sweep that resolves the disagreement is what would earn the default.

This also retires a number: the full-suite run reported LIVE p95 **838 ms vs
tesseract 841 ms**, a 3 ms pass that read as alarming. On a quiet box the same
gate is 241 vs 321. That entire figure was contention from concurrent work on
this machine, and it is exactly the failure the interleaved-A/B discipline
exists to prevent — sequential arms sampling different machines.

### 8.19 Three hunts: one big win, one mechanism, one clean refutation

#### Unclip belongs to the RECOGNIZER, not the detector — 8.6 points

The reference pins a single `unclip_ratio = 1.5` and we had inherited it. The
hypothesis going in was that 1.5 over-expands for our axis-aligned crops.
**Refuted, and inverted.** Swept on the CORD train split:

| unclip | 0.0 | 0.5 | 0.8 | 1.1 | 1.5 | 1.9 | 2.3 | 2.8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mobiledet-**crnn** | 70.3 | 47.4 | 41.6 | 37.0 | **32.0** | 31.8 | 32.6 | 35.2 |
| mobiledet-**parseq** | 45.6 | 32.8 | **32.3** | 36.5 | 41.3 | 45.4 | — | — |

Two optima, on the same detector, ten values apart. CRNN's sits at the
reference's 1.5 (1.9 is 0.26 pp better — inside the noise of 15 clips and not
worth moving a default for); **PARSeq's sits at 0.8**, nine points better than
the value it had been given. Both ranges are now enclosed rather than
open-ended, which is what caught the crnn optimum being effectively at 1.5.

The mechanism is the crop each recognizer consumes: CRNN reads a whole line and
wants context around it, while PARSeq reads WORDS recovered by an ink-gap
projection *inside* that box — and a loose box fills the gaps with background
until the projection stops finding them.

Confirmed on a second corpus class before shipping, not assumed to generalise:
on frames, PARSeq reads 1.83 % at 0.8 against 5.42 % at 1.5. Landed as
`UNCLIP_LINE` / `UNCLIP_WORD`, and measured on **holdout**:

| mobiledet-parseq | before | after |
|---|---:|---:|
| frames | 10.91 % | **2.29 %** |
| CORD | 38.13 % | **34.60 %** |

§8.17 called mobiledet-parseq "dominated everywhere and not to be promoted".
On frames that is now false: 2.29 % beats craft-parseq's 2.84 %. The pairing
was not bad; it was mis-parameterised, by a constant copied from a reference
whose recognizer is not ours.

#### The frames batch-vs-LIVE disagreement — mechanism found

§8.18 left mobile-det better in batch (1.71 % vs 1.80 %) and worse in LIVE
(1.83 % vs 1.74 %), unexplained. The harness prints the answer:

| | band coverage of detected boxes | calibration cost |
|---|---:|---:|
| CRAFT | **86.7 %** | 2132 ms |
| mobile-det | 69.8 % | 1093 ms |

LIVE recognises auto-ROI **bands**, not frames, and mobile-det's calibration
produces bands covering only 69.8 % of the text it detected. Roughly 30 % of
the text sits outside every band and is never re-read in steady state. **The
regression is missed text, not misread text** — batch mode reads the whole
frame, so the same detector looks better there. `calibrate_bands` unions line
y-extents with a +-8 px tolerance, which was built for CRAFT's many small word
boxes and does not fit DBNet's few tall ones. Band construction, not the
detector, is the open work — and it is worth doing, because the same run also
halves calibration cost.

#### Composing the lineages — refuted

`composed-*` runs BOTH detectors: CRAFT supplies the word boxes, DBNet's
regions decide which words share a line. It isolates whether `group_lines`'
heuristic or CRAFT's box geometry is what loses.

| CER % holdout | frames | CORD |
|---|---:|---:|
| craft-parseq | 2.84 | **21.96** |
| **composed-parseq** | 2.84 | 27.22 |
| mobiledet-parseq | **2.29** | 34.60 |
| composed-crnn | — | 30.02 |

**Neither.** On frames composed ties craft-parseq to the second decimal, so
DBNet grouping and `group_lines` are indistinguishable there; on receipts
composed is 5.3 points WORSE than craft-parseq, so DBNet's regions are the
inferior grouping. `group_lines` was never the binding constraint, and paying
two detector forwards buys nothing on either class.

Reverted as measured-worse (not as within-noise): `composed-*` is unregistered
and does not ship. `CraftCrnn::new_composed` stays so the probe reproduces.

#### Where the dispatch stands

| corpus class | best engine | CER |
|---|---|---:|
| frames / screens | **mobiledet-crnn** | **1.71 %** |
| receipts / photos | **craft-parseq** | **21.96 %** |

Two lineages, opposite trades, both now parameterised for the recognizer they
actually feed.

### 8.20 calibrate_bands — the fix landed, and §8.19's stated mechanism was wrong

**Correction first.** §8.19 reported that LIVE's mobile-det regression was
missed text: auto-ROI bands covering 86.7 % of CRAFT's boxes against 69.8 % of
mobile-det's, so "~30 % of the text is never re-read". **That mechanism is
wrong.** The harvest scores coverage by STRICT containment — the whole later
box must sit inside a calibrated band — while `calibrate_bands` assigns lines
to bands by their CENTRE. Measured on the rule the system actually uses:

| | strict containment | CENTRE containment |
|---|---:|---:|
| CRAFT | 86.7 % | **100.0 %** |
| mobile-det | 69.8 % | **100.0 %** |

Every box's centre lands in a band, for both detectors. Six bands each, in the
right places. No text was being missed, and the number that said otherwise was
an observe-only ceiling estimate measured with a stricter rule than the code
obeys. A metric that does not match the code path is not evidence about the
code path.

**The right place to look was still the pad, for a different reason.** Bands
are what the recognizer is CROPPED to, so a band that is tight around its line
clips ascenders. Mobile-det's bands measured tighter (median 45 px vs CRAFT's
50) around boxes of the same size, because DBNet's few line-level boxes union
tightly where CRAFT's many word boxes union generously — and the pad absorbing
that was a flat 8 px, tuned against CRAFT.

Made proportional to the band's own height and swept (CER on change frames):

| frac | 0.0 | 0.12 | 0.25 | 0.29 | **0.33** | 0.36 | 0.40 | 0.50 | 0.65 |
|---|---|---|---|---|---|---|---|---|---|
| CRAFT | 1.74 | 1.74 | 1.77 | 1.74 | **1.43** | 1.47 | 1.54 | 2.47 | 3.42 |
| mobile-det | 1.83 | 1.83 | 1.83 | — | **1.80** | — | 2.21 | — | — |

A basin rather than a spike — three values inside, monotone rise on both sides
— and confirmed on the second detector rather than assumed to generalise.

**Correction (§8.21): the row above was measured under `FFAI_DET_SCALE=0.5`,
which is not the shipped default.** Re-run at the adaptive default the pad
still wins and the basin sits in the same place, but the margin is smaller:

| frac (adaptive default) | 0.0 | 0.25 | **0.33** |
|---|---:|---:|---:|
| CRAFT | 1.81 | 1.82 | **1.67** |

So the honest figure for LIVE's quality gate is **1.81 % -> 1.67 %**, a 7.7 %
relative improvement, not the 18 % the pinned-scale pair implied. The direction,
the basin's location and the cliff all reproduce; only the size shrinks. Filed
as a correction rather than an edit because quoting a gate number from a
non-default configuration is the same error class as §8.15's reference flag.

The cliff past ~0.45 is mechanical: bands are clamped to the midpoint of the
gap to their neighbour, so a large fraction eventually grows a crop halfway
into the next line and the CRNN reads two lines as one. Its position therefore
depends on the corpus's line spacing; 0.33 keeps ~0.12 of margin here, and
`FFAI_BAND_PAD` is the knob for a denser one.

#### Tried and reverted: a shape-aware detector floor

Mobile-det's remaining LIVE deficit had an obvious suspect: LIVE hands the
detector band STRIPS (1280x45 is an aspect of 28), and a minimum-SHORT-side
floor of 736 demands a 16x upscale that lands straight on the 4000 px cap. So
the floor was made shape-aware — a smaller value for strips than for pages.

**Measured inert.** LIVE stayed at 1.83 % while a globally lower floor reaches
1.62 %, which places the effect in the full-frame CALIBRATION call rather than
in the strips. Reverted rather than kept as a plausible knob that buys nothing.

#### Left open, and stated as open

A global `FFAI_DET_MIN_SIDE` of 48, 96 or 320 all give mobile-det **1.62 %** in
LIVE against **1.83 %** at the 736 default — three settings agreeing exactly, so
the effect is real. Batch wants the opposite: on full frames 736 reads 1.71 %
and 320 reads 1.80 %. Same detector, same corpus, opposite optima, and the
shape probe rules out the strips. The remaining difference is what the boxes
are USED for — final crops in batch, band GEOMETRY in LIVE's calibration — but
that is a hypothesis, not a measurement, and it is recorded as one. Plumbing a
per-call floor needs an `OcrOptions` field, which is a shared-crate change.

### 8.21 The ledger sweep — every engine, every corpus, one configuration

The campaign had drifted ahead of its own evidence: §8.19 and §8.20 were argued
from `.tools-bench/score_corpus.py`, a scorer explicitly labelled "A/B only,
never quoted as a claim", while the ledger held **zero** lines for
`mobiledet-parseq` and one stale line for `mobiledet-crnn`. Twelve runs later
it does not. One configuration throughout — adaptive detection default,
best-of-1, PaddleOCR-mobile as the single reference — because a per-engine env
makes the arms incomparable, which is exactly how the earlier suite ended up
with craft-crnn at 1.76 % and this one at 1.72 %.

| CER % (pages/s, steady MiB) | frames | render | CORD |
|---|---|---|---|
| craft-crnn | 1.72 (0.11, 1500) | 0.67 (0.10, 1957) | 27.27 (0.05, 2055) |
| craft-parseq | 2.87 (0.13, 1497) | **0.49** (0.08, 1897) | **20.93** (0.09, 1782) |
| **mobiledet-crnn** | **1.70** (**0.64**, **123**) | 0.54 (**0.59**, 210) | 34.08 (0.36, 646) |
| mobiledet-parseq | 2.48 (0.57, **102**) | 0.68 (0.47, **100**) | 35.17 (0.36, 631) |
| paddleocr-mobile | 1.31 (0.26, 739) | 0.02 (0.14, 736) | 15.62 (0.16, 810) |

**Mobile-det passes the speed AND footprint gates on all three corpora against
PaddleOCR** — the first Carmenta configuration to do so anywhere, and it does
it everywhere. On frames it runs **5.8x faster than CRAFT at 1/12th its steady
memory** (123 MiB against 1500), and it is faster and leaner than the reference
itself. Quality is the only gate still open, and only on photographs.

Three things the sweep settled that the ad-hoc scorer could not:

1. **The unclip fix is real and banked.** mobiledet-parseq reads 2.48 % on
   frames where the reference's 1.5 would have put it near 10.9 %.
2. **The pad fix is real but smaller than reported** — see the correction in
   §8.20. Measured at the default rather than at a pinned scale.
3. **The dispatch rule I proposed does not survive.** §8.17 read "Rendered ->
   mobile-det", resting on the frames corpus alone. Render is also Rendered,
   and there craft-parseq wins (0.49 vs mobiledet-crnn's 0.54). No single
   engine takes both Rendered corpora, and the two winners are *opposite*
   pairings. So content class is NOT a sufficient dispatch key.

   What survives is weaker and more useful: mobiledet-crnn is first on frames
   and a 0.05 pp second on render, at ~6x the speed and a fifth of the memory.
   It is never far from best on rendered content and never close on
   photographs. A dispatch built on that is defensible; one built on "Rendered
   -> mobile-det wins" is not, and I had the second one written down.

**Scorer calibration, now that both exist.** The ad-hoc scorer ran high by
0.08–1.03 pp (frames craft-crnn 1.80 vs 1.72; CORD craft-parseq 21.96 vs 20.93;
frames mobiledet-parseq 2.29 vs 2.48 — that one ran LOW). Every A/B conclusion
drawn from it survives the ledger, but no absolute number from it should be
quoted, which is what it was labelled for.

### 8.22 The CORD gap is not a recognition gap — 97 % of it is INSERTIONS

§8.7 listed "blur-filtered CORD GT (corpus honesty first)" as open brick #1 and
it was never done. Doing it changes what the campaign's only open gate is
about.

CORD's receipts are privacy-blurred: store headers and footers are smeared to
illegibility. The ground truth, built from CORD's own `valid_line`
annotations, correctly omits them — but the pixels are still in the image, so
every engine detects those regions and emits something for them. That text has
no counterpart in the reference and scores as pure insertion.

Decomposing the edit distance instead of quoting its total, on 15 CORD holdout
clips (1951 GT chars):

| | CER | substitutions | deletions | **insertions** |
|---|---:|---:|---:|---:|
| craft-parseq | 20.55 % | 3.54 pp | 5.48 pp | **11.53 pp** |
| paddleocr-mobile | 14.61 % | 3.08 pp | 5.74 pp | **5.79 pp** |

**Insertions are 56 % of our error, and 97 % of our gap to PaddleOCR** — 5.74 of
the 5.94 pp that separate us. Our substitutions sit within 0.46 pp of the
reference's and our deletions are *better* than its: we miss less of the real
text than PaddleOCR does.

So the standing story — "full-pipeline photo accuracy trails PaddleOCR" — is
true in the number and wrong in the cause. **Our reading is at parity. We emit
roughly twice as much text that should not have been emitted at all.** The
reference suppresses low-confidence regions and we do not: paddle gates each
box on `box_thresh = 0.6` of DB probability, while CRAFT's component walk keeps
anything clearing a region-score threshold and nothing downstream ever asks the
recognizer how confident it was.

This does not overturn §8.13. That probe matched the SAME words on both sides,
so it measured substitutions, and a 16-point crop-geometry effect on matched
words is consistent with a 0.46 pp substitution gap on whole pages — the
matched-word probe simply cannot see insertions, because it never scores a
region the ground truth does not contain. Two true measurements of different
things; only the full-page decomposition says which one the gate is made of.

**Named next brick: rejection** — and §8.23 measured it and refuted it. The
ceiling quoted here ("~14.8 %, at parity with PaddleOCR") assumed insertions
could be removed for free. They cannot: every mechanism that removes them
removes real text at a similar or faster rate. The decomposition in this
section stands; the prize computed from it did not survive contact.

### 8.23 Rejection — refuted, and the ceiling with it

§8.22 named a prize: our insertions are twice the reference's, so suppress them
and CORD lands at parity. Three levers, swept on the CORD train split, plus two
separability probes. **All negative.**

| lever | best | vs off | what happened |
|---|---|---|---|
| recognition confidence (PARSeq) | 18.10 % @ 0.8 | −0.70 pp | insertions 9.87 -> 8.99; at 0.9 insertions reach 7.22 but deletions go 4.68 -> **9.49** |
| recognition confidence (CRNN) | 21.20 % @ off | — | every threshold made it worse |
| CRAFT detection threshold | 18.54 % @ 0.78 | −0.26 pp | insertions 9.87 -> 6.46 at 0.94, deletions 4.68 -> **13.73** |
| the other detector | 32.03 % | — | mobile-det's insertions are 13.73–18.61 pp, WORSE than CRAFT's 9.87 |

Every one trades insertions for deletions at roughly 1:1. That is the signature
of a constant applied to a content-dependent question, so the next question was
whether any per-page signal separates the two populations at all. Classifying
each emitted line by whether the ground truth contains it:

| signal | in-GT lines | NOT-in-GT lines |
|---|---|---|
| PARSeq confidence, absolute | median 0.997, p10 0.970 | median 0.976, p10 0.853, **p90 0.998** |
| confidence, relative to page median | median +0.000, p10 −0.021 | median −0.005, p10 −0.132 |
| crop sharpness, absolute | median 0.546 | median 0.414 |
| sharpness, relative to page | median 1.015 | median 0.933 |

**Nothing separates them.** A tenth of the bad lines carry confidence 0.998 —
inside the good lines' body, not their tail. An autoregressive decoder handed
an illegible crop returns a fluent word and full confidence, and **its opinion
is not evidence about its input**. Sharpness, which is a property of the pixels
rather than of a model's belief, does better on medians and still overlaps
almost completely.

So the honest verdict: **the insertion gap is real, the ceiling computed from it
is not.** §8.22's "~14.8 %, parity with PaddleOCR" assumed those 5.74 pp could
be removed for free; they cannot be removed at all by any gate tried, and the
number should never have been stated as a prize without a mechanism that
collects it. Recorded as a correction to my own section, one entry later.

`FFAI_REJECT` stays as a knob and stays **off** by default. A 0.70 pp gain on 15
train clips, bought by trading one error class for another, is exactly the
fragile tuning this campaign is supposed to refuse.

**What this leaves.** Paddle emits half our junk and no gate explains how, which
means the difference is upstream of anything a threshold can see — most likely
in what its detector proposes at all on smeared regions, since its recogniser
is measurably worse than ours on matched crops (1.5 % vs 3.0 %, §Status). The
next probe is not another threshold: it is a direct comparison of the two
detectors' box SETS on a blurred region, which is a different question from the
box GEOMETRY §8.13 measured.

### 8.24 Quad crops — ceiling measured at 1.26 pp, then pruned on structure

We compute DBNet's rotated quad and discard it for its axis-aligned bounds, and
the ground-truth quads that read at 1.30 % in §8.13 *are* quads. §8.8 pruned a
deskew fix, but that was line GROUPING; this is crop SHAPE — a different lever
wearing a similar name, which is how levers get wrongly refuted. So: ceiling
first, cost second, exactly as the campaign rule says.

Same 300 CORD ground-truth words, cut two ways, read by the same recogniser:

| crop | CER |
|---|---:|
| axis-aligned bounds | 14.91 % |
| **perspective warp of the quad** | **13.65 %** |

**1.26 pp, and it is real.** It is also concentrated: CORD's words are
essentially upright — tilt median **0.00 deg**, p90 4.21, max 8.69 — so a
warp and an AABB are the same rectangle for the typical word, and the whole
gain comes from a ~10 % tilted tail.

**Pruned anyway, on structure rather than on size.** The engine that would use
it is the wrong one. Our best CORD engine is craft-parseq at 20.93 %, and CRAFT
emits axis-aligned character components — **there is no quad to warp**. The
detector that does emit quads is mobile-det, which sits 13 points behind on
this corpus, so the 1.26 pp would be spent making the loser slightly less far
behind. Building it now buys nothing on the number the gate reads.

Where it *should* return is skewed DOCUMENT scans, where a line detector is the
right tool and page skew is real rather than a 10 % tail. The corpus that could
fail it does not exist yet: `carmenta-doclaynet-v1` is rendered from PDFs at
zero skew (§8.17). So this is recorded as a measured, quantified lever waiting
on M-C3's corpus, which is this plan's own rule — nothing lands without a
corpus that can fail it.

### 8.25 The footprint soak — M-C2's last gate, and it was never a pass before

M-C2 has been reported as green with `footprint SKIP` since it closed. A
skipped gate is never a pass — this plan says so in its own §8 preamble — so
the milestone was never actually complete. Two 30-minute soaks, cycling the
screencast corpus through a `LiveSession` and sampling resident memory:

| detector | frames | first-window | last-window | ratio | gate (<= 1.10) |
|---|---:|---:|---:|---:|---|
| mobile-det | 70 716 | 285 MiB | 309 MiB | 1.085 | **PASS** |
| CRAFT | 57 483 | 1602 MiB | 1646 MiB | 1.027 | **PASS** |

**All four M-C2 gates are now green for both detectors, on evidence rather than
on an exemption.** Mobile-det also pushed 23 % more frames through the same
wall clock at a fifth of the memory.

One honest caveat rather than a clean win: **mobile-det passes at 1.085 against
a 1.10 bar** — 24 MiB of growth across 70k frames, and only 1.5 % of margin.
CRAFT's 1.027 is comfortable; mobile-det's is not. Thirty minutes cannot
distinguish a plateau from a slow leak, and the gate's shape means a longer run
is the only thing that can. Recorded as a pass with a thin margin, which is
what it is, and a longer soak is named work rather than an assumed formality.

### 8.26 carmenta-doc-v1 — the corpus M-C3 was actually blocked on

Two prunes in a row pointed at the same thing. §8.24 measured a 1.26 pp prize
for perspective-warped crops and had to park it because no corpus had any skew
to fail on. §8.17 pinned DocLayNet and then measured that it cannot gate what
it was pinned for — 1025x1025 full pages put a body line at 9-12 px, every
engine lands near 50 % (PaddleOCR at 53.94 %), and upscaling was refuted across
a 3.5x sweep. The blocker was never the code.

So the generator that already renders `carmenta-render` and `carmenta-frames`
now renders documents too: **32 pages across 8 documents at 1700x2200** (letter
at ~200 dpi, body text ~28 px). Verified: 0 hash mismatches, 8 train / 24
holdout, 3775 ground-truth chars per page, 550 regions across six classes.

Rendering rather than collecting buys four things no public set gave us:

| | why it matters |
|---|---|
| **exact reading order** | two-column pages lay out the left column fully, then the right — the order a raster-order reader gets wrong, which is the measurement M-C3 exists to make |
| **exact region boxes**, per class, as quads | the layout gate has ground truth instead of a proxy |
| **arbitrary resolution** | the CER gate measures reading, not legibility — DocLayNet's defect, fixed by construction |
| **controlled skew** | half the pages rotate 0.8-3.6 deg, and BOTH splits carry both kinds, so §8.24's parked lever finally has something that can fail it |

Documents split **whole** — none straddles — because M-C4 will group pages by
`doc_id` and a split document leaks the moment it does. Running headers and
page numbers are rendered deliberately: suppressing them across pages is LONG's
job, so the corpus that will test it should contain them.

The text is synthesised from the same fixed lexicon as the other two corpora
rather than drawn from Gutenberg. Deliberate: a downloaded book adds a fetch
gate to a corpus whose whole point is regeneration from a formula, and a
controlled vocabulary keeps out-of-vocabulary surprises from confounding a
measurement about LAYOUT. Regenerating leaves the other three corpora
byte-identical — checked, not assumed.

**And a document-tier reference to compete against.** Every reference so far is
line-level: they answer "did you read the characters". `ppstructure-v3` answers
whether the characters came out IN READING ORDER, having first decided the page
has two columns, a running header and a footer that is not prose — which is the
question M-C3's gate actually asks, and which the plan has named since it was
written. It is matched to `paddleocr-mobile` tier-for-tier and
handicap-for-handicap, its text is taken from PP-Structure's own ordered
`parsing_res_list` rather than re-sorted by us, and a fallback to unordered
text is logged loudly because it would silently demote the comparison back to
line-level. It costs ~170 s per page on this box, which is a fact about the
reference and is recorded rather than hidden.

Two defects of my own on the way, both worth the entry. The manifest shipped
with `class = "document_1col"` against a closed enum, so the harness rejected
the corpus I had just committed. And the first fix appeared not to work because
cargo reported `Finished` while the example binary stayed **24 minutes older
than its source** — the §8.12 failure again, and the same tell: a result that
contradicts a change you just made indicts the artifact before the code.

### 8.27 First read of carmenta-doc — the corpus attributes, and M-C3 has its number

`mobiledet-crnn` on the 24-page holdout, split by the two design variables:

| | upright | skewed |
|---|---:|---:|
| **1-column** | **0.34 %** | 39.59 % |
| **2-column** | **71.20 %** | 73.06 % |

Three things fall straight out, none of which the flat 37 % average could say.

**1. The resolution decision is vindicated.** 0.34 % on upright single-column
pages. The same engine reads DocLayNet at ~49 % because a 9-px line is not
legible (§8.17). The CER gate now measures READING, which is what it was
supposed to measure all along.

**2. Multi-column reading order is the dominant failure, at ~71 points, and it
is not skew.** A two-column page costs 71.20 % with *zero* skew. We emit
regions in raster order, so a two-column page interleaves the columns line by
line and the text comes out shuffled. That is precisely the defect M-C3 exists
to fix, it now has a number, and the number is enormous. Everything Carmenta
has measured until now was single-column by construction, so this failure has
been invisible for the entire campaign.

**3. Skew is a real, separate second cause — 39 points.** On single-column
pages, 0.34 % becomes 39.59 % at 0.8-3.6 degrees. That re-opens §8.24's parked
quad-crop lever on a corpus that can finally fail it: the 1.26 pp ceiling
measured on near-upright CORD words was measuring the wrong population.

Note the interaction, because it disciplines the order of work: skew adds only
~2 points ON TOP of two columns (71.20 -> 73.06). Reading order dominates so
completely that fixing skew first would be almost invisible. Layout first, then
deskew.

**The corpus earned its design in one run.** The first cut confounded columns
with skew — both keyed off `doc % 2` — and produced a real separation
(0.40 % vs 71.61 %) that could not be attributed to either variable. Crossing
them properly turned one impressive number into three actionable ones. That is
the difference between a corpus that measures and a corpus that only scores.

### 8.28 Column-aware reading order — 71 points for a histogram

§8.27 measured a 71-point CER penalty on two-column pages. The token-order
probe attributed **55.75 pp of it to ordering alone**: we read the words
correctly and emitted them interleaved, alternating columns line by line.

LIVE's `calibrate_bands` already finds horizontal text bands by projecting
detected line boxes onto the y-axis. A column gutter is the same operation on
the other axis — a vertical strip no line box crosses — and the boxes are
already computed, so it costs a histogram over them.

| CER % | upright | skewed |
|---|---:|---:|
| 1-column | 0.34 -> **0.34** | 39.59 -> **39.59** |
| 2-column | 71.20 -> **0.32** | 73.06 -> **15.20** |

**Two-column upright goes 71.20 % -> 0.32 %**, indistinguishable from
single-column, and single-column does not move at all. Against Baidu's
Unlimited-OCR at ~103 s per page on a GPU, this is a projection over boxes we
already had.

Three defects on the way, each caught by a measurement rather than by reading
the code, and each worth more than the fix:

1. **The gutter did not exist.** The first implementation found no interior gap
   at all — free runs were `(0,125)` and `(1566,1700)`, the margins. DBNet's
   boxes arrive unclipped by 1.5x (§8.19), which widens every line by ~0.7x its
   own height per side and closes a 60 px gutter completely. The projection now
   runs on ERODED boxes, proportional to height because that is what the unclip
   scales with.
2. **One centred element vetoed the whole column break.** Eroding opened gaps at
   `(800,826)` and `(873,894)` with 47 px occupied between them — the centred
   page-number footer, sitting exactly in the gutter. Binary occupancy lets a
   single stray element deny a column split, so the projection COUNTS crossings
   and tolerates a few.
3. **The gutter was found and the output stayed interleaved.** `spans()` tested
   the RAW box against a gutter computed from ERODED boxes, so every unclipped
   column line read as spanning, every line flushed the stripe, and the
   column-major sort never ran. Both now use one criterion — width against the
   same threshold that excludes a box from the projection — so they cannot
   disagree by construction.

And the unit test earned its place: gating the outlier tolerance on `n_lines`
came from `spanning_title_separates_stripes` failing, because with three lines
a tolerance of 1 erases an entire single-line column and the gutter merges into
the right margin. A rule that is right on a page can be wrong on a fixture, and
only the fixture says so.

**What remains is skew**, now cleanly separated: 39.59 % on skewed
single-column pages and 15.20 % on skewed two-column. That is §8.24's parked
quad-crop lever, which was measured at a 1.26 pp ceiling on near-upright CORD
words and pruned for want of a corpus with real skew. It has one now.

### 8.29 Recursive XY-cut on real documents — and the confound that nearly buried it

§8.28's one-level cut computed ONE set of gutters per page. On OmniDocBench
that is right for a report and wrong for a newspaper, which is a headline over
a 3-column block beside a boxed sidebar. Made recursive: at each node compare
the widest horizontal valley against the widest vertical one, cut along the
larger, recurse. Spanning elements force a horizontal split whatever the valley
measures, because a headline is a separator even with ordinary leading.

All three strategies, ONE binary, one variable (`FFAI_ORDER`):

| CER %, 236 pages | raster | one-level | **recursive** |
|---|---:|---:|---:|
| newspaper (47) | 76.92 | 34.51 | **20.07** |
| magazine (60) | 72.49 | **20.69** | 24.34 |
| book (33) | 64.14 | **42.78** | 44.36 |
| colorful_textbook (23) | 43.89 | 27.27 | 27.29 |
| PPT2PDF (20) | **21.96** | 23.70 | 22.48 |
| exam_paper (10) | **33.80** | 34.22 | 34.30 |
| academic_literature (43) | 85.66 | 75.93 | **73.16** |
| **ALL** | **74.39** | **39.17** | **33.02** |

**33.02 % against 39.17 %** — 6.15 points over the one-level cut, 41 points over
no ordering at all, and newspapers 34.51 -> 20.07 with their order component
falling from 21.40 pp to 6.96. Kept.

#### The confound, which is the real lesson

The recursive cut first measured 35.46 %, then 33.02 %, against a "baseline" of
29.61 % — so it read as a clear regression and was nearly reverted twice. The
baseline was wrong. It had been measured on an EARLIER BINARY: a sibling
session swapped the PNG and JPEG decoders in the same window, so the two arms
differed by an image decoder as well as by the thing under test.

The tell was in the data and I nearly talked past it. Academic literature's
**order-free** score also doubled, 34.92 % -> 73.66 %. Order-free is a
bag-of-tokens comparison; reordering lines cannot move it. A metric that is
order-independent changing under an order-only change means something ELSE
changed — and that is not a subtle inference, it is arithmetic.

On one binary the true baseline is 74.39 %, not 29.61 %, and the recursive cut
is a 41-point win rather than a 3-point loss.

Two things landed to stop it recurring:

* `FFAI_ORDER=raster|onelevel|xycut` keeps all three strategies in ONE binary.
  The one-level cut had been *replaced* rather than kept, so the comparison
  required two builds — and in a shared worktree two builds is two variables.
* Recorded as §8.12's law restated: when a number contradicts a change you just
  made, indict the artifact before the code. This time the artifact was not
  stale, it was a *different binary* — same failure, new disguise.

#### Still open, and now cleanly separated

`academic_literature` at 73.16 % is the worst cell by 29 points, and its order
component is **-0.49 pp** — ordering is not its problem. Same for book (0.62)
and exam_paper (-3.12). Those are recognition, and the three-way against
PP-StructureV3 and Unlimited-OCR is what will say whether 73 % on academic text
is our recognizer or the corpus being genuinely hard.

### 8.30 The three-way on real documents — measured, and the gap is 10.4 points

Carmenta against both document-tier references on 43 OmniDocBench pages,
identical pixels, one binary, ledger-recorded:

| | CER | WER | pages/s | peak MiB | steady MiB |
|---|---:|---:|---:|---:|---:|
| **Unlimited-OCR** (Baidu, 3B MoE, GPU) | **15.51 %** | 23.02 | 0.01 | 8745 | — |
| PP-StructureV3 | 19.14 % | 30.65 | 0.02 | 4972 | 1481 |
| **mobiledet-crnn** (ours, CPU) | 25.91 % | 41.58 | **0.17** | 2369 | **425** |
| craft-crnn (ours, CPU) | 55.06 % | — | 0.10 | 1821 | — |

**correctness PASS · quality FAIL · speed PASS · footprint PASS.**

The honest shape of it: **10.4 points behind the model that holds the
OmniDocBench record, 6.8 behind PP-Structure, at 17x its throughput on a CPU
and 0.29x PP-Structure's steady memory.** Not parity. The same order of
magnitude, from 4.7 MB of detector weights against 6.4 GB, on a machine with no
GPU.

Three things any use of these numbers has to carry:

1. **Name the engine.** craft-crnn reads 55.06 % where mobiledet-crnn reads
   25.91 %. "Carmenta scores X" is meaningless; the detector choice is worth 29
   points on documents.
2. **43 pages, not 316** — a stratified subset so two reference models could run
   in hours rather than a day, with every source represented.
3. **One page excluded**, and it is a defect not a choice: `rusty_jpeg` 0.1.5
   panics on progressive JPEG (SOF2, `decode/decoder.rs:1449`, unwrap on None).
   49 of the parent corpus's 316 pages are progressive. A panic costs the whole
   batch rather than one page, which is why four benchmark arms died silently
   before this was found.

**RESOLVED, and I had it backwards.** Both parseq arms reported
`correctness FAIL 0/43` beside a footprint-accounting note, and I read the note
as the cause and called the engines "unmeasured rather than broken". They were
broken. The harness pushes the real error onto the same `notes` vector the
footprint note already occupies, and the correctness gate prints
`notes.first()` — so a genuine failure was masked by a memory sentence pushed
earlier. Inserting the error at the FRONT surfaced it:

    omni-0023: candle: narrow invalid args start + len > dim_len:
    [1, 26, 192], dim: 1, start: 0, len: 27

A real defect in our PARSeq port. The AR loop runs `MAX_STEPS` (26) times and
exits early only on EOS, so a word that never terminates yields `[BOS]` + 26
ids = 27 positions against a `pos_queries` holding 26. Every synthetic corpus
we own produces words short enough to hit EOS; real documents do not. Clamped
to the model's capacity — both PARSeq oracles still pass, so the
reference-exact path is untouched — and the arm now completes:

| | CER | WER | pages/s |
|---|---:|---:|---:|
| mobiledet-parseq | 31.56 % | 61.91 | 0.16 |

Two lessons, both about diagnostics rather than decoding. **A failure message
assembled from a shared notes list will eventually report the wrong note**, and
the one it reports will be the one pushed earliest, not the one that matters.
And **one clip's error aborts the whole corpus run** — 42 good pages were
discarded because page 23 failed, which is why the defect looked like a harness
quirk instead of a crash.

Both are now fixed (`b4dc952`). The recognize loop records a clip failure and
continues, and decode is wrapped in `catch_unwind` so a panicking dependency
costs one page rather than the corpus — rusty_jpeg 0.1.5 unwraps a `None` on
progressive JPEGs, and 49 of these 316 pages are progressive, so an upstream
defect had been producing *no data at all* and no indication of which page did
it. The correctness gate already expressed partial success as
`clips_ok < clips_total`; aborting was discarding exactly the information the
gate exists to report, and paying 42 good measurements to learn one bad one.
Verified on a six-page canary carrying one progressive JPEG: 5/6 processed,
correctness FAIL naming the culprit, CER still measured on the five.
`catch_unwind` is deliberately **not** wrapped around the engine — a panic in
Carmenta is our defect and must abort loudly.

### 8.31 academic_literature at 73 % was a decoder crash, not a recognizer

**The cell does not exist. `academic_literature` reads 25.88 %, not 73.16 %,
and the difference is a decoder crash scored as a wrong answer.**

| mobiledet-crnn | pages | CER | order-free |
|---|---:|---:|---:|
| academic, **decodable** | 16 | **25.88 %** | 27.24 % |
| academic, decode failed — scored as empty | 27 | 100.00 % | 100.00 % |
| academic, blended (the old figure) | 43 | 73.16 % | 73.66 % |
| newspaper, zero decode failures | 47 | 20.07 % | 13.11 % |

`omni_split.py` reads a subprocess's output as `out.stdout or ""`. When
rusty_jpeg 0.1.5 panics on a progressive JPEG the process dies, stdout is
empty, and an empty hypothesis scores ~100 % CER — silently, as though the
recognizer had read the page and got every character wrong. 27 of this cell's
43 pages are progressive. **Newspaper, the control I compared against, has
zero.**

| source | pages | undecodable |
|---|---:|---:|
| academic_literature | 43 | **62.8 %** |
| magazine | 60 | 8.3 % |
| book | 33 | 6.1 % |
| colorful_textbook | 23 | 4.3 % |
| PPT2PDF / exam_paper / newspaper | 77 | 0.0 % |

The blend reproduces 73.16 % to two decimals, which is what confirms this is
the same quantity the split scorer was reporting and not a second measurement
that happens to disagree. At 25.88 % the cell sits *at* the corpus average
(25.91 % on the three-way), 5.8 points off newspaper rather than 53. It was
never the worst cell and never a recognition problem.

**Both hypotheses in the original descent were chasing this artifact**, and
they are recorded rather than deleted because the reasoning was sound and the
target was not:

- *Small type* — refuted on its own terms and the refutation still holds: the
  character cell is **19.10 px academic against 19.00 px newspaper**, measured
  as `sqrt(region area / characters)` from the annotations. Type size genuinely
  is not a differentiator here. But the 53-point gap it was invoked to explain
  was mostly manufactured.
- *Inline mathematics* — the corpus filter excluded `equation_isolated` regions
  while inline formulas live inside `text_block` and were never filtered, so
  ground truth really does ask for `$ \mathrm{N i S O}_{4} $` where the page
  shows NiSO4, at **15.08 markers per 1k characters against newspaper's 0.91**.
  That remains a true fact about the corpus and a real capability gap —
  PP-StructureV3 and Unlimited-OCR ship formula heads that emit LaTeX and can
  match this reference; we emit glyphs and structurally cannot. It is worth
  perhaps 8 % of characters. It is not worth 47 points, and it is now a
  known-size tax rather than a mystery.

**The lesson is the one this campaign keeps relearning, and the skill states
first: run depth 6 before depths 3–5.** Two hypotheses were built, measured,
and written up — one refuted, one bounded — while the actual cause was that
62.8 % of the cell never reached the recognizer. The tell was available for
free the whole time and nobody asked for it: *does every page in both arms
produce output at all?* A cell that is 62.8 % one file format and a control
that is 0 % is not a comparison. This is the fifth time in this campaign that
the decisive number was one that had been filtered, swallowed, or never
printed.

**Every per-source figure taken before `b4dc952` carries an unknown decode tax,
and the tax is not evenly distributed** — so the splits are not comparable to
each other, which is exactly the thing splits exist to do. The three-way in
§8.30 is unaffected: all 43 of its pages completed, which under the old
abort-on-first-failure code they could not have done otherwise.

**Full holdout re-baseline**, the first one obtainable at all — 35 progressive
JPEGs would each individually have killed the run before `b4dc952`:

    mobiledet-crnn   201/236 pages   CER 31.81 %   0.18 pg/s   456 MiB steady

`correctness FAIL` and correctly so: 35 pages are unreadable, and the gate says
so instead of the run silently pricing them at 100 %.

### 8.32 The progressive-JPEG panic, root-caused and fixed — one guard

Returning those 35 pages is 15 % of the corpus for zero model work, and
`rusty_jpeg` is ours, so this is a fix at the source rather than a workaround.
The crate already *implements* progressive decoding; it was a hoisted `unwrap`,
not a missing feature.

`decode_block_progressive`, `src/decode/decoder.rs:1449`:

```rust
    let mut index = cmp::max(spectral_selection.start, 1);
    ...
    let ac_table = ac_table.unwrap();          // <-- panics
    while index < spectral_selection.end {
```

A progressive image's **first scan is DC-only**: `spectral_selection` is `0..1`,
so `index` starts at 1, `1 < 1` is false, and the loop never executes — and a
DC-only scan legitimately declares **no AC table**, so `ac_table` is `None`. The
unwrap was hoisted above the loop guard as a speed optimisation (its comment
cites ~362k calls per 1080p frame), which is correct for baseline JPEG, where
every scan carries an AC table. That is also why every baseline fixture passed.

The fix takes the guard first, keeping the hoist for the hot path and costing a
comparison the loop was about to make anyway:

```rust
    if index >= spectral_selection.end {
        return Ok(());
    }
    let ac_table = ac_table.unwrap();
```

Verified four ways, because "stopped panicking" is not "decodes correctly":

| check | result |
|---|---|
| reproduces unpatched | panic at `decoder.rs:1449` on 3/3 files |
| decodes patched | 4/4, e.g. `1653x2339 RGB24` |
| **pixel-correct vs Pillow** | **max abs diff 1**, mean 0.009, zero pixels off by >2 — IDCT rounding |
| no baseline regression | crate's own suite, **58 passed / 0 failed** |

Pending upstream publication, so FFai still carries the `catch_unwind` from
`b4dc952` — which stays regardless. Surviving a dependency's panic is a
property the harness should have on its own, not one borrowed from a fixed
dependency.

**The prize, measured rather than predicted.** `cargo --config
"patch.crates-io.rusty_jpeg.path='…'"` injects the patched crate for one build
without editing a single file in the worktree — which matters here, because a
sibling campaign shares it. Same corpus, same engine, same box:

| full holdout | clips | CER | pg/s | steady |
|---|---:|---:|---:|---:|
| unpatched decoder | 201/236 · **FAIL** | 31.81 % | 0.18 | 456 MiB |
| patched decoder | **236/236 · PASS** | **33.70 %** | 0.16 | 453 MiB |

**The headline CER got worse by 1.89 points, and that is the honest result.**
The 35 recovered pages were never scored badly before — they were *absent from
the denominator*. Fixing the decoder moves them from "not measured" to
"measured, and hard", which is what a real corpus figure looks like. A change
that repairs a defect and improves the headline should be suspected of having
repaired the measurement in its own favour; this one did the opposite, which is
the weaker claim and the trustworthy one.

This is also the first `correctness PASS` on the full OmniDocBench holdout.

### 8.33 Two averagings, and the 6.7 points we spend reading figures

The full-holdout run and the split probe reported **33.70 %** and **24.77 %**
on the same 236 pages, same engine, same box. Nine points is not rounding, and
until it was explained neither number was quotable.

**It is the averaging, and only the averaging.** `runner.rs:1363` computes
`sum(per-clip CER) / clip_count` — MACRO, every page weighted equally. The
probes compute `sum(edits) / sum(chars)` — MICRO, every character weighted
equally. Recomputing both from one pass reproduces the harness figure exactly:

| | micro | macro |
|---|---:|---:|
| ALL 236 pages | 24.77 % | **33.70 %** |

Exact reproduction is what rules out a second defect. Both are defensible; they
are not interchangeable, and this campaign had been quoting both — §8.29's
XY-cut deltas are micro, §8.30's three-way is macro. **The three-way is
unaffected**: engine and references both go through the same `mean`
(`runner.rs:851` and `:958`), so all three arms are macro and the comparison is
internally consistent. From here every figure states its averaging.

**What macro exposed was worth the confusion.** Per cell the two diverge
wildly — `book` reads 27.83 % micro and 59.20 % macro — because ground truth
runs from **55 to 39686 characters, a 722x spread**, and `metrics.rs`
deliberately does not cap the error rate (capping "would hide gross
over-generation"). One short page where the detector over-fires can score
several hundred percent and outweigh dozens of long ones:

| page | source | ref | ours | CER |
|---|---|---:|---:|---:|
| omni-0245 | book | 315 | **2972** | **843.5 %** |
| omni-0055 | academic_literature | 3437 | 11013 | 221.5 % |
| omni-0213 | book | 418 | 1248 | 199.0 % |

Nine pages exceed 100 %. `omni-0245` alone carries ~25 of `book`'s 59.20 %.

**And the mechanism is not bad reading.** `omni-0245` is a book page containing
a screenshot of a Google results window. The annotation carries three prose
regions — 315 characters; we transcribe the whole page, browser chrome and URL
bar included. `omni-0213` is the same shape: four annotated regions, and we
read the unannotated figure too. **A large part of our measured error is text
that is genuinely on the page and deliberately not in the answer key.**

That names the phenomenon precisely. **It does not name a gap against the
references — see §8.35, which measured that claim and refuted it.** They pay
the same tax we do.

**The lever, with a measured ceiling** (micro, all 236 pages):

| | value |
|---|---:|
| pages emitting >1.5x the reference | **16 / 236** |
| share of reference characters they hold | 2.8 % |
| **share of all edits they cause** | **13.3 %** |
| CER on those 16 pages | 119.00 % |
| CER on the other 220 | 22.08 % |
| **ceiling if over-generation were perfectly suppressed** | **24.77 % -> 18.07 %** |

**6.7 points of micro CER for a layout decision rather than a better
recognizer** — the largest single lever this campaign has measured since
reading order, and unlike reading order it needs no new model: the detector
already emits the boxes, and the question is which of them to keep. Note the
ceiling is micro and therefore NOT comparable to §8.30's macro three-way; what
it bounds is the prize, not the ranking.

Hypotheses for all 236 pages are cached under `.tools-bench/cache/`, so
re-scoring under any metric is now instant. Every question in this section
previously cost a 25-minute re-run, which is why so few of them had been asked.

### 8.34 The 6.7 points are not cheap — size and confidence both refuted

§8.33 priced over-generation at 6.7 points of micro CER and named the mechanism
on two inspected pages. The obvious follow-up is whether the offending lines can
be filtered by something the pipeline already computes. Two candidates, both
free: recognition **confidence** (`FFAI_OCR_CONF` exists for exactly this) and
line **height**.

**First attempt, and why it was thrown away.** Labelling a line as in- or
out-of-reference by matching its first 18 characters against ground truth looks
obvious and does not work on output with ~20 % CER — one misread character
moves a correct line into the wrong bucket. Measured on six pages that emit
**no** excess text, that matcher still called **18-48 % of lines unmatched**.
Both its groups are mixtures, so every statistic built on it was void, including
a one-page result (`omni-0245`, height 32 px against 14 px, 2.29x) that looked
like a finding and generalised to nothing.

**The instrument that works is geometric.** Each sidecar carries a `poly` per
annotated region; a detected line's centre is inside one or it is not, and OCR
errors cannot change that. The control confirms it: pages that do not
over-generate show **1-13 out-of-region lines against 29-90 inside**, where the
string matcher claimed 18-48 %.

**With a sound instrument, both levers are refuted:**

| separator | pages where it separates |
|---|---|
| line height >1.3x either way | **2 / 16** |
| confidence, +0.05 or better | **1 / 16** |

Typical page: `omni-0055` has 48 in-region and 113 out-of-region lines, heights
22 px against 21 px, confidence 0.986 against 0.976. `omni-0092`'s out-of-region
text is *more* confident than its body text (1.000 against 0.989).

**The mechanism explains the refutation.** Out-of-region text is not degraded
text. It is real text, correctly read, sitting where the benchmark declines to
annotate — figure interiors, screenshots, page furniture. Nothing about its
*appearance* differs from body text, so no appearance-based filter can find it.
The only property that distinguishes it is **position relative to a region a
layout model would have identified**.

So the 6.7 points require layout analysis — region detection and
classification — and cannot be had with a threshold. **Recorded as refuted
rather than open**, with the measurements above, so the cheap version is not
re-attempted: it has now failed once on a broken instrument and once on a sound
one. (An earlier draft added "the same capability PP-StructureV3 and
Unlimited-OCR already ship". §8.35 measured that and it is false.)

`.tools-bench/region_label.py`. Runs off the cached dumps; no recognizer.

### 8.35 PP-Structure reads the screenshot too — the lever is unexploited, not lost

§8.33 and §8.34 both asserted that PP-StructureV3 and Unlimited-OCR suppress
figure regions, and that this was the source of their lead. It was the one
claim in either section not backed by a measurement — plausible, because they
run layout classification and we do not. Measured on the six worst
over-generating pages, output length against reference length:

| page | ref | ours | x ref | PP-Structure | x ref |
|---|---:|---:|---:|---:|---:|
| omni-0245 | 315 | 2972 | 9.43x | 2929 | **9.30x** |
| omni-0055 | 3437 | 11013 | 3.20x | 11123 | **3.24x** |
| omni-0039 | 2228 | 6289 | 2.82x | 6216 | **2.79x** |
| omni-0092 | 1121 | 1841 | 1.64x | 1820 | 1.62x |
| omni-0025 | 3924 | 6320 | 1.61x | 6029 | 1.54x |
| omni-0213 | 418 | 1248 | 2.99x | **421** | **1.01x** |

**It reads the Google screenshot too**, and on `omni-0055` emits *more* than we
do. It suppressed the figure on one page of six.

So the claim is refuted, and two conclusions change:

- **Over-generation is not why the references are ahead.** They pay the same
  tax. Their lead comes from somewhere else, and this campaign has not yet
  established from where — a question §8.30 left open and this closes one wrong
  answer to.
- **The lever survives, and is better than it looked.** PP-StructureV3's
  measured 19.14 % *includes* this tax. A suppression pass would move our
  absolute CER against a reference number that does not benefit from the same
  fix — so the 6.7 points are ground nobody has taken, rather than ground we
  are behind on.

The character-count comparison needs no text matching, which is why it is
trustworthy where §8.34's first instrument was not. Sample is six pages chosen
as the worst offenders, not a random draw: it establishes that PP-Structure
does not systematically suppress, not the rate at which it sometimes does.

**The general lesson, third instance this session.** A mechanism that explains
the data is not evidence for the mechanism. "They classify layout, we do not,
therefore they suppress what we read" is coherent, fits every number in §8.33,
and is false. It survived two sections and two commits because it was never
given its own measurement — and it was cheap to measure, six pages and fifteen
minutes.

### 8.36 The gap is reading order — 13.7 of 15.5 points, recognition at parity

§8.35 refuted the previous answer to "why are the references ahead" and left
the question open. Measured, on 28 pages stratified across all seven sources
(evenly spaced by id within each, so the sample cannot have been picked to
flatter either side), our engine against PP-StructureV3, micro CER, with the
order-free column beside it:

| cell | ours | PP-V3 | gap | ours o-free | PP o-free | gap |
|---|---:|---:|---:|---:|---:|---:|
| academic_literature | 63.4 % | 29.7 % | **+33.7** | 33.7 % | 32.8 % | **+0.9** |
| magazine | 40.2 % | 20.4 % | **+19.8** | 15.8 % | 15.5 % | **+0.3** |
| newspaper | 29.7 % | 12.2 % | +17.5 | 14.5 % | 10.5 % | +4.0 |
| PPT2PDF | 10.7 % | 3.4 % | +7.3 | 10.6 % | 13.5 % | −2.8 |
| book | 36.8 % | 31.7 % | +5.2 | 23.9 % | 23.5 % | +0.5 |
| colorful_textbook | 15.6 % | 12.4 % | +3.1 | 21.9 % | 22.6 % | −0.7 |
| exam_paper | 22.9 % | 25.6 % | **−2.8** | 24.5 % | 26.0 % | −1.6 |
| **ALL** | **34.1 %** | **18.6 %** | **+15.5** | **19.9 %** | **18.1 %** | **+1.8** |

**Our recognition is at parity. 13.7 of the 15.5 points is sequence.**
Destroying order in both outputs collapses the gap from 15.5 pp to 1.8 pp.
Put the other way: reading order costs us **14.2 points** and costs
PP-StructureV3 **0.5**. Their layout model emits a near-perfect sequence; our
geometric XY-cut does not.

The per-cell pattern is the same finding stated seven times. `magazine` is
+19.8 pp raw and **+0.3 pp** order-free — we read a magazine as accurately as
PP-Structure does and assemble it wrongly. `academic_literature`, the cell
§8.31 spent a whole descent on, is +33.7 pp raw and **+0.9 pp** order-free: it
was never a recognition problem there either. Where we already order well
(`exam_paper`, `colorful_textbook`, `PPT2PDF`) we match or beat them outright,
and `exam_paper` we simply win.

**This redirects the campaign.** §8.34 priced layout-region suppression at 6.7
points and found no cheap route to it; this is twice that, on the same
detector output, and §8.29 already showed the lever moves — recursive XY-cut
was worth 6 points when it landed. It has considerably further to go, and the
cells that need it most (magazine, academic, newspaper) are the three largest
in the corpus.

Caveats, stated: 28 pages not 236; micro CER, so not comparable to §8.30's
macro three-way; PP-StructureV3 at the mobile tier with oneDNN off, the
handicaps pinned in `ppstructure_ref.py`. What the sample establishes is the
*decomposition* — order versus characters — which is a ratio and far more
robust to sample size than either absolute number.

### 8.37 The ordering defect is the algorithm, and it has a sign flip — **RETRACTED, see §8.39**

§8.36 put 13.7 points on sequence without saying which half of the pipeline
owns it, and the two answers need different work: either the rule mis-orders
even perfect input, or the rule is fine and it is being handed detected LINES
where it wants REGIONS. The corpus settles this without a recognizer — every
sidecar carries each region's polygon *and* its annotated reading order, so
feeding true regions in and counting inversions out isolates the algorithm
completely.

Inversions rather than exact match: one region out of place should cost one
unit, not void the page. Normalised by `n(n-1)/2`, so 0 % is a perfect sequence
and 50 % is random.

| cell | pages | SHIPPED | raster | pages perfect |
|---|---:|---:|---:|---:|
| newspaper | 47 | 13.42 % | 29.65 % | 19 % |
| magazine | 60 | 9.40 % | 20.43 % | 32 % |
| colorful_textbook | 23 | 8.08 % | 8.67 % | 61 % |
| academic_literature | 43 | 7.44 % | 16.24 % | 58 % |
| book | 33 | 4.74 % | 11.66 % | 73 % |
| **PPT2PDF** | 20 | **2.50 %** | **0.14 %** | 85 % |
| exam_paper | 10 | 0.35 % | 0.00 % | 90 % |
| **ALL** | 236 | **8.10 %** | 16.55 % | **50 %** |

**Given perfect regions, the shipped rule still mis-sequences half of all
pages.** So the algorithm is the defect and improving it does not depend on
better detection first. It is also clearly worth having — it halves raster's
inversions overall, and on newspapers cuts them from 29.65 % to 13.42 %.

**And it has a sign flip.** On `PPT2PDF` raster scores **0.14 %** against
XY-cut's 2.50 %, and on `exam_paper` 0.00 % against 0.35 %; `colorful_textbook`
is a wash. Recursive cutting *invents* structure on pages that have none, and
a slide or an exam paper read top-to-bottom is simply correct. This is the
same shape as every adaptive-dispatch win this campaign has taken: a lever that
wins big on one content class and loses on another, with the classes
separable. §8.36's per-cell gaps line up with this table — `exam_paper` is
0.35 % here and the one cell where we *beat* PP-StructureV3, `PPT2PDF` is
2.50 % and nearly tied — which is independent corroboration, since those
numbers come from CER against a different reference.

**Two instrument failures on the way here, both caught by an assertion rather
than by inspection.** First, the testbed was written as a Python
reimplementation of `boxes.rs`; it reproduced the shipped order on **2 pages of
12** and every number from it was discarded. `examples/order_probe.rs` now
exposes the real `order_reading` and the testbed calls it. Second, that probe
parsed box coordinates as `usize` while OmniDocBench polygons are floats, so
`filter_map` silently dropped every row and the probe emitted nothing — which
the caller's permutation check reported as `order_reading` losing boxes. A
shipped-code bug was nearly written up. **A parse that discards what it cannot
read is the same defect as a scorer treating a dead process as empty output**,
which is §8.31, which is the fifth instance this campaign has recorded.

Next: dispatch on measured structure rather than always cutting — the sign flip
says the win is available and the classes are separable.

### 8.38 The adaptive gate is refuted, and the sign flip is worth ~0.2 pp — **measured on the void instrument of §8.37; see §8.39**

§8.37's sign flip suggested the obvious lever: do not cut pages that have no
columns. `adaptive_cut` implements it with the separator the cut already
computes — no vertical valley at the top level means no column structure, so
read top-to-bottom. Measured on both splits, inversion rate over annotated
regions:

| | default | adaptive | raster |
|---|---:|---:|---:|
| holdout, 236 pages | **8.10 %** | 8.49 % | 16.55 % |
| train, 80 pages | **7.53 %** | 7.54 % | 15.74 % |

**Worse on holdout, a wash on train. Refuted.** Both ends of the hypothesis
were wrong, and the per-cell numbers say why:

- `PPT2PDF` did not move at all (2.50 % either way). Slides *do* have a
  top-level vertical valley — two-column bullet layouts — so the gate never
  fired on the cell it was built for.
- `newspaper` got **worse**, 13.42 % → 15.14 %. Its columns often appear only
  *after* a horizontal cut separates the masthead, so testing the whole page
  once gates off the recursion exactly where it was earning its keep.

Only `exam_paper` improved (0.35 % → 0.00 %), and that is 10 pages.

**The more useful finding is that the sign flip is not worth chasing.**
`PPT2PDF` costs 2.36 pp on a cell holding 8 % of the corpus — about **0.2 pp**
overall — while `newspaper` alone still carries 13.42 % and `magazine` 9.40 %.
The prize is making the cut *better on multi-column pages*, not switching it
off on simple ones. §8.37 framed this as an adaptive-dispatch opportunity
because that is this campaign's usual shape; measured, the asymmetry is 60:1
the other way.

`adaptive_cut` stays as a named `FFAI_ORDER` mode rather than being deleted —
the modes exist so variants can be A/B'd on one binary, and a refuted variant
that can be re-measured is worth more than a paragraph saying it was tried.
The default is unchanged.

**A discipline note, because this nearly went wrong.** The testbed ran on
HOLDOUT, and the adaptive hypothesis was read off a holdout table — searching
for a threshold there would have burned the split that has to certify the
answer. It now takes an explicit `split` argument and **defaults to train**.
The testbed is fast enough (no recognizer, 236 pages in seconds) that it
actively invites the mistake.

### 8.39 §8.37 and §8.38 are void — the testbed fed regions to a line-scale rule

Both sections rest on one number: 8.10 % inversions when the shipped
`order_reading` is handed perfect annotated regions, from which §8.37 concluded
"the algorithm mis-sequences half of all pages, so the algorithm is the
defect". **That number is an artifact of the testbed and the conclusion does not
follow.**

Diagnosing the worst train newspaper page showed the signature immediately.
`omni-0288` has three columns at x0 ≈ 121 / 611 / 1103 and a true order that is
column-major (ranks 0-5, 6-12, 13-16). The cut emitted **0, 6, 13, 1, 7, 14, 2,
8, 15** — straight across all three columns, row by row. That is raster
behaviour: the vertical gutters were never found at all.

They could not be. `xy_cut` measures every valley in units of the median box
height, which is correct when the boxes are text LINES — the units the shipped
pipeline actually produces — and nonsense when they are REGIONS:

| page | median box height | column gutters | gutter in those units | V_GAP_MIN 0.55 |
|---|---:|---|---|---|
| omni-0288 | **311 px** (region) | 19 px, 14 px | 0.06, 0.05 | **never fires** |
| omni-0288 | ~30 px (line) | 19 px, 14 px | 0.64, 0.47 | fires |
| omni-0292 | **405 px** (region) | 11 px, 18 px | 0.03, 0.04 | **never fires** |
| omni-0292 | ~30 px (line) | 11 px, 18 px | 0.38, 0.59 | fires |

So the testbed asked the algorithm to find column gutters while telling it a
column gutter would have to be ~170 px wide. It answered "there are no columns"
and fell back to raster, on every multi-column page. **The shipped code is not
implicated: it only ever receives lines.**

**What is withdrawn:**

- §8.37's headline — "given perfect regions the rule still mis-sequences half
  of all pages" — and with it "the algorithm is the defect, improving it does
  not wait on better detection". Unmeasured either way.
- §8.37's sign flip and its per-cell table. `PPT2PDF` scoring worse than raster
  is exactly what a crippled cut does on pages whose regions are large.
- §8.38 entirely. The adaptive gate was measured against a rule that had already
  degenerated to raster, so "worse on holdout, a wash on train" describes
  nothing about the real pipeline. Its *reasoning* survives — a whole-page gate
  cannot help a newspaper whose columns appear only under the masthead — but it
  was not tested.

**What survives untouched:** §8.36. That decomposition — 13.7 of 15.5 points on
sequence, recognition at parity — comes from end-to-end CER through the real
pipeline against PP-StructureV3, with no testbed involved. The target is
unchanged; only the tool built to attack it was wrong.

**The correct instrument** feeds LINES, because that is what the algorithm
consumes, and takes their ground-truth order from the region each line falls
inside — the geometric labelling §8.34 already validated. That is buildable from
the `FFAI_OCR_CONF` dumps and is the next step.

**Third instrument failure in this thread, and the most expensive.** The first
two were caught by assertions — a reimplementation that reproduced the shipped
order on 2 pages of 12, a parse that silently dropped every row. This one
produced a *plausible* number instead of an obviously broken one: 8.10 % against
raster's 16.55 % looked like the cut working, just not well enough. **A wrong
instrument that fails loudly costs an hour; one that returns a believable number
costs two sections and a commit.** The tell was there and unexamined — a rule
that halves raster's inversions but leaves half of all pages imperfect, on
*perfect input*, should have prompted "what does it think a line height is?"
before it prompted a fix.

### 8.40 Reading order, measured properly — XY-cut is worth 3.4x (cell ranking corrected in §8.41)

The instrument §8.39 called for: lines from `FFAI_OCR_CONF` dumps — real
detections in the order the engine emitted them, so `order_reading` sees the
units its thresholds are calibrated for — with each line's true rank taken from
the annotated region its centre falls inside (§8.34's geometric labelling, which
has a validated control). Lines inside no region are dropped rather than
defaulted; they are §8.33's over-generation and have no true rank, and mixing
the two defects into one number is how §8.31 happened. Inversions count only
between lines in DIFFERENT regions. TRAIN split.

**It passes the check set before its output was seen:** the cell ranking
reproduces §8.36's independently measured order cost — academic worst, exam and
colorful_textbook best — on a different split, through a completely separate
path (end-to-end CER against PP-StructureV3). The voided testbed never had that
corroboration, which is precisely why a plausible number survived two sections.

| cell | pages | raster | XY-cut | perfect (XY) |
|---|---:|---:|---:|---:|
| newspaper | 17 | 31.20 % | **5.25 %** | 47 % |
| magazine | 21 | 26.67 % | **6.07 %** | 57 % |
| **academic_literature** | 15 | 24.61 % | **12.01 %** | 60 % |
| book | 11 | 11.85 % | 3.24 % | 73 % |
| colorful_textbook | 9 | 5.62 % | 1.22 % | 78 % |
| exam_paper | 2 | 0.09 % | 0.88 % | 50 % |
| **PPT2PDF** | 5 | **0.00 %** | **6.27 %** | 60 % |
| **ALL** | 80 | **20.51 %** | **5.96 %** | 60 % |

**Three conclusions, all of which differ from what the void instrument said:**

1. **The recursive cut is worth far more than previously credited** — 20.51 % to
   5.96 %, a 3.4x reduction, against the 2x the region-fed testbed reported.
   Newspapers improve 6x. §8.29 shipped it on a 6-point CER gain; on the units
   it actually consumes it is doing much more than that number implied.

2. **`academic_literature` is the worst ordering cell at 12.01 %**, double
   magazine and newspaper, and the *weakest beneficiary* of the cut (2x where
   newspapers get 6x). This is a genuine surprise: newspapers have been treated
   as the ordering problem since §8.29. It also closes §8.31's loop — that cell
   does have a real defect, it is ordering rather than recognition, and it is
   the largest single one on the board.

3. **The sign flip is real, and is reinstated.** §8.37 claimed it, §8.39
   withdrew it as an artifact of region-scale input — correctly, on the evidence
   then available — and the valid instrument shows it plainly: raster orders
   **100 % of slides perfectly** and the cut breaks 40 % of them (0.00 % ->
   6.27 %). `exam_paper` flips the same way at negligible size. Withdrawing it
   was right; it was unmeasured, and "unmeasured" is not "false".

   §8.38's adaptive gate remains untested — it was measured against a degenerate
   rule — and its stated mechanism is now sharper: the gate must not be a
   whole-page test, because newspapers find their columns only *below* the
   masthead. What PPT2PDF needs is for the recursion to decline to cut when the
   valley it found is not evidence of columns, which is a per-node question, not
   a page-level one.

Corpus-weighted, PPT2PDF is 8 % of pages and its regression is ~0.5 pp of the
5.96 %; academic_literature is 18 % and carries ~2.2 pp. The ranking of work is
unambiguous.

### 8.41 The column-first cut wins on train and LOSES on holdout — refuted

§8.40 made `academic_literature` the top target and `line_diag.py` found a
clean mechanism. `omni-0038` emits region sequence
`[0, 5, 1, 5, 1, 5, 1, 5, ...]` — strict left-right alternation between a
left column (regions 0-4) and a right column (region 5). `xy_cut` takes
whichever valley is wider, and a two-column paper with figures between its
captions has horizontal gaps of ~19 line-heights against a ~2 line-height
gutter, so it cuts horizontally into thin bands. Each band then orders its own
left and right lines correctly, and the page comes out interleaved.

The argument for the fix is that those two valleys are not the same kind of
evidence. A vertical gap `best_gap` returns is a band **no box crosses** — a
gutter running the full height of the node, i.e. structure. A horizontal gap is
whitespace, and whitespace is wider on pages containing figures. `xy_cut_vfirst`
therefore prefers the column cut, except when something spans the node, which is
what §8.29 built horizontal-first to protect (a vertical cut assigns a headline
to whichever side its centre lands on, slicing it in half).

**On train it worked, and on holdout it does not.**

| cell | holdout base | holdout vfirst | delta | train delta |
|---|---:|---:|---:|---:|
| colorful_textbook | 8.06 % | 10.24 % | **+2.18** | +2.11 |
| PPT2PDF | 3.37 % | 5.50 % | **+2.13** | **−1.21** |
| magazine | 4.10 % | 5.22 % | +1.12 | +0.27 |
| newspaper | 5.05 % | 4.13 % | −0.92 | −0.75 |
| academic_literature | 4.39 % | 3.75 % | −0.64 | **−3.76** |
| book | 3.54 % | 2.30 % | −1.24 | −1.05 |
| exam_paper | 0.52 % | 0.77 % | +0.25 | −0.79 |
| **ALL** | **4.44 %** | **4.65 %** | **+0.21 WORSE** | −0.80 better |

**Refuted. The default stays `xy_cut`;** `vfirst` remains a named `FFAI_ORDER`
mode so the next attempt starts from a measured position rather than this
paragraph.

**And the premise it was built on does not generalise.** §8.40 called
`academic_literature` the worst ordering cell at 12.01 %. That was 15 TRAIN
pages. On holdout's 43 it reads **4.39 %** — mid-table — and the worst cell is
`colorful_textbook` at 8.06 %, which §8.40 listed as second-best. The whole
diagnostic chain (worst cell -> diagnose its pages -> find the interleave ->
build the fix) was chasing a cell that is not unusually bad.

The mechanism is still real: the interleave in `omni-0038` is not an artifact,
and `vfirst` does fix it (academic improves on BOTH splits, and `book` and
`newspaper` improve on both). What fails is the trade. Preferring columns costs
more on pages whose "gutter" is a figure margin than it gains on genuine
two-column text, and `PPT2PDF` flipping from −1.21 to +2.13 across splits says
the effect there is noise on 5 train pages.

**What survives from §8.40:** the 3.4x figure for the cut against raster, which
was a like-for-like comparison on identical pages. What is corrected: the cell
ranking, which was train-only and is superseded by the holdout column above.

**The discipline is the finding.** A −0.80 pp train win with a coherent
mechanism, a signature visible in the raw sequence, and improvements in the
predicted cells — everything a result is supposed to have — is +0.21 pp on the
split that decides. Had `line_order.py` defaulted to holdout (it did until
§8.38 caught it), this would have shipped.

### 8.42 Cell-level ordering numbers are noise — tau +0.05 across splits

§8.41's fix failed to transfer, and the reason is not the fix. The per-cell
inversion rates this thread has been steering by are not resolvable at these
page counts, and their cross-split ranking is indistinguishable from random.

| cell | train | holdout | rank train -> holdout |
|---|---:|---:|---|
| colorful_textbook | 1.22 % ±1.07 | 8.06 % ±2.47 | **5th -> 0th** |
| newspaper | 5.25 % ±1.71 | 5.05 % ±1.43 | 3rd -> 1st |
| academic_literature | 12.01 % ±4.73 | 4.39 % ±1.27 | **0th -> 2nd** |
| magazine | 6.07 % ±2.24 | 4.10 % ±1.14 | 2nd -> 3rd |
| book | 3.24 % ±2.26 | 3.54 % ±1.54 | 4th -> 4th |
| PPT2PDF | 6.27 % ±4.90 | 3.37 % ±2.09 | **1st -> 5th** |
| exam_paper | 0.88 % ±0.88 | 0.52 % ±0.37 | 6th -> 6th |
| **ALL** | **5.96 % ±1.23** | **4.44 % ±0.60** | |

(± is the standard error of the per-page mean.)

**Kendall tau +0.05 — 11 of 21 pairs concordant against 10.5 expected by
chance.** `academic_literature` carries ±4.73 on a 12.01 % mean and
`PPT2PDF` ±4.90 on 6.27 %; those intervals swallow every difference the thread
treated as a finding. Only the aggregate is stable, and even it moves 1.5 pp
between splits.

**What this invalidates.** §8.40's "academic_literature is the worst ordering
cell", already corrected in §8.41, was not merely wrong about which cell — the
question has no reliable answer at 9-23 pages per cell. §8.36's per-cell CER
table is subject to the same caveat: its ALL row (13.7 of 15.5 points on
sequence) is an aggregate over 28 pages and stands, but the cell-by-cell gaps
are single-digit-page estimates and should not have been read as a ranking.

**The rule this establishes.** On `carmenta-omnidoc`, ordering work is directed
by the **aggregate** number and validated on holdout. A cell-level difference
needs roughly n ≈ 60 pages to resolve 2 pp at these variances — more than any
cell in the train split has, and more than four of seven have in holdout. The
per-cell columns stay in the tooling because they localise a MECHANISM worth
inspecting (`omni-0038`'s interleave is real and was found that way), but they
do not rank work and must not select a target.

**The cost of learning it, stated plainly.** §8.40 picked a target cell,
§8.41 diagnosed it, built a principled fix, confirmed the mechanism in the raw
output, measured a −0.80 pp train win in exactly the predicted cells — and lost
on holdout. Every step was sound except the first, and the first was reading a
ranking out of numbers whose error bars overlap completely. **A confidence
interval on §8.40's table would have cost one line of code and saved the entire
sequence.** This is the sixth instrument failure recorded in this campaign and
the first where the instrument was not broken — it was simply asked a question
finer than it can answer.

### 8.43 The directing finding, now powered — order owns 89 % of the gap

§8.42 established that this corpus cannot resolve cell-level differences, and
that work must be directed by aggregates carrying error bars. Applying that
standard to §8.36 — the result that redirected the whole campaign from
recognition to ordering — showed it had been reported without one, and on 34
pages it did not survive:

| n = 34 | value | verdict |
|---|---:|---|
| ORDER share | +11.68 pp | 95 % CI **[−0.74, +23.15]** — includes zero |
| paired sign test | 22/34 pages | z = +1.71 — **not** significant |

Directing four sections of work on an interval spanning zero is exactly what
this discipline exists to prevent, and the interval should have been computed
when §8.36 was written, not four sections later. The character-weighted metric
is dominated by pages like `omni-0245` at 843 % CER, so 34 pages could not
resolve it — the same 722x page-length spread §8.33 found distorting the macro
number, resurfacing as variance.

It was also cheap to settle: at the observed rate, n ≈ 50 crosses z > 2. Thirty
more stratified pages, none previously measured:

| n = 69 | value | verdict |
|---|---:|---|
| raw gap (ours − PP-StructureV3) | **+13.29 pp** | |
| recognition, order-free | **+1.40 pp** | |
| **attributable to ORDER** | **+11.89 pp** | 95 % CI **[+3.88, +19.45]** — excludes zero |
| paired sign test | **44/69 pages** | **z = +2.29 — significant** |

**Established.** Order owns **89 %** of the deficit against PP-StructureV3, and
recognition sits 1.40 pp from parity — tighter than §8.36's 1.8 pp estimate and
now with an interval behind it. The campaign's direction is correct; it was
correct on 34 pages too, but it was not yet *known* to be.

Two things this fixes going forward. Every headline claim gets an interval at
the time it is made — the bootstrap and the paired sign test here are four lines
each and run off cached hypotheses. And when the two disagree, the sign test is
the one to trust on this corpus: it is immune to the length outliers that make
the character-weighted bootstrap so wide, which is why it reached significance
first (z = +1.71 at n = 34 against a CI that spanned zero).

**Where this leaves the campaign.** Ordering is confirmed as the lever, worth
~11.9 pp of the ~13.3 pp gap. §8.41's column-first attempt at it is refuted on
holdout, and §8.42 explains why the target was mis-picked. The next attempt is
directed by the aggregate inversion rate (holdout 4.44 %), validated on holdout,
and must clear an interval — not a cell ranking.

### 8.44 Four ordering variants, none survive — and the aggregate lied twice

§8.43 confirmed order owns 89 % of the gap. Four attempts at it, all measured on
the line-level testbed, train first and holdout deciding:

| aggregate inversions | train | holdout |
|---|---:|---:|
| baseline `xy_cut` | 5.96 % | **4.44 %** |
| `vfirst` — prefer columns everywhere | 5.16 % | 4.65 % |
| `vtop` — prefer columns at depth 0 | 5.28 % | 4.85 % |
| `onelevel` — explicit column grid | 5.72 % | not run |
| `hybrid` — route on spanning element | **4.60 %** | **4.25 %** |

`onelevel` is architecturally distinct from the recursion — one x-projection,
gutters found once, lines assigned to column bands — and its train profile
confirmed BOTH halves of a prediction made before measuring: it fixes the
column interleave (`academic_literature` 12.01 % -> 4.51 %, `magazine` 6.07 % ->
4.06 %) and fragments layouts whose structure changes down the page
(`newspaper` 5.25 % -> 9.13 %, the stated falsifier). Two architectures failing
on mechanistically opposite layouts is the sign-flip this campaign routes on, so
`hybrid` dispatches on the property that separates them — a page-wide element
means the layout changes vertically and only the recursion can follow it —
reusing §8.29's existing `is_spanning` test rather than introducing a threshold.

On the aggregate it worked: the only variant to beat baseline on holdout,
4.44 % -> 4.25 %, keeping `academic_literature` (4.39 % -> 2.19 %) AND
`newspaper` (5.05 % -> 4.20 %), which no single architecture could hold
together.

**The paired test says no.**

| | better | worse | tied | sign test | mean Δ, 95 % CI |
|---|---:|---:|---:|---|---|
| **holdout** | 25 | **33** | 178 | **z = −1.05** | +0.19 pp, **[−0.93, +1.34]** |
| train | 12 | 8 | 60 | z = +0.89 | +1.36 pp, [−0.72, +3.68] |

**More holdout pages get worse than better.** The +0.19 pp aggregate is a few
large pages outvoting a majority of small regressions — the same 722x
page-length spread that has distorted every character-weighted number in this
campaign (§8.33 as macro/micro, §8.43 as an interval spanning zero, here as an
aggregate pointing the wrong way from the page count). Only 58 of 236 pages
change at all, and among those the router is a coin flip landing slightly
negative.

**Refuted. The default stays `xy_cut`.** All four variants remain named
`FFAI_ORDER` modes so the next attempt starts from measurements rather than
this paragraph.

**Two lessons, both about the aggregate.** The train margin (−1.36 pp) was
never significant either — its CI spans zero — and running the sign test on
TRAIN would have cost nothing and saved a 24-minute holdout run. Train is for
selecting; it still needs an interval before it earns a holdout. And a
character-weighted aggregate can move opposite to the page majority, so on this
corpus the page-level test is not a confirmation of the aggregate — where they
disagree, it is the verdict.

**Where this leaves ordering.** Four attempts, three architecturally distinct,
none transferring. The failure is no longer plausibly "the wrong rule": routing
at PAGE granularity cannot work when the same page contains both a uniform grid
and a structure change, which is what a figure inside a two-column paper IS.
The next attempt should route per NODE inside the recursion rather than once
per page — the same discriminator, applied where the ambiguity actually lives.

### 8.45 Per-node routing refuted — five variants, and the aggregate lies twice more

§8.44 ended by naming the next attempt: route per NODE inside the recursion
rather than once per page, because a single page routinely contains both a
uniform grid and a structure change. `pernode` does that — at every node, a set
with no page-wide element and real gutters is split into ALL its columns at once
(column-major, recursing inside each), and anything else falls through to the
ordinary larger-valley cut.

**It produced the best aggregate of the session and is still refuted.**

| | train | holdout |
|---|---:|---:|
| baseline `xy_cut` | 5.96 % | 4.44 % |
| `hybrid` | 4.60 % | 4.25 % |
| **`pernode`** | **4.16 %** | **3.74 %** |

| paired, holdout | better | worse | tied | sign test | mean Δ, 95 % CI |
|---|---:|---:|---:|---|---|
| `hybrid` | 25 | **33** | 178 | z = −1.05 | +0.19 pp, [−0.93, +1.34] |
| **`pernode`** | 20 | **24** | 192 | **z = −0.60** | +0.69 pp, **[−0.28, +1.73]** |

A 3.74 % against a 4.44 % baseline is a 16 % relative improvement and it is not
real: **more pages get worse than better.** Both routing variants help a handful
of large pages substantially and hurt slightly more small ones slightly, and
character-weighted averaging over a 722x page-length spread reports that as
progress. This is the third distinct way that spread has produced a misleading
number (§8.33 macro/micro, §8.43 an interval spanning zero, here an aggregate
pointing opposite to the page count).

**Five variants, three architectures, none transfers:**

| variant | idea | holdout verdict |
|---|---|---|
| `vfirst` | prefer columns at every node | 4.65 %, worse |
| `vtop` | prefer columns at depth 0 | 4.85 %, worse |
| `onelevel` | explicit column grid, no recursion | falsifier fired on newspapers |
| `hybrid` | route per page on spanning element | aggregate up, pages down |
| `pernode` | route per node | aggregate up, pages down |

**The conclusion is architectural, not another rule.** Every variant reorders
the same geometric evidence — whitespace valleys and box extents. The two that
improved the aggregate did so by trading many small regressions for a few large
wins, which is what happens when the decision is right on the pages where
structure is obvious and arbitrary everywhere else. **Projection-based ordering
has a ceiling here**, and clearing it needs information the projection does not
carry: which boxes belong to the same logical block, which PP-StructureV3 gets
from a layout model and geometry cannot infer.

That is a real deliverable — it says the next investment is a layout classifier,
not a sixth cut rule — and §8.43's decomposition still stands behind it: order
owns 89 % of the gap, so the prize remains the largest on the board.

**A power note.** `pernode`'s train sign test (z = +1.15) was declared a stopping
condition beforehand and then overridden, correctly: 61 of 80 pages were TIED,
so at n = 19 changed pages the test needed ~15-4 to reach |z| > 2 and could not
have confirmed anything. Refuting there would have been refuting on an
instrument that could not confirm — and a wrong refutation is permanent. Holdout,
with 44 changed pages, could decide, and did.

### 8.46 rusty_jpeg 0.2.1 — progressive fixed upstream, 80/80 byte-identical

§8.32's guard is upstream. `rusty_jpeg` 0.2.1 decodes progressive JPEGs with no
local patch; the `"0.1"` -> `"0.2"` bump needed no code change in `ffai-media`.

Verified against the locally-patched build across the whole train split:
**80/80 byte-identical**, including all five progressive pages. Every measurement
in §8.33-§8.45 therefore stands without re-baselining. (0.1.6 and 0.1.7 did NOT
contain the fix — the unwrap was still hoisted above the loop guard in both, and
both still panicked.)

**One false diff, worth recording.** `omni-0130` initially differed — a single
confidence, 0.9646 against 0.9647, on one line of 350. It is a real PNG and never
touches the JPEG decoder. The dump had been written while a second `ocr_text`
process was running; re-run in isolation it is identical, and the binary
reproduces 3/3. **Our recognition confidences are not bit-reproducible under CPU
contention** — non-deterministic parallel reduction order, ~1e-4, far too small
to move text, boxes or ordering. Byte-identity checks must therefore run in
isolation, or they report differences that are not there.

### 8.47 The ceiling: a perfect layout model is worth 11.95 points

§8.45 concluded the next investment is a layout classifier. Before building one,
this campaign's rule is ceiling first, then cost — measure what a perfect fix is
worth before building the thing that approximates it.

The ceiling is computable exactly, with no model and no new runs. Take our OWN
detected lines carrying our OWN recognized text, and emit them in the order the
annotation says the regions should be read, each line inheriting the `order` of
the region its centre falls inside (§8.34's validated geometric labelling). That
is our pipeline with an ORACLE layout model bolted on: identical detection,
identical recognition, identical over-generation, perfect sequence.

| 236 holdout pages, micro CER | |
|---|---:|
| shipped ordering | **24.77 %** |
| **ORACLE ordering** | **12.83 %** |
| **prize for all ordering work** | **11.95 pp** |

**Nearly half our CER is sequence.** Nothing built can exceed this, and every
variant in §8.44-§8.45 was competing for a share of it — the best of them
(`pernode`, refuted) captured about 0.7 pp of aggregate against an 11.95 pp
prize, which is roughly 6 %.

This is the number that justifies the investment. A layout classifier is a large
build; a prize of 11.95 pp on the metric the campaign is judged by is large
enough to warrant it, and it also reconciles with §8.43 from a different
direction: order owns 89 % of the gap to PP-StructureV3, measured against a
reference, and 48 % of our absolute CER, measured against an oracle. Both say
the same thing about where the work is.

**A correction to the probe, made before recording the result.** The order-free
column (20.33 %) was initially labelled a floor. It is not — the oracle beats it,
12.83 % against 20.33 %. Sorting tokens on BOTH sides destroys the alignment
structure Levenshtein exploits, so order-free is a DIAGNOSTIC separating
sequence errors from character errors, not a bound. Reading it as a floor would
have understated the prize by half and made the layout classifier look barely
worth building.

**What it does not include.** The oracle inherits our detection and our
over-generation: 3382 holdout lines fall outside every annotated region and are
kept at the end, so §8.33's separate 6.7 pp over-generation prize is still on the
table and is largely independent of this one.

### 8.48 The prize is 90 % ORDERING, 10 % grouping — the cheap build is refuted

§8.47 priced a perfect layout model at 11.95 pp. Such a model supplies two
things this campaign had been treating as one: **grouping** (which lines form a
block) and **ordering** (what sequence the blocks go in). The distinction
decides the build. Grouping is geometric clustering — cheap, no model. Ordering
is what a learned layout model predicts, and is not cheap.

It is also the obvious hypothesis after §8.44-§8.45: all five refuted variants
reorder individual LINES while the oracle orders REGIONS, and §8.39 already
established that on this problem the units matter more than the rule.

Measured by inserting a grouping oracle and nothing else — our rule's actual
line ordering, with blocks merely made contiguous, each block taking the
position of its first-emitted line:

| 236 holdout pages, micro CER | |
|---|---:|
| A — shipped (our lines, our rule) | 24.72 % |
| B — oracle **grouping**, our ordering | **23.47 %** |
| C — oracle grouping **and** ordering | 12.83 % |

**Grouping alone is worth 1.24 pp (10 % of the prize). Ordering is 10.65 pp
(90 %).**

**The cheap build is refuted.** A geometric block detector — connected
components, proximity clustering, whatever — buys about a point. Knowing which
lines belong together barely helps; the difficulty is entirely the sequence.
That kills the shortcut this probe was written to find, which is what it was
for.

It also explains the five refutations more precisely than §8.45 did. Those
variants were not failing to group; they were competing for the 90 % and
capturing ~6 % of it. The ceiling is not in how lines are bundled, it is in the
evidence available to decide sequence — and a projection over box geometry
simply does not contain "this column continues from that one".

**A first attempt at B was thrown away, and the reason is now familiar.** It
ordered one representative box per block and scored **51.88 %** — twice as bad
as shipped — because `order_reading`'s projection is calibrated for dense line
sets, and ~10 sparse representatives make every gap look like structure. That
measures "our rule on a sparse sample", not "our rule on blocks". It is the same
error as §8.39 (region-scale boxes into a line-scale rule) and would have
reported grouping as catastrophically harmful. **Seventh instrument failure in
this campaign, caught before it was recorded rather than after.**

**Where this leaves the build.** The next investment is reading-order
prediction, not region detection: the model has to output a SEQUENCE, and
geometry cannot supply it. That is the same capability PP-StructureV3 ships and
the reason §8.43 finds order owning 89 % of the gap to it.

### 8.49 Geometry refuted at every granularity — and what the fix costs

§8.48 concluded that the prize is 90 % ordering and that reading-order
PREDICTION is needed. That conclusion had a hole: PP-StructureV3 does not run a
sequence model — it detects regions and orders them geometrically — so if
region-scale ordering works when properly calibrated, region DETECTION would be
the whole build and no sequence model is required. §8.39's failure at region
scale was a calibration bug (line-height units applied to region boxes), not
proof the approach fails.

Tested directly, with true regions and the page's true LINE height as the scale:

| 236 holdout pages, micro CER | |
|---|---:|
| shipped (our lines, our rule) | **24.72 %** |
| true regions, XY-cut at line scale | **35.94 %** |
| true regions, true order | 12.83 % |

**Given perfect regions AND correct calibration, geometric ordering is worse
than what we ship.** The mechanism is arithmetic: with few large units every
wrong decision relocates a whole block of text, so cost-per-mistake grows faster
than the projection's accuracy does. Ordering 100 lines badly is cheaper than
ordering 8 regions badly.

**Geometry is now refuted at all three granularities it can operate on:**

| units | attempts | result |
|---|---|---|
| lines | 5 variants (§8.44, §8.45) | none transfers; best captured ~6 % of the prize |
| contiguous blocks | grouping oracle (§8.48) | 10 % of the prize |
| true regions | XY-cut at line scale | **−94 %** — worse than shipped |

So the missing ingredient is not a better projection, a better granularity, or
better regions. PP-StructureV3's layout models emit CLASSES — title, text,
figure, caption, header — and its ordering uses that semantics, which is
information no projection over box geometry contains. That is what §8.43's 89 %
is measuring.

**The cost, since the ceiling is known.** PP-StructureV3's layout stack as
installed here:

| model | size |
|---|---:|
| PP-DocLayout_plus-L | 130.4 MB |
| PP-DocBlockLayout | 129.9 MB |
| *(our entire detector)* | *4.7 MB* |

**~260 MB against our 4.7 MB** — a 55x increase in weights to buy 11.95 pp. That
is a real tension with the position §8.30 actually won on: 425 MiB steady and
0.17 pages/s on CPU, against 8745 MiB and 0.01 pages/s. Buying the CER at that
price would forfeit the deployment class that is our advantage.

**So the decision is a trade, not a task**, and it should be made deliberately:

1. **Port a large layout model** — closes most of 11.95 pp, costs the edge story.
2. **Find or train a SMALL reading-order head** — the prize is sequence over
   already-detected boxes, which is a much smaller problem than layout detection
   from pixels; a few MB is plausible. Unmeasured.
3. **Bank §8.33's 6.7 pp instead** — over-generation suppression is independent
   of ordering, needs no new model class, and is unexplored since its cheap
   filters were refuted.

Option 2 is the one that preserves the position and is unpriced; option 3 is
smaller but cheap and orthogonal. Both are better next moves than a sixth cut
rule, which §8.44-§8.49 have now closed off with measurements at every level.

### 8.50 The 6.7 pp over-generation prize is benchmark-fitting, not a fix

§8.49 recommended banking §8.33's 6.7 pp over-generation prize as the cheap
parallel win. **That recommendation was wrong and is withdrawn.** Before
building the suppressor, the question worth asking is what the suppressed text
actually IS — and the answer is that it is real, correctly-read page content.

1807 out-of-region holdout lines of six characters or more, median confidence
**0.981**, random sample:

```
JIGME DORJE/XINHUAAP                                 photo credit
PA Tate, J.G. Dorsey / J Chromatogr. A 1103 (2006)   running header
02018 KPMG International Cooperative ("KPMG Int…")   copyright footer
622   Chapter 10   Conics                            running head
(242) 635-9988                                       contact detail
Attribution - NonCommercial License, where it is …   licence footer
Siemens 1996-2007                                    footer
```

Running heads, footers, page numbers, photo credits, copyright lines. This is
precisely OmniDocBench's `abandon` class, which the corpus builder drops "as the
benchmark intends" (`tools/carmenta_omnidoc_corpus.py`). **We are charged for
reading text that is genuinely on the page and that the benchmark chose not to
score.**

So suppressing it improves the CER without improving the extraction. §8.35
already showed PP-StructureV3 reads the same furniture, so the points are
available — which is exactly what makes this tempting and exactly why it should
be named. **Taking 6.7 pp by matching an annotation policy is fitting the
benchmark, and this campaign's ledger is supposed to mean something outside it.**

**What is legitimate here, stated separately.** Furniture suppression IS a real
product behaviour — prose extraction for RAG usually does not want a running
header repeated on every page. The honest form is a FLAG with its own
justification, measured on whether users want furniture-free output, and its CER
effect reported as a consequence of that choice rather than as a quality
improvement. What is not legitimate is turning it on by default because it
happens to score better on OmniDocBench.

**This leaves §8.49's option 2 as the only clean lever:** a small reading-order
head over already-detected boxes. The 11.95 pp ordering prize is not
benchmark-fitting — reading a two-column paper in the wrong sequence is wrong by
any measure, for any consumer, and `omni-0038`'s left-right-left-right interleave
would be wrong if no benchmark existed. That distinction is the one that matters
when choosing what to build.

**A note on how close this came to being built.** §8.49 recommended it in a
sentence, on the strength of a number (6.7 pp) that was correctly measured and
whose MEANING had never been examined. The measurement was right; the inference
from it was not. Checking what a number is made of costs one sample and is the
last thing that gets done.

### 8.51 A learned ranker over geometry loses to the hand rule — the limit is the input

§8.49 refuted geometric ordering, but every attempt was a hand-written RULE.
That does not distinguish "the information is absent from the geometry" from
"the information is there and hard to express in a projection", and the two cost
very differently: a few-MB head over boxes, or PP-DocLayout's ~260 MB of pixels
and classes.

Priced with a pairwise ranker — for every pair of lines on a page, predict from
geometry alone whether A precedes B, then order by Copeland score. Trained on
TRAIN, evaluated on HOLDOUT, scored with §8.40's inversion metric so it sits
directly beside the shipped rule. Features are deliberately the same information
the cut rules had, so the comparison isolates RULE vs LEARNED on identical input.

| model | params | train | **holdout** |
|---|---:|---:|---:|
| logistic | ~15 | 5.77 % | **7.00 %** |
| GBDT-100 | ~1500 nodes | 2.60 % | **4.80 %** |
| *shipped hand rule* | 0 | 5.96 % | **4.44 %** |

**Neither beats the hand-written rule.** GBDT fits train to 2.60 % and lands at
4.80 % on holdout — it learns the training pages, not the problem. So the
limitation is not that `xy_cut` is poorly written. **The reading order is not
recoverable from box geometry**, and every approach that consumes only boxes —
rule or model, cheap or expensive — is bounded by the same wall. That closes
§8.49's option 2 as stated.

**What is NOT refuted, and is the obvious next probe.** These features were
positions and sizes only. We also recognise the TEXT, for free, before ordering
is decided — and text carries exactly the signal geometry lacks: "Chapter 4",
"continued on page 7", a sentence ending mid-clause, a caption beginning
"FIGURE 3". LayoutReader-class models use text plus layout for precisely this
reason. **Geometry + text is unpriced, needs no new pixels, and reuses output the
pipeline already produces.** That is the cheapest remaining shot at the 11.95 pp,
and it should be measured the same way before anything is built.

**Standing back.** The ordering campaign now has a clean shape: 11.95 pp
available (§8.47), 90 % of it sequence rather than grouping (§8.48), geometry
refuted at three granularities as a rule (§8.49) and as a learned model (§8.51),
and the one benchmark-fitting shortcut named and declined (§8.50). What remains
is a genuine fork — text-aware ordering at unknown cost, or a large layout model
at 55x our weights — and it is now a decision with numbers on both sides rather
than a hunch.

### 8.52 Hand-crafted text features add nothing — no cheap lever survives

§8.51 closed boxes-only ordering and named the one free input left: the TEXT,
which the pipeline recognises before ordering is decided and which carries what
geometry cannot — a line ending mid-clause, a caption announcing itself, a bare
number that is furniture.

Added to the same ranker as eight per-line flags (ends in `.!?`, trailing
hyphen, starts capitalised, starts with a digit, matches
`figure|table|chapter|section|…`, ALL CAPS, bare number, saturating length):

| features | model | **holdout** |
|---|---|---:|
| geometry only (15) | logistic | 7.00 % |
| geometry only (15) | GBDT-100 | 4.80 % |
| geometry + text (31) | logistic | **6.96 %** |
| geometry + text (31) | GBDT-100 | **4.71 %** |
| *shipped hand rule* | — | **4.44 %** |

**0.09 pp.** Both still lose to the hand-written rule.

**What this does and does not refute.** It refutes *hand-crafted* text flags,
not text. LayoutReader-class models consume learned token embeddings over the
line's actual words, not eight booleans — "continued from page 7" and a
mid-clause break are semantic facts a regex cannot see. So the honest statement
is that the CHEAP form of the text lever is dead, and the informative form
requires a language model over the recognised text, which is no longer a
few-MB head.

**Every cheap path to the 11.95 pp is now closed by measurement:**

| lever | result |
|---|---|
| better cut rule | 5 variants, none transfers (§8.44-§8.45) |
| finer/coarser granularity | refuted at lines, blocks, regions (§8.48-§8.49) |
| learned ranker over boxes | loses to the hand rule (§8.51) |
| + hand-crafted text features | +0.09 pp, still loses (§8.52) |
| suppress over-generation | benchmark-fitting, declined (§8.50) |

What remains costs real weights: a layout model with semantic classes (~260 MB
as PP-StructureV3 ships it), or a text-embedding sequence model over recognised
lines. **The 11.95 pp is real and so is the wall in front of it**, and that is a
more useful position than an untested sixth idea.

**Method note.** The two conditions were run as separate invocations rather than
the single-pass A/B intended — a patch to `main()` failed silently and was not
checked. Same script, data, models and metric, and the feature counts (15 vs 31)
identify which is which, so the comparison stands; but a silently-skipped edit is
the same class of defect this campaign has now recorded eight times, and it was
caught here only because the output had one block where two were expected.

### 8.53 The refutations were on the wrong metric — `pernode` is worth 4.5 points

§8.44 and §8.45 refuted five ordering variants on a paired sign test over
INVERSIONS. That test weights every page equally. **CER does not — it is
character-weighted by definition**, and it is the metric in the ledger, the four
gates, and the three-way against Unlimited-OCR. "Helps a few large pages a lot,
hurts slightly more small pages a little" is a loss under page-counting and a
large win under the metric that ships.

Measured on the same holdout dumps, no re-runs:

| variant | CER | delta | 95 % CI on improvement |
|---|---:|---:|---|
| baseline `xy_cut` | 24.77 % | — | — |
| `vfirst` | 23.25 % | −1.52 pp | [−0.37, +3.39] — spans zero |
| `vtop` | 25.03 % | +0.25 pp | worse |
| `hybrid` | 25.85 % | +1.08 pp | worse |
| **`pernode`** | **20.27 %** | **−4.50 pp** | **[+2.10, +7.27] — excludes zero** |

Train agrees: 28.74 % → **21.67 %**, CI [+2.86, +12.11]. Both splits, interval
clear of zero. **Shipped as default**; `FFAI_ORDER=xycut` keeps the old path.

The four-gate ledger run confirms end to end: **236/236 PASS**, macro CER
33.70 % → **30.87 %**, WER 49.33 % → 46.38 %, steady memory 453 → 373 MiB.

**`hybrid` proves the two metrics can disagree in SIGN** — better on inversions
(4.25 % vs 4.44 %), worse on CER (+1.08 pp). So this is not a rescaling; the
proxy can point the wrong way. On the 44 pages whose order changed, Δinversions
against ΔCER correlates at **r = +0.853**: the proxy tracks direction and
understates magnitude ~6.5x. It is a diagnostic, not a verdict.

**And there was no speed cost.** Measured in isolation, interleaved:
`pernode` 162 s / 24 pages against `xycut` at 176 s and 205 s — two runs of the
SAME config differing by 17 %. The 0.11 vs 0.16 pg/s in the contaminated
four-gate run was CPU contention. **The noise floor is larger than the effect
anyone was worried about**, which is the depth-6 question that should have been
asked before any of it.

### 8.54 The page-relative gutter thresholds are a brake, and the obvious fix is worse

`find_gutters` scales `SPAN_FRAC`, `GUTTER_MIN_FRAC` and `MARGIN_FRAC` by
`page_w`, while `xy_cut_pernode` calls it on node SUBSETS. That is plainly
inconsistent, and the six-whys reasoning for fixing it is sound: on a half-page
node, a heading spanning the whole node measures under `page_w * 0.60`, so it is
projected instead of skipped and **vetoes every gutter in that node** —
reintroducing at depth the exact veto §8.28 added `is_spanning` to prevent. A
real column gutter inside a narrow node likewise falls under a page-relative
minimum and is discarded.

Made node-relative and measured:

| | train | holdout |
|---|---:|---:|
| `pernode` (page-relative) | 21.67 % | **20.27 %** |
| node-relative | 21.36 % | **20.66 %** |
| delta | −0.30 pp | **+0.39 pp WORSE** |
| 95 % CI | [−0.41, +1.56] spans zero | **[−0.70, −0.12] excludes zero** |
| pages | 3 better, 11 worse | 8 better, **30 worse** |

**Refuted, significantly, and reverted.** Train already showed the tell — a
correct-looking change making 11 pages worse against 3 better — and holdout
confirmed it with an interval clear of zero.

**The inconsistency is load-bearing.** A page-relative fraction is STRICTER when
applied to a smaller node, and that strictness is the only thing suppressing
spurious gutters deep in the recursion, where a handful of lines can leave an
accidental vertical band. Relaxing it finds more columns and most are not real.
So the "bug" is a depth-dependent brake that happens to be spelt as a page
fraction. If it is ever made explicit it must stay a brake — a node-relative
minimum with a floor, not a pure proportion.

**Two of the three proposed `find_gutters` optimisations are also moot.** The
per-node `vec![0u32; page_w]` allocation and the O(ink-width) inner loop are
real waste, but §8.53 measured no speed problem to solve: at a 17 % noise floor
the ordering change is not detectable in wall-clock at all. Optimising them
would be optimising something nothing is paying for.

### 8.55 Prometheus on the cut-axis rule — valid, and worth nothing

The full refinery loop, run on Carmenta's most obviously hand-guessed decision:
`xy_cut`'s `if a.1 >= b.1 { Horizontal } else { Vertical }`, the comparison
§8.41 identified as the ordering defect.

| stage | result |
|---|---|
| **harvest** | 1178 nodes, traced from the SHIPPED `xy_cut`, page-tagged |
| **label** | 25 usable; **979 dropped for having only ONE candidate gap** |
| **distill** | unanimous — all 25 prefer horizontal; the shipped rule gets 18 |
| **forge** | `FFAI_ORDER=hfirst`: take horizontal whenever both valleys exist |
| **trial, train** | 21.67 % -> 21.11 %, 3 pages better, **0 worse** |
| **trial, holdout** | 20.27 % -> **20.25 %**, −0.02 pp, CI [−0.56, +0.70], 12 better / 5 worse |

**Directionally right on both splits and worth nothing.** Not refuted — no page
count went negative, unlike §8.54 — simply too small to measure. It stays a
named mode; the default is unchanged.

**The distilled formula was a CONSTANT, and that is the interesting part.** Once
`pernode` routes column layouts to the grid path, the nodes still reaching
`xy_cut` are predominantly the spanning ones (56 % carry a spanning element) —
exactly the case §8.29 built horizontal-first for. The data agreed without a
single exception, so symbolic regression had nothing to discover beyond "always
horizontal". **This session's own earlier win had already removed the contested
population.**

**The mistake was skipping §8.47's rule on a sub-decision.** Ceiling first, then
cost — and the ceiling here was computable before any harvesting: 25 contested
nodes across 80 pages, of which the rule gets 7 wrong, bounds the prize at a
handful of pages. Minutes of counting would have priced it. Instead the loop ran
end to end (two harvests, a labeller rewrite, train and holdout trials) to reach
a number that was knowable up front. **A ceiling check is cheap at every scale,
not just for campaign-sized levers.**

**Two harvest defects worth recording**, both caught by drop-reason counting
that the first version did not have:

* the trace emitted no page identifier and the labeller joined on `page_w` —
  many pages share a width, so it silently attached the wrong regions and left
  **2 labellable rows out of 315**. A join key that has to be guessed is not a
  join key.
* `trace_node` fired in both `xy_cut_pernode` and `xy_cut`, duplicating 149 of
  315 rows.

Neither was visible in the output until the labeller printed WHY each row was
dropped. "313 dropped" is not a diagnostic; "979 one_gap, 162 no_split, 12
no_signal" is.

**What Prometheus is genuinely suited to here**, on this evidence: decisions
that fire on most inputs, where the current rule is a hand-set constant, and
where a ceiling check says the prize is worth the loop. The cut-axis rule failed
the third test, not the first two.

### 8.56 Prometheus cannot close this gap — the gap is not in a constant

§8.55 ran the refinery loop end to end and got "valid, and worth nothing",
because the target fired on 25 of 1178 nodes. Its stated lesson gives three
criteria for a Prometheus target: it must **fire on most inputs**, be a
**hand-set constant**, and have a **ceiling worth the loop**.

The best remaining candidate was the DBNet unclip ratio — `UNCLIP_LINE = 1.5`,
applied to every detected box on every page, controlling crop tightness and
therefore feeding recognition directly. It passes the first two criteria
outright, and there was positive reason to expect slack: the ledger shows unclip
was swept on **CORD receipts**, never on documents, and this campaign has
measured content sign-flips repeatedly. A constant tuned for one content class
sitting unexamined on another is exactly the shape of an easy win.

Ceiling checked FIRST this time, via the existing `FFAI_DB_UNCLIP` override, 30
train pages:

| unclip | CER |
|---|---:|
| 1.2 | 21.42 % |
| 1.35 | 21.18 % |
| **1.5 (shipped)** | **21.10 %** |
| 1.7 | 24.97 % |
| 1.9 | 25.34 % |

**The shipped constant is already at the optimum.** Flat below (0.32 pp across
1.2–1.5), sharply worse above (+3.87 pp by 1.7). No prize in re-tuning it, so
the target fails criterion three and the loop does not run. **Fifteen minutes of
sweeping instead of a full harvest-label-distill-forge-trial cycle** — which is
the entire point of ceiling-first, applied at the right moment for once.

A useful side result: 1.5 is now validated on documents, not just inherited from
a receipt sweep.

**The general answer, which the two checks together support.** Prometheus
replaces *human-guessed formulas* with discovered ones. That is only worth doing
where a guessed formula is leaving value on the table, and Carmenta's are not:
the cut-axis rule is right on the population that reaches it (§8.55, 25/25), and
the unclip constant is at its optimum (here). Meanwhile **89 % of the remaining
gap is sequence** (§8.43), and §8.51/§8.52 measured that reading order is *not
recoverable from box geometry at all* — not by a rule, not by a learned ranker,
not with hand-crafted text features.

**So the gap is not a badly-tuned formula; it is missing information.** No
symbolic-regression pipeline can discover a fact its inputs do not contain.
Closing the rest needs layout semantics — region classes, or a text-embedding
sequence model — which is a new model, not a better constant. Prometheus is the
right tool for the codec work it was built for and the wrong tool for this gap,
and that is now measured rather than asserted.

**What did close 2.15 points**, for the record: per-node reading-order routing
(§8.53), re-measured on the three-way subset that had never been re-run since it
landed — 25.91 % -> **23.76 %**, deficit against Unlimited-OCR 10.40 pp ->
**8.25 pp**. The lesson there is smaller and duller than a refinery: **when a fix
lands, re-run the benchmarks that quote the old number.**

### 8.57 Engine selection re-measured — the CRNN default is right on the merits

§8.56's closing lesson is that a landed fix invalidates every benchmark quoting
the old number. Applied to engine SELECTION, which had never been re-run since
three defects were fixed: the PARSeq long-word overflow (which crashed
`mobiledet-parseq` mid-page), `pernode` (worth 4.5 pp, applied to every arm),
and the progressive-JPEG panic (35 unreadable pages).

`mobiledet-parseq`'s 31.56 % had been measured under all three. Re-measured on
the same 43-page subset with today's build:

| engine | CER | correctness |
|---|---:|---|
| **`mobiledet-crnn`** | **23.76 %** | 43/43 |
| `mobiledet-parseq` | 31.05 % | 43/43 |

**Still 7.3 points worse.** The gap narrowed slightly (31.56 -> 31.05) but the
ranking is unchanged, so the CRNN default was correct on the merits rather than
by accident of a crashing competitor. `composed-crnn` turns out not to be a
registered engine at all — a name in the example's dispatch that never reached
the registry, worth noting so it is not benchmarked again.

That closes the last cheap lever. Engine selection, unclip, and the cut-axis
rule are all measured and none moves the deficit; §8.51/§8.52 measured that
ordering cannot be improved from box geometry by any means tried. **What remains
is layout semantics — a new model.**

**The gap today**, on the subset where all three engines ran through one harness
and one metric:

| | CER | |
|---|---:|---|
| Unlimited-OCR (3B MoE, GPU) | 15.51 % | |
| PP-StructureV3 | 19.14 % | |
| **Carmenta** | **23.76 %** | was 25.91 % this morning |

Deficit **10.40 pp -> 8.25 pp**, from ordering alone, no new weights.

### 8.58 Four constants ceiling-checked, all at optimum — the answer is settled

§8.55 gave three criteria for a Prometheus target: fires on most inputs, is a
hand-set constant, has a ceiling worth the loop. Four of Carmenta's constants
have now been checked against the third criterion — the only one that decides
whether a refinery loop is worth running.

| target | fires on | ceiling check | verdict |
|---|---|---|---|
| cut-axis rule (§8.55) | 25 of 1178 nodes | full loop run: −0.02 pp holdout | too rare |
| unclip ratio (§8.56) | every box | 1.2–1.9 sweep: shipped **1.5 optimal** | no slack |
| DB binarisation | every pixel | 0.20–0.40: **0.09 pp** total spread | flat |
| detector `min_side` | every image | 736 native **21.10 %**, upscaling worse (1800 → 21.13, 2200 → 21.52) | native optimal |

**Every hand-set constant tested is already where it should be.** That is a real
result about the codebase, not a failure of the method: these values were set by
measurement in earlier campaigns and the measurements held.

**So the answer to "can Prometheus close this gap" is no, and it now rests on
four independent checks rather than an argument.** Symbolic discovery replaces
guessed formulas with better ones. There are no badly-guessed formulas left in
the path. Meanwhile §8.43 attributes **89 % of the remaining deficit to
sequence**, and §8.51/§8.52 measured that reading order is not recoverable from
box geometry by a rule, by a learned ranker, or with hand-crafted text features.
**No symbolic-regression pipeline discovers a fact its inputs do not contain.**
What remains needs layout semantics — region classes or a text-embedding
sequence model — which is a new model, not a better constant.

**A methodological note, because it nearly cost a false refutation.** The first
`min_side` sweep returned **21.10 % at 640, 736, 960 and 1280 — four identical
figures to the decimal**. That is not a flat response curve; it is a knob that
never engaged. `min_side` is a MINIMUM short side and these pages are ~1650 px,
so every value tested was a no-op. The binding range is above 1650, where it
upscales — and only the corrected sweep is the measurement.

That is the fourth time this session a probe returned a plausible number while
measuring nothing (§8.39's region-scale testbed, the string-match labeller at 2
rows of 315, the page-width join, and this). **The tell every time was a number
too clean to be real.** A refutation recorded from any of them would have been
permanent and wrong.

**What actually moved the gap today**, for contrast: `pernode` (§8.53), and then
re-running the benchmark that still quoted the pre-`pernode` number (§8.57).
25.91 % → **23.76 %**, deficit against Unlimited-OCR **10.40 pp → 8.25 pp**. No
new weights, no discovered formula — a fix that had landed and a benchmark that
had not been re-run.

### 8.59 The last two constants — and the definitive answer on Prometheus

§8.47's ordering prize, re-measured after `pernode`:

| 236 holdout pages | CER |
|---|---:|
| shipped | **20.27 %** |
| oracle ordering | 12.85 % |
| **remaining prize** | **7.42 pp** |

`pernode` captured **4.53 pp** of the original 11.95 — matching its −4.50 pp CER
exactly, from an independent measurement. So slack remains, and with the axis
rule exonerated (§8.55, 25/25 correct) it lives in the GRID path: `find_gutters`.

Its two governing constants are the only ones on the hot path never swept in any
campaign, and §8.54 gave positive reason to expect drift — they are load-bearing
in a way nobody designed, acting as an accidental depth-dependent brake:

| GUTTER_MIN_FRAC | CER | MARGIN_FRAC | CER |
|---|---:|---|---:|
| 0.012 | 21.10 % | 0.04 | 21.10 % |
| **0.025 (shipped)** | **21.10 %** | **0.08 (shipped)** | 21.10 % |
| 0.045 | 25.47 % | 0.14 | 21.10 % |

`GUTTER_MIN_FRAC` sits at the edge of a plateau — lowering changes nothing,
raising costs 4.4 pp. At optimum. `MARGIN_FRAC` is **identical across 3.5x**: the
gutters on these pages sit well inside the page margins so the exclusion never
fires. **A non-binding knob, the fifth this session**, and it would have been
recorded as "flat, refuted" without checking the digits.

**The definitive answer.** Prometheus has now been applied to Carmenta across a
full refinery loop and six constants:

| target | fires on | verdict |
|---|---|---|
| cut-axis rule | 25 / 1178 nodes | full loop run; −0.02 pp holdout |
| unclip ratio | every box | shipped 1.5 optimal |
| DB binarisation | every pixel | 0.09 pp spread — flat |
| detector `min_side` | every image | native optimal; upscaling worse |
| `GUTTER_MIN_FRAC` | every grid node | at plateau edge — optimal |
| `MARGIN_FRAC` | every grid node | non-binding on this corpus |

**Not one has slack.** These constants were set by measurement in earlier
campaigns and the measurements held. Symbolic discovery replaces badly-guessed
formulas; **Carmenta has none left on this path.**

Meanwhile the 7.42 pp that remains is ordering, and §8.51/§8.52 measured that
reading order is not recoverable from box geometry — not by a rule, not by a
learned ranker over the same features, not with hand-crafted text flags. **A
refinery cannot discover a fact its inputs do not contain.** Closing the rest
needs layout semantics: region classes, or a text-embedding sequence model. That
is a new model, and it is the honest recommendation.

**What closed 2.15 pp today**, by contrast, was neither clever nor discovered:
`pernode` landed (§8.53), and the benchmark still quoting the pre-`pernode`
number was re-run (§8.57). **25.91 % -> 23.76 %, deficit 10.40 pp -> 8.25 pp.**

### 8.60 The missing input, located and priced — figures, not formulas

§8.59 closed the Prometheus question with "a refinery cannot discover a fact its
inputs do not contain". The productive follow-up is not another constant but
**which fact is missing**, and that is measurable.

`find_gutters` projects BOXES, not pixels. A figure sitting between two columns
occupies a band containing no text boxes — so the projection reads it as a
gutter and cuts there. The image contains the fact that it is a figure; the
projection discards it before the decision is made.

Split the residual ordering slack by whether a page carries figures (detected via
`figure_caption` regions, since the figure itself is not an annotated text
region):

| page class | shipped | oracle | ordering slack |
|---|---:|---:|---:|
| **has figures** | 23.25 % | 12.12 % | **11.13 pp** |
| no figures | 18.15 % | 13.38 % | 4.77 pp |

**Figure pages carry 2.3x the slack.** The mechanism §8.41 found on `omni-0038` —
a figure's whitespace outbidding a real column gutter — is not one dramatic page,
it is the dominant residual failure across the corpus.

**This is the first target this campaign has found with measured slack**, after
six constants that had none. And it is a fact the pipeline already possesses:
the image is in hand at detection time and thrown away before ordering. The
discriminator is ink — a real gutter is empty in the ORIGINAL IMAGE; a
figure-induced band is full of it.

**Why this is cheaper than the layout model §8.58 recommended.** It needs no
region classifier and no new weights — only the mean ink density of a candidate
gutter band, sampled from the image the detector already decoded. That is a
formula over a new measurable input, which is precisely the Prometheus shape
that the six exhausted constants were not.

**Cost, stated honestly.** `find_gutters` takes boxes and `page_w`; it has no
access to pixels. Plumbing the image through `order_reading` into the cut is a
real API change across `boxes.rs` and its callers, and this campaign has learned
not to price a build by its plausibility. The ceiling is 11.13 pp on figure
pages; whether an ink test captures it is the next measurement, not a claim.

**Sequence for that work**, following the discipline that produced today's two
retractions: measure the ink separation on annotated figure bands FIRST (a
probe, no plumbing), and only build if the classes separate.

### 8.61 The ink discriminator separates — and is too rare to pay for the plumbing

§8.60 located the residual slack on figure pages (11.13 pp against 4.77 pp) and
recorded the sequence: measure ink separation FIRST, build only if the classes
separate. Both halves now measured.

**The signal separates.** Ink in box-free x-bands, sampled from the original
image across 60 holdout pages, by decile:

```
0.000 0.000 0.000 0.000 0.000 0.000 0.004 0.036 0.083 0.818
                                          ^ clean valley
```

Bimodal with an empty valley between 0.004 and 0.036. About 70 % of candidate
bands are blank paper — real gutters — and 26 % carry ink above 0.02: figures,
rules, shading. **A threshold in that valley would reject exactly the bands that
should not be gutters.** The mechanism §8.60 proposed is real.

**The frequency kills it.** Counting what `find_gutters` actually sees, via the
existing `FFAI_COL_DEBUG`, over 12 train pages:

| | |
|---|---|
| pages where `find_gutters` is called at all | **6 of 12** |
| total candidate free-runs | **22** |
| mean, over all pages | **1.8 bands/page** |

Six of twelve pages never reach the grid path. On those that do, ~3.7 candidate
bands each, of which ~26 % carry ink — so the discriminator would act on roughly
**one band per two pages**, and only some of those change a reading order.

**Priced against the cost:** `find_gutters` takes boxes and `page_w` and has no
pixel access. Threading the image through `order_reading` into every recursion
node is an API change across `boxes.rs` and all its callers, plus a per-node
image sample. That is a substantial change for a lever that fires on half a page
in one, against a corpus where §8.53 measured **run-to-run CER variance of
~0.5 pp**. The expected effect is at or below the noise floor of the instrument
that would have to certify it.

**Refuted on cost, not on mechanism** — a distinction worth preserving, because
the mechanism is sound and the situation changes if the input gets cheaper. If
the detector ever surfaces a probability map or an ink summary alongside its
boxes (it computes one already), the sample becomes free and this is worth
revisiting. Recorded so the idea is not re-derived from scratch, and not rebuilt
without re-checking the frequency.

**The honest close on the Prometheus question.** Seven targets examined: one full
refinery loop, six constants, and this — the only one with measured slack, which
turns out to be unreachable at acceptable cost from the inputs available. The
gap that remains is layout semantics, and it needs a model, not a formula. That
conclusion has now survived every cheap alternative this campaign could
construct.

### 8.62 The Unlimited-OCR decomposition attempt — control botched, see §8.63

§8.61 concluded "what remains is layout semantics". That rests on §8.43's
decomposition — 89 % of the gap is sequence — which was measured against
**PP-StructureV3**, not against Unlimited-OCR. Unlimited-OCR is 3.63 pp better
than PP-Structure, and whether THAT advantage is ordering or characters has
never been measured. A 3B multimodal model plausibly reads glyphs better, not
merely orders them better.

The two answers fund different work:

* advantage is ordering -> our order-free CER sits near theirs, and the layout
  model is the only path;
* advantage is recognition -> there is character gap no ordering work can touch,
  and the next build is the recognizer, not layout semantics.

**Attempted, and the run failed its own control.** Twelve pages through the
`unlimited_ocr_ref` adapter:

| | this run | ledger |
|---|---:|---:|
| Unlimited-OCR CER | **26.59 %** | **15.51 %** |
| pages returning empty | **2 of 12** | — |
| its order-free CER | 35.16 % | — |

Eleven points worse than the pinned figure, two empty pages, and an order-free
score WORSE than its raw score — which a real output cannot produce, since
removing sequence can only help a text-vs-text edit distance here. The arm is
degenerate: the adapter defaults to `--device cuda`, and its `--base-size` /
`--image-size` may not match the pinned run either. **The numbers are not
reported as a finding.**

Validating a new reference run against its own known figure before believing it
is what caught this — the sixth instrument catch of this campaign, and the
cheapest: one comparison against a number already in the ledger.

**What this leaves open, stated plainly.** The recommendation "layout semantics,
a new model" is inferred from the PP-StructureV3 decomposition and **assumes it
transfers to Unlimited-OCR**. That assumption is now explicitly unverified. It is
the same shape as §8.35, where "PP-Structure suppresses figure regions" was
coherent, consistent with every number then available, and false when finally
measured.

**Next**, before any layout-model work is funded: re-run the Unlimited-OCR arm
with the pinned configuration from its ledger row, confirm it reproduces
15.51 %, and only then decompose. If its edge turns out to be recognition, the
whole §8.58–§8.61 recommendation redirects.

### 8.63 VERIFIED — 84 % of the Unlimited-OCR gap is ordering, not recognition

§8.62 declared the run void on a control that was itself wrong. Two of twelve
pages returned EMPTY from Unlimited-OCR, and an empty hypothesis scores ~100 %
CER — dragging its average to 26.59 % against a pinned 15.51 %. **That is §8.31,
committed inside the instrument check written to catch exactly that.** The
`batch_command` in `references.toml` uses the defaults, CUDA was available, and
the command matched the ledger's: there was no config drift.

Rescored on the 10 pages where BOTH engines produced output:

| | CER | order-free | = order |
|---|---:|---:|---:|
| Unlimited-OCR | **13.92 %** | 23.98 % | −10.05 pp |
| ours | 24.34 % | 25.61 % | −1.27 pp |
| **gap** | **+10.41 pp** | **+1.63 pp** | |

Unlimited-OCR's 13.92 % against the ledger's 15.51 % on a different 43-page
subset is ordinary page-set variation, so the arm is sound.

**84 % of the deficit is sequence.** Independently measured against a second
reference, it reproduces §8.43's 89 % against PP-StructureV3 — and order-free we
sit **1.63 pp** from a 3B multimodal model on a GPU. **We read the characters
about as well as Unlimited-OCR does. We assemble them worse.**

**This closes §8.62's open question and CONFIRMS the recommendation rather than
redirecting it.** The concern was that Unlimited-OCR's 3.63 pp edge over
PP-StructureV3 might be recognition, which no ordering work could touch. It is
not: its order-free CER is *worse* than ours (23.98 % vs 25.61 % is within the
noise of a 10-page sample, and its raw score beats ours by 10.41 pp). The
advantage is almost entirely in assembling the page.

So the whole §8.58–§8.61 conclusion stands, now verified against the reference it
is actually measured against: **the remaining gap is layout semantics.** Six
constants have no slack, geometry cannot express reading order at any
granularity, the ink discriminator is too rare to pay for its plumbing — and the
one thing that would close it is a model that knows a caption from a column.

**Recognition is explicitly NOT the target.** 1.63 pp order-free against a 3B
model is the strongest single result this campaign has produced, and it says
plainly where not to spend: not the recognizer, not the detector constants, not
another cut rule.

### 8.64 The layout-model path needs NEAR-ORACLE ordering — a good ranker is not enough

§8.63 verified that 84 % of the Unlimited-OCR gap is sequence and recommended
layout semantics. Two measurements now sharpen that into something buildable —
or not.

**Region CLASSES add nothing.** A pairwise ranker over region geometry, trained
on train and evaluated on holdout, orders regions to **5.26 %** inversions.
Adding the TRUE region class (title / text_block / header / footer /
figure_caption / page_number) as features: **5.24 %**. A 0.02 pp difference.
Knowing a box is a caption rather than a body block does not help order it — so
"layout semantics" in the sense of a region CLASSIFIER is refuted before anything
is built.

**And region-level ordering is savagely error-sensitive.** Feeding that ranker
perfect region grouping and scoring end-to-end CER:

| 236-page holdout, micro CER | |
|---|---:|
| shipped (line-level ordering) | **20.27 %** |
| oracle grouping + learned region ranker (5.26 % inversions) | **30.63 %** |
| oracle grouping + oracle order (0 % inversions) | 12.85 % |

**Perfect grouping plus a good ranker is 10 points WORSE than what we ship.**
5.26 % of region inversions costs 17.8 pp of CER, because a misplaced region
relocates a whole block of text — the same arithmetic §8.49 found, now priced
end-to-end. Our line-level cut makes far more ordering mistakes and each costs
far less.

**What this means for the build.** "Detect regions and order them" — PP-Structure's
architecture — only beats line-level ordering when the ordering is essentially
perfect. The gradient is unforgiving: between 0 % and 5.26 % region inversions
lies the entire difference between 12.85 % and 30.63 % CER. Any region-based
approach must land in the first percent or it is worse than doing nothing.

That is a far harder target than "port a layout model", and it explains
something §8.49 left open: PP-StructureV3 scores 19.14 % with this architecture,
so its region ordering must be very close to oracle. Matching it means matching
that, not merely detecting regions.

**Where this leaves the campaign, honestly.** Every path has now been measured:

| path | verdict |
|---|---|
| better cut rule | 5 variants refuted on holdout |
| finer/coarser granularity | refuted at lines, blocks, regions |
| learned ranker over boxes | loses to the hand rule |
| hand-crafted text features | +0.09 pp |
| six hand-set constants | none has slack |
| ink discriminator | separates, too rare to pay for |
| region classes | +0.02 pp — no signal |
| **region detection + learned ranker** | **10 pp WORSE than shipped** |

The 7.42 pp of ordering slack is real and every cheap route to it is closed. What
remains is an ordering model good enough to beat 1 % region inversions, which is
a research problem, not an integration.

**And the honest headline stands:** order-free, we are **1.63 pp** from a 3B
multimodal model on a GPU, at 17x its throughput and 4.7 MB of detector weights
against 6.4 GB. The characters are not the problem and have not been for some
time.

### 8.65 The spec: what any future ordering model has to hit — UNITS CORRECTED IN §8.66

§8.64 showed region-level ordering is savagely error-sensitive but gave only two
points. Measured properly by degrading the oracle order with controlled random
swaps, 236 holdout pages:

| swaps/page | region inversions | CER |
|---:|---:|---:|
| 0 | 0.00 % | **12.85 %** |
| 1 | 15.61 % | 23.10 % |
| 2 | 20.44 % | 29.92 % |
| 3 | 28.02 % | 35.42 % |
| 5 | 34.47 % | 44.54 % |
| 8 | 38.21 % | 47.67 % |

**Shipped line-level ordering reads 20.27 %.** Interpolating the first segment, a
region-based approach must stay under roughly **11 % region inversions** — about
ONE swap every one-and-a-half pages — merely to MATCH what a projection over
line boxes already achieves. To reach the 12.85 % oracle it must be essentially
perfect.

That is the spec, and it is a demanding one. The learned ranker of §8.64 sits at
5.26 % inversions and still lands at 30.63 % CER — worse than shipped — because
the curve above is measured on RANDOM swaps while a ranker's errors are
correlated and concentrated on the hard pages. **A model can be well inside the
random-swap budget and still lose.**

**What the whole campaign now says, in one line each:**

* the characters are fine — order-free we are **1.63 pp** from a 3B multimodal
  model on a GPU (§8.63);
* **84 %** of the remaining deficit is sequence, verified against that model
  directly, reproducing the 89 % measured against PP-StructureV3 (§8.43, §8.63);
* every cheap route to it is measured and closed — five cut variants, three
  granularities, a learned ranker, text features, six hand-set constants, the
  ink discriminator, region classes, and region-detection-plus-ranker
  (§8.44–§8.64);
* the slack is **7.42 pp** and reaching it needs an ordering model whose errors
  stay inside a budget that a good geometric ranker already fails.

**And what shipped today**: `pernode`, worth **4.5 points** of CER, taking the
holdout from 201/236 FAIL at 24.77 % to **236/236 PASS at 20.27 %**, and the
deficit against Unlimited-OCR from **10.40 pp to 8.25 pp** — with no new weights,
no new model, and no speed cost.

### 8.66 Inversion rate does not predict CER — spec in CER, not in inversions

§8.65 offered a budget ("stay under ~11 % region inversions") derived from a
random-swap curve. Two measurements show that budget is unsound, and the second
one is a unit error of my own.

**The character-weighting hypothesis is refuted.** The learned region ranker's
errors, counted two ways on holdout:

| | |
|---|---:|
| raw inversions (pooled over pairs) | 11.54 % |
| character-weighted inversions | 12.40 % |

Only 1.1x. The ranker is **not** preferentially misplacing the large regions, so
that is not why it produces 30.63 % CER.

**And the spec mixed units.** §8.64's 5.26 % was a MEAN OF PER-PAGE inversion
rates; the 11.54 % above is POOLED over all pairs. Same errors, two averagings —
precisely the macro/micro distinction that already produced a wrong verdict in
§8.53, committed again three sections later.

**What survives is the negative result, and it is the useful part.** A ranker at
5.26 % page-mean inversions produces **30.63 %** CER while random swaps at
15.61 % produce **23.10 %**. Whatever the averaging, **a lower inversion rate is
giving a much worse CER** — so the inversion metric does not order candidate
orderings the way CER does, and no budget expressed in it is trustworthy.

**The corrected spec is therefore trivial and honest: evaluate on CER.** Any
future ordering model must be measured end-to-end against the 20.27 % that
line-level ordering already achieves, on holdout, with a bootstrap interval.
Inversions remain useful for LOCALISING a defect — that is how §8.41 found
`omni-0038`'s interleave — but not for gating a change. That is the same
conclusion §8.53 reached about a different proxy, and it is now the second time
this campaign has been misled by optimising a stand-in for the shipping metric.

**The durable rule**: this campaign has now been burned twice by proxies —
page-weighted inversions for character-weighted CER (§8.53), and region
inversions for end-to-end CER (here). **Gate on the metric that ships. Use
proxies to find bugs, never to accept changes.**

### 8.67 The eighth target is not on this path — and the search is closed

`group_lines` was the strongest remaining candidate on paper: two hand-set
constants (a 50 % vertical-overlap merge rule and a 3-line lookback), firing on
every box, and independently implicated — §8.36 named "tilt-sensitive line
grouping" as the diagnosed cause of the photo-domain gap.

Exposed via `FFAI_LINE_OVERLAP` / `FFAI_LINE_BACK` and swept:

| overlap | CER | lookback | CER |
|---|---:|---|---:|
| 0.3 | 21.10 % | 2 | 21.10 % |
| **0.5 (shipped)** | 21.10 % | **3 (shipped)** | 21.10 % |
| 0.7 | 21.10 % | 6 | 21.10 % |

Six identical figures — the "too clean to be real" tell, for the sixth time this
session. Checked with extremes: `overlap=0.01` and `overlap=0.99` produce
**byte-identical output**. The knob does not bind.

**Because `group_lines` is not on the document path at all.** `DetStage::MobileDet`
builds boxes straight from DBNet's probability map, and DB emits LINE-level
regions — each detected box already is a line. `group_lines` serves the CRAFT
path, which is precisely where §8.36 diagnosed the tilt sensitivity, on CORD
photographs with `craft-parseq`.

**Not applicable, not refuted** — a distinction worth keeping, because the
constants may well have slack on the photo path where they do bind. That is a
different corpus and a different campaign.

**The search is closed.** Eight targets examined against §8.55's criteria:

| target | verdict |
|---|---|
| cut-axis rule | full loop run; fires on 25/1178 nodes; −0.02 pp |
| unclip ratio | shipped 1.5 optimal |
| DB binarisation | 0.09 pp spread — flat |
| detector `min_side` | native optimal; upscaling worse |
| `GUTTER_MIN_FRAC` | at plateau edge — optimal |
| `MARGIN_FRAC` | non-binding on this corpus |
| ink discriminator | separates, ~1.8 bands/page — too rare to pay for |
| `group_lines` constants | **not on the document path** |

**Prometheus cannot close this gap**, and that is now measured eight ways rather
than argued. Its premise is that a human-guessed formula is leaving value on the
table; on this path, none is. What remains is 7.42 pp of ordering slack that
§8.51/§8.52/§8.64 show is not reachable from box geometry by rule, by learned
ranker, by text features, or by region detection — and §8.66 established that
even measuring progress on it requires the shipping metric, because every proxy
tried has misled.

**What did close the gap today**: `pernode` (§8.53) and re-running the benchmark
that still quoted the pre-`pernode` number (§8.57). **25.91 % -> 23.76 %,
deficit 10.40 pp -> 8.25 pp**, holdout **201/236 FAIL at 24.77 % -> 236/236 PASS
at 20.27 %** — no new weights, no new model, no speed cost.

### 8.68 The gap floors are a peak, and axis-biasing cannot fix figures

The ninth and last hand-set constant pair, and the most directly aimed at
§8.41's diagnosis: `H_GAP_MIN` (1.35) and `V_GAP_MIN` (0.55), the minimum valley
widths in line-heights before a cut is admissible on each axis. **Their
asymmetry IS the axis bias**, and neither had ever been swept.

| H_GAP_MIN | CER | V_GAP_MIN | CER |
|---|---:|---|---:|
| 0.80 | 22.51 % | 0.30 | 21.10 % |
| **1.35 (shipped)** | **21.10 %** | **0.55 (shipped)** | **21.10 %** |
| 2.50 | 26.40 % | 1.00 | 24.92 % |
| 4.00 | 27.36 % | | |

`H_GAP_MIN` is a **genuine peak** — worse below (0.80 -> 22.51) and sharply worse
above (2.50 -> 26.40). `V_GAP_MIN` sits at a plateau edge. Both optimal.

**And this refutes the obvious fix for the figure problem.** §8.41 diagnosed a
figure's ~19 line-height horizontal gap outbidding a ~2 line-height column
gutter; the direct remedy is to raise the horizontal floor so that whitespace
stops qualifying. Measured, that costs **5.3 points** (21.10 -> 26.40 at 2.5).
The horizontal cut is doing necessary work elsewhere — separating headers,
footers and section breaks — and starving it loses more than the figures cost.

So **the figure problem cannot be solved by shifting a global threshold.** A fix
must reject SPECIFIC bands on evidence about those bands, which is what the
figure-vs-gutter plan (`docs/whys/figure-vs-gutter-test-plan.md`) tests.

Nine Prometheus targets now, none with slack. Every hand-set constant on the
document path is at or beside its optimum.

### 8.69 Figure-vs-gutter Stage 0 — the prize is 4.63 pp

Stage 0 of the plan, on cached `pernode` output, no recognizer runs:

| 236 holdout pages | CER | delta |
|---|---:|---:|
| shipped | 20.27 % | — |
| **all figure pages perfectly ordered** | **15.64 %** | **4.63 pp** |
| every page perfectly ordered | 12.85 % | 7.42 pp |

**62 % of the remaining ordering slack is on figure-bearing pages**, and all 79
of them are mis-ordered — which is why the loose bound (fix every figure page)
and the tight bound (fix only mis-ordered figure pages) coincide exactly.

**PASSES the 1.0 pp gate** — twice the ~0.5 pp run-to-run variance of §8.53 —
so the plan proceeds to Stage 1. This is the first target this campaign has
found with both measured slack AND enough size to certify.

### 8.70 Stage 1 kills the gutter fix — the damage is not where the fix would act

§8.69's ceiling passed at 4.63 pp, so Stage 1 asked the question that decides
buildability: do the gutters `find_gutters` ACCEPTS separate on ink? Measured on
the full train split, sampling the original image over each accepted band:

| | n | median ink | inky (>0.02) |
|---|---:|---:|---:|
| figure pages | 18 | 0.0726 | **94 %** |
| clean pages | 35 | 0.0439 | **86 %** |

Separation is **1.7x** and the distributions overlap heavily — 86 % of gutters on
CLEAN pages are also inky, so a threshold rejects most legitimate gutters, and
the single inkiest band in the sample (0.926) is on a clean page. Narrow gutters
catch descenders and antialiasing; that is why §8.61's clean bimodal split over
ALL box-free bands does not survive restriction to the accepted ones.

**But the count is what kills it.** Across 80 train pages, 29 carry figures and
they contribute **18 accepted gutters between them** — so **38 % of figure pages
have no accepted gutter at all**. A gutter-rejecting discriminator cannot reach
them however well it separates.

**So the figure damage is not happening in `find_gutters`.** It happens in
`xy_cut`'s horizontal-vs-vertical valley competition — §8.41's original
diagnosis, where a figure's ~19 line-height horizontal gap outbids a ~2
line-height column gutter — and §8.68 measured that shifting that threshold
costs **5.3 points**, because the horizontal cut is doing necessary work on
headers, footers and section breaks.

**Stage 1 kill gate fires. Stages 2 and 3 are not run, and no plumbing is
written.** That is what the staging was for: the plan cost about an hour of
measurement instead of a wide API change across `boxes.rs` and every caller,
followed by a train run, a holdout run, and a revert.

**What the 4.63 pp would actually need.** Not a scalar over the accepted band —
the evidence has to arrive BEFORE the valley contest, so the cut can know that a
19-line-height horizontal gap is a figure rather than a section break. That is a
statement about a REGION of the page, not about a candidate cut, and it is the
same conclusion §8.64 reached from the other direction: something that knows a
caption from a column. The prize stays on the books at 4.63 pp; the cheap route
to it is now measured shut.

**Plan doc** `docs/whys/figure-vs-gutter-test-plan.md` updated with the outcome
so the staging and its gates survive for the next attempt.

### 8.71 Ink is dead on both axes — and the page can choose its own ordering

Two ideas run against the 4.63 pp of §8.69.

**Idea 1: the corrected ink test. REFUTED.** §8.70's Stage 1 sampled VERTICAL
gutters, but a figure's damage comes from the HORIZONTAL valley it opens — the
gap that outbids the column gutter. Re-run on the right axis, over the 437
traced nodes where the horizontal cut wins:

| | n | median ink | inky (>0.02) |
|---|---:|---:|---:|
| figure pages | 225 | 0.0300 | 52 % |
| clean pages | 212 | 0.0170 | 42 % |

1.8x separation with p25 ≈ 0 and p75 ≈ 0.09 on BOTH. **Ink is refuted on both
axes**, now properly rather than by aiming at the wrong one. Incidental and
useful: the median winning horizontal valley is **3.0 line-heights, not 19** —
`omni-0038`'s figure gap is atypical, which is why §8.68's global threshold shift
could never work.

**Idea 2: self-diagnosis. The trigger failed; the SELECTION won.**

As a detector, column alternation is weak — r = +0.10, and high-alternation pages
are 40 % of the corpus carrying 53 % of the damage, only 1.3x enrichment. (The
first attempt looked worse still, at r = +0.12 with 2 % of pages: it thresholded
|dx| > 0.5 page widths, while a two-column page has centres 0.45 apart and a
four-column page 0.16. **Every real alternation went uncounted.** Corrected to
LEFTWARD RESETS.)

But detection was the wrong question. The right one is whether a page can pick
between orderings at RUN TIME with no ground truth — and it can:

| 236-page holdout | CER |
|---|---:|
| `xycut` | 24.77 % |
| `vfirst` | 23.25 % |
| `hfirst` | 20.25 % |
| `pernode` (was default) | 20.27 % |
| **selection** | **18.88 %** |

**−1.39 pp, 95 % CI [+0.47, +2.54] excluding zero, 23 pages better against 15
worse.** A positive page count — which `hybrid` and every inversion-judged
variant lacked (§8.53). Implemented in `order_by_selection`; the Rust reproduces
the Python simulation to 0.01 pp.

**Robust to its constant:** every reset threshold from 0.03 to 0.30 beats
`pernode` (−1.32 to −2.03 pp). The RULE carries it, not the tuning — which is
what distinguishes this from the six constants §8.55–§8.68 found already optimal.

**The idea worth keeping.** Five ordering variants were "refuted" individually
(§8.44, §8.45) — but each was the best available choice on PART of the corpus.
They were never wrong rules; they were rules with no arbiter. Leftward resets are
a runtime-computable proxy for column coherence, so the page selects its own
strategy. **This is the first thing in the campaign to beat the shipped default
since `pernode`, and it needed no new information at all** — only a second look
at output we already produce. Detection and recognition run once and are shared;
only the ordering repeats, over boxes already in hand.

`FFAI_ORDER=noselect` reverts to `pernode`; `FFAI_ORDER_SELECT_EPS` tunes the
reset threshold.

### 8.72 Ideas 3 and 4 — both refuted, and one of them taught the selection its limit

**Idea 4: cost-based cut selection. REFUTED, twice over.**

§8.68 showed that raising `H_GAP_MIN` globally costs 5.3 pp because horizontal
cuts do necessary work on headers and footers — the floor is global, the damage
is local. `xy_cut_cost` makes each cut pay for its own harm instead: count the
x-bands present on BOTH sides of a candidate horizontal cut, and prefer the
vertical one when the horizontal severs more than one live column.

| holdout, 236 pages | CER | |
|---|---:|---|
| `pernode` | 20.27 % | |
| **selection, 3 candidates** | **18.88 %** | shipped |
| `cost` alone | **24.73 %** | +4.46 pp, CI [−7.99, −1.23] |
| selection + `cost` | 19.66 % | |

Worse standalone, significantly, with 56 pages worse against 44 better —
counting severed bands sounds principled and fires on section breaks that
legitimately span columns.

**And adding it to the pool made the SELECTION worse**, 18.88 % → 19.66 %. That
is a property of §8.71's architecture worth stating plainly: **the reset score
can prefer a wrong-but-column-coherent ordering, so more candidates is not
free.** A candidate earns its place by being best SOMEWHERE, not by existing.
Removed from the pool, kept as `FFAI_ORDER=cost`; the 3-candidate default
re-measures at exactly 18.88 %.

**Idea 3: the detector probability map. REFUTED, structurally.**

The hope was that DBNet's map sees what ink and boxes cannot:

| region | ink | text-probability |
|---|---|---|
| blank gutter | low | low |
| body text | high | HIGH |
| **figure** | **high** | **low** |

Measured over the 105 winning horizontal valleys with a dumped map:

| signal | figure median | clean median | separation |
|---|---:|---:|---:|
| ink | 0.0089 | 0.0169 | 0.5x |
| **text-probability** | **0.0000** | **0.0000** | **0.0x** |
| ink x (1 − prob) | 0.0089 | 0.0169 | 0.5x |

**The map reads zero in every winning valley, figure and blank alike.** Not a
weak signal — no signal, and the reason is structural: DBNet is a TEXT detector.
A figure is not text, so it looks exactly like blank paper. The one distinction
needed is the one the map cannot make.

That also explains §8.70/§8.71 in retrospect: boxes are THRESHOLDED FROM this
map, so "absence of boxes" and "low text-probability" were always the same
signal in two coats. Testing ink was testing the map, twice, without knowing it.

**Where this leaves the four ideas** (`FFAI_DUMP_PROB` added for the probe):

| idea | verdict |
|---|---|
| 1 — ink on the horizontal axis | refuted, 1.8x |
| **2 — oracle-free ordering selection** | **SHIPPED, 20.27 % → 18.88 %** |
| 3 — detector probability map | refuted, structurally (0.0x) |
| 4 — cost-based cut | refuted, and degrades the pool |

Everything that distinguishes a figure from a gutter WITHOUT a layout model has
now been measured: ink on both axes, box geometry at three granularities, region
classes, learned rankers, text features, nine hand-set constants, and the
detector's own probability map. The one thing that worked needed no new
information at all — it re-read output the pipeline already produces.

### 8.73 What Unlimited-OCR actually does — and why segmentation is the wrong axis

Their adapter leaks the model's raw generation, which settles how they organise a
page. Every region is emitted as a token span:

```
<|det|>CLASS [x0, y0, x1, y1]<|/det|>TEXT
```

**The order of emission IS the reading order.** No sort, no column detection, no
merge — a single autoregressive sequence, each region conditioned on everything
already emitted. Ten classes, not OmniDocBench's six: `text, header, title,
footer, page_number, image, page_footnote, image_footnote, image_caption,
aside_text`. Note `image` — emitted with EMPTY text, purely to localise the
figure, the exact object our box projection cannot see (§8.72: text-probability
reads 0.0000 in every valley, figure or blank).

On `omni-0001` the emission runs page_number, header, then the LEFT column
top-to-bottom (x≈50, y 86→613), then the RIGHT column (x=500, y 85→508), then
`image` and `image_caption` together. **Column-major, furniture first, figure
with its caption.**

**And it is correct.** Mapping their emitted regions onto the annotation by text
match, on four multi-column figure-bearing holdout pages — the population where
§8.60 measured our slack concentrating:

| page | matched regions | their inversions |
|---|---:|---:|
| omni-0001 | 9 | **0.0 %** |
| omni-0006 | 6 | **0.0 %** |
| omni-0007 | 11 | **0.0 %** |
| omni-0013 | 9 | **0.0 %** |

Perfect sequences. They are not winning a benchmark convention; they are reading
the page in the order it is meant to be read.

**Our failure is localised precisely.** Holdout, 236 pages:

| | CER |
|---|---:|
| shipped | 18.88 % |
| true region order + OUR within-region sequence | **12.89 %** |
| true region order + y-sorted within region | 12.84 % |

**0.05 pp apart** — re-sorting inside a region changes nothing. Our
within-column ordering is already right; **the entire 5.99 pp is which region
comes next.**

**Segmentation into bands is REFUTED, and it is the bug itself.** Simulated on
cached output — split the page into N horizontal bands, read column-major within
each, concatenate:

| bands x cols | CER |
|---|---:|
| 1 x 3 | **35.09 %** |
| 2 x 3 | 51.79 % |
| 3 x 3 | 55.54 % |
| 6 x 3 | 62.63 % |

**Monotonically worse with more bands.** Every band boundary severs a column, and
band-major reading crosses columns by construction — that is exactly
`omni-0038`'s `[0,5,1,5,1,5…]`. A 3x3 grid reproduces the figure bug
deliberately. The family's best member is ONE band (no horizontal segmentation
at all), which is pure column-major.

**So the decomposition axis is vertical, not horizontal.** A column is a
contiguous run of the true reading order and can be ordered independently then
concatenated. A band is a slice ACROSS several runs and cannot.

**`order_one_level` in the selection pool: tried, not kept.** Holdout 18.88 % ->
18.65 %, but the CI spans zero ([−0.90, +1.44]), the page count is NEGATIVE
(3 better, 4 worse), only 7 pages changed, and −0.23 pp is under §8.53's ~0.5 pp
run-to-run variance. It is almost never the most column-coherent candidate, so
the reset score rarely selects it. Reverted; reachable as `FFAI_ORDER=onelevel`.

**What remains.** The one case where a band boundary is genuinely correct is
where the column COUNT changes — under a masthead. One boundary on a newspaper,
zero on an academic paper. Distinguishing that from a figure's whitespace is the
open problem, and §8.71/§8.72 measured that neither ink nor the detector's
probability map can do it. Their `image` class can, because it was trained to.

### 8.74 The failure, drawn — and the compound selection score refuted

`.tools-bench/render_order.py` draws the worst-ordered holdout pages twice: our
emission sequence beside the annotation's, each region numbered and joined by a
path, coloured blue -> red along the reading order. Recognition and detection are
identical on both sides, so the picture is sequence and nothing else.

`omni-0001` (+62.1 pp, the worst on holdout) is the whole campaign in one image.
Ours reads **1 left, 2 RIGHT, 3-4 left, 5-6 right, 7-8 left, 9 caption** — the
path crosses the gutter six times. Correct reads **1-5 straight down the left
column, 6-8 down the right, 9 caption** — two clean vertical runs. Same boxes,
same text, 62 points.

Six pages rendered, 41.7 to 62.1 pp: `omni-0001, 0069, 0003, 0116, 0015, 0191`.
Every one is the same shape — a multi-column page read across instead of down.

**The compound selection score is REFUTED.** §8.73 measured 1.91 pp of headroom
in the selection RULE (shipped 18.88 %, oracle pick 16.97 %), which looked like
the cheapest lever left. Seven criteria tested offline on cached dumps, over four
candidates:

| criterion | CER |
|---|---:|
| **leftward resets (shipped)** | **18.78 %** |
| resets + y-backtracks | 18.78 % |
| resets + inverse run-length | 18.86 % |
| resets + 0.5x switches + ybacks | 18.86 % |
| all column switches | 19.07 % |
| inverse run-length | 19.07 % |
| oracle pick | 17.70 % |

**Nothing beats counting leftward resets, and adding signals makes it worse.**
The reason is instructive: `switches` counts EVERY column change, including the
one legitimate switch at a column boundary, so it penalises exactly the
column-major reading it was meant to reward. Only a LEFTWARD move is evidence of
going backwards in the reading order.

So the 1.08 pp still in the rule is real and none of these signals reach it. The
shipped criterion is not a first guess that happened to work — it is the best of
seven, and the only one that distinguishes "moved to the next column" from
"went back".

### 8.75 Two corrections from looking at the pictures, and span-banding refuted

Rendering the failures (§8.74) produced two findings that no aggregate had.

**1. Some annotations read RIGHT-TO-LEFT, and on those pages we are correct.**
`omni-0015` orders its right column before its left; for an English-language
document that is not the practical reading order, and our left-then-right output
is the sensible one. Counted across the holdout:

| two-column holdout pages | count |
|---|---:|
| annotation reads left column first | 139 (95 %) |
| **annotation reads right column first** | **8 (5 %)** |

`omni-0003` and `omni-0015` — two of the six worst-damaged pages — are both in
that set. But the correction is small:

| | ours | annotation | damage |
|---|---:|---:|---:|
| all 236 pages | 18.88 % | 12.89 % | **6.00 pp** |
| excluding the 8 reversed | 18.77 % | 12.95 % | **5.82 pp** |
| the 8 reversed alone | 27.47 % | 8.43 % | 19.04 pp |

They are savaged individually (19 pp) and rare (3 % of pages), carrying **4 %**
of the damage. So the gap is 5.82 pp of genuine error, not 6.00 — real, and not
a rescue. **Chasing those 8 pages would be §8.50's benchmark-fitting.**

**2. Span-banding — band at the ELEMENT, not the gap. REFUTED.**

Every remaining failure is 2-column -> 1-column -> 2-column, and the 1-column
part is a real full-width ELEMENT. Every cut we have built chose the widest
VALLEY instead, which is why a figure wins: a figure is a wide gap with NO text,
a masthead is a wide element WITH text. Measuring the element rather than the
hole makes them different OBJECTS rather than different sizes — which is exactly
why §8.68's threshold shift, §8.71's ink test and §8.72's probability map all
failed, and it looked like the missing distinction.

`xy_cut_span` cuts immediately above the topmost spanning element, falling
through to the vertical cut when none exists. Measured on holdout:

| | CER |
|---|---:|
| shipped selection | **18.88 %** |
| span-banding alone | **25.31 %** (+6.42 pp, CI [−10.05, −3.14], 63 worse / 26 better) |

**And it fails hardest exactly where it was aimed: on multi-column pages WITH
figures it wins 8 and loses 51.**

The mechanism is already in the ledger. Running headers, footers and wide
captions are all spanning elements, so cutting above each one manufactures
MANY bands — and §8.73 measured bands as monotonically harmful (1 band 35.09 %,
6 bands 62.63 %). **The fix reintroduces the disease.** A spanning element is
necessary evidence for a band boundary but nowhere near sufficient; what is
missing is knowing which spanning elements are STRUCTURAL and which are
furniture, which is a class question, not a geometry question.

Kept as `FFAI_ORDER=span`, not in the pool: §8.72's rule is that a candidate
earns its place by being best somewhere, and this is best nowhere that matters.

### 8.76 The §8.75 refutation was WRONG — it tested a defective implementation

§8.75 closed span-banding on ONE measurement, which is exactly what the
three-probe rule forbids, and the follow-up found the stated mechanism to be
wrong. Descending on "how is structural told from furniture":

| | spanning elements | a true boundary |
|---|---:|---:|
| single-column pages | 133 | **94 %** |
| multi-column pages | 95 | **94 %** |

**There is no structural-vs-furniture distinction to find.** A spanning element
is a genuine reading-order boundary 94 % of the time, in BOTH buckets — so the
"which are furniture?" question §8.75 ended on is not the open problem. The
column-structure-changes test proposed as the discriminator is *anti*-correlated
(47 % of structural vs 64 % of furniture), and is refuted.

Nor is detection at fault. Our line-level span test against region truth:

| | count |
|---|---:|
| spanning regions found | 271 |
| flagged but not spanning | 7 |
| missed | 67 |

**precision 97 %, recall 80 %.** So the rule is right and the detection is good,
yet the implementation measured +6.42 pp worse — a contradiction, and therefore
an instrument fault (D6). It is:

`is_spanning()` is evaluated on **LINES, not elements.** A full-width paragraph
is one element but N spanning *lines*:

| per page | median | max |
|---|---:|---:|
| spanning REGIONS | 0 | 13 |
| spanning **LINES** | 0 | **35** |

Each cut consumes one recursion level against `MAX_CUT_DEPTH = 12`, so on
**9 % of pages the recursion is exhausted peeling single lines** and the page
falls through to `sorted_by_y` — raster, the worst ordering measured in this
campaign. Span-banding degenerated into raster order on the pages with the most
full-width text.

**The idea was never tested; a defective implementation of it was.** Lines must
be grouped into elements (contiguous full-width runs) BEFORE banding. §8.75's
"it manufactures bands" was right by accident and wrong in substance — it bands
per line, not per element. A wrong refutation is permanent, so this one is
reopened rather than left to look settled.

### 8.77 What LIVE can and cannot lend the document path

`live.rs` (582 lines) is a temporal-reuse layer: decimated grayscale frame
differencing, `calibrate_bands`, `crop_rows`, and reuse of `prev_out` when the
frame is unchanged. It calls the SAME detector and recognizer and contains no
region classifier and no layout model. **It cannot supply region semantics; it
is strictly less informed, not more.**

Its *premise* does transfer. LIVE bets that layout persists across FRAMES; a
document's layout persists across PAGES. A running header repeats at the same y
on every page and a masthead appears once — which is the structural-vs-furniture
signal obtained from repetition instead of from a 3B model, and the one signal
that needs no semantics at all.

**OmniDocBench cannot reward it:** its pages are sampled individually from
different documents, so there is no cross-page context and the measured gain
would be zero. Worth building for real PDFs, which is what ships; recorded here
so it is not mistaken for a benchmark lever.

### 8.78 Element grouping built and measured — the idea is refuted, properly this time

§8.76 reopened span-banding because the refutation had tested a defective
implementation. Both halves of the fix were built and measured.

**A faster instrument.** Reading order is a pure function of the line boxes, so
an ordering A/B needs no detector and no recognizer — only the lines the engine
already emitted, re-ordered. `.tools-bench/reorder_cer.py` drives the shipped
`order_reading` through `examples/order_probe.rs`; ~27 minutes per arm becomes
seconds, which is the difference between measuring three variants and one.
`order_probe` gained an explicit page-width argument: `is_spanning` scales by
`page_w`, and inferring it from the boxes' own extent makes every full-width
element span by construction.

**Its instrument check FAILED, and the reason is worth keeping.** The dumps were
written in shipped order, so re-ordering them with the default should reproduce
them exactly. It does not: 18.88 % as written against 18.63 % re-ordered.
`sorted_by_y` is a STABLE sort, so ties break by input order — the engine feeds
lines in DETECTION order, the probe feeds them in SHIPPED order, and the two
disagree on ties. The probe is therefore valid for relative A/B against its own
18.63 % baseline and is NOT bit-exact with the engine. Every number below is
probe-relative; the shipped figure remains 18.88 %.

**1. Element grouping works.** `element_tops` merges contiguous full-width lines
into one element, ending a run at the first non-spanning line or a gap wider
than `H_GAP_MIN`:

| | before | after |
|---|---:|---:|
| cuts per page (mean) | 3.1 | **1.4** |
| pages exceeding `MAX_CUT_DEPTH` | 21 | **2** |
| pages whose cut list changes | — | 62 of 236 (26 %) |

The §8.76 defect was real and is fixed.

**2. And the idea still loses.** Standalone, span-banding measures **25.31 %**
against the default's 18.63 % — **+6.68 pp**. `FFAI_ORDER=raster` is 70.38 %, so
this is not the degeneration §8.76 predicted; it is simply a worse ordering.

**3. The pool addition loses too.** The pool does not require a candidate to be
good on average, only to be best SOMEWHERE, and span beat the default on 21
pages standalone — so this was the test that mattered:

| | CER |
|---|---:|
| 3 candidates | **18.63 %** |
| 4 candidates (+ span) | **19.16 %** |

delta **+0.53 pp**, CI **[−1.34, −0.04]** excluding zero, and the page count is
decisive: **1 better, 16 worse**, 219 unchanged.

This is the THIRD candidate refused by this pool after `xy_cut_cost` and
`order_one_level`, always by the same mechanism: **the reset score rewards
column-coherent output, and a wrong ordering can be more column-coherent than a
right one.** Offering a rule per page does not make a bad rule safe, because the
selector is itself only a proxy — §8.74 already measured 1.91 pp of headroom in
the selection RULE, and this is that headroom biting.

Kept as `FFAI_ORDER=span` with the grouping fix. The shipped default is
untouched and verified byte-identical to the pre-change output on 12 of 12
spot-checked pages.

**What this closes.** Every geometric route to the remaining gap has now been
measured and refused: threshold shift (§8.68), ink test (§8.71), probability map
(§8.72), compound selection score (§8.74), element banding (§8.78). The 0.92 pp
ceiling §8.77 priced is real but is not reachable by geometry. What is left is
region CLASS, and the only source of it that needs no model is cross-page
repetition — which this benchmark, sampling pages individually, cannot reward.

### 8.79 Column-relative spanning is a tautology, not a signal

After §8.78 the natural next question is whether the same element grouping
applies one scale down: a "full COLUMN line" rather than a full-page one. It is
a real object — on the 179 multi-column holdout pages, **30 % of lines fill
>= 90 % of their column and 49 % fill >= 60 %** — so the question is fair.

Two uses, two answers.

**As a grouping unit it is capped before it is written.** §8.48 measured an
ORACLE grouping — perfect knowledge of which lines form a block — at 1.24 pp,
10 % of the prize, against ordering's 10.65 pp. A width heuristic captures some
fraction of what the oracle gets for free, so this is bounded well under a
point.

**As a cut signal it is worse than useless, and the reason generalises.** The
routing test is `lines.iter().any(|l| is_spanning(...))`. A column's width is
DEFINED by its widest line, so that line spans 100 % of it **by construction**:
rescaling `is_spanning` to the node makes `any()` unconditionally true.

Today it is unconditionally FALSE below the root for the mirror-image reason —
`is_spanning` is measured against `page_w` at every depth, a column is ~45 % of
page width, and `SPAN_FRAC` is 0.60. In `xy_cut_pernode`, our first pool
candidate, `spans_something` is therefore dead everywhere except depth 0.

That looks like a defect and is not one. **The page-level test is informative
precisely BECAUSE spanning is rare there** — median 0 spanning regions per page
(§8.76). Rarity is what makes it evidence, and rarity is a page-scale property
that does not survive rescaling. Constant-false and constant-true carry the same
zero bits; the root is the only regime where the test means anything.

The general form, which has now cost this campaign three sections (§8.39, §8.78,
this one): **a geometric predicate is evidence only at the scale where its
positives are rare.** Before reusing one at a new scale, measure its base rate
there — if it fires on half the units, or on all of them, it is not a signal
however sound the intuition behind it.

### 8.80 Furniture barely exists — and ordering runs BEFORE recognition

Two facts found while inventorying what could supply region CLASS. The first
corrects §8.75-§8.77.

**Furniture is 0.8 % of this corpus.** Holdout region classes:

| class | n |
|---|---:|
| text_block | 2865 |
| title | 816 |
| figure_caption | 137 |
| header / page_number / footer | **13 / 9 / 8** |

Thirty regions of 3848. §8.75 through §8.77 reasoned at length about separating
structural elements from furniture, and §8.76 measured 94 % of spanning
elements to be true boundaries in BOTH the single- and multi-column buckets —
this is why. **There was almost nothing to separate.** Cross-page repetition
(§8.77) would work and would be chasing 0.8 % of regions; it stays a real
product feature for PDFs and is not a lever on this benchmark or, more
honestly, on this problem.

The live class question is `text_block` vs `title`, 96 % of all regions, and
free text features separate them:

| class | n | Fig/Table prefix | numeric only | <= 60 chars |
|---|---:|---:|---:|---:|
| text_block | 2865 | 0 % | 0 % | **18 %** |
| title | 816 | 1 % | 2 % | **88 %** |
| figure_caption | 137 | **24 %** | 0 % | 45 % |
| page_number | 9 | 0 % | **100 %** | 100 % |

**`order_reading` runs inside the DETECTION stage** (`engine.rs:323`, `:385`) —
before recognition. Ordering sees geometry and nothing else, and the text is
computed immediately afterwards and never fed back. The richest signal we own is
discarded by construction.

**Inventory of what could supply class, and what each is worth:**

| tool | have it? | sized to the problem? |
|---|---|---|
| recognised TEXT (length, prefix, numerals) | yes, free | class yes -> but class mostly buys GROUPING, capped 1.24 pp (§8.48) |
| text CONTINUITY across a column join | yes, free | **yes — attacks ordering, the 90 %** |
| line height | yes | weak: 1.15x median structural vs 1.08x furniture (§8.76) |
| layout model (DocLayout-YOLO, PP-Structure head) | **no** — none on disk | real semantics, ~10-30 M params on the candle spine; new dependency |
| train a classifier on the train split | labels exist | same 1.24 pp grouping cap |

**The conclusion is that CLASS is the wrong target.** Class buys grouping and
§8.48 capped all grouping — with an ORACLE — at 1.24 pp. What is large is
ordering (10.65 pp), and §8.74 localised 1.91 pp of it in the SELECTION RULE,
which today scores candidates by leftward resets: pure geometry, the exact
channel §8.68-§8.79 exhausted.

Text continuity is the one available signal that is not geometry. A column
ending `"incon-"` and one beginning `"clusive"` is direct evidence of
adjacency, and no projection over boxes can contain it. That requires ordering
to move after recognition, or a re-ordering pass to be added — which is an
architectural change, and the first one this campaign has had a measured reason
to make.

### 8.81 Text continuity REFUTED — reading order is not textual continuity

§8.80 argued that text is the one signal left outside the exhausted geometric
channel: a column ending `"incon-"` and one beginning `"clusive"` is direct
evidence of adjacency that no projection over boxes contains. It was the first
idea in this campaign with a measured reason to change the architecture. It is
wrong, and three probes on three different axes say so.

**Probe 1 — as a SELECTION rule.** The pool emits three candidates per page and
picks by leftward resets. Replacing or augmenting that rule with a continuity
score, over 234 holdout pages:

| selection rule | CER |
|---|---:|
| resets (ships today) | **18.63 %** |
| continuity only | 18.89 % |
| continuity primary, resets tie-break | 18.89 % |
| continuity gated on margin (0.02 / 0.05 / 0.10) | 18.63 % |
| resets + 3.0 x continuity, blended | **18.59 %** |
| **ORACLE — pick the lowest-CER candidate** | **17.47 %** |

The entire selection-rule prize is 1.16 pp and the best continuity variant
captures **0.04 pp of it — 3 %.**

**Probe 2 — as an ORDERING OBJECTIVE.** A selection test caps the idea at the
3-candidate oracle. The larger claim is optimising continuity directly, which
is bounded by §8.48's 10.65 pp. That requires the TRUE order to score better
than ours. It does not:

| 201 holdout pages | continuity (0 = every join reads clean) |
|---|---:|
| our order | 0.1699 |
| the TRUE order | 0.1687 |

Mean advantage +0.0012, **median 0.0000**, and the true order reads WORSE on
77 % of pages. An objective that does not separate the right answer from ours
cannot guide a search toward it.

**Probe 3 — undiluted.** The obvious rescue is dilution: a 100-line page has
~4 column joins, so a page-averaged score is swamped by the 96 % of joins both
orderings get right. Scoring ONLY the joins where the two orderings differ:

| 184 pages where they differ | |
|---|---:|
| ours | 0.3193 |
| the TRUE order | 0.3168 |

True order better on **26 %** of them. Dilution was not the explanation.

**The premise was wrong. Correct reading order is not textually continuous.**
Documents are full of legitimate discontinuities — a heading follows a
paragraph, a caption interrupts a column, a section starts mid-page — so the
true sequence has broken joins BY CONSTRUCTION. "Reads smoothly" was never a
property that distinguishes the right order from a wrong one. A secondary cause
compounds it: at 18.6 % CER the text is itself noisy, and terminal punctuation
and initial case are exactly the characters an OCR engine misreads.

One positive fragment, recorded because it is real and does not convert:
continuity ranks candidate PAIRS better than resets does — 67 % concordant
against 60 % — but decides far fewer pairs (79 against 171). Higher precision,
lower coverage, and no combination tried turned that into CER.

**What this closes.** §8.80 named text as the last non-geometric signal
available without a new model. It is measured and it is not one. The remaining
gap needs region semantics from a LAYOUT MODEL — DocLayout-YOLO or a
PP-Structure head, ~10-30 M params on the candle spine — or it does not get
closed. That is now a decision about dependencies and scope, not about
algorithms, and every cheaper route has been measured rather than assumed.

### 8.82 The problem was never columns — it is NEWSPAPERS AND MAGAZINES

Every section from §8.68 to §8.81 attacked column geometry: figure-vs-gutter,
threshold shifts, ink tests, probability maps, per-node routing, element
banding, column-scale banding, text continuity. The corpus carries a per-page
`layout` label that was never read. Reading it ends the campaign's central
assumption.

Ordering damage (ours minus the oracle on the SAME lines), 236 holdout pages:

| layout | pages | CER | damage | share |
|---|---:|---:|---:|---:|
| **other_layout** | 84 | 19.57 % | **4.90 pp** | **82 %** |
| `1andmore_column` (the 2->1->2 case) | 25 | 26.19 % | 0.72 pp | 12 % |
| double_column | 36 | 28.37 % | 0.28 pp | 5 % |
| single_column | 58 | 31.29 % | 0.07 pp | 1 % |
| three_column | 33 | **5.30 %** | 0.02 pp | 0 % |

**Column layouts are SOLVED.** Single, double and three-column pages together
carry 6 % of the ordering damage; three-column pages read at 5.30 % CER. The
2->1->2 transition that §8.74-§8.78 were built around is 12 %. The recursive
XY-cut, per-node routing and runtime selection did their job — on the layouts
they were designed for, the remaining error is recognition, not sequence.

**82 % is `other_layout`, and it is editorial design:**

| data_source | pages | share of the 4.90 pp |
|---|---:|---:|
| **newspaper** | 24 | **69 %** |
| **magazine** | 37 | **27 %** |
| colorful_textbook / academic / PPT / exam | 23 | 4 % |

**61 pages — newspapers and magazines, 26 % of the holdout — carry 78 % of all
ordering damage.** Worst pages: `omni-0116` 44.9 pp, `omni-0191` 41.7 pp,
`omni-0185` 40.4 pp, all magazine.

**Why no cut sequence can fix it.** A newspaper page is not a decomposition of
one text into columns; it is SEVERAL INDEPENDENT ARTICLES sharing a sheet, with
text flowing around pull-quotes, advertisements and images. The correct order
requires knowing which block belongs to which STORY. That is not a property of
the geometry — two adjacent columns may belong to different articles, and two
distant ones to the same. Every technique this campaign refuted was searching
for a partition that does not exist on these pages.

It also explains §8.81 cleanly. Text continuity failed because the true order of
a newspaper page is discontinuous by design, and it explains §8.48's 90/10
split: grouping is easy (blocks are visually obvious), sequencing them is the
whole problem.

**This retargets everything.** The gap against Unlimited-OCR is not a column
algorithm we have not thought of; it is that a 3 B VLM reads a newspaper as a
newspaper. Options, in honest order:

1. **Scope the claim.** We are at parity-or-better on academic and business
   documents — the layouts most OCR is actually run on — and behind on
   editorial pages. Saying so is accurate and cheap, and it is what the numbers
   support today.
2. **A layout model** (DocLayout-YOLO / PP-Structure head, ~10-30 M params on
   the candle spine) supplies region class but NOT story membership; it would
   help the 12 % and 6 % buckets that are already nearly solved. **It does not
   obviously address the 82 %** — which should be measured before it is built.
3. **Story segmentation** is the thing that would actually close it, and no
   cheap version of it exists.

The campaign's real result is that the algorithmic work is done and the residue
has a name.

### 8.83 A layout model buys 0.56 of a 7.39 pp prize — and the corpus splits in two

§8.82 named newspapers and magazines as 78 % of the ordering damage and left one
question open: does a layout model (DocLayout-YOLO, a PP-Structure head) touch
it? Priced before building, per §8.77.

A layout model supplies REGIONS and CLASSES. It does not supply reading order.
So its ceiling is §8.48's construction — oracle grouping with our own ordering,
blocks merely made contiguous, each taking the position of its first-emitted
line — measured on the population in question:

| newspaper + magazine, 632 770 chars | CER | |
|---|---:|---|
| A shipped | 14.28 % | |
| B **oracle grouping, our ordering** | **13.72 %** | what a layout model buys: **0.56 pp** |
| C oracle grouping AND order | 6.89 % | the whole prize: **7.39 pp** |

**A layout model captures 7.6 % of what is available on the pages it would be
bought for.** That is a new model, new weights and a new dependency in a
pure-Rust toolkit, for half a point. **Refused on price**, and refused on the
same evidence that §8.48 produced corpus-wide: grouping is not the scarce
ingredient, sequence is.

**The corpus splits into two populations with opposite bottlenecks:**

| | CER | ordering headroom |
|---|---:|---:|
| newspaper + magazine | **14.28 %** | **7.39 pp** |
| everything else | **29.09 %** | 2.90 pp |

Editorial pages are our BEST-RECOGNISED and WORST-ORDERED; everything else is
the reverse. Editorial is ordering-limited, the rest is recognition-limited.
This retires a standing assumption of the campaign — that ordering work helps
the document corpus generally. It helps a quarter of it, and on the other
three-quarters the remaining error is characters, not sequence.

**Two failure shapes seen by rendering (`render_layout.py`), both distinct from
§8.82's story-membership argument:**

* `omni-0069` (+56.3 pp, `double_column`) — a plain two-column page where we
  ZIG-ZAG across the gutter: our sequence alternates left, right, left, right
  while the truth runs down the left column then down the right. A column
  failure, not an editorial one, and the worst magazine page in the holdout.
* `omni-0116` (+44.9 pp) — a contents GRID of tiles. The truth reads
  **row-major** across each row of tiles; we read **column-major** down each
  column. A grid reads across, a column layout reads down, and the XY-cut
  prefers the vertical cut in both.

The grid case is worth pricing separately and is NOT one of the refuted levers:
a true grid has regions aligned on BOTH axes, whereas text columns have no
row alignment, so the distinction is geometric and detectable. §8.79's rule
applies before any build — measure the base rate of grid-aligned pages first,
because a predicate that fires on everything or nothing is not a signal.

### 8.84 The grid lever REFUTED — row/column-major is not a page-level property

§8.83 flagged `omni-0116`, a contents grid whose truth reads ROW-major while we
read column-major, as the one unpriced lever left. Priced, with §8.79's base-rate
check first.

**Base rate passes.** Of 173 holdout pages whose regions form a 2-D arrangement,
the truth reads row-major on **66 (38 %)** and column-major on 102. Non-degenerate
— the predicate fires on neither everything nor nothing, which is the test
§8.79 exists to impose. Row-major pages span every source (magazine 15,
academic 14, book 10, newspaper 10).

**And the ceiling is NEGATIVE.** The decision only exists at block level, so it
is measured with grouping given, isolating the row/column choice:

| newspaper + magazine, 632 770 chars | CER |
|---|---:|
| oracle grouping, **our** block order | **13.72 %** |
| all blocks ROW-major | 58.43 % |
| all blocks COLUMN-major | 25.76 % |
| **ORACLE — pick row/col-major per page** | **23.42 %** |
| oracle TRUE order | 6.89 % |

**Even a perfect per-page choice loses 9.70 pp to what we already do.** Swept
across 12 row/column banding thresholds (row gap 0.02-0.15, column gap
0.05-0.20) the best oracle is 19.21 %, still 5.5 pp worse than baseline. Not a
parameterisation artifact.

**Why: real pages are not globally row- or column-major.** A magazine page is a
title THEN a grid; an article is a masthead THEN columns. A global 2-D sort
cannot express that hierarchy, and expressing it is precisely what the recursive
XY-cut does — which is why it beats both orderings by a wide margin. The
`omni-0116` observation was a correct description of ONE PAGE'S TILE REGION and
wrong as a page-level rule: grid-ness is a LOCAL property of a sub-node, not of
a page.

The narrower form — row-major applied only inside a node detected as a grid —
remains logically open, but it is a poor bet: it would enter the candidate pool,
and that pool has now refused three candidates (`xy_cut_cost`, `order_one_level`,
`xy_cut_span`) because the reset score cannot reliably pick a winner. §8.81
priced the entire selection-rule prize at 1.16 pp.

**This closes the ordering campaign.** Every lever is measured: threshold shift
(§8.68), ink test (§8.71), probability map (§8.72), compound score (§8.74),
element banding (§8.78), column-scale banding (§8.79), text continuity (§8.81),
layout model (§8.83, 7.6 % of its prize), grid detection (§8.84). What remains
on editorial pages is reading order that requires knowing the document, and
§8.83's split says the other three-quarters of the corpus is
recognition-limited anyway.

### 8.85 A real defect found by LOOKING: lines merged across the gutter

Reviewing the rendered failures turned up something no aggregate had, and it is
upstream of every lever §8.68-§8.84 measured.

On `omni-0069` — the worst magazine page, +56.3 pp — the emitted sequence makes
**79 column changes in 90 lines**: near-perfect raster. The cause is in the first
line of the dump, which spans 77 % of the page:

```
'pher Script +10, Gather Information +9, Knowledge   except that the range is 0 and the effect is a cloud that'
   ^ left column                                       ^ right column
```

**`group_lines` merged a left-column line with a right-column line.** Two
consequences, both fatal and neither recoverable by ordering:

1. the text is interleaved INSIDE the line, so no permutation can fix it;
2. the merged box spans 77 %, so `is_spanning` fires, `xy_cut_pernode` sees
   `spans = true`, abandons the gutter grid, and the page falls to raster.

**Three bad lines out of ninety destroy the ordering of the whole page.**

**Scale.** Gutter-merged lines are rare and concentrated:

| | |
|---|---:|
| merged lines | 20 of 26 671 (**0.1 %**) |
| pages affected | 10 of 236 (4 %) |
| CER on those pages | **22.76 %** vs 18.56 % |
| share of ALL ordering damage | **1.03 pp of 6.00 pp = 17 %** |

Six of the ten are newspaper — the population §8.82 identified. This is a larger
prize than several levers chased this campaign (§8.77's 0.92 pp, §8.83's
0.56 pp) and it comes from 0.1 % of lines.

**The amplification is real; the routing logic is NOT the fix.** `any()` looks
fragile and replacing it with a fraction was the obvious repair. Measured, it is
refuted:

| `FFAI_SPAN_TOL` | CER |
|---|---:|
| 0.0 (shipped `any()`) | **18.63 %** |
| 0.01 | 21.08 % (+2.45) |
| 0.10 | 21.75 % (+3.12) |
| 0.50 | 21.87 % (+3.24) |

Every value worse. The strictness is load-bearing for exactly the reason §8.29
built it: a genuine spanning headline sliced by a vertical cut costs far more
than the rare merged line saves. Reverted, with the numbers recorded in the
code so the "obvious" repair is not attempted a third time.

**So the fix belongs in `group_lines`** — do not join boxes across a column
gutter — which is detection, not ordering, and is the first defect in this
campaign that is neither a threshold nor a rule but a straightforward bug. Its
ceiling is at least the 1.03 pp above, and likely more: the oracle-order
measurements in §8.68-§8.84 all reorder these same broken lines, so their
ceilings are understated on these pages too.

**Method note.** Fourteen sections of aggregates never surfaced this. One look
at a rendered page did. `render_layout.py` and `render_order.py` exist for that
reason and should be run FIRST on any new failure population, not last.

### 8.86 The gutter split: diagnosis CONFIRMED, every predicate refused

§8.85 found DBNet components bridging the column gutter. Building the fix took
four attempts, and the three that failed are worth as much as the one that ran.

**Wrong file.** The obvious fix is `group_lines`, which merges on VERTICAL
overlap with no horizontal constraint. But `mobiledet-crnn` never calls it —
"DBNet emits text LINES, so there is nothing to group", `engine.rs:385`. Each
detector box IS a line. Fixing `group_lines` would have edited dead code and
measured nothing.

**Wrong rule, twice.** On `omni-0069`'s merged box the gutter is **9 map
columns** against a 30-row box — far under a line height, so any absolute rule
keyed to line height rejects it. And the widest WORD SPACE on the same line is
**8**, so "the gutter is the outlier gap" does not separate them either.

**Wrong evidence.** At the true gutter the merged boxes read **max probability
1.000**. DBNet hallucinates ink across the columns — which is precisely why the
component bridged. A map-based split looks principled and searches for evidence
that is not there.

**What fires.** A word space falls at a random x on each line; a column gutter
sits at the SAME x on every line. Taking the gutter from the COVERAGE PROFILE of
all boxes and splitting any box that straddles one works: `omni-0069` goes
90 lines -> 93, the merges resolved.

**And it is REFUSED. Holdout A/B, one variable:**

| population | OFF | ON | delta | pages |
|---|---:|---:|---:|---|
| all holdout | **18.88 %** | 21.44 % | +2.56 pp | 13 better, 100 worse |
| the 10 merge pages | 22.76 % | 42.84 % | +20.09 pp | 3 better, 6 worse |
| the other 226 | 18.56 % | 19.65 % | +1.10 pp | 10 better, 94 worse |

The OFF arm reproduces 18.88 % exactly — the instrument is sound.

**But it is BIMODAL, and that vindicates the diagnosis completely:**

| page | OFF | ON | |
|---|---:|---:|---|
| `omni-0069` | 70.26 % | **1.92 %** | **-68.34 pp** |
| `omni-0001` | | | -53.72 pp |
| `omni-0185` | | | -38.85 pp |
| `omni-0140` | 13.03 % | 69.26 % | +56.23 pp |
| `omni-0123` | | | +42.05 pp |

`omni-0069`'s ENTIRE failure was this defect. Recognition survives the split —
the ON text reads cleanly — so the damage elsewhere is the rule firing where no
gutter exists and shattering every line on the page.

**The adaptive switch does not exist on this signal.** Swept on TRAIN, because
tuning on holdout would contaminate every number in §8.68-§8.85:

| `FFAI_GUTTER_COV` | train CER | pages |
|---|---:|---|
| off | **19.90 %** | — |
| 0.10 | 21.89 % | 5 better, 31 worse |
| 0.05 | 20.73 % | 3 better, 22 worse |
| 0.02 | 20.79 % | **0 better**, 13 worse |
| 0.01 / 0.00 | 19.90 % | never fires |

**Tightening kills the wins before the losses.** And the population check passes
— train carries the defect on **10 of 80 pages (12.5 %)**, three times holdout's
rate — so this is not a sample artifact. Gutter cleanliness does not separate
the pages the split helps from the pages it wrecks.

Kept as `FFAI_GUTTER_SPLIT`, defaulted OFF; nothing ships. The untried signal is
`boxes::find_gutters`, the ordering path's own calibrated gutter finder, rather
than the ad-hoc coverage profile built here.

**The standing lesson.** A -68 pp page proves the defect is real and worth
fixing; it does not license shipping the fix. Both facts are in the same
measurement and only the paired, population-split A/B shows them together.

### 8.87 Why no trigger exists: 4 merges hide among 737 look-alikes

§8.86 proved the OPERATION and refused two triggers. Two more were built and
refused, and then the base rate explained all four at once.

**`find_gutters` as the gate — refused.** The ordering path's own finder is
calibrated across §8.54/§8.55 and does what the ad-hoc coverage profile
structurally could not: it ERODES each box by ~0.6x its height before
projecting, undoing the `UNCLIP_LINE` dilation that closes a real gutter, and it
skips spanning lines so a merged box cannot veto the gutter it straddles. Train:
**19.90 % -> 23.35 %, +3.45 pp, 4 better and 42 worse** — worse than the profile
it replaced.

**Height as the gate — refused, and it closes the question.** Among boxes
spanning >= 60 % of the page on the holdout:

| | count | height / page median line |
|---|---:|---:|
| gutter MERGES | **4** | 1.05 |
| legitimate spans | **737** | 1.03 |

| rule | catches | wrongly splits |
|---|---|---|
| height < 1.2x median | 4 of 4 merges | **599 of 737** real spans |
| height < 1.5x median | 4 of 4 | 675 of 737 |
| height < 2.0x median | 4 of 4 | 701 of 737 |

**The merges are 0.5 % of the population the rule fires on, and no geometric
feature separates them** — the two classes differ by 0.02 line heights. A
trigger would need better than 99 % precision to break even, because each false
positive shatters a real masthead and §8.29 measured that as expensive.

This is §8.79's rule for the third time (§8.39, §8.79, here): **a geometric
predicate is evidence only at the scale where its positives are rare.** Here the
positives are rare in the wrong direction — the thing we want to catch is the
0.5 %, not the 99.5 %.

**Final state of the lever.** The defect is real and worth 1.03 pp (17 % of all
ordering damage); the surgery is proven — `omni-0069` 70.26 % -> **1.92 %**,
recognition intact; and five independent triggers are refused: absolute width,
outlier-dominance, probability-map emptiness, coverage cleanliness (5
thresholds), the calibrated `find_gutters`, and box height. What separates a
merge from a masthead is whether the text on either side belongs to the same
sentence — which is CONTENT, and §8.81 measured content as unable to arbitrate
reading order at all.

Kept as `FFAI_GUTTER_SPLIT`, defaulted OFF, with `find_gutters` as the gate
since it is at least principled. Nothing ships.

### 8.88 The best discriminator yet, and it still does not convert

§8.87 closed the gutter-merge lever on a base-rate argument. Two corrections to
that section came from pushing on it, and both were mine.

**The target was under-counted.** The 4-merges-vs-737-spans figure covered only
boxes spanning >= 60 % of the page. There are **20 merges**; the other 16 bridge
two columns of a THREE-column layout and never reach 60 %. The height test
measured a fifth of the problem and was reported as the whole of it.

**And the right test was refused on circular evidence.** Gutter-emptiness was
rejected because the probability map reads 1.000 at the true gutter — but that
hallucination IS the defect. Asking the broken detector whether it is broken
cannot work. **The source pixels were never consulted.**

They separate the classes, where nothing else did:

| widest internal white corridor, source pixels | n | median (line-heights) |
|---|---:|---:|
| gutter MERGES | 20 | **1.14** |
| legitimate spans | 22 588 | **0.33** |

**3.5x**, against the 1.05-vs-1.03 that box height gave. At 0.8x it catches 17
of 20 merges and touches 1.9 % of everything else.

**Built** (`boxes::split_at_white_corridor`, hooked in the MobileDet path where
the image is in scope) and it works: `omni-0069` goes 90 lines -> 93 with the
merge resolved. **And it is REFUSED.** Swept on TRAIN:

| `FFAI_WHITE_SPLIT` | CER | vs off | pages |
|---|---:|---:|---|
| off | **19.90 %** | — | |
| 0.6 | 19.89 % | -0.02 pp | 10 better, 37 worse |
| 0.8 | 19.89 % | -0.01 pp | 8 better, 26 worse |
| 1.0 | **19.85 %** | **-0.05 pp** | **5 better, 15 worse** |
| 1.3 | 20.02 % | +0.11 pp | 2 better, 5 worse |

Best case 0.05 pp — about 90 characters — with a NEGATIVE page count at every
threshold. §8.78's standard refuses this.

**Why a good discriminator still loses.** 20 merged lines out of 26 671.
Repairing every one caps near 1 pp, while a 1.9 % false-positive rate applies to
the other 99.9 % of lines. **The quality of the discriminator was never the
binding constraint — the base rate was**, which is why five geometric triggers
and one pixel trigger all land in the same place.

Kept as `FFAI_WHITE_SPLIT`, defaulted OFF. The lever is closed for the second
and final time, now from the best position available rather than the worst.

**What is left, and it is not layout.** §8.83 measured three-quarters of the
corpus as RECOGNITION-limited (29.09 % CER, only 2.90 pp reachable by ordering).
Every ordering and detection lever is now measured and closed. PARSeq is already
built, already shipped, and is not the default on the document path — that is
the next thing to price.

### 8.89 Gutter merge CLOSED after six triggers; PARSeq refused on documents

**PARSeq, re-tested rather than inherited.** The ledger's refutation was a
43-page subset at an older version, which §8.53's rule says expires. Current, on
the full document holdout:

| | CER | |
|---|---:|---|
| shipped `mobiledet-crnn` | **18.88 %** | |
| `mobiledet-parseq` | **26.05 %** | +7.17 pp, 32 better / **199 worse** |

Refused, and now on current evidence.

**The white-corridor split, confirmed on holdout.** Selected on train at 1.0
(§8.88) and carried across exactly once:

| population | OFF | ON | delta | pages |
|---|---:|---:|---:|---|
| all holdout | 18.88 % | 19.77 % | +0.89 pp | 9 better, 29 worse |
| the 10 merge pages | 22.76 % | 29.45 % | +6.70 pp | **5 better, 3 worse** |
| the other 226 | 18.56 % | 18.96 % | +0.40 pp | 4 better, 26 worse |

**`omni-0069` reproduces at -68.3 pp**, with `omni-0137` -18.6 and `omni-0140`
-8.8, and MORE merge pages improve than regress. The diagnosis is right and the
surgery works. It is broken by `omni-0144` (+37.7) and `omni-0148` (+34.7) —
themselves merge pages, where the same trigger fires in the wrong place.

**The width gate — principled, and anti-correlated.** The damage mechanism is
specifically that a box wider than `SPAN_FRAC` makes `is_spanning` fire and
collapses the page to raster, so a narrow box carrying a corridor is all risk
and no reward. Restricting the split to wide boxes is derived, not fitted. It is
WORSE:

| population | OFF | ON | delta | pages |
|---|---:|---:|---:|---|
| all holdout | 18.88 % | **20.03 %** | **+1.15 pp** | 3 better, 8 worse |
| the 10 merge pages | 22.76 % | 32.99 % | +10.23 pp | 2 better, 2 worse |
| the other 226 | 18.56 % | 18.94 % | +0.39 pp | 1 better, 6 worse |

It narrowed the blast radius exactly as intended — 38 pages touched down to 11 —
and kept the wrong 11: `omni-0069` -68.3 retained, `omni-0144` +37.7 and
`omni-0148` +34.7 retained, `omni-0137` -18.6 and `omni-0140` -8.8 LOST.
`omni-0137` and `omni-0140` are THREE-column pages whose merges bridge two
columns and never reach `SPAN_FRAC` — the 16 of 20 merges §8.88 identified —
while the damaging splits are genuinely wide. **Width is anti-correlated with
correctness.**

**Closed.** Six independent triggers, all refused: absolute width,
outlier-dominance, probability-map emptiness, coverage cleanliness (5
thresholds), the calibrated `find_gutters`, box height, source-pixel white
corridor (4 thresholds), and that corridor width-gated. The best is +0.89 pp.

The defect is real, worth ~1 pp, and its surgery demonstrably repairs the worst
page in the corpus. What no geometry supplies is WHICH wide box with a white
corridor is a merge — because the answer is whether the text either side
continues the same sentence, and §8.81 measured content as unable to arbitrate
reading order.

Kept as `FFAI_WHITE_SPLIT` / `FFAI_WHITE_WIDE`, both defaulted OFF.

**Campaign status.** Ordering, detection and recognition levers are now all
measured and closed: §8.68, §8.71, §8.72, §8.74, §8.78, §8.79, §8.81, §8.83,
§8.84, §8.86, §8.87, §8.88, §8.89. Shipped default remains **18.88 %**, 236/236
PASS, 5.82 pp behind Unlimited-OCR — a gap that §8.82 localised to newspapers
and magazines and §8.83 showed a layout model does not close.

### 8.90 Editorial pages ARE detectable — and page type is not the gate

If newspapers and magazines carry 78 % of the ordering damage (§8.82), the
natural question is whether they carry a signature we could gate on. They do,
and it does not help.

**The signature, all computable at runtime from our own boxes:**

| feature | editorial (n=107) | everything else (n=123) | ratio |
|---|---:|---:|---:|
| lines per page | 171 | 62 | **2.76x** |
| regions per page | 18 | 8 | 2.25x |
| ink density | 0.454 | 0.284 | 1.60x |
| **line-width CV** | **0.322** | 0.573 | **0.56x** |
| fraction display type | 0.008 | 0.000 | — |
| line-height CV | 0.184 | 0.184 | **1.00x** |

The strongest feature is inverted from intuition: editorial pages have MORE
UNIFORM line widths, because columns force every line to one measure while
ragged single-column prose varies freely. And type-size variety — the obvious
guess for "magazines have mixed fonts" — is identical at 1.00x, refuted.

**It is not the gate for the gutter split.** Every page the split helps and
every page it hurts is a NEWSPAPER:

| page | source | lines | |
|---|---|---:|---|
| omni-0069 | magazine | 90 | helps |
| omni-0137 | newspaper | 251 | helps |
| omni-0140 | newspaper | 265 | helps |
| omni-0144 | newspaper | 447 | **hurts** |
| omni-0148 | newspaper | 456 | **hurts** |
| omni-0136 | newspaper | 301 | **hurts** |
| omni-0131 | newspaper | 295 | **hurts** |

Page type is constant across the outcome being predicted, so it carries zero
information about it.

**A tempting separation, refused.** Helps <= 265 lines, hurts >= 295. Seven
points, one threshold, and no mechanism for why 265 lines is splittable and 295
is not. Fitting it would also mean tuning on HOLDOUT, contaminating every number
from §8.68 onward. This is §8.50's benchmark-fitting exactly, and the fact that
it appears after six honest refutations is what makes it dangerous rather than
promising.

**The structural reason a page-type gate cannot pay.** A gate is only worth
having if there is a different TREATMENT to apply behind it. There is not:
every ordering alternative is refuted (§8.68-§8.84), a layout model buys 7.6 %
of its prize on exactly this population (§8.83), and the split has no setting
that wins (§8.89). Detecting the population we are worst on does not help while
we have nothing better to do with it.

Recorded so the question is not re-opened as though unexamined: the signature
exists, it is strong, and it gates nothing we currently possess.

## 9. Pure-Rust boundary and watchlist

**Decisions, recorded:**

- **No ONNX Runtime, no MNN.** Candle is the spine (Principle 3); Mercury
  proved the spine competitive with hand-tuned C++. A future accelerator
  backend is a feature flag with a measured justification, or it does not
  exist.
- **Raster in, structure out.** PDF rasterization, camera capture, and
  screen capture are input plumbing outside the OCR crate; `ffai-media` owns
  ingest as it does for audio.

**Watchlist** (adopted when mature, checked each milestone):

| Item | Status |
|---|---|
| `ocrs` (pure-Rust OCR) | candidate zero-download baseline engine; measured at M-C0 either way |
| rff image decoders (PNG/JPEG/WebP) | ROADMAP Phase 3 — Carmenta's ingest route |
| rff video ingest (frame iteration for LIVE) | needed by M-C2; scope it early |
| **Pure-Rust PDF rasterizer** | does not exist at production grade; candidate future Remade-With-Rust project (`rff-pdf` or sibling). Until then, LONG consumes pre-rendered page images |
| Deferred feature map | preprocessing beyond §3 (dewarp, super-res), ensemble/voting, handwriting + vertical text, seal/chart heads, embeddings output for RAG, VLM-refinement hybrid — parked until a function's gate demands one; nothing lands without a corpus that can fail it |

---

## 10. What "FFmpeg-grade" means here, restated

Everything optional and composable; explicit measured trade-offs; rich
structured output by default; engines swappable by name; the same mental
model as Mercury — and every number on this page eventually replaced by a
ledger line or deleted.
