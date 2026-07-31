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
| **OmniDocBench** | Purpose-built for document parsing and the better fit on paper — reading order, formulas, tables. **Rejected: no licence stated** in either the HF card data or the README. Same rule that disqualified CRAFT's click-gated weights in §7.1 — an unstated licence is not a permissive one, and a corpus whose terms we cannot name cannot back a public claim. |
| **PubLayNet** | Layout only, no text ground truth, and the HF mirrors return 401. |

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

**Named next brick: rejection.** A confidence gate — CTC/AR score per line,
detection score per box, or both — swept on the CORD train split. The ceiling
is unusually well defined: driving our insertions to the reference's 5.79 pp
takes CORD from 20.55 % to ~14.8 %, at parity with PaddleOCR, without touching
detection geometry or the recognizer. That is the largest single prize left on
the board and it was invisible for the whole campaign because CER was only ever
read as a total.

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
