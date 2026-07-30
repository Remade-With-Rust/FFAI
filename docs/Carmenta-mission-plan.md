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
| **M-C1** | Det+rec core on candle: lineage selected by audit, `OcrOutput` v2 hierarchy in `ffai-core`, per-stage oracles vs references on the synthetic corpus, stub names reconciled | CER within the 5 % relative band of Tesseract on the printed holdout; all four gates run and recorded (speed may fail honestly at bring-up — it did for Mercury) |
| **M-C2** | **LIVE**: streaming loop, frame sampler + change gate, stabilizer, timed SRT/VTT output, `--roi`; auto-ROI observe-only harvest + ceiling sweep, then opt-in `--auto-roi` if the sweep pays | p95 warm frame latency ≤ Tesseract per-frame on the same frames; zero churn on identical frames; CER parity with own batch mode on the frame holdout; footprint flat over a 30-min synthetic stream; auto-ROI sweep in the ledger win or lose |
| **M-C3** | **DOCUMENT**: layout stage, reading order, `--layout`, Markdown/JSON structured output | reading-order accuracy + end-to-end CER vs Tesseract and PP-Structure on the document holdout; structured output round-trips losslessly to JSON and back |
| **M-C4** | **LONG**: multi-page state, cross-page repair, bounded-memory streaming, intermediate caching | full-document CER ≤ the same engine's per-page score (coherence costs nothing); footprint flat over 500 pages; vs Tesseract/docTR on the multi-page holdout |
| **M-C5** | **FORMULA**: LaTeX head, `--formula` routing from layout regions | edit distance + ExpRate vs pix2tex on the pinned holdout; composes with `--layout` (inline `$...$` in Markdown out) |
| **M-C6** | Carmenta `stable`: docs, library examples, claims page generated FROM the ledger; tables/TEDS if M-C3's table work matured | every public claim maps to a ledger line id |

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
