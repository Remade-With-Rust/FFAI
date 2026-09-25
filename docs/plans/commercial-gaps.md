# Commercial gaps — found while using FFai in a real pipeline

A running list of gaps found when an application tried to use FFai for real work. Each entry names the use, what was missing or weak, the evidence, and what would close it.

Evidence labels:

- **source**: read in FFai's own code or README.
- **observed**: seen in the input documents.
- **measured**: run and counted.

---

## Progress

Executing the whole list, in three batches. Branch `feat/commercial-gaps`.

| Batch | Gap | Work | Status |
|---|---|---|---|
| 1 quick | 9 | `ffai_media::decode_image(&[u8])`: same decoders and output as `load_image`, dispatch by magic bytes | **done**: byte-identical to `load_image` on JPEG and PNG, tests pass |
| 1 quick | 7 | `CraftCrnn::offline(rec, det, weights_root)`: weights only from `root/<model>/<file>`, no cache, no network. `preload()` fails before page one. `ffai models --verify <name>` checks without downloading. Errors name the exact expected path | **done**, details below |
| 1 quick | 10 | The six OCR manifests are compiled into `ffai-carmenta` (`manifests::EMBEDDED`), so no `models/` directory is needed. A test keeps the crate's copies byte-identical to `models/`. A missing manifest directory is an error that names it | **done**; `cargo package --list` confirms the manifests ship (hosted weights: batch 3) |
| 1 quick | 3 (part) | `OcrOptions::charset`: decoding restricted to a character set, for CRNN and SVTR. Confidence is not renormalised, so a forced character reads as uncertain. PARSeq and routing refuse rather than ignore it. `ffai ocr --charset` | **done**, details below |
| 1 quick | 6 | Determinism test: same page, two calls and two instances, text and boxes must be identical | **done: already deterministic**, details below |
| 1 quick | 8 | Confidence always reported? | **answered from source**: CRNN and SVTR report a confidence on every emitted line (a line with no kept character is dropped, never emitted with `None`). Only PARSeq can return `None`, when no word carried one |
| 2 measured | 12 | Remove form underlines before recognition; calibrated confidence | **rule removal done**, opt-in: fixes CRNN, neutral or harmful on SVTR (below). **Biggest measured fix: use `mobiledet-svtr`.** Full-width fold done and on by default (OmniDocBench: 5 pages better, 1 worse, the worse one a ground-truth quirk) |
| 2 measured | 5 | Orientation (0/90/180/270) detection | **done: 144/144** on turned pages and form lines, at 48 % of a read per document page. `OcrOptions::auto_orient`, `ffai ocr --auto-orient` |
| — | 13 | Throughput with orientation search | **done via gap 5**: about 1.5 reads per page, against the 3 reads fraud-alert pays |
| 3 build | 1, 11 | PDF ingest returning the page **as rendered** (redactions composited), not the raw scan XObject | **done**: `ffai_media::pdf`, wired into `ffai ocr`; born-digital text layers returned with a `redaction_risk` flag (below) |
| 3 build | 2 | CCITT G3/G4 decoder (then JBIG2) | **G4 and G3-1D done** via the pure-Rust `fax` crate inside `pdf`; a house `rusty_ccitt`, G3-2D and JBIG2 are open |
| 3 build | 4, 14 | Checkbox, table and key-value output | **checkboxes done**: 100 % recall and state on 960 synthetic boxes; `口` is the known false positive. **Key-value pairing done** (`forms::pair_fields`; 220/221 correctly-read fields paired). Free hand ticks and tables in `OcrOutput`: open |
| 3 build | 3, 12 (rest) | Lexicon constraint; confidence calibration; digit CER in the bench | **all three done**: lexicon; calibration measured (min-character confidence changes sign by engine, so it is refuted as a default); digit error rate in every OCR ledger line |
| 3 build | 10 (rest) | Hosted converted weights, so manifests name a download source | not started |

Out of scope for batch 1, noted: region routing (`FFAI_ROUTE`) still loads its
ONNX stages from `corpora/refs/fixtures`, a repository path. It is opt-in and
off by default.

### Batch 1 results (2026-09-24)

All measured on this machine with the cached weights, in
`crates/ffai-carmenta/tests/production.rs`.

- **Determinism (gap 6): closed; there was nothing to fix.** OmniDocBench page
  `omni-0000` has 120 lines, enough for the parallel recognition path. It was
  read twice by one `mobiledet-svtr` instance and once by a second instance.
  Text and boxes were identical on all three reads, and the largest confidence
  difference was **0**. The test now guards it. Caveat, from §8.46: the reads
  ran in isolation, and under contention from another process confidences have
  been seen to move by about 1e-4. That is far below any gate a caller would
  set, and text did not move.
- **Offline resolution (gap 7): it found a defect in its own first version.**
  The `ppocrv5-mobile-det` manifest also lists `det.safetensors` and the
  training checkpoint `det-train.safetensors`, for provenance. Neither is
  loaded at runtime. Strict offline resolution demanded all three files, which
  would have forced a deployment to ship two unused checkpoints. Offline
  loading now asks only for the files it opens
  (`ModelManifest::resolve_only_in`). A name the manifest does not declare is
  refused, since it has no checksum. The fix also removed a doubled
  "model error: model error:" prefix from load errors.
  - Offline output is identical to the cache-backed constructor's on the
    120-line page.
  - An empty weights directory fails at `preload()`, naming
    `<root>/ppocrv5-mobile-det/det-fused.safetensors`.
- **Charset constraint (gap 3, part).**
  - With the full English charset allowed, text and confidences are identical
    to the free decode on the real page.
  - Restricting to `0123456789$,.- ` emitted nothing outside the set.
  - On a page of prose, mean line confidence fell from **0.918 to 0.781**. The
    forced characters show up in the confidence, as intended.
  - A character the model cannot emit (`中` on the English CRNN) is an error
    naming it.
  - Not yet measured: whether constraining fixes the underline look-alikes
    ("1117" → "TTT7"). That needs the KY crops and belongs with gap 12 in
    batch 2.
- **Gates:** the three blocking CI clippy commands, `cargo check --workspace
  --all-targets`, rustfmt on `ffai-media` and on every file that was clean
  before, and `tools/lint_invariants.py` all pass. `ffai models --verify`
  passes for cached models and exits 1 naming the searched path for a missing
  one.
- **Breaking change:** the new public field on `OcrOptions` breaks
  `ffai-core`'s semver (see CHANGELOG).

### Batch 2: form rules (gap 12)

**Corpus.** The Kentucky filings are public records that carry personal
details, and fraud-alert reads them without keeping them, so they are not
test data here. `tools/carmenta_form_corpus.py` builds `carmenta-form-v1`
instead, with ground truth known by construction:

- 30 invented values: amounts, numbers and names chosen for the confusions
  seen in the field.
- Each value is rendered with no rule (the paired control) and three rule
  kinds: an underline touching the glyph bottoms, a rule through the
  descenders, and a strikethrough.
- Clean grayscale, plus 1-bit to mimic CCITT fax scans.
- Courier, Arial and Times, 10-12 pt at 300 ppi. 240 clips.

The bench is `examples/form_bench.rs`. It scores the VALUE: exact read, CER,
and **wrong>gate**, meaning the value was wrong but its line passed a 0.85
confidence gate.

**Baseline, all 240 clips (measured).**

| condition | exact, `mobiledet-crnn` | wrong>gate | exact, `mobiledet-svtr` | wrong>gate |
|---|---:|---:|---:|---:|
| no rule | 70 % | 17 / 60 | 93 % | 4 / 60 |
| underline | 13-20 % | 40 / 60 | 93 % | 4 / 60 |
| rule through descenders | 47-50 % | 28 / 60 | 97 % | 2 / 60 |
| strikethrough | 7-10 % | 43 / 60 | 77-87 % | 10 / 60 |
| amounts, no rule | 15 % | 16 / 20 | 85 % | 3 / 20 |

- **The corpus reproduces the field failure**, and only on the recognizer
  fraud-alert uses. `OcrPages::new` builds `variant_in(RecStage::Crnn,
  DetStage::MobileDet, ..)`. On that engine, ruled values are read exactly
  7-20 % of the time, and most of the wrong ones pass the gate.
- **`mobiledet-svtr`, the shipped document default since §8.171, is already
  robust to underlines and to rules through descenders.** Switching
  fraud-alert to it is the largest single fix available, and it is one line in
  their code. Strikethrough remains the weak case for both engines.
- CRNN also reads amounts badly **with no rule at all** (15 % exact). That is a
  recognizer weakness on `$`-prefixed values, not a rule artefact.

**Rule removal, `OcrOptions::remove_rules` (measured, all 240 clips).**
`carmenta::rules` erases thin, long horizontal dark runs. It keeps any pixel
whose vertical dark run is tall, so glyph strokes standing on or crossing a
rule survive, and a filled box such as a redaction is never touched. It runs
before detection.

| condition | CRNN | CRNN + removal | SVTR | SVTR + removal |
|---|---:|---:|---:|---:|
| no rule | 70 % | 70 % | 93 % | 93 % |
| underline | 13-20 % | **70 %** | 93 % | 90-93 % |
| through descenders | 47-50 % | **67-70 %** | 97 % | 93-97 % |
| strikethrough | 7-10 % | **47-50 %** | 77-87 % | 73 % |
| wrong>gate, underline | 40 / 60 | **18 / 60** | 4 / 60 | 5 / 60 |

- **On CRNN it removes the whole underline penalty.** Underlined values read
  exactly as often as the no-rule control.
- **It is a true no-op on unruled text**: the "no rule" rows match the baseline
  to the digit.
- **It is neutral-to-harmful on SVTR.** Strikethrough gets worse (77-87 % to
  73 %), because erasing a strike through the middle of the glyphs also erases
  the horizontal strokes it coincides with (`e` reads as `c`, `C` as `G`).

**Verdict: opt-in.** It helps a caller on CRNN, and the better fix is SVTR
without removal.

**The per-field charset (`--charset-by-kind`, amounts restricted to
`0123456789$,.- to`) changed nothing on SVTR.** Its remaining amount errors
are not look-alikes a charset can exclude.

**What SVTR still gets wrong, and the next targets:**

- **Full-width forms.** `"$25，０00"` and `"12／31／2025"`: SVTR's multilingual
  charset sometimes emits CJK full-width digits and punctuation (U+FF0C,
  U+FF10, U+FF0F). A number parser rejects them, or worse, drops them silently.
  The charset constraint excludes them for fields known to be numeric. A
  full-width-to-ASCII fold for Latin pages is the general fix; it needs the
  OmniDocBench gate because CJK text uses full-width forms correctly.
- **`1` read as `l`** in `"1121 River Road"`, in every condition including no
  rule. This is a lexicon or pattern constraint problem (gap 12).
- **A value split by detection.** `"$5,001 - $10,000"` loses its first half,
  also with no rule.

### Batch 2: orientation (gap 5)

`CraftCrnn::detect_orientation` and `OcrOptions::auto_orient` use no new
model:

1. One detection pass decides the axis from box aspect, weighted by area.
2. The 12 largest boxes on that axis are read in both candidate turns, and
   the higher mean recognition confidence wins.
3. With `auto_orient`, boxes are mapped back to the input's coordinates.

**Accuracy (measured, `examples/orient_bench.rs`, ground truth by
construction).** 24 OmniDocBench pages and 12 form lines were each turned
0/90/180/270 and had to be turned back.

| source | 0 | 90 | 180 | 270 | smallest confidence margin |
|---|---:|---:|---:|---:|---:|
| document pages | 24/24 | 24/24 | 24/24 | 24/24 | 0.545 |
| single form lines | 12/12 | 12/12 | 12/12 | 12/12 | 0.148 |

**144 of 144 correct, including upside-down (180), which fraud-alert's
workaround never tries.** The margin is the gap between the chosen and
rejected turn's confidence. It is wide on pages and narrower, but still
unanimous, on a single sparse line.

**Cost, first version: missed.** A decision took 122 % of a full read. The
axis vote ran detection at full page resolution, and the sample crops were
read one after another.

**Second version: fixed.** Detection runs on a page decimated to about a
640 px short side (box aspect survives), and the crops are read in parallel,
on the same pages:

| source | accuracy | smallest margin | decision cost |
|---|---:|---:|---:|
| document pages | 96/96 | 0.465 | **48 % of a read** (7.8 s vs 16.3 s) |
| single form lines | 48/48 | 0.148 | 104 % of a read (nothing to decimate on a one-line image) |

`auto_orient` therefore costs about **1.5 reads per document page**.
fraud-alert's workaround reads each landscape page three times, and never
checks upside-down. (Absolute seconds are from a contended machine; only the
ratios are claims.)

### Batch 3: checkboxes (gaps 4, 14)

`ffai_carmenta::checkbox::find` and `ffai ocr --checkboxes` find boxes in the
pixels and report each frame with a marked or empty state, instead of the
`@` / `0` / `D` / `B` a text reader makes of them. No model is involved. A
frame is accepted only when all of these hold:

- four sides, each at least 85 % covered;
- four sharp, inked corners;
- clear space just outside every side: a box's top edge is its topmost ink,
  while a round letter bulges past any run cut from its own contour;
- a mark that overflows the box is trimmed back to the frame's real right
  side.

A box counts as marked when its interior, inset past the frame, is at least
6 % ink.

**Corpus.** `tools/carmenta_checkbox_corpus.py` builds `carmenta-checkbox-v1`:
80 pages and 960 boxes, ground truth by construction.

- Boxes are drawn both as stroked rectangles and as the font's own `☐` glyph.
- Marks are X strokes, ticks that overflow the box, solid fills and `☒`, or
  the box is left empty.
- Distractors: `O D 0 8` in running text, `口 回`, a table, and a rule.
- Every page comes in gray and 1-bit.

**Measured, in four iterations. Each earlier result stays recorded because
each was wrong in a different, instructive way.**

| version | recall | state right | false positives |
|---|---:|---:|---:|
| four sides only | 99.4 % | 96.6 % | 1544 |
| + generator fix (Courier questions ran into the "empty" boxes; the detector was right and the ground truth was not) | 98.8 % | **100 %** | 1540 |
| + clear space outside each side (corners alone passed `o d g R`: a run cut from a curve has dark ends by construction) | 98.8 % | 100 % | 240 |
| + trim an overflowing tick back to the frame | **100 %** | **100 %** | 240 |

- **All 240 remaining false positives are the `口口 回` distractor, three per
  page.** Those ideographs are geometrically square frames, and no shape test
  can separate `口` from an empty box. On CJK pages, cross-check each box
  against the OCR text at its position. That is the known limit, stated.
- **First real page, first new false-positive class.** On a real scientific
  page (the smoke test below), two 13x12 px squares were reported as empty
  boxes. They are most likely square plot markers or legend symbols, a class
  the synthetic corpus does not contain. `CheckboxParams::min_side` is the
  knob. The default is left alone rather than tuned to one page.
- **Caveats.** The detector's structure was developed while looking at this
  corpus's failures, so these are not yet holdout claims. Nothing has been
  run on real scans yet. Hand ticks with no box at all (Indiana's columns,
  gap 14) are not handled: this finds boxes, not free marks.

### Batch 3: lexicon constraint (gap 12)

`ffai_carmenta::lexicon::Lexicon` takes the values a caller already knows a
field can hold (counties, known creditors) and snaps a read to the nearest
entry.

- The distance is Levenshtein, except that look-alike substitutions cost half
  an edit: `1 l I i | T`, `0 O o D U Q`, `5 S s`, `8 B $` and the rest.
- It snaps **only when unambiguous**: within 25 % of the entry's length, with
  the runner-up at least one edit further away. Otherwise it returns the
  candidates.
- It never rewrites what the engine read. It is advice with its evidence
  attached.
- **Every misread in this document snaps to its entry:** "Trish HTlI" →
  Irish Hill, "GuIF Shores" → Gulf Shores, "MerrITLynch" → Merrill Lynch,
  "Cabe]l County" → Cabell County, "Tr1-State Bank" → Tri-State Bank. A tie
  between "Unit 10" and "Unit 1O" is reported as ambiguous, not guessed.

### Batch 3: confidence calibration (gap 12), measured on SVTR

The line's mean character probability is compared with its weakest
character's (`FFAI_CONF_REDUCE=min`), on the 240 form clips (19 wrong values):

| confidence | AUROC | at the 0.85 gate |
|---|---:|---|
| mean (shipped) | 0.746 | keeps 216/221 right, **19/19 wrong** |
| min character | 0.764 | keeps 125/221 right, 3/19 wrong |

- **This reproduces the incident exactly.** With mean confidence, a 0.85 gate
  keeps every wrong value.
- **Min separates only slightly better** (AUROC 0.746 to 0.764). At the same
  gate it removes 16 of 19 wrong values, but also 43 % of the right ones,
  because it is a different scale and its gate has to be set separately.
**On CRNN it reverses** (240 clips, 154 wrong values):

| CRNN confidence | AUROC | at the 0.85 gate |
|---|---:|---|
| mean (shipped) | **0.873** | keeps 86/86 right, 128/154 wrong |
| min character | 0.828 | keeps 57/86 right, 20/154 wrong |

- **The min reduction's effect changes sign by engine**: slightly better on
  SVTR, worse on CRNN. So it is **refuted as a general fix**. The mean stays
  the default, and `FFAI_CONF_REDUCE=min` stays an experiment.
- Neither reduction is a calibrated probability, and no single gate protects a
  field. The measured protections are:
  - a constraint that knows the field (charset or lexicon);
  - `mobiledet-svtr`, which makes far fewer wrong reads to gate in the first
    place (19 against CRNN's 154 on the same clips).

### End-to-end smoke test: a sideways scanned PDF

An OmniDocBench page was rotated 90 degrees and saved as an image-only PDF
with PIL, the way a scanner stores a portrait form landscape.
`ffai ocr -i sideways.pdf`, release build:

| flags | first lines read |
|---|---|
| `--auto-orient` | "Antibody—antigen interactions I C. Bich et al. 1 Anal. Biochem. 375 (2008) 35–45", then the figure's axis labels |
| none | `3`, `4`, `行`, `o`, `0`, `2`, `3`, `n`, `间`, `m`, `E` |

The second row is the silent failure fraud-alert hit: no error, just
confident fragments. The first row is the PDF path (`ffai_media::pdf`), the
orientation decision and the document default recognizer working together.

### Batch 3: the full-width fold, census on OmniDocBench (gap 12)

`examples/fold_census.rs` runs `mobiledet-svtr` with the fold off
(`FFAI_NO_FOLD=1`), then applies it offline, so both arms come from one
exactly-paired pass.

**Interim, first ~176 of 316 pages: the fold fired on 32 lines.** Where the
ground truth settles the question (the raw or the folded form appears in it,
not both):

| verdict | count | examples |
|---|---:|---|
| folded form is in the ground truth | 5 | `"prｅviously"` → `"previously"`, `"corｐorate"` → `"corporate"`, `"water/sewer，"` → `"water/sewer,"`, `"QQ：87124697"` → `"QQ:87124697"` |
| raw form is in the ground truth | 1 | `"e)，and"`: the annotators themselves typed a full-width comma |

- **SVTR puts full-width LETTERS inside English words** (`prｅviously`,
  `corｐorate`). They look identical and break every search, lexicon match and
  parser downstream. That is a bigger class than the digits the fold was
  written for.
- **The CJK-page worry did not materialise.** `QQ：87124697` sits on a Chinese
  page on a line with no CJK characters, and its ground truth uses the ASCII
  colon.
**Final, all 316 pages:**

- The fold fired on **35 of 34,852 lines (0.1 %) across 19 pages**.
- Changed pages: **5 better, 1 worse, 13 unchanged in CER**. The one worse
  page is the annotators' full-width comma above.
- Corpus CER: 14.2460 % with the fold off, 14.2453 % with it on.
- On the form corpus it turned `"$25，０00"` and `"12／31／2025"` into correct
  reads (strikethrough/gray 76.7 % to 83.3 % exact).

**Verdict: on by default, earned.** The corpus-wide effect is small, as
expected for something that fires on 0.1 % of lines. It is never worse except
where the ground truth itself is full-width. On the lines where it fires it
turns an invisible, parser-breaking character into the one a reader sees.
`FFAI_NO_FOLD=1` turns it off.

### Batch 3: label → value pairing (gap 4)

`ffai_carmenta::forms::pair_fields(&OcrOutput) -> Vec<Field>`:

- A line ending in `:` is a label.
- Its value is the nearest unlabelled line to its right on the same row,
  otherwise the nearest one directly below.
- A single `Label: value` line is split at its colon. A time such as `10:30`
  is not a field.
- Each value line is claimed once, closest label first.
- A label with nothing near it comes back with an **empty value** rather than
  a neighbour's: a blank field is information.

**Measured on `mobiledet-svtr` over the 240 form clips.** The clip's label had
to come back with its exact value: **220 / 240**, against 221 values read
correctly. So pairing loses **one field out of 221 correct reads (99.5 %)**, a
single amount on a "rule through descenders" gray clip. Every other miss is
the recognizer's, not the pairing's. Per row, pairing equals the exact-read
rate in 7 of 8 conditions.

### Batch 3: digit error rate in the bench (gap 3)

`ffai_bench::metrics::digit_errors_with` aligns the reference and the read
and counts only digit errors:

- a substitution where either side is a digit, so a phantom digit counts as
  well as a misread one (`$` → `8` turns "$10,000" into "810,000");
- a dropped digit;
- an invented digit.

Every OCR bench run, engine and reference arm alike, now adds `digit error
rate X % over N reference digits`, micro-averaged over the corpus, to its
ledger notes. ASR ledger lines are unchanged. The first version counted only
reference-digit errors. Its own test (`no` → `n0`) showed that misses
phantom digits, and it was corrected before being wired in.

### Batch 3: born-digital text layers, and their own redaction trap (gaps 1, 11)

`PdfPage` now carries the page's `text` layer (`lopdf`'s bounded extractor)
and a `redaction_risk` flag.

**A text layer has the scan incident's twin.** A black box drawn over
born-digital text hides it on screen and leaves it in the content stream,
where any text extractor reads straight through it. That is the classic
failed redaction. `redaction_risk` is set when:

- the page carries a redaction-style annotation;
- a dark fill is drawn after text was shown;
- the drawing cannot be walked at all (fail-safe).

A background fill drawn before the text does not set it. `ffai ocr` prints a
page's text layer only when the risk is clear. Otherwise it withholds the text
with a warning, and the rendered image is what gets read. Tested on in-memory
PDFs covering all three cases.

### Batch 3: PDF ingest (gaps 1, 2, 11)

`ffai_media::pdf::pdf_pages(bytes)`, behind the opt-in `pdf` feature.
It depends on `lopdf` without its default features, and on `fax` for CCITT;
both are pure Rust and MIT.

- Every page comes back as a viewer shows it. The scan is composited with
  every filled path, image and form `XObject` drawn after it, plus `Redact`,
  `Square`, `Circle`, `Polygon` and `Stamp` annotations. It is then turned
  upright from its placement matrix and `/Rotate`.
- `has_text_layer` flags born-digital pages, so they can be routed to text
  extraction instead of OCR.
- **A page whose drawing cannot be walked returns no image** and a note. It
  never falls back to the raw scan.
- Decoders: DCT (JPEG), CCITT G4 and G3-1D, and Flate/raw samples at 1 or 8
  bits in gray, RGB or CMYK, including stencil masks. JBIG2, JPEG 2000 and
  G3-2D are reported per page, not guessed.
- Bounded against hostile files:
  - images are capped at 100 MP;
  - decompression is capped at the size the dimensions justify;
  - form content is capped at 64 MiB;
  - form nesting is capped at 8;
  - `/N`, bits per component and CCITT `/Columns` are validated before any
    size arithmetic.
- 9 tests on PDFs built in memory, covering the incident itself: a white patch
  image over a scan must come out black.
- **Two defects found by those tests before anything shipped:**
  - Stencil-mask polarity was inverted, so every stencil scan would have come
    out as a negative.
  - `lopdf`'s annotation helper skips annotations stored inline in `/Annots`,
    so an inline redaction would have been read through. `/Annots` is now
    resolved directly, accepting both forms.
- **Two hostile-input paths found while fixing lints:** an unbounded ICC `/N`
  feeding row-size arithmetic, and uncapped form-content decompression. Both
  are now bounded.
- Clippy-clean under `ffai-media`'s pedantic and nursery lints. The casts are
  checked conversions or individually justified, not a blanket allow (open
  audit item R-006 criticises exactly that for parsers).

---

## 2026-09-24 — fraud-alert: reading scanned financial-disclosure filings

**The use.** `fraud-alert` (F:\coding\fraud-alert) reads state officials' financial disclosure filings for all 50 states. About a dozen states publish them only as scans or phone photos:

- CO: phone photos.
- KY, WV: copier scans.
- LA: scans with a vendor OCR layer.
- NY, SD: image-only pages.
- IN: legislators' filings are image-only.

The job is to pull specific fields (income sources, business interests, debts, real estate by county, dollar ranges) into records whose values must be exact. A value that cannot be verified is held back, not published. Carmenta was the intended engine. At the time of writing, Carmenta has been read but not yet run on these documents; measured entries will be added once it has.

### 1. No PDF ingest (source, observed)

**Gap.** Neither `ffai-carmenta` nor `ffai-media` opens a PDF; `OcrEngine::recognize` takes an `ImageBuffer`. The caller has to find each page's image XObject, decode it, and handle page rotation and the image's placement on the page itself.

**Evidence.** `pdfimages -list` on the sample filings shows three encodings:

| Encoding | Where seen |
|---|---|
| DCTDecode (JPEG), RGB and gray | KY, WV, IN |
| CCITTFaxDecode (Group 4, 1-bit) | KY page 2, IN, WV stencils |
| Raw 1-bit bitmaps (Flate) | IN |

Page sizes range from 1275×1650 at 150 ppi to 3304×2550 at 300 ppi. KY pages are stored landscape.

**To close it:** a `ffai-media` (or `ffai-carmenta`) entry point that takes PDF bytes and returns the page images, with orientation, in page order. It could also pass through the text layer of born-digital PDFs, so a caller does not OCR a page that already has text. Among our sample states, LA ships an OCR text layer, and FL, NC, and IA are born-digital.

### 2. No CCITT Group 3/4 or JBIG2 decoder in the house stack (source)

**Gap.** Fax-encoded bilevel images are the norm for black-and-white document scanners. The house stack has `rusty_jpeg` and `rusty_png`, but no CCITT or JBIG2 decoder, so a caller has to reach for a third-party crate from crates.io.

**To close it:** a `rusty_ccitt` (G3/G4) crate, and `rusty_jbig2` later. Both are small, well specified, and C-free. Every scanned-PDF user needs them.

### 3. Document accuracy is short of the bar for exact values (source)

**Gap.** The README reports the following. Neither figure distinguishes digits from letters.

| Benchmark | Carmenta CER | Reference CER |
|---|---|---|
| OmniDocBench English holdout | 18.88 % micro | — |
| 43-page set | 23.76 % | 15.51 % (3B GPU reference) |

A disclosure field such as "$150,000 to $249,999" is useless unless every digit is right. At roughly a 1-in-5 character error rate, most multi-digit amounts would fail.

**To close it:**

- **Digit error rate.** Report digit/numeric-field CER separately (the metric a financial-forms buyer asks for first).
- **Field-level confidence.** Report per-field confidence that a caller can gate on.
- **Constrained decoding.** Allow a numeric-only or charset-restricted mode for fields known to be amounts or dates.

### 4. No form structure: key-value pairs, checkboxes, tables as output (source, observed)

**Gap.** `OcrOutput` is blocks → lines → words, with boxes. The `OcrBlock` docs say region classification (title, table, figure) "arrives as typed payloads with the DOCUMENT milestone — deliberately absent from v1". `table.rs` exists (`recognize` → `TableStructure`), but it is not part of `OcrEngine`'s output.

The disclosure forms are form grids:

- labelled cells ("Name of Creditor", "Amount of Liability") with bordered rows;
- ☐ / ☒ Yes/No checkboxes that decide whether a section applies.

**Evidence.** On born-digital forms, the drawn table rules decide row and column membership. The fraud-alert table reader depends on them because centered headings do not line up with left-aligned values. On scans that geometry has to come from the image.

**To close it:**

- **Tables in OCR output.** Put table structure (cells with row and column indices) into `OcrOutput`.
- **Checkbox output.** Add a checkbox or mark detector, emitting checked/unchecked state with a box.
- **Key-value pairing.** Emit label → value pairing for form fields.

These are the features that make a document OCR engine commercially useful for forms.

### 5. Tilt and phone photos (source, observed)

**Gap.** The README states the photographed-receipt pipeline trails PaddleOCR (20.9 % vs 15.6 % CER), and names tilt-sensitive line grouping as the cause, with deskew as the fix. CO's filings (2023 on) are phone photos: the PDF producer is "iOS Quartz PDFContext", and the pages carry no text layer.

**To close it:** deskew plus perspective correction before detection. Rotation detection (0/90/180/270) is also needed, because KY stores portrait forms as landscape pages. Whether Carmenta detects rotation today is **unverified**.

### 6. Run-to-run non-determinism (source)

**Gap.** The README says run-to-run CER varies about 0.5 pp on identical inputs, from a non-deterministic parallel reduction. fraud-alert's rule is that re-running an extraction on the same bytes gives the same record, so a stored value can be re-derived and audited. Output that changes between runs breaks that.

**To close it:** a deterministic mode (fixed reduction order) that yields byte-identical output for identical input and weights, even at some throughput cost.

**Correction (2026-09-24, from the Carmenta log).** The "~0.5 pp run-to-run CER variance" traces to the mission plan's §8.53, which measured a 17 % **speed** noise floor, not CER variance. What the log does record:

- §8.46: recognised text is byte-identical when run alone. Only line confidences move, by about 1e-4, when another process competes for the CPU.
- §8.100: text is byte-identical across threading modes.

So the premise is probably wrong. The batch 1 determinism test measures it directly rather than trusting either reading.

### 7. Weights fetched at runtime into a relative `models` dir (source)

**Gap.** `CraftCrnn::new()` uses `Path::new("models")`, relative to the working directory. The default `fetch` feature downloads weights at first use from hash-verified manifests. A scheduled batch job needs to know where weights live, and must fail fast (not download) when it is offline.

**To close it:**

- **Explicit weights dir.** Document the manifest dir as a required, explicit parameter in production use.
- **Prefetch.** Offer a `prefetch` / `verify` command.
- **No fetch mid-job.** Allow a build without runtime fetch that errors cleanly when weights are missing.

### 8. Per-line confidence is optional (source)

**Gap.** `OcrLine.confidence` and `OcrWord.confidence` are `Option<f32>`, "when the engine reports one". A caller that gates on confidence (keep a value only above a threshold, hold the rest) needs to know which engines always report it. For the CRNN recognizer this is **unverified**.

**To close it:** guarantee confidence from every document engine, and document how it is calibrated.

### 9. No in-memory image decode in `ffai-media` (source)

**Gap.** `ffai_media::load_image` takes only a file path. The byte-level `decode_jpeg` / `decode_png` it dispatches to are private. A caller whose pages come out of a PDF as bytes must either write them to a temporary file or call `rusty_jpeg` directly, which duplicates the grayscale/RGB handling `ffai-media` already does.

**To close it:** a public `decode_image(bytes: &[u8]) -> Result<ImageBuffer>` with the same magic-byte dispatch.

### 10. Model manifests are not shipped with the published crates (source, measured)

**Gap.** `CraftCrnn::new_mobiledet` looks for manifests in `./models/*.toml`. Those TOMLs exist only in the FFai repository (`F:\coding\FFai\models`), not in the `ffai-carmenta` or `ffai-models` crates from crates.io.

Some manifests also name no download source, because the weights are converted locally by FFai's `tools/*.py`, for example `det-fused.safetensors`. On a machine without the FFai repo and a warm model cache, a crates.io user gets `i/o error: The system cannot find the path specified`, which is what fraud-alert got until it copied five manifests by hand.

**To close it:**

- **Ship manifests.** Embed the OCR manifests in the crate (`include_str!`).
- **Hosted weights.** Publish converted weights somewhere a manifest can point to.
- **Clear error.** Make the error name the missing manifest file.

### Measured, 2026-09-24 (Carmenta 0.10.2, `mobiledet-crnn`, CPU)

The pages came from a Kentucky legislator filing (4 pages, 300 ppi, first page JPEG RGB, the rest CCITT G4) and an Indiana one (6 pages, 200 to 400 ppi). Only counts and confidence were recorded, not text.

| Page | As stored: lines, mean conf | Rotated 90°: lines, mean conf | Rotated 270°: lines, mean conf |
|---|---|---|---|
| KY p1 (landscape-stored) | 5, 0.575 | 34, 0.831 | 34, **0.976** |
| KY p2 | 3, 0.380 | 26, 0.824 | 26, **0.984** |
| KY p3 | 16, 0.573 | 24, 0.807 | 23, **0.987** |
| KY p4 | 35, 0.577 | 17, 0.803 | 19, **0.953** |

- **Orientation is the whole difference.** On upright clean scans Carmenta's line confidence is 0.95 to 0.99, with 0 to 1 lines per page under 0.8. On the same pages sideways it is 0.38 to 0.58 and it finds a fraction of the lines. This confirms gap 5: **Carmenta does not detect or correct rotation.**
- **A cheap workaround exists.** The right orientation wins on mean confidence by 0.15 or more on every page. A caller can run 0°, 90°, and 270° and keep the best, at three times the cost. A built-in orientation classifier (or a cheap detector-only pass per orientation) would remove that cost.
- **Confidence is always reported** by `mobiledet-crnn` (every line carried one), which answers gap 8 for this engine.
- **Speed:** about 2 to 3.5 s per page per orientation on CPU for 2550×3300 pages after the first page's model load (17 s cold).
- **Mixed sources within one state (observed):** West Virginia's filings come from very different tools. A pipeline needs to route each file to text extraction or OCR, which is gap 1 again.

  | Producer | Kind of file |
  |---|---|
  | iOS scanner | photos with a text layer |
  | Konica Minolta copier, HP scanner | scans |
  | Chrome/Skia print-to-PDF | born-digital text, no images |
  | iText | generated |

  A page the probe first reported as undecodable was a Skia text PDF with no image at all, not a JPEG decode failure.

### 11. Extracting a page's raw image reads through redactions (measured) — privacy

**Gap.** Gap 1's workaround is to pull the page's image XObject out and OCR it, and that is unsafe. A published Kentucky filing hides the filer's home address under a black patch, and a viewer shows the patch. Extracting the scan and OCRing it returned the full address at high confidence, because the patch is a separate object drawn on top:

- **What the patch is:** a second, small image placed over the scan in an incremental save, not a vector rectangle.
- **Other ways to redact:** a filled rectangle, or a white-out box. All of these exist in public-records PDFs.

fraud-alert now:

1. walks the page's content stream;
2. takes the largest image's transform;
3. paints every rectangle filled after it, and every smaller image drawn after it, onto the pixels (image overlays in black, the fail-safe choice);
4. only then runs OCR.

A page whose drawing can't be walked is not read. After the fix, the same line reads "Home address:" and nothing more.

**To close it:** the PDF entry point from gap 1 must return **the page as rendered** (the scan composited with everything drawn over it), never the raw image XObject. Annotations need the same treatment: a `/Redact` or `/Square` annotation with an appearance stream is also drawn over the page. For a product sold to anyone handling public records, legal discovery or HR files, this is the first thing a privacy review will test.

### 12. Confidence does not flag wrong names (measured)

**Setup.** One Kentucky legislator's filing (3 pages, JPEG and CCITT, 300 ppi, stored landscape) was read by hand against the Carmenta output. There are 17 answer lines (typed entries under the form's questions), and every one passed the 0.85 line-confidence gate.

| Measure | Result |
|---|---|
| Lines exactly right | 7 of 17 (41 %) |
| Entity name right, punctuation aside | 13 of 17 (76 %) |
| Wrong entity name | 4 of 17: "401K" → "Z0TK", "Irish Hill" → "Trish HTlI", "I/ONX" → "IJONX", "Gulf Shores" → "GuIF Shores" |
| Symbols in the printed form | "$10,000" → "810,000", "$1000" → "S1OOO", "n/a" → "nla" |
| Clean printed paragraphs | near-perfect |

- **Where the errors are.** They cluster on typed lines whose underline rule runs through the descenders. That is a form-specific condition, and the detector/recognizer doesn't separate the rule from the text.
- **Confidence doesn't help.** It did not separate these lines from correct ones, so a caller cannot gate names on it.

**A second filing, read the same way** (18 answer lines, all passing the gate):

| Measure | Result |
|---|---|
| Entity name right | 14 of 18 |
| Wrong | "Tri-State" → "Tr1-State", "Cabell" → "Cabe]l", "Merrill Lynch" → "MerrITLynch", and one handwritten line turned into noise ("N/A (… permission 2/11/26)" → "Nlt (p mpermssion 4"/a)") |
| Street numbers on struck-through lines | **every one lettered**: "1000" → "TOOU", "1117" → "TTT7", "1121" → "TT2T", "#305" → "#30S" |

**Across both filings:** 27 of 35 entity names right (77 %); 8 wrong, and none flagged by confidence.

- **Numbers on underlined forms can't be used.** On lines the form's underline runs through, digits come out as look-alike letters (1 → T/I, 0 → O/U, 5 → S). Any numeric field on an underlined form is unusable as read.
- **Why it matters for privacy.** A digits-based address filter misses these, so a caller has to treat digit look-alikes as digits when deciding what is an address.
- **Handwriting.** Handwritten notes are read as confident noise rather than rejected. A handwriting detector (or a low confidence on them) would let a caller drop them.

**To close it:**

- **Rule removal.** Strip horizontal rules from the image before recognition, since form underlines are universal.
- **Calibrated confidence.** Make confidence track character errors, most importantly for the `$`/`S`/`8` and `I`/`l`/`1` confusions.
- **Lexicon constraint.** Allow a caller-supplied lexicon, such as the form's own words or a list of known entity names.

Until then, fraud-alert stores OCR'd names only as flagged, unverified leads.

### 13. Throughput with orientation search (measured)

Three orientation tries on landscape-stored 3-page filings took **23 to 30 s per filing** on CPU, about 8 to 10 s per page. The first filing was 64 s, including the model load. Kentucky's 259 legislator filings come to about 2 hours for a full re-read. That is acceptable for a batch job, but an orientation classifier (gap 5) would cut it by about two-thirds.

### 14. Checkbox and tick marks read as characters (observed)

Indiana's statement (a RICOH copier scan at 200 ppi) marks "Your interest / Spouse's interest / over $500,000" with hand ticks in table columns. Carmenta returns them as characters ("@", "0", "D", "B") at the confidence of text, so a caller can't tell a tick from a letter. This is the same form-structure gap as gap 4, measured: **the one value field on Indiana's form is unreadable through Carmenta.**

**Workaround in use:** orientation found on a document's first page is tried first on the rest, and kept when mean confidence is at least 0.9. That turns three readings per page into about one, because a scanner feeds a document's pages the same way. An engine-level orientation classifier would still remove the first page's extra readings.

---

*Next:* when Carmenta runs on the sample KY, WV, IN, and CO filings, fraud-alert will add measured numbers here:

- lines per page;
- mean confidence;
- the share of dollar amounts that parse and match the form's printed range choices;
- pages per second on CPU;
- whether rotated KY pages read correctly.
