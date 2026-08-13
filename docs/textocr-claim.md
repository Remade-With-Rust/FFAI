# Carmenta on the OmniDocBench Text OCR task

**Claim.** On OmniDocBench v1.6's Text OCR task, restricted to English pages,
Carmenta's `craft-crnn` engine reads **0.1051** normalized
edit distance (95 % CI [0.1013, 0.1091],
7019 block-level regions across 694 pages) against
**published EasyOCR's 0.26** on the same task's English column.

`craft-crnn` **is** the EasyOCR model stack — CRAFT detection + the `english_g2`
CRNN recognizer — reimplemented in pure Rust on candle. Same models, same task,
different engineering.

## Published comparison column (Pipeline Tools & Expert Vision Models, EN)

| model | EN edit distance |
|---|---:|
| Mathpix | 0.033 |
| GOT-OCR | 0.041 |
| Surya | 0.057 |
| OpenOCR | 0.07 |
| PaddleOCR | 0.071 |
| Tesseract-OCR | 0.096 |
| EasyOCR | 0.26 |
| **Carmenta `craft-crnn`** | **0.1051** |

## Scope — this travels with the number

* **English pages only** (694 pages, 7019 `text_block`
  regions). Not the full 1651-page benchmark, not the composite Overall score.
* **Block-level** regions, matching their stated protocol: "evaluates OCR at
  the text paragraph level".
* Ground-truth boxes are supplied by the benchmark; this task measures
  RECOGNITION, not detection or reading order.

## Disclosures

1. **We reproduced the metric.** The v1.6 repository does not ship this task's
   pipeline (`configs/ocr.yaml` and its recognition dataset are absent), so the
   formula was taken from their `src/metrics/cal_metric.py`:
   `Levenshtein(pred, gt) / max(len(pred), len(gt))`, averaged over regions.
   Whitespace is normalized on both sides.
2. **The published rows predate the v1.5/v1.6 annotation corrections.** Their
   changelog records OCR ground-truth fixes in both releases, so the comparison
   is directional rather than exact. A same-release re-run of EasyOCR would be
   required for an exact claim, and we have not done one.
3. **Our document default is a different engine.** `mobiledet-svtr` (DBNet +
   SVTR) is what ships for document OCR and it scores
   **0.4133** on this task — *worse*. The cause is measured:
   on tight, margin-free crops DBNet fragments text into ~5-character pieces
   (median 5.0 chars/line, against 28.0 on full pages, a 5.6x difference), and
   the fragment joins inject spurious spaces. That is a property of the crop
   condition, not of page parsing: the same engine reads 0.1073 end-to-end on
   full English pages where it must also do its own detection. We report
   `craft-crnn` here because it is the like-for-like EasyOCR comparison, and we
   disclose that it is not our default.
4. **Environment deviations** from their pinned setup: `lxml>=5.2` (the pinned
   4.9.1 has no Python 3.11 wheels) and `PYTHONUTF8=1` (their harness reads the
   GT JSON without an explicit encoding and fails on Windows cp1252). Neither
   touches text scoring.

## Reproduction

```
commit          510261e27da0
engine          craft-crnn (CRAFT + english_g2 CRNN, pure Rust on candle)
dataset         OmniDocBench v1.6, 1651-page release (Apache-2.0, research use)
manifest sha256 e414e1c68c346f157d2277884bdab61d5c20bbb61421fc26ec60a60c7bb7eca6
filelist sha256 f1336ba9ea1e898b2212ab422877f5c5312a856cad7b1cef91ba226ba0440ea1

python .tools-bench/textocr_task.py --stage crops
python .tools-bench/textocr_task.py --stage run --engine craft-crnn
python .tools-bench/textocr_task.py --stage score
```

**Dataset licence:** OmniDocBench is research-use-only; the evaluation code is
Apache-2.0. This result is a research measurement.
