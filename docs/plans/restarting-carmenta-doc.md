# Carmenta OCR — launchpad for the next optimizer

You are picking up a document-OCR engine mid-campaign. This page is the
shortest path to being productive: what the thing is, **how to measure it
without being lied to**, what has already been tried, and where the remaining
value is. Read §1–§3 before running anything.

The blow-by-blow campaign log is `docs/plan/benching-history-made.md` (§1–§32).
This is the summary; that is the evidence.

---

## 1. What Carmenta is, and where it stands

A pure-Rust document OCR pipeline on candle. The default document engine is
`mobiledet-svtr`: a DBNet detector (~4.7 MB) plus PP-OCRv5 SVTR recognition
(~16 MB, an 18 385-class CJK+Latin head). No VLM, CPU only, ~9–17 s/page.

**Current banked standing** — OmniDocBench v1.6, all 1 651 pages, scored by
their official evaluator:

| | score | context |
|---|---:|---|
| Text^Edit | **0.1307** | published range 0.0326 (PaddleOCR-VL) – 0.157 (Marker) |
| ReadOrder^Edit | **0.2348** | published range 0.116 – 0.243 |

Ahead of the worst published row on both columns, well behind the leaders.
English-only the engine reads ~0.107 text; non-English ~0.15.

**Speed is not the problem.** Roughly 9× faster than the reference VLM — our
CPU against its GPU. Every open lever is quality. Spend speed for quality
without hesitation; that trade has been made twice and was right both times.

---

## 2. THE INSTRUMENT — read this before you measure anything

The most expensive lesson of this campaign: **our own scorer was biased 2.8× in
our favour** and produced a year of confident, wrong competitive conclusions
(§8.173). Four shipped mechanisms turned out to be artifacts of it.

### Rules that are not optional

1. **Only OmniDocBench's own evaluator counts.** It lives at
   `C:/Users/talmo/coding/omnidocbench-eval` with its own venv. Drive it through
   `.tools-bench/score_arm.py`, never by hand.
2. **`PYTHONUTF8=1` always** — their harness reads the GT JSON without an
   explicit encoding and dies on Windows cp1252.
3. **A missing prediction file scores 1.0, it is NOT omitted.** Verified: 296
   absent pages all read exactly 1.0000. This has produced two false results.
   The only real exclusion is filtering the GROUND TRUTH to the pages both arms
   produced — which is what `score_arm.py` does.
4. **One binary, two env settings.** Never compare across builds. Every lever
   ships behind an env toggle so the A/B is one binary.
5. **Bootstrap CI over pages must exclude zero.** Point estimates are not
   verdicts.
6. **Price the ceiling BEFORE building — on every column the change can reach.**
   §17 priced reading-order, the payoff landed on text, and a real win was
   nearly discarded.
7. **Share of error is not recoverable prize.** Three levers died on this: a
   segment carrying 21.6 % of order error had a +0.0049 ceiling. Rank by
   contribution to choose what to PRICE, never what to BUILD.
8. **Suspect any number that agrees too exactly, or moves too much.** A guard
   toggle that changes 6 pages cannot move the metric 5× — when it appeared to,
   the aggregate file was stale from a previous run.

### Harness traps, all hit at least once

- `ocr_batch` crashes on long runs (three signatures: timeout, `0xC0000409`,
  `0xC0000005`). Chunk it, stream to disk, resume. **Output that exists only
  inside a running process is not output.**
- A crashed page written as `""` scores 1.0 — a subprocess crash masquerading
  as a quality failure. It cost 0.0056 of the headline before it was caught.
- Aggregate `metric_result.json` is not deleted on rescore. Read per-page files,
  or verify the timestamp against the run you care about.

---

## 3. Tooling

| tool | what it does |
|---|---|
| `.tools-bench/gg_arm.py` | run an engine ARM (env-toggled) over a language subset, parallel, GT-filtered |
| `.tools-bench/score_arm.py` | **the correct scorer** — population-matched GT, both arms, CIs |
| `.tools-bench/odb_segment.py` | decompose the banked run by language / doc type / layout |
| `.tools-bench/order_ceiling.py` | oracle ordering ceiling per segment |
| `.tools-bench/textocr_task.py` | the region-level Text OCR task (crops) |
| `_greatgate/gate-calculator` | offline rule search over a feature CSV (gitignored, local only) |
| `FFAI_GATE_HARVEST` | per-page decision-time feature tap, inert unless set |

Engine toggles that exist: `FFAI_ORDER_GATE`, `FFAI_ORDER_PROBE`,
`FFAI_ORDER_VERIFY`, `FFAI_ORDER_GUARD`, `FFAI_CJK_FLUENCY`, `FFAI_BODY_ONLY`,
`FFAI_DB_BIN`, `FFAI_DB_BOX`, `FFAI_DB_UNCLIP`, `FFAI_ARM_ENGINE`.

---

## 4. The ledger — what is already answered

### Banked wins (do not re-litigate)

| change | effect | where |
|---|---|---|
| SVTR recognizer over CRNN | +1.521 pp on the old scorer; re-confirmed by §32 | §8.170 |
| Script-aware `join_fluency` (CJK arm) | +0.0073 text, CI excludes 0 | §18 |
| **Body-only suppression OFF** | **+0.0089/+0.0582 EN, +0.0486/+0.1096 non-EN** | §16, §19 |
| Harness repair (crashes scored as 1.0) | +0.0056 text, +0.0033 order | §28 |

### Confirmed real on the correct instrument

- The §8.160 **text verifier** — disabling costs 0.0083 text, CI excludes 0.
- The §8.157/§8.160 **probe reorder** — disabling costs 0.0126 text.
- The **engine choice**: `mobiledet-svtr` beats `craft-crnn` end-to-end by 3×
  (0.1073 vs 0.3522). CRAFT is better at RECOGNISING a supplied region and far
  worse at FINDING regions — different jobs, and the crop task only measures
  the first.

### Refuted, with mechanism — do not retry without new information

| lever | why it died |
|---|---|
| Widening the ordering gate axis | the reorder is invisible to the metric at our granularity |
| **Block-level grouping** | MGAM merges but cannot split: coarse output is irrecoverable. Would have DOUBLED text error |
| Raster ordering candidate | regresses good crops 10×, closes only 570 of 2 911 bad |
| Ordering on 3+ float pages | ceiling +0.0049 — the error is text, not sequence |
| Leading/trailing furniture strip | −0.0041 text, CI excludes 0 |
| `FFAI_DB_BIN` / `BOX` / `UNCLIP` sweeps | flat — detection thresholds are not the mechanism |
| Dropping the §8.157 guard | changes **6 of 755 pages**, net −0.000023. Inert |

---

## 5. Known defects still open

1. **A dead constant.** `probe_gate_fires` tests `body_frac > 0.85`, where
   `body_frac = n_body / n_all` measures how much SUPPRESSION removed. Since
   body-only went off (§19), `n_body == n_all`, so it is **always 1.0 and the
   term is always true**. Harmless in practice — the guard-off arm proved it
   moves 6 pages — but it is a lie in the source and should be deleted or
   re-fitted. **General form: a constant is fitted against a whole
   CONFIGURATION, not one component. Any config flip must re-audit every
   constant whose inputs it touches.**
2. **54 non-English pages under-emit** (median 0.48 of GT length). Not
   detection thresholds, not post-processing. Unexplained, and the largest
   single unexplained segment. Prize if fixed to typical: ~+0.017.
3. **`ocr_batch` crashes** on long runs. Tooling debt, not an engine defect.
4. **`num_seq_monotone` is ASCII-only** — cannot see 一二三. Small but real.

---

## 6. Where the remaining value is

Segments of the banked 0.1307 text, by share of error:

| segment | share | notes |
|---|---:|---|
| non-EN, all float counts | **55 %** | the language is the driver, not floats |
| EN no-float | 14 % | worst English segment (0.1656), also the biggest ordering ceiling |
| EN with floats | 25 % | ordering ceiling here is ~0 — it is a text problem |
| degenerate (≤1 region) | 5 % | metric artifact, unfixable, exclude from targets |

**Start with the non-English 55 %.** It is over half the error, it is NOT
post-processing (§31 audited every filter), it is NOT detection thresholds
(§28), and recognition is demonstrably good — sampled CJK output is
character-accurate against GT. Something else is wrong and nobody has found it.
That is the honest frontier.

**A quarter of remaining ORDER error is the metric's degenerate case** — pages
with ≤1 orderable region scoring 0.6–0.7 where no sequence exists to get wrong.
Excluding them, the order column reads 0.1941, mid-pack on the published board.
Know that number; do not target it.

**Publishable now:** `docs/textocr-claim.md` — `craft-crnn` (our Rust port of
the EasyOCR stack) reads 0.1051 on the benchmark's own Text OCR task against
published EasyOCR's 0.26, with four disclosures attached.

---

## 7. How to run your first experiment

```bash
# 1. an arm: one env toggle, one language subset
FFAI_ARM_ENGINE=mobiledet-svtr python .tools-bench/gg_arm.py \
    --name myarm --lang english --env FFAI_SOMETHING=0 \
    --base-save odb_pred_nb_en_quick_match --workers 3

# 2. score it CORRECTLY (population-matched GT, both arms, CIs)
python .tools-bench/score_arm.py --arm myarm --base nb_en
```

Bank it only if the CI excludes zero on `text_block` **and** `reading_order`
does not regress. Text is the primary objective; order is a non-regression gate.

Record every result — **including refutations, which are most of them** — in
`docs/plan/benching-history-made.md`. A refutation with a measured mechanism is
worth as much as a win: this campaign killed two multi-day engine builds with
analysis scripts, and one of them would have doubled our text error.
