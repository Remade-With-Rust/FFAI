# Carmenta recognition — the unowned stage

**Created 2026-08-16, because nothing owned it.**
`carmenta-ordering-audit.md` runs Stages 0–6: Detection, Line identity, Corridor
splitting, Cut strategy, Selector objective, Post-recognition reorder, Emission.
Stage 5's name places RECOGNITION as an unnumbered given between Stage 0 and
Stage 5. That was deliberate — the campaign had concluded "the characters are
not the problem" (§8.53: order-free we sit 1.63 pp from Unlimited-OCR).

**That conclusion has expired, and three independent measurements now say so.**

---

## The measurement that opened this

| probe | finding |
|---|---|
| Stage 0 / 0.1 split | detection owns **0.40 %** of missing text, recognition **7.99 %** |
| Stage 1 / 1.2 | **62 %** of missing characters are in perfectly TILED blocks |
| geometry elimination | **93.3 %** of missing characters are in blocks that are geometrically perfect |

The last one is the headline. Those blocks have:

* one box per row-band (no horizontal fragmentation)
* normal single-line height (no vertical merging — merged boxes read **1.000**)
* x-extent **0.995** of the GT block width (no clipping)
* aspect nowhere near the 2048 input clamp (that binds on 0.66 % of lines, and
  those blocks read **1.000**)

…and they lose **9.5 % of their characters** anyway.

**Four geometric hypotheses were tested and every one came back backwards.** The
strongest — that 35–50 px boxes were merged line pairs — inverted completely:
boxes at 1.8–2.5x the page's normal line height read PERFECTLY and carry 0.6 %
of the loss.

There is no geometry story left. The recognizer drops roughly one character in
eleven from a correctly-cropped, correctly-detected single line.

**Floor, not ceiling:** char-ratio counts LENGTH, so it sees deletions and
length-changing substitutions but not equal-length ones. True recognizer error
is ≥ 9.5 %.

---

## The first question, and it is not "which knob"

Our SVTR is a hand-port matched to a paddle fixture (§8.167). **A fixture is one
crop.** A corpus-wide 9.5 % deletion rate is invisible to it. So before any
sweep, split the two candidates that cannot be told apart from inside:

1. **Our PORT loses characters** → ours reads worse than the reference running
   the same weights. Fixable by us, and free.
2. **The MODEL is the ceiling** → ours matches the reference exactly, and the
   mobile tier simply cannot do better. Fixable only by changing models.

`onnxruntime` is installed (2026-08-16, for CDM). `PP-OCRv5_mobile_rec` ships as
ONNX. So this is a direct comparison on identical input, exactly like the
layout/table/formula validation that found four silent defects in one day:

> **R.1 — Port oracle on REAL crops.** Take ~500 line crops spanning the sources
> that lose most (`exam_paper` 0.898, `note` 0.923, `book` 0.944), feed the
> BYTE-IDENTICAL tensor to our SVTR and to `PP-OCRv5_mobile_rec` under ORT, and
> score both against GT. *Ours ≈ reference means the port is clean and the model
> is the wall. Ours worse means we have been leaving characters on the floor
> since the port landed.*
>
> Hand the reference OUR tensor, not our image — an oracle validates exactly the
> span you hand it (§48, learned by letting a white-padding bug survive a
> token-perfect match).

> **R.2 — Tier check.** `PP-OCRv5_server_rec` exists. We ship MOBILE. If R.1 says
> the model is the wall, the question becomes what the server tier reads on the
> same crops and what it costs. Mobile-vs-server was a SPEED decision for the
> detector (§8.19-era); it was never priced for the recognizer.

> **R.3 — Where the loss is, inside the line.** If the port is clean, is the
> drop at line ends (crop geometry after all) or distributed (model capacity)?
> Align our output to GT per line and histogram deletion POSITION. Distinguishes
> a fixable margin from a genuine accuracy ceiling.

---

---

## R.1 RESULT (2026-08-16) — the port is CLEAN, and the GT is not what I compared against

**Our SVTR matches the reference.** 25 of 29 real crops decode identically to
`PP-OCRv5_mobile_rec` under onnxruntime on byte-identical tensors, and all four
differences are SPACES our decode keeps and the reference's config drops — we
emit MORE characters, not fewer. There is no port defect to fix.

**Then the root cause looked like GT LaTeX**, and blocks carrying inline `$..$`
did hold 77.1 % of the missing-character mass at 25.8 % of their own characters,
against 3.2 % for plain text. That independently reproduced §40's 87 %.

**⚠ AND THAT MEASUREMENT USED THE WRONG GT.** `end2end_dataset.py:1956` runs
every text block through `textblock2unicode` BEFORE scoring: inline `$...$` spans
whose content contains `\`, `^` or `_` are converted to UNICODE and spliced back
in. `$d_{w}$` is normalised away; `$(D)$` is not (no `\^_`). I compared our
output to the RAW `text` field — a string the benchmark never scores — having
written down "an oracle validates exactly the span you hand it" the same day.

`rec_gt_normalized.py` re-prices against the scorer's own normaliser. **Every
number below is provisional until it lands.**

---

## The fix strategy — staged, cheapest kill first

Each step must be able to KILL the ones after it. Nothing is built before the
step that prices it has run.

### F.0 — Re-price against normalised GT *(RUNNING)*
`rec_gt_normalized.py`, offline, the evaluator's own `textblock2unicode`.
Reports missing characters against RAW and NORMALISED GT side by side.

> **KILL GATE.** If normalisation removes most of the apparent gap, inline maths
> is largely a non-problem and this entire plan stops here. The remaining loss
> would then be ordinary recognition error on plain text at ~3.2 %, which is a
> model-tier question (R.2), not an inline-LaTeX one.

### F.1 — Price the ceiling with an oracle
Synthesize a prediction set identical to `fullnull` EXCEPT that every inline
maths span carries GT's own (normalised) content, and score it. That is the
absolute upper bound of a perfect inline-formula pipeline on Text^Edit.

> **KILL GATE.** §37's ordering oracle was worth +0.0378 and justified a
> campaign; if this oracle is worth +0.002, no amount of engineering pays.
> Price before building — the standing law that saved Stage 0.

### F.2 — Character x-positions from CTC *(the enabling capability)*
Splicing LaTeX into a line needs to know WHERE in the emitted string a pixel
span falls. CTC already provides it: `ctc_greedy` iterates timesteps, and each
kept timestep IS an x-position in the resized crop, mappable back to page pixels
through the crop rectangle. Return `(char, x0, x1)` alongside the string.

> **GATE: byte-identical text output.** This is a pure addition; if the emitted
> string changes at all, the change is wrong. Deterministic, no scoring run.

### F.3 — Inline region detection: recall and false-positive rate
Layout DOES emit inline formula regions — a 144x34 box sitting inside a GT text
block, confirmed. But they score **0.43–0.45**, below both `FFAI_ROUTE_SCORE`
(0.60) and near the layout floor (0.45). Measure, against GT inline spans: what
fraction do we find at each threshold, and how many false positives come with
them?

> **GATE.** A false inline formula REPLACES correct words with a LaTeX
> rendering of them. Same asymmetry as the table guard, same discipline: measure
> the blast radius before lowering any threshold.

### F.4 — Splice, with a refusal guard
Replace the character range a formula region covers with `$...$`. Reuse the
`retain`-style guard: if the LaTeX carries less content than the characters it
displaces, refuse and keep the text. The router already proves this pattern —
and §49 proves that shipping without it costs +0.0347.

### F.5 — Gate on Text^Edit, not on character counts
char-ratio localises; Text^Edit decides. Screen on the 305-page subset
(±0.005), bank on the full corpus at `--max-cores 8`.

---

## What this stage must NOT repeat

* **Do not sweep first.** Stage 0's step 0.2 named CRAFT knobs on a DBNet path;
  it would have swept four values of something never executed and "refuted" it.
  Trace the call path, then measure, then sweep.
* **Do not trust a proxy.** char-ratio localises; ReadOrder/Text^Edit decide. Five
  refutations were bought by judging ordering on inversions while CER is
  character-weighted.
* **Do not generalise from a page.** `page-3ecc67a1` argued the thin-block ceiling
  was an artifact; the corpus said the opposite.
* **Bound the cores.** `--workers N` is processes, not cores: candle hands
  `Parallelism::Rayon(num_cpus::get())` to every matmul. `gg_arm.py --max-cores`
  (default 8) exists now; capture scripts still need the same treatment.

---

## Standing context

* Baselines: `odb_pred_fullnull` (routing OFF) text **0.1281** / order **0.2339**;
  `odb_pred_fullroute` (routing ON, guarded) **0.1298** / **0.2153**. Same binary.
* Scorecard: `.tools-bench/scorecard.py` — text, order, TEDS, TEDS-structure,
  CDM, Overall, fallback counts. Refuses to print Overall unless all three terms
  exist.
* Probes written for the ordering audit that apply here unchanged:
  `stage0_coverage.py` (char-ratio per GT block), `stage1_tiling.py`
  (boxes per row-band vs char-ratio).
