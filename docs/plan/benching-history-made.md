# Benching History — the Carmenta campaign on a correct instrument

**Opened 2026-08-08, immediately after §8.173 withdrew every competitive claim
this campaign had made.** The engineering was sound and the scoreboard was not.
This plan restarts the competitive half of the campaign on OmniDocBench's own
evaluator, and carries forward every lesson the wrong scoreboard taught us.

---

## 1. Where we actually stand

Measured by OmniDocBench's official harness, 236 held-out English pages, both
engines through one evaluator, paired per page:

| | Text^Edit ↓ | ReadOrder^Edit ↓ |
|---|---:|---:|
| **Unlimited-OCR** (3B MoE VLM, GPU) | **0.0406** | **0.0522** |
| **Carmenta** (shipped: mobiledet + SVTR, CPU) | 0.1084 | 0.1749 |
| gap | **+0.0679** CI [+0.0485, +0.0884] | **+0.1227** CI [+0.0960, +0.1506] |
| pages they win / we win | 172 / 35 | 96 / 7 |

Both intervals exclude zero. **We are 2.7x behind on text and 3.4x behind on
reading order.** Their 0.0406 on our subset matches their published 0.038, so
the reference is sound and the gap is ours.

Context on the public v1.6 board — and note our category:

| category | n | |
|---|---:|---|
| Specialized VLMs | 23 | |
| General VLMs | 7 | |
| **Pipeline Tools** | **2** | **ours** |

| Pipeline Tools | Text^Edit | ReadOrder^Edit |
|---|---:|---:|
| MinerU-Pipeline | 0.0550 | 0.1530 |
| **Carmenta** | **0.1084** | **0.1749** |
| Marker | 0.1570 | 0.2430 |

Published Text^Edit spans 0.0326 (PaddleOCR-VL-1.6) to 0.157 (Marker);
ReadOrder^Edit spans 0.116 to 0.243. **Every published row averages all five
languages; ours is English-only, so no row-by-row rank is honest yet.**

---

## 2. What we are playing for

`Overall = ((1 - text_edit) x 100 + table_TEDS + formula_CDM) / 3`

**Overall is not computable for Carmenta** — we emit neither tables nor
formulas, so two of three terms are zero by construction. Chasing Overall means
building a table and formula engine, which is a different project.

**Reading order is NOT in Overall.** It is a published column but not part of
the headline. So the two candidate objectives pull apart:

| objective | argument for | argument against |
|---|---|---|
| **Text^Edit** | the published headline; 172 of 235 page losses live here | we are 2.7x behind, the smaller of the two gaps |
| **ReadOrder^Edit** | our worst column (3.4x); four shipped mechanisms already aimed here | not in Overall; a leaderboard reader may never see it |

**DECISION: primary objective is Text^Edit; reading order is tracked as a
secondary gate and must not regress.** Rationale: it is the number anyone
checking our claim will look at, and it carries the majority of our per-page
losses. Reading order stays instrumented because it is where our existing
machinery lives and because a text win bought by scrambling order is not a win.

A second, independent front is available and cheap — see §7.

---

## 3. The instrument contract

Non-negotiable rules for every number in this campaign. Each is paid for.

1. **Every competitive number comes from the reference's own evaluator.**
   §8.173: our concatenate-then-edit-distance scorer inflated our error 1.58x
   and Unlimited-OCR's 4.37x — a 2.8x bias in our favour that inverted the
   standing. Running their harness took under an hour and overturned fifty
   sections of framing.
2. **One instrument on both arms is NECESSARY AND NOT SUFFICIENT.** The
   instrument must also be *neutral with respect to how the arms differ*. Our
   scorer had a systematic ordering axis and our detector sat closer to it by
   construction. Before any cross-engine claim, ask: *what does this instrument
   reward that is not the thing I am measuring?*
3. **Fit on train, judge ONCE on holdout.** Splits in §4. A number computed on
   pages used for fitting is a training loss, not a result.
4. **Bootstrap CI over pages must exclude zero.** Point estimates are not
   verdicts. §8.171 shipped a "+0.60 pp lead" whose CI was [-3.13, +4.89].
5. **Rank targets by ABSOLUTE contribution, never by ratio.** A 0.9 mean over
   13 pages is noise; 0.15 over 148 is the campaign.
6. **Pre-register the prediction before running.** State what you expect and
   why. §8.171's prediction was wrong in our favour and saying so out loud is
   what made the result informative rather than a rationalisation.
7. **Positive control on every A/B.** The unchanged arm must reproduce its known
   number. If it does not, the run is void before you read the changed arm.
8. **Suspect the artifact when a number AGREES too exactly** — unless it is the
   unchanged arm, where exact agreement is the control passing. A rebenchmark
   once returned the disabled arm's figure to three decimals because the default
   was flipped in source and built only into `target-probe`.
9. **An impossible number is the instrument asking for help.** 730 % and
   6 204 % CER were noted as curiosities and left; they were the alignment bug
   that voided a whole per-class table.
10. **Never silently narrow the population.** Report every skip, drop, timeout
    and fallback. A shrinking denominator turns a claim into a claim about a
    different set.
11. **Three probes before any refutation, varied on an axis that could flip the
    answer.** A wrong keep gets caught by the next gate; a wrong refute is
    permanent and removes a lever from the search space forever.
12. **Land structural changes INERT first**, then move one number. §8.171's
    per-recognizer threading shipped equal-to-old so the behaviour change was a
    single constant.
13. **Record WHY a revert happened** — "measured worse" and "inside the noise"
    are different states and only one of them is re-openable.
14. **Watch the encoding.** cp1252 destroyed 59 of Unlimited-OCR's 236 pages in
    one run and kills OmniDocBench's own harness on Windows. `PYTHONUTF8=1`.

---

## 4. Splits — re-cut for the real benchmark

| set | pages | status |
|---|---:|---|
| fitted on (permanently dirty) | 80 | fit only, never judge |
| already judged (v1 holdout) | 236 | judged once; reusable only for diagnosis |
| **NEVER SEEN — the real holdout** | **439** | **judge once, at the end** |
| Chinese / mixed / other | 894 | out of scope: English-only recognizers |

The 439 were excluded from the old corpus *only* because they contain tables or
equations. `text_block` and `reading_order` are scored per-category, so those
pages are fully usable for both our metrics — a clean holdout **2.3x** the size
of the old one, weighted toward exactly what we are worst at:

`academic_literature 148 · PPT2PDF 95 · book 87 · exam_paper 70 ·
colorful_textbook 21 · newspaper 13 · magazine 3 · research_report 1 · note 1`

**Discipline:** develop against the 80 + diagnose on the 236; the 439 is spent
ONCE, at the end, and its number is the one we publish.

---

## 5. Phase 0 — validity checks BEFORE any optimisation

Both are config-only. Both could change what we optimise, so neither is
skippable.

**0A. Re-validate the shipped ordering mechanisms on the TRUE metric.**
§8.156 (+0.562), §8.157 (+0.231) and §8.160 (+0.987) — **+1.78 pp combined** —
were fitted *and* judged against the scorer §8.173 discredited. The official
harness grades reading order in its own column. Those mechanisms may be weaker,
neutral, or harmful on the real metric. Run each `FFAI_ORDER_*` arm through the
official evaluator on the 236.

> This is the highest-stakes unknown on the board. It is 1.78 pp of shipped
> behaviour resting on a proxy we now know was distorted. Whatever it says, it
> is a finding: confirmation makes the machinery trustworthy, refutation frees
> us from defending it.

**0B. `FFAI_BODY_ONLY` on/off under the official protocol.**
Their GT scores `header`, `footer`, `page_number`, `figure_caption` and
`reference` as their own categories. Body-only *deletes* that text. Under our
scorer it was a large win because it stripped junk; under theirs it may be
discarding content that is being counted. Zero code, potentially large.

---

## 6. Phase 1 — decompose, then attack in contribution order

**1A. Segment the full 1651-page run** by every axis their harness exposes:
`language`, `data_source` (10 doc types), layout, `text_background`,
`text_rotate`. For each segment report **n, mean edit, and share of total
error**, for text_block and reading_order separately. Partition explicitly into:

- **out of scope** — Chinese/mixed (894), rotated text. Quantified, never chased.
- **addressable** — everything else, ranked by absolute contribution.

**1B. Carry forward the one diagnosis that survives the instrument change.**
The 72 small pages (<1500 chars) reading 34.19 % against 28.32 % was measured on
the biased scorer, so the *margin* is void — but page-weighting genuinely does
charge a mangled 200-character slide as much as a clean 5 000-character journal
page, and that is a property of the metric, not of our scorer. Re-measure it
first on the official numbers.

**1C. Attack in contribution order**, one lever per brick, each gated by:
fit on the 80 → confirm on the 236 → bootstrap CI excluding zero → reading order
must not regress → record the number and the revert reason if it fails.

---

## 7. Second front — the Text OCR task

OmniDocBench has **six separate tasks**, not one. `### Text OCR Evaluation` is
region-level recognition: the harness hands the model ground-truth boxes and
scores the text. **That is exactly what Carmenta's recognizer is**, with layout
and reading order removed as confounders — and it publishes per-language columns.

| model (Text OCR task, EN) | Edit ↓ |
|---|---:|
| Mathpix | 0.033 |
| GOT-OCR | 0.041 |
| Surya | 0.057 |
| OpenOCR | 0.070 |
| PaddleOCR | 0.071 |
| Tesseract-OCR | 0.096 |
| **EasyOCR** | **0.26** |

**`craft-crnn` IS the EasyOCR model stack, ported to Rust.** A direct Text OCR
number against EasyOCR's 0.26 is the cleanest apples-to-apples claim available
to this project, and it isolates recognition from the ordering problem entirely.
Cheap to run — no detection, no ordering, just crops. **Do this early:** it may
be a publishable result long before end-to-end parity is plausible.

---

## 8. Definition of done

A claim is publishable when ALL hold:

- computed by OmniDocBench's official evaluator, unmodified except documented
  environment fixes (`lxml>=5.2` for py3.11 wheels, `PYTHONUTF8=1` for their
  cp1252 bug);
- on the 439-page never-seen holdout, or the full 1651 with language stated;
- compared against numbers from the same task and the same language scope —
  never an all-language row against our English-only figure;
- accompanied by the reproduction kit: corpus manifest SHA-256s, commit hash,
  exact command, and the adapter used for any engine we measured ourselves;
- carrying its scope in the sentence: *English, text-only, no tables or
  formulas, N pages* — not in a footnote.

Note the dataset licence: **research use, not commercial.** Apache-2.0 code,
restricted data. Check before it becomes load-bearing in anything public.

---

## 9. Standing log

| date | event |
|---|---|
| 2026-08-08 | §8.173 — instrument bias found; all competitive claims withdrawn |
| 2026-08-08 | official baseline: Text 0.1084 / Order 0.1749 (236 EN pages) |
| 2026-08-08 | full 1651-page run in flight |
