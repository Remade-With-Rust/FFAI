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

---

## 10. The Great Gate calculator — the instrument for Phases 0A and 1C

Imported 2026-08-08 from `remade_ffmpeg_rs@ae6b5ce` to `/_greatgate/`
(**gitignored, `publish = false`, detached workspace, outside
`members = ["crates/*"]`** — three independent guards; offline analysis tooling
and its harvests never ship).

**This is our own tool coming home generalized.** Its header cites this
campaign by section — §8.106–§8.119 — and it kept our vocabulary: the CSV's
`clip` column takes alias **`page`**, and `clip_total` takes **`page_chars`**.
It went FFai suppress-optimizer → `rs_h264/_greatgate/` → generalized here, and
independently re-derived the same laws the codec dispatch skill distilled.

**What it adds over the optimizer we already had:**

| capability | why it matters HERE |
|---|---|
| `shipped` column → **incremental** scoring | scores only units the shipped gate still keeps. This is the ghost-branch trap exactly: 92 % of a candidate's targeted lines were already dropped, replay read −0.057 and the engine −0.988. A standalone table cannot see that. |
| `top3` concentration column | §8.114 — a rule scored +4 407 micro and three clips carried all of it; removing them took it to −0.01. Half a corpus's residual routinely sits on a handful of pages. |
| MICRO **and** MACRO with sign-disagreement flags | §8.119 — the two pick different rules and flip signs, and this campaign fitted the wrong one for weeks. |
| instrument audit that withholds "bankable" | refuses to certify a rule until quality, macro, split, work counter and clock are all present. Prints HYPOTHESES ONLY otherwise. |
| `gate_refit` (verification stage) | **this is Phase 0A** — "re-ask whether an already-SHIPPED threshold is still right by running the engine end-to-end." |

**Where it plugs in:**

- **Phase 0A** — `gate_refit` is the right shape for re-validating §8.156/§8.157/§8.160 against the true reading-order metric.
- **Phase 1C** — `gate_calculator` is where every new lever is tried BEFORE any of it reaches `suppress.rs` as a transcribed branch (depth ≤ 4, one doc-comment per feature, one test per branch).

**The harvest Carmenta must emit** (per page, at decision time):

```
gain          signed effect on the OFFICIAL metric if the gate fires on this unit
page          the macro denominator (alias of `clip`)
page_chars    reference mass, for macro_gain (alias of `clip_total`)
split         train (the 80) / holdout — anything else counts as holdout
shipped       does the currently shipped gate already route this unit?
work          deterministic work saved (positive = cheaper)
cpu_ms        pinned-CPU delta, confirmatory only
<features>    every other numeric column, harvested AT DECISION TIME
```

### Two honest limits, recorded so we neither fake a number nor dismiss the audit

1. **Most Carmenta levers have NO speed component by construction.** A
   reordering gate permutes lines the engine already recognized — the arm costs
   the same either way, so `work = 0` is the truthful value, not a missing
   measurement. The audit will print HYPOTHESES ONLY for these, and for a
   quality-only lever that verdict should be read as *"this gate has no speed
   half"* rather than *"the harvest is incomplete."* Suppression gates DO have a
   real counter (lines dropped) and must carry it. **Do not invent a `work`
   column to silence the audit.**
2. **We took a mid-flight snapshot.** `refit.rs` and `bin/gate_refit.rs` were
   written 2026-08-08T17:50 against a `Cargo.toml` that does not yet declare the
   `bd` feature they are gated behind, so only
   `--no-default-features --bin gate_calculator` builds. The discovery half is
   complete and verified (`--demo` runs clean). Re-sync from the source repo
   before relying on refit, and re-record the source commit when you do.

### Reconciliation with §3

The Great Gate's seven architecture laws and §3's fourteen instrument rules
overlap but are not duplicates — **§3 governs whether a MEASUREMENT is valid;
the seven laws govern whether a GATE is well-formed.** Both bind. The two that
§3 does not otherwise state, and which this campaign should adopt:

- **Population-relative normalization** — thresholds as percentiles of *this
  page's own* signal distribution, never absolute values. §8.171 proved the
  cost of the alternative: `VERIFY_LOWCONF = 0.96` was bound to CRNN's
  confidence distribution and silently abstained on 66 pages instead of 19 when
  a different recognizer arrived. **Choose the normalization for TRANSFER, not
  for in-corpus fit.**
- **Abstention beats fit** — prefer the gate that refuses when unsure over the
  best-fitted one; the abstaining gate is the one whose train and holdout
  numbers converge. §8.160's competence abstain is exactly this and it is the
  single largest ordering win we have (+0.987 pp).

---

## 11. Deploying the Great Gate — the work items, in order

§10 says what the tool is. This says what we BUILD to use it, and when. The
governing economics: engine runs are expensive (236 pages x ~9 s/page, plus the
official harness on top), rule search is free. **So harvest ONCE with a WIDE
feature set and search many rules offline** — the CSV is schema-agnostic and
discovers features from its header, which is exactly what makes that possible.
Harvesting per-lever would spend the expensive resource on the cheap problem.

### W1 — the feature tap (engine, env-gated, inert by default)

`FFAI_GATE_HARVEST=<path>` makes the engine append one row per page carrying
every decision-time signal it already computes, plus the ones we know we want:

```
page, n_col, cover, aspect, body_frac, n_lines, n_all,
sparse_scatter,                 §8.156's gate signal
join_margin, numseq_margin,     §8.160's text-verifier margins
conf_median, conf_frac_low,     the abstain's inputs, both engines' scales
mean_line_h, page_w, page_h, ink_extent, col_gap_max, ...
```

Rules that bind it: **harvested AT DECISION TIME** (a tap after the action
measures the action), and **inert when the env var is unset** — no allocation,
no branch cost in the shipped path. One test asserting the tap changes no
output.

### W2 — the harvest builder (`.tools-bench/gg_harvest.py`)

`gain` must now be the signed effect on the **OFFICIAL** metric, which our old
optimizer could not produce. The pipeline:

1. run the engine with the lever's arm **OFF** -> predictions -> official
   harness -> per-page `text_block` and `reading_order` edit;
2. run it **ON** -> same -> same;
3. `gain = off - on` per page (positive = the arm helps), for BOTH metrics;
4. join with W1's features on `page`;
5. emit the CSV with `page`, `page_chars`, `split`, `shipped`, `work`,
   `cpu_ms`, and every feature column.

Two arms, one binary, ABBA-interleaved, freshness-asserted — §3 applies to the
harvest exactly as it applies to a result. **`gain` is computed from a paired
per-page difference, so the ordering bias that voided §8.119–§8.171 cannot
recur here: both arms are the same engine on the same pages, differenced.**

### W3 — Phase 0A, run through the gate

Re-validating §8.156/§8.157/§8.160 needs an ON/OFF pair per mechanism, which is
exactly what W2 produces. Prefer `gate_refit` once the source repo's `bd`
feature is declared and we have re-synced; until then W2 + `gate_calculator`
answers the same question from the same data. **Do not block Phase 0A on the
refit binary** — the harvest is the evidence, refit is a convenience over it.

### W4 — Phase 1C, every new lever

No lever reaches `suppress.rs` without first surviving `gate_calculator` on the
W2 harvest: both splits agreeing in SIGN, `top3` well under 100 %, MACRO and
MICRO not disagreeing in sign, and — where the lever has a speed half — the
work counter present. Then transcribed as a branch of depth <= 4, one
doc-comment per feature, one test per branch, threshold in
**population-relative** form.

### Sequencing

| # | item | depends on | cost |
|---|---|---|---|
| W1 | feature tap | — | small, inert |
| W2 | harvest builder | W1 | one engine pass per arm |
| W3 | Phase 0A verdict on the shipped +1.78 pp | W2 | 2 arms x 3 mechanisms |
| W4 | Phase 1C lever search | W2 | free per rule |

**W1 and W2 are on the critical path for everything else in this plan** —
Phase 0A, Phase 1C, and any future gate all consume the same harvest. Build
them while the full 1651-page run finishes; neither needs its output.

### The one thing that would make this go wrong

Harvesting a NARROW feature set. Every feature omitted at harvest time costs a
full engine pass to add later, and the tool cannot search for a signal that is
not in the CSV. **When in doubt, emit the column.** Storage is free; the
expensive resource is the engine pass, and a wide harvest is the same pass.

---

## 12. Phase 1A RESULT — the full benchmark, decomposed

Official evaluator, all 1651 pages, shipped engine, no Carmenta changes.
Population verified: 1557 pages carry a `text_block` region and 1638 a reading
order; the 94 and 13 absent have no such region in the GT, so nothing was
silently dropped.

| | ours (full) | published range |
|---|---:|---|
| Text^Edit | **0.1636** | 0.0326 (PaddleOCR-VL-1.6) – 0.157 (Marker) |
| ReadOrder^Edit | **0.3226** | 0.116 – 0.243 (Marker) |

**On the full benchmark we are last on both columns.** That is the honest
leaderboard-comparable figure and it goes in the record first.

**Scope carries two thirds of it:**

| | n | mean | share of total error |
|---|---:|---:|---:|
| OUT OF SCOPE (non-English) | 836 | 0.2052 | **67.3 %** |
| ADDRESSABLE (English) | 721 | **0.1155** | 32.7 % |

English-only: **Text 0.1155**, **ReadOrder 0.2839**. Both numbers are true —
0.1636 describes the SCOPE, 0.1155 describes the ENGINE, and any public claim
must say which it is quoting.

### The finding that inverts a campaign assumption

| reading_order by layout | n | mean | pp_of_total |
|---|---:|---:|---:|
| **single_column** | 881 | **0.3404** | **0.1831 (56.8 %)** |
| other_layout | 367 | 0.3264 | 0.0731 |
| double_column | 182 | 0.3118 | 0.0346 |
| 1andmore_column | 155 | 0.2793 | 0.0264 |
| **three_column** | 53 | **0.1626** | 0.0053 |

**Three-column pages are our BEST layout for reading order; single-column is
our WORST — by mean, not merely by volume.** Roughly fifty sections and four
shipped mechanisms targeted dense 3+ column pages, and that work shows: 0.1626
against a corpus mean of 0.3226. Meanwhile single-column pages, where reading
order should be trivially top-to-bottom, carry **56.8 % of all reading-order
error**.

This was invisible under the old scorer, which conflated text and order into
one number, and it re-points the campaign: **the remaining reading-order work
is not in dense columns. It is in ordinary single-column pages**, and a page
whose order should be trivial reading 0.34 says something is wrong upstream of
the ordering machinery — suppression dropping content, or headers, footers and
captions interleaving into the body sequence. Investigate before building.

### The addressable work queue, by contribution

| doc type | text share | order share |
|---|---:|---:|
| **academic_literature** (185 EN pages) | **26.2 %** | **36.3 %** |
| PPT2PDF | 20.4 % | 10.0 % |
| book | 16.3 % | 19.2 % |
| exam_paper | 8.3 % | 14.4 % |
| newspaper | 12.9 % | 7.3 % |
| colorful_textbook | 9.5 % | 6.4 % |
| magazine | 6.5 % | 6.0 % |

`academic_literature` leads BOTH axes. It is the first target.

---

## 13. §12 CORRECTED — it is floats, not single-column, and the gate excludes them

§12 read "single_column is our worst layout for reading order (0.3404)" off the
raw per-layout means. Decomposing further shows part of that was a DEGENERATE
CASE of the metric, and the real driver is something else entirely.

**The first hypothesis died on inspection.** The metric's own pairing filters
out `header`, `footer`, `page_number`, `figure_caption`, `figure_footnote`,
`table_caption`, `table_footnote` and `page_footnote` before comparing order
(`end2end_dataset.py` ~2258). `FFAI_BODY_ONLY` dropping those therefore CANNOT
cost us reading order — our scope choice is ALIGNED with the metric, not
fighting it. That was the obvious explanation and it was wrong.

**English pages, order error decomposed:**

| segment | n | mean | share of total |
|---|---:|---:|---:|
| DEGENERATE — <= 1 orderable body region | 48 | 0.7135 | 16.1 % |
| clean — 2+ regions, no float | 185 | 0.1838 | 16.0 % |
| **has float — figure / table / equation** | **517** | **0.2798** | **67.9 %** |

**1. Sixteen percent is the evaluator's degenerate case, not our defect.** A page
with one orderable region has no sequence to get wrong, yet these score 0.71.
`page-035cb436` reads **text 0.000 and order 1.000** — read perfectly, scored as
entirely misordered. EXCLUDE these from any optimisation target; a lever fitted
to them is fitted to an artifact.

**2. Sixty-eight percent is floats.** Pages carrying a figure, table or isolated
equation read 0.2798 against 0.1838 for clean pages, and they are two thirds of
the error mass. Text flowing around and between floats is the real defect.

**3. The gate excludes exactly that population.** `probe_apply` guards on
`n_col >= 3 && cover >= 0.18`, so the reordering machinery NEVER RUNS on one-
and two-column pages. The gating variable is COLUMN COUNT; the thing that
actually predicts damage is FLOAT INTERRUPTION. That is a Great Gate variable
problem in the precise sense §10 describes — the gate is fitted on the wrong
axis for this population — and NOT a detection or box-sizing problem.

**Per-layout, with degenerates removed:**

| layout | all | non-degenerate | n |
|---|---:|---:|---:|
| double_column | 0.3151 | **0.3085** | 104 |
| single_column | 0.3181 | 0.2635 | 297 |
| 1andmore_column | 0.2674 | 0.2634 | 100 |
| other_layout | 0.2412 | 0.2278 | 153 |
| **three_column** | **0.1480** | **0.1480** | 48 |

**DOUBLE-column is our worst, not single.** And **three_column is our best by a
wide margin** — the dense-column campaign delivered on exactly the population it
was aimed at. §12's framing was partly the degenerate case talking; this table
supersedes it.

### What this makes the next lever

Not more ordering machinery for dense columns — that population is solved.
**Widen the gate's axis:** the candidate is a float-interruption signal (a wide
horizontal band with no text, or a text block that fails to span the expected
measure) admitting 1- and 2-column pages that currently return early. That is a
W2 harvest away: the features are decision-time and cheap, and the gate
calculator can search them offline against `gain_order` before any of it
reaches `suppress.rs`.

**Harvest note:** exclude the 48 degenerate pages, or `top3` and the macro
column will both be dominated by an artifact that no rule can fix.

---

## 14. §13's lever REFUTED — the reorder is invisible to the metric, and why

The gate-axis hypothesis was: extend the reordering machinery to the 1- and
2-column pages the `n_col >= 3` guard excludes, where 67.9 % of English
reading-order error lives. `FFAI_ORDER_PROBE_ALL=1` measured the ORACLE gain of
doing exactly that, over 755 English pages, scored by the official evaluator.

**Four probes, varied, all agreeing — the refutation bar (§3 rule 11):**

| # | probe | result |
|---|---|---|
| 1 | oracle gain, 750 pages | mean **-0.0007**; 18 helped, 23 hurt, **709 unchanged** |
| 2 | did the probe actually move anything? | different order on **361/730** pages, median `frac_moved` 0.455 |
| 3 | of those 361 moved pages | **321 were score NO-OPS** |
| 4 | exhaustive search, 39 features, both splits | best rule **+0.001 MACRO**, `top3` 95-128 % |

Probe 4 is the level-above check: it is not that we failed to find the
threshold, it is that no combination of 39 decision-time signals separates a
population worth routing. `top3` above 100 % means the crumbs are carried by
three pages and the rest are negative — a clip list, not a rule.

### The mechanism, which is the valuable part

The probe reorders nearly half the lines on half the pages and the metric does
not notice. Splitting the moved pages by whether their score responded:

| | GT body regions | our lines | **lines per region** |
|---|---:|---:|---:|
| score CHANGED (n=40) | 12 | 28 | **4.9** |
| score NO-OP (n=321) | 9 | 67 | **27.4** |

**Our line-level output is far finer than the granularity the metric scores.**
OmniDocBench compares the order of MATCHED REGIONS; MGAM merges our many lines
into one GT paragraph, and permuting lines INSIDE a merged region is invisible.
The score only responds when our lines sit near 1:1 with regions.

So the reordering machinery is operating below the level the metric measures.
That is not a gate problem and never was — no threshold on any axis can fix a
granularity mismatch.

### What this refutes, precisely, and what it does not

REFUTED: *widening the gate's axis so the existing line-level probe runs on
1- and 2-column pages.* Measured, mechanism identified, four probes.

NOT refuted: *that reading order on float-interrupted pages is recoverable.* We
showed the EXISTING machinery cannot recover it, not that nothing can. The
remaining 68 % of error is real and still on the board.

### The lever this points at instead

**BLOCK-LEVEL GROUPING.** Group lines into paragraph-scale blocks first, then
order the BLOCKS — operating at the granularity the metric actually scores.
§8.160 already named this as the next candidate ("within-column continuation of
the join score at the BLOCK level, or upstream line grouping"), from the
engine's side; this is the same conclusion arriving from the METRIC's side,
which is the stronger form of the argument.

Note that block grouping is also what would make our output match the GT's
structure generally — so it is likely to move `text_block` as well, not only
reading order.

### Harvest hygiene, recorded

`gain_text` shipped as a column in the first cut and the calculator picked it up
as a candidate PREDICATE — an outcome predicting an outcome, the §8.161
contamination, in a file written by the tool built to avoid it. Removed. The
rule stands: **the calculator treats every numeric non-required column as a
feature, so an outcome must never be in the same file.**

---

## 15. CORRECTION — we are NOT English-only, and 894 pages are back in scope

§4, §12 and §13 all partition the corpus into "ADDRESSABLE (English)" and "OUT
OF SCOPE (non-English), because our recognizers are English-only by deliberate
design". **That premise is false.** The shipped recognizer's charset:

| block | entries |
|---|---:|
| CJK unified ideographs | **15 565** |
| CJK punctuation / other | 2 530 |
| ASCII | 95 |
| katakana | 94 |
| hiragana | 86 |
| cyrillic | 11 |

**SVTR is PP-OCRv5 — a Chinese/Japanese/Latin model.** It is not an English
recogniser with a big table; CJK is 96 % of its head.

**And we read Chinese well.** Sampled against GT, one page returns
`2．学会“茬、砖、涎”3个会认字。` character-for-character. Simplified Chinese
scores **0.2038** against English's 0.1155 — 1.8x worse, not the ~1.0 a model
that could not read the script would produce. The errors on those pages are
ORDERING and OVER-EMISSION (we emit a `WWW.1PPT.COM` watermark the GT excludes),
exactly the failure modes we have on English pages.

**WITHDRAWN:** every statement that non-English pages are out of scope, and the
attribution of 67.3 % of text error to a scope decision. Those 894 pages are
addressable and their error is ours.

### The mechanism, and it is in our source not the model

`join_fluency` — the text verifier behind §8.160's +0.987 pp, our largest
ordering win — is **structurally inert on CJK**:

```rust
const TERM: &[char] = &['.', '!', '?', ':', ';', ...];   // Latin punctuation only
...
if first.is_lowercase() { s += 1.0 } else if first.is_uppercase() { s -= 1.0 }
```

Chinese uses `。！？：；`, which never match `TERM`; and CJK characters have NO
CASE, so `is_lowercase()` and `is_uppercase()` are both false and NEITHER branch
fires. The function returns ~0 for EVERY candidate ordering on a Chinese page.
**Our best ordering mechanism cannot distinguish a good order from a bad one on
54 % of the benchmark.** §8.160's `verifier_blind` abstain was a symptom of this
that we diagnosed as a confidence problem.

The same Latin assumption runs through the suppression branches
(`bibliography_branch`, `after_refs_branch`, `year_paren_syntax`) and through
`num_seq_monotone`, which does not know 一二三.

### On building a language detector — we need detection, not a detector

- **Not for model routing.** SVTR already IS the Chinese model; there is nothing
  better to route to. A pre-pass would spend compute choosing between models we
  do not have.
- **Script comes FREE from the output.** After one recognition pass, counting
  Unicode blocks in the recognised text classifies the page with near-certainty
  — no second pass, no model, no image-feature threshold. That is the signal
  script-aware post-processing needs.
- **A pre-pass is only required for scripts SVTR CANNOT read** (Arabic,
  Devanagari, Cyrillic), where output-based detection fails by construction: the
  model will not emit Arabic, it will emit confident garbage in a script it
  knows. Worth building for graceful refusal as a product feature; NOT a
  benchmark lever — OmniDocBench carries only English, Chinese, mixed and 2
  "other" pages.

### The lever this opens

**Script-aware post-processing**: a CJK branch for `join_fluency` (CJK
terminators, no case test, no hyphenation), and script guards on the Latin-only
suppression branches. Unlike §13's gate-axis hypothesis, this has a mechanism
visible in the source rather than a correlation — and it addresses a population
carrying two thirds of our text error that we had written off.

---

## 16. Phase 0B RESULT — body-only suppression is HURTING us

`FFAI_BODY_ONLY` off vs the shipped baseline, 755 English pages, official
evaluator, one engine pass against the existing baseline:

| metric | shipped | body-only OFF | gain | 95 % CI | |
|---|---:|---:|---:|---|---|
| text_block | 0.1155 | **0.1066** | +0.0089 | [+0.0004, +0.0173] | excludes 0 |
| reading_order | 0.2839 | **0.2257** | **+0.0582** | [+0.0469, +0.0703] | excludes 0 |

**Suppression is a net HARM on both published metrics**, and reading order
improves by 20 % relative. This is the largest lever the campaign has found on
the official instrument.

**It confirms the §13 correlation was causal.** `body_frac` (the fraction of
detected lines kept) tracked reading-order error monotonically — 0.144 at 0.95+
against 0.404 below 0.5 — and survived controls for floats, confidence and line
count. The A/B says that relationship was not pages-that-need-suppression-are-
hard; it was suppression doing the damage.

**And it is the THIRD mechanism fitted to the discredited scorer.** §8.119
measured body-only as a large win (30.12 -> 25.85 macro) on our own
concatenate-then-edit-distance metric, which charged the extra headers, footers
and captions as INSERTIONS. The official metric matches regions first, so that
material costs nothing and gives the matcher more to align against. After the
competitive standings (§8.173) and the ordering verifier (§15), this is the
third time a shipped behaviour turns out to have been optimising the instrument
rather than the output.

### Stated honestly

- The **text** gain is MARGINAL: CI lower bound +0.0004 is one page of noise
  from spanning zero. The reading-order gain is the solid one.
- **English only** (721/750 pages). §15 showed our post-processing is
  Latin-centric, so transfer to the 894 non-English pages must be MEASURED, not
  assumed.
- Body-only exists for a real reason — a document-to-text user usually does not
  want page furniture. The right end state is probably a SCOPE FLAG that is off
  for benchmark parity and available for users who want it, not deletion.

### Interaction to watch

The §15 CJK pricing run is measuring its ceiling with `FFAI_BODY_ONLY=1`, i.e.
under a configuration this result says we should abandon. The ordering ceiling
is broadly independent of scope, but any number taken from that run carries the
caveat until re-measured against the new baseline.

---

## 17. §15's CJK fix PRICED — the mechanism is real, the prize is not

The lever: `join_fluency` is measurably blind on CJK (§15, pinned by a unit test
with a Latin control), so the §8.160 verifier cannot rank two orders on 54 % of
the benchmark. The fix is built and tested (`caec493`). Before shipping it, the
ceiling — what a PERFECT gate on this lever could ever be worth.

`FFAI_ORDER_PROBE_ALL=1` with `FFAI_CJK_FLUENCY=0` (old behaviour pinned so the
measurement contains the ceiling and NOT the fix), 896 non-English pages:

| | shipped | probe-all | CEILING (perfect gate) | blind application |
|---|---:|---:|---:|---:|
| text_block | 0.2052 | 0.2100 | **+0.0028** | -0.0048, CI [-0.0094, -0.0009] |
| reading_order | 0.3553 | 0.3639 | **+0.0077** | -0.0087, CI [-0.0150, -0.0025] |

reading_order: helped 30 pages, hurt 55.

**The ceiling is +0.0077.** Body-only OFF (§16) buys **+0.0582** on the same
metric. The CJK ordering lever is worth **7.5x less at its theoretical maximum**
than a lever we already have in hand — and the maximum is unachievable, since it
assumes a gate that is never wrong.

**This is the check §14 was refuted for skipping.** The mechanism was real,
measured, and correctly fixed; the prize was never sized first. Fifteen minutes
of ceiling probe against an hour of building. Price BEFORE building, not after.

### What is refuted, and what is not

REFUTED: *CJK-aware `join_fluency` as a QUALITY lever worth shipping on its
merits.* Its ceiling is negligible against the alternatives on the board.

NOT REFUTED: *that CJK reading order is improvable.* At 0.3553 it is our worst
population by a wide margin. What the ceiling says is that the PROBE'S
ALTERNATIVE ORDER is rarely better than what we already emit on those pages —
the same shape as §14. A different ordering approach is untested, not refuted.

### Disposition of `caec493`

The code is correct, tested, and fixes a proven defect. But "revert if unproven"
binds: a change whose ceiling is +0.0077 cannot produce a CI excluding zero on
888 pages, so it cannot be shipped as a measured win.

**PRE-REGISTERED before the A/B ran:** the arm will land inside noise and its CI
will SPAN ZERO. If it excludes zero in either direction, my model of this lever
is wrong and that is the more interesting result.

The intended disposition is to keep the CJK arm behind its toggle, DEFAULT OFF,
with this ceiling recorded as the reason — so the mechanism is available if a
future lever makes CJK ordering matter, and no unproven behaviour ships.

---

## 18. §17's disposition REVERSED — the CJK arm ships, and why the pricing missed it

**Pre-registered in §17:** "the arm will land inside noise and its CI will SPAN
ZERO. If it excludes zero in either direction, my model of this lever is wrong
and that is the more interesting result."

It excluded zero. The model was wrong.

| metric | shipped | CJK arm | gain | 95 % CI | |
|---|---:|---:|---:|---|---|
| **text_block** | 0.2052 | **0.1979** | **+0.0073** | [+0.0035, +0.0116] | **excludes 0** |
| reading_order | 0.3553 | 0.3520 | +0.0033 | [-0.0002, +0.0068] | spans 0 |

836 non-English pages, helped 23 / hurt 5 on text, 22 / 5 on order.

### Why §17's ceiling missed it

§17 priced the ceiling on **reading_order**, because that is the metric the
mechanism targets, measured +0.0077, and recommended not shipping. The payoff
arrived on **text_block** instead.

**The metrics are COUPLED through the matcher.** MGAM aligns our output to GT
regions before scoring text; a better line order produces better region
alignment, so more of our text matches the right region and the TEXT edit
distance falls. Fixing reading order improves the text score without the
reading-order score moving much.

**A ceiling probe aimed at the wrong column is barely better than none.** §14
refuted a lever for skipping the ceiling; §17 ran one and still nearly killed a
real win by pricing the metric the mechanism TOUCHES rather than the metric the
campaign OPTIMISES. Price every column the change can reach, not the obvious one.

### The control that made this bankable

v1 compared the new binary against `odb_pred_full`, built from older source —
two builds, the §8.53 violation, with the extra code merely ARGUED to be inert.
The `FFAI_CJK_FLUENCY=0` toggle existed precisely so that argument could be
replaced by a measurement:

| cjkoff vs baseline | delta | pages changed |
|---|---:|---:|
| text_block | **+0.0000** | 0 |
| reading_order | **+0.0000** | 0 |

Byte-identical. The W1 tap and the §13 probe-all arm are PROVEN inert when
unset, so the only active difference in v1 was the CJK arm and its result
stands. **This is what a positive control is for** — it converted a suspect
cross-binary comparison into a valid one without re-running it.

### Shipped

`caec493` is already default-on and stays. Per §2: text is the primary
objective and it gains with a CI excluding zero; reading order is the
non-regression gate and it improves.

**Benchmark impact, CJK arm alone:** 836 of 1557 scored pages are non-English,
so +0.0073 there is **+0.0039 on the full metric — 0.1636 -> 0.1597**.

Body-only OFF (§16) would add more, but it is measured on ENGLISH ONLY. If it
transferred it would give 0.1556; that number is **not banked** and must be
measured on non-English before anyone quotes it.

---

## 19. Item 1 BANKED — body-only OFF clears both halves; the benchmark config flips

`FFAI_BODY_ONLY` off on the 896 non-English pages, differenced against the
same-binary `cjkfluency` baseline so the delta contains only this lever:

| metric | shipped | body-only OFF | gain | 95 % CI | pages |
|---|---:|---:|---:|---|---|
| text_block | 0.1979 | **0.1492** | **+0.0486** | [+0.0389, +0.0584] excludes 0 | 505 helped / 86 hurt |
| reading_order | 0.3520 | **0.2424** | **+0.1096** | [+0.0968, +0.1227] excludes 0 | 388 helped / 31 hurt |

**5.5x the English text gain and 2x the English order gain.** The Latin-fitted
suppression heuristics misfire harder on CJK — §15's mechanism, showing up in
§16's lever exactly where it predicted.

**The mechanism has a name now (item 4's "worst text" mystery).** The §8.135
bibliography branches delete `reference` regions, and the official GT scores
`reference` as text (it is NOT in the metric's ignore list). Verified per page:
0.824 -> 0.005, 0.689 -> 0.020, 0.636 -> 0.001 with body-only off. Only 16
English pages carry references (2 % of pages, 2.2 % of chars) but they are
individually savaged — a concentrated slice of the win. One counter-example
recorded (0.549 -> 0.615): suppression is not uniformly harmful.

**Banked.** Both populations, both metrics, CIs excluding zero. Note the
engine's default was ALREADY off (§8.106 made it opt-in); it was the BENCHMARK
config that opted in, because the old biased scorer rewarded deletion by 4+ pp.
The harnesses now run body-only OFF as the shipped config; legacy arms pass
`--env FFAI_BODY_ONLY=1`. This is the FOURTH mechanism exposed as an artifact
of the §8.173 scorer, and the largest.

**Projected standing, both banked changes (CJK arm + body-only off):**

| | text_block | reading_order |
|---|---:|---:|
| session start | 0.1636 | 0.3226 |
| **projected now** | **~0.1295** | **~0.2348** |
| Marker (worst published row) | 0.157 | 0.243 |

Off the bottom of the published board on both columns. PROJECTION ONLY — the
arithmetic composite of separately measured arms. The merged full-population
run scores next and becomes the banked number and the new baseline for every
subsequent arm.

---

## 20. THE NEW BANKED STANDING — off the bottom of the board

Merged full-population run (755 EN from `nb_en` + 896 non-EN from
`bodyoff_zh`, coverage-asserted, control-attested), scored once by the
official evaluator. Config: CJK-aware `join_fluency` (§18) + body-only OFF
(§19).

| | session start | **BANKED** | Marker (worst published) |
|---|---:|---:|---:|
| text_block | 0.1636 | **0.1363** | 0.157 |
| reading_order | 0.3226 | **0.2381** | 0.243 |

**Ahead of Marker on both columns.** Two banked changes, both with CIs
excluding zero on their measured populations, took text down 17 % relative and
reading order 26 % relative in one day of work on the correct instrument.

The §19 projection (0.1295 / 0.2348) was optimistic by ~0.007 / ~0.003 —
arithmetic composites of arm means do not survive contact with the merged
official pass, which is why the plan banks only the scored number.

`odb_pred_newbase_quick_match` is now the baseline every subsequent arm
differences against.

---

## 21. Item 3 REFUTED — coarse granularity is irrecoverable, and line-level was already right

The blockgroup pricing arm (no engine changes: identical text, identical order,
lines grouped 3.1-per-block by vertical gap + column overlap, blocks as
markdown paragraphs), 755 English pages vs the new baseline:

| metric | base | blockgroup | gain | CI | pages |
|---|---:|---:|---:|---|---|
| text_block | 0.1073 | 0.2051 | **-0.0978** | [-0.1098, -0.0852] excludes 0 | 537 hurt / 77 helped |
| reading_order | 0.2257 | 0.2840 | **-0.0583** | [-0.0704, -0.0463] excludes 0 | 282 hurt / 73 helped |

Ceiling with a PERFECT per-page gate: +0.0120 / +0.0165 — not worth a gate
even if one existed.

**The mechanism is the metric's merge asymmetry.** MGAM searches segmentation
granularity on the PREDICTION side by MERGING: fine output is safe (our lines
merge up to GT regions), coarse output is irrecoverable (a block crossing a GT
boundary cannot be split back). So the optimal prediction granularity is AT OR
BELOW GT granularity, never above. §14's "permutations inside a merged region
are invisible" was the benign face of the same property; this is the hostile
face.

REFUTED: block-level grouping of our output, both as presentation and as the
in-engine stage §14 proposed — the pricing arm exists precisely so that stage
was never built. Line-level emission stands as the correct granularity.

Item 4's remaining real target (order on 3+ float academic pages, 0.4493)
therefore needs a lever that reorders LINES without regrouping them; the §14/
§8.160 "block grouping" hypothesis is closed.

---

## 22. Phase 0A COMPLETE — the ordering machinery is REAL

The last open question from §8.173: the +1.78 pp of ordering mechanisms were
fitted AND judged on the discredited scorer. Four other mechanisms from that
era fell as artifacts (standings, Latin verifier blindness, body-only, and the
old competitive frame). These three were the remaining suspects. Each arm
disables ONE mechanism against the new baseline (`nb_en`, body-only OFF, CJK
arm on), 721/750 English pages, official evaluator:

| mechanism disabled | text_block delta | reading_order delta | verdict |
|---|---:|---:|---|
| §8.160 verifier (`FFAI_ORDER_VERIFY=0`) | -0.0083 [-0.0153, -0.0018] | -0.0075 [-0.0140, -0.0015] | **CONFIRMED REAL — keep** |
| §8.157/160 probe (`FFAI_ORDER_PROBE=0`) | -0.0126 [-0.0187, -0.0071] | -0.0093 [-0.0154, -0.0036] | **CONFIRMED REAL — keep** |
| §8.156 sparse gate (`FFAI_ORDER_GATE=0`) | -0.0006 [-0.0036, +0.0028] | -0.0041 [-0.0096, +0.0008] | inconclusive — stays, no evidence either way |

(Deltas are the COST of disabling: negative = the mechanism helps.)

**The ordering machinery survives the instrument it was never fitted to.**
The verifier and the probe are confirmed on both official metrics with CIs
excluding zero; the sparse gate is small and unproven in both directions
(point estimates lean keep: 13 hurt vs 3 helped on order). The fifty sections
of ordering work were real engineering — it was the scoreboard around them
that was broken, which is §8.173's conclusion completing its own audit.

Notable: the probe's contribution on the TRUE metric (+0.0126 text) is larger
than the verifier's, the reverse of their old-scorer ranking — one more case
of the §18 lesson that the metrics are coupled through the matcher and levers
land on columns they do not aim at.

---

## 23. Item 5 RESULT — the Text OCR task isolates our real defect: line order in PLAIN TEXT

7 019 block-level `text_block` crops from all 755 English pages, ground-truth
regions handed to the engine, scored by their `cal_metric.py` formula
(disclosed: the v1.6 repo does not ship this task's pipeline).

**Headline: `mobiledet-svtr` reads 0.4133 sample-avg — and that number is NOT
a recognition score.** The same engine scores 0.1073 end-to-end on the same
pages where it must also detect. An engine cannot be four times worse when
handed the regions. The instrument was asking for help, and the samples answer:

```
GT  : When an attempt is made to form the product BA, we discover that the...
PRED: BA, we discover made to form the product dimensions When ana attempt...
```

**Every word present, order scrambled.** 2 279 long crops show it. On a
paragraph crop there are no floats, no columns, no furniture — and the emitted
line order is still wrong. This is §12's single-column anomaly reproduced in a
minimal test case: the residual reading-order defect is not about layout
complexity at all. It is in how detected boxes are sequenced within PLAIN
TEXT, and it only escapes end-to-end notice because MGAM's merging forgives
within-region permutation (§14) — the crop task scores the string directly and
exposes it.

Secondary: crops with GT <= 10 chars read 0.915 (n=206) — detection fails on
tiny context-free images. Real, small, separate.

**What this opens: the next campaign's target, with 2 279 ready-made repro
cases** — each a small image where recognition is proven right and ordering
provably wrong. No harvest needed; the failure set exists on disk.

**What this blocks: any public Text OCR claim from these numbers.** 0.4133
must not be quoted against PaddleOCR's 0.071 — it measures our box
sequencing, not our recognizer. The craft-crnn pass over the SAME crops is
still valid as a PAIRED recognizer comparison (same detector, same ordering,
differenced), running now.

---

## 24. CORRECTION to §23's pairing claim — and the finding it uncovered

§23 said the craft-crnn crop pass "stays valid as a PAIRED recognizer
comparison (same detector, same ordering, differenced)". **That was wrong.**
`craft-crnn` is CRAFT detection + CRNN; `mobiledet-svtr` is DBNet + SVTR. The
pair changes DETECTOR and RECOGNIZER together — it isolates nothing. The
one-variable pair on crops is `mobiledet-crnn` vs `mobiledet-svtr`; a targeted
sample is running.

**The confound turned out to be the diagnostic.** On the same 7 019 crops:

| engine | sample_avg | page_avg |
|---|---:|---:|
| **craft-crnn** (CRAFT + CRNN) | **0.1051** | 0.1235 |
| mobiledet-svtr (DBNet + SVTR) | 0.4133 | 0.3399 |

CRAFT+CRNN reads the crops nearly clean — NO scrambling. The §23 defect is
therefore SPECIFIC TO THE MOBILEDET/DBNET PATH: its line formation on cropped
paragraphs produces fragments the orderer then scrambles. It is not a
universal boxes.rs flaw, which reshapes step 2: the fix candidates are (a) a
raster candidate that orders fragments correctly, and/or (b) the mobiledet
line-grouping itself — step 1b's ceiling decides which, exactly the branch it
was pre-registered to decide.

**And the accidental headline: our EasyOCR-stack port reads 0.1051 where
published EasyOCR reads 0.26 EN** — 2.5x better on the benchmark's own Text
OCR task. Caveats attach (their row predates the v1.5/v1.6 annotation
corrections; our scoring reproduces their formula because v1.6 does not ship
the task pipeline) — but as a like-for-like model-stack claim it is the
cleanest one this project has: same models, our engineering, their task.

---

## 25. Step 1 KILLS the ordering fix — the defect is FRAGMENTATION, not sequence

The ceiling probe (7 019 crops, engine's own lines reordered offline, no engine
changes) against the rules pre-registered in §23 step 1:

| segment | ENGINE | RASTER | YSORT | ORACLE |
|---|---:|---:|---:|---:|
| ALL (n=6875) | 0.4010 | 0.3818 | 0.4420 | 0.2971 |
| BAD (n=2911) | 0.6645 | 0.5343 | 0.6364 | 0.4880 |
| GOOD (n=1448) | 0.0169 | **0.1668** | 0.1097 | 0.0166 |

**Both pre-registered rules fail.** RASTER regresses the good crops 14 better
against 643 worse (0.0169 -> 0.1668) and closes only 570 of 2 911 bad crops
under 0.2 — it is not best-somewhere-and-harmless-elsewhere, which is the
boxes.rs pool's own admission rule. And **2 185 of the 2 911 bad crops keep
ORACLE >= 0.3**: no permutation of the engine's own lines recovers them, so the
defect is upstream of ordering exactly as the kill branch anticipated.

**The mechanism, measured:** median GT 130 chars, median 19 DETECTED LINES —
about 7 characters per line. DBNet shatters a paragraph into ~19 fragments;
59 % of bad crops still carry >= 80 % of the text. The words survive and the
JOINS do not: every fragment boundary injects a spurious space or line break,
which no reordering can undo. §23 read "scrambled" off a few samples; it is
OVER-FRAGMENTATION, and the two have opposite fixes.

**Refuted:** a raster candidate in the ordering pool, and with it §23's whole
"fix our ordering" framing for this defect. Cost: one analysis script. The
engine campaign it prevented would have been days.

**Still open, and the right question before anything is built:** does this
fragmentation hurt SHIPPED FULL PAGES, or is it an artifact of the crop
condition (tight bbox, no margin, small input — all scale inputs DBNet is
sensitive to)? Our end-to-end English text is 0.1073, which is not what a
badly fragmenting detector produces, and §14 measured 27.4 lines per GT region
on full pages where MGAM's merging absorbs it. **Price that before treating
0.4133 as a shipping defect** — it may be a valid measurement of an invalid
condition.

---

## 26. Fragmentation PRICED — crop-condition only; and the Text OCR claim banked

**#1 — is §25's fragmentation a shipping defect?** No. Measured on data already
on disk, no engine run:

| | chars per detected line |
|---|---:|
| crops | median **5.0** |
| full pages | median **28.0** |
| GT `text_block` regions | median 183 chars |

**5.6x.** On full pages DBNet forms proper text lines; on tight margin-free
crops it shatters into ~5-character fragments. The defect is CROP-CONDITION
SPECIFIC — DBNet's scale sensitivity on small inputs — and our shipped page
pipeline is unaffected (0.1073 end-to-end English text is not what a
fragmenting detector produces). **`0.4133` is a valid measurement of a
condition we do not ship. Walked away from.**

This also retires §23's "next campaign" framing entirely: there is no
plain-text ordering defect, there was never a raster fix worth building, and
the 2 279 "repro cases" are reproductions of an artificial condition. Three
sections of hypothesis closed by two analysis scripts and zero engine changes.

**#2 — the claim, banked with its kit** (`docs/textocr-claim.md`):

> On OmniDocBench v1.6's Text OCR task, English pages, Carmenta's `craft-crnn`
> reads **0.1051** normalized edit distance (95 % CI [0.1013, 0.1091], 7 019
> regions / 755 pages) against **published EasyOCR's 0.26**.

`craft-crnn` IS the EasyOCR model stack (CRAFT + `english_g2` CRNN) in pure
Rust on candle — same models, same task, different engineering. That places it
between Tesseract (0.096) and OpenOCR (0.070) on their published column.

Four disclosures travel with it, all in the document: we reproduced the metric
(v1.6 ships no pipeline for this task); their rows predate the v1.5/v1.6
annotation corrections so it is directional; **our document default
(`mobiledet-svtr`) scores worse on this task, 0.4133, for the crop-condition
reason above** — we report `craft-crnn` because it is the like-for-like
comparison and we say plainly that it is not our default; and the two
environment deviations (`lxml>=5.2`, `PYTHONUTF8=1`), neither touching scoring.
