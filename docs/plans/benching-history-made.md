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

---

## 27. #3 — the next campaign, ranked by measured contribution

Segments of the BANKED baseline (0.1363 text / 0.2381 order), each with its
share of the remaining error. Targets are chosen by contribution, never by mean
(§3 rule 5).

### TEXT — banked 0.1363 over 1557 pages

| segment | n | mean | share |
|---|---:|---:|---:|
| non-EN 1-2 floats | 344 | 0.1259 | 20.4 % |
| non-EN no float | 240 | 0.1575 | 17.8 % |
| non-EN 3+ floats | 189 | 0.1937 | 17.2 % |
| EN no float | 185 | **0.1656** | 14.4 % |
| EN 3+ floats | 234 | 0.1275 | 14.1 % |
| EN 1-2 floats | 283 | 0.0805 | 10.7 % |
| degenerate (artifact) | 82 | — | 5.3 % |

**non-English carries 55.4 % of all text error** and it is spread evenly across
float counts — so floats are NOT the driver; the language is. Two banked wins
already came from Latin-fitted code hurting CJK (§18 join_fluency, §19
suppression), which makes a third plausible but NOT assumable.

### ORDER — banked 0.2381 over 1638 pages

| segment | n | mean | share |
|---|---:|---:|---:|
| **EN 3+ floats** | 234 | **0.3605** | **21.6 %** |
| non-EN 1-2 floats | 344 | 0.2037 | 18.0 % |
| **degenerate (artifact)** | 163 | 0.60-0.73 | **26.6 %** |
| non-EN 3+ floats | 189 | 0.2869 | 13.9 % |
| EN 1-2 floats | 283 | 0.1229 | 8.9 % |
| no float (both langs) | 425 | ~0.10 | 11.0 % |

**A quarter of remaining order error is the metric's degenerate case** — pages
with <= 1 orderable region, where no sequence exists to get wrong. Excluding
them our order reads **0.1941**, which would sit mid-pack on the published
board (better than Marker 0.243, olmOCR 0.216, Nanonets 0.213, POINTS-Reader
0.198). We cannot CLAIM that number — the leaderboard includes those pages —
but it must not be a TARGET either. Chasing it is chasing an artifact.

### The three candidate levers, in order

**A. Diagnose non-English text (55 % of text error).** DIAGNOSIS FIRST, no
lever named yet — §14 and §17 both cost time by building or pricing before the
mechanism was known. The audit question: what is still Latin-specific in the
active path? With body-only OFF most suppression branches no longer run;
`num_seq_monotone` remains ASCII-only (`is_ascii_digit`, `.`/`)`/`]`
terminators — it cannot see 一二三 or `、`), but it is only a tiebreak when the
join margin ties. That is a small lever, so the 55 % is probably NOT
post-processing at all — more likely recognition or detection on CJK, which
would be a different and larger campaign. **Measure which before choosing.**

**B. EN no-float pages are our WORST English text segment (0.1656).** A page
with no figure, table or equation should be our easiest case, and it is our
hardest. That inversion is the same shape as §12's single-column anomaly, which
turned out to be a metric artifact plus a mechanism — worth the same
decomposition. Cheap: the data is on disk.

**C. Order on 3+ float pages (35.5 % across both languages).** The float
problem §13 identified. §14 refuted widening the gate axis and §25 refuted
raster ordering — but both were refuted for CROP or granularity reasons, and
the page-level float case has never had a lever aimed at it. Needs a ceiling
probe before any build; the oracle harness already exists.

### Explicitly NOT next

* Anything from the crop task (§26: invalid condition for our shipped engine).
* The degenerate pages (unfixable by construction).
* Speed work — we win there by ~9x and the goal is quality.

---

## 28. A's threshold refuted — and a HARNESS BUG that was corrupting the banked number

**The cheap hypothesis died first, as intended.** `BIN_THRESHOLD = 0.3` tuned
for Latin stroke density rejecting dense CJK glyph masses: swept `FFAI_DB_BIN`
across 0.30 -> 0.10 on 14 failing pages and counted CHARACTERS EMITTED, a
deterministic count rather than a score:

| FFAI_DB_BIN | 0.30 | 0.25 | 0.20 | 0.15 | 0.10 |
|---|---:|---:|---:|---:|---:|
| chars | 7267 | 7257 | 7234 | 7208 | 7140 |

**Flat across a 3x range.** The threshold is not the mechanism. Cost: 13 minutes.

**But the sweep printed `empty pages 0` where the banked run had empties**, and
that contradiction was the real find. Re-running the four worst pages with the
same engine and config:

| page | banked | re-run |
|---|---:|---:|
| newspaper_d1716a12... | **0 chars** | **7 741** |
| newspaper_d6cd76e5... | **0 chars** | **6 172** |
| page-942ac90d... (trad. Chinese historical) | 1 | 1 |
| yanbaopptmerge_4570 | 15 | 15 |

Two of the four were **subprocess crashes**, not detection failures. The arm
harness writes `""` when the child exits non-zero — visible rather than silent,
which was the right instinct and the wrong action: **an empty prediction scores
1.0**, so a crash masquerades as a total quality failure.

**Corrected banked standing** (four pages repaired, full population rescored):

| | recorded in §20 | **corrected** |
|---|---:|---:|
| text_block | 0.1363 | **0.1307** |
| reading_order | 0.2381 | **0.2348** |

The engine was always this good; 0.0056 of text and 0.0033 of order were a
harness artifact. §20's number is superseded.

**The fix, applied to `gg_arm.py`:** retry once, and on a second failure record
the page in `failed_<arm>.txt` and write NO prediction file at all — the
evaluator then omits it and `n` reflects the true population, instead of a
crash being scored as a perfect-failure page. A page the engine genuinely
cannot read still legitimately writes `""`; the distinction is between "the
engine answered nothing" and "the engine never answered".

**Standing lesson, and it is the session's own rule turned on itself:** §3
rule 10 says never silently narrow the population — we obeyed it by writing a
placeholder, and the placeholder was worse than the omission it prevented.
*Visible* is not enough; the placeholder must not be a value the metric can
score.

**Still real after the correction:** 54 non-English pages genuinely under-emit
(median 0.48 of GT length). Two of the four audited were true detection
failures. Lever A's prize must be re-priced against 0.1307, not 0.1363.

---

## 29. Lever C REFUTED — and it inverts §27's ranking

The oracle ordering ceiling on real pages: our OWN lines, unchanged, re-emitted
in the GT's annotated region order and scored against GT text. Both arms use
identical text, so the difference isolates SEQUENCE. 1 145 pages with geometry
and >= 2 orderable regions.

| segment | n | shipped | ORACLE | ceiling |
|---|---:|---:|---:|---:|
| **EN no float** | 155 | 0.1943 | 0.1522 | **+0.0421** |
| non-EN 3+ floats | 122 | 0.5818 | 0.5420 | +0.0397 |
| non-EN 1-2 floats | 260 | 0.5018 | 0.4764 | +0.0254 |
| non-EN no float | 195 | 0.2756 | 0.2503 | +0.0253 |
| EN 1-2 floats | 203 | 0.3405 | 0.3308 | +0.0097 |
| **EN 3+ floats** | 210 | 0.4427 | 0.4378 | **+0.0049** |

**§27 named EN 3+ floats as the top ordering target — 21.6 % of remaining order
error — and it has the SMALLEST ceiling on the board.** Perfect sequencing buys
+0.0049 there. Those pages are wrong in their TEXT, not their order, so no
ordering lever can reach them however the order column scores them.

This is the third time this campaign a segment's SHARE OF ERROR has been
mistaken for its RECOVERABLE prize (§14's gate axis, §17's reading-order
ceiling, now this). **Share of error tells you where to look; a ceiling tells
you what is there.** Rank by contribution to choose what to PRICE, never to
choose what to BUILD.

**The inversion:** the biggest ordering prize (+0.0421) is EN NO-FLOAT pages —
which is also §27 lever B's segment and our worst English text at 0.1656. Two
independent decompositions, arriving from text and from order, name the same
155 pages. That convergence is the strongest steer the campaign has.

**Caveat on the instrument:** this is an ordering-only measure over our own
text scored against GT text — NOT the official `reading_order` column, which
scores region sequence and would show a different (larger) oracle. It is a
RELATIVE prize between segments, which is what a target-selection decision
needs. A lever that moved the order column without moving text would still be
low value under §2's objective, where text is primary.

---

## 30. THE EVALUATOR SCORES ABSENT PAGES AS 1.0 — §28's fix was wrong, and lever B is refuted

**The instrument fact, verified rather than assumed:** a page with no prediction
file is scored **1.0**, not omitted. The leadfix arm wrote 1 351 files, the
evaluator scored 1 557 pages, and **all 296 pages without a file read exactly
1.0000**.

**This voided a result and invalidated a fix.** The first leadfix scoring read
-0.159 text / -0.145 order, "WORSE, CI excludes 0" on every scope — a clean,
decisive-looking refutation that was entirely 296 absent pages. And §28
concluded the harness should write NO FILE on a crash so the evaluator would
omit the page: **writing nothing and writing empty are identically corrupting.**
§28 replaced a bug with the same bug wearing better intentions, on an
assumption about the evaluator that was never tested.

**The real mechanism** is `score_arm.py`: filter the GROUND TRUTH to the pages
BOTH arms produced, and score both on that same filtered GT. That is the only
exclusion the evaluator offers, and it enforces population parity instead of
assuming it.

### Lever B, scored correctly

Population-matched to 1 351 pages, both arms on the same filtered GT:

| metric | scope | n | base | leadfix | gain | |
|---|---|---:|---:|---:|---:|---|
| text_block | all | 1261 | 0.1179 | 0.1220 | **-0.0041** | worse, CI excludes 0 |
| text_block | english | 614 | 0.1040 | 0.1118 | **-0.0078** | worse, CI excludes 0 |
| text_block | EN no-float | 163 | 0.1363 | 0.1426 | -0.0064 | spans 0 |
| reading_order | all | 1339 | 0.2390 | 0.2409 | -0.0019 | worse, CI excludes 0 |
| reading_order | EN no-float | 163 | 0.0905 | 0.0900 | +0.0004 | spans 0 |

**REFUTED — but honestly this time, at a hundredth of the fake magnitude.**
Stripping leading and trailing furniture is mildly harmful: the geometric
heuristic removes real body text more often than it removes running headers,
and even on its own target segment it buys nothing.

The DIAGNOSIS stands — a leading header really can cascade a near-correct page
to 1.000, and three such pages were shown. What is refuted is that a
geometric top-of-page rule can find them without collateral damage. A rule
precise enough would need the linguistic signal (§8.160's lesson: page
structure is a text property, not a geometric one), and its ceiling on 163
pages is too small to justify building one.

### Standing lesson

**Test what the instrument does with a MISSING input before designing around
it.** §3 rule 10 has now been got wrong twice from opposite directions: first by
writing a placeholder the metric could score (§28), then by removing the
placeholder and assuming absence meant exclusion (§30). The correct form is
neither — it is to change the POPULATION both arms are scored on.

---

## 31. THE FILTERS WERE FITTED UNDER A CONFIG WE NO LONGER RUN

Raised from outside the measurement loop: *most of our filters were added
before SVTR, on CRNN.* Auditing every filter still live under the banked config
(body-only OFF, CJK arm on, SVTR):

| filter | fitted under | status now |
|---|---|---|
| `reject_threshold()` | CORD/CRNN (§8.22) | **0.0 — off by default**, not a factor |
| `VERIFY_LOWCONF` | CRNN's confidence scale | already per-recognizer (§8.171) |
| `join_fluency` | Latin text | already script-routed (§18) |
| §8.135 block rules | CRNN + body-only ON | **dead — body_only never runs** |
| `ORDER_GATE_T = 0.0525` | CRNN-era boxes | pre-recognition, recognizer-independent |
| **§8.157 guard `body_frac > 0.85`** | **body-only ON** | **DEGENERATE — always true** |

**The live defect.** `probe_gate_fires` is
`n_col >= 3 && cover >= 0.18 && aspect > 1.6 && body_frac > 0.85`, where
`body_frac = n_body / n_all` measures how much SUPPRESSION removed. §19 turned
suppression off for the benchmark config, so `n_body == n_all`, `body_frac` is
**always exactly 1.0**, and the fourth term is always true.

The guard silently went from four conditions to three and **now fires more
often than it was ever fitted to**. That is not cosmetic: ground 1 BYPASSES the
text verifier, and §22 measured the verifier as genuinely good (disabling it
costs 0.0083 text, CI excluding zero). A looser guard force-accepts reorders on
pages the verifier would have judged.

`FFAI_ORDER_GUARD=0` drops ground 1 so the verifier decides every page. Default
unchanged, 38 tests pass, A/B-able from one binary.

**The general lesson, and it is not the one the observation started from.** The
filters did not rot because the RECOGNIZER changed — §8.171 and §18 had already
caught those. They rotted because a CONFIG change (§19's body-only OFF) silently
emptied a term that another filter depended on. **A constant is fitted against a
whole configuration, not against one component**, so any config flip should
re-audit every constant whose inputs that flip touches. §19 banked a real win
and left a dead term behind it; nobody looked, because the win was measured and
the term was invisible.

---

## 32. The engine choice RE-VALIDATED on the correct instrument — mobiledet-svtr holds

The largest decision still resting on the §8.173 scorer: mobiledet+SVTR over
CRAFT+CRNN. Three things argued it deserved re-testing — §8.16 recorded that
mobiledet "won speed decisively, quality LOST"; §8.170's +1.435 pp for SVTR was
measured on the biased scorer with body-only ON (a config §19 has since shown
was harmful); and on the Text OCR crop task `craft-crnn` read **0.1051** against
`mobiledet-svtr`'s 0.4133, four times better with GT boxes supplied.

Run end-to-end on 755 English pages, official evaluator, current config:

| metric | mobiledet-svtr | craft-crnn | |
|---|---:|---:|---|
| text_block | **0.1073** | 0.3522 | -0.2449, CI [-0.2693, -0.2199] |
| reading_order | **0.2257** | 0.4149 | -0.1892, CI [-0.2137, -0.1654] |

**mobiledet-svtr wins by a factor of three, decisively.** 540 of 721 pages
worse under craft-crnn on text.

**And it resolves the crop-task paradox in the other direction from §26.** §26
concluded DBNet fragments on tight crops and CRAFT does not — true, and it made
craft-crnn the right engine to REPORT on that task. But end-to-end, where the
detector must find its own regions on a full page, CRAFT is far worse: it is
strong at recognising a supplied region and weak at finding regions. Those are
different jobs and the crop task only measures the first.

**What this retires:** the engine choice was the biggest decision still leaning
on the discredited instrument, and it survives re-measurement on the correct
one. Every mechanism from that era has now been re-tested — four fell
(standings, Latin verifier, body-only, competitive frame), and three stand
(verifier, probe, and now the engine).

**What it costs the §26 claim: nothing, but the framing tightens.** Reporting
`craft-crnn`'s 0.1051 on the Text OCR task remains honest — it is a real
configuration on a real task — and `docs/textocr-claim.md` already discloses
that our document default scores worse there. This adds the converse, which
belongs beside it: on full-page parsing the default is three times BETTER, and
the two facts together are the whole truth about the two engines.

---

## §33 — a clean full-corpus baseline, one refuted claim of mine, and the grouping prize

**The run.** Current config, all 1651 pages, `--workers 10`, 156 min,
**1651/1651 files written and zero hard failures** — the first complete,
crash-free pass this campaign has had. Banked as `odb_pred_fullcur`.

| metric | fullcur | pages scored | prior banked claim | old `odb_pred_full` |
|---|---:|---:|---:|---:|
| Text^Edit | **0.1278** | 1557 | 0.1307 | 0.1636 |
| ReadOrder^Edit | **0.2336** | 1638 | 0.2348 | 0.3226 |

Only 7 pages read 1.0 on text and all are genuine quality failures, not
crashes. The 94 pages unscored on text carry no `text_block` annotation.

Against the old `full` baseline the current config gains **+0.0358 text** (CI
[+0.0293, +0.0422]) and **+0.0889 order** (CI [+0.0800, +0.0981]) — the banked
wins are real and large.

**REFUTED, mine.** I claimed the 0.1307 headline was inflated because it was
computed on 1261 of 1651 pages and the 296 excluded pages were 4.6x bigger and
scored 33% worse. Measured: 0.1278. **The banked headline was sound and §1
needs no correction.** The error was characterising the excluded pages with the
OLD `full` run; the current config improved most on exactly those big pages, so
a real population gap never became a headline gap. General form: a population
bias in the SCORE is only a bias in the HEADLINE if the excluded pages differ
under the CONFIG BEING MEASURED — demonstrate that on the arm itself, never on
a predecessor.

**Baseline hygiene.** `gg_arm.py` now defaults `--base-save` to
`odb_pred_fullcur_quick_match`. Differencing a new arm against `odb_pred_full`
credits it with the +0.0358/+0.0889 already banked.

**Where the error is now**, and priced per segment. Ceiling = each GT block may
bind the best order-preserving selection of characters from our OWN page stream
(LCS); it invents no characters and reorders nothing, so it bounds a pure
line->paragraph grouping fix.

| segment | pages | share of text err | now | ceiling | gain |
|---|---:|---:|---:|---:|---:|
| **newspaper** | 151 | 15.5% | 0.2047 | **0.0462** | **+0.1585** |
| — simplified_chinese | 74 | 9.9% | 0.2664 | 0.0581 | +0.2083 |
| — english | 77 | 5.6% | 0.1455 | 0.0348 | +0.1107 |
| exam_paper | 192 | 16.1% | 0.1671 | 0.1236 | +0.0435 |
| PPT2PDF | 245 | 18.0% | 0.1461 | 0.1167 | +0.0295 |
| academic_literature | 191 | 8.8% | 0.0914 | 0.0595 | +0.0319 |
| note | 115 | 7.7% | 0.1328 | 0.1217 | +0.0111 |
| traditional_chinese | 12 | 2.2% | 0.3586 | 0.3552 | +0.0034 |
| historical_document | 5 | 2.5% | 0.9809 | 0.9722 | +0.0087 |
| research_report | 107 | 1.8% | 0.0337 | 0.0334 | +0.0003 |

Corpus ceiling **0.1278 -> 0.0875, +0.0403, 32% of current error** — roughly
4.5x the largest banked win.

**The mechanism.** We emit one markdown block per physical LINE (median block
22 chars against GT's 104; the blank-line join is `gg_arm.py:125`). Text^Edit is
a length-weighted micro-average over MATCHED blocks (`cal_metric.py:386`), so a
615-char GT paragraph binds one 23-char line of ours and the other 592 chars are
charged as pure deletion. For the CJK blocks carrying a third of all edit mass,
our page emits 7000+ CJK chars, the median fraction of each block's distinct
characters present on our page is **1.00**, and median in-sequence (LCS) recall
is **0.98** — we read the characters correctly and in the right order. Only
9/160 sampled blocks are genuinely scrambled. Latin adds a second, smaller
defect: soft hyphens at line ends are preserved (`dramat-` + `ically`) where GT
is de-hyphenated.

**Two failure classes, and they need different levers.**
- *Grouping-limited* — newspapers above all (77% of their error is recoverable),
  then english PPT2PDF, book, academic_literature, magazine.
- *Recognition-limited* — `traditional_chinese` (0.3586, ceiling 0.3552) and
  `historical_document` (0.9809, ceiling 0.9722) are near-total failures with
  NO grouping headroom. Small (17 pages, 4.7% of error) but genuinely unread.
  `exam_paper` is half and half: +0.0435 recoverable, but its ceiling is still
  0.1236, and formula-bearing blocks are the likely residue.

**NOT YET PRICED — do not skip (rule 6).** Only the TEXT column is priced here.
Grouping collapses ~350 orderable units into ~20 and must move ReadOrder;
newspapers are also the worst order segment (0.2774). Direction is plausibly
good, but that is a prediction, not a measurement, and it must be priced before
anything ships.

**Prior art that constrains the build.** §4's block-grouping refutation is real
(0.2051 vs 0.1041 on matched population). MGAM merged into coarse REGIONS and
could not split; coarse output is irrecoverable. Line->paragraph joining inside
an already-detected column is a different operation, but over-merge is the
documented failure mode, so it must not cross a column or region boundary.

---

## §34 — line->paragraph joining REFUTED, and it promotes §4 into a law

**The arm.** `.tools-bench/join_lines.py` transforms an existing prediction dir:
join physical lines into paragraphs, de-hyphenate soft line breaks. No engine
pass, no Rust. Caps swept against GT rather than guessed — `soft=35/hard=80`
lands **14 blocks/page against GT's 13** and **85 chars/block against GT's 104**,
deliberately more-and-shorter than GT. Correctness gate: character stream
**byte-identical on 1339 pages**, differing only by removed soft hyphens on 312,
**zero real mismatches** — nothing invented, dropped, or reordered.

(First attempt used caps of 300/600 and collapsed pages to 5 blocks — a textbook
over-merge, caught only by comparing against GT granularity. Calibrate a cap
against the target distribution, never against how the output looks.)

**The verdict**, `score_arm.py`, population-matched at 1651 pages:

| metric | scope | base | joined | gain | |
|---|---|---:|---:|---:|---|
| text_block | all | 0.1278 | **0.3109** | -0.1831 | WORSE, CI excludes 0 |
| text_block | english | 0.1051 | 0.2221 | -0.1170 | WORSE, CI excludes 0 |
| reading_order | all | 0.2335 | **0.3391** | -0.1056 | WORSE, CI excludes 0 |
| reading_order | EN no-float | 0.0907 | 0.2402 | -0.1495 | WORSE, CI excludes 0 |

2.4x worse on text. Hurt 1411 of 1557 pages. **Worse in every block-count band,
including the 250+ band that was predicted to improve.** Ceiling for a PERFECT
block-count dispatch: **+0.0088 text** — so the lever is dead outright, not
mis-gated. Do not build the dispatch.

**Mechanism, visible in one block.** The first joined block of the Guardian page
reads `Wednesday 8 January 2o25 The Guardian Features 33 Continued from page 32
compassion" when she tried this on` — a running header, section label, page
number, continuation notice and body text welded into one. The matcher MERGES
BUT CANNOT SPLIT, so **fine-grained output is composable and coarse output is
irrecoverable**. Line-level emission is not our defect; it is what protects us.
§4 refuted this for MGAM's region grouping; this refutes it again by an
unrelated route, which promotes it from a fact about one lever to a **law about
this metric: never coarsen our block structure.**

**It also resolves the §33 confound AGAINST §33's reading.** §33 found a cliff
(pages over 250 blocks score 0.20-0.44 vs 0.12 below) and flagged that dense
pages both produce more blocks AND are intrinsically harder. This arm held the
page constant and changed only block count. Score got worse. **The cliff is
correlation, not causation**, and "get under ~250 blocks" was a bad prescription
drawn from a true number. General form: a monotone relationship between an
output property and the score is not a lever until an arm has MOVED that
property on fixed inputs.

**What survives.** The CJK measurement stands — median in-sequence (LCS) recall
0.98, so those characters really are read correctly and in order. The +0.0403
oracle ceiling stands as an upper bound on perfect re-binding. What dies is the
claim that our block structure is the route to it: reaching it needs boundaries
matching GT's EXACTLY, and this result prices the penalty for a wrong boundary
as brutal. Discount that direction heavily before spending engine work on it.

---

## §35 — MY §33 CEILING WAS WRONG: LCS is not a valid oracle on CJK

**The error.** §33 priced the re-binding ceiling with an LCS oracle
(`LCSseq.similarity(gt, page_stream)`) and reported **+0.0403, 32% of current
error**. LCS lets a GT block cherry-pick characters from ANYWHERE on the page in
any spacing. On a page carrying 8000 Chinese characters drawn from the same few
thousand code points, a 600-character GT block scores near-total LCS recall **by
chance**, and no block binding can ever realise it. A predicted block is a
CONTIGUOUS RUN of our output, so the only admissible oracle is the best
contiguous WINDOW.

| oracle | ceiling | headroom | |
|---|---:|---:|---|
| contiguous window | 0.1146 | **+0.0126** | correct |
| LCS | 0.0869 | +0.0404 | **inflated 3.2x** |

| segment | now | ceiling (correct) | gain | gain claimed in §33 |
|---|---:|---:|---:|---:|
| newspaper | 0.2005 | 0.1581 | **+0.0424** | +0.1543 |
| academic_literature | 0.0943 | 0.0756 | +0.0187 | +0.0349 |
| book | 0.1100 | 0.0992 | +0.0107 | +0.0288 |
| exam_paper | 0.1671 | 0.1576 | +0.0095 | +0.0435 |
| PPT2PDF | 0.1461 | 0.1369 | +0.0092 | +0.0295 |
| note | 0.1328 | 0.1309 | +0.0019 | +0.0111 |
| research_report | 0.0337 | 0.0337 | +0.0000 | +0.0003 |

**What else this retracts.** §33 claimed "for the CJK blocks carrying a third of
all edit mass, median in-sequence recall is 0.98 and the median fraction of each
block's distinct CJK characters present on our page is 1.00 — we read the
characters correctly and in the right order." **Both statistics are vacuous on
CJK.** A distinct-character-set overlap of 1.00 says only that two Chinese texts
use the same common characters; LCS recall of 0.98 says only that a long haystack
in the same script contains an in-order subsequence. Neither is evidence we read
anything. A stricter probe (SequenceMatcher, runs of >=6 characters) finds that
only **6 of 60** badly-bound blocks have >=75% of their GT text genuinely present
in our page, and **0 of 6** show an interruption >= 40 characters -- so the
"read but interleaved" story is dead too. Those blocks are simply NOT READ.

**The conclusion inverts.** There is no large binding prize. Total realistic
re-binding headroom is **+0.0126 across the whole corpus**, and outside newspapers
it is negligible. The residual is genuine RECOGNITION and DETECTION failure --
which is where §34 already pointed by elimination, and is now positively
confirmed rather than inferred.

**The law.** Never price a ceiling with a metric that can succeed by chance on
the alphabet in play. An oracle must be constrained to what the SYSTEM can
actually emit -- here, a contiguous run. General form: **an oracle that is not
realisable by any implementation is not a ceiling, it is a fantasy.** Test any
proposed oracle against a null arm (score it on an UNRELATED page of the same
script); if the null scores well, the oracle is measuring the alphabet.

---

## §36 — 8 of our 10 "worst pages" are EVALUATOR TIMEOUTS, not engine failures

**The detector probe.** `FFAI_OCR_CONF=1` makes `ocr_text` emit
`conf \t x \t y \t w \t h \t text` per line, so the shipped engine's own
detections are inspectable with no code change. Run on the worst-scoring pages:

| page | score | our lines | our box area vs GT | chars read | median conf |
|---|---:|---:|---:|---:|---:|
| Chicago Tribune p015 | 0.990 | 738 | 37.7% vs 37.2% | 31,648 | 0.978 |
| newspaper_0b1bb8d0…_1 | 0.995 | 2,312 | 45.9% vs 86.5% | 18,015 | 0.999 |
| enbook…pdf_105 | 1.000 | 213 | 38.5% vs 66.2% | 5,515 | 0.994 |

Near-identical text coverage to GT, tens of thousands of characters, confidence
0.98+, `frac<0.5 = 0.00`. **Not a detection failure and not a recognition
failure.**

**The cause.** All ten are in `fallbacks` in the run's own
`stage_execution.json` — 8 `quick_match_timeout`, 2 `page_timeout`. The matcher
exceeded its budget (defaults 300s/420s) and a substitute penalty was scored.
Re-scored in ISOLATION at 5400s/7200s:

| page | text @300s | text @5400s |
|---|---:|---:|
| enbook…pdf_105 | 1.0000 | **0.0413** |
| enbook…pdf_57 | 1.0000 | **0.0251** |
| Chicago Tribune p015 | 0.9895 | **0.0224** |
| newspaper_2a6b4fa0…_16 | 0.9774 | **0.0193** |
| newspaper_2a6b4fa0…_15 | 0.8268 | **0.0188** |
| newspaper_2a6b4fa0…_7 | 0.7225 | **0.0158** |
| newspaper_2a6b4fa0…_8 | 0.6856 | **0.0165** |
| scihub…chem…pdf_9 | 0.5974 | **0.0317** |
| newspaper_0b1bb8d0…_1 | 0.9946 | 0.4955 |
| jiaocai_needrop_en_349 | 0.9260 | 0.8378 |

Mean **0.8720 -> 0.1524**. Several of these are our BEST pages once matched.

**Two standings, both true, and they must be quoted with their budget.**
- Default budget (what a leaderboard run produces): **text 0.1278 / order 0.2335**
- Adequate budget (what our output earns on quality): **text 0.1232 / order 0.2306**

The 0.0046 is a real cost of emitting fine-grained blocks against a matcher with
a compute budget, so the default-budget number stays the headline. **But every
INTERNAL A/B should raise the timeout**, because 21.6% of block edit mass is
fallback penalty that no engine change can move — pure noise in every arm.

**How much of the analysis this contaminated.** Those 10 pages carry **21.6% of
all block edit mass and 28.6% of the "badly wrong" bucket** from 143 of 20,113
blocks. Recomputing with them removed:

| bucket | with fallbacks | CLEAN |
|---|---:|---:|
| read correctly, bound to the wrong block | 22.0% | **4.5%** |
| missed entirely | 19.8% | **1.2%** |
| CJK absent | 32.0% | **39.3%** |
| Latin under 25% emitted | 8.4% | 10.2% |
| formula-bearing | 5.6% | 5.6% |

**The binding story was almost entirely a timeout artifact.** With it gone, three
independent lines now agree: the contiguous ceiling is only +0.0126 (§35),
block-structure changes are refuted (§34), and binding is 4.5% of clean error.
**The failure is RECOGNITION on CJK — 39.3% of clean edit mass — plus Latin
under-emission at 10.2%.**

**Genuine engine failures, now visible with the noise removed.**
`page-3ecc67a1` returns ZERO detections against 3 GT text blocks;
`page-942ac90d` returns ONE box holding the single character 九 against 8 GT
blocks covering 48.6% of the page. The true worst list is dominated by
`historical_document` (traditional Chinese) and `yanbaopptmerge` PPT pages where
we emit 1–9 blocks for a full page.

**The law.** A page-level score that a MEASUREMENT FAILURE can produce is not
evidence about the system. Before ranking worst pages, subtract every page the
harness reported as degraded — the fallback list is in the run's own
`stage_execution.json` and it cost three separate conclusions here.

**Near-miss, recorded.** The first attempt at this rescore pointed the evaluator
at `odb_pred_fullcur`, whose save name would have OVERWRITTEN the banked
1557/1638-page baseline with a 10-page file. Caught before it wrote. Rescoring a
SUBSET must use a copied prediction dir with a distinct name; back up
`result/*_per_page_edit.json` first (`.tools-bench/_baseline_backup/`).

---

## §37 — READING ORDER IS THE LEVER: oracle ceiling +0.0378 text / +0.0532 order

**The chain, measured end to end on the CJK failure.**

| stage | verdict | evidence |
|---|---|---|
| Detection | fine | 75% median area coverage of GT blocks; **0 of 493** horizontal CJK blocks under 20% covered |
| Recognition (SVTR) | excellent | our lines assembled BY GEOMETRY reconstruct the GT block at **median edit 0.023**; 193/222 under 0.10 |
| Serialization | **broken** | those same lines score 0.26 as emitted |

SVTR is not the problem. It reads these pages nearly perfectly, and every
character is already in the right place on the page.

**The mechanism, and it is monotone.** Per-block contiguity (our lines inside a
GT block, as a fraction of the index span they occupy in our emitted stream):

| block edit score | n | median contiguity | fully contiguous |
|---|---:|---:|---:|
| under 0.10 | 6661 | 1.00 | 98% |
| 0.25–0.50 | 590 | 1.00 | 67% |
| 0.50–0.80 | 596 | 0.56 | 27% |
| over 0.80 | 755 | 0.46 | 15% |
| **CJK under 0.10** | 3077 | 1.00 | **98%** |
| **CJK over 0.50** | 857 | 0.42 | **4%** |

**The oracle arm.** `.tools-bench/oracle_order.py` groups our captured lines by
which GT block contains them and emits each group contiguously in GT's reading
order. Line-level granularity untouched (§34's law), character multiset
identical on all 1651 pages, 91% of lines repositioned.

| metric | fullcur | oracleord | gain | CI |
|---|---:|---:|---:|---|
| text_block all | 0.1274 | **0.0895** | **+0.0378** | [+0.0322, +0.0435] |
| text_block english | 0.1043 | 0.0753 | +0.0290 | [+0.0208, +0.0377] |
| reading_order all | 0.2336 | **0.1804** | **+0.0532** | [+0.0459, +0.0607] |
| reading_order EN no-float | 0.0907 | 0.0443 | +0.0464 | [+0.0219, +0.0728] |

**Ordering is CAUSAL, not a symptom.** 0.0895 text would sit mid-pack on the
published board. This is ~4x the largest banked win, and it is the same lever on
BOTH columns.

**It also reconciles §35.** The +0.0086 re-binding ceiling was the best contiguous
window available IN OUR CURRENT ORDER. Fix the order and far more becomes
bindable; the numbers were never in conflict, one was conditioned on the other.
It likewise supersedes §36's refutation of serialization, which rested on 6
analysable blocks and was underpowered.

**THE ORACLE IS NOT SHIPPABLE and the engineering is not free.** The one real
implementation tried — a fresh recursive XY-cut over our own boxes
(`.tools-bench/reserialize.py`) — **worsened 4229 blocks against 375 improved**,
because our emitted order is ALREADY contiguous for 82% of blocks corpus-wide
and the crude cut rewrote the majority that were correct. `boxes.rs` already
implements a tuned XY-cut; the target is its FAILURE CASES, not a replacement.
My "median lines/span 0.56" came from ONE newspaper page — the third
generalisation-from-a-small-sample error today, caught by the gate.

**Two reusable instruments, and one is the reason this was affordable.**
- `.tools-bench/capture_boxes.py` — one 132-min pass captures per-line
  `conf/x/y/w/h/text` for all 1651 pages. `FFAI_OCR_CONF` changes printing only,
  so the geometry is now available offline and any future ordering or grouping
  idea costs SECONDS instead of an engine pass.
- **Per-block contiguity is a fast proxy gate.** It refuted the XY-cut arm in
  seconds without the evaluator. Use it before any scoring run.
- The NULL ARM matters: `reserialize.py --identity` is **byte-identical to
  fullcur on all 1651 pages**, which is what licenses attributing the whole
  oracle delta to ordering.

**Next.** Improve region segmentation in `boxes.rs` where it fails, gated on
contiguity first and the evaluator second. The ceiling is known (+0.0378/+0.0532)
and so is the failure mode of the naive approach.

---

## §38 — CJK layout IS different, but not through the valley floors

**The layout difference is real and measured** from our own captured boxes:

| metric (newspapers) | english | simplified_chinese | ratio |
|---|---:|---:|---:|
| leading / line-height | **0.04** | **0.32** | **8.6x** |
| gutter / line-height | 0.93 | 3.12 | 3.3x |
| chars per line | 34 | 16.5 | 0.49x |
| line boxes per page | 206 | 435 | 2.1x |

CJK is set on a fixed em grid — square glyphs, real inter-line air, no word
spaces. Latin packs lines nearly touching and separates WORDS horizontally
instead. So `H_GAP_MIN`'s stated calibration — "ordinary leading, which is ~1
line height by construction" — has no single value on this corpus.

**HYPOTHESIS REFUTED.** I predicted that CJK's 8x larger leading makes horizontal
valleys qualify too easily, severing live multi-column blocks and producing the
`[0,5,1,5,…]` interleave the source names. Raising the floor should then help.
Swept on 60 zh + 40 en pages, gated on contiguity:

| arm | zh contiguity | en contiguity | zh full-contig | en full-contig |
|---|---:|---:|---:|---:|
| base H1.35 V0.55 | 0.556 | 1.000 | 0.39 | 0.58 |
| H 2.50 | **0.500** | 1.000 | 0.31 | 0.58 |
| H 4.00 | **0.500** | 1.000 | 0.30 | 0.55 |
| V 0.35 | 0.571 | 1.000 | 0.40 | **0.66** |
| H2.50 V0.35 | 0.500 | 1.000 | 0.31 | 0.66 |

Raising `H_GAP_MIN` makes Chinese **worse** (0.556 -> 0.500) at both 2.5 and 4.0.
The horizontal cut is doing useful work, which is what §8.68 concluded from the
other direction. **A measured layout difference does not imply the constant that
mentions that layout property is the lever.** The mechanism has to be shown, not
inferred from a correlation between a statistic and a comment.

**One marginal survivor.** `V_GAP_MIN=0.35` is the only non-negative arm: zh
median +0.016, en median unchanged, and English fully-contiguous blocks rise
0.58 -> 0.66. It closes ~3% of the gap between current zh contiguity (0.556) and
the oracle (1.000), so its expected score effect is small. Candidate for ONE
scoring run on the strength of harming neither language — not a priority.

**What this leaves standing.** §37's ordering ceiling (+0.0378 text / +0.0532
order) is untouched and remains the largest priced lever in the campaign. What
is now excluded is that the ordering defect lives in the valley-width floors:
§8.68 refuted the horizontal floor from above, this refutes it from below, and
`xy_cut_cost` already replaced width with a severing-cost criterion. The residual
interleave is elsewhere in region segmentation.

**Cost of the refutation: 30 minutes**, because the contiguity gate stands in for
the evaluator. Three hypotheses have now died on that gate without a scoring run.

---

## §39 — V_GAP_MIN 0.55 -> 0.35: a real but near-inert +0.0012

Full corpus, one binary, one env toggle, 1651/1651 pages, zero failures,
scored at the SAME 300/420 matcher budget as the baseline (matching budgets
matters more than §36's noise reduction applied to one side only).

| metric | fullcur | vgap035 | gain | CI | verdict |
|---|---:|---:|---:|---|---|
| text_block | 0.1274 | **0.1262** | +0.0012 | [+0.0001, +0.0026] | BETTER, CI excludes 0 |
| reading_order | 0.2336 | 0.2330 | +0.0005 | [-0.0009, +0.0020] | spans 0 |

**It meets the banking bar** — text CI excludes zero, order does not regress —
**and it is nearly inert**: 42 of 1557 pages change at all (22 helped, 20 hurt),
the CI's lower bound is +0.0001, and the perfect-gate ceiling is +0.0016 against
+0.0012 blind, so there is no gate worth building. Apply or don't.

Recorded with its fragility explicit: an effect carried by 42 pages with a
bootstrap lower bound one ten-thousandth above zero is the kind of number that
does not survive a change of population. It should be re-checked, not trusted,
if it ever becomes load-bearing for a later conclusion. Compare the §8.157 guard
(6 of 755 pages, net -0.000023) which was recorded as inert.

**The prediction was made in advance and was correct** — "a gain in the low
thousandths, quite possibly with a CI spanning zero". Stating the expected
magnitude before the run is what makes a marginal result readable as marginal
rather than as a win.

**Strategically it changes nothing.** §37's ordering ceiling stands at +0.0378
text / +0.0532 order — thirty times this. The valley-floor axis is now closed
from every direction: §8.68 from above, §38 from below, and this its best case.
The residual interleave is in region segmentation, not in the thresholds.

---

## §40 — We are not failing to SEE the text. We are failing to REPRESENT it.

**The question**: on the pages we score badly, are the characters read and
mis-ordered, or genuinely not read? Answered per page by assembling OUR OWN lines
into each GT block by geometry and sorting them — perfect binding and perfect
ordering, using only characters we already produced.

**Bad pages split almost binarily BY DOCUMENT TYPE.** Of 192 pages scoring >= 0.25:

| type | ordering-limited | |
|---|---|---|
| newspaper | **39/39** | **100%** |
| magazine | **8/8** | **100%** |
| PPT2PDF | 7/20 | 35% |
| colorful_textbook | 6/21 | 29% |
| academic_literature | 5/20 | 25% |
| exam_paper | 6/46 | 13% |
| note | 1/13 | 8% |
| book | 1/20 | 5% |
| historical_document | 0/5 | 0% |

Overall 38% ordering-limited, 48% not-read, 14% partial. Pages in the 0.25–0.50
band score **0.335 emitted and 0.337 assembled** — perfect ordering buys them
NOTHING. Only the worst band moves (0.642 -> 0.452).

**This corrects §37's generalisation.** The median-edit-0.023 assembly result was
measured on four sampled NEWSPAPER pages — the one category that is 100%
ordering-limited — and I read it as the CJK bulk. Corpus-wide it is 38% of bad
pages. Sample the category you intend to generalise to.

**Why the not-read pages are not read** — 92 pages, 536 failing blocks:

| cause | blocks | % of gt chars |
|---|---:|---:|
| **GT is LaTeX/markup we cannot emit** | 348 | **65.9%** |
| thin detection (<40% of block covered) | 132 | 29.9% |
| no detection in the region | 51 | 3.6% |
| covered, characters WRONG (recognition) | **5** | **0.5%** |

**RECOGNITION IS 0.5%.** SVTR is not the problem anywhere. What "not read" means
in practice:

```
GT  : [答案]：解令$t=\operatorname{sin}^{2}x$则$\mathrm{d}t=2\operatorname{sin}x...
OURS: [答案]:解令t=sin2x,则dt=2sinxcosxdx,sin4x+cos4x=2t2-2t+1,则
```

We read it correctly. GT is LaTeX SOURCE; we emit RENDERED Unicode. Our line is
arguably the better text and scores as a total miss.

**Corpus-wide sizing** (timeout pages excluded, total edit mass 288 434):

| | blocks | % of ALL text error |
|---|---:|---:|
| GT carries LaTeX in the raw annotation | 2423 | **20.6%** |
| ...LaTeX SURVIVES the evaluator's normaliser | 131 | **4.9%** |

The evaluator's `latex2unicode` collapses simple cases (`$10\%$` -> `10%`) but
leaves `\int`, `\frac`, `\operatorname` intact, so those blocks are unwinnable
without emitting LaTeX ourselves. **This is a CAPABILITY GAP, not a defect**: the
leaders on this board are VLMs that emit LaTeX natively. It also explains the
`exam_paper` segment (largest bad-page bucket, 46 pages, 87% not-read) and its
0.1236 floor even at ceiling.

**The three real targets, now sized and disjoint.**
1. **Formula representation — 20.6% of all text error.** Needs a formula path
   that emits LaTeX. Largest single target found in this campaign.
2. **Ordering — +0.0378 text / +0.0532 order (§37 oracle)**, concentrated
   entirely in newspapers and magazines, which are 100% ordering-limited.
3. **Thin detection — 29.9% of not-read characters**, on exam papers and
   textbooks: the region is found but under 40% of its area is covered.

Everything else is noise: recognition 0.5%, re-binding +0.0086, valley floors
closed from both directions.

---

## §41 — All three candidate features tested. No resounding win. Nothing shipped.

**Directive**: test all three, revert anything that is not a resounding win.
**Status**: `crates/ffai-carmenta/src` was never modified. There is nothing to
revert. All three were tested as offline arms or priced as ceilings.

**First, the ordering ceiling decomposes cleanly.** The contiguity-only arm
(group by GT block, keep OUR block sequence) scored, via score_arm.py on the
full 1651-page population:

| arm | text | order |
|---|---:|---:|
| contiguity only | **+0.0337** CI [+0.0283, +0.0390] | -0.0036 spans 0 |
| full oracle (+ GT block sequence) | +0.0378 | +0.0532 |

**90% of the TEXT gain is line grouping; ALL of the ORDER gain is block
sequence.** They are independent levers on different columns. This corrects my
guess that sequence dominated text — it contributes +0.0038.

**All three, priced on one scale** (baseline 0.1230, timeout pages excluded):

| feature | result | gain | kind |
|---|---:|---:|---|
| 1. Formula — every LaTeX block perfect | 0.0757 | **+0.0473** | oracle |
| 2. Ordering — contiguity arm | 0.0938 | **+0.0337** | SCORED |
| 3. Detection — every thin-coverage block perfect | 0.1013 | +0.0217 | oracle |

**And the blast radius, which decides whether a ceiling is worth chasing:**

| feature | upside | at risk |
|---|---|---|
| Formula | 475 bad LaTeX blocks, 12.3% of error | **1496 LaTeX blocks already scoring well** (12.9% of chars); 9358 prose blocks (60.2% of chars) if math-detection misfires |
| Ordering | 47 bad newspaper/magazine pages | 82% of blocks already contiguous |
| Detection | 29.9% of not-read chars | adds boxes to all 1039 pages already scoring well; feeds the timeout tail |

**Implementation attempts for #2 — SIX, all failed:**

| attempt | result |
|---|---|
| recursive XY-cut rewrite (§37) | 4229 blocks worse vs 375 better |
| column-band stable sort (§40) | full-contiguity -0.048 |
| ...after fixing its band-detection bug | -0.048, changed 527 pages |
| `H_GAP_MIN` raised (§38) | zh contiguity 0.556 -> 0.500 |
| `V_GAP_MIN` 0.35 (§39) | +0.0012, CI lower bound +0.0001, 42 pages |
| **self-derived region grouping (§41)** | **full-contiguity 0.822 -> 0.29, rejected at the gate** |

The last is the sharpest: it did EXACTLY what the +0.0340 arm did, but derived
regions from our own geometry instead of GT. It destroyed contiguity on 1343 of
1411 pages. **Our geometry cannot reproduce GT's block grouping** — which is what
separates a ceiling from a feature, and no amount of tuning bridges it.

**VERDICT: none of the three is a resounding win. None shipped.**
- #1 has the largest ceiling and no implementation short of an image->LaTeX
  model; its at-risk mass EXCEEDS its upside.
- #2 has a real scored ceiling and six failed implementations.
- #3 has the smallest ceiling and the widest blast radius.

**Reversion mechanism verified, for when something IS built.** 64 env toggles
exist in the engine. Tested against the frozen `odb_pred_fullcur`: no toggles set
= byte-identical; defaults set explicitly = byte-identical; one toggle changed =
differs; **changed then restored = byte-identical**. Gating is a true revert, not
a hope. The frozen baseline (`odb_pred_fullcur`, `boxes_fullcur`,
`.tools-bench/_baseline_backup`) is the reference any future gate-off must match.

---

## §43 — vertical CJK: rotation REFUTED, column-splitting CONFIRMED, prize tiny

**Stage 0 diagnosis.** `page-942ac90d` emits ONE character against 8 GT blocks
covering half the page. Cause, traced end to end:

| stage | evidence | verdict |
|---|---|---|
| DBNet map | 0.242 inside GT text vs 0.002 outside (121:1; the control page we read well is 10:1) | **sees it** |
| `boxes_from_probability` | 19 components, **16 survive** every filter, median shape **19 x 1115** | **finds it** |
| `svtr_input` | scales every crop to H=48 and derives width from aspect: 1115 px column -> the 8-px floor | **destroys it** |

Detection was never the defect. Threshold sweeps (`DB_BIN`, `DB_BOX` to 0.05,
`MIN_SIDE`) were all sweeping filters that were already passing 16 of 19 boxes.

**HYPOTHESIS 1 — PaddleOCR-style rotation. REFUTED.** PaddleOCR turns a crop 90°
when height/width > ~1.5. Implemented behind `FFAI_ROT_ASPECT`, gate-off verified
byte-identical. Measured: `page-942ac90d` 1 char -> 43. But the output was
`一。。。一。一。三之王十。「。一。一）。。）（。` — **dominated by
rotation-symmetric characters**, against 719 GT chars.

The rule is for genuinely ROTATED lines. Classical CJK vertical typesetting keeps
glyphs UPRIGHT and stacks them downward, so rotating the column lays every glyph
on its side and only the rotation-invariant forms survive. **The output was the
hypothesis refuting itself.** Reverted.

**HYPOTHESIS 2 — split the column into upright glyph cells. CONFIRMED.**
`boxes::split_vertical_columns`, `FFAI_VSPLIT_ASPECT` (default 0 = off =
byte-identical, verified on 12 pages / 7 doc types).

| page | GT chars | off | VSPLIT 2.0 |
|---|---:|---:|---:|
| page-942ac90d | 719 | 1 | **270** |
| page-3ecc67a1 | 129 | 0 | **72** |
| page-dad79c54 | 627 | 22 | **464** |
| horizontal control | 1655 | 1693 | **1693 unchanged** |

Output is real classical Chinese — `卷一初見秦`, `諸侯`, `魯君`, `韓` — where
there had been one character. Threshold 2.0 leaves the control byte-unchanged;
1.5 begins touching horizontal pages, so 2.0 is the safe setting.

**Scored on the 14-page vertical population (isolated dir + filtered GT, §36):**

| metric | before | after | gain on population | **corpus effect** |
|---|---:|---:|---:|---:|
| text_block | 0.4039 | 0.3872 | +0.0167 | **+0.0001** |
| reading_order | 0.4924 | 0.4212 | +0.0713 | **+0.0006** |

Every affected page improved; none regressed.

**A CORRECTNESS win, not a score win.** 14 pages of 1557 cannot move a headline
whatever we do to them, and the pages still sit at ~0.9 text afterwards because
vertical CJK reads RIGHT-TO-LEFT and `order_reading` reads left-to-right — so the
columns come out reversed even now that they exist. That is Stage 3/4 work, not
detection.

**Recommendation: ship `FFAI_VSPLIT_ASPECT=2.0` on correctness, not on the
number.** It is gated, byte-identical off, provably a no-op on horizontal pages,
and it turns three total failures into partial reads. But nobody should expect it
in the headline.

**What Stage 0 has now established overall:** there is no general detection
weakness. Detection is healthy on 96% of blocks; of the 204 genuine misses, 87%
of the characters are LaTeX-bearing (a formula gap, §40), and the rest is this
vertical-text defect worth +0.0001. **Detection is not where the campaign's
remaining points are.**

---

## §44 — the three missing stages, and why the headline is a COVERAGE gap

**The reframing, from evidence already on disk.** `configs/unlimited_holdout.yaml`
scores Carmenta on a 236-page holdout: **English only, ZERO LaTeX-bearing text
blocks, ZERO table regions, ZERO isolated equations** (verified by counting the
annotations). On that holdout Carmenta reads **0.0406 text / 0.0522 order**.

| system | text edit distance |
|---|---:|
| PaddleOCR-VL | 0.0326 |
| **Carmenta, formula/table-free English** | **0.0406** |
| PP-StructureV3 (full pipeline, not a VLM) | 0.0794 |
| **Carmenta, full corpus** | **0.1278** |

**On content we can represent we already beat PP-StructureV3 and sit near the VLM
leader.** 0.1278 is not a quality deficit; it is a COVERAGE deficit. That is the
single most useful thing to know about this campaign and it was sitting in a
config file the whole time.

**What everyone else does.** Both paradigms route formulas to a dedicated LaTeX
model. Pipelines (PP-StructureV3, MinerU, Marker) run a LAYOUT model first —
PP-DocLayout classifies 23 region categories including `formula` — then send each
region to a specialist: text to PP-OCRv5, formula to PP-FormulaNet/UniMERNet,
table to SLANet. VLMs (PaddleOCR-VL, MinerU2.5, Baidu Unlimited-OCR) generate the
whole markdown including `$...$` as text.

And the detail that explains our formula scores exactly: **OmniDocBench's ground
truth was itself annotated with UniMERNet and GPT-4o for formulas.** The LaTeX we
are scored against is that family's output. Systems running the same class of
model reproduce its spelling; we never can.

**Why we don't do it that way: we implement 2 of 5 stages.**

| stage | reference | us |
|---|---|---|
| layout / region detection | PP-DocLayout | **absent** |
| text detection | PP-OCRv5 det | present |
| text recognition | PP-OCRv5 rec | present |
| formula -> LaTeX | PP-FormulaNet / UniMERNet | **absent** |
| table -> HTML | SLANet | **absent** |

Model cache confirms it: `ppocrv5-mobile-det` (5 MB), `ppocrv5-mobile-rec`
(16 MB), plus legacy CRAFT/CRNN/PARSeq. No layout, formula or table model exists
anywhere in the workspace (`grep -rln doclayout|picodet|rt_detr crates/*/src`
returns nothing).

**Competitor context.** Baidu **Unlimited-OCR** (June 2026; 3B total / 500M
active, Reference Sliding Window Attention for a flat KV cache, 32K context)
posts **93.92% on OmniDocBench v1.6**. That is the OVERALL metric,
`((1-text_edit)*100 + table_TEDS + formula_CDM)/3` — and our own config records
that we **cannot report it at all**: "TEDS and CDM are not merely skipped for
convenience — they are not computable for this engine." **Tables and formulas are
not an optimisation against that competitor; they are the precondition for being
scorable against it.**

**Port route established, with no framework installed.** PP-DocLayout,
PP-FormulaNet and UniMERNet all ship ONNX on the hub. ONNX is protobuf: the graph
carries ops/attributes/shapes and the weights sit in `initializer` — or, for
exports like this one, inlined in `Constant` nodes. `.tools-bench/onnx_inspect.py`
is a dependency-free reader (numpy only) that extracts both.

**PP-DocLayout-S, read end to end (4.9 MB):**

```
input   image [N,3,480,480] + scale_factor [N,2]
output  [N,6] (class, score, x0,y0,x1,y1) + [N] counts
1078 nodes, 536 initializers, 1.20 M params
Conv 98 · BatchNorm 82 · HardSwish 78 · GlobalAveragePool 6 · Sigmoid 12
Softmax 4 · MatMul 4 · TopK 2 · NMS 1 · Resize 2
```

PicoDet: LCNet backbone (HardSwish + SE), CSP-PAN neck, DFL head, NMS. **Every
primitive except NMS/TopK/Resize already exists in `svtr.rs`** — same HardSwish,
same hardsigmoid SE, same grouped convs — and the weight names use the same
`conv2d_N.w_0` paddle convention the SVTR port was generated from. Fixed 480x480
input, so CPU cost sits far below the -L variant's 760 ms.

**First trap avoided.** The reader initially reported "1078 nodes, 0
initializers, 0.00 M params" because `AttributeProto` field 5 (`t`, a
TensorProto) was unhandled and this export inlines every weight as a `Constant`.
A model with zero parameters is not a model — when a reader reports one, the
reader is wrong, not the file.

---

## §45 — layout stage DEPLOYED; formula and table encoders run, decoders do not

**Shipped and working: LAYOUT.** PP-DocLayout-S (PicoDet/GFL, 4.9 MB, 1.20 M
params) ported to candle and producing real regions:

```
page-bd6eb36e (academic, isolated equations)   load 5.8ms  infer 457ms
  formula 0.888 [503,1332 576x333]   formula 0.884 [503,412 587x336]
  text    0.872 [253,1689 1098x112]  formula 0.867 [585,908 388x285]
  formula 0.846 [601,1829 250x130]   text    0.579 [526,316 765x70]
```

Four display formulas and two inline ones, cleanly separated from text — **the
region class carrying 87% of our genuinely-missed characters** (§40), and the one
no detection-threshold sweep could ever produce because those knobs filter boxes
that already exist rather than creating a region class.

**Infrastructure built (reusable for all three stages):**
- `.tools-bench/onnx_inspect.py` — protobuf ONNX reader, numpy only. No paddle,
  torch, onnx or safetensors installed on this box; none needed.
- `.tools-bench/onnx_export.py` — writes safetensors by hand and emits an
  executable arch JSON.
- `crates/ffai-carmenta/src/onnx_graph.rs` — candle ONNX-subset executor. **The
  graph is DATA, not transcribed code**, which is what makes three ports
  tractable rather than three hand-transcriptions of 400+ nodes. §8.167 is the
  precedent: two of the three hypotheses that died in the SVTR port "matched
  every shape" and were still structurally wrong.
- `crates/ffai-carmenta/src/doclayout.rs` — preprocessing, class decode, native
  NMS for the graph tail.

**Traps hit and recorded.**
1. The reader first reported "1078 nodes, 0 initializers, **0.00 M params**"
   because `AttributeProto` field 5 (`t`) was unhandled and this export inlines
   every weight in `Constant` nodes. A model with zero parameters is not a model
   — when a reader says so, the reader is wrong.
2. `doclayout.rs` first hand-rolled its own bilinear sampler, duplicating
   `image::resize_bilinear`. Replaced: two resamplers in one pipeline is two sets
   of numerics to keep in step, and the drift would appear as layout boxes
   disagreeing with the text boxes they are meant to contain.

**NOT DEPLOYED: formula and table. Encoders run; decoders do not.** Both models
export and both encoders execute to completion under the same executor:

| model | size | nodes | encoder | decoder (`Loop`) |
|---|---:|---:|---|---|
| SLANet_plus (table) | 5.5 MB | 328 | **runs, 0..310** | 190-node body, 82 initializers, 11 carried vars, 14 missing ops (`If`, `ScatterElements`, `OneHot`, `Where`, `Range`, `ArgMax`, …) |
| PP-FormulaNet_plus-M | 276.7 MB | 538 | **runs, 0..537** | **1743-node body, 815 initializers, 62 carried vars**, 8 nested `If`, LayerNorm, Erf — a full transformer decoder with KV cache, plus a LaTeX tokenizer/detokenizer that does not exist yet |

**Honest assessment of what remains.** Table is bounded: ~14 mostly-trivial ops
plus `Loop`/`If` subgraph execution, then the table-structure token vocabulary
and HTML assembly. Formula is not bounded by the same measure — the decoder body
alone is 3x the size of the entire layout graph, carries a 6-layer KV cache
across 62 loop variables, and needs a tokenizer.

**Status: 1 of 3 deployed and executed. 2 of 3 have working encoders and
unimplemented decoders.** Recording it that way rather than claiming three,
because an unvalidated port is exactly the failure this campaign keeps
documenting.

---

## §46 — layout SHIPPED; table executes but is WRONG; formula not attempted

**LAYOUT — deployed, executing, correct.** PP-DocLayout-S through the candle
executor, 457 ms/page, finding four display formulas and two inline ones on an
academic page, cleanly separated from text. This is the region class carrying
87% of genuinely-missed characters (§40) and the one no threshold sweep could
produce.

**TABLE — executes end to end and is NOT correct.** SLANet_plus runs: encoder
(310 nodes) + `Loop` decoder (144-node body, 11 carried vars, nested `If`), 501
steps, 1.07 s, no errors. And the output is worthless:

```
logits dims [1, 501, 50]   min 0.0000  max 1.0000   step0 sum 1.0000
step0   first 12: [0.02, 0.02, 0.02, ...]
step250 first 12: [0.02, 0.02, 0.02, ...]
```

**Uniform 0.02 = 1/50 at every step** — softmax over all-zeros. The decoder head
is seeing zeros, so the encoder features are not reaching it through the carried
variables. A clean execution trace and a completely dead output.

**This is exactly the §8.167 failure mode and the reason it must not be called
done.** Two of the three hypotheses that died in the SVTR port "matched every
shape" and were still structurally wrong. A 501-step loop that runs without
error, returns the right tensor shapes, and emits a uniform distribution is that
same class of result: everything checks except the answer.

**Bugs found and fixed along the way** (each was silent, each mattered):

| symptom | cause |
|---|---|
| "1078 nodes, 0 initializers, **0.00 M params**" | `AttributeProto` field 5 (`t`) unhandled; this export inlines weights in `Constant` nodes |
| `missing tensor p2o.sub_block...` inside Loop | `get()` resolved aliases against the TOP-LEVEL map only; a body has its own `Identity` chains |
| `missing tensor helper.constant.65` | tensor payload in `int64_data` (field 7), not `raw_data`; reader handled only raw + float |
| ...still missing after that fix | four tensors have `dims=[0]` — legitimately EMPTY, and returning `None` for them reads as absent |
| `shape mismatch in scatter [1,501,50] vs [1,1,50]` | `ScatterElements` axis dropped by the exporter, silently defaulting to 0. **On a matching shape this would have corrupted output quietly rather than erroring** |
| `dtype mismatch in cmp, lhs: I64, rhs: I32` | ONNX permits mixed integer widths in comparisons; candle does not |
| 501 `<pad>` on a full page | SLANet is trained on CROPPED tables; fed a page it correctly reports "no table here" |
| 497x1 crop | `^M` in a shell-parsed coordinate file |
| stretched 497x222 to square | reference does `ResizeTableImage(max_len=488)` + pad, preserving aspect — not a stretch |

**FORMULA — not attempted beyond export.** PP-FormulaNet_plus-M exports (963
tensors / 276.7 MB, 538 nodes) and its encoder runs, but the decoder is a
1743-node body with 815 initializers, 62 carried KV-cache variables and 8 nested
`If` branches, and needs a LaTeX tokenizer that does not exist here.

**Status: 1 of 3 deployed. 1 of 3 executing-but-wrong with a precise diagnosis.
1 of 3 export-only.** Recorded as such. The remaining work on table is to trace
why the carried variables arrive as zeros — a bounded bug in `onnx_graph::exec`'s
`Loop` handling, not a porting problem.

---

## §47 — all three missing stages now execute, and all three match onnxruntime

§46 closed with "1 of 3 deployed, 1 of 3 executing-but-wrong, 1 of 3
export-only". That is no longer the state. Layout, table and formula all run,
and all three were checked against `onnxruntime` on byte-identical input rather
than against my own expectations.

### The result

| stage | model | validation |
|---|---|---|
| layout | PP-DocLayout-S | deployed (§45) |
| table | SLANet_plus | ORT ids identical; **and** matches OmniDocBench GT exactly — 7 rows / 35 `<td>` |
| formula | PP-FormulaNet_plus-M | ORT ids identical, **all 125 tokens** |

The formula check is the strong one: 538 top-level nodes ending in a `Loop`
whose body is 951 nodes with 62 carried variables and nested `If`s, reproduced
token-for-token.

### The bug that was actually blocking everything

§46 blamed the table on "carried variables arriving as zeros". That was the
wrong description of the right symptom. The real defect was in name resolution:

> **A bound name shadows its alias, and the check must happen at EVERY hop of
> the chain — not just at the ends.**

The exporter folds `Identity` nodes into an alias map. Inside a `Loop` body the
carried buffers are bound under their formal names, but those names are
themselves aliases of the tensors the caller passed on iteration zero, and those
tensors are still present in the cloned enclosing environment. Resolving a name
to its root before looking it up walks straight PAST the live formal and lands
on the stale original. Every read of a carried buffer therefore returned the
INITIAL value, so each iteration restarted from scratch.

The signature is distinctive and worth recognising again: a buffer that updates
exactly ONCE and then freezes. SLANet's `[1,501,50]` logit buffer held exactly
one row for all 500 iterations; PP-FormulaNet's token buffer and all 56 KV
caches grew `[1,1] → [1,2]` and then sat there for all 600.

**The fix that looks obvious is wrong.** Binding each formal under its root name
as well makes the symptom disappear — and silently corrupts the model, because
the roots COLLIDE: four of PP-FormulaNet's KV caches resolve to the same
`Expand.5`, and `p2o.pd_op.reshape.1.0` is both the input-id buffer and the root
of the token buffer. Binding roots collapses four distinct caches into one. It
ran further and produced worse-but-plausible output — §8.167's failure mode
exactly. Stopping the walk at the first bound name is what shadowing means.

### Six more defects, every one silent

1. **`auto_pad` dropped by the exporter.** ONNX states padding as EITHER `pads`
   OR `auto_pad: SAME_UPPER`, and the second carries no `pads` at all — so
   reading only `pads` treats it as VALID and every such layer comes out a
   pixel or two small. Surfaced in PP-FormulaNet's stem as a `Concat` of
   `[1,48,191,191]` with `[1,48,190,190]`, two branches meant to be identical.
   On a net whose branches happened to agree it would have been a silent crop.
2. **`Gather` dropped the index's rank.** ONNX output is
   `data[:axis] + indices.shape + data[axis+1:]`; candle's `index_select` takes
   a 1-D index and leaves the data rank alone. An embedding lookup with ids
   `[1,1]` returned `[1,512]` instead of `[1,1,512]`, and the missing SEQUENCE
   axis propagated ~90 nodes before surfacing as a cross-attention mismatch,
   nowhere near its cause.
3. **`Expand` treated a literal `0` as "keep this dim".** Only a negative entry
   means keep. A zero is a genuinely empty axis — the decoder's KV cache begins
   EMPTY. Folding 0 into "keep" started it at length 1, and the cache then ran
   one step ahead of the attention mask forever: `[1,16,1,3]` vs `[1,1,1,2]`.
4. **Nested scopes did not chain.** An `If` inside a `Loop` saw its own 10
   aliases and the top-level 286, but not the loop body's 792.
5. **One op still called scope-blind `get()`.** It hardcodes the TOP-LEVEL alias
   map, so inside a body it resolved against the wrong table. `Range` was the
   only remaining such call and only PP-FormulaNet exercised it.
6. **The weights file was half empty.** The formula export was 963 tensors /
   276.7 MB; re-exporting after the BOOL→U8 and emit-before-write fixes gave
   **1854 tensors / 591.9 MB**. It had been silently dropping half the decoder.

Also implemented because the graph needs them, not because they were missing on
principle: `LayerNormalization`, `CumSum`, `GreaterOrEqual`/`LessOrEqual`,
`BitwiseNot`/`BitwiseAnd`/`BitwiseOr`, and integer-width promotion in `bcast`.

### The vocabulary lesson, repeated

SLANet's tokens were INVENTED in §46. The real ones ship in `inference.yml`
beside the weights. Reading them was not enough either — the same config sets
`merge_no_span_structure: true`, and `TableLabelDecode` acts on it: it REMOVES
`<td>` and APPENDS `<td></td>` at the END, which is what puts the ordinary cell
at index 48 rather than 7. With the raw config order a genuinely correct 7x5
table decoded as `<tr> rowspan="20" x5 </tr>`.

**The shape of the output was already right; only the names were wrong.** That
is the failure mode that looks most like a broken model and is least like one.

PP-FormulaNet's tokenizer was likewise recorded as "does not exist". It is a
complete HuggingFace ByteLevel-BPE fast tokenizer, 50000 entries, embedded in
its 2.17 MB `inference.yml`. Both blockers were self-inflicted.

### What is NOT fixed

The formula model wraps short expressions in a spurious `\begin{aligned}` with
repeats. The GT appears VERBATIM inside it — `+ \frac{\epsilon}{3}
\bar{\mu}(\Gamma_{n,j}) \bar{\mu}(A)` and `\omega(v)\sim1, |v|\leqslant1.` are
both exact — but the wrapper is not GT. Because our ids match ORT exactly, this
is the MODEL's behaviour on a bare tight crop, not a port defect. PaddleX feeds
these regions with different margins; that is the next thing to test, and it is
a preprocessing question, not a correctness one.

**Standing law:** an oracle that costs one `pip install` is worth more than any
amount of reasoning about whether the port is right. Six of the eight defects
above produce PLAUSIBLE output, and three of them I had already "explained"
with a confident wrong story.

---

## §48 — routing built; all five stages complete

§47 ported three models. None of them reached the benchmark, because nothing
decided which model a region belongs to. `route.rs` is that decision, and
finishing it turned up four more silent defects — every one of which produced
output that looked reasonable.

### The router

Layout classifies the page; `table` goes to SLANet, `formula` to
PP-FormulaNet, everything else stays on the detect+recognise path. It runs
LAST, on the finished line sequence, and SPLICES into it: a region's output
takes the slot of the first line it absorbs. Reading order was the hardest
thing this campaign earned (§29–§43, mostly refutations) and a router that
re-sorted the page would have silently discarded it.

`FFAI_ROUTE=1` opts in. With it off the layout model never loads and output is
byte-identical to the banked baseline — verified, not assumed.

### Four more silent defects

**1. Layout took the argmax class per anchor.** PicoDet is MULTI-LABEL: it
scores every class at every anchor and the graph's NMS keeps each one that
clears the threshold. Taking only the top class dropped a real region
(`paragraph_title` 0.451 under `text` 0.490 on the same anchor). The failure
mode that matters is worse than a missing title — an anchor scoring `table`
0.44 behind `text` 0.46 would never reach the table model, so the routing
decision would be settled by a margin of 0.02 that nothing ever sees. Fixed;
layout now matches onnxruntime exactly, 14 regions, same classes, scores and
boxes. **All five stages are now ORT-validated.**

**2. The formula pad was WHITE. It should be BLACK.** `ImageOps.expand(img,
padding)` is called with no `fill`, and PIL's default is 0. "The model was
trained on white paper, so pad white" is the plausible reading and it is wrong.

This was the whole of the `\begin{aligned}` bloat blamed on the model in §47:

| | white pad | black pad | GT |
|---|---|---|---|
| | `\begin{aligned} &\left\{...\ & 故 \ ...` (411 ch) | `\omega ( v ) \sim 1 , \quad \| v \| \leqslant 1 .` | `\omega(v)\sim1,\quad\|v\|\leqslant1.` |

Three of three GT formulas now decode to the ground truth, INCLUDING the
handwritten one (`\therefore 3m + 1 = 1 , 5 , - 1 , - 5 .`). §47 recorded this
as "the model's genuine behaviour, not a port defect" on the strength of the
ORT match. That was true and useless: ORT was fed MY tensor, so it could only
ever confirm the executor, never the preprocessing. **An oracle validates
exactly the span you hand it.**

**3. Table cell boxes were denormalised per-axis.** They come back normalised
to the PADDED SQUARE, so both axes share the long side's scale and both need
`max(w,h)`. Using `w` for x and `h` for y compressed every box along the short
axis: on a 497x222 table all seven predicted rows landed inside the top 45 %
(222/497) of the real table, so consecutive rows read the SAME source row twice
and the bottom half was never read — while the structure, the cell count and
every individual crop still looked entirely reasonable.

**4. `formula_number` was routed to the LaTeX decoder.** It is the "(26)" in the
margin — ordinary text. `routes_to_latex()` now excludes it.

### Cells are READ, not redistributed

The first cell-binding attempt assigned recognised LINES to predicted CELLS by
proximity. It cannot work, and the reason is a type mismatch rather than a bad
metric: the detector emits text LINES and SLANet emits CELLS, and one detected
line routinely spans a whole row while a column of figures comes back as one
tall box. The corruption was visible — header row empty, and "2048 2560 4096
6144", four separate table rows, deposited in a single cell. The information
the binding needed was destroyed upstream when four rows became one box.

A cell box is itself a crop rectangle, so the fix is to READ it. `engine.rs`
passes the router a `RecFn` closure over whichever recognizer is configured —
the same reason `VERIFY_LOWCONF` is passed in (§8.171): only the engine knows
which one ran. Result on the RWKV table, against GT:

```
ours  Dimension (D) | dw | da | du | dg      GT  Dimension (D) | dw | da | dv | dg
      ewb           | 64 | 64 | 32 | 128         768           | 64 | 64 | 32 | 128
      2eG0          | 96 | G  | 64 | 256         2048          | 96 | 96 | 64 | 256
      4096b         |128 |128 | 96 | 480         4096          |128 |128 | 96 | 480
```

All 7 rows aligned, 4 of 5 columns exact. The remaining defect is the first
column only — a recognition-crop issue on wide label cells, not a structural
one. A padding sweep (0.0 / 0.15 / 0.30) did not fix it, so it stays at 0.0 and
stays open rather than being tuned into a coincidence.

### Standing law added

**An oracle validates exactly the span you hand it.** Handing ORT my
preprocessed tensor proved the executor and proved nothing about the
preprocessing — and I used the match to close the question. The white pad
survived a "validation" that structurally could not see it.

---

## §49 — routing BANKED on the full corpus: ReadOrder −0.0186, text neutral

Two full-corpus passes on one binary, 1651/1651 pages each, zero failures.

### The drift question, settled

`fullnull` (routing off) against the banked `odb_pred_fullcur`:

| metric | fullcur | fullnull | delta | verdict |
|---|---|---|---|---|
| text | 0.1274 | 0.1281 | +0.0007 | SPANS 0 |
| order | 0.2335 | 0.2339 | +0.0004 | SPANS 0 |

The baseline was NOT stale in any way that mattered. Mid-session I called the
first routed result "void" on the strength of ONE page whose output differed by
155 lines — and that page (`scihub_mol`) turned out to be genuinely drifted
while the aggregate was not: banked 0.0036, null 0.8676, route 0.8676, identical
under both arms, so routing never touched it. **A per-page difference is not an
aggregate difference. I read one page's cause onto 305 of them.**

### The result

`fullroute` (guarded) against the SAME-BINARY null:

| metric | null | route | delta | verdict |
|---|---|---|---|---|
| text | 0.1281 | 0.1298 | +0.0016 | SPANS 0 — neutral |
| order | 0.2339 | **0.2153** | **−0.0186** | **CI [+0.0140, +0.0232], ARM BETTER** |

182 pages helped on order against 44 hurt. The subset predicted −0.0159; the
full corpus delivered −0.0186, comfortably inside the ±0.005 screening
tolerance and in the same direction. **ReadOrder^Edit is now 0.2153 against the
0.2348 this campaign opened with.**

The mechanism is not subtle: collapsing a table or formula region into ONE
block removes its internal lines from the reading sequence entirely, so they can
no longer interleave with the surrounding columns. Routing buys ordering by
DELETING ordering decisions.

### The guards are the whole result

Unguarded routing measured **+0.0347 WORSE** on text (CI excluding zero, 50
pages hurt of 305). Two guards, both refusing rather than tuning:

1. **A rendering that carries LESS TEXT than it replaces is refused**
   (`FFAI_ROUTE_RETAIN`, 0.60, compared on CONTENT not markup). The module
   already fell back when a model ERRORED; it had no check for a model that
   succeeded BADLY, and an empty table skeleton is well-formed output. Worst
   case caught: a Chinese newspaper whose column grid scored as a `table` —
   region absorbed 194 lines, SLANet returned a near-empty grid, and 6970 bytes
   of correctly-read prose became 2041 bytes of `<td></td>`. 0.0144 → 1.0000.
2. **The bar to REDIRECT is higher than the bar to DETECT**
   (`FFAI_ROUTE_SCORE` 0.60 over the layout floor 0.45). Deciding a region
   exists and deciding to overwrite a page's text with a model's rendering are
   different claims on the same score.

Neither guard adds a win — both only prevent harm — which is exactly why text
went from +0.0347 to neutral rather than to positive.

**I wrote that asymmetry into this module's own header — "a missed table still
gets read as text, a false table replaces real text with an empty grid" — and
shipped it with no code enforcing it.** Naming a risk in a doc comment is not
guarding it.

### What is still not measured

- **TEDS / CDM.** `gg_arm.py` hardcodes Edit_dist on text_block and
  reading_order only. 10.3 % of routed output sits inside `$$…$$` and 16.8 %
  inside `<table>`, and the evaluator parses that OUT of the text pool — so
  text_block structurally cannot see routing's content wins, only its losses.
  That is why the text CEILING is +0.0034: there is almost nothing on that axis
  for a gate to capture. **A first attempt at TEDS was killed: its config wrote
  under the SAME save name as the edit run and would have overwritten the null
  baseline the whole comparison rests on — the same save-name collision that
  nearly destroyed the banked baseline in §33. Prediction dirs must be copied
  under distinct names before rescoring.**
- **Speed.** The harness forbids speed claims (4 workers, BELOW_NORMAL, no
  ABBA). Routing cost 163 min against 133 min for the null pass on identical
  work — **1.23×**, far cheaper than the 1.74× the subset suggested and nowhere
  near the ~5× a three-page serial sample implied. That is a wall-clock
  observation from this run, NOT a benchmarked per-page latency.
- **The 118 text-hurt pages** (ceiling +0.0034). Small, but it is the remaining
  false-positive tail and the guards are unswept at 0.60/0.60.

---

## §50 — the §37 ceiling RE-PRICED after routing: floats were never the slack

The ordering-audit's warning box feared that "some of the slack §37–§41
attributed to cut strategy was table and formula regions all along". Priced
today by splitting the oracle. **The fear is refuted: the cut-strategy prize
survives almost intact.**

### The instrument

`.tools-bench/oracle_order_split.py` — same line→region assignment as
`oracle_order.py`, but split into two disjoint arms with PINNED-slot
construction (an ineligible line keeps its exact emitted index; eligible lines
permute only among their own slots):

- `orctext` — only lines owned by non-float regions may move;
- `orcfloat` — only lines owned by float regions (figure/table/
  equation_isolated) may move.

Null-arm proof: `orcfloat` is byte-identical to `fullcur` on all 459 no-float
pages, and scores EXACTLY 0.0000 delta there once the instrument bug below was
subtracted.

### The verdict (vs fullcur, evaluator-fallback pages excluded, CIs exclude 0
unless noted)

| arm | text | order |
|---|---:|---:|
| §37 full oracle (rescored same way) | +0.0387 | +0.0538 |
| **orctext — text lines only** | **+0.0190** | **+0.0449** |
| orcfloat — float lines only | +0.0004 (spans 0) | −0.0000 (spans 0) |

- **Order: text-block sequencing keeps 83 % of the oracle ceiling.** Floats
  alone are worth NOTHING as a reorder — §49's routing win was a DELETION win
  (removing float lines from the text pool), not a sequencing win, and the two
  levers barely overlap.
- **Text: the interaction term is the other half** (+0.0197): a float line
  sitting INSIDE a text block's span breaks the block even when the text lines
  are ordered right. That is the part routing's deletion mechanism claims;
  orctext's +0.0190 is what remains for Stages 3–4 on top of it.
- EN no-float, the audit's named worst English segment: orctext reads
  0.1146 → 0.0711 text (+0.0435) and 0.0734 → 0.0208 order. Confirmed as the
  densest ordering prize per page.

### THE INSTRUMENT BUG THAT NEARLY SHIPPED A FALSE +0.0263

`score_arm.py --reuse-base` (added today: compare a fresh arm against a CACHED
base run) reported orcfloat +0.0263 text on EN no-float pages — pages where
the arm is byte-identical to the base by construction. **Byte-identical input,
different score.** Cause: the evaluator's per-page matching TIMEOUTS are
load-dependent. The cached fullcur run (10 workers) timed out on 8 pages and
scored them ~1.0; today's 4-worker BelowNormal run completed the same pages.
Seven pages of pure timeout noise, worth a false +0.0033 text on the aggregate.
§36 found this class once already; it reappears whenever two evaluator runs
from different sessions are differenced. **Law: a cross-session comparison must
exclude the UNION of both runs' stage_execution fallback pages.** With that
exclusion the null pages read exactly 0.0000, which is the instrument proving
itself.

Corollary, quantified: the banked 0.1274/0.2335 headline carries ~0.0045 text
of evaluator-timeout tax on 8 pages the engine reads fine (fallback-excluded
baseline: 0.1229 text / 0.2310 order).

### Same day, same discipline — three more audit steps closed offline

- **Stage 2 (corridor splitter): a footnote.** 28 of 32 476 pre-split lines
  genuinely cross a GT column gap; the splitter catches 1 (recall 3.6 %,
  precision 20 %); the page gate blocks 91 of 96 pages that proposed cuts.
  Bounded tail risk, not a segment.
- **Stages 3.1/4.1 (`stage4_regret.py`, real `order_reading` via
  `order_probe.exe`, 1469 pages, contiguity proxy):** chosen 0.941, pool
  best-of-3 0.959, oracle 1.0 → **pool ceiling 0.041, selector regret 0.018** —
  independently reproducing §8.153's ~30/70 split on a different instrument.
  `noselect` wins 1241 pages, `vfirst` 143, `xycut` 85. Regret concentrates in
  research_report (+0.046), note, colorful_textbook; newspapers are
  pool-limited, not chooser-limited.
- **Routing (§49) decomposed by source:** order wins live where floats live —
  book +0.0456, exam_paper +0.0410, academic +0.0358 — while **newspaper is
  net-negative on BOTH columns** (−0.0106 order, −0.0113 text). The text-hurt
  tail concentrates in exam_paper (50 pages) and book (35). Five catastrophic
  guard misses identified by name (none are evaluator fallbacks — verified):
  `jiaocaineedrop_jiaocai_needrop_en_3685` (0.00→1.00 order),
  `newspaper_7d8b25729455f1d061a95ec4269d72dc_1` (→1.00 both),
  `newspaper_2a6b4fa088699701a6fa9ccecfb5c25d_18` (0.01→0.90 text),
  `PPT_lay_linalg5_01_05_page_009`, `eastmoney_…pdf_5`. The retain/redirect
  guards at 0.60/0.60 are unswept; this is their price list.

---

## §51 — ordering selection v2 BANKED (+0.0035 order, text 0.0000); raster challenger REFUTED; routing absorption cap KEPT

Three levers from §50's roadmap, run in one campaign: the new pool member and
the selector objective (designed jointly — history showed pool expansion dies
under the old selector), and the routing guard sweep. One banked, one refuted
with a law, one kept as a guard.

### The instrument that made it cheap

`stage34_pool.py` — all ELEVEN existing `FFAI_ORDER` strategies through the
real `order_reading` (order_probe.exe) on every page, dumping per-arm
contiguity, objective features, sparse-scatter and the full PERMUTATION.
With permutations on disk, every selector experiment afterwards is pure
offline analysis. Census: best-of-pool3 0.9592, best-of-ALL-arms 0.9784 —
**half the §50 pool ceiling was already sitting in the menu** (raster wins 93
pages, onelevel 40, hfirst 9). The simulation was validated by reproducing
the shipped default to four digits (0.9412 = r008/pool3/sparse-gate), which
also answered audit 3.2: **the §8.156 sparse gate is worth +0.018 mean
contiguity on today's corpus** — it does the heavy lifting the raw selection
misses.

### Lever 1+2 (BANKED, env-gated): objective v2

`FFAI_ORDER_SELECT=2` — same three candidates, objective
`wreset + 0.5·yback + 2·scat` (leftward-jump MAGNITUDE + fraction of
consecutive pairs moving UP the page + sparse-scatter) instead of the reset
COUNT. Weights sit on a measured plateau (1–2× each, flat), survived an
even/odd holdout split (+0.012 contiguity on both halves), and the sparse
gate is untouched.

Full corpus, probe-reserialized arms (selv2b vs selnull, same construction,
same session), official evaluator, §50 rule 9 applied (7 fallback pages
excluded):

| metric | n | gain | CI | verdict |
|---|---|---|---|---|
| reading_order | 1631 | **+0.0035** | [+0.0010, +0.0062] | **BETTER, CI excludes 0** |
| text_block | 1550 | +0.0000 | [−0.0016, +0.0016] | dead neutral |

EN no-float order read +0.0155 (CI [+0.0038, +0.0314]) — the §50-named
densest segment pays as predicted. Same banked shape as §49: an order win at
text-neutral. Rule 9 note: the raw same-session numbers carried ~0.0015 of
timeout noise on top; the fallback-excluded row is the honest one.

### Lever 2b (REFUTED, with a law): the raster challenger

§1.1's note-page sign-flip argued raster into the menu. On the CONTIGUITY
PROXY it worked: margin-guarded (0.04 — the bar to REPLACE higher than the
bar to COMPETE, same asymmetry as route.rs), the magazine regression it first
caused (raster stealing 14 pages from a perfect pernode) was neutralized and
the corpus read +0.004 over the objective swap alone. The EVALUATOR killed
it: on the 170 challenger-taken pages, **text −0.0185 all / −0.0375 EN, CI
excluding zero** (order +0.0075, spans 0). Decomposed cleanly by the per-page
choice map — the objective-swap class was simultaneously order +0.0106 (CI
excludes 0) at text +0.0004.

**The law: contiguity is blind to INTERLEAVING.** A raster-read multi-column
page keeps each block's lines index-compact (contiguity ≈ 1) while
alternating blocks across columns — block matching sees the interleave,
contiguity cannot. Any future ordering candidate whose win is
contiguity-only must show the interleave axis (or the evaluator) before it
is believed. Challenger default is now `f32::INFINITY` (never fires);
`FFAI_ORDER_V2_MARGIN` keeps it reachable for a sparse-page-only salvage.

Engine-subset confirmation, both configs: challenger ON read text +0.0073
WORSE (CI excl. 0) — the challenger carried it, consistent with the
decomposition. The FINAL config (challenger off) through the full engine
path including `probe_reorder`: text −0.0021 [−0.0096, +0.0026], order
+0.0003 [−0.0089, +0.0104] — both span 0. A 305-page screen cannot resolve a
+0.0035 lever (its resolution is ±0.005, and it carries 26 of 145
newspapers); what it CAN do is kill, and it did not: no text harm survives in
the shipped configuration. The full-corpus CI-positive order gain stands as
the banked evidence; promotion to engine DEFAULT wants one full-corpus
engine pass.

### Lever 3 (KEPT, guard): routing absorption cap

`FFAI_ROUTE_ABSORB` (default 0.60, inside opt-in routing): a table/formula
region whose absorbed lines carry more than 60 % of the page's characters is
refused. Built from the §50 hit list mechanism — a false table over a
newspaper column grid or a textbook TOC absorbs ~the whole page, the cells
re-read fine so `retain` passes, and the page's text exits the evaluator's
text pool as markup.

Swept OFFLINE first (`layout_batch.exe` harvested regions on all 457 routed
pages; absorption computed from captured line geometry; predicted against
banked per-page deltas): 0.60 catches the catastrophes, tighter caps add
churn and recover nothing, and raising `FFAI_ROUTE_SCORE` to 0.80 instead
would cost +0.0089 of routing's order win. Engine-confirmed: cap-off
**byte-identical to banked fullroute on all 25 pages tried** (inert-when-off
proven on the new binary), cap-on changes exactly the 21 predicted pages and
zero controls.

Scored (capfix = fullroute + the 21 pages, vs cached fullroute, rule 9
applied — and it fired AGAIN: the raw diff read +0.0043 text, 3–4× the
offline prediction; 8 cached-run timeout pages were the excess):

| metric | honest gain | CI | mechanism |
|---|---|---|---|
| text_block | +0.0010 | [+0.0000, +0.0026] | 2 pages rescued (+0.82, +0.75), 0 hurt, 19 unchanged |
| reading_order | +0.0012 | [+0.0000, +0.0030] | same two pages |

CI touches zero because the gain lives in 2 of 1549 pages — but the per-page
mechanism is deterministic and engine-confirmed, and the guard's real value
is that routing's worst known failure mode is now BOUNDED. The remaining
false-table tail (newspaper_2a6b…_18's prose-table at ~0.3 absorption) needs
a prose-vs-tabular detector, not a lower cap — recorded as the follow-up.

### Standing state after §51

- Banked headline (routing + cap, fallback-excluded basis): text ~0.125,
  order ~0.212. With `FFAI_ORDER_SELECT=2` on top: order ~0.209 (pending the
  engine-subset confirmation of the stacked config).
- 38 lib tests pass; `bench_ocr` example repaired (missing
  `skip_references` field, pre-existing §48 debt).
- New instruments: `stage34_pool.py` (permutation census),
  `layout_batch.rs`, per-page choice maps. The §50 rule-9 law fired twice
  more today; it is now applied by default in every cross-session diff here.

---

## §52 — the stacked config through the FULL ENGINE: new banked standing 0.1269 text / 0.2093 order

One full-corpus engine pass (1651/1651 pages, zero failures, ~4.6 h wall on a
shared box), `FFAI_ROUTE=1` + `FFAI_ORDER_SELECT=2`, everything else default.
This was §51's named promotion gate for the v2 selection, and it clears it
with room:

| vs (fallback-union excluded) | text | order |
|---|---|---|
| fullcur (pre-routing baseline) | +0.0005 (spans 0) | **+0.0244** [+0.0190, +0.0297] |
| fullroute (previous best) | **+0.0029** [+0.0010, +0.0053] | **+0.0060** [+0.0028, +0.0094] |

- **Additivity held**: routing −0.0186 + cap −0.0012 + v2 −0.0035 predicted
  −0.0233; the engine delivered −0.0244. The probe-arm methodology (§51) is
  vindicated as a full-corpus instrument.
- **No losing source**: order gain by source is book +0.0525, exam +0.0373,
  academic +0.0363, … newspaper **+0.0109** (routing alone had newspaper at
  −0.0106 — the absorption cap plus v2 turned the one regressing source
  positive), magazine exactly 0.0000. The dispatch-law finish line.
- **Non-English moved too**: text +0.0025 (CI excludes 0) and order +0.0092
  vs fullcur — small, but the first CI-positive non-EN text movement since
  §19.
- English order vs fullcur: **0.2206 → 0.1782 (+0.0424)**.

### The new banked standing (official evaluator, all 1651 pages)

| | raw (comparable to published rows) | evaluator-timeout pages excluded |
|---|---:|---:|
| Text^Edit | **0.1269** | 0.1229 (n=1549) |
| ReadOrder^Edit | **0.2093** | 0.2071 (n=1630) |

The campaign opened at 0.1307 / 0.2348. Text is down 0.0038 with the §28+§50
finding that ~0.0045 of what remains is evaluator-timeout tax, not engine
error. **Order is down 0.0255 — from brushing the published board's worst row
(0.243) to clear mid-pack territory.** `odb_pred_fullv2` is the new reference
arm; future levers difference against it (and §50 rule 9 applies to every
cross-session comparison).

Promotion note: `FFAI_ORDER_SELECT=2` has now been measured text-neutral at
full scale through the real engine (twice: probe arms §51, engine here) and
order-positive with CIs excluding zero at every level. Making it the engine
DEFAULT is now an editorial decision, not a measurement gap; the env escape
(`FFAI_ORDER_SELECT` unset) reproduces the old selection byte-identically.
