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
