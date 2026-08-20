# Carmenta ordering — stage-by-stage audit plan

The reading-order path is the largest priced lever in the campaign
(§37: oracle **+0.0378 text / +0.0532 order**; §41: the implementable half of it,
line grouping, is worth **+0.0337 text** and was scored end to end). Six
implementations have failed. This plan audits the path one stage at a time so the
seventh attempt is aimed by measurement rather than by guess.

> ### ⚠ BASELINE MOVED — §49, 2026-08-16
>
> **Order is now 0.2153, not 0.2335.** Region routing (`FFAI_ROUTE=1`,
> §47–§49) banked **−0.0186 order** on the full corpus, CI [+0.0140, +0.0232],
> 182 pages helped against 44 hurt, with text neutral (+0.0016, spans zero).
> Any arm below must difference against **`odb_pred_fullnull`** (text 0.1281 /
> order 0.2339, routing OFF) or **`odb_pred_fullroute`** (text 0.1298 / order
> 0.2153, routing ON) — matched to whether the arm runs with routing.
>
> **That win came from OUTSIDE this plan.** Routing is not one of Stages 0–6: it
> collapses a table or formula region into a SINGLE block so its internal lines
> leave the sequence altogether. It did not sequence anything better; it removed
> ordering decisions. The consequence for this document is that the ceilings
> below were measured with those lines still in the sequence, so **some of the
> slack §37–§41 attributed to cut strategy was table and formula regions all
> along, and nobody has separated the two.** Re-price before executing Stages 3–4.
>
> ✅ **RE-PRICED 2026-08-19 (§50) — the fear is refuted.** The oracle was split
> (`oracle_order_split.py`): reordering ONLY float-region lines is worth
> **nothing** (+0.0004 text / −0.0000 order, CIs span 0); reordering ONLY text
> lines keeps **+0.0190 text / +0.0449 order** of §37's +0.0387/+0.0538
> (fallback-excluded rescore). Routing's §49 win was a DELETION win, not a
> sequencing win — the two levers barely overlap, and **Stages 3–4 keep their
> prize**. The remaining gap between orctext and the full oracle (+0.0197 text)
> is float-lines-inside-text-blocks, which is routing's territory, not this
> plan's. Cross-session scoring trap discovered doing this: two evaluator runs
> from different sessions MUST exclude the union of both runs'
> `stage_execution` fallback pages — the matching timeout is load-dependent and
> 7 timeout pages faked a +0.0263 "gain" on byte-identical predictions.
>
> Architectural note: layout runs at `engine.rs:809`, AFTER detection,
> recognition and the Stage 5 reorder — it is effectively Stage 5.5, not a
> pre-Stage-0 step. A post-process can only DELETE units, never repair them.
> Region-first was measured at 30.63% in §8.53 (ten points worse than shipped)
> and is not being revisited; the win is kept as a post-process.

Baseline for everything below: `odb_pred_fullcur`, text **0.1278** / order
**0.2335**, 1651/1651 pages, zero engine failures. *(Superseded for order — see
the box above.)*

---

## How to run any step in this plan

**The gate order is not optional.** Today four levers were refuted in seconds
each, and one wasted a 2-hour engine pass plus a scoring run because it was taken
straight to the evaluator.

1. **Offline first.** `.tools-bench/boxes_fullcur` holds per-line
   `conf/x/y/w/h/text` for all 1651 pages. Any reordering or regrouping idea can
   be tested against it in seconds with no engine pass.
2. **Contiguity gate.** Median lines/span of GT blocks in emitted order, and the
   fraction fully contiguous. Baseline **1.000 / 0.822**. A change that lowers
   full-contiguity does not proceed. This gate has now rejected four levers.
3. **Engine subset.** If the change needs the engine (a new env toggle), run
   20–30 pages of the affected population and re-gate on contiguity.
4. **Scoring run.** `score_arm.py --arm X --base fullcur` only after the gate
   passes. Both columns, bootstrap CIs, population-matched.
5. **Null arm.** Any transform harness must reproduce the baseline byte-identical
   with the change disabled. `reserialize.py --identity` does; that is what
   licensed attributing §37's whole delta to ordering.

**Standing traps, all hit at least once:**
- Generalising from a small sample. §37's "median edit 0.023" came from four
  newspaper pages and was wrong for the corpus (§40).
- Oracles that succeed by chance. LCS on CJK inflated a ceiling 3.2x (§35).
  An oracle must be realisable by some implementation.
- Correlation read as mechanism. The 250-block cliff was real and causally
  backwards (§34); the CJK leading difference was real and acted through nothing
  (§38).
- Measurement failures scored as quality. 10 evaluator timeouts carried 21.6% of
  block edit mass (§36). Subtract `stage_execution.json` fallbacks before ranking.
- Blast radius. ~75% of blocks are already near-perfect; a global change is
  paying for the minority with the majority.

---

## Stage 0 — Detection (DBNet → `DetBox`)

`mobiledet_input` ([image.rs:490](../../crates/ffai-carmenta/src/image.rs)) →
`MobileDet` → connected components.

**Established.** Healthy on the population whose problem is ordering: 75% median
GT-block area coverage on horizontal CJK, **0 of 493** blocks under 20% covered,
box area 37.7% against GT 37.2% on the Chicago Tribune page. Calibration point:
GT polygons are region rectangles including inter-line whitespace, so **0.77 is
what "fully covered" looks like** — measured on blocks scoring under 0.10.

**Broken elsewhere.** 132 blocks under 40% covered (29.9% of not-read
characters); 51 with nothing. Ceiling if all thin/no-coverage blocks were
perfect: **+0.0217** — but see the re-measurement below, which puts that number
in doubt.

**⚠ RE-MEASURED 2026-08-16 — the two named pages are FIXED, and one was never
broken in the way this section claims.**

| page | this plan said | measured now |
|---|---|---|
| `page-3ecc67a1` | **zero** boxes | **76** boxes, 286 chars emitted |
| `page-942ac90d` | **one** box (`九`) vs 8 GT blocks | **310** boxes, 1034 chars |

`page-942ac90d` is §43's vertical-column split working: one-box-per-glyph is
resolved, so "vertical text yields one box per glyph" no longer holds.

`page-3ecc67a1` is the more interesting correction. Its ground truth is **136
characters across 3 text blocks** — we emit **286**. The page is not under-read,
it is **over-read by more than 2x**. It is a sparse vertical-CJK page, and
reading its 286 chars as "still thin for a whole page" (as this plan's framing
invites) is an assumption about page density that nobody checked.

**That is direct evidence for the hypothesis step 0.1 exists to test:** low
GT-coverage may mean GT polygons include inter-line whitespace we correctly do
not cover, not that we are missing text. This section already carries the
calibration — 0.77 is what "fully covered" looks like. If the artifact reading
holds, **the +0.0217 ceiling shrinks or disappears and Stage 0 is largely
finished.** One page is suggestive, not conclusive (generalising from a handful
of pages is the first standing trap in this document). Run 0.1 before spending
anything else here.

**Refuted already.** `FFAI_DB_BIN` / `BOX` / `UNCLIP` sweeps (flat, launchpad §4);
`FFAI_DET_MIN_SIDE` 736→2200 (coverage unmoved on failing pages, §38).

### Exploration steps

- **0.1 Is thin coverage vertical or horizontal?** ✅ **RUN 2026-08-16**
  (`.tools-bench/stage0_coverage.py`, offline, 20 961 GT text blocks over 1551
  pages). **The +0.0217 is CONFIRMED, and the framing around it is wrong.**

  Area coverage cannot separate "missed a line" from "correctly skipped the gap
  between two lines", so the probe scores CHARACTERS instead: what fraction of
  each GT block's text do our boxes actually emit?

  | population | n | median char-ratio | median area_cov |
  |---|---|---|---|
  | all blocks | 20 961 | **0.98** | 0.78 |
  | thin (`area_cov < 0.40`) | 915 | **0.40** | 0.29 |
  | rest | 20 046 | 0.99 | 0.79 |

  Median area coverage 0.78 across all blocks **confirms this section's 0.77
  calibration** — that is what a healthy line detector scores against a region
  rectangle.

  **The artifact reading is REFUTED.** Only 136 of 915 thin blocks are fully
  read; the rest genuinely lose text. Missing characters in thin blocks total
  **58 882 = 1.99 % of all text**, so a perfect Stage 0 on them is worth at most
  **~0.0199** — within noise of the plan's +0.0217. *(I leaned the other way on
  the strength of `page-3ecc67a1` alone. One page is not a corpus — the first
  standing trap in this document, and I walked into it.)*

  **The finding that matters is the one the thin-block framing hides.** Missing
  characters CORPUS-WIDE are **9.3–10.1 % of all text** (297 085 chars; the
  range is strict vs loose box-to-block attribution, so the estimate is robust
  to the rule). Thin blocks are only **~20 %** of that mass. **Four fifths of
  the text we fail to emit sits in blocks whose area coverage looks fine.**

  So `area_cov < 0.40` is a poor targeting instrument: it selects a fifth of the
  problem and calls it the problem. The median block reads 0.98 of its
  characters, so the loss is a concentrated tail that area coverage does not
  find.

  **⚠ AND THE CHARACTER RATIO CONFLATES TWO STAGES.** "Characters emitted per GT
  block" scores DETECTION AND RECOGNITION TOGETHER. A block with perfect boxes
  whose recognizer returns short text looks identical to a block that was never
  detected. Stage 0 owns only the second. Splitting them by whether our boxes
  cover the block's ROWS:

  | bucket | missing chars | share of all text | blocks |
  |---|---|---|---|
  | no boxes at all | 5 912 | 0.20 % | 201 |
  | boxes cover < 40 % of rows | 6 038 | 0.20 % | 168 |
  | **boxes cover the rows, text short** | **236 310** | **7.99 %** | **4 928** |
  | read fine (≥90 %) | 48 825 | 1.65 % | 15 664 |

  > **DETECTION owns 0.40 % of the missing text. RECOGNITION owns 7.99 %.**

  **Stage 0 is finished.** Its true ceiling is ~**0.004** on Text^Edit, not
  +0.0217 — that figure was detection and recognition added together, and 95 %
  of it belongs to the recognizer. Perfect detection everywhere would buy four
  thousandths. There is no knob sweep, no threshold and no probe in this stage
  worth the engine time.

  This also independently reproduces §8.9's conclusion by a completely different
  route: *"our cropping is proportionally BETTER; the CRNN is the deficit."*
  Two unrelated instruments now agree the characters are where the loss is.

  *Caveat on the 7.99 %: box-to-block attribution requires ≥50 % of a box inside
  the block, so text on a boundary can go uncounted. Loosening the rule to any
  overlap moves TOTAL missing 10.05 % → 9.33 %, so the recognition bucket
  survives the rule change and stays dominant. It is not an artifact of the
  attribution.*
- **0.2 Threshold recall on thin blocks only.** ~~`FFAI_DET_TEXT_THR` /
  `FFAI_DET_LOW`~~ — **DO NOT RUN AS WRITTEN. Those are the wrong knobs.**

  Both exist ([boxes.rs:29-33](../../crates/ffai-carmenta/src/boxes.rs)) but
  they wrap `TEXT_THRESHOLD` / `LOW_TEXT`, which are **CRAFT** constants. This
  stage is DBNet on the `mobiledet-svtr` path, where they are never read — a
  sweep would return four identical numbers and "refute" a lever it never
  applied. This is the same trap §43 caught by checking use sites, and the trap
  that burned three hypotheses on `group_lines`, which is likewise not on the
  shipped path (`engine.rs:405`: "DBNet emits text LINES, so there is nothing to
  group"). **The plan preserved the mistake; a knob existing is not evidence it
  is wired.**

  DBNet's real knobs are `FFAI_DB_BIN` / `FFAI_DB_BOX` / `FFAI_DB_UNCLIP`, and
  this section already records them as swept flat. So 0.2 is either unrunnable
  or already refuted. **Treat it as closed unless 0.1 says the thin blocks are
  genuinely unread**, in which case the question is a detector-recall one and
  the sweep needs the DB knobs, on the thin-block pages only.
- **0.3 Vertical-text detection.** ~~Does DBNet produce column-shaped
  probability regions on `page-942ac90d` that our connected-component walk then
  shatters?~~ **CLOSED BY OUTCOME (§43), not by this probe.**

  It was a grouping problem, and `boxes::split_vertical_columns`
  (`FFAI_VSPLIT_ASPECT`, default 4.0) ships the fix: a box taller than 4x its
  width is cut into roughly square glyph cells. `page-942ac90d` went from **one**
  box to **310**, 1034 chars. The threshold is 4.0 rather than the more
  aggressive 2.0 because 2.0 was measured to corrupt **199 tall digits across 92
  pages** while recovering the same vertical text — blast radius priced before
  shipping.

  Note the probe was never run; the fix was found from the other end. That is
  fine for the outcome but leaves the stated QUESTION unanswered, so if vertical
  pages resurface as a defect, the probability-map dump is still the right
  instrument.

---

## Stage 1 — Line identity

> **CORRECTED (§42). There is no grouping stage on the shipped engine.** An
> earlier draft of this plan named `group_lines` the prime suspect and proposed
> sweeping `FFAI_LINE_BACK` / `FFAI_LINE_OVERLAP` and adding an x-adjacency
> condition to its join test. **All three are inert for `mobiledet-svtr`**,
> because `group_lines` is never called on that path.
>
> Caught by measurement before the call path was read: the x-condition was
> built, gated, and swept over four tolerances down to 0.5 line-heights, and
> every arm returned numbers identical to the last digit — contiguity 0.522,
> 444 lines/page, 7338 chars/page. **A code path that cannot move any number at
> any setting is not being executed.** Reverted; engine source is clean and
> verified byte-identical to `odb_pred_fullcur` on 12 pages across 7 doc types.
>
> General form: trace the call path before attributing a defect to a function.
> A comment naming a stage ("the defect is upstream, in line grouping") is
> evidence about the path it was written for, not the path you are running.

**What actually happens** ([engine.rs:405–430](../../crates/ffai-carmenta/src/engine.rs#L405)):

```rust
// DBNet emits text LINES, so there is nothing to group — each region IS a line.
b.sort_by_key(|r| (r.y0, r.x0));                    // RASTER order
let mut b = boxes::split_at_white_corridor(b, &gray, w, h);
let lines: Vec<Vec<DetBox>> = b.into_iter().map(|r| vec![r]).collect();
Ok((boxes::order_reading(lines, w), ...))
```

Line identity on the shipped engine is decided **entirely by DBNet's regions plus
`split_at_white_corridor`**. `group_lines` serves the CRAFT path
([engine.rs:358](../../crates/ffai-carmenta/src/engine.rs#L358)) and orphan boxes
in Composed; its constants matter only if `craft-crnn` returns as a candidate,
which §32 makes unlikely (it loses end-to-end by 3x).

**Where the concurrency finding really belongs.** Newspapers carry a median of
**4** lines sharing a y-band (p90 5; 53% of newspaper pages above 3) against a
corpus median of 1. That measurement stands, but it does not indict grouping — it
indicts the **input handed to ordering**. The pre-sort is `(r.y0, r.x0)`, i.e.
RASTER order, which on a page with 4 lines per y-band is maximally interleaved by
construction. De-interleaving it is the entire job of `order_reading`, and
§37–§41 measured that it fails there. Concurrency is therefore a **difficulty
predictor for Stages 3–4**, not a Stage 1 defect.

### Exploration steps

- **1.1 How much interleaving is INHERITED rather than caused?** Score the raster
  pre-sort `(y0, x0)` against GT contiguity, then the final emitted order. The
  difference is what `order_reading` actually contributes. Raster ≈ final on
  newspapers means ordering is doing nothing and Stages 3–4 are the whole target;
  final much better but still poor means the strategies help and fall short.
  *Offline — `boxes_fullcur` holds the final order and raster is a re-sort of it.
  Cheapest high-information probe in the plan. **Run this first.***
- **1.2 Region↔GT-line correspondence.** For each GT text block, do our regions
  TILE it — one per text line — or split and merge lines? Compare region count in
  a block against its expected line count (GT chars ÷ chars per emitted line).
  Splitting inflates the units ordering must sequence; merging across a gutter is
  Stage 2's problem. *Offline, minutes.*
- **1.1 How much interleaving is INHERITED?** ✅ **RUN 2026-08-16**
  (`stage1_contiguity.py`). Instrument VALIDATED against this plan's recorded
  baseline: emitted median 1.000 / fully-contiguous 0.828 vs the stated 0.822.

  | | median | mean | fully contiguous |
  |---|---|---|---|
  | EMITTED | 1.000 | 0.904 | 0.828 |
  | raster `(y0,x0)` | 0.500 | 0.562 | 0.285 |

  **`order_reading` contributes +0.342 mean contiguity — it is NOT inert**, and
  this section's hypothesis ("raster ~= final on newspapers means ordering is
  doing nothing") is REFUTED: newspapers are where it works HARDEST, lifting
  0.316 -> 0.870. They stay second-worst because they start hardest.

  **The finding is `note`: ordering makes it WORSE than the order it was handed**
  (raster 0.940 -> emitted 0.785, a fourth content sign-flip in this component).
  Localised: `probe_reorder` contributes EXACTLY 0.000 there, so it is
  `order_reading`. `FFAI_ORDER=raster` already exists, and the §8.156 sparse-page
  gate already fires on this population but escapes to `xy_cut`, never to raster.
  **Untested on ReadOrder^Edit — contiguity is a proxy and the gate's own comment
  warns these pages barely move MICRO (+0.069 pp).**

  | source | n | emitted | raster | gain |
  |---|---|---|---|---|
  | note | 637 | 0.785 | 0.940 | **-0.155** |
  | newspaper | 6340 | 0.870 | 0.316 | +0.554 |
  | magazine | 1065 | 0.967 | 0.615 | +0.352 |
  | PPT2PDF | 449 | 0.947 | 0.945 | +0.002 |

- **1.2 Region<->GT-line correspondence.** ✅ **RUN 2026-08-16**
  (`stage1_tiling.py`). **Our units are RIGHT in 89 % of blocks.**

  | boxes/row-band | blocks | median char-ratio | missing chars |
  |---|---|---|---|
  | 1.0-1.2 (tiled) | 18 380 | 0.988 | 178 707 |
  | 3.0+ (shattered) | 464 | 0.582 | 41 409 |

  Fragmentation is real where it occurs — a shattered block loses 42 % of its
  characters — but explains a MINORITY: 62 % of missing characters sit in
  perfectly tiled blocks, and blocks reading <90 % have the SAME median
  boxes/row-band (1.00) as blocks reading >=90 %.

  Also: `newspaper` reads a median char-ratio of **1.000** while being the worst
  ordering class. Its entire deficit is SEQUENCE, not characters.

- **1.3 Region-boundary oracle.** ✅ **RUN 2026-08-16**
  (`f13_region_oracle.py`, via `examples/order_probe.rs` so the REAL
  `order_reading` is exercised). Perfect units in — OmniDocBench's own region
  polygons, GT text — and the shipped orderer sequences them.

  Two arms so the harness proves itself first: `f13gt` (GT text in GT `order`,
  must score ~0) and `f13ours` (GT text in OUR order).

  **The harness is only clean on 136 of 269 pages** — the `ORDERED` category set
  does not match the evaluator's expectations on pages carrying tables and
  figures. Restricted to the pages where `f13gt` scores EXACTLY 0.0000:

  > **f13gt 0.0000 -> f13ours 0.1589. 42 pages hurt, ZERO helped.**

  **THE ORDERER IS THE CONSTRAINT, NOT THE UNITS.** Given perfect detection,
  perfect line identity and perfect recognition, `order_reading` still costs
  **+0.1589 ReadOrder^Edit**. Against a shipped ReadOrder of 0.2153, most of the
  sequence error is the ordering ALGORITHM, not the boxes handed to it.

  Stages 3-4 therefore own real, priced slack. *(Note the unrestricted figure is
  +0.1152 — the dirty pages DILUTE the finding rather than inflate it, so the
  restriction is conservative in the right direction.)*

  *(Superseded plan text: "Replace our regions with GT line boxes, keep the
  existing orderer, score. Separates 'ordering cannot sequence our units' from
  "our units are wrong". *One scoring run, and only after 1.1 and 1.2.*
- **1.4 CRAFT-path constants.** ✅ **IMPLEMENTED AND MEASURED 2026-08-16.**

  `FFAI_LINE_XGAP` added to `group_lines` ([boxes.rs:100](../../crates/ffai-carmenta/src/boxes.rs)).
  The join test was VERTICAL OVERLAP ALONE — a box at the right margin joins a
  line built entirely at the left margin whenever they share a y-band, which is
  how a two-column page becomes one line per row across both columns. The new
  condition requires the box to sit within N LINE HEIGHTS of the line's current
  extent (scale-free across a 640 px frame and a 2400 px receipt). Default
  **0.0 = OFF**; 38 lib tests pass unchanged.

  **REFUTED ON RECEIPTS (CORD holdout, craft-parseq, 45 clips):**

  | `FFAI_LINE_XGAP` | CER |
  |---|---|
  | 0 (null) | **23.40 %** |
  | 1.0 | 58.89 % |
  | 2.0 | 56.00 % |
  | 3.0 | 50.83 % |

  2.5x worse at 1.0, recovering MONOTONICALLY toward baseline as the condition
  loosens. Mechanism, and the monotone trend confirms it rather than merely the
  sign: a receipt line is `ITEM NAME ......... $5.00`, so a wide intra-line gap
  is NORMAL and x-adjacency splits it in two. **Do not re-run this on
  single-column content.** The lever can only cost where there are no gutters to
  defend against.

  **⚠ NULL-ARM CAVEAT.** `XGAP=0` reads **23.40 %** against the ledger's 21.96 %
  for this engine/corpus, and `score_corpus` is documented to read ~1 pp BELOW
  the ledger. The A/B is valid (both arms, one binary) but the ledger figure is
  NOT reproducible from this binary — §43's vertical split and the §47-49 work
  all postdate that entry. Re-bank the CORD number before quoting 21.96 again.

  **INSTRUMENT FIX, and it fired immediately.** `score_corpus.py` hardcoded
  `target/release` while every other harness builds into `target-gate`. It now
  takes `FFAI_OCR_EXE` (defaulting to target-gate), REFUSES to run when any
  `.rs` is newer than the exe, and prints the binary and its timestamp. Its
  first run rejected a stale `ocr_text.exe` — without it the whole sweep would
  have returned four identical numbers and "refuted" 1.4 for the second time for
  the same reason §42 did.

  **ALSO REFUTED ON SCREENS** (`carmenta-frames-v1` holdout, craft-crnn, 23
  clips), which is where I predicted it would pay:

  | `FFAI_LINE_XGAP` | frames CER | CORD CER |
  |---|---|---|
  | 0 (null) | **1.92 %** | **23.40 %** |
  | 1.0 | 2.84 % | 58.89 % |
  | 2.0 | 2.31 % | 56.00 % |

  **1.4 IS DEAD ON EVERY CRAFT CORPUS.** Not a dispatch, not a tuning problem —
  the condition costs on both, with the same monotone recovery toward baseline.

  **THE PREMISE WAS BACKWARDS, and the sweep says so quantitatively.** At
  `XGAP=3.0` CORD still costs 27 pp, which means `group_lines` routinely joins
  boxes MORE THAN THREE LINE-HEIGHTS APART — and those joins are CORRECT,
  because removing them is what does the damage. 1.4 assumed the function was
  wrongly merging word boxes across columns. It is not: CRAFT's affinity already
  yields LINE-LEVEL components (§8.10), so the wide-gap joins are item->price on
  a receipt and label->value on a HUD. They are load-bearing.

  **Kept at default 0.0** — the knob stays for reproduction, the condition never
  fires. Do not rebuild this argument; it has now been implemented twice (§42
  inert on the wrong path, and here on the right one) and refuted on measurement
  the second time.

---

## Stage 2 — Corridor splitting (`split_at_white_corridor`)

[boxes.rs:1282](../../crates/ffai-carmenta/src/boxes.rs#L1282). Splits a line that
crosses a detected white corridor, so one line cannot span two columns.

**Open.** Never audited in this campaign. It is the guard against exactly the
"gutter-merged line" that `xy_cut_pernode` says costs 56.3 pp, so its recall
directly determines how often the `any()` strictness misfires.

### Exploration steps

- **2.1 How often does it fire, and does it need to?** ✅ **RUN 2026-08-19**
  (`stage2_corridor.py`, offline over the existing 305-page pre/post-split
  capture). **The splitter is nearly inert, and the problem it guards is
  nearly absent.** Of 32,476 pre-split lines, only **28** genuinely cross a GT
  column gap. The splitter fires 5 times: recall **3.6 %**, precision 20 %.
  The page-level gate proposed cuts on 96 pages and **blocked 91 of them**.

  The absolute prize is bounded by those 28 lines (~1 every 11 pages) times the
  spanning-veto blast radius — real when it lands (§ omni-0069's 56.3 pp) but a
  tail risk, not a segment. **Stage 2 is a footnote, not a campaign.** If the
  veto ever shows up in a worst-page audit, the fix is the gate (91/96 blocked
  is the anomaly), not corridor detection.
- **2.2 Spanning-line census.** On pages where `xy_cut_pernode` abandons the
  column grid, is the triggering line a real headline or a merge artifact?
  Classify against GT categories. *Decides whether to fix the splitter or the
  veto. Given 2.1's count of 28 crossing lines, run only if a worst-page audit
  implicates the veto.*

---

## Stage 3 — Cut strategy (`order_reading` → `order_by_selection`)

[boxes.rs:380](../../crates/ffai-carmenta/src/boxes.rs#L380) dispatches on
`FFAI_ORDER` across eight strategies; the default runs three candidates and
selects. Afterwards the §8.156 sparse-page gate may re-run plain `xy_cut`.

**Established / refuted.** `H_GAP_MIN` refuted from above (§8.68, −5.3 pp) and
below (§38); `V_GAP_MIN=0.35` is +0.0012 with a CI lower bound of +0.0001 and 42
pages changed (§39) — recorded, not shipped. `xy_cut_cost`, `order_one_level` and
`xy_cut_span` were each tried as a fourth candidate and each made selection worse.

### Exploration steps

- **3.0 Extended-pool census.** ✅ **RUN 2026-08-19** (`stage34_pool.py`, all
  11 `FFAI_ORDER` strategies, permutations dumped): best-of-ALL-arms 0.9784 vs
  best-of-pool3 0.9592 — half the pool ceiling was already in the menu. But
  §51 then showed the biggest outside winner (raster, 93 pages) FAILS the
  evaluator when actually selected (4.4 below), and `hfirst` added nothing
  over the v2 objective. **The remaining pool ceiling (~0.02 contiguity) needs
  a genuinely NEW strategy, and any candidate must be judged on the evaluator,
  not the proxy.**
- **3.1 Per-candidate oracle.** ✅ **RUN 2026-08-19** (`stage4_regret.py`, all
  1469 pages with ≥1 multi-line GT block, real `order_reading` via
  `order_probe.exe`, contiguity proxy). **Both are broken, and the menu is the
  bigger half.** Per-page mean contiguity: chosen 0.941, pool best-of-3
  **0.959**, oracle 1.0. Of the 0.059 chosen-vs-oracle deficit, **0.041 is pool
  ceiling** (no candidate produces it) and **0.018 is selector regret**. This
  independently reproduces §8.153's 1.90 pp regret / 2.90 pp ceiling split on a
  different instrument — two unrelated probes now agree on ~30/70.

  The pool is not monolithic: `noselect` wins 1241 pages but `vfirst` wins 143
  and `xycut` 85, so selection is load-bearing on a fifth of the corpus. A new
  pool member has more to win than a better chooser — but the chooser's 0.018
  is cheaper (Stage 4.2 needs no new strategy).
- **3.2 Sparse-page gate audit.** How often does it fire on the current corpus,
  and does it still help? It was fitted on 236 holdout pages pre-§33.
  `FFAI_ORDER_GATE=0` is a one-toggle A/B.

---

## Stage 4 — The selector objective (the reset score)

[boxes.rs:722](../../crates/ffai-carmenta/src/boxes.rs#L722). Scores an ordering
by the fraction of consecutive lines whose x-centre jumps left by more than
`FFAI_ORDER_SELECT_EPS` (0.08) of page width, and keeps the **minimum**.

**Why it is suspect.** The file records three separate candidates rejected with
the same diagnosis: *"the reset score rewards column-coherent output, and a wrong
ordering can be more column-coherent than a right one."* We select orderings with
a proxy that is documented to prefer wrong answers.

### Exploration steps

- **4.1 Selector regret.** ✅ **RUN 2026-08-19** with 3.1 (`stage4_regret.py`,
  same run — the two questions share one instrument). **Mean regret +0.018
  contiguity**, median 0.000: the selector is right on most pages and loses
  concretely on 121 pages with regret > 0.05. Regret by source:
  `research_report` +0.046, `note` +0.024, `colorful_textbook` +0.022 —
  NOT the newspapers (+0.008), which are pool-limited instead. Validation:
  the probe's default arm reads 0.941 against the emitted order's 0.945
  (the gap is Stage 5's `probe_reorder`, absent from the probe path), so the
  harness reproduces the engine. Regret is real, bounded, and 4.2 (swap the
  reset-score objective, keep the pool) is the direct next step — the reset
  score's documented failure ("a wrong ordering can be more column-coherent
  than a right one") now has a per-page price list.
- **4.2 Replace the objective with contiguity-like self-evidence.** ✅ **DONE
  AND BANKED 2026-08-19 (§51).** `FFAI_ORDER_SELECT=2`: objective
  `wreset + 0.5·yback + 2·scat` over the same three candidates, sparse gate
  untouched. Full-corpus evaluator (probe-reserialized arms, fallback-excluded):
  **ReadOrder +0.0035, CI [+0.0010, +0.0062]; text exactly 0.0000.** EN
  no-float order +0.0155. Designed offline against the `stage34_pool.py`
  permutation census (weights on a flat plateau, even/odd holdout held).
- **4.3 `FFAI_ORDER_SELECT_EPS` sweep.** Subsumed by 4.2 — the objective the
  eps belonged to was replaced entirely; the census showed eps 0.04/0.08/0.16
  within 0.001 contiguity of each other under the old objective.
- **4.4 (new, REFUTED — do not retry without the interleave axis):** RASTER as
  a margin-guarded challenger won the contiguity proxy (+0.004) and lost the
  evaluator: text **−0.0185 / EN −0.0375, CI excluding 0** on the 170 pages it
  took (§51). **Law: contiguity is blind to interleaving** — a raster-read
  multi-column page keeps blocks index-compact while alternating them across
  columns. Challenger default is off (`FFAI_ORDER_V2_MARGIN=∞`); salvage, if
  any, is a sparse-page-only condition.

---

## Stage 5 — Post-recognition reorder (`probe_reorder`)

[suppress.rs:819](../../crates/ffai-carmenta/src/suppress.rs#L819). A second
reorder after recognition, accepted by three grounds: the §8.157 guard, a
`verifier_blind` competence check, then the text verifier.

**Established.** Disabling the probe costs 0.0126 text; disabling the verifier
costs 0.0083 (§22). Both are real and shipped.

**Known defect.** `probe_gate_fires` tests `body_frac > 0.85`, and since
body-only suppression went off `n_body == n_all`, so the term is permanently
true. The guard fires more often than it was fitted for.

### Exploration steps

- **5.1 Re-fit or delete the dead term.** `FFAI_ORDER_GUARD=0` already routes
  every page to the verifier; §31 measured it moves 6 of 755 pages. Re-measure on
  the current baseline and either delete the term or re-fit it on live inputs.
- **5.2 Does the probe help or hurt the newspaper population?** It fires on large
  dense pages, which is exactly where ordering fails. Split its effect by document
  type rather than the aggregate. *One toggle, one scoring run.*

---

## Stage 6 — Emission

`gg_arm.py:125` joins non-empty stdout lines with `\n\n`, so every physical line
becomes its own markdown block.

**Established.** Coarsening this is refuted twice — MGAM region grouping (§4,
0.2051 vs 0.1041) and line→paragraph joining (§34, −0.1831). The matcher merges
but cannot split, so fine-grained output is protective. **Do not coarsen.**
Re-sequencing without coarsening is the permitted move, and is Stages 1–4.

---

## Sequencing — REWRITTEN 2026-08-19, all screening probes now run

Stages 0, 1, 2 and the re-price are DONE. What the probes left standing, in
order of expected value per unit of work:

1. **Stage 3 — a new pool member** owns the largest measured slack: pool
   ceiling 0.041 contiguity (~70 % of the ordering deficit), worth up to
   ~+0.019 text / +0.045 order (§50's orctext ceiling bounds all of Stages
   3–5 combined). The 1.3 oracle says the failure is the ALGORITHM given
   perfect units, and 1.1 says newspapers/notes are where the pool falls
   short. Candidate design should start from the per-page winner data in
   `stage4_regret.py` — what do the 143 vfirst-pages and 85 xycut-pages have
   in common that pernode lacks?
2. **Stage 4.2 — swap the selector objective** for the cheaper ~30 %: regret
   0.018 contiguity, no new strategy needed, offline iteration against
   `stage4_regret.py` before any scoring run. The reset score's documented
   failure mode now has a price list (research_report +0.046). 4.3 (EPS sweep)
   rides along free.
3. **Routing guard sweep (outside this plan, §50):** five named catastrophic
   guard-miss pages, newspaper net-negative on both columns, text-hurt tail in
   exam_paper/book. ~+0.003 order / +0.002 text, and it converts routing's
   text-neutral toward positive. Cheap: `FFAI_ROUTE_RETAIN`/`FFAI_ROUTE_SCORE`
   are env toggles; screen on the 5-page hit list before any corpus pass.
4. **`FFAI_ORDER=raster` on `note` pages** — 1.1's fourth sign-flip: ordering
   makes notes WORSE than the raster order it was handed (0.940 → 0.785
   contiguity, 637 blocks). The gate exists (§8.156 fires there but escapes to
   xy_cut); pointing it at raster is a one-line arm. Untested on the metric —
   contiguity is a proxy, and these pages barely move micro. Screen first.
5. **Stage 5.1** — delete or re-fit the dead `body_frac` term (hygiene, §31
   priced it at 6 pages).

Stage 0 is FINISHED (ceiling ~0.004, §0.1). Stage 2 is a footnote (28 crossing
lines corpus-wide, 2.1). Stage 6 stays closed — do not coarsen. The §36/§50
timeout law applies to every scoring run above: exclude the union of both runs'
fallback pages, or score arm and base in one session.

**Screen on `bench_subset_300.json`** (305 pages, ~17 min against 132) for
anything needing the engine; bank on the full corpus. The subset reproduces
known A/B deltas within ±0.005 — enough to kill a lever, not enough to ship a
marginal one. Caveat for this plan specifically: it under-represents the largest
pages and carries only 26 of 145 newspapers, so ordering work wants a
newspaper-weighted subset validated the same way.

Formula representation — the largest ceiling at +0.0473 — is **not** in this plan.
It is a capability gap (image→LaTeX), not an ordering defect, and its blast radius
exceeds its upside (§41): 475 bad LaTeX blocks to win against 1496 already-good
ones to lose.
