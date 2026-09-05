# Carmenta math campaign — win the 0.0415

**Created 2026-08-20, §57's #1 block.** Inline-math text blocks carry **36.1 %
of all remaining Text^Edit loss** (0.0415 of 0.1150) — measured twice by
independent instruments (§54 spliced-GT oracle +0.0416; §57 per-block ledger
0.0415). CJK 0.0228 / latin 0.0187, in 3,295 GT blocks on 562 pages. Fully
captured, the headline reads ≈ 0.074; realistically captured, under 0.10 —
past Marker, into the published pipeline pack. This document is the staged
plan, with a kill gate in front of every build.

---

## 0. The measured starting position — read before building anything

| fact | number | where |
|---|---|---|
| The pool | 0.0415 Text^Edit, 3,295 blocks, 562 pages | §57 ledger |
| Ceiling (perfect fix of those blocks) | +0.0416, CI [+0.0352, +0.0483] | §54 F.1 oracle |
| v1 splice capture (`FFAI_ROUTE_INLINE=0.4`) | +0.0023 (6 %) | §56 |
| **Limiter 1 — RECALL** | **only 498 / 3,209 math blocks touched (15.5 %)** | fullv4-vs-wmfinal block diff |
| **Limiter 2 — QUALITY** | **touched blocks recover only 18 % of their loss** | same |
| Untouched blocks under v1 | −1 % (exactly nothing, as designed) | same |
| Order tax | −0.0022, DIAGNOSED: matcher ambiguity, not our sequence | §56, Evans-page proof |
| CJK capture so far | +0.0003 (nothing) | §56 |
| The strategic unlock | the evaluator normalises PREDICTIONS with the same `textblock2unicode` as GT — emit `$latex$`, the evaluator does the unicode | match.py:545 |

**Capture = recall × splice-quality.** 0.155 × 0.18 ≈ the 6 % we got. Neither
lever alone can win; the plan runs both tracks.

Existing machinery this campaign inherits: `ctc_greedy_spans` (char
x-positions, F.2), the guarded splice in `route.rs` (`FFAI_ROUTE_INLINE`,
`FFAI_ROUTE_INLINE_ANCHOR`), the ONNX graph loader that already runs
DocLayout/SLANet/FormulaNet, onnxruntime as the port oracle, and the §57
ledger as the campaign scoreboard (re-run it after every banked lever — it is
free and it cannot lie about shares).

---

## 1. Gates (inherited, plus math-specific)

1. Text^Edit is the objective; CI must exclude zero. **Order is a
   non-regression gate** — §56's tax blocked v1 from default and the same bar
   holds here.
2. Rule 9 (fallback-union exclusion) on every cross-session comparison.
3. Splice-off must stay byte-identical — every lever env-gated, one binary.
4. **No threshold is lowered before its blast radius is measured** (the §49
   unguarded-routing precedent: +0.0347 text harm; v1's zero false fires on
   no-math pages is the bar to keep).
5. A lever that fires on some content and hurts other content is an
   UNFINISHED DISPATCH, not a result (§55's word-merge chain is the template).

---

## 2. R-track — recall (85 % of the pool is currently unreachable)

### R1. Reachability census *(offline + one 40-min layout pass — DO FIRST)*
Run `layout_batch` at floor 0.30 over all 1,651 pages. For each of the 3,295
math blocks: does ANY formula region overlap it, at what score, covering what
fraction of its inline spans? Output: the pool split into
region-reachable / region-missed, by score band and script.
> **KILL GATE:** if ≤ 40 % of the pool's loss mass is reachable at any layout
> floor, the layout-only road cannot win and R2/R3 are mandatory, not
> optional. This census also prices exactly what R3 must beat.

### R2. Text-driven span finder *(no new models — the cheap recall unlock)*
The recognizer already READS the math badly — which means it knows where it
is. Candidate spans from the SVTR decode itself: maximal runs of math-shaped
characters (`= ^ _ \ / ± × ÷ ≤ ≥ ∑ ∫ √ π α–ω`, digit-symbol mixtures,
sub/superscript unicode), located on the page via `ctc_greedy_spans`, cropped,
confirmed by FormulaNet, spliced under the existing guard stack (single host,
minority span, decode-identity, anchor guard). A FormulaNet output that is
degenerate or non-mathy refuses the splice — the confirmation IS the
false-positive guard.
> Screen on the F.1 562-page set + 20 no-math pages (blast radius), then the
> affected-pages splice-arm pattern (§55). Prize if recall reaches ~60 % at
> v1 quality: ~+0.008–0.012. With Q-track quality: multiplicative.

### R3. Port a dedicated MFD (math formula detection) model *(the reference answer)*
Every reference pipeline (PP-StructureV3, MinerU) runs a purpose-trained
formula detector, not a layout model, for exactly this reason. PaddleOCR
ships MFD weights (PP-DocLayout family / YOLO-MFD class). Same port
discipline that landed DocLayout, SLANet and FormulaNet in three sessions
(§45–§47): arch JSON + safetensors via `onnx_graph`, oracle-matched against
onnxruntime, then swapped in as the inline-region source.
> Build ONLY if R1+R2 leave ≥ 0.015 of the pool unreachable — R1's census
> makes that arithmetic explicit before a line is written.

---

## 3. Q-track — splice quality (82 % of fired-block loss survives)

### Q1. Residual decomposition on fired blocks *(offline, free — DO FIRST)*
On the 498 touched blocks: how much of the residual is (a) span coverage
(region covered only part of the math), (b) LaTeX mismatch after
normalisation (run the evaluator's own `normalized_formula` over our latex vs
GT's — offline, per block), (c) collateral plain-text error in the block?
> This decides whether Q2 (bigger formula model) or R-side span-completion
> pays first. Do not port a model before this says the latex is the problem.

### Q2. FormulaNet tier upgrade
We ship PP-FormulaNet-**S**. The **plus/L** tiers exist in the same family and
load through the same graph machinery. Oracle-match against onnxruntime on
the §47 fixtures, then A/B behind `FFAI_FORMULA_MODEL`.
> Speed budget exists (9×); the §1 law — spend speed for quality — applies.

### Q3. UniMERNet-class model *(the GT-alignment play)*
OmniDocBench's formula GT was annotated by the UniMERNet/GPT-4o family — §44
recorded this. A recognizer from the annotator's own family matches its LaTeX
conventions where FormulaNet's dialect differs in ways normalisation does not
forgive. Bigger port; justified only if Q1 shows large post-normalisation
LaTeX mismatch AND Q2 does not close it.

### Q4. Normalisation-aware LaTeX hygiene *(cheap, do with Q1)*
Diff our spliced latex against GT latex AFTER both pass `normalized_formula`.
Any systematic surviving artifact (spacing conventions, `\mathrm` wrappers,
delimiter styles) is a free postprocessing fix — measure, don't guess.

---

## 4. O-track — the order tax (must stay ≤ 0)

O1. **Anchor-guard verdict** (`FFAI_ROUTE_INLINE_ANCHOR`, fullv5 arm —
in flight as this is written). If it neutralises the −0.0022 at acceptable
text cost, it is the standing guard for every R/Q lever.
O2. If a residual tax survives: the §56 method (per-page evaluator order
records) localises WHICH splices flip matcher assignments; candidate rules —
never splice the line's leading tokens, require the host's neighbours to keep
distinct anchors — are swept offline on cached per-page scores (the free
recombination trick from §55).

---

## 5. C-track — CJK math (0.0228, half the pool, zero captured)

C1. **Diagnose before building**: 20 worst CJK math blocks — are regions
detected there at all (R1 answers by script)? Are the spans mixed
CJK-prose + math on one line (R2's finder handles; layout regions do not)?
Is the anchor guard refusing CJK hosts? Is the loss actually display-`$$`
sitting inside GT text blocks (a different splice shape)?
C2. Apply the winning R/Q levers with a CJK screen set; the §18 CJK-arm
precedent (script-aware behavior behind one env) is the pattern if CJK wants
different guards.

---

## 6. Sequencing and the prize ladder — REVISED 2026-08-21 (§59)

**Execution inverted the order.** R1/Q1/Q4 ran; R2 was built, bug-fixed
(§59's early-return find) and REFUTED at every configuration: at current
LaTeX quality a splice nets zero text (FormulaNet-S ≈ SVTR on the broad
candidate population) and real order tax (matcher ambiguity). **Quality
first, then recall**: Q2/Q3 must prove BETTER-THAN-SVTR on the hard
candidate population by an OFFLINE crop-level A/B (normalised-GT-span edit
distance — no engine runs) before any recall lever re-runs. The recall
machinery (span finder, counters, censuses) is built and waiting.


| step | cost | gate it must pass | cumulative prize (est.) |
|---|---|---|---|
| R1 census + Q1 decomposition + Q4 diff | 1 day, mostly offline | — (they ARE the gates) | — |
| O1 verdict | pending run | order ≥ 0 | keeps +0.0023 |
| R2 span finder | 1–2 sessions | blast radius + affected-page arm | +0.008–0.012 |
| Q2 FormulaNet upgrade | 1–2 sessions | Q1 says latex-limited; oracle match | +0.015–0.020 |
| R3 MFD port | 2–4 sessions | R1 says ≥0.015 unreachable | +0.025–0.030 |
| Q3 UniMERNet | larger | Q1+Q2 leave latex mismatch | toward +0.033 |

Milestones on the headline: **+0.010 → 0.105; +0.020 → 0.095 (past Marker,
under 0.10); +0.030 → 0.085.** Each banked lever re-runs the §57 ledger so
the shares stay honest and the next step is chosen by measurement.

Side quest, same corpus passes: the display-math column (CDM) and TEDS have
never been scored — every Q-track improvement also moves them, and scoring
them is one `scorecard.py` session under distinct save names (§49's
collision warning applies).

## 7. What this campaign must NOT do

- Lower a detection floor without the blast-radius measurement (gate 4).
- Trust a proxy: char-ratio undersold fragmentation (§53); contiguity was
  blind to interleaving (§51); the matcher-ambiguity tax (§56) is this
  campaign's own proxy trap — ONLY the paired text+order evaluator verdict
  banks a lever.
- Coarsen output granularity chasing matcher behavior (§4/§34 law).
- Generalise from the Putnam page. Screen sets: the 562 math pages, a CJK
  slice, and no-math pages, every time.
