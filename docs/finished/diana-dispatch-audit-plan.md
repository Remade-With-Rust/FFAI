# Diana — a dispatch audit of every settled lever

**Status:** plan, nothing executed
**Prompted by:** 2026-08-05, when a lever refuted on wall turned out to be
1.12x FASTER once its arm was parallelised, and two more turned out to be CPU
operating points rather than losses.

---

## 0. Why this is worth running

Every keep/revert decision in the Diana campaign was made on **wall time on one
machine at one thread count**. That is one cell of a four-cell table, and the
other three have now each produced a result that the first one missed:

| axis | what it asks | first found |
|---|---|---|
| **wall** | is it faster? | always measured |
| **CPU-seconds** | does it burn less machine? | 2026-08-05 — three levers are trades |
| **parallelism** (`cpu/wall`) | were the arms even comparable? | 2026-08-05 — overturned a refutation |
| **per-shape / per-content** | does the answer vary by unit? | measured for 4 levers, refuted for all 4 |

The audit is **cheap**: five levers, one harness, minutes of machine time. It
produced two strengthened refutations, one caveat, and two shipped operating
points, with zero new kernel code. That ratio is the argument for doing the
rest.

**It is also the highest-leverage kind of work available right now**, because
every structural lever is closed. There is nothing left to build; what remains
is finding out whether things already built were judged on the wrong axis.

---

## 1. The four dispatch axes, and the rule each one needs

### A. Deployment intent — latency vs CPU
An arm that is slower in wall while using less CPU is **not a loss**. It is the
arm a throughput-bound host should pick. Costs nothing to dispatch: the embedder
knows its own regime, so the selector is a config constant with no runtime probe.
*(codec-measurement §2b)*

### B. Parallelism mismatch — the `cpu/wall` column
Two arms at different thread counts are not comparable, and work-COUNT parity
cannot catch it because both arms did identical work. **An arm that looks slower
is often just under-parallelised.** Check before recording any loss.
*(codec-measurement §2a)*

### C. Per-shape / static dispatch
The shape is known at model load, so a per-shape choice is a build-time constant
with zero runtime cost. **Requires sign-consistency across >= 4 independent
runs** — tonight, 3 of 4 candidates that looked bimodal in one run sign-flipped
across four. *(codec-content-adaptive-dispatch)*

### D. Per-content dispatch
**Already refuted for this engine, and the mechanism is absent rather than
unfound**: static shapes, dense arithmetic, no data-dependent branching, 0
denormals in 80.6M values, and latency correlating -0.009 with detection count.
**Do not re-test D per lever.** It is closed at the engine level.

---

## 2. The harness — build once, before touching any lever

`tools/diana_lever_audit.py`, generalising tonight's throwaway script. For any
toggle it must report, in one table:

| column | why it is mandatory |
|---|---|
| wall ms/frame | the shipped objective |
| **cpu ms/frame** | axis A |
| **cores busy** (`cpu/wall`) | axis B — the column whose absence cost an hour |
| **detections per pass** | work parity (§4); a differing count voids the row |
| per-rep spread | so a 25 % spread is never read as a verdict |
| ABBA order | drift lands on both arms equally (§3) |

Two modes:
* **sweep** — every lever vs the shipped default, 4 ABBA reps. Minutes.
* **per-shape** — one lever, `FFAI_DIANA_ROOFLINE=1`, 4 independent runs,
  sign-consistency required. Only for levers the sweep flags.

**The harness prints its own method line** (§13) so any number it emits is
auditable later.

---

## 3. Lever inventory, with priors

Ordered by expected yield, not alphabetically. `[done]` marks tonight's five.

### High prior — a physical reason to expect a CPU or parallelism story

| lever | what it switches | why it might have been misjudged |
|---|---|---|
| `FFAI_DIANA_NO_ZEROCOPY` | `SliceOp` zero-copy vs copy-in/out | Removes two multi-MB copies per conv. Pure work removal — should win on BOTH axes, and confirming that validates the harness against a known-good. |
| `FFAI_DIANA_ZEROFILL` | the redundant im2col zero-write | 3.1-9.4 ms/image of pure writes removed. Same shape of claim as the epilogue, which the CPU column vindicated. |
| `FFAI_DIANA_NO_DWCONV` | our depthwise vs candle's grouped | candle's path was 72.9x slower per FLOP — but it is a *different algorithm*, so its CPU and parallelism profile are unknown. |
| `FFAI_DIANA_NESTED_PAR` | nested per-layer fan-out inside a batch | Refuted on a **2.32x CPU tax**, so axis A was checked. Axis B was not — and this lever is *about* parallelism, so cores-busy is the natural instrument. |
| `FFAI_DIANA_DIRECT` `[done]` | direct conv vs im2col+GEMM | 0.95x CPU at 0.86x wall. **Re-test fanned out** — its row-blocked structure may be under-parallelised exactly as Winograd's was. |

### Medium prior — a per-shape question never asked

| lever | note |
|---|---|
| `FFAI_DIANA_NO_PW` | our pointwise vs candle, dispatched by KIND but never **within** kind. 49 1x1 convs/image spanning cout 8-256. |
| `FFAI_DIANA_NO_S2` | same, for the 7 stride-2 convs — the widest spatial shapes in the graph. |
| `FFAI_DIANA_NO_CONV3` | same, for the 32 dense 3x3. |
| `FFAI_DIANA_NO_AVX2`, `SILU_ROUND` | AVX2 SiLU vs scalar. SiLU is 1.0 % of detect, so **prune on arithmetic first** — the ceiling is 1 %. |

### Already settled tonight — do not re-run

`NO_INPLACE` (right on both axes), `TILE` (0.70x wall AND 1.18x CPU, closed),
`THREADS` (documented as a curve, not a constant), `NHWC` (break-even 5.1 convs
vs an architecture offering 2).

### Non-toggle decisions worth the same treatment

* **`MIMALLOC_PURGE_DELAY`** — halves page faults for +57 MiB. Faults are
  *kernel* CPU, and only wall was measured. One CPU run settles it.
* **LIVE gate threshold** — a work-REMOVAL feature, so its CPU saving is by
  construction larger than its wall saving. Currently priced on wall and mAP only.
* **Geometry (rect vs square)** — rect does 63 % of square's model work at 1080p.
  Never framed as a CPU trade.

---

## 4. Classification — what each result means

Run the sweep, then sort every row into exactly one bucket:

| result | bucket | action |
|---|---|---|
| worse on wall AND CPU | **refutation confirmed** | record the CPU number beside the old one; close permanently |
| better CPU, worse wall | **deployment-intent trade (A)** | document as an operating point with the curve; ship no default change |
| worse wall, **cores-busy well below the winning arm** | **suspect (B)** | re-test fanned out before believing the loss |
| better on both | **missed win** | gate, measure at the level above, then default it |
| sign-flips across reps | **per-shape candidate (C)** | run the 4-run reproducibility mode; require every rep to agree |

**A row may only leave the table via one of these five.** "Inconclusive" is not
a bucket — it means more reps, not a shrug.

---

## 5. Stop rules, so this does not become its own campaign

1. **Prune on arithmetic first.** A lever whose stage is under ~2 % of detect
   cannot repay a dispatch however the axes land. SiLU at 1.0 % is the example:
   compute the ceiling, write it down, skip the measurement.
2. **The sweep is the budget.** One pass over all levers, 4 reps each. Deeper
   work only for rows the sweep flags.
3. **No building in this phase.** The audit's output is a classified table and
   documentation. Anything in the "missed win" bucket gets its own gated brick
   afterwards, with a break-even computed before a line is written — the lesson
   NHWC cost four turns to learn and Winograd honoured in twenty minutes.
4. **Expected yield is documentation, not speed.** Tonight: 5 levers, 3 trades,
   0 new code. Plan for that, and treat a genuine "missed win" as a bonus.

---

## 6. Order of work

1. Build `tools/diana_lever_audit.py` with the mandatory columns. Validate it on
   `NO_ZEROCOPY`, which is expected to win on both axes — **a harness that cannot
   reproduce a known-good result is not yet trustworthy.**
2. Sweep the high-prior levers (5). Classify.
3. Re-test every axis-B suspect fanned out, starting with `DIRECT`.
4. Sweep medium-prior levers; skip anything the arithmetic prunes.
5. Per-shape mode only for sign-flippers, 4 runs, sign-consistency required.
6. Fold the results into both published READMEs as operating points, and append
   any new law to `codec-measurement` / `codec-content-adaptive-dispatch`.

---

## 7. What this plan is really defending against

Every wrong verdict this campaign produced had the same shape: **a number taken
honestly on one axis, generalised to a decision that spanned several.** The
1.68x synthetic orientation win, the 1.645x loaded-box GEMM ratio, the
"dead on arithmetic" Winograd verdict, the v2-vs-v3 im2col revert — all four
were true as measured and wrong as applied.

The audit does not add a new instrument. It applies the ones already written
down to decisions that were made before they existed.

---

# EXECUTED — 2026-08-05

`tools/diana_lever_audit.py`, 10 levers, ABBA-interleaved, CPU via
`GetProcessTimes`, work parity checked per arm. Flagged rows re-run at 8 reps.

## The sweep

| lever (arm) | wall x | cpu x | cores | spread | pol | verdict |
|---|---:|---:|---:|---:|---|---|
| no zero-copy SliceOp | 1.08 | 1.11 | 2.22 | 7 % | off | confirmed both |
| im2col zero-fill back | 1.13 | 1.15 | 2.28 | 3 % | off | confirmed both |
| candle grouped depthwise | 1.38 | 1.30 | 2.31 | 9 % | off | confirmed both |
| candle pointwise | 1.49 | 1.21 | 1.76 | 10 % | off | confirmed both |
| candle stride-2 | 1.32 | 1.46 | 2.13 | 3 % | off | confirmed both |
| candle dense 3x3 | 1.47 | 1.40 | 2.21 | 8 % | off | confirmed both |
| scalar SiLU (no AVX2) | 1.10 | 1.18 | 2.48 | 22 % | off | confirmed both |
| nested parallelism | 1.01 | 1.00 | 2.10 | 24 % | alt | no effect |
| direct convolution | 0.99 | 1.03 | 2.14 | 16 % | alt | no effect |
| **mimalloc purge off** | **1.00** | **0.95** | 2.15 | **9 %** | alt | **TRADE** |

## Result: no missed wins, one real trade

**Seven shipped defaults are now confirmed on BOTH axes**, where before they
were confirmed on wall alone. That is the bulk of the value: zero-copy,
zero-fill removal, and all four of our kernels beating candle's are correct
choices for a latency user AND a CPU-constrained one.

**`MIMALLOC_PURGE_DELAY=-1` is the only genuine trade: 0.95x CPU at 1.00x
wall** — 5 % less machine for free, in exchange for +57 MiB steady RSS. It
reproduces an independent same-binary ABBA from earlier the same day (0.968x
CPU, 1.033x wall), and it has the lowest spread of any flagged row. A
CPU-constrained host with memory headroom should set it; the footprint gate is
why it is not a default.

**No lever anywhere in the engine is a missed win.** Every shipped default
survives both axes.

## Two flags evaporated when the reps doubled

| lever | 4 reps | 8 reps |
|---|---|---|
| nested parallelism | 0.91 wall / 1.01 cpu, **24 % spread** | 1.01 / 1.00 |
| direct convolution | 0.97 / 1.11, 6 % spread | 0.99 / 1.03 |

Both looked like trades and neither is. **The spread column is what identified
them as untrustworthy before anything was built on them** — and direct
convolution's earlier "0.95x CPU at 0.86x wall" reading, taken on a throwaway
script without a spread column, was noise in both directions.

## The harness caught two bugs in itself, and that is the point

1. **Inverted ratio.** The first classifier read `wx > 1` (the disabled arm is
   slower) as "missed win" when it means the optimisation is WINNING. Caught
   immediately by the `NO_ZEROCOPY` validation row, which is in the sweep for
   exactly that reason — **a harness that cannot reproduce a known-good result
   is not yet trustworthy.**
2. **Ignored polarity.** Three levers (`DIRECT`, `NESTED_PAR`,
   `MIMALLOC_PURGE_DELAY`) ENABLE a refuted alternative rather than disabling a
   shipped optimisation, so their verdicts invert. The first sweep labelled all
   ten as "disable" and read three backwards. Polarity is now a required field
   on every lever.

Both bugs produced plausible, confidently-worded verdicts. Neither would have
been caught by staring at the code.

## What is still unaudited

* **The LIVE gate threshold** — a work-REMOVAL feature, so its CPU saving is by
  construction larger than its wall saving, and it has only ever been priced on
  wall and mAP.
* **Geometry (rect vs square)** — rect does 63 % of square's model work at
  1080p and was never framed as a CPU trade.
* **Per-shape mode** was not needed: nothing in the sweep sign-flipped, so
  there was no candidate to run it on.
