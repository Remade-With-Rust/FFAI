# Open work — Mercury speed campaign

State at `cf821cc`. Gap to `whisper-cpp-tiny-greedy-t24`: **1.13–1.15×**
(from 2.84× when this descent started). Quality **PASS** on both corpora.
Speed gate **FAIL**. 62 tests, 0 failures.

Every stage is instrumented per-op and every residue is closed to 1–3 %, so
what remains is not "find the time" — it is these specific, sized items.

---

## 1. Four refutations that were never properly probed

Each was killed on **one measurement**. This campaign found **four wrong
refutations out of nine**, so a single-probe rejection is not evidence. Under
the three-probe rule (`codec-six-whys-unknowns`), each needs three varied
probes, **at least one at the level above the change**.

Listed with the reason I expect each to lose. **A reason to expect a loss is
not a measurement** — that distinction is why these are "unexamined", not
"refuted".

| lever | why it was killed | why the kill may be wrong | expected |
|---|---|---|---|
| **NT-form kernel** (K in natural layout, no transpose) | value-per-effort; ~5 ms = 1.5 % | never actually measured | loses — replaces AXPY-over-keys with dot-products-plus-reduction, the structure that measured **46 GFLOP/s vs 73** early on |
| **int8 decoder MLP** | 1.005×, inside noise | delivery was `to_vec1` + rayon; `CustomOp1` + serial swung the f16 GEMV 0.946 → 1.144× | loses — baseline moved f32 → f16, so it now buys 2× not 4×, for the quality risk that failed at 8.39 % WER |
| **parallel decode heads** | 1.14× stage, 0.975× pipeline | measured at f32 cache size | loses harder — the f16 cache **halved per-head work**, so fork-join is now relatively more expensive |
| **f16 KV pipeline claim** | 13/21, z = +1.1, inconclusive | kept anyway, on stage win + halved memory | unresolvable at ~2.5 % without a quieter machine |

## 1b. NT-form kernel — BUILT, MEASURED, REVERTED

The micro said **1.95x** (33 -> 64 GFLOP/s on a 64x256 tile). Ported it in
full; in context `ea_kernel` went **0.038 -> 0.054 s** — slower.

The micro used a 64x256x64 tile, ~80 KB, entirely L1-resident. In context K is
384 KB per head, 2.3 MB per layer. The access patterns then diverge:

- **AXPY form** reads `kt[t*seq + k0+j]` — ONE contiguous stream of 256 keys.
- **NT form** reads `k[(k0+j)*HD + d]` — FOUR rows 256 B apart, four streams.

In L1 that costs nothing. At 2.3 MB it is 4x the cache pressure, and it eats
the register-tiling win plus the transpose saving.

So the ORIGINAL refutation was right, and the probe that "reversed" it was
right too — about a tile that fits in L1, which is not the tile that runs.
Both my refutation and my reversal were single-level measurements.

**Caveat on this entry:** the revert rests on profile samples, not a paired
A/B. Under the three-probe rule that is not sufficient to call it settled —
it is sufficient to justify not shipping it. If revisited, block K so the
four streams stay resident (smaller BK, or pack K into tiles at load).

## 2. Sized but not built: q/k/v fusion

**Do not rebuild this without reading why it died.** The refutation's
*conclusion* is right; its *stated reason* was wrong, and so was the re-probe's.

- Fused transposed `(1152,384)@(384,1500)`: splits fall on dim 0 and are
  **contiguous**; q and k land in the kernel's layout. Projections alone:
  **5.24 ms vs 10.40 shipped, 1.98×.**
- In the pipeline: **0.963×, 2/21, z = −3.7.**
- Mechanism: the kernel broadcasts one q value per (row, dim). Row-major
  `q[i*64 + t]` walks 64 **contiguous** floats — 4 cache lines, hot across the
  key loop. Transposed `q[t*1500 + i]` jumps **6 KB per t-step**: 64 distinct
  lines, no reuse.
- **The projection wants (HD, seq); the kernel wants (seq, HD).** Two
  consumers of one tensor, opposite layouts.

Only worth revisiting if the kernel's tiling is redesigned to walk q
column-wise — at which point re-measure both consumers, not one.

## 3. Unexplained: the quantization asymmetry

The **harder** corpus is **less** sensitive to quantization. Measured twice,
predicted backwards both times, still no mechanism:

| | test-clean | test-other |
|---|---:|---:|
| argmax flips vs f32 oracle | 1.77 % | **1.07 %** |
| WER cost of all-decoder int8 | +0.62 pp | **+0.05 pp** |

The intuition — harder audio → flatter logits → closer top-2 → more flips — is
wrong, or is swamped by something larger. Worth understanding before trusting
any future quantization decision, because it means **test-clean is the
sensitive gate and test-other will not catch what test-clean misses.**

## 4. Instrumentation gaps

- **Footprint gate is SKIP** — never built. The f16 KV cache halves cache
  memory (18.4 → 9.2 MB/token) and that benefit is currently unclaimable.
- **The argmax-flip audit only sees the final projection.** It is blind to
  compounding error upstream — which is exactly the class that failed the
  all-decoder int8 experiment.
- **The reference's throughput spreads 25–37× across runs** on this machine.
  Cross-run gap ratios carry that; single-process stage deltas do not. Report
  progress with the latter.

---

## What would actually move the gate

Nothing on this list closes 1.13×. Both remaining candidates sit outside what
was measurable here:

1. **A quantized KV cache that survives the quality gate.** int8 did not
   (8.39 % WER). Would need a finer scheme — per-head scales, or int8 keys
   with f16 values — and the argmax-flip instrument **cannot see this class**,
   so it needs corpus WER on both sets per iteration.
2. **Work inside candle's CPU GEMM.** Our kernels now run at 550 GFLOP/s
   against its ~615 square-matmul peak; the remaining headroom is in the
   framework, not above it.
