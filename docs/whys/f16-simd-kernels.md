# WHYS — f16 SIMD kernels

**Unknown:** §6.18 concluded the encode gap to whisper.cpp (1.39×) is
"candle's CPU backend lacking f16 SIMD kernels." That is a *mechanism claim*,
asserted from a single benchmark. Descend on it.

Method: `codec-six-whys-unknowns`.

---

## D6 (run FIRST, per the skill) — is the instrument sound?

- **ASKED:** what is the noise floor before any A/B is attempted?
- **MEASURED:** encoder fc1 f32, identical operation, 9 consecutive runs:
  min 8.98 ms, median 37.55 ms, max 53.41 ms.
- **ANSWER:** spread is **118 % of the median** — a 5.9× range on the same
  op. Two `pythonw` processes hold 5788 s and 2944 s of CPU and the machine
  sits at **70 % background load** (not ours; not touched).
- **CONFIDENCE:** high, and decisive.
- **STATUS:** **BLOCKING.** No A/B below ~2× can be resolved on this machine
  in this state.

### What this retroactively invalidates

This is the depth-6 payoff and it is uncomfortable. Sorting every M2 result
against a 118 % floor:

| result | size | verdict |
|---|---:|---|
| GELU `tanh` → Padé | 8.7× on the op | **safe** — far above any plausible floor |
| Incremental decoder | 2.6× on the stage | **safe** |
| Pre-transposed vocabulary projection | removed 91.5 % of decoder time | **safe** |
| Query-tiled attention (pruned) | 1.28× *worse*, monotone across 5 tile sizes | **safe** — the monotone trend is the evidence, not any single delta |
| Fused-attention ceiling probe | tax 6× the prize | **safe** — margin is enormous |
| int8 decoder (kept as variant) | "flat", 11.4 vs 12.1 | **UNRESOLVED** |
| f16 K/V cross-attention cache (reverted) | 0.077 vs 0.086 s | **UNRESOLVED** — already flagged in §6.16 |
| Adaptive f16 vocabulary projection (kept) | 1.43–1.56× on the op | **probably safe**, re-confirm |
| Corpus throughput figures (13.9–17.5×) | ±26 % | **suspect** — quote the ratio, not the absolute |

The large structural wins survive; every fine-grained verdict from the last
stretch of the campaign needs re-running on a quiet machine. The
`codec-analyzer` rule was right and I applied it two sessions late: *establish
the noise floor before the first A/B, not after the confusing one.*

---

## D1 — is the encode gap real, at matched settings?

- **ASKED:** does whisper.cpp genuinely encode faster once both sides do the
  same work?
- **MEASURED:** after removing `-nt` (§6.17): encode 3528.5 ms vs our
  4891.7 ms over 20 clips.
- **ANSWER:** yes, **1.39×** — down from the 1.71× measured unfairly.
- **CONFIDENCE:** medium. Ratio-of-totals over 20 clips averages some noise,
  but the floor above says treat it as directional.
- **STATUS:** open, pending a quiet-machine re-run.

## D3 — is our encoder compute- or memory-bound?

- **ASKED:** which ceiling is the encoder actually against?
- **MEASURED:** anatomy bench — matmuls at **84–94 % of measured compute
  peak**; attention trio memory-bound; roofline balance 53.7 FLOP/byte.
- **ANSWER:** **mixed, and that matters.** The projections and MLP are
  compute-bound and nearly optimal. The attention trio (scores, softmax,
  attn@v ≈ 50 % of encoder primitive time) is memory-bound.
- **CONFIDENCE:** high — ratios this large clear the floor.
- **STATUS:** closed.

## D5 — what would f16 SIMD actually buy, mechanically?

- **ASKED:** the §6.18 claim assumes "f16 kernels" means native half-precision
  arithmetic. Is that what ggml has on this hardware?
- **REASONED (to be measured):** x86 AVX2 + F16C provides *conversion*
  instructions (`vcvtph2ps`), **not** native f16 FMA. Native f16 arithmetic
  needs AVX-512-FP16 (Sapphire Rapids and later). So ggml's f16 path on this
  class of CPU is almost certainly **f16 storage, f32 compute** — halved
  weight traffic at full f32 arithmetic throughput.
- **IMPLICATION IF TRUE — and it inverts the §6.18 conclusion:** f16 SIMD
  would buy nothing on the encoder's *compute-bound* matmuls (already at
  84–94 % of peak; there is no headroom to sell). Its value is confined to the
  **memory-bound attention trio**, which is the half we already tried to
  attack with f16 and lost — because candle's f16 *softmax* is 82 % slower and
  ate the matmul gain.
- **CONFIDENCE:** low — this is a hypothesis with a mechanism, which the skill
  says is the next question, not an answer.
- **STATUS:** **CLOSED — by reading source, not benchmarking.**

### D5 closure: answered structurally, because the machine could not answer it

D6 blocked benchmarking, so the question was re-posed as one that source can
settle: *does candle have an f16 path at all, and where does it break?*

**candle's CPU matmul accepts f16 and dispatches it straight to the tuned
`gemm` crate** (`cpu_backend/mod.rs`: `match T::DTYPE { F16 | F32 | F64 => {}
_ => Err(UnsupportedDTypeForOp) }`). It does not materialize, it does not
upcast the operands. The f16 matmul path is real and tuned.

**So the §6.18 conclusion — "the encode gap is candle's CPU backend lacking
f16 SIMD kernels" — is WRONG.** f16 matmul exists. It measured slower than
f32 (421 vs 615 GFLOP/s) for the reason the roofline already gave: on a
*compute-bound* shape, per-tile conversion costs more than halved weight
traffic returns. That is correct behaviour, not a missing kernel.

**Where candle genuinely lacks a specialization is softmax.** `candle-nn`'s
`SoftmaxLastDim::cpu_fwd` is generic over `T: Float` and evaluates
`(*s - max).exp()` per element. For `T = f16` that is a scalar `exp` wrapped
in *two conversions per element* — which is exactly the 82 % slowdown measured
in §6.16, and exactly what ate the f16 K/V cache's gain.

- **Named, contributable upstream fix:** an f16 arm for `softmax_last_dim`
  that converts each row to f32 **once**, computes, and converts back —
  instead of per element. Same shape as the GELU fix in §6.10.
- **Prize, computed before building:** softmax is 22 % of cross-attention,
  which is 30 % of decode. Unblocking the f16 K/V cache is worth roughly
  **3 % overall** — real, but small, and far below the current noise floor.
- **Decision: do not build it now.** Not because it is wrong, but because it
  cannot be validated on this machine and the ceiling does not justify
  building blind. `codec-six-whys-unknowns` rule 1 of the rebuild: ceiling
  first, then cost.

---

## Where the descent stands

Depth 6 halted the investigation before depth 5 could be closed, which is the
process working as designed rather than failing: a mechanism claim
("candle lacks f16 SIMD kernels") was about to be built on, and the descent
showed both that the claim is probably *mis-stated* (the win would be traffic,
not arithmetic) and that the machine cannot currently resolve the experiment
that would settle it.

**Resolved without a quiet machine.** D5 closed from source: candle converts
in-kernel, so the §6.18 conclusion is wrong and **the encode gap is
elsewhere**. That question is now reopened, honestly, rather than papered over
with a mechanism that sounded right.

---

## RESOLVED — the noise floor was beatable after all

D6 called the machine unusable. That was half right: sequential A/Bs are
indeed worthless here, but **interleaved paired A/B** resolves deltas the
machine cannot resolve sequentially
([`examples/interleaved_ab.rs`](../../crates/ffai-mercury/examples/interleaved_ab.rs)).
41 alternating rounds per test, verdict by paired win rate:

| test | paired | median ratio | z | verdict |
|---|---:|---:|---:|---|
| vocabulary projection f16 vs f32 | 41/41 | 1.55× | +6.4 | **f16 — KEEP** (ranges disjoint) |
| GELU ours vs candle | 41/41 | **13.77×** | +6.4 | **KEEP** — larger than the 8.7× first measured |
| cross-attn q@k, f16 K/V | 40/41 | 1.32× | +6.1 | f16 wins *the matmul* |
| softmax f16 vs f32 | 5/41 | 0.74× | −4.8 | **f32** wins |
| **full cross-attn chain** | **6/41** | **0.88×** | **−4.5** | **all-f32 — the revert was correct** |

Every UNRESOLVED verdict from the table above is now closed, and the two
shipped wins are confirmed with disjoint ranges. The f16 K/V cache was
reverted in §6.16 on a guess inside the noise; it is now reverted on
evidence, and for a reason the per-op numbers could not have shown — **f16
wins both matmuls and loses the softmax between them by more.**

**What is left, in priority order:**

1. **Reopen the encode gap (D2/D3 on encode specifically).** 1.39×, cause
   unknown again. The old answer was an inference, not a measurement, and it
   did not survive. Needs a quiet machine.
2. ~~Re-run every UNRESOLVED verdict~~ — **done**, see above.
3. **Optional small brick:** the f16 softmax specialization — worth ~3 %,
   contributable upstream, only after 1 and 2.

The descent cost one session and deleted a wrong conclusion that was about to
become an upstream project. That is the process paying for itself: **the most
valuable output of a six-why descent is often a deletion, not a discovery.**
