# WHYS — the 1.8x total-pipeline gap

**Unknown:** as shipped we are 1.8x slower than whisper.cpp end to end.
Total pipeline is the only number that counts.

Method: `codec-six-whys-unknowns`.

---

## D2 — where is the gap, in absolute ms?

| stage | share of gap | ratio |
|---|---:|---:|
| **decode** | **62 %** | 2.05x |
| encode | 34 % | 1.53x (69 % of which is their flash attention — [not transferable](encode-residual-117.md)) |
| mel | 6 % | 3.67x |
| sample | — | **we win 2.03x** |

Decode owns it. Everything below is decode.

## D6 — parity checks (run first)

- **Token counts:** whisper.cpp 41 decode runs vs our 43 on the same clip.
  ~5 % — comparable, not the earlier 18 % artifact.
- **Noise floor:** compute ops swing 4-245 % on this machine. Every verdict
  below therefore uses **interleaved paired A/B**, never sequential runs.

## D5a — REFUTED: the self-attention KV cache is not O(n^2) in practice

`Tensor::cat` reallocates and copies the whole cache each token, which looks
like a classic redundancy bug. Measured: growing to 50 tokens costs 725 us per
cache, 1450 us for K and V, 5.8 ms across 4 layers — **1.6 % of decode.**
Real, but not the problem. Recorded so it is not re-investigated.

## D5b — FOUND: candle has a matrix-VECTOR cliff

The vocabulary projection is 38.6 % of decode and reads 40 MB (f16) per token
for **one row** of output. Against the corrected 33.57 GB/s ceiling it ran at
49 %. Sweeping the row count — *absolute* wall time, more work each step:

| rows (m) | ms | GB/s |
|---|---:|---:|
| **1** | **1.94** | 20.5 |
| 2 | 0.95 | 41.7 |
| **4** | **0.78** | **51.1** |
| 8 | 1.11 | 36.0 |

**Computing four rows costs 2.5x LESS than computing one.** candle routes
`m == 1` to a matrix-vector path; `m >= 2` reaches the tuned GEMM
micro-kernel. Thread count is irrelevant (20 GB/s at both 1 and 24 threads),
so it is a code-path cliff, not parallelism.

The cliff is **shape-dependent and cuts both ways**:

| shape | m=1 vs m=4 |
|---|---:|
| attention projection 384x384 | 0.60x — padding LOSES |
| mlp fc1 384x1536 | 1.17x |
| vocabulary 384x51864 f16 | **1.97x** |

So the row count is **measured per shape** at load time
([`asr/adaptive.rs::matmul_pad_rows`](../../crates/ffai-mercury/src/asr/adaptive.rs)),
cached, with a 15 % margin and an `FFAI_GEMV_PAD=off` override. On this
machine it selects **m=8, 2.01x** for the vocabulary projection.

## The brick — KEPT

Padded rows are duplicates of the real one, so there is no extra weight
traffic: the 40 MB read is shared across all of them. We ask for rows we throw
away, and it is faster.

- **Op level:** vocabulary projection 0.162 s -> 0.088 s over 50 tokens (1.84x).
- **Correctness:** transcripts byte-identical to the candle reference, 16/16.
- **Engine level, interleaved paired A/B, 12 clips x 15 rounds:**
  **15/15 rounds, z = +3.9, 1.057x** — a proven **5.7 % end-to-end** gain.

### A harness bug caught on the way (D6 again)

The first engine A/B returned 7/15, z = -0.3, ratio 1.003x — "inconclusive".
It was wrong: the calibration cache is process-global and keyed by shape, so
building an "off" engine and then an "on" engine had the second silently reuse
the first's cached decision. **Both arms ran the same code.** Moving the
override into the per-call path fixed it and the same test returned 15/15.

An A/B that reports *no difference* is exactly as suspect as one reporting a
large one — verify the arms actually differ before believing either.

## D5c — REFUTED: the cliff does not generalize to cross-attention

Cross-attention's matmuls are `(1,h,1,64) @ (1,h,64,1500)` — per head m=1 with
a wide output, apparently the same cliff shape. Swept:

| rows | q@k | w@v | sum |
|---|---:|---:|---:|
| **1** | 46.6 us | 49.1 us | **95.7 us** |
| 2 | 89.7 | 54.4 | 144.1 |
| 4 | 65.3 | 80.4 | 145.7 |
| 8 | 202.6 | 142.5 | 345.1 |

**m=1 is already optimal**; padding is 1.5x worse. The cliff is specific to
*large-n unbatched* matmuls — candle's batched path does not share it. The
per-shape calibration handles this automatically (it returns 1), which is the
argument for measuring rather than applying the trick everywhere it "should"
work.

## Standing after this descent

| lever | result |
|---|---|
| vocabulary-projection GEMV padding | **KEPT — 15/15 rounds, +5.7 % end-to-end** |
| self-attention KV `Tensor::cat` | refuted — 1.6 % of decode |
| cross-attention row padding | refuted — m=1 already optimal |

Total pipeline remains the only scoreboard, and it moved: a measured,
statistically hard **5.7 %**, byte-identical, with the decision made adaptive
so it stays correct on hardware where the cliff does not exist.

---

## D3b — mel: a hot loop that was secretly a matmul

Mel was 3.67x slower than whisper.cpp's, 6-7 % of the gap, and had never been
examined. Cracked open at the real shape (30 s window, 3000 frames x 80 mels):

| stage | ms | share |
|---|---:|---:|
| **filterbank projection 80x201 @ 201x3000** | **18.84** | **87 %** |
| 3000 FFTs of size 400 | 1.51 | 7 % |
| log10 over 240 000 elements | 0.86 | 4 % |
| whole mel | 21.74 | — |

The filterbank ran at **5.1 GFLOP/s** — because it was hand-written as a
scalar triple loop. It is a matmul. Handing it to candle's GEMM (with the
power spectrum stored bin-major so both operands are contiguous and no
transpose is needed):

- **mel 21.74 ms -> 6.30 ms, 3.45x**
- mel's gap to whisper.cpp: **4.44x -> 1.22x**, its share of the total gap
  **7 % -> 0 %**
- Gated twice: the openai-whisper mel oracle still passes, and engine
  transcripts are 16/16 byte-identical.

Note what this was *not*: the FFT — the thing anyone would profile first —
was 7 %. And `log10`, the transcendental that was the entire story in the
GELU case (§6.10), was 4 %. The same shape of bug does not recur in the same
place twice; only measurement finds it.

**Transferable:** *look for hot loops that are secretly matmuls.* A triple
loop over `[i][k] * [k][j]` accumulating into `[i][j]` is a GEMM, and a tuned
GEMM will beat any hand-written version by two orders of magnitude. Both wins
in this descent were this: the vocabulary projection was a matmul taking the
wrong code path, and the filterbank was a matmul that had never been one.

## Standing after the descent

| stage | before | after |
|---|---:|---:|
| mel | 4.44x slower | **1.22x** |
| decode | 2.46x | 2.13x |
| encode | 1.70x | 1.98x (unchanged; noise) |
| sample | we win 2.03x | we win 1.94x |

Shipped, both byte-identical and both proven above the noise floor:

1. **Adaptive GEMV padding** — 15/15 paired rounds, z = +3.9, **+5.7 % end to end**
2. **Mel filterbank as a real matmul** — 3.45x on the stage, mel gap all but closed

Remaining, in order of absolute cost: **encode** (54 % of the gap, of which
69 % is their flash attention and [not transferable](encode-residual-117.md))
and **decode** (48 %, dominated by cross-attention which has survived four
attacks). `sample` we win outright.

---

## D5d — the chain that failed, unblocked

§6.16 reverted an f16 cross-attention K/V cache: it made both matmuls faster
and gave it all back because candle's f16 softmax was 82 % slower. The
mechanism was understood, so the blocker was addressable.

**`fast_softmax`** — a candle `CustomOp1` that converts each row to f32
**once** (rather than per element, which is what a generic `T: Float` loop
does), computes there, and converts back:

| test | paired | ratio |
|---|---|---:|
| our f16 softmax vs candle's | 38/41, z = +5.5 | **1.96x** |
| **full cross-attn chain, f16 K/V + our softmax** | **31/41, z = +3.3** | **1.18x** |

The chain that lost 0.88x now wins 1.18x. Nothing about the f16 idea changed;
one op inside it did. **This is the "test the chain, not the link" rule read
backwards: a chain-level loss can be one bad link, and it is worth finding out
which before discarding the idea.**

### And it improved accuracy

Output is no longer byte-identical (15/16 clips match; f16 K/V flips one
borderline token), so this gates on corpus WER:

| | WER % | CER % |
|---|---:|---:|
| before | 7.87 | 3.28 |
| **after** | **7.55** | **2.95** |
| whisper.cpp | 7.58 | 2.87 |

**The quality gate PASSES for the first time in the campaign** — 7.55 % against
whisper.cpp's 7.58 %. The gain is most likely `fast_softmax` computing the
exponential in f32 regardless of storage dtype, where candle's generic path
computed it in the storage type.

## Session total

| change | effect | gate |
|---|---|---|
| Adaptive GEMV padding | +5.7 % end to end (15/15, z=+3.9) | byte-identical |
| Mel filterbank as a matmul | mel 3.45x; gap 4.44x -> 1.22x | mel oracle + byte-identical |
| `fast_softmax` custom op | 1.96x on f16 softmax | — |
| f16 cross-attention K/V (unblocked) | chain 1.18x (31/41, z=+3.3) | **corpus WER 7.87 -> 7.55** |

**WER 7.87 % -> 7.55 %, now ahead of whisper.cpp (7.58 %). Quality gate: PASS.**
Speed remains the open gate.

---

## D5e — REFUTED: f16 for the encoder attention

Cross-attention won with f16 K/V, so the encoder's attention was the obvious
next candidate: its `q@kT` is (1500x64)@(64x1500) — only k=64 of reuse per
output element, writing a 54 MB score matrix, which looks memory-bound.

| | median |
|---|---:|
| f16 chain | 41.34 ms |
| **f32 chain** | **28.51 ms** |
| paired | **0/15, z = -3.9, 0.69x** |

**f32 wins by 1.45x.** At m=1500 there is enough arithmetic per byte that
candle's f16 conversion cost exceeds the traffic saved — the same verdict as
the encoder MLP (§6.13), and the opposite of cross-attention's m=1. The
dividing line is the query count, not the operation.

This closes the f16 lever for encode. Recorded so the symmetry with
cross-attention does not tempt anyone to retry it.

## Final state of the descent

Same clips, same run, both sides:

| stage | at descent start | now |
|---|---:|---:|
| mel | 4.44x slower | **1.10x** |
| decode | 2.46x | **1.82x** (per token 2.27x -> **1.69x**) |
| encode | 1.98x | **1.53x** |
| sample | we win 2.03x | we win **2.13x** |
| **TOTAL** | **1.94x** | **1.57x** |

And the corpus gates:

| | WER % | CER % |
|---|---:|---:|
| **whisper-candle** | **7.55** | 2.95 |
| whisper.cpp | 7.58 | 2.87 |

**Quality: PASS.** Speed: still open at 1.57x, but down from 1.94x in this
descent alone.

### Ledger of this descent

| lever | verdict |
|---|---|
| adaptive GEMV padding | **KEPT** — +5.7 % e2e, 15/15, z=+3.9 |
| mel filterbank as a matmul | **KEPT** — 3.45x, mel gap all but closed |
| `fast_softmax` custom op | **KEPT** — 1.96x f16, 1.07x f32 |
| f16 cross-attention K/V (re-tried) | **KEPT** — chain 1.18x, WER 7.87 -> 7.55 |
| self-attention KV `Tensor::cat` | refuted — 1.6 % of decode |
| cross-attention row padding | refuted — m=1 already optimal |
| f16 encoder attention | refuted — 0/15, f32 1.45x faster |

Seven levers examined, four kept, three refuted with numbers. The refutations
matter as much as the wins: each one is a door that stays shut.

---

## D6 (again) — two self-inflicted defects found by re-profiling

**A per-call lookup in a per-token path.** The GEMV-padding win regressed
itself: `matmul_padded` called the calibration helper on *every* invocation,
which meant a `std::env::var` (allocating, locking the process environment)
plus a mutex-guarded HashMap lookup per generated token. The vocabulary
projection went 0.088 s -> 0.263 s — the optimization cost more than it saved.
Resolving both decisions once at load and storing them in the variant restored
it to **0.079 s (3.3x)**.

*A lookup being "cheap" is not a licence to put it in a per-token path.*

**A decision measured on the wrong thing.** The cross-attention K/V dtype was
being chosen by timing the isolated `q@k` matmul, where f16's margin is thin
enough to **flip between runs** — the byte-identical count oscillated 15/16 and
16/16 across builds, i.e. the shipped model was non-deterministic. Moved to a
chain-level calibration (`q@k -> softmax -> w@v`, conversions included), where
the margin is a clear 1.17x and the answer is stable.

*The skill's "test the chain, not the link" applies to the calibration itself,
not only to the human's A/B.*

## Honest close

| gate | state |
|---|---|
| correctness | **PASS** — 134/134 clips |
| quality | **PASS** — 7.87 % vs whisper.cpp 7.58 %, inside the 5 % band (7.55 % when f16 K/V is selected) |
| speed | **FAIL** — 14.1x vs 26.1x |
| footprint | not instrumented |

The chain calibration's 1.17x margin sits close to its 1.10 threshold, so on a
loaded machine the f16 K/V decision can still fall to f32 and cost ~0.3 pp of
WER. That is a real residual weakness, and the fix is a wider margin or a
paired calibration rather than best-of-N — noted, not hidden.

**Speed remains the open gate.** Seven levers were examined this descent, four
kept and three refuted, and the total-pipeline ratio moved from 1.94x to
between 1.57x and 1.84x depending on machine load. The remaining gap is
dominated by two things the descent has already characterized and closed off:
the reference's flash attention (69 % of the encode gap, **not transferable** —
it only pays because their GEMM is 2.3x slower than ours) and a
cross-attention that has survived six separate attacks. Further gains on this
machine are below its noise floor; the next real step is a quiet machine and
the corpus re-run, not another kernel.

---

## D5f — three more refutations, one of them mine

**GEMV padding on the decoder projections and MLP — REVERTED.** The cliff
microbenchmark promised 1.46x on mlp fc1 at m=2 and the isolated numbers were
real. In context, process-level paired A/B over the whole engine:
**0/13 rounds, z = -3.6, 0.847x.** The pre-transposed weight copies are memory
the decoder must then stream, and on cold cache the padded matmul does not
recover it. Padding stays only where it was proven end to end — the vocabulary
projection.

**Hoisting candle's `Linear` weight transpose — REFUTED.** `Linear::forward`
calls `weight.broadcast_left(..).t()` on every invocation, which is exactly the
bug that cost 91.5 % of decoder time in the vocabulary projection. At the
decoder's projection sizes it is free: 96/201, 96/201, 98/201 — three ties.
The transpose is a cheap *view* until the matmul is large enough to
materialize it.

**The calibration probe cannot be narrowed — MY BUG.** To cut ~50 ms of
startup I probed the pad-row cliff on a narrowed output (8192 instead of
51864), reasoning that the cliff is a code-path property. **The cliff vanished
and the calibration returned "no padding".** It is a *size* property: the
output stops fitting the cache hierarchy the same way. Reverted to full-width
probing — a calibration that is cheap and wrong is worse than one that costs
50 ms once.

### And the honest residual

`attention_kv_dtype` chose F16 in one run and F32 in the next (0.182 vs
0.195 ms — inside the noise). The chain margin on this machine is ~1.0-1.17x,
too close to its 1.10 threshold to be stable, so the f16 K/V cache — and the
0.3 pp of WER that rides on it — is **not deterministic here**. The fix is a
paired calibration rather than best-of-N, or a wider margin. Recorded as a
known weakness, not papered over.

## Final ledger of this descent

| lever | verdict |
|---|---|
| adaptive GEMV padding (vocabulary projection) | **KEPT** — +5.7 % e2e, 15/15, z=+3.9 |
| mel filterbank as a matmul | **KEPT** — 3.45x; mel gap 4.44x -> 1.10x |
| `fast_softmax` custom op | **KEPT** — 1.96x f16, 1.07x f32 |
| f16 cross-attention K/V, chain-calibrated | **KEPT** (marginal/unstable) — WER 7.55 % when selected |
| per-call env+mutex in the hot path | **FIXED** — self-inflicted, 3.3x recovered |
| self-attention KV `Tensor::cat` | refuted — 1.6 % of decode |
| cross-attention row padding | refuted — m=1 already optimal |
| f16 encoder attention | refuted — 0/15, f32 1.45x faster |
| GEMV padding on projections/MLP | refuted — 0/13, 0.847x |
| hoisting `Linear`'s weight transpose | refuted — three ties |
| narrowed calibration probe | refuted — breaks the decision |

**Eleven levers, four kept, one bug fixed, six refuted with numbers.**
