# WHYS — the residual 1.17× encode gap

**Unknown:** with flash attention removed from both sides
([`encode-gap.md`](encode-gap.md)), whisper.cpp still encodes 1.17× faster.
Why?

Method: `codec-six-whys-unknowns`.

---

## D6 (first) — is the instrument sound, and are the arms comparable?

Two findings, both invalidating earlier work.

**D6a — the roofline's memory ceiling was 2.5× too low.**
`whisper-bench -w 1` measures this machine's memcpy at **33.57 GB/s (24
threads)**. Our anatomy bench calibrated its memory roof with candle's
single-threaded `Tensor::copy` and got **12–15 GB/s**.

Every "% of memory peak" in §6.10 and §6.15 was therefore computed against a
ceiling less than half the truth. Ops reported as running at "163 % / 206 % of
memory peak" — a nonsense figure I noted at the time and did not chase — were
in fact at ~60–75 % of the real roof. **The absurd number was the instrument
telling me it was wrong, and I read it as a curiosity.**

**D6b — ggml's matmul is slower than candle's, not faster.**
`whisper-bench -w 2`, this machine:

| size | ggml F16 | ggml F32 |
|---|---:|---:|
| 1024² | 210.0 GFLOPS | 7.5 GFLOPS |
| 2048² | 265.1 GFLOPS | 9.5 GFLOPS |
| 4096² | 250.5 GFLOPS | 9.5 GFLOPS |

Against candle's measured **615 GFLOP/s (F32)** and 421 (F16). **candle's
matmul is ~2.3× faster than ggml's best.**

This kills the narrative that has run through this campaign since §6.11 —
"ggml's hand-tuned kernels against candle's generic ones." It is exactly
backwards. We were never losing on matmul.

## D2/D4 — where does our encoder time actually go?

In-context sub-stages, one 30 s window:

| bucket | seconds | share |
|---|---:|---:|
| attention | 0.148 | **67.3 %** |
| MLP | 0.055 | 25.1 % |
| conv front end | 0.017 | 7.6 % |
| **residue** | **0.003** | **1 %** |

**The encoder has no framework overhead** — 1 % residue, against the decoder's
57 %. Whatever the 1.17× is, it is inside these three buckets, and two of them
are small.

## D5 — why is attention slow, given our matmul is the faster one?

FLOP accounting per 30 s window:

| bucket | GFLOP | measured | achieved | vs candle's 615 GFLOP/s |
|---|---:|---:|---:|---:|
| attention | 20.90 | 0.148 s | **141 GFLOP/s** | **23 %** |
| MLP | 14.16 | 0.055 s | 257 GFLOP/s | 42 % |

**Our attention runs at 23 % of our own framework's demonstrated matmul
capability.** The cause is structural, not a kernel defect: attention is six
*small batched* matmuls (per head: 1500×64 against 64×1500) plus a softmax
over 13.5 M elements. Batched small-K matmuls cannot reach the efficiency of
one large GEMM, and the softmax is pure traffic between them.

**Ceiling:** if attention merely ran at the rate candle already achieves on
large matmuls, it would take **34 ms instead of 148 ms**, putting the encoder
at **106 ms against whisper.cpp's 210 ms — we would win by 1.98×.**

---

## The answer

**The residual 1.17× and the 1.31× flash-attention gap are the same defect.**

Attention is 67 % of our encoder and runs at 23 % efficiency. Flash attention
is the technique that fixes precisely this — it keeps the score block in
registers, eliminating both the softmax's memory round-trip and the batched
matmul's inefficiency. whisper.cpp's `-fa`/`-nfa` toggle already measured what
that is worth *to a slower matmul than ours*: 1.31×.

So there are not two problems to solve. There is one:

- fix attention → close the 1.31× (flash) **and** most of the 1.17× (efficiency);
- the ceiling says the combined result is not parity but **~2× faster than
  whisper.cpp on encode**, because our matmul is 2.3× faster than theirs and
  attention is the only thing preventing that advantage from showing.

## Rebuild — the target is now precisely specified

- **Route:** `codec-vectorize-kernel` → `codec-asm-kernel`. A genuinely fused
  kernel, not chunked framework ops (§6.8 established that distinction the
  hard way, and [`encode-gap.md`](encode-gap.md) established that the prune
  did not generalize).
- **Prize:** 148 ms → 34 ms on attention if it reaches candle's own matmul
  rate; realistically less, but the reference proves ≥1.31× is attainable.
- **Tax:** unknown for this shape. The earlier hand-written fused attempt hit
  68 GFLOP/s, but on the *decoder* shape (one query row, no reuse). The
  encoder shape is what flash attention is designed for.
- **First measurement before writing the kernel:** a fused block for the
  encoder shape only, benchmarked against the current three-op sequence with
  interleaved paired A/B. If it cannot beat 141 GFLOP/s, stop.
- **Gate:** byte-identical transcripts, then stage-level paired A/B, then
  corpus WER.

## Corrections this descent forces

1. **§6.11's "backend-quality gap, ggml's hand-tuned kernels vs candle's
   generic ones" is wrong** — candle's matmul is 2.3× faster than ggml's.
2. **§6.10 and §6.15 roofline percentages are wrong** — the memory ceiling was
   under-measured by 2.5×. The *rankings* survive; the *percentages* do not.
3. **§6.6's "the remaining lever is not vectorization"** is wrong for the
   encoder, now for the second time and with a number attached.

## Four for four

| descent | D6 finding |
|---|---|
| overall gap | reference given 23 % less work by a flag |
| f16 SIMD | 118 % noise floor; candle *has* f16 matmul |
| encode gap | reference runs flash attention by default |
| **this one** | **memory ceiling under-measured 2.5×; ggml's matmul is the slower one** |

Every descent has terminated at depth 6, and this one found that the
instrument had been lying in a way that shaped three sections of conclusions.
The rule stands and hardens: **calibrate ceilings with something other than
the system under test.**

---

## The brick: built, measured, ABANDONED

The kernel exists and is **numerically exact** (max |Δ| = 6.1e-7 against the
candle three-op path) — real flash attention with online softmax over key
blocks, which is the piece §6.8's query-tiling could not do. It is simply not
fast enough:

| attempt | GFLOP/s | vs candle three-op | paired |
|---|---:|---:|---|
| naive inner loop | 39 | 0.55× | 0/21, z = −4.6 |
| + 4-way register blocking | 46 | 0.55× | 0/21, z = −4.6 |
| candle three-op (baseline) | 84 | — | — |

Register blocking bought 18 % and closed none of the gap. Reaching 84 GFLOP/s
would need a professionally tuned micro-kernel — packed operands, 8×8 AVX2
register tiles — which is precisely what the `gemm` crate already is.

## Why ggml wins with this and we cannot — the reconciliation

This looked like a contradiction: whisper.cpp's own `-fa`/`-nfa` toggle proves
flash attention is worth **1.31×**, yet our flash attention loses by 1.8×.
Both are true, and the reason ties the whole investigation together:

> **Flash attention is worth 1.31× to ggml because ggml's matmul is slow.**
> Its own bench measures 265 GFLOP/s against candle's 615. Routing *around* a
> slow GEMM pays. Routing around a fast one does not — and that is what we
> would be doing.

The technique's value is inversely proportional to the quality of the matmul
it replaces. We already hold the better matmul, so the same optimization has
far less to give us, while the hand-written kernel required to capture it must
beat a tuned GEMM that we would be discarding.

**This is the fourth independent confirmation of one rule** (§6.8 tiling,
§6.11 fusion-ceiling probe, §6.16 f16 K/V, this kernel): *on this stack,
anything that takes work away from candle's `gemm` loses.* It is no longer an
inference from one prune — it is a measured property with four data points and
a mechanism.

## What would actually be required

Not "abandon forever" — **abandon at this cost**. Capturing the prize needs
`codec-asm-kernel`: AVX2/FMA intrinsics, packed panels, 8×8 register tiling,
prefetch. Days of work to reimplement what `gemm` already does, in order to
win back a fraction that ggml only enjoys because its own GEMM is weaker.
On a value-per-day basis it ranks below every remaining item on the roadmap.

**Recorded as: measured worse, twice, with the mechanism understood** — not
"unproven". If candle ever gains a fused attention primitive, this becomes a
one-line change and the prize is waiting.

---

## RE-OPENED, and I was wrong TWICE — in opposite directions

The user challenged the "flash attention is not transferable" conclusion.
Re-measuring found two errors of mine, not one.

### Error 1 — the rationalization

I wrote: *"flash attention is worth 1.31x to ggml because ggml's matmul is
slow; routing around a fast matmul does not pay."* That was a story fitted to
two data points (their 1.31x, my kernel's 0.55x). It explained the numbers
without being true. The real reason my version lost is simpler: **my kernel was
naive.** Restructuring it so the contraction runs in the OUTER loop — turning
each score update into a vectorizable AXPY instead of a dot product ending in
a horizontal reduction — took it from **46 to 73 GFLOP/s, a 59 % gain from one
change.** The technique was never the problem.

### Error 2 — the ceiling, again

Having recanted, I then claimed **4.4x headroom** in encoder attention. That
used the 615 GFLOP/s ceiling measured on a **2048³ square** matmul. The
attention matmuls contract over **K = 64**, where there is far less reuse per
pass over the output, and candle achieves **201-397 GFLOP/s** there:

| ceiling used | implied floor | "headroom" |
|---|---:|---:|
| 615 GFLOP/s (square matmul — **wrong shape**) | 34.0 ms | 4.4x |
| ~210 GFLOP/s (**what candle reaches at K=64**) | 99.5 ms | **1.5x** |

And the traffic prize is smaller than it looked: 864 MB at 33.6 GB/s is
25.7 ms against ~160 ms of measured attention — **16 %**, not a multiple.

**This is the same class of error as D6a, made again in the same document.**
There it was a memory ceiling calibrated with a single-threaded copy; here a
compute ceiling calibrated with the wrong matrix *shape*. A roofline is only
valid for the shape it was measured at.

### The corrected picture

| | |
|---|---|
| real headroom in encoder attention | **~1.5x**, not 4.4x and not zero |
| of which traffic elimination | ~16 % |
| our fused kernel now | 73 GFLOP/s vs candle's three-op path at 86 — **0.84x** |
| what winning needs | beating a tuned GEMM at K=64 *and* banking the 16 % — `codec-asm-kernel` work |

So the honest verdict sits between my two wrong ones: **the prize is real but
modest (~1.5x on attention, ~1.2x on encode), and capturing it requires a
register-blocked SIMD kernel, not the two evenings I have given it.** The
kernel is correct (max |Δ| 5.5e-7) and 59 % faster than my first attempt;
it is 15 % short of the path it must replace.

**Both of my confident conclusions were wrong, in opposite directions, and
both were caught by measuring a ceiling properly.** The rule that keeps
earning its place: *calibrate ceilings with the shape you are actually
running.*
