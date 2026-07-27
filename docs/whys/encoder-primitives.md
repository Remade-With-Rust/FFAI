# WHYS — the encoder primitives against whisper.cpp

**Unknown:** with `OPEN.md`'s levers settled, why is the pipeline still ~1.2×
behind whisper.cpp, and specifically what is wrong with `scores q@kT`,
`softmax`, `attn@v` and `qkv/out proj`?

Method: `codec-six-whys-unknowns`, depth 6 first. Companion to
[`open-campaign.md`](open-campaign.md).

---

## D6c — is the comparison fair? (asked first, again)

- **ASKED:** three previous descents terminated here. What does whisper.cpp's
  own configuration say this time?
- **MEASURED:** `whisper-cli -h` plus the init log:
  - `-fa, --flash-attn [true]` — **on by default**, and our reference adapter
    does not pass `-nfa`. §6.18's finding is live in every number we quote.
  - `whisper_backend_init: using BLAS backend` — a 51 MB OpenBLAS build.
  - `ggml-cpu-alderlake.dll` chosen from nine ISA variants; `system_info`
    reports **AVX_VNNI = 1** and **REPACK = 1**.
  - `use gpu = 1` but `no GPU found` — CPU path confirmed.
  - **41 decode runs on both sides** for the same clip — the token-count check
    that caught the `-nt` disaster passes.
- **ANSWER:** fair on work done, structurally unequal on capability. They have
  fused attention, an int8 path with VNNI, blocked weight repacking, and an f16
  model. Those are differences to **price**, not defects to fix.
- **CONFIDENCE:** high — read from their help text and init log.
- **STATUS:** closed.

## D2b — which stage owns the gap, in ABSOLUTE ms?

- **MEASURED:** best-of-7 both sides, same clip, matched settings:

  | stage | whisper.cpp | Mercury | ratio | gap |
  |---|---:|---:|---:|---:|
  | encode | 132.8 | 159 | 1.20× | **+26.2 ms** |
  | decode | 100.7 | 120 | 1.19× | **+19.3 ms** |
  | mel | 5.4 | 7 | 1.29× | +1.6 ms |
  | sample | 20.3 | 10 | 0.49× | **−10.3 ms (we win 2.0×)** |

- **ANSWER:** encode and decode are **co-equal**. There is no single stage
  whose fix wins. Ranking by *ratio* would have pointed at mel — 1.29×, the
  worst ratio on the board, worth 4 % of the gap.
- **A correction this produced:** one profiled run of ours (142.75 ms) against
  their median (132.14) briefly read as "1.08× behind". Best-of-7 both sides
  says **1.20×**. Never pair a single run against a median.
- **STATUS:** closed.

## D3b — how much of encode is an algorithm we lack?

- **MEASURED:** their own flag, three runs each. Encode **132.8 ms** with flash
  attention, **218.8 ms** without — **1.65×**. Decode unaffected (111.8 vs
  116.5, inside spread).
- **ANSWER:** we are **1.38× faster than their unfused encoder** and 1.20×
  behind their fused one. The §6.22 kernel is real and has captured most of
  what the technique buys; the residue is implementation, not a missing idea.
- **CONFIDENCE:** high — one variable, their build, their flag.
- **STATUS:** closed.

## D6d — our roofline was wrong on BOTH axes, and had been for months

- **ASKED:** the anatomy bench printed `softmax 220 % of memory peak` and
  `attn@v 158 %`. Impossible. §6.9 corrected this in prose — did the correction
  ever reach the instrument?
- **MEASURED:** no.
  - **Memory axis:** calibrated with candle's single-threaded `Tensor::copy`
    → 9 GB/s. ggml's `whisper-bench -w 1` measures this box at **30.8 GB/s**.
    Every "% of memory peak" was ~3.4× too high.
  - **Compute axis:** a 2048³ square matmul (734 GFLOP/s) scoring an attention
    kernel that contracts over **K=64**, where the batched shape tops out at
    **229 GFLOP/s** — 31 % of hardware. §6.20's error, still in the tool.
- **FIXED** — and the fix caught two further errors of mine: a single-head
  ceiling probe used for a 6-head batched op (printed "112 % of peak"), and a
  memcpy probe taking min-of-5 on a 64 MB buffer, which reported 66.9 GB/s by
  measuring L3. The tool now prints `<<< IMPOSSIBLE — ceiling is
  miscalibrated` instead of letting an impossible number pass as a curiosity.
- **CONSEQUENCE:** the `BOUND` column routes work — memory-bound to
  `codec-cache-tiles`, compute-bound to `codec-vectorize-kernel`. Corrected:

  | op | ms/pass | rate | reading |
  |---|---:|---:|---|
  | scores q@kT | 33.77 | 205 GFLOP/s | **89 % of what the SHAPE allows** — no implementation headroom |
  | softmax | 17.90 | 24.1 GB/s | 44 % of memory peak — the only one with real headroom |
  | attn@v | 16.07 | 430 GFLOP/s | 59 % of hardware |
  | qkv/out proj | 14.76 | 479 GFLOP/s | 66 % of hardware |

- **STATUS:** closed. Fourth ceiling error of the campaign; the first the
  instrument caught by itself.

## D4b — the glue, not the arithmetic

- **MEASURED:** isolated primitives reconciled against the in-context profiler:

  | op | isolated | in context | ratio |
  |---|---:|---:|---:|
  | qkv/out proj | 14.76 ms | 16 ms | 1.08× |
  | `ea prep` | 4.95 ms | **11 ms** | **2.2×** |
  | `ea merge` | 1.92 ms | **6 ms** | **3.1×** |

- **ANSWER:** `codec-analyzer`'s "in-context ns/call ≫ kernel ns/call". The
  shipped prep materialized three ~2.3 MB tensors per layer:

  ```rust
  (split_heads(&q)? * (scale * scale))?.contiguous()?,  // multiply THEN copy
  split_heads(&k)?.transpose(2, 3)?.contiguous()?,      // strided gather
  split_heads(&v)?.contiguous()?,                       // copy
  ```

- **MECHANISM:** *the projection writes `(seq, d)`; the fused kernel reads
  `(heads, seq, HD)` for q and v and `(heads, HD, seq)` for k. Nobody writes
  the layout the consumer wants, so every layer re-materializes ~7 MB purely to
  change indexing.* That is what ggml's graph executor plus REPACK avoids.
- **STATUS:** closed.

## D5b — the ceiling probe, brick by brick

`examples/enc_prep_ceiling.rs`, each arm deleting exactly the cost its brick
would delete (the rule: a probe that removes more prices a brick you are not
building):

| brick | saves/layer | saves/pass | % encoder | % pipeline |
|---|---:|---:|---:|---:|
| 1 fold `scale` into the q weight | 0.227 ms | 0.91 ms | 0.6 % | 0.3 % |
| **2 produce K transposed (`Wᵀ @ xᵀ`)** | **1.314 ms** | **5.26 ms** | **3.3 %** | **1.7 %** |
| 3 head-layout projection (custom op) | 0.995 ms | 3.98 ms | 2.5 % | 1.3 % |
| 1+2+3 | 2.536 ms | 10.14 ms | 6.4 % | 3.4 % |

The finding behind brick 2: `Wᵀ.matmul(&x.t())` costs **0.951 ms against 2.265
for projection + gather** — candle takes the strided `x.t()` view without
materializing it and writes the kernel's layout directly.

**`OPEN.md` §2 refuted fusing ALL of q/k/v transposed, and was right: the
kernel wants q as `(seq, HD)`, where `q[i*64 + t]` walks four contiguous cache
lines. That refutation never reached K ALONE — the one tensor the kernel
genuinely wants transposed.** A refutation constrains the approach it tested,
not the idea it belonged to.

Bricks 1 and 2 collapse into one tensor: k is consumed only by `q@k`, so
folding the scale into k's weight is equivalent to scaling q, and k's weight is
already being materialized for the transposed projection.

## D1b — the brick, and three instrument failures on the way

Landed as `key_wt_scaled` in `EncoderAttention`, escape hatch `FFAI_ENC_KT=off`.

| gate | result |
|---|---|
| vs candle reference encoder | 15/16 identical — **the pre-change baseline is also 15/16, same clip** → no regression |
| unit tests | 30/30 |
| pipeline paired A/B, 4 clips × 41 rounds | **29/41, z = +2.7, 1.053×** |
| pipeline paired A/B, 9 clips × 31 rounds | **24/31, z = +3.1, 1.045×** |
| corpus WER (test-clean-v2) | **PENDING — not default-on until this passes** |

**Three instrument failures, all mine, in one brick:**

1. Three runs of the profiler showed `ea prep` unchanged and I nearly recorded
   the brick as dead. Three runs is not a verdict.
2. A cross-process arm-by-arm profile comparison (five "new", then five "old")
   showed the new path *slower*. The "new" block rose monotonically
   0.167 → 0.200 — the machine heating through the block. **This is the exact
   instrument proved unsound at D6a of the companion descent, and I ran it
   anyway.**
3. Interleaved at N=10 it read 2/10 (z = −1.9) against the brick. At **N=25 it
   reads 14/25 (z = +0.60)** — inconclusive, sign flipped. The N=10 sample was
   noise and came within one decision of becoming a recorded refutation.

Final state: the cross-process stage probe cannot resolve a ~5 ms change on a
165 ms stage and is **silent, not contradictory**; the two in-process pipeline
probes agree at ~1.05 ×. Recorded as a provisional KEEP on pipeline evidence,
**pending the corpus quality gate** — a non-bit-identical change does not get
to be a default before that returns.

## What remains, sized

- `softmax` at 44 % of memory peak is the only one of the four ops with real
  roofline headroom — but in the shipped path it lives *inside* the fused
  kernel with a vectorized `exp`, so the anatomy figure is the cost of a path
  that no longer runs. Any attack must be inside the kernel.
- `scores q@kT` at **89 % of its shape ceiling** is closed to implementation
  work. Only a different algorithm (which is what fusion already is) moves it.
- Brick 3 (a projection writing the head-split layout directly) is worth 1.3 %
  of pipeline and needs a `CustomOp` — the largest remaining named item here.
- **Their Q8_0 path at 487 GFLOP/s, 1.74× their own F16, is the biggest
  capability gap on the board** — AVX_VNNI plus REPACK, which candle lacks.
  §6.5 measured int8 on our encoder as a 2.4× regression because candle's
  quantized kernels are built for the decode shape and re-quantize activations
  per call. The headroom is real and sits in a kernel family we do not own.
