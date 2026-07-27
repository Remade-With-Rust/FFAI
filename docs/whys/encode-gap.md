# WHYS — the 1.39× encode gap

**Unknown:** after §6.18's "missing f16 SIMD kernels" was retracted
([`f16-simd-kernels.md`](f16-simd-kernels.md)), the encode gap to whisper.cpp
had no explanation at all.

Method: `codec-six-whys-unknowns`.

---

## D6 (first, per the skill) — do both arms do the same work?

- **ASKED:** the last two descents both ended here. Is there *another* config
  difference in the encoder path?
- **MEASURED:** `whisper-cli -h`:

  ```
  -fa,  --flash-attn      [true  ] enable flash attention
  -nfa, --no-flash-attn   [false ] disable flash attention
  ```

  Encoder pass counts: whisper.cpp **20 runs**, ours **20 passes** — fair.
- **ANSWER:** **flash attention is ON BY DEFAULT** in whisper.cpp v1.9.1.
  Every encode comparison in this campaign has been our materialized
  attention against ggml's *fused* attention.
- **CONFIDENCE:** high — read from the tool's own help output.
- **STATUS:** closed. **Third consecutive descent to terminate at D6.**

## D1 — how much of the gap is that, measured?

- **ASKED:** toggle the reference's own flag and watch its encode time.
- **MEASURED:** 20 clips, alternating rounds:

  | round | `-fa` (default) | `-nfa` |
  |---|---:|---:|
  | 1 | 3018.0 ms | 4194.0 ms |
  | 2 | 3275.8 ms | 4460.2 ms |
  | 3 | 3195.3 ms | 4137.6 ms |

  Flash attention faster **3/3 rounds**, median **1.31×**.
- **ANSWER:** the decomposition:

  | | ms | vs ours |
  |---|---:|---:|
  | whisper.cpp, flash attention on (its default) | 3195 | **1.53×** |
  | whisper.cpp, flash attention off | 4194 | **1.17×** |
  | ours (no flash attention) | 4892 | — |

  **Flash attention accounts for ~69 % of the encode gap.** The residual
  implementation gap against a like-for-like ggml is **1.17×**, not 1.39×.
- **CONFIDENCE:** high — the reference toggles its own feature, so nothing
  about the two builds differs except the thing under test. This is the
  cleanest measurement in the campaign.
- **STATUS:** closed. **The unknown is answered.**

---

## Why this did not surface earlier — and why §6.8 is not a contradiction

We *did* try flash attention (§6.8, query tiling) and measured it
**monotonically slower**, then pruned it. Both results are correct, and the
difference between them is the whole lesson:

| | §6.8 query tiling | ggml `flash_attn_ext` |
|---|---|---|
| built from | chunked candle tensor ops | one hand-written fused kernel |
| per block | 3 framework calls + allocation | registers, no intermediate |
| score matrix | still materialized, in pieces | **never materialized** |
| result | 1.28× **slower** | 1.31× **faster** |

Tiling a framework's ops is not flash attention; it is the same work in
smaller, less efficient pieces, plus dispatch. Flash attention is a *kernel*.
§6.8's conclusion — "anything that takes the matmul away from candle's tuned
`gemm` loses" — is true for everything built out of `gemm` calls, and says
nothing about a kernel that replaces them entirely. The prune was right; the
generalization drawn from it was too broad, and it is what stopped this being
found three sessions ago.

## Rebuild — ceiling first (skill rule 1)

- **Prize, measured on the reference rather than estimated:** 1.31× on encode.
  Encode is ~35 % of total time, so ≈ **10 % end-to-end** — and unlike every
  other candidate this campaign produced, the reference has already
  demonstrated it is achievable on this exact hardware, model and workload.
- **Tax, partially known:** our earlier hand-written fused attempt reached
  68 GFLOP/s against candle's 285 — but that was the **decoder** shape
  (one query row), where there is no reuse to exploit and fusion cannot pay.
  The **encoder** shape (1500 queries × 1500 keys) is precisely what flash
  attention exists for, and ggml's 1.31× is the existence proof.
- **Route (skill rebuild step 2):** `codec-vectorize-kernel`, then
  `codec-asm-kernel` if intrinsics top out. This is a genuine fused-kernel
  job — the first one in this campaign that the evidence supports.
- **Gate:** transcripts byte-identical (attention output is deterministic
  given the same maths), then interleaved paired A/B at the stage level, then
  corpus WER.

## Standing conclusions to correct

1. §6.18's retraction stands, but its replacement — "the encode gap is
   unexplained" — is now resolved: **69 % flash attention, 31 % implementation.**
2. §6.6's "the remaining lever is not vectorization" was wrong for the
   encoder. It is *exactly* vectorization, of a kernel we do not yet have.
3. The **real** like-for-like standing against ggml is **1.17× on encode**,
   which is much closer than any number this campaign has quoted.

## The pattern, three for three

Every descent run with this skill has terminated at depth 6, on a
configuration difference rather than a code defect:

| descent | D6 finding | worth |
|---|---|---|
| overall gap | reference ran with `-nt` — 23 % less decode work | a third of the headline |
| f16 SIMD | machine noise floor 118 %; then candle *does* have f16 matmul | deleted an upstream project |
| encode gap | reference runs flash attention by default | 69 % of the encode gap |

**The instrument and the configuration are where codec campaigns actually
lose time** — not in the kernels everyone reaches for first. Depth 6 is not
the last thing to check. On this evidence it should be the first.
