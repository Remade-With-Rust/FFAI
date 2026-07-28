# M-X2 batched encoder — BUILT, correct, and worth ~0 for speed

**Date:** 2026-07-28
**Verdict:** implemented; correctness and null-arm gates PASS; speed prize
measured at 1.00-1.06x, i.e. nothing. The milestone's own rule was "must move
xRT materially or it is pruned like §6.8" — by that rule the *optimisation* is
pruned, while the *capability* ships because building it fixed a live
correctness bug (§7).
**Cost:** four profiler runs, a window count, two probes, and one afternoon.

---

## 1. What M-X2 specified

> *Independent segments → batch the encoder. Exit gate: transcripts
> byte-identical to unbatched; speed must move ×RT materially or it is pruned
> like §6.8.*

The premise, taken from WhisperX: VAD produces independent speech segments,
those segments batch through the encoder, and the batching is where the
headline throughput comes from.

---

## 2. The blocker: there is nothing to batch

Encoder invocations per clip, measured across the corpus:

```
  10.4s -> 1 encoder call      5.4s -> 1 encoder call
   7.9s -> 1 encoder call      4.8s -> 1 encoder call
   4.0s -> 1 encoder call      9.0s -> 1 encoder call
   4.4s -> 1 encoder call     11.2s -> 1 encoder call
```

Every clip is **one** 30 s window. Mean clip duration is 7.5 s and Whisper's
context is 30 s, so VAD packing puts each clip's speech into a single window —
by design, since `pack()` closes a window only when adding the next region
would exceed the context.

A batch of one is not a batch. **On this corpus the applicable work is zero,
whatever the implementation.**

---

## 3. And the target stage is already saturated

Even where multiple windows exist — long-form audio — the encoder is the wrong
place to look. From §6.10's anatomy, its dominant ops at batch 1:

| op | bound |
|---|---|
| mlp fc1 / fc2 | **CPU-bound, 87–94 % of compute peak** |
| qkv / out projection | **CPU-bound, 84 % of compute peak** |

Batching raises arithmetic intensity: more work per weight load. That helps an
op starved of work. It cannot help an op already issuing FLOPs at 90 % of what
the machine can retire — the headroom to 100 % *is* the ceiling, ~1.11×, and
only on part of the stage.

The reason is shape. The encoder's matmuls are already large: `(1500 × 384) @
(384 × 1536)`. Batching takes M from 1500 to B·1500, and GEMM efficiency at
M = 1500 is already at its asymptote. **Nothing about batching changes a
well-shaped large GEMM.**

This is where WhisperX's number comes from and why it does not transfer: on a
GPU, batch 1 leaves most of the device idle, so batching is close to free
throughput. On CPU at 90 % of peak, the machine is already busy.

---

## 3b. And measured directly, not just inferred

§3 leaned on the roofline table from mission-plan §6.10 — an existing
measurement, not one taken *as batching*. A borrowed number is not a probe, so
`examples/batch_ceiling.rs` measures it: stack B copies of one 30 s mel window
and time the encoder, best-of-3, three independent probes.

Our own encoder cannot serve as the instrument — `conv1d_gemm` would silently
return item 0 — so this runs candle's reference encoder, which goes through
`candle_nn::conv1d` and is genuinely batch-capable. That answers "does this
machine reward batching this shape" without first committing to a kernel
rewrite.

**Per-window encoder time, ms:**

| batch | probe 1 | probe 2 | probe 3 |
|---:|---:|---:|---:|
| 1 | 288.8 | 291.4 | 293.0 |
| 2 | 279.8 | 281.2 | 294.5 |
| 4 | 294.6 | 286.7 | 285.8 |
| 8 | 285.0 | 283.3 | 286.8 |

Flat from 1 to 8. Best speedup anywhere is **1.036×**, and the response is
non-monotonic inside every probe (up at 2, down at 4, up at 8) — which by this
campaign's own rule means noise, not a trade to fit a curve to. The refutation
bar was set at 1.15× before running; nothing came close.

A fourth run taken from a cold cache read 1.065× / 0.921× / 1.056× — wider
scatter around the same flat line, and worth recording as the reason the
probe warms before timing.

**The prune's second leg is now measured, not inferred.**

## 4. Where the prize actually is, if anywhere

Measured stage split (tiny.en, one clip, `FFAI_PROFILE=1`):

```
mel 2.6 %   encoder 47.8 %   decoder 45.8 %   sampling 3.8 %
```

and inside the decoder:

| op | share of decoder | of total |
|---|---:|---:|
| cross-attn | 44.2 % | 20.2 % |
| mlp | 25.2 % | 11.5 % |
| self-attn | 17.6 % | 8.1 % |
| final-proj | 12.7 % | 5.8 % |

The decoder is where batching pays on a CPU, and it is the **opposite** of the
encoder's situation: every decoder op is a GEMV — one token, M = 1 — which
streams the full weight matrix per token. The vocabulary projection alone
moves ~80 MB per token (§6.13). Those are memory-bound, so batching B windows
amortises one weight read across B tokens.

Rough ceiling if B windows could decode in lockstep: `mlp` + `final-proj` is
17.3 % of total, going perhaps 2–3× → **~11 % of total**, plus some of the
projections inside attention.

Two things make that harder than it sounds, and neither is addressed by the
milestone as written:

- **Windows decode different numbers of tokens.** Lockstep batching needs
  padding, masking and per-item early exit. That is real machinery, and it
  changes the decode path — precisely where the byte-identical gate is
  hardest to hold.
- **It still needs ≥ 2 windows**, so §2's blocker applies unchanged.

---

## 5. Proposed redefinition

M-X2 as written targets the encoder, which is saturated, using a batch size
that on this corpus is always 1. Both halves are wrong. What survives:

**M-X2′ — throughput across *files*, not windows.** Decode N independent
clips concurrently, sharing weight reads in the decoder. This is the workload
that actually has parallelism (a transcription queue, a batch job), and it is
the honest version of WhisperX's claim on a CPU.

It cannot be gated by the current harness, which measures per-clip ×realtime
and would show nothing. That is a harness change, and it raises a fairness
question worth settling *before* building: whisper.cpp is not batching, so a
batched-versus-unbatched comparison measures a different thing than the
per-clip gates do. It belongs beside them as a throughput number, not inside
them.

**Recommendation: defer M-X2′ until there is a workload that needs it.** No
current corpus, gate, or user-facing claim depends on multi-file throughput.

---

## 6. What the analysis cost, and the rule it confirms

Four profiler runs and a window count, before any kernel was touched. Had the
prize been real the modification would have been substantial: `conv1d_gemm` flattens and indexes
as `(cin, l_in)` with no batch stride, and the transposed-K attention path
reshapes to `(1, heads, hd, seq)`. Both silently assume batch 1 while the
encoder's `forward` documents `(batch, n_mels, frames)`. Extending four
hand-written AVX2 kernels to batch *for speed*, on a measured prize of
approximately zero, is exactly what the prune-on-arithmetic rule exists to
prevent — and §7 records what happened when the correctness case for building
it turned out to be independent of the speed case.

Same outcome as §6.8 (query-tiled attention) and §6.25: **compute the prize
before writing the code.** The prize here is `stage share × speedup`, and
`speedup` was capped at 1.11× by a roofline measurement that already existed
in the mission plan, on a stage whose applicable batch size was 1.

Worth noting the near-miss: the encoder's signature *says* it takes a batch
dimension. Trusting the signature instead of reading `conv1d_gemm` would have
produced a batched call that silently returned results for item 0 only — no
error, no shape mismatch, just wrong transcripts for every window after the
first.

---

## 7. Built anyway — and it was worth building

The analysis above says batching buys no speed, and that held: our AVX2 path
measures **136.6 / 140.1 / 128.7 ms per window** at batch 1 / 2 / 4 —
1.00× / 0.98× / 1.06×, matching the independent ceiling probe.

It was implemented regardless, because §6's near-miss was not hypothetical. It
was a **live bug**:

```rust
// before — signature says (batch, n_mels, frames), code says otherwise
let src: Vec<f32> = x.flatten_all()?.to_vec1()?;
for c in 0..cin {
    let s = &src[c * l_in..(c + 1) * l_in];   // no batch stride
```

Any caller passing batch > 1 received **item 0's features for every item** —
no error, no shape mismatch, no panic. The encoder's own doc comment promised
a batch dimension it did not honour. The fused encoder-attention path carried
the same assumption in `kt.reshape((1, heads, hd, seq))`.

**What shipped:**

- `conv1d_gemm` does one im2col across the batch into a single
  `(cin·3) × (batch·l_out)` GEMM — the weight matrix is read once for the
  whole batch rather than per item.
- Encoder attention loops per item, because the measurement says there is no
  shared work between windows worth restructuring four AVX2 kernels for.
- The batch-1 path is unchanged in shape and allocation.

**Both of the milestone's gates pass, on our own kernels:**

| gate | result |
|---|---|
| correctness — byte-identical, not "close" | **PASS**, `max |Δ| = 0.000e0` on all 4 windows |
| null arm — batch of 1 reproduces unbatched | **PASS**, bit-exact |

The correctness gate uses four **distinct** windows on purpose, and asserts
they are distinct before comparing: with identical inputs, the exact bug being
guarded against would have passed silently.

`examples/batch_encoder.rs` runs both gates and the speed table.

**The distinction worth keeping.** "Batching is not worth building" and
"batching is not worth *enabling for throughput*" are different claims, and
only the second survived. The capability is correct and now provable; the
optimisation is not there, and the ms/window column says so in the same
breath. Shipping the first while honestly reporting the second is the
outcome — not a speed win, and not a silent landmine either.

One incidental datapoint from running both probes: our AVX2 encoder is
**136.6 ms/window against candle's 285 ms/window**, 2.1× faster on the same
shape and machine.
