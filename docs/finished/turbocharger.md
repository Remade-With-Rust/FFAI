# Turbocharger — systematically replacing scalar transcendentals

**Status:** plan, 2026-08-22. Written after the Argus vision campaign
(`argus-launch-plan.md` §19–20) turned one activation from **44.01 ms to
1.22 ms — 32x** — and, just as usefully, refuted four neighbouring ideas that
looked identical.

---

## 1. The one-sentence thesis

`exp`, `ln`, `log10`, `tanh`, `sin`, `cos` and `powf` **have no SIMD
instruction at any width**. Every call is a scalar libm call — ~40 ns, no lanes,
and a hard barrier to vectorising the loop it sits in. Replacing them with
range-reduced polynomials is the single highest-yield mechanical change
available in this workspace.

**`sqrt`, `min`, `max`, `mul`, `add` are the opposite** — SSE2 baseline, already
vectorised by the compiler. Rewriting those is how a campaign wastes a week
(mp3 `xrpow`: hand-AVX2 measured **0.97x** and was reverted).

That distinction is the whole targeting rule.

---

## 2. What already exists — read this before writing anything

Three independent implementations of this idea are already in the tree, all
**wired to production**, none aware of the others:

| crate | kernel | reachable from | AVX2 twin |
|---|---|---|---|
| `ffai-diana` | `silu::exp_fast` | `silu.rs:140` (sigmoid) | ✅ `silu_avx2.rs` |
| `ffai-mercury` | `fast_tanh_exp`, `fast_sigmoid` | `decoder_kernels.rs:1125` | ✅ same file |
| `ffai-argus` | `siglip::exp_poly` | GELU `CustomOp1` | ✅ `gelu_chunk_avx2` |

**Task 0 of this plan is therefore consolidation, not creation.** Three copies
of a range-reduced `exp` will drift, and two of them already differ in how they
handle the `round()` step. One `ffai-core::fastmath` (or a small `ffai-simd`
crate) with `exp_poly`, `tanh_poly`, `sigmoid`, `ln_poly` — each with its
scalar oracle test and its AVX2 twin — is the foundation everything below
depends on.

Existing SIMD infrastructure to reuse rather than reinvent:

```
ffai-argus/src/siglip.rs            ffai-mercury/src/asr/f16_gemv.rs
ffai-carmenta/src/conv3x3.rs        ffai-mercury/src/asr/flash_attn.rs
ffai-diana/src/direct3x3_avx2.rs    ffai-mercury/src/asr/vocab_int8.rs
ffai-diana/src/silu_avx2.rs         ffai-mercury/src/tts/decoder_kernels.rs
```

---

## 3. The targets, ranked

### Tier 1 — candle activations (17 call sites, every engine)

Each of these is candle computing a scalar libm call **per element**:

| call | count | what candle does |
|---|---:|---|
| `.gelu_erf()` | 7 | scalar `erf` per element |
| `.tanh()` | 6 | scalar `tanhf` per element |
| `.gelu()` | 3 | scalar `tanhf` per element |
| `.erf()` | 1 | scalar `erf` per element |

By crate: **mercury 35 tensor-op activations**, carmenta 4, diana 4, argus 1.

This is the tier Argus already proved: candle's `gelu` measured **44.01 ms** on
`(1,1024,3072)` against **1.22 ms** for a polynomial + AVX2 `CustomOp1` — the
same arithmetic, 32x apart, bit-identical between its own scalar and AVX2 paths.

The mechanism is fully general and the delivery is already written: a
`CustomOp1` that takes candle's `CpuStorage` directly, a `#[target_feature]`
kernel, a scalar fallback, a `*_matches_scalar` test.

### Tier 2 — hot scalar loops outside candle

Verified per-element, in a loop, on a shipping path:

| site | call | why it is hot |
|---|---|---|
| `mercury/asr/decoder.rs:858` | `((l - max)/temp).exp()` | **softmax over a 51 865-token vocab, every decode step** |
| `mercury/asr/decoder.rs:523,774` | `(v - denom).exp()` | per candidate token |
| `mercury/asr/flash_attn.rs` ×4 | `(*x - mx).exp()` | attention softmax — already has SIMD infra, so the exp is the odd one out |
| `mercury/tts/vits.rs:406` | `(*pj - m).exp()` | per element |
| `mercury/tts/vits.rs:799` | `.tanh()` | candle tensor op |
| `mercury/asr/mel.rs:217` | `v.max(1e-10).log10()` | per mel bin, every frame |
| `mercury/asr/fbank.rs:171` | `energy.max(AMIN).log10()` | per mel, per frame |
| `carmenta/svtr.rs:233` | `swish(...)` | SiLU per element, every layer |

### Tier 3 — do NOT touch (recorded so nobody spends a day here)

One-off setup arithmetic, correctly written as it is:

* `mel.rs:48,59,127` — hz↔mel and the Hann window, built once per filterbank.
* `fbank.rs:42,46,58` — the same, once.
* `svtr.rs:213` — one `powf` for the attention scale, per forward. (Better still:
  **fold it into the q weights at load**, as Argus did — that deletes it rather
  than accelerating it. See §5.)
* `decoder.rs:628` — length penalty, once per beam.
* `diana/depth_head.rs` — `exp`/`powf` on whole tensors via candle's ops, at the
  head only.

---

## 4. The rules this plan runs under

Earned the expensive way in `argus-launch-plan.md` §19–20. Each has a body.

### ★ Replace expensive arithmetic; do NOT re-implement a good kernel

The most important line here. Same polynomial, same `target_feature` dispatch,
same zero-copy hook, same author:

| op | result |
|---|---:|
| GELU (candle used scalar `tanhf`) | **32.4x** |
| softmax (candle already vectorised) | **0.5x**, refuted over **four** attempts |

Each softmax attempt fixed a real defect — the marshalling, the libm call, the
missing `target_feature`, the loop-carried `sum` — and it still lost by >10x.
**A win on one op is not a licence to rewrite its neighbour.** Before writing a
kernel, measure what the framework already does with that op.

### `round()` is SSE4.1, not SSE2

A range-reduced `exp` contains `round()`. Outside a `#[target_feature]`
function that lowers to a **call to `roundf`** — so a "no libm calls" kernel
quietly makes one libm call per element anyway. This cost a whole measurement
cycle: 12.6 M elements × ~15 ops at ~1 op/cycle ≈ 63 ms, exactly what it
measured. **Every polynomial kernel must be behind `target_feature` or it is
not a polynomial kernel.**

### An f32 reduction blocks vectorisation

`sum += e` is loop-carried and f32 addition is not associative, so LLVM keeps
the loop scalar **even inside an AVX2 function**. Split into N independent
accumulators — and know it is a reassociation, so gate on tolerance, not
bit-identity. (In the softmax case this made things *worse*; the split is a
tool, not a rule.)

### Delivery can cost more than the kernel

Routing a tensor through `to_vec1()` and `Tensor::from_vec` is a fixed per-call
tax. Argus's GELU: **3.03 ms → 2.30 ms** from switching to `CustomOp1`, then
**→ 1.08 ms** from writing src→out in one pass instead of copy-then-modify.
More than half the win was glue. **Find the zero-copy hook before benchmarking
the kernel, or you are benchmarking the glue.**

### Dispatch placement can flip the sign

A `#[target_feature]` function cannot inline into a caller lacking the feature.
Hoist the decision above the loop; keep the boundary only where the guarded
work amortises it (8192 elements is ~120 K ops against a ~9-cycle call — fine;
a per-block call around a handful of ops is fatal).

### Measure with counters; sign-test the clock

This box swings **±12 %**, which is larger than most wins worth having. Use
`ffai_argus::cost`-style counters (matmul FLOPs, elementwise visits, **scalar
vs vectorised** transcendentals, bytes moved, copies) — exactly reproducible,
no quiet box needed. Where only the clock can see it (locality), use a **sign
test over interleaved rounds**: 20/20 survives any symmetric noise; a min-of-3
does not.

Count scalar and vectorised transcendentals **separately** — they differ ~36x
(75 M/s libm vs 2.7 G/s in a kernel). Folding them predicted 32 s against a
16 s tower.

---

## 5. Better than accelerating: deleting

Two wins in the Argus campaign removed the work rather than speeding it up, and
both generalise:

* **Fold constants into weights at load.** `(q·kᵀ)·s ≡ (q·s)·kᵀ`, and `q·s`
  folds into `Wq`. A 12.6 M-element pass per layer became a one-off on a
  768×768 matrix at startup — **15.6 % of a layer, deleted.** `svtr.rs:213`'s
  `powf(-0.5)` scale is the same shape and the same fix.
* **Do not zero a buffer you fully overwrite.** `vec![0.0; n]` before a kernel
  that writes every element is a discarded pass — 2.6 GB per image in Argus.
  Use `spare_capacity_mut` (**not** `set_len` after `with_capacity`; clippy's
  `uninit_vec` is right that the latter is UB-adjacent).

Always ask "can this be deleted?" before "can this be vectorised?".

---

## 6. Sequencing

| step | work | gate |
|---|---|---|
| **0** | `ffai-core::fastmath`: one `exp_poly`/`tanh_poly`/`sigmoid`/`ln_poly`, scalar oracle + AVX2 twin + `*_matches_scalar` tests. Migrate diana/mercury/argus onto it | each crate's existing gates unchanged |
| **1** | Deterministic counters (port `ffai-argus::cost`) into whichever crate is next, so tier ordering is measured | counters reproducible, asserted |
| **2** | Tier 1: a shared `Gelu`/`Tanh`/`Silu` `CustomOp1`, applied at all 17 candle activation sites | per-crate oracle tests; ASR WER / OCR CER / detection mAP unchanged |
| **3** | Tier 2, highest first: `decoder.rs:858` vocab softmax, then flash_attn, then vits, then mel/fbank `log10` | Mercury's existing gates |
| **4** | §5 deletions: fold `svtr.rs:213`'s scale into the weights; audit `vec![0.0; n]` before full-overwrite kernels | byte-identical or token-identical |

**One kernel per commit**, each independently revertible, each carrying its
before/after in the message. Revert any brick that is not faster —
`codec-vectorize-kernel`'s law, and this campaign refuted four ideas that all
looked obviously right.

---

## 7. What NOT to expect

* **No large multiple in matmul.** candle's CPU GEMM measures **589–702 GF/s**
  against PyTorch's 680–697 — parity. That is not where the time is.
* **Blocked/flash attention is refuted here** — 0.23–0.51x, bit-identical but
  2–4x *slower*; candle's batched matmul beats blocking at every tile size.
* **Tile batching is refuted** — 1.07x, and it costs +384 MiB.
* **Threading beats batching for independent work** — Argus got **2.5x** from
  running tiles on separate threads, because candle's CPU backend uses rayon
  for `conv2d` and **nothing else**. Every elementwise kernel in candle is
  single-threaded; that is a general property worth exploiting wherever a
  workload has independent items.

---

# EXECUTION RECORD — 2026-08-22

The plan above was written and then executed in the same session. What follows
is what actually happened, including the parts that went differently.

## Step 0 — consolidation ✅

`ffai_core::fastmath` — `exp`, `exp2`, `ln`, `log10`, `erf`, `tanh`, `sigmoid`,
`silu`, `gelu_tanh`, `gelu_erf`, plus `round_ties_even_fast`. **10 gated tests.**
All three crates migrated onto it.

**★ The consolidation paid for itself immediately, in the direction the plan
predicted but larger.** Diana's implementation turned out to be the best of the
three, and the reason was one line: it had replaced `round()` with the
`+1.5*2^23` trick, having found — and documented — the exact trap Argus hit
independently weeks later. Mercury had the same bug in a different spelling
(`floor()`, which needs SSE4.1).

Deploying Diana's rounding to Argus, measured on the same box and binary:

| | before (`round()`) | after (magic number) |
|---|---:|---:|
| GELU scalar | 13.77 ms | **5.68 ms** |
| GELU avx2+fma | 5.98 ms | 4.76 ms |
| AVX2 advantage | **2.30x** | **1.19x** |

**The scalar path got 2.42x faster**, and Argus's hand-written AVX2 twin went
from load-bearing to marginal in the process. Recorded against the kernel: by
`codec-vectorize-kernel`'s Step 0 a new twin would not be written for 1.19x.

Two accuracy findings while gating the shared kernel, both from an oracle rather
than the implementation:

* **`tanh` via `1 - 2/(e^{2x}+1)` catastrophically cancels near zero** —
  measured **2.5e-4 relative at x = 2.4e-4**, i.e. a ~100 % error on a value the
  caller believes is exact. Fixed with a Maclaurin branch below `|x| = 0.02`.
  Neither of the two pre-existing implementations had it.
* **The erf oracle failed at exactly its own cutoff**, reporting an error of
  precisely `1 - erf(3)`. The magnitude naming the cutoff is what identified the
  fault as the oracle's, not the kernel's.

## Step 1 — deterministic counters ✅

`ffai_core::cost` — matmul FLOPs/calls, elementwise visits, **scalar vs
vectorised** transcendentals, bytes moved, layout copies. Moved out of
`ffai-argus` unchanged so Mercury and Carmenta can rank their own targets the
same way; `ffai_argus::cost` is now a re-export, so the ~30 instrumented call
sites in `siglip.rs` did not churn.

Counting scalar and vectorised transcendentals apart is the part that matters:
they differ ~36x (75 M/s libm vs 2.7 G/s inside a kernel), and folding them into
one unit once made a model predict **32 s against a ~16 s stage**.

## Step 2 — Tier 1, all 17 candle activation sites ✅

`ffai_core::fastops` — `gelu_erf`, `gelu_tanh`, `tanh`, `silu`, `erf` as
`CustomOp1`s (zero-copy, rayon, spare-capacity output). **4 gated tests**,
including one asserting the two GELUs stay *different* and one asserting a
strided view is refused rather than misread.

Applied at every production site: carmenta `parseq` x2 and `onnx_graph` x2,
mercury `wav2vec2` x4, `vits` x2, `speaker`, `text_decoder`, `decoder_kernels`.

**★ A test caught a real bug I introduced.** Unwrapping
`ff_in.forward(..)?.gelu_erf()?` into a nested call put the GELU **before**
`ff_in` instead of after — a valid-looking rearrangement that changes the
network. `wav2vec2`'s pre-norm test failed on it. *Unwrapping a method chain
into a nested call is exactly where operation order inverts, and only in the
branch that was rearranged.*

## Step 3 — Tier 2, complete ✅

* `decoder.rs` sampling — **~51 865 scalar `exp` per decode step**, the densest
  transcendental site in the workspace. Also hoisted `1/temperature` out of the
  map.
* `decoder.rs` no-speech probability, and `logsumexp` (`exp` **and** `ln`) —
  over the whole vocabulary, per beam, per step.
* `mel.rs` and `fbank.rs` — `log10` per mel bin per frame.
* `vits.rs` — the softmax loop at 406.
* `flash_attn.rs` — all four online-softmax `exp` sites.

**`flash_attn` was initially left out on the reasoning that "it already has SIMD
infrastructure, so it is a kernel to measure rather than a call to substitute".
That was wrong.** Having its own AVX2 elsewhere in the file says nothing about
four `expf` calls sitting in the softmax; the substitution is the same one-line
change as everywhere else and the surrounding structure does not change. The
original note is left in §"What is left" below, struck through, because
"deliberately skipped" and "not got to" are different claims and it was the
second one wearing the first one's clothes.

## Step 4 — deletions, partial ⚠

* `svtr.rs` attention scale: the `powf(-0.5)` is now a `const` with a
  `debug_assert` guarding it against `HEAD_DIM` moving. ✅
* **NOT done: folding that scale into the q weights.** Argus deletes the
  equivalent pass entirely (~15 % of a layer) via `(q·kᵀ)s == (q·s)·kᵀ`, but
  `svtr::linear` re-fetches weights **by name from the `VarBuilder` on every
  call**, so there is nowhere to fold them into. This needs a weight cache
  first — which is `codec-eliminate-redundancy` work, not turbocharger work, and
  would also delete a `format!` and a hashmap lookup per layer per forward.
  **Left as the next thing to do here.**
* `vec![0.0; n]` audit: `fastops` and Argus's GELU use `spare_capacity_mut`.
  `preprocess.rs:83`'s zero-init is **required** (only `n` of `width` taps per
  row are written; the rest must stay zero) — a case where the "wasted memset"
  pattern is load-bearing. The remaining three are in preprocessing, which
  measures 54 ms of a 14 561 ms caption (0.4 %); not worth the `unsafe`.

## What was left — both follow-ups now closed (2026-08-22)

Every tier in the plan was applied; two follow-ups remained. Both are done, and
**both turned up something bigger than the question that opened them.**

---

### Follow-up 1 — the `svtr` weight cache ✅ (and the fold REFUTED)

**The premise was understated.** The note above said caching weights would
"delete a `format!` and a hashmap lookup per layer per forward." It deletes far
more than that: `VarBuilder::get_unchecked` on a mmaped safetensors backend
calls `MmapedSafetensors::load` -> candle's `convert`, which **allocates a new
tensor and copies the weight out of the mapping** on every call. Twelve fetch
sites, encoder blocks in a loop.

Counted with `ffai_core::cost` (deterministic, not a stopwatch) via the new
`examples/svtr_weight_traffic.rs`:

| | weight copies | bytes |
|---|---:|---:|
| forward 0 (cold) | 234 | 32.91 MB |
| **forward 1, 2, 3** | **0** | **0.00 MB** |

**32.91 MB copied per forward, per crop.** A page of text is hundreds of crops.
The cache is a `RwLock<HashMap<String, Tensor>>` on `Svtr` — a `RwLock` and not
a `RefCell` because the recognizer is shared across rayon workers, and
poison-tolerant (`unwrap_or_else(PoisonError::into_inner)`) because the map is a
pure memo and `unwrap()` there would turn any unrelated panic into a permanently
dead recognizer.

Gate: the paddle oracle reads **8.857e-5, mean 3.141e-9, identical with the
cache and without it** — as it must, since the weights were the same on every
call. That is the whole point: it was pure overhead.

**The fold it was supposed to unlock is refuted, and the cache is not why.**
With `linear` instrumented, the q-scale multiply is 9 600 elements against
194.93 MFLOP of matmul — **1.28 %** of the modelled linear-algebra cost, and
that denominator excludes the entire convolutional backbone, so the real share
is lower. Argus's ~15 % came from a different geometry (1088 tokens x 768 dims,
against this model's 8 heads x 40 timesteps x 15 dims). And it would not be
free: `ATTN_SCALE` is not a power of two, so the fold reassociates
`(Σ x·w + b)·s` into `Σ x·(w·s) + b·s`, against an oracle with **12 % of its
error budget left**. Recorded in the code so it is not proposed a third time.

### ⚠ And the reason the oracle could be run at all was a bug it caught

Running the gate before changing anything found it **already red on the working
tree** — `max abs 6.760e-2` against a `1e-4` bar. The cause was this campaign's
own §"Step 4" entry above:

```rust
const ATTN_SCALE: f64 = 0.125;   // WRONG
```

`0.125` is `64^-0.5`. `HEAD_DIM` is **15**, whose inverse square root is
`0.2581988897…` — a **2.07x error on every attention score in the model**, i.e.
a temperature change on every softmax. Fixed to the derived value; the oracle
went `6.760e-2 -> 8.857e-5` and the mean improved three orders of magnitude
(`3.854e-6 -> 3.141e-9`).

Three things this cost, all worth keeping:

1. **The guard was a `debug_assert!`, and it could not fire.** It compiles out
   in release, and release is the only profile the oracle, the benches and
   production ever use. It is now a real `#[test]`, which `cargo test` builds in
   both profiles. **A guard that cannot fire where the code runs is not a
   guard.**
2. **100 % argmax agreement was reported the whole time.** The oracle's own
   header says agreement is not evidence, and it was right — only the tensor
   distance caught it.
3. **The "optimization" was worth nothing to begin with.** The `powf` it
   replaced was one scalar call *per attention*, not per element, in front of a
   matmul. `rusty-fast-transcendentals` §1 already says to check that the call
   is in a per-element loop before replacing it; this one was not, and the note
   is now in the code.

---

### Follow-up 2 — Diana's `silu_avx2` ✅ KEEP, and a 1.92x found next to it

**The twin keeps its `unsafe`: 2.15–2.22x over a scalar path with nothing left
in it, 20/20 on an interleaved sign test.** That is a different answer from
Argus's twin (1.19x, marginal) and the reason is the kernel shape — eight lanes
of a polynomial, no gather, no cross-lane step, which keeps scaling with width.

**But the first measurement read 4.23x, and that did not fit.** Argus had just
demonstrated that a call-free branch-free scalar loop closes most of that gap on
its own. Arithmetic that does not close indicts the decomposition
(`rusty-curiosity`), and one layer down `exp_fast` was reading `old_rounding()`
— **a relaxed atomic load, which LLVM will not hoist out of a loop** — once per
element:

| | scalar | avx2 | advantage |
|---|---:|---:|---:|
| toggle read per element | 14.30 ms | 3.40 ms | 4.21x |
| toggle hoisted per call | **7.05 ms** | 3.18 ms | **2.22x** |

**1.92x on the scalar path, bit-identical** (`max_abs` exactly 0) — the toggle
never changed the arithmetic, it only decided whether to look.

And it was not a cold fallback. `silu_scalar_pub` is called per-element from
**seven loops in `epilogue.rs` and `conv3x3.rs`** and from the AVX2 kernel's own
scalar tail, so the barrier was in the shipping path on **every** CPU, this one
included. The toggle now lives in `silu_fill`, read once per slice, with both
arms monomorphic (a `fn` pointer would be tidier and would reintroduce the
barrier in a different spelling).

Two new tests pin the claim the hoist rests on: the two roundings are
bit-identical over 60 000 points, and the legacy arm is still a correct `exp`
rather than a rotted A/B arm.

**The general law, added to `rusty-fast-transcendentals`:** *the guard around
the fix can be the barrier the fix removed.* §4 already says removing `exp` and
leaving `round` removes nothing; this is the same trap wearing an
observability hook instead of a libm call.

---

## Verification

| | |
|---|---|
| `ffai-diana` | 84 tests, 5 suites, 0 failures (incl. the parity oracle) |
| `ffai-carmenta` | 49 tests, 0 failures |
| SVTR paddle oracle | **8.857e-5 < 1e-4 — MATCH** (was `6.760e-2`, failing) |
| clippy | 0 errors on both crates |

New instruments, both deterministic and both kept:
`crates/ffai-diana/examples/silu_avx2_worth.rs`,
`crates/ffai-carmenta/examples/svtr_weight_traffic.rs`.
