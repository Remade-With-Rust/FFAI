# Argus — fifteen veins for deterministic instruction-count wins

Campaign plan, 2026-11-07. Scope: `crates/ffai-argus` (and, by construction,
`crates/ffai-argus-wasm`, which is the same code run single-threaded — every
instruction removed here lands 1:1 in the browser, where there are no cores to
hide behind).

The instrument is **callgrind Ir** (`instruction-counting`), exactly as the
Diana and Mercury campaigns ran it: same binary twice reads a spread of 0,
every arm goes through one script, work-parity anchors printed beside every
count. The verdict is a **count, not a clock** — this box has read identical
Argus code at 8172 and 14382 ms in one session (README), so no stopwatch can
resolve anything under ~2×. Where a vein's prize is *bytes* rather than
instructions (zero-filled allocations), the count is a counting allocator's,
and the Ir number is quoted beside it with its blind spot named.

Standard for a counted win, carried over from Diana: one change, counted
alone, > 100 k Ir on the instrument it targets (same-binary spread ≤ 10 k),
gated as stated per vein. Three per vein.

## What is already done (do not re-mine)

The crate has had two rounds of wall-clock work. Landed and shipping: fused
q/k/v with the attention scale folded into the weight; bias folded into the
packed-qkv copy, the residual adds and the GELU; AVX2/NEON/simd128 GELU and
`exp-sub-sum` kernels in `ffai_core::fastmath`; deferred softmax
normalisation (`AttnMergeOp`); in-place softmax; patch conv as a matmul;
fused pixel shuffle; 2-D (not broadcast) matmul for the connector and every
text `linear`; causal mask fused into softmax; `1/sqrt(64)` folded into q
(exact); GQA without `repeat_kv`; preallocated KV cache; in-place residual and
SwiGLU; slice-then-normalise before `lm_head`; six-way tile threading.

**Refuted on the clock, with numbers — do not re-propose as wall wins:**
per-head attention (0.888×), query-row-blocked vision attention (0.73–0.93×,
monotone), rank-3 attention (0.74×), hand-written blocked `q.kᵀ` (0.11×),
candle's softmax replaced (four attempts, ≤0.71×), fused q/k/v in the **text**
tower via `narrow`+`reshape` (0.94–3.52× vs 1.46–3.62×), tile batching
(1.07×), `kernels_parallel=1` under six workers (0.932×), fused LayerNorm
(parity on the clock — but see vein 5), serial-threshold and eight-accumulator
causal softmax **at decode shape** (3.80 → 4.04 ms — but see vein 1).

Two of those refutations were wall-clock verdicts on ideas whose *instruction*
count is strictly lower; the clock could not see it because the data was
cache-resident. Veins 1 and 5 re-try them on the instrument that can.

## The harness (build first — nothing below is decidable without it)

`tools/argus_ir.sh <label> [tile|prefill|caption|all]`, a copy of
`tools/diana_ir.sh` (WSL, `~/ffai-target`, `~/ffai-ir`, `taskset -c 2`,
`CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`), plus:

```
RAYON_NUM_THREADS=1 FFAI_ARGUS_TILE_WORKERS=1 FFAI_ARGUS_KERNELS_PARALLEL=0
```

`taskset` to one core also makes candle's gemm pick `Parallelism::None`
(it reads `available_parallelism`). callgrind's header thread count must read
1 (rayon's single worker is a second thread; if the count ever reads > 1,
collect with `--separate-threads=yes` and sum, or route the region through a
serial scope as Diana does). Weights resolve through the same manifest the
oracle tests use; `HF_HOME`/`FFAI_CACHE` point at the Windows caches under
`/mnt/c/Users/talmo/...`.

Three instruments, three corpus shapes (`instruction-counting` §4 — a third
of Diana's probes moved two instruments in opposite directions):

| instrument | region | what it covers | expected size |
|---|---|---|---|
| `tile` | `ir_region_tile` | one 512×512 tile: `SmolVlmVision::forward` (tower + connector), serial | ~20 G Ir (212 GFLOP of gemm + the row kernels) |
| `prefill` | `ir_region_prefill` | `TextTower::forward` on a **1142-token** embedding (the 17-tile prompt shape; build it from `.oracle/smolvlm-prompt/inputs_embeds.f32`) + 8 decode steps at pos 1142 in a second region `ir_region_decode` | ~30 G + ~1.3 G Ir |
| `caption` | `ir_region_caption` | `describe_image_unsplit` end to end on one real image: preprocess, 1 tile, ~120-token prefill, decode to stop | ~28 G Ir |

Anchors printed on every run: fnv64 of the region's output bytes (tile
embeddings / logits / token ids), the token ids themselves, and
`ffai_core::cost::stop()` (matmul flops, elementwise ops, copies, copy bytes)
— the deterministic counters already instrumented at ~30 sites in `siglip.rs`
and `text.rs`. A moved anchor means the experiment changed, not the code.

Censuses after the baseline, before any brick: self-cost
(`callgrind_annotate --threshold=94`), call counts, caller→callee edges, and
the line census (`--auto=yes`). The three "census-first" items at the bottom
of this file are questions the baseline answers for free and that re-rank
veins 2, 5 and 15.

Gates, per vein: **bit-identical** (fnv unchanged) where stated;
**token-equal** (`tests/decode_oracle.rs` 32/32, `tests/engine_oracle.rs`)
where a reassociation is involved; plus `siglip_parity` / `vision_oracle` /
`preprocess_oracle` / `prompt_oracle` as named. Every float-close vein also
runs the fused-bias and parity tolerances the crate already has.

---

## The fifteen veins, highest return first

### 1. Causal softmax rows onto the vectorised kernels (text prefill)

**Where.** `text.rs:399-425` (`CausalSoftmaxInplace`, the shipping kernel)
and `text.rs:313-344` (`CausalSoftmax`, the probe arm). Per row: a scalar
branchy max, then `for x in &mut row[..lim] { let e = fastmath::exp(*x - m);
*x = e; sum += e; }`.

**Mechanism.** `sum += e` is a loop-carried non-associative reduction, so
LLVM refuses to vectorise the loop — and that drags the polynomial `exp` down
to scalar with it (~20 Ir/element). `ffai_core::fastmath::max_f32` and
`exp_sub_sum_inplace` (AVX2 / NEON / simd128, scalar oracle, tolerance-gated)
exist precisely for this shape and are already what the **vision** softmax
calls (`siglip.rs:1303-1316`). They take a slice, so `row[..lim]` is a
one-line change; the masked tail becomes `fill(0.0)`.

**Prize, deterministic.** Exps per prefill at seq 1142: `9 × 1142 × 571.5 ×
30 = 176 M`, every one scalar today. At ~20 → ~3 Ir/element that is of the
order of **−3 G Ir per prefill**, which is ~10 % of the whole instrument.
Decode rows (9 × 1142 per layer) shrink too, by a smaller factor.

**Gate.** Token-equal (the lane split reassociates the row sum). The
in-kernel `max` is exact.

**Why the earlier refutation does not apply.** The "eight-accumulator +
target_feature" arm was priced at **decode** shape (9 rows of 1142, where
rayon dispatch dominates) on a clock. This is prefill, 10 278 rows of up to
1142, and the verdict is Ir. Price decode separately and keep whichever wins
each shape — one kernel per shape, not one kernel.

### 2. Uninitialised GEMM outputs (direct `gemm`), and the zeroed-allocation census

**Where.** Every `matmul` in the tower and text paths lands in candle's
`vec![T::zero(); dst_len]` (fork `fa366d8`, `candle-core/src/cpu_backend/mod.rs:615/675/815`)
and is then fully overwritten by gemm with `beta = 0`.

**Mechanism.** Call `gemm::gemm` (0.19.0, the version candle links) directly
into `Vec::with_capacity` + `set_len`, strides passed explicitly. Diana already
did this — `ffai-diana/src/cpuop.rs:67` (`matmul2`), `:185` (`uninit_f32`),
`:276` (`matmul2_into`), **bit-identical to candle** and counted at −20 M Ir
on YOLO shapes. Lift it into `ffai-core` (next to `fastmath`) so Argus,
Diana and Mercury share one copy with one oracle test.

**Prize, deterministic.** Bytes zero-filled then overwritten, per vision tile:
qkv 9.4 MB + out 3.1 + fc1 12.6 + fc2 3.1 + scores **50.3** + attn·v 3.1 =
**81.6 MB/layer, 979 MB/tile, 16.6 GB per 17-tile caption**. Per prefill at
seq 1142: ~73 MB/layer, **2.2 GB**. Plus the KV cache's `Tensor::zeros`
(43 MB per caption, every slot written before it is read) and preprocess's
`vec![0.0; tiles*per]` (53 MB, vein 11).

**The instrument's blind spot, stated.** `vec![0; n]` is `calloc`; for
allocations above the mmap threshold glibc hands back kernel-zeroed pages and
emits **no memset instructions** — Ir sees nothing, while the page faults
(12 800 per 50 MB) are real wall. Below the threshold (and under
`rusty_alloc`, the production allocator) it is a `memset` that Ir
**over**-prices (`codec-memory-copies`, rusty_aac note). So the primary count
for this vein is a counting allocator's "bytes zero-allocated per region",
with Ir and a wall sign-test beside it. Both arms, same binary.

**Gate.** Bit-identical (fnv), proven on Diana. Keep candle's `matmul` as the
oracle behind the existing `FFAI_ARGUS_*` arm pattern.

**Follow-on it unlocks.** `matmul2_into` lets a GEMM write straight into a
destination row range — Diana's concat-free blocks. In Argus the candidates are
the text `o_proj` output written into the residual buffer (then `AddInplace`
becomes the GEMM's own accumulate with `beta = 1`, deleting a 2.6 MB pass per
layer — float-close, since the add moves inside the K-loop), and the
per-chunk outputs of vein 3.

### 3. Causal block-skip attention (text prefill)

**Where.** `text.rs:957` (`qg.matmul(&k.t())`) and `:975-978`
(`att.matmul(&v)`): both run over the full `1142 × 1142` square, after which
`CausalSoftmaxInplace` zeroes the upper triangle and `attn.v` multiplies those
zeros.

**Mechanism.** Chunk the query rows (e.g. 128): chunk `[s0, s1)` needs keys
`[0, s1 + offset)` only. Per chunk: `q_chunk` (contiguous copy of
`3 × 3 × 128 × 64` = 295 KB — the same copy the vision blocked arm made),
`q_chunk · k[..lim]ᵀ`, causal softmax with `offset = index_pos + s0`,
`p · v[..lim]`, write the chunk's rows into the output (`matmul2_into` from
vein 2, or `Tensor::cat`). Decode (`seq = 1`) is untouched.

**Prize, deterministic.** FLOPs: `q.kᵀ` and `attn.v` are `2 × 9 × 1142² × 64
= 1.5 GFLOP` each per layer, of which ~half is above the diagonal: **−1.5
GFLOP/layer, −45 GFLOP per prefill** (of ~330 — the prefill's largest
single deterministic reduction), plus ~23 MB/layer of softmax tail writes and
`rows × k_len / 2` elementwise visits deleted. `cost::matmul_flops` is the
anchor that moves, and must move by exactly that.

**Gate.** The `q.kᵀ` half is exact (entries never read are not computed).
The `p·v` half sums over a shorter K, so gemm's blocking changes the
association: **token-equal**, not bit-identical.

**Why the three vision refutations do not transfer.** Those blocked arms moved
*no* arithmetic — they bought locality and paid gemm re-packing and lower
`M`. This one deletes 45 GFLOP; the re-packing of a 0.9 MB `k` nine times is
8 MB. Size it with a throwaway first (`instruction-counting` §7.2): run the
chunked loop *without* the causal limit and confirm it reads as the unblocked
count plus the pack cost, then add the limit and read the FLOP drop.

### 4. The Lanczos resampler's two passes, vectorised bit-exactly

**Where.** `preprocess.rs:336-395` (`resize_lanczos_u8`): per output sample,
`for (kk, &w) in row.iter().enumerate().take(n) { acc += i32::from(src[(y*sw +
start + kk)*channels + c]) * w; }` — index arithmetic, a bounds check and an
`imul` per tap, scalar.

**Mechanism.** The arithmetic is integer (PIL's 22-bit fixed point), so a
vector twin is **bit-identical by construction** — the gate is `assert_eq!`
on bytes, not a tolerance. The vertical pass is the easy half: for fixed
output row `y`, `acc[x] += mid[(start+kk)*W + x] * w_kk` is a contiguous
AXPY over `x` — 8-wide `vpmulld`/`vpaddd` under `avx2`, SSE4.1 at 4, and
auto-vectorisable at SSE2 if the inner loop is written over `x` with the tap
loop outside. The horizontal pass wants the samples de-interleaved per
channel (three planes) so the tap window is contiguous; or compute 8 output
`x` at once with the taps as a small transposed table.

**Prize, deterministic.** For the shipping 512→2048→512 path: upscale ~110 M
MACs, thumbnail ~98 M MACs, **~210 M scalar MACs per image at ~8 Ir each —
of the order of 1.7 G Ir**, i.e. ~6 % of the `caption` instrument and, in
the browser (serial, `describeFast`), the same share of wall. On the 24-core
box it is 0.5 % of a 17-tile caption, which is why it never ranked before.

**Gate.** `tests/resize_oracle.rs` + `tests/preprocess_oracle.rs` (PIL-exact
bytes); the `the_fixed_point_kernel_preserves_a_constant_image` test; fnv on
`pixel_values`.

### 5. LayerNorm: eight-lane reductions, re-judged on Ir

**Where.** `siglip.rs:884-992` (`LayerNormOp`, **off by default**,
`FFAI_ARGUS_FUSED_LN`), called 25× per tile.

**What the census must settle first.** The doc comment says candle's
`LayerNorm` is six single-threaded broadcast passes. It is not, on this
fork: `candle-nn/src/layer_norm.rs:118-121` routes a contiguous input with a
bias to the fused `ops::layer_norm`, one pass per row. That is why the fused
arm read as parity on the clock — it was fused against fused. **But candle's
fused kernel accumulates `sum` and `sum2` as scalar loop-carried `f32`
reductions** (`candle-nn/src/ops.rs:704-711`), so its two reductions are
scalar; ours has eight independent lanes and vectorises. Same formulation
(`E[x²] − E[x]²`, `eps` inside the sqrt), so the "it changes the variance
formulation" objection in the doc is also stale — fix the doc with the brick.

**Prize, deterministic.** `768 × 1024 × 25 = 19.7 M` elements per tile
through two scalar reductions → lanes: of the order of **−60 to −80 M Ir per
tile**; the same shape is the text tower's `rms_norm` (vein 8).

**Quality.** Eight lanes shorten the dependency chain eightfold; the lane
sum is *more* accurate than the serial fold, not less. Perf and quality
together, which is the brief.

**Gate.** `tests/siglip_parity.rs` tolerance, token-equal caption. Flip the
default; keep candle's as the `=0` arm.

### 6. Eight-wide twins of every row kernel (bit-identical, body compiled twice)

**Where.** The crate is built for the x86-64 **baseline** (SSE2), so every
hand-written row loop auto-vectorises 4-wide at best: `PackedQkvOp`
(`siglip.rs:1647-1659`), `AddBiasOp` (`:1753-1757`, ×2 per layer),
`EmbedAddOp`, `AttnMergeOp` (`:1380-1388`), the bias pre-add in `GeluBiasOp`
(`:1837-1839`), `AddInplace` (`text.rs:474-478`), `RmsNorm` (after vein 8),
`CausalSoftmax`'s tail, `KvAppend` (already `memcpy` — leave it).

**Mechanism.** Diana's trick, already counted there: write the body once as
`#[inline(always)] fn body(..)`, wrap it in a plain function and in a
`#[target_feature(enable = "avx2")]` twin (**no `fma`** → no contraction →
bit-identical), dispatch once per call on the cached probe. No intrinsics.

**Prize, deterministic.** Elements through these kernels per vision layer:
packed 2.36 M + bias adds 1.57 M + gelu pre-add 3.15 M + merge 0.79 M ≈
7.9 M → **~95 M per tile**; halving the vector-op count is of the order of
−1 Ir/element → ~100 M Ir per tile. Each kernel is its own counted brick.

**Gate.** fnv unchanged (bit-identical). `cargo test --no-default-features`
and the wasm32 check, because every one of these needs its serial/simd128
arm to still compile — the ISA asymmetry that bit the browser once.

### 7. Activation fusions in `fastmath`: `silu·up` twin, `gelu(x + b)` in one pass

**Where.** `text.rs:699-703` (`SwiGluInplace`: `*g = fastmath::silu(*g) * u`,
call-free so SSE2 auto-vectorises it 4-wide, but no AVX2/NEON/simd128 twin —
unlike `gelu_tanh_inplace` and `exp_sub_sum_inplace`); `siglip.rs:1836-1846`
(`GeluBiasOp`: a `v + b` pass over the row, **then** `gelu_tanh_inplace(o)`
re-reads and re-writes it — two passes where the L1-resident second pass
costs a load and a store per element in Ir).

**Mechanism.** Add `silu_mul_inplace(gate, up)` and `gelu_tanh_bias(dst, src,
bias)` to `ffai_core::fastmath` with the same four arms the existing kernels
have (scalar oracle, AVX2+FMA, NEON, simd128), tolerance-gated like their
siblings. Without `fma` the AVX2 arm is bit-identical to today; with it,
fewer instructions and float-close.

**Prize, deterministic.** SwiGLU: `1142 × 1536 × 30 = 52.6 M` elements per
prefill, 8-wide vs 4-wide. GELU: `3.15 M × 12 = 37.7 M` elements per tile,
one load+store pair deleted each.

**Gate.** Token-equal (FMA arm) or fnv (no-FMA arm); `fastmath`'s
twin-vs-oracle tests.

### 8. `RmsNorm`'s serial accumulator → lanes

**Where.** `text.rs:239-243`: `let mut acc = 0f32; for &v in i { acc += v*v }`
— loop-carried, scalar, 576 wide, `1142 × 60` rows per prefill and `60` per
decode token. (The comment says the reduction is "accumulated in f64 exactly
as candle's"; it is `f32`, and so is candle's — `ops.rs:480-487`. Fix the
comment.)

**Mechanism.** Eight partial sums, as `LayerNormOp` and `softmax_row_scalar`
already do; then vein 6's twin.

**Prize, deterministic.** `576 × 1142 × 60 = 39.5 M` scalar multiply-adds per
prefill → lanes; of the order of −80 M Ir per prefill, and the decode steps
proportionally. More accurate (shorter chain).

**Gate.** `rms_norm_matches_candles` tolerance, token-equal.

### 9. Text-tower layout passes: transpose + RoPE fused, k and v straight into the cache

**Where.** `text.rs:860-871` (three `reshape → transpose → contiguous`
copies), `:876-877` (candle `rope` allocates and writes q and k again),
`:920-921` (`KvAppend` copies k and v a third time into the cache),
`:987` (`y.transpose(1,2).reshape(..)` — another strided copy through
candle's generic `copy_strided_src_`).

**Mechanism.** One `CustomOp` per tensor: q — read the `(seq, 9·64)`
projection, write `(9, seq, 64)` with RoPE applied in the same pass (replicate
candle's non-interleaved rope order exactly: `x1·cos − x2·sin`, `x1·sin +
x2·cos`, no `mul_add`, so it stays bit-identical); k — the same pass, written
**directly into the cache slot** at `pos`; v — a plain permute-copy into the
cache. The attention-output transpose becomes a hand permute of 64-float
runs (the `PixelShuffleOp` shape), and with vein 2's `matmul2_into` the
`o_proj` can consume it strided and skip the copy entirely.

**Prize, deterministic.** Per layer per prefill today: q 2.6 + 2.6, k 0.88 ×
3, v 0.88 × 2, out 2.6 MB ≈ **12.4 MB copied in eight passes, five
allocations**; after: q 2.6, k 0.88, v 0.88 (+ out 2.6 or 0) in three to four
passes, one or two allocations. Same per decode token ×30 layers ×33 steps,
where the allocation count is a larger share.

**Gate.** Bit-identical if the rope order is replicated; `cost::copies` and
`copy_bytes` are the anchors that must drop by exactly the deleted passes.

### 10. Patch-embedding im2col as a hand permute, and zero-copy tile tensors

**Where.** `siglip.rs:166-170`: a 6-D `reshape → permute → contiguous` that
candle executes through its generic strided walker (`copy_strided_src_`,
`MultipleBlocks` — per-element index iteration) for `786 K` elements per tile.
`engine.rs:461-462` and `:518-519`: each tile's 3 MB of `pixel_values` is
`to_vec()`'d and re-wrapped with `Tensor::from_vec`.

**Mechanism.** The permutation is `(c, gh, p, gw, p) → (gh, gw, c, p, p)`:
for fixed `(gh, gw, c, py)` it is a contiguous run of 16 floats — one
`copy_from_slice` per run, both streams sequential at run granularity (the
same lesson as `PackedQkvOp`'s 20.4 → 4.0 ms). Wrap the whole `pixel_values`
once with `from_vec` and `narrow(0, t, 1)` per tile.

**Prize, deterministic.** Of the order of 10–20 M Ir per tile from the
walker (candle's per-element strided index is ~15–25 Ir) and 786 K elements
of copy per tile deleted. Diana's biggest single win (im2col −114 M) was this
shape.

**Gate.** Bit-identical (pure permutation); the existing
`a_non_overlapping_conv_is_a_matmul_over_permuted_patches` test extended to
the hand kernel.

### 11. Preprocess plumbing: the zero-fill, the intermediate `Vec`, the row copies, the no-op resizes

**Where.** `preprocess.rs:530` (`vec![0.0f32; tiles * per]` — 53 MB zeroed
then fully overwritten); `:532-541` (`write_tile`: `normalize_u8` allocates a
3 MB `f32` `Vec` per tile, then a scalar HWC→CHW gather `norm[i*3 + c]` over
it); `:543-552` (`tile_buf`: 512 row copies into a scratch tile, then
`write_tile` reads the scratch — the scratch is a pure extra pass);
`:130-132`/`:344-346`/`:348-350` (`src.to_vec()` on an identity resize — for a
square image the step-2 resize is **always** an identity and copies 12 MB;
the thumbnail's horizontal pass when `sw == dw` too); `engine.rs:775`
(`to_rgb8` copies an RGB8 input it could borrow — `Cow`).

**Mechanism.** One pass from the `u8` source region straight into the planar
`f32` destination (`v / 127.5 − 1`, unchanged), reading rows of `b` at their
stride; `Vec::with_capacity` + `spare_capacity_mut` for `pixel_values`;
`Cow<[u8]>` for identity resizes and RGB8 input.

**Prize, deterministic.** Per image: 53 MB memset, 17 × 3 MB `Vec`, 16 × 0.8 MB
row-copy, 12 MB + 12 MB identity copies, one input copy — ~110 MB of pure data
movement with no arithmetic, all bit-identical to delete.

**Gate.** `preprocess_oracle` bytes; `preprocessing_fills_every_tile_it_promises`.

### 12. Fused `gate|up` (and `q|k|v`) GEMMs in the text tower — this time with our own consumer

**Where.** `text.rs:1006-1007`: two `linear` calls over the same `normed`
lhs; gemm packs the 2.6 MB lhs twice per layer. `:852-854` likewise, three
times.

**Why the earlier refutation does not apply.** Fusing q/k/v lost because the
three parts had to be `narrow`ed out of the fused result and candle's
`reshape` then **copied** the strided views. Here the consumer of `gate|up`
is `SwiGluInplace` — our kernel — which can read `gate` at columns
`[0, 1536)` and `up` at `[1536, 3072)` of one `(seq, 3072)` output with a row
stride of 3072 and never copy. With vein 9's layout pass as the consumer of
`q|k|v`, the same holds there.

**Prize, deterministic.** `matmul_calls` −30 (gate|up) and −60 (q|k|v) per
prefill and per decode token; lhs packing −2.6 MB and −5.2 MB per layer per
prefill. FLOPs identical.

**Gate.** Bit-identical in practice (each output element's K-loop is
unchanged by N-partitioning — verify with fnv, as Diana did for `matmul2`).

### 13. Prompt assembly: ids from the layout, one buffer for the merge

**Where.** `engine.rs:673-678`: the tokenizer runs over an 8.5 KB string that
is 1088 copies of `<image>` plus markers whose ids are fixed by the layout;
`prompt.rs` `merge_image_embeddings`: 1200 `Tensor::i()` row views and a
`Tensor::stack` of 1200 tensors (one candle dispatch + layout allocation
each) to interleave two already-contiguous buffers.

**Mechanism.** Tokenize only the user text and the template fragments
(~100 B); splice the known special-token ids from `PromptLayout`. Merge as
one `Vec<f32>` filled by `copy_from_slice` per row from the two sources, one
allocation.

**Prize, deterministic.** Tokenizer input 8.5 KB → ~100 B; 1200 small tensor
allocations + 1200 strided copies → 1 allocation + 1142 memcpys. Small in
absolute terms (README: assemble = 4 ms); counted because it is a per-caption
fixed cost that is pure overhead on the unsplit/video path, where the caption
is short.

**Gate.** `tests/prompt_oracle.rs` — the ids must be **equal**, and
`inputs_embeds` bit-identical to `.oracle/smolvlm-prompt/ours_inputs_embeds.f32`.

### 14. Dispatch hoisting: per-row feature probes, per-row rayon, per-layer switches

**Where.** `ffai_core::fastmath::exp_sub_sum_inplace` and `max_f32` re-run
`is_x86_feature_detected!` on **every call**, and the vision softmax calls
them once per row — 12 288 rows per layer (`siglip.rs:1297-1316`), and after
vein 1 so will 10 278 prefill rows per layer. `SoftmaxExpInplace` is
`par_chunks_mut` unconditionally (ignores `kernels_parallel`, so under six
tile workers it nests a rayon region 12 times per tile — the exact cost
`AddInplace` measured and avoided, `text.rs:466-473`). Six arm switches
(`fuse_bias`, `late_normalize`, `head_attn`, `attn_block`,
`inplace_softmax`, `kernels_parallel`) are each an atomic load per layer, and
~26 `Instant::now()` + `prof::add` pairs per layer run whether or not
profiling is on.

**Mechanism.** The skill's "runtime switch" shape (`codec-eliminate-redundancy`
§ four redundancies, #1): a `Plan` struct read **once** per forward and
passed by value; row kernels that take a block of rows inside one
`#[target_feature]` function so the probe is per call, not per row; the
softmax honouring `kernels_parallel`.

**Prize, deterministic.** Small by Ir — of the order of 10 Ir per row probe
(~250 K per layer, ~3 M per tile) and the rayon split/join overhead per row
block — but it is the one vein whose prize is also a *wall* fix on the
shipping six-worker path (the nested region). Counted honestly as the
smallest vein; three bricks exist (probe hoist, nested rayon, Plan).

**Gate.** Bit-identical.

### 15. The decode step's fixed overhead, by allocation census

**Where.** `decode.rs:448-482` (the per-token loop), `:500-511` (`argmax`
copies 49 280 logits out through `to_vec1` and scans them with a branchy
scalar loop), `text.rs:1025-1030` (`rope` at seq 1 — an allocation per call
for 576 floats), `rms_norm` (`xs.contiguous()?` + `w.contiguous()?` clones).

**Mechanism.** Install a counting `GlobalAlloc` in the `prefill` instrument's
decode region and print allocations per token beside Ir (the rusty_sloth
method: 30 994 → 2 allocations per 32 tokens came from exactly this census).
Then: `argmax` on the storage via a `CustomOp1` (or `max_f32` + position —
`>` keeps the first maximum, which is the tie rule the comment pins); embed
the single id by a direct row copy; fold `rope` into vein 9's pass; drop the
`contiguous()` clones where the input is already contiguous.

**Prize, deterministic.** Allocations per token (count first; expect several
hundred of ours plus candle's per-op `Shape`/`Layout` vectors) and ~200 KB of
logits copy per token. Low by Ir — decode is `≈ 270 MFLOP` of GEMV per token
and these are < 1 % of it — ranked last for that reason, and kept because the
video path runs 8× fewer image tokens and proportionally more decode.

**Gate.** Token-equal; `tests/decode_oracle.rs`.

---

## Census-first items (the baseline answers these for free)

1. **Is the zero-fill a memset or a page fault?** Line census on
   `__rust_alloc_zeroed` / `memset` / `calloc` in the `tile` and `prefill`
   regions, and the same regions with the counting allocator reporting bytes
   zero-allocated. Decides how vein 2 is *quoted*, not whether it is built.
2. **Does candle's gemm pack the weight at `m = 1`?** Edge census from
   `TextTower::block` into `gemm::*` at decode shape. If the `(1, 576) ·
   (576, N)` calls go through the packed GEMM rather than a gemv path, the
   decode step copies ~540 MB of weights per token into pack buffers, and
   a direct-gemv brick joins vein 2 at the top of the list; if they take
   gemm's gemv dispatch, vein 15 stays last.
3. **Where does `LayerNorm` actually land?** Confirm `ops::layer_norm` (not the
   broadcast chain) is the callee from `Layer::forward`, and read its line
   census for the two scalar reductions before pricing vein 5.

## Ledger

Harness: `tools/argus_ir.sh build|count <label> [modes]` (WSL, callgrind,
one core, one-thread rayon pool) and `tools/argus_ir_queue.sh` for parallel
counts; instrument `crates/ffai-argus/examples/ir_argus.rs`. Every row is ONE
change counted alone against the binary immediately before it in the chain
(label in brackets). Same-binary null: `pre` 7,320,860,790 twice, spread 0;
an extra env var moves a count by a fixed ~300 Ir (environ pedestal).

Baseline (label `base`): tile 67,900,223,220 · prefill 120,527,101,420 ·
decode 929,267,187 · caption 81,982,867,848 · pre 7,320,860,790 · assemble
26,520,813. Anchors: tile fnv 5f03ea273c7b18b0, prefill fnv b9bd2d34eb922b6a
argmax 378, decode tokens [378, 2443, 10637, 354, 7545, 28, 4785, 29], pre fnv
63698d7ff5f507ba.

Census facts that re-ranked the plan: the tile is 94 % gemm microkernel; the
1.06 GB/tile zero-fill is only 63 M Ir of memset (fresh mmap pages are zeroed
by the kernel — invisible to Ir); decode is gemv 234 M + gemm pack_operands
173 M of 805 M — the attention products repack the whole K/V cache each step.

| vein | brick | instrument | before | after | Δ Ir | gate |
|---|---|---|---:|---:|---:|---|
| 1 | 1a causal row max -> `max_f32` | prefill | 120,527,101,420 [base] | 120,262,493,740 | **−264.6 M** | fnv unchanged |
| 1 | 1b exp+sum -> `exp_sub_sum_inplace` | prefill | 120,262,493,740 [v1a] | 107,554,835,080 | **−12,707.7 M** | tokens equal |
| 1 | 1c deferred divide (A/B arm, same binary) | prefill | 94,372,767,990 [early arm] | 93,981,237,620 | **−391.5 M** | tokens equal |
| 2 | 2a text projections, uninit gemm out | prefill | 108,078,260,505 [v1c] | 107,905,118,421 | **−173.1 M** | fnv unchanged |
| 2 | 2b vision projections, uninit gemm out | tile | 67,900,223,220 [v2a] | 67,866,829,539 | **−33.4 M** | fnv unchanged |
| 2 | 2c attention products, uninit gemm out | tile | 67,866,829,539 [v2b] | 67,846,494,691 | **−20.3 M** | fnv unchanged |
| 3 | 3a causal block-skip prefill (chunk 128) | prefill | 107,896,771,880 [v2c] | 94,695,313,470 | **−13,201.5 M** | argmax equal |
| 3 | 3b chunks merge into one buffer | prefill | 94,695,313,470 [v3a] | 94,528,756,959 | **−166.6 M** | fnv unchanged |
| 3 | 3c chunk 64 | prefill | 94,528,756,959 [v3b] | 93,966,439,120 | **−562.3 M** | argmax equal |
| 4 | 4a Lanczos vertical as row AXPY | pre | 7,320,860,790 [v3c] | 4,741,984,790 | **−2,578.9 M** | fnv unchanged |
| 4 | 4b Lanczos horizontal, 3-ch slices | pre | 4,741,984,790 [v4a] | 2,156,992,986 | **−2,585.0 M** | fnv unchanged |
| 4 | 4c AVX2 twins of both row kernels | pre | 2,156,992,986 [v4b] | 1,156,606,141 | **−1,000.4 M** | fnv unchanged |
| 5 | 5b residual+bias+ln2 fused | tile | 67,970,990,037 [v5a] | 67,969,895,221 | **−1.10 M** | fnv unchanged |
| 5 | 5c fc2 residual + next LN fused | tile | 67,969,895,221 [v5b] | 67,968,627,514 | **−1.27 M** | fnv unchanged |
| 5 | 5e layer-0 ln1 row AVX2 twin | tile | 67,368,685,732 [v5rev] | 67,355,545,790 | **−13.14 M** | fnv unchanged |
| 6 | 6a packed-qkv copy+bias AVX2 twin | tile | 67,968,627,514 [v5c] | 67,919,991,508 | **−48.6 M** | fnv unchanged |
| 6 | 6b residual+LN row AVX2 twin | tile | 67,919,991,508 [v6a] | 67,559,757,172 | **−360.2 M** | fnv unchanged |
| 6 | 6c attention-merge AVX2 twin | tile | 67,559,757,172 [v6b] | 67,542,013,576 | **−17.7 M** | fnv unchanged |
| 6 | 6c (text merge-into, same change) | prefill | 93,981,237,620 [v6b] | 93,948,084,147 | −33.2 M | fnv unchanged |
| 7 | 7a `fastmath::silu_mul_inplace` (SwiGLU) | prefill | 93,948,084,147 [v6c] | 92,580,043,947 | **−1,368.0 M** | argmax equal |
| 7 | 7b `fastmath::gelu_tanh_bias_into` | tile | 67,669,593,870 [v5d] | 67,574,657,130 | **−94.9 M** | fnv unchanged |
| 7 | 7c exp without the dead upper clamp | tile | 67,574,657,130 [v7b] | 67,496,210,538 | **−78.4 M** | fnv unchanged |
| 7 | 7c (text, same change) | prefill | 92,580,043,947 [v7a] | 92,485,691,907 | −94.4 M | fnv unchanged |
| 8 | 8a RMSNorm eight-lane sums | prefill | 92,485,691,907 [v7c] | 92,438,411,809 | **−47.3 M** | argmax equal |
| 8 | 8b RMSNorm row AVX2 twin | prefill | 92,438,411,809 [v8a] | 92,307,264,682 | **−131.1 M** | fnv unchanged |
| 8 | 8d residual add inside the AddRms twin | prefill | 91,506,977,031 [v14c] | 91,471,689,231 | **−35.3 M** | fnv unchanged |
| 9 | 9a q transpose + RoPE in one pass | prefill | 92,317,498,132 [v8c] | 91,727,927,317 | **−589.6 M** | fnv unchanged |
| 9 | 9b k transposed+rotated into the cache | prefill | 91,727,927,317 [v9a] | 91,499,548,399 | **−228.4 M** | fnv unchanged |
| 9 | 9c v transposed into the cache | prefill | 91,499,548,399 [v9b] | 91,444,928,714 | **−54.6 M** | fnv unchanged |
| 10 | 10a im2col as a hand permute | tile | 67,355,545,790 [v5e] | 67,355,392,320 | **−0.15 M** | fnv unchanged |
| 10 | 10b tile tensors from one move of pixel_values | caption | 74,870,928,277 [v10a] | 74,870,464,800 | **−0.46 M** | text equal |
| 10 | 10c embed-add row AVX2 twin | tile | 67,355,392,320 [v10a/b] | 67,353,469,289 | **−1.92 M** | fnv unchanged |
| 11 | 11a identity resizes borrow | pre | 1,156,606,141 [v10c] | 1,154,167,670 | **−2.44 M** | fnv unchanged |
| 11 | 11b normalise fused into CHW scatter | pre | 1,154,167,670 [v11a] | 1,093,950,186 | **−60.2 M** | fnv unchanged |
| 11 | 11c tiles read in place (no tile_buf) | pre | 1,093,950,186 [v11b] | 1,092,306,745 | **−1.64 M** | fnv unchanged |
| 12 | 12c down projection accumulates the residual in gemm | prefill | 91,576,644,437 [v12b] | 91,521,369,418 | **−55.3 M** | argmax equal |
| 12 | 12b′ fused q\|k\|v GEMM at prefill only | prefill | 91,515,823,499 [v12a′] | 91,514,127,313 | **−1.70 M** | fnv unchanged |
| 13 | 13a memoised image-block token ids | assemble | 26,520,437 [v12c] | 22,010,924 | **−4.51 M** | fnv unchanged |
| 13 | 13b merge as one buffer | assemble | 22,010,924 [v13a] | 11,552,274 | **−10.46 M** | fnv unchanged |
| 13 | 13c embed only the text ids | assemble | 11,552,274 [v13b] | 10,957,394 | **−0.59 M** | fnv unchanged |
| 14 | 14a vision softmax row-kernel table | tile | 67,353,469,289 [v15c] | 67,349,046,329 | **−4.42 M** | fnv unchanged |
| 14 | 14b text causal-softmax row-kernel table | prefill | 91,521,384,538 [v14a] | 91,516,486,738 | **−4.90 M** | fnv unchanged |
| 14 | 14c row ops serial below 16 K elements | decode | 1,282,776,140 [v14b] | 1,282,593,052 | **−0.18 M** | tokens equal |
| 15 | 15c argmax in place, vectorised max | decode | 1,282,013,693 [v15b] | 1,277,181,200 | **−4.83 M** | tokens equal |
| 15 | 15g text linears without reshape round-trips | decode | 849,915,020 [v15f] | 849,345,375 | **−0.57 M** | tokens equal |
| 12 | 12a″ gate\|up fused at decode only (A/B arm, same binary) | decode | 850,612,276 [arm never] | 849,521,516 | **−1.09 M** | tokens equal |
| 15 | 15h decode attention ops serial below 16 K elements | decode | 787,011,381 [v12a3, gemm arm] | 786,868,363 | **−0.14 M** | tokens equal |

### Per-vein totals (the three credited wins, each counted alone)

| vein | wins | instrument(s) | Σ credited |
|---|---|---|---:|
| 1 causal softmax on vector kernels | 1a, 1b, 1c | prefill | −13,363.8 M |
| 2 uninitialised gemm outputs | 2a, 2b, 2c | prefill, tile | −226.8 M |
| 3 causal block-skip prefill | 3a, 3b, 3c | prefill | −13,930.4 M |
| 4 Lanczos resampler | 4a, 4b, 4c | pre | −6,164.3 M |
| 5 LayerNorm | 5b, 5c, 5e | tile | −15.5 M |
| 6 AVX2 twins of row kernels | 6a, 6b, 6c | tile | −426.5 M |
| 7 activation kernels in `fastmath` | 7a, 7b, 7c | prefill, tile | −1,541.3 M |
| 8 RMSNorm | 8a, 8b, 8d | prefill | −213.7 M |
| 9 text layout passes | 9a, 9b, 9c | prefill | −872.6 M |
| 10 patch embedding / tile tensors | 10a, 10b, 10c | tile, caption | −2.5 M |
| 11 preprocess plumbing | 11a, 11b, 11c | pre | −64.3 M |
| 12 projection structure | 12c, 12b′, 12a″ | prefill, decode | −58.1 M |
| 13 prompt assembly | 13a, 13b, 13c | assemble | −15.6 M |
| 14 dispatch overhead | 14a, 14b, 14c | tile, prefill, decode | −9.5 M |
| 15 decode-step overhead | 15c, 15g, 15h | decode | −5.5 M |

### Whole pipeline, baseline -> final head (label `final`, after cleanup)

The cleanup after the last brick (dead code removed, clippy fixes, and the
chunked-attention output buffer ZEROED instead of left uninitialised — sound,
costs one memset per prefill layer) was rebuilt and recounted:

| instrument | baseline | final | Δ |
|---|---:|---:|---:|
| caption (unsplit, end to end) | 81,982,867,848 | 74,854,008,769 | **−8.7 %** |
| prefill (1142 tokens) | 120,527,101,420 | 91,348,653,870 | **−24.2 %** |
| decode (8 steps at 1142) | 929,267,187 | 786,865,933 | **−15.3 %** |
| tile (one 512x512 tile) | 67,900,223,220 | 67,349,036,293 | −0.8 % (94 % gemm) |
| pre (split, 17 tiles) | 7,320,860,790 | 1,092,306,745 | **−85.1 %** |
| assemble (17-tile prompt) | 26,520,813 | 10,957,437 | **−58.7 %** |

Gates at `final`: `cargo test --release -p ffai-core -p ffai-argus` — 98 unit
tests and every oracle pass with NO skips (decode_oracle 32/32 tokens on the
decoder-only, whole-pipeline and full-content paths; engine, vision, prompt,
preprocess, resize, video, siglip-parity). CI's blocking clippy
(`--no-deps -p ffai-core -p ffai-argus --lib -- -D warnings -A unsafe_code`)
is clean; `cargo check --target wasm32-unknown-unknown` and
`--target aarch64-unknown-linux-gnu` compile (wasm has one pre-existing warning
in `SoftmaxLastDimOp`).

Anchors at the head: caption text "A rainbow." (= baseline); decode tokens
identical to baseline; prefill argmax 378 (= baseline); pre and assemble fnv
identical to baseline (byte-identical); tile fnv 3fdc5685a4a10dfd (float-close
since 5a; the LayerNorm reductions are eight-lane).

Decode, same bricks (tokens identical throughout): base 929,267,187 · 1a
919,641,417 · 1b 826,087,767 · 1c 824,230,512 · 2a 816,726,299 · 2c
804,743,309; deferred-divide A/B −2.48 M.

**Refuted, with numbers (do not retry as written):**

- 1c as first written: +523.4 M prefill — `row[lim..].fill(0.0)` lowered to a
  `memset` call per row (245 M -> 1,398 M Ir) plus 1,154 M in `slice::fill`;
  the explicit store loop it replaced was inline vector stores. Fixed, then the
  divide was measured alone (row above).
- 5a `LayerNormOp` (eight-lane sums) on at the SSE2 baseline: +124.5 M tile
  (21 Ir/element against candle's fused 15). Not the idea's fault: under the
  AVX2 twin (6b) the same lanes are worth −360 M.
- 5d candle's exact LayerNorm arithmetic in the shared row: +127.6 M tile —
  bit-identical to candle (fnv back to 5f03ea273c7b18b0) but its scalar sums
  gain nothing from AVX2. Reverted (the revert, −127.5 M, is not credited).
- 8c `AddRmsOp` (residual + ln2 in one pass) with only the norm half behind the
  twin: +10.2 M prefill, +1.23 M decode. 8d moved the add into the twin.
- 15a hand-written streaming decode attention: +955.6 M decode. gemm's FMA
  microkernels beat a lane-array loop at the SSE2 baseline even while packing;
  the AVX2 twin (15b) recovered 468 M of it but not the rest. Replaced by 15d
  (per-head m = 1 products: gemm's gemv path, no packing).
- 14b text row-kernel table: +5.60 M decode. Calling `max_f32` /
  `exp_sub_rowmax_sum` through function pointers blocked inlining of their
  dispatch; the inlined feature probe was already cheap. (The same table won
  −4.42 M on the vision softmax, 14a, where rows are long.)
- 12a fused gate|up with SwiGLU in place and `down` reading the strided gate
  half: +131.8 M prefill (gemm packs a row-strided lhs through its generic
  path). 12a′ with a dense product: still +175.4 M prefill against separate
  GEMMs (91,340,449,205 -> 91,515,823,499) — but −1.23 M on decode. Split by
  call site as 12a″ (fused at decode only; env arm `FFAI_ARGUS_GATEUP`).
- 12b fused q|k|v for every call: −0.12 M prefill, +2.02 M decode. Split as
  12b′ (prefill fused, decode three m = 1 products over row views).
- Decode attention without gemm's per-step cache repacking, three ways, all
  against the batched path in one binary (`FFAI_ARGUS_DECODE_ATTN`): 15a
  hand-written stream +955.6 M (15b's AVX2 twin recovered 468 M of it), 15d
  per-head m = 1 products as tensors +93.2 M (882,241,263 vs 789,079,570), 15f
  direct m = 1 `gemm::gemm` calls in one op +62.5 M (849,915,020 vs
  787,451,923). The packing is real (173 M) but gemm's gemv path computes
  these strided products worse than the packed microkernel. Batched stays.
- 14d profiling clock reads gated: −84 K decode, −19 K prefill, −10 K tile —
  real but under the 100 K bar on every instrument; kept, not credited.
- 15e decode softmax serial: −0.27 M, but only on the per-head-tensor arm
  that is no longer the default; not credited.
