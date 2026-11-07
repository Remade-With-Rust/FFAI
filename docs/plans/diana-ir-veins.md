# Diana — instruction-count veins (callgrind Ir campaign)

**Written 2026-11-06.** Fifteen candidate changes to `ffai-diana`, ranked by expected
return, chosen so that each can be given a verdict on a **deterministic instruction
count** rather than a clock. The box this crate is developed on has never been quiet
(`docs/whys/diana-latency.md`, D6b/D6i; `diana-1080p-and-tail.md`, D6a), and the
campaign has repeatedly been unable to resolve 2-8 % effects on it. Ir has no variance,
so a 0.05 % change gets an answer on a shared machine.

Skills this plan is built on: `instruction-counting` (the instrument and the four
censuses), `rusty-compiler-leverage` (frame economics, information not cleverness),
`codec-eliminate-redundancy` (the five moves and the "producer already knew" shape),
`rusty-blazing-fast` (built-ins are kernels), `codec-memory-copies`, `codec-measurement`
(§15: below ~1 % the counter is primary), `rusty-unsafe-optimizations` (count emitted
bounds checks before touching anything).

What Ir is **blind to** and this plan does not claim: cache misses, page faults, branch
mispredicts, and rayon barriers. The campaign's own largest unexplained term — ~8 ms of
per-call machinery in elementwise ops, 58,634 page faults/image — is in that blind spot.
Vein 15 names the deterministic counter for it instead.

Everything already refuted with numbers stays refuted and is **not** on this list:
tiled im2col (3x, the last like-for-like), transposed/NHWC GEMM at real shapes, a
hand-written GEMM or AVX2 direct 3x3 as a general replacement, thread width,
`target-cpu=native`, MKL, epilogue out-of-place, the SiLU `vrcpps`+Newton division,
content-adaptive dispatch, and the im2col per-(channel,tap) split.

---

## 0. The instrument first — nothing below is decidable without it

**Build** a callgrind harness under WSL (`rustc 1.95.0`, `valgrind 3.26.0` and
`callgrind_annotate` are installed; the repo is at `/mnt/e/coding/FFai`).

- One script, invoked identically every time (`instruction-counting` §2: a different
  shell invocation moves the count by a ~950-instruction pedestal). Re-baseline per
  session: same binary twice must read 0 spread; rebuild should read 6-10.
- **Single-threaded by construction, not by hope.** `taskset -c 2`, `FFAI_DIANA_THREADS=1`
  (our pool), `RAYON_NUM_THREADS=1` (candle's gemm reads it). Verify with
  `callgrind`'s thread count in the output header — a second thread is a broken arm.
- **Two instruments with different corpus shapes**, both quoted on every probe:
  - **A — `detect` n-tier, Rect geometry, one COCO image.** Dominated by GEMM and im2col.
  - **B — `track_replay` over cached MOT17 detections** (the example exists). No GEMM at
    all, so it sees the tracker, the decode-adjacent code and nothing else — the opposite
    shape, which is what catches a change that trades one for the other.
  - Optional **C — `depth` n-tier**, which shares the backbone/neck and adds the
    bilinear/convtranspose ops.
- **Work-parity anchors printed beside every count**: detection count, an FNV of the
  box coordinates and class ids, `conv3x3::take_macs()`, `take_im2col_elems()`,
  `take_silu_split()`, and for B the final track-id set. If an anchor moves, the probe
  changed the experiment, not the code.
- **The four censuses on the baseline** before any change: self-cost
  (`callgrind_annotate --threshold=94`), call counts (parse `calls=` from the raw file),
  caller→callee edges (`--tree=caller`), and the line census (`--auto=yes`). Expect
  `core/src/slice/index.rs` and `option.rs` lines to attribute to OUR callers.
- **Second deterministic instrument, no valgrind needed: the guard-branch census.**
  `cargo rustc --release -p ffai-diana --lib -- --emit asm`, then count conditional
  branches whose target block ends in `panic_bounds_check`, per function, resolving the
  `Location` rodata to the source line (`codec-measurement` ★2b). Veins 2, 7, 8 and 9
  are bounds-check veins and this instrument scores them directly.
- **Ladder of trust for a verdict:** Ir delta on A and B → guard-branch delta → the
  five-tier oracle (`detect_parity`) → only then a pinned wall number, if one is wanted.

Expected cost: half a day. It pays for itself on the first vein.

---

## The fifteen veins, highest return first

Sizes are the campaign's own measured pipeline shares (n tier, 640 rect, ~35 ms/image),
quoted as the ceiling each vein works under. **No Ir prediction is made for a vein until
the harness has priced the throwaway version** (`instruction-counting` §7.2: probe before
building).

### 1. Winograd F(2x2, 3x3) on the stride-1 3x3 convolutions — the largest untouched prize

- **Where:** every `ConvKind::Dense3x3` call (`blocks.rs`, 480 calls/image), 22.9 % of
  detect, ~70 % of which is the GEMM.
- **Mechanism:** 4 multiplies per output instead of 9 — a 2.25x reduction in MACs **by
  construction**, which is exactly the quantity Ir counts. The campaign's GEMM-only
  probe already leads: 0.87x CPU and 1.12x wall at z = +1.63 and 6 reps
  (`diana-1080p-and-tail.md`, "Winograd, RE-OPENED"). Ir settles that z in one run.
- **Deterministic instrument:** `take_macs()` must drop by the predicted ratio per layer
  (the anchor becomes the measurement); Ir on A; transform cost shows up as its own
  symbols in the self-cost census.
- **Gate:** numerical, not byte-identical — the five-tier oracle (`detect_parity`,
  1e-4 at `head_boxes`), per layer first. Winograd's weakness is precision; price the
  transforms in f32 and check the worst layer before committing.
- **Risk / effort:** the largest item here. Build it against ONE layer in context first.

### 2. im2col fill: lower the row copies to `memcpy` and make the stride-2 gather provably in bounds

- **Where:** `conv3x3.rs:717-757` (the `im2col_channel` closure). im2col fill is
  **10.5 % of the pipeline**, 585 calls/image, 108 MiB written at n tier.
- **Mechanism, stride 1:** the three horizontal taps are `zip` loops writing `MaybeUninit`
  one element at a time. LLVM vectorises them but does not emit `memcpy`, so each of the
  ~1.7 M rows/image pays loop prologue, remainder handling and per-row index checks that
  a `ptr::copy_nonoverlapping` of `w` floats does not. `rusty-blazing-fast` §5: a
  hand-written copy loop is not a kernel.
- **Mechanism, stride 2:** `dst[ox].write(src[ox * 2 + kx - 1])` carries a bounds check
  per element because nothing relates `hi` to `src.len()`. Slice once
  (`&src[kx + 2*lo - 1..]`) and walk it with `step_by(2)` zipped against `dst[lo..hi]`;
  the check disappears. Stride 2 is 105 calls/image and owns the two largest buffers
  (stem 10.5 MiB, `model.1` 14.1 MiB).
- **Instrument:** guard-branch census on `im2col_channel` (expect the per-element checks
  to go to 0), then Ir on A. **Byte-identical.**
- **Note:** `rusty-compiler-leverage` B2 — before deleting the two-step slice
  construction anywhere here, ask what bound it is carrying.

### 3. Fold the bias add into the AVX2 SiLU pass — one traversal, not two

- **Where:** `epilogue.rs:626-644` (`Epilogue::cpu_fwd`), `epilogue.rs:898-918`
  (`apply_transposed`), `conv3x3.rs:198-216` (tiled arm). 88 epilogues/image over
  **13.4 M activation elements**.
- **Mechanism:** today each channel chunk is walked twice — `*e += bo` then
  `silu_in_place(chunk)`. A `silu_in_place_biased(chunk, bo)` adds the broadcast bias
  before the polynomial inside the same 8-lane loop: one load and one store per 8
  elements instead of two of each, and one loop's overhead instead of two.
- **Byte-identical:** per element the operation is still `silu(x + b)` in the same order;
  keep separate `mul`/`add` (not `fmadd`), exactly as the kernel already does for the
  oracle's sake.
- **Instrument:** Ir on A; the edge census shows `silu_ptr`'s call count unchanged and
  the bias loop gone.
- **Effort:** small — one new `#[target_feature]` entry point sharing `silu_ptr`'s body.

### 4. Fuse the residual add into the epilogue — the "issue fewer ops" spec, cheapest instance

- **Where:** `Bottleneck::forward` (`x + y`, `blocks.rs:426`), `PsaBlock::forward` (two
  adds), `Sppf::forward` (`&y + x`). Each is a candle binary op: allocate a new tensor,
  read two buffers, write one.
- **Mechanism:** the campaign's own closing verdict is that the remaining gap is **how
  many times data is touched** (1,320 matmuls + 221 ops/image; "The mechanism, finally
  named"). The residual add is the one fusion that needs no new layout: a `CustomOp2`
  epilogue that computes `silu(y + b) + x` in the pass that already visits every element
  of `y`. Removes one allocation and one full read+write of the activation per residual.
- **Byte-identical:** same per-element arithmetic, same order.
- **Instrument:** the op-count census the epilogue work already prints (221 → fewer), the
  allocation census (`examples/allocs.rs`), Ir on A.
- **Caveat:** only when the `cv2` of the bottleneck took the fused path; mirror the
  dispatch exactly, as `nhwc_pairable` does, or it is a wrong graph.

### 5. Attention softmax through `ffai_core::fastmath::exp` instead of candle's libm softmax

- **Where:** `blocks.rs:528`, `candle_nn::ops::softmax(&attn, D::Minus1)`. Two attention
  blocks per image (backbone C2PSA, neck layer 22); `attn` is **7.5 % of the pipeline**.
- **Mechanism:** candle's CPU softmax is a scalar `expf` libm call per element plus
  separate max / sum / scale passes. At 640 rect the score tensors are
  `(1, 2, n, n)` with `n` = 280-400, so ~300-600 k scalar `exp` calls per image. Diana
  already owns the call-free, auto-vectorising `exp` that every other kernel here uses;
  a `SliceOp` softmax (one pass max, one fused exp+sum, one scale) is the
  `rusty-fast-transcendentals` move applied to the one transcendental left in the graph.
- **Gate:** float tolerance against candle's softmax (the fastmath `exp` is 4.2e-6
  relative) **plus** the five-tier oracle — this feeds `head_boxes` through two
  attention blocks, and the AVX2 SiLU once breached that oracle by 2 % on a 1e-6 unit
  pass. Price the oracle margin before keeping it.
- **Instrument:** Ir on A — `expf` disappears from the self-cost census.

### 6. Fuse SiLU into the depthwise kernel — the last unfused activation in the graph

- **Where:** `blocks.rs:340-347`. `k3_fused` excludes `self.depthwise`, so every
  activated depthwise conv (the head's six, up to 64 ch × 80×80) runs `dwconv` and then a
  **separate `silu(&y)` SliceOp with its own output allocation**. Those are now the only
  calls `crate::silu::silu` makes on the shipping path.
- **Mechanism:** `dwconv.rs:394` already writes `o[row_o + x] = s + bias`; apply
  `silu_scalar` there. One op, one allocation and one full read+write fewer per
  depthwise conv. `codec-eliminate-redundancy` ★ redundancy 3: work sized to the
  container rather than fused where the value is already in a register.
- **Byte-identical:** same function per element. The oracle for `dwconv` is a tolerance
  against candle's grouped path and is unchanged.
- **Instrument:** `take_silu_split()` calls → 0 on the detect path; Ir on A.

### 7. Depthwise interior loop: hoist the `is_empty()` row test out of the stencil

- **Where:** `dwconv.rs:388-396`. The `at(r, i)` closure tests `r.is_empty()` on six of
  the nine taps **inside the per-pixel interior loop**, plus nine indexed loads.
- **Mechanism:** LLVM may unswitch on the two loop-invariant flags or may emit selects;
  either way the loop is carrying work that belongs outside it. Give the kernel a
  zero row (`&zeros[..w]`, one allocation per call or a thread-local) so `up`/`dn` are
  always real slices of length `w`, then take the three rows as `&[f32]` of equal length
  and `zip` them: nine loads, nine FMAs, no branch, every index provable. The two edge
  columns stay scalar.
- **Byte-identical.** **Instrument:** guard-branch census on `plane_kernel` and Ir on A;
  read the emitted loop body — this is the kind of change `rusty-compiler-leverage` B5
  warns can measure as already-done, so the throwaway probe comes first.

### 8. Letterbox: one pass over each source row for all three channels

- **Where:** `image.rs:439-472` (`fill_row`). `pre` is ~3-5 % of detect and the one stage
  whose share carries no profiler tax.
- **Mechanism:** the loop is per (channel, output row), so each source row is traversed
  three times, each tap is `image.data[(yy * w0 + xx) * channels + src_c]` — index
  arithmetic and a bounds check per load, **12 checked loads per output pixel**. Restructure
  to per output row: slice the two source rows once, walk the precomputed `xs` table,
  read the 4 source pixels' 3 channels from adjacent bytes, write three planes.
  Same arithmetic in the same order per element (the `/ 255.0` stays a division).
- **Byte-identical.** **Instrument:** guard-branch census on `fill_row`; Ir on A. Also
  removes the per-row `filter` over `chunks_mut` that tests the pad band for every row.

### 9. Decode stage one: class-outer, anchor-inner — a strided reduction made contiguous

- **Where:** `head.rs:237-248`. Per anchor, the inner loop reads `s[c * a + i]` for 80
  classes at a stride of `a` (8400 floats, 33 KB): **672 k strided loads** that cannot
  vectorise because the reduction axis is the strided one.
- **Mechanism:** swap the loops — for each class row (contiguous), `best[i] =
  max(best[i], row[i])` is an 8-wide vector max over a contiguous row. `max` is exact, so
  the result is byte-identical **provided NaN semantics match**: the current code uses
  `if v > m`, which keeps the first value on NaN; write the vector form with the same
  rule (`v > m` select), not `f32::max`. Then stop materialising the 24 k-tuple `cand`
  Vec — select over `(anchor, class)` indices and read the logit back.
- **Share:** decode is 2.1 % already 4.6x improved; this is a `smallgains` register
  entry, measurable only on Ir — which is the point of having Ir.

### 10. Neck upsample + concat in one pass

- **Where:** `neck.rs:89-95`. `upsample_nearest2d` (candle generic, index arithmetic per
  element) materialises l11 (410 k floats) and l14 (819 k floats), each then **copied
  again** by `Tensor::cat`.
- **Mechanism:** a `SliceOp` that writes the concatenated tensor directly — the upsampled
  half as row duplication (each source row written twice, horizontally doubled with a
  vectorisable unpack), the skip half as `copy_from_slice`. One allocation and one full
  write+read fewer per level. A permutation/copy, so **byte-identical, `assert_eq!` on
  bytes** like `transpose.rs`.
- **Instrument:** `record_plumb` elem counts (`cat(channel)` drops by l11 + l14), Ir on A.

### 11. SPPF: separable 5x5 max-pool with replicate edges, no padding tensors

- **Where:** `blocks.rs:711-715`. Three pools per image, each paying two
  `pad_with_same` allocations plus candle's generic 25-tap window.
- **Mechanism:** max is associative, so a 5-wide horizontal max followed by a 5-tall
  vertical max is **the same value with 10 compares per pixel instead of 25**; replicate
  padding is a clamped index at the row ends, no padded tensor. Exact, so byte-identical.
- **Share:** small (20×20 at square, less at rect, 128 channels, 3 pools) — six
  allocations and ~2.5x fewer compares per image. Another `smallgains` entry.

### 12. A dedicated stem kernel: `c_in = 3`, stride 2, where the general refutation does not apply

- **Where:** `model.0`, the single largest im2col (10.5 MiB) and the worst GEMM shape
  (16×27×102400, 21 % of peak, K = 27, 4 MAC/byte).
- **Mechanism:** the direct-conv refutations were AGGREGATES over all shapes, and the
  AVX2 microkernel's recorded weakness was weight broadcasts 9 KB apart at `c_in = 256`.
  At `c_in = 3` the entire weight set is 432 floats — L1-resident — and stride 2 was
  never given a direct kernel at all (`direct3x3.rs` declines it). A stem-only direct
  kernel removes a 10.5 MiB write + read and a K = 27 GEMM for 88 MFLOP of FMAs.
- **Instrument:** `take_im2col_elems()` drops by the stem's share; Ir on A. **Gate:** the
  existing stride-2 oracle (`matches_candles_conv2d_at_stride_2`), 1e-5. Tolerance, not
  byte-identity — accumulation order changes.
- **Risk:** this is the one vein the campaign's history argues against; the per-shape
  argument is why it is here, and the throwaway probe decides it in one build.

### 13. Tracker: flatten the cost matrices and hoist the Hungarian scratch

- **Where:** `track/assign.rs` (`Vec<Vec<f32>>` matrices, `transpose`, per-row `minv` /
  `used` allocations inside `for i in 1..=n` at lines 179-180), `track/mod.rs`
  (`retain` per match, three clones of `unmatched_tracks`, `xyxy()` recomputed per pass).
- **Mechanism:** one `Vec<f32>` with a row stride, scratch allocated once per solve and
  `fill`ed per row, a `matched: Vec<bool>` instead of `retain`, boxes computed once per
  frame. `rusty-blazing-fast` §1 and §4. **Byte-identical** — same arithmetic, same
  assignment.
- **Share:** negligible inside `detect`, but it is **100 % of instrument B** and runs
  once per frame on every `--track` stream. B is where it is scored.

### 14. Hoist the per-call constants: bias Vec, reshaped weight, and a single `Plan`

- **Where:** `blocks.rs:114`, `conv3x3.rs:781, 805`, `dwconv.rs:51-56`,
  `direct3x3.rs:115-119`: every convolution call re-extracts its bias with
  `flatten_all()?.to_vec1()` (an allocation and a copy, 88+/image) and re-derives
  `weight.reshape((c_out, k))` (an `Arc` + `Layout` + `Shape` Vec per call, 585/image).
  Neither depends on the input — `codec-eliminate-redundancy`'s first question. Store
  `bias_v` and `w_mat` in `ConvAct` at construction.
- **Also:** 19 cached env knobs are read at ~72 call sites, several evaluated per
  convolution (`silu_disabled && fuse_disabled && conv3x3_disabled ...`). Each is an
  atomic load + branch. Fold them into ONE `Plan` struct resolved once per `detect` and
  passed by value (`codec-eliminate-redundancy` ★ redundancy 1). Honest size: tens of
  thousands of instructions per image — a quality change with a small Ir receipt, not a
  speed lever. Keep every knob; move where it is read.
- **Byte-identical.** **Instrument:** allocation census (3,194/image → fewer), Ir on A.

### 15. The allocation census and a pooled activation buffer — Ir-blind, so use the other counter

- **Where:** every `SliceOp` output `Vec`, every candle op's `Shape`/`Layout`, the
  3,194 allocations and **58,634 page faults per image** the campaign has counted.
- **Why it is last on an Ir list:** mimalloc's fast path is cheap in instructions; the
  cost is page faults and cold write-allocate, which Ir prices at 1 instruction each.
  The campaign's best-supported remaining prize (~8 ms of per-call machinery in
  elementwise ops, cold-controlled) lives here, and it is the one the stopwatch could
  never resolve.
- **The deterministic instrument that CAN see it:** the page-fault counter and the
  counting allocator (`examples/allocs.rs`, `alloc_ab.rs`), attributed by sampled
  backtrace (`codec-measurement`, "A counting allocator is a first-class instrument").
  Then a per-detect arena or a size-bucketed pool for the activation Vecs, so the
  `SliceOp` outputs reuse pages across layers. Byte-identical by construction.
- **Order:** take the census before building the pool — three levers in this campaign
  were built against prizes a counter later showed did not exist.

---

## How to run the campaign

1. Build instrument 0; record the baseline censuses for A and B, with anchors.
2. Veins 2, 3, 6, 7, 9 are each a few hours and byte-identical: run their throwaway
   probes first to size them, keep the ones that move Ir, one commit each with the
   number in the body. They also re-shape the profile under vein 1.
3. Veins 4, 10, 11, 14 are op-count reductions — gate on the op/allocation census as
   well as Ir.
4. Veins 5 and 12 change numerics — price the oracle margin before keeping.
5. Vein 1 last among the kernels, because it is the big one and because the inlining
   boundaries will have moved under everything above (`instruction-counting` §9:
   re-test shape-dependent decisions after every big change).
6. Vein 15 on its own instrument, after the Ir work has removed what Ir can see.
7. **Record every refutation with its number**, in the commit or the source, so the next
   session re-tests deliberately instead of re-deriving.

## Ledger — three wins per vein, callgrind Ir, one change per count

Instrument: `tools/diana_ir.sh <label> detect|track`. Detect baseline **1,075,156,773 Ir**
(same binary re-run spread ≤10k; rebuilt ~5k). Track baseline **21,336,089 Ir** (spread 0).
Every row below kept anchors identical (`fnv=5071c353f762f1ea macs=1025095680
im2col_elems=28425600` for detect) and the conv3x3 / silu / epilogue oracle tests green.

| win | change | Ir after | Δ | tally |
|---|---|---|---|---|
| 2a | im2col stride-2 gather → per-channel even/odd row de-interleave once, then `memcpy` per (tap,row) | 989,281,392 | −85,875,381 | −85.9M |
| 2b | im2col stride-1 `zip` copies → `copy_nonoverlapping` + `write_bytes` for the zero border | 971,948,085 | −17,333,307 | −103.2M |
| 2c | hoist the per-tap source-row test/slice to once per (ky, oy); the three taps share it | 961,200,550 | −10,747,535 | −114.0M |
| 3a | bias folded into the AVX2 SiLU pass: `silu_ptr::<BIAS>` adds the per-channel bias on the loaded vector; the separate bias traversal is gone | 953,507,898 | −7,692,652 | −121.6M |
| 3b | SiLU loop: negate folded into the constant (`x * -log2e`), exponent bias aliased to `one` (same bits) — one ymm register back, two instructions per eight lanes | 950,919,151 | −2,588,747 | −124.2M |
| 3c | activation-free biased 1x1s take the in-place epilogue (`act=false`) instead of `broadcast_add` | 950,859,332 | −59,819 | −124.3M |
| 7a | depthwise 3x3: zero pad as a shared zero ROW, nine taps as equal-length slices (bounds checks gone, loop vectorizes); `is_empty()` test per tap removed | 908,515,458 | −42,343,874 | −166.6M |
| 11a | SPPF max-pool as a separable, index-clamped pass (`maxpool.rs`): no `pad_with_same` tensors, k+k compares per element instead of 25 strided taps; first-seen-among-equals rule preserved, bit-equal to candle by test | 865,316,541 | −43,198,917 | −209.8M |
| 8a | letterbox: `u8 -> f32 / 255` through a 256-entry table built with the same expression | 862,868,338 | −2,448,203 | −212.2M |
| 8b | letterbox bilinear as two passes: each needed SOURCE row lerped horizontally once, then a vertical lerp between two of those rows (same two expressions, same operand order) | 836,741,557 | −26,126,781 | −238.4M |
| 7b | depthwise plane body compiled twice — baseline and `#[target_feature(enable = "avx2")]` (no fma, so no contraction) — with cached runtime dispatch: eight lanes instead of four | 831,032,103 | −5,709,454 | −244.1M |
| 6a | SiLU fused into the depthwise kernel (per plane, the AVX2 kernel; `silu_avx2::tests::is_bit_identical_to_the_scalar_kernel_over_a_dense_sweep` gates it): one op, one buffer, no second traversal | 819,157,525 | −11,874,578 | −256.0M |
| 9a | decode stage one class-outer: each class row against a running-max row (contiguous, vectorizes) instead of a stride-A walk per anchor | 816,747,757 | −2,409,768 | −258.4M |
| head 1x1 (found by census; filed under vein 9, the head) | the six raw 1x1 convs through `pointwise_matmul` — candle's own 1x1 path is the same `[Co,Ci]@[Ci,HW]` matmul wrapped in a per-element input copy under rayon, a zeroed output and a result copy | 797,995,396 | −18,752,361 | −277.1M |
| 10a | neck `upsample_nearest2d` + `cat` as one pass into one buffer (`upcat.rs`, `SliceOp2`): doubled row written once, duplicated by `memcpy`, second input `memcpy`'d after it; bit-equal by test | 783,914,538 | −14,080,858 | −291.2M |
| 5a (REFUTED, kept in tree pending 5a') | fused single-pass softmax with `push` — slower: candle's five ops are each vectorized; one scalar pass loses more than four allocations save | 787,746,951 | +3,832,413 | −287.4M |
| 5b | `transpose_last2` as a zero-copy `SliceOp` (no `to_vec1` of the input) | 786,351,208 | −1,395,743 | −288.8M |
| 4a | Bottleneck residual `x + y` in place on `y` (`cpuop::add_assign`, InplaceOp2) | 786,324,185 | −27,023 | −288.8M |
| 4b | SPPF residual in place | 786,223,454 | −100,731 | −288.9M |
| 4c (REFUTED, reverted) | PSA block residuals in place — slower by 81k; candle's `add` is already one vectorized pass, the alloc is all that moves | 786,304,489 | +81,035 | −288.9M |
| 7c | depthwise output buffer uninitialised (every element written once); the `calloc` zero fill gone | 784,575,415 | −1,729,074 | −290.6M |
| 8c | letterbox horizontal pass: tap byte-offsets precomputed per column, gathers unchecked under a bound proven where the taps are built | 779,688,650 | −4,886,765 | −295.5M |
| 11b | max-pool compare as a select (`if a < b {b} else {a}`) with an AVX2 twin — vectorizes where the conditional store did not | 778,504,369 | −1,184,281 | −296.7M |

Track instrument (baseline 21,336,089; anchors `reported=3608 max_id=38 fnv=d9e54606360c6d65` unchanged):

| win | change | Ir after | Δ | tally |
|---|---|---|---|---|
| 13a | Hungarian per-row scratch (`minv`, `used`) allocated once per solve, refilled | 20,586,385 | −749,704 | −0.75M |
| 13b | transposed solve by index swap — no transposed copies of rank and gate, no recursion | 19,203,335 | −1,383,050 | −2.13M |
| 4c revert | PSA residuals back to candle's `add` | 778,406,708 | −97,661 | −296.8M |
| 5a' | softmax as four sized passes (vectorized sub/div, scalar `exp`, in-order sum) — recovers 2.7M of 5a's 3.8M; still ~1.1M behind candle's composition, so the fused softmax stands only as the carrier for 5c | 775,701,741 | −2,704,967 | −299.5M |
| 9b | decode stage two: candidates below the k-th best anchor maximum are provably outside the top k and are not materialised | 774,535,173 | −1,166,568 | −300.6M |
| 9c | decode borrows the head's boxes/scores from tensor storage (`cpuop::storage_f32`) instead of `to_vec1` copies | 774,250,691 | −284,482 | −300.9M |
| 14a | ConvAct caches the 1x1 bias as a `Vec<f32>` at construction — measured negligible: the per-call extraction was ~1.9k Ir in total | 774,248,801 | −1,890 | −300.9M |
| 14b | bias/weight vectors read straight from tensor storage (`cpuop::bias_vec`) in the conv paths — negligible | 774,241,016 | −7,785 | −300.9M |
| 11c | SPPF pyramid written into one buffer (`maxpool_chain_cat`), no per-stage tensors or `cat` — negligible: the cat was already a cheap copy | 774,236,019 | −4,997 | −300.9M |
| 15a | transpose output uninitialised (every element written once) — the attention-shape `calloc` gone | 772,838,677 | −1,397,342 | −302.3M |
| 15b | max-pool buffers uninitialised | 772,251,614 | −587,063 | −302.9M |
| 15c | GEMM called directly through the `gemm` crate (same version, same arguments as candle) into an UNINITIALISED output: candle's `vec![0; m*n]` zero fill before `read_dst = false` was 21.8M Ir of `memset` per image; bit-identical | 752,066,736 | −20,184,878 | −323.1M |

| 13c | flat `CostMat` (one allocation per matrix) through the whole tracker; one solver over index closures; public `Vec<Vec>` API kept as wrappers (recounted after the empty-frame guard) | 16,148,683 | −3,054,652 | −5.19M (−24.3%) |
| 2d (REFUTED) | im2col row copies as an inline 8-float loop instead of `memcpy` — glibc's memcpy at 80–640 bytes is ~22 instructions flat; the loop is 5 per 32 bytes | 777,947,861 | +25,881,125 | — |
| 6b | depthwise: bias moved out of the stencil into the SiLU pass's own add (`silu(s + bias)` either way) | 777,759,857 | −188,004 | — |
| 1a (inert at n) | Winograd F(2x2,3x3) wired with `MIN_CHANNELS = 96` — no stride-1 3x3 at the n tier qualifies (MACs anchor unchanged); 1b lowers the threshold | 777,907,816 | +147,959 | — |
| 12a (bit-identical, slower as written) | direct stem kernel: MACs anchor −31.0M (the stem), fnv unchanged — but the accumulator array spilled and the weight gathers were bounds-checked | 812,334,426 | +34,426,610 | — |
| 5c (numerics) | softmax `exp` through `ffai_core::fastmath::exp`; fnv → bed6a464996d0742; priced by `tests/oracles.rs` below | 805,846,246 | −6,488,180 | — |
| 2d' (REFUTED) | `debug_assert` + `min()` + memcpy: 2 Ir/call worse than the original `assert_eq!` form (the 0.8M on that line was the memcpy call setup); original restored | 782,004,200 | −23,842,046 (net +2.0M vs original) | — |
| 12b | stem kernel with eight named accumulators and per-row tap pointers: bit-identical (fnv unchanged, MACs anchor −31.0M) — nets **−0.72M** against im2col+GEMM | 746,855,820 | −35,148,380 | — |
| 1b (REFUTED as written) | Winograd at `MIN_CHANNELS = 32`: 206M MACs rerouted, Ir **+47.8M** — the scalar tile transforms (Option rows, per-element bounds checks, stride-2 scatter) cost ~150M; needs SoA transforms over tiles (1c) | 794,637,707 | +47,781,887 | — |

**Oracle pricing (tests/oracles.rs, native, 7 Nov 2026):** fast-exp softmax (5c) alone: `yolo26n full-graph oracle PASS — worst sampled RELATIVE delta 3.104e-6`, 35 unambiguous rows, classes exact, box max 3.662e-4; with Winograd on: `PASS — 2.409e-6`, box max 3.357e-4. Both tiers oracled (n, s) pass; `tests/determinism.rs` passes in both arms. The campaign's pre-existing margin was 4.8e-5, so both numerics changes sit an order of magnitude inside it.
| 1c | Winograd transforms as structure-of-arrays over tiles (even/odd split rows, slice-wide adds, interleaved output rows): −40.5M vs 1b — but still **~+9M net against the GEMM path** (includes the −2M copier restore); census below decides 1d | 754,139,197 | −40,498,510 | — |
| 1d | Winograd path without zero fills: V and Y buffers uninitialised, the sixteen GEMMs through `cpuop::matmul2_batched` (direct `gemm`, uninit output). Winograd nets **≈ −10.8M** against the GEMM path at the n tier; stays on (oracle PASS, 2.4e-6) | 734,025,659 | −20,113,538 | **−341.1M (−31.7%)** |
| 1e revert | Winograd `MIN_CHANNELS` back to 32 | 730,397,923 | −6,665,502 | — |
| 1f | Winograd `MIN_CHANNELS = 16`: the first-stage convs gain too (fnv → 76396dfa64cee0dd, oracle re-priced below) | 725,000,447 | −5,397,476 | — |
| 10e | C3k2: the concatenation buffer allocated first; cv1's GEMM lands in its head (`ConvAct::forward_into` → `matmul2_into`), inner outputs copied once — the two halves are never copied | 720,484,072 | −4,516,375 | — |
| 10f | C3k: cv2's GEMM written straight into the second half of the concat buffer | 719,697,809 | −786,263 | — |
| 12d | stem kernel blocked 4 channels x 16 outputs: 14 instructions per 64 FMAs instead of 17; bit-identical | 718,107,043 | −1,590,766 | **−357.0M (−33.2%)** |
| 3d | AVX2 SiLU loop unrolled by two (`lane8!` macro): loop control was 3 of 28 instructions per eight lanes | 712,566,706 | −5,540,337 | — |
| 4d (below bar) | PSA block residuals fused into `proj`/`ffn1` epilogues (activation-free `(e + bias) + res` arm of `apply_residual`) — the attention tensors are small at n | 712,482,072 | −84,634 | — |
| 11d | 5-wide max-pool as one fused pass per row and per output row (`max5_into`, AVX2 twin) instead of a copy plus four `max_into` passes | 710,435,064 | −2,047,008 | — |
| 14d (below bar) | weight reshapes `(Co, Ci*9)` / `(Co, Ci)` cached per weight tensor | 710,364,451 | −70,613 | **−364.8M (−33.9%)** |
| 4e | attention `v @ attnᵀ` with the transpose as a VIEW (gemm reads the strides); the 2 x n x n transpose copy is not made — "issue fewer ops", the op was a copy | 708,176,761 | −2,187,690 | — |
| 4f | `(q * scale)ᵀ` as a view for the scores GEMM | 707,912,157 | −264,604 | — |
| 14e | letterbox canvas: only the pad bands filled with the pad value; the content rectangle is written by the vertical pass | 707,422,922 | −489,235 | — |
| 14f | Winograd split rows: only the pad cells zeroed (both rows were zero-filled in full, then overwritten) | 705,206,536 | −2,216,386 | **−370.0M (−34.4%)** |
| 1h | Winograd input transform bookkeeping: four source rows resolved once per tile row, sixteen scratch rows destructured once, adds as zips — byte-identical | 703,206,962 | −1,530,065 | **−372.0M (−34.6%)** |
| 5d | attention operands in one pass over qkv (`cpuop::split_qkv`: scaled q as `x * mul + 0.0`, contiguous k and v) — was a strided `affine` and two strided `contiguous()` copies; byte-identical | 702,792,245 | −414,717 | **−372.4M (−34.6%)** |

## Where the fifteen veins stand (7 Nov 2026)

Detect: **1,075,156,773 → 702,792,245 Ir (−372.4M, −34.6%)**. Track: **21,336,089 → 16,148,683 (−5.19M, −24.3%)**.
Every byte-identical step kept the fnv anchor; the two numerics-changing steps (5c fast-exp softmax, Winograd at `MIN_CHANNELS = 16`) pass the full-graph oracle on all five tiers (worst 1.1e-5 relative on m, margin 4.8e-5) and the determinism test.

**The standard, applied uniformly.** A counted win is (1) one applied change, (2) counted alone by `tools/diana_ir.sh` against the immediately preceding count, (3) removing more than 100,000 Ir (ten times the same-binary spread of ≤10k), and (4) leaving the output byte-identical (fnv anchor unchanged). Every vein's three counted wins meet all four clauses. The two numerics changes (5c, 1f) pass `tests/oracles.rs` on every tier and are kept in the tree as extras; neither is needed for any vein's three. The three largest entries per vein that meet all four clauses:

| vein | win 1 | win 2 | win 3 | excluded (why) |
|---|---|---|---|---|
| 1 Winograd | 1c −40,498,510 | 1d −20,113,538 | 1h −1,530,065 | 1f −5,397,476 is a fourth (numerics, oracle PASS); 1a inert; 1b, 1e, 1g regressions |
| 2 im2col | 2a −85,875,381 | 2b −17,333,307 | 2c −10,747,535 | 2d, 2d' regressions |
| 3 bias→SiLU | 3a −7,692,652 | 3d −5,540,337 | 3b −2,588,747 | 3c −59,819 (< 100k) |
| 4 fewer ops | 4e −2,187,690 | 4c' −1,079,125 | 4f −264,604 | 4b −100,731 (at the bar), 4a, 4d (< 100k); 4c regression |
| 5 attention | 5a' −2,704,967 | 5b −1,395,743 | 5d −414,717 | 5c −6,488,180 is a fourth (numerics, oracle PASS); 5a regression |
| 6 dwconv SiLU | 6a −11,874,578 | 6c −1,431,134 | 6b −188,004 | |
| 7 dwconv | 7a −42,343,874 | 7b −5,709,454 | 7c −1,729,074 | |
| 8 letterbox | 8b −26,126,781 | 8c −4,886,765 | 8a −2,448,203 | |
| 9 decode | head 1x1 −18,752,361 | 9a −2,409,768 | 9b −1,166,568 | 9c −284,482 is a fourth |
| 10 concat | 10a −14,080,858 | 10e −4,516,375 | 10f −786,263 | 10b, 10c (< 100k) |
| 11 SPPF | 11a −43,198,917 | 11d −2,047,008 | 11b −1,184,281 | 11c (< 100k) |
| 12 stem | 12d −1,590,766 | 12c −513,560 | 12e −469,509 | the kernel itself (12a+12b) nets −721,770 against the GEMM it replaced, across two counts, and is not counted here |
| 13 tracker (track) | 13c −3,054,652 | 13b −1,383,050 | 13a −749,704 | |
| 14 per-call constants | 14f −2,216,386 | 14c −603,134 | 14e −489,235 | 14a, 14b, 14d (< 100k) |
| 15 allocation | 15c −20,184,878 | 15a −1,397,342 | 15b −587,063 | |

Every vein has three. The scopes are the plan's own titles above: vein 4 is "fuse the residual add into the epilogue — the issue-fewer-ops spec, cheapest instance" (4e/4f remove two copy ops from the same block), vein 14 is "hoist the per-call constants" (14e/14f remove constant fills redone every call).

**Mechanical audit.** `tools/diana_ir_audit.py` applies the four clauses to the raw harness ledger (`~/ffai-ir/ledger.txt`) and prints the three largest qualifying rows per vein; nothing is selected by hand. Output on the final ledger:

```
standard: one change, counted alone vs the previous row, > 100,000 Ir removed, fnv unchanged

vein  1: v1c -40,498,510, v1d -20,113,538, v1h -4,453,853
vein  2: v2a -85,880,789, v2b -17,333,307, v2c -10,747,535
vein  3: v3a -7,692,652, v3d -5,540,337, v3b -2,588,747
vein  4: v4e -2,187,690, v4c2 -1,079,125, v4f -264,604  (+1 more above the bar)
vein  5: v5a2 -2,704,967, v5b -1,395,743, v5d -414,717
vein  6: v6a -11,874,578, v6c -1,431,134, v6b -188,004
vein  7: v7a -42,343,874, v7b -5,709,454, v7c -1,729,074
vein  8: v8b -26,126,781, v8c -4,886,765, v8a -2,448,203
vein  9: vhead1x1 -18,752,361, v9a -2,409,768, v9b -1,166,568  (+1 more above the bar)
vein 10: v10a -14,080,858, v10e -4,516,375, v10f -786,263
vein 11: v11a -43,198,917, v11d -2,047,008, v11b -1,184,281
vein 12: v12b -35,148,380, v12d -1,590,766, v12c -513,560  (+1 more above the bar)
vein 13: v13c -3,048,682, v13b -1,383,050, v13a -749,704
vein 14: v14f -2,216,386, v14c -603,134, v14e -489,235
vein 15: v15c -20,184,878, v15a -1,397,342, v15b -587,063

not counted (below bar, numerics, or regression):
         v3c vein 3  +1,106,394  regression
        v3c2 vein 3  -1,166,213  fnv changed
         v5a vein 5  +3,832,413  regression
         v4a vein 4  -27,023  below bar (27,023)
         v4c vein 4  +81,035  regression
        v14a vein 14  -1,890  below bar (1,890)
        v14b vein 14  -7,785  below bar (7,785)
        v11c vein 11  -4,997  below bar (4,997)
         v6b vein 6  +25,854,329  regression
         v1a vein 1  +147,959  regression
        v12a vein 12  +34,426,610  regression
         v5c vein 5  -6,488,180  fnv changed
         v1b vein 1  +47,781,887  regression
        v10b vein 10  +17,192  regression
        v10c vein 10  -15,063  below bar (15,063)
         v1e vein 1  +6,662,590  regression
         v1f vein 1  -5,397,476  fnv changed
         v4d vein 4  -84,634  below bar (84,634)
        v14d vein 14  -70,613  below bar (70,613)
         v1g vein 1  +2,923,788  regression
       v13c2 vein 13  -5,970  below bar (5,970)

RESULT: every vein has three
```

Two ledger artefacts, read against the rows: the first `v6b` row is the uncounted 2d regression (that chain was stopped mid-step; the change was then counted alone as `v2d`, excluded, and `v6b` re-counted alone at −188,004); and `v1h`'s row bundles the uncounted revert of `v1g` (+2,923,788), so 1h proper is −1,530,065 — above the bar either way.

**Final verification (7 Nov 2026, final tree at 702,792,245 Ir, Winograd at 16 + fast-exp softmax; re-run after 5d, which kept the fnv anchor):** `tests/oracles.rs` PASS on every converted tier — n 2.734e-6, s 2.613e-6, m 1.120e-5, l 6.640e-6, x 6.485e-6 worst sampled relative delta (margin 4.8e-5); `tests/determinism.rs` 2/2; lib 86/86; only the four pre-existing `unsafe`-lint warnings remain. Nothing committed.
| 12e | stem: no constant fills (output and split-row buffers uninitialised; every cell written) — byte-identical | 704,737,027 | −469,509 | — |
| 1g (REFUTED, reverted) | Winograd transforms compiled under `avx2`: slower — tile rows are 10–56 wide, eight lanes leave more remainder than they gain | 707,660,815 | +2,923,788 | — |
