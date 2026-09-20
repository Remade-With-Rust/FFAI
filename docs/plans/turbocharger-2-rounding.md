# Turbocharger 2 — the rounding class

**Status: 18 of 89 sites converted and gated; the other 71 have a recorded
verdict. 2026-09-19.**

**The vectorisation premise is REFUTED for the resize loops, and the calls are
genuinely gone.** §7 has the disassembly A/B. Half the mechanism delivered — two
`callq floorf` per function became zero — and half did not: the loops did not
widen, because `floor` was never what was stopping them. No timing has been
taken, and on this evidence the expected effect is small. The finding that
matters is next door, in §7.3.

The first turbocharger campaign
([docs/finished/turbocharger.md](../finished/turbocharger.md)) is closed and it
closed properly: Tier 1 took **all 17 candle activation sites**, Tier 2 took
**every hot scalar `exp`/`log` loop**, Tier 3 recorded the setup arithmetic that
must not be touched. `ffai-core`'s
[`fastmath`](../../crates/ffai-core/src/fastmath.rs) and
[`fastops`](../../crates/ffai-core/src/fastops.rs) are the consolidated kernel
and its zero-copy delivery.

**What it never targeted is the rounding class.** Its §4 states the rule —
*"`round()` is SSE4.1, not SSE2"* — but every target in its three tiers was an
`exp`, a `log` or an activation. `floor`, `ceil`, `trunc` and `round` were left
where they sat: **89 of them on shipping paths.**

---

## 1. Why this bites in *this* workspace

The rule is not universal — it depends on the compile baseline, so it has to be
checked per repo. Here it holds.

`.cargo/config.toml` scopes its rustflags to **`wasm32-unknown-unknown`**
(`+simd128`) and to **Linux/GNU linking** (RELRO, noexecstack, frame pointers).
It sets **no `target-cpu` and no `target-feature` for native builds**, and
there is none in `Cargo.toml`, `rust-toolchain.toml` or the CI workflows. So
the native x86-64 baseline is the default one:

| op | instruction | at this baseline |
|---|---|---|
| `floor` / `ceil` / `trunc` | `roundss` / `vroundps` | **SSE4.1 — above baseline.** Lowers to a call or a branchy sequence. |
| `round` | *none* | Rust's `f32::round` is ties-**away-from-zero**; `vroundps` is ties-to-**even**. No x86 instruction implements it at any baseline. |
| `sqrt`, `min`, `max`, `abs`, FMA | SSE2 / FMA3 | Already vectorised. **Do not touch.** |

Two qualifications, both load-bearing:

* **This is a native-only win.** wasm32 has `f32.floor` as a real instruction
  and `f32x4.floor` under `+simd128`, which we already enable. The wasm modules
  do not pay this tax; the native `ffai` binary does.
* **Inside an existing `#[target_feature(enable = "avx2")]` kernel, `floor` is
  free** — the attribute lifts the baseline for that function. None of the
  converted sites were inside such a function; they are plain safe Rust.

---

## 2. The fix is usually *deletion*, not the magic number

This is the difference from the first campaign, and it is why the conversion
was cheap and byte-identical.

`rusty-fast-transcendentals` §2 offers the magic-number round
(`(x + 1.5*2^23) - 1.5*2^23`). That is the right tool when you need
round-to-nearest. **Most sites here did not.** They compute a *floor of a
non-negative value*, and an `as u32` cast already truncates toward zero — which
**is** floor when the value cannot be negative. The `.max(0.0)` on the line
above is the whole licence, and it was re-read at each site rather than assumed.

Two primitives now live in `fastmath`, both exact against libm over a dense
sweep:

| primitive | contract |
|---|---|
| [`floor_frac_nonneg(x) -> (u32, f32)`](../../crates/ffai-core/src/fastmath.rs) | `(floor(x), x - floor(x))` for `x >= 0`. No correction term, no call. Debug-asserts `0 <= x < 2^24`, so a caller that drops its clamp fails loudly instead of silently truncating. |
| [`floor_i32(x) -> i32`](../../crates/ffai-core/src/fastmath.rs) | Exact floor for **any** sign: `x as i32` then subtract the comparison. Branchless, so the loop still widens. |

`a_cast_alone_is_not_floor` pins the trap the second primitive exists for:
`-0.5 as i32` is `0`, `floor(-0.5)` is `-1`. Two of the converted sites are on
values that genuinely go negative and would have been wrong under the cast.

---

## 3. What landed

Six functions, 18 call sites.

| # | site | op | frequency | conversion |
|---|---|---|---|---|
| A1 | [`carmenta/src/image.rs`](../../crates/ffai-carmenta/src/image.rs) `resize_bilinear()` | `floor` ×4 | **per output pixel** | `floor_frac_nonneg` |
| A2 | [`carmenta/src/image.rs`](../../crates/ffai-carmenta/src/image.rs) `resize_bilinear_u8()` | `floor` ×4 | **per output pixel**, ×3 channels per page | `floor_frac_nonneg` |
| A3 | [`carmenta/src/image.rs`](../../crates/ffai-carmenta/src/image.rs) `kernel()` (bicubic) | `floor` ×2 | **per output pixel** | `floor_i32` — **`fy` is not clamped and goes negative at `oy = 0`** |
| A4 | [`carmenta/src/svtr.rs`](../../crates/ffai-carmenta/src/svtr.rs) `svtr_input()` | `floor` ×4 | **per output pixel, per text crop** | `floor_frac_nonneg` |
| A5 | [`diana/src/depth_ops.rs`](../../crates/ffai-diana/src/depth_ops.rs) `bilinear2x_align_corners()` | `floor` ×2 | **per output pixel**, twice per depth forward | `floor_frac_nonneg` |
| A6 | [`diana/src/depth_engine.rs`](../../crates/ffai-diana/src/depth_engine.rs) `to_source_resolution()` | `floor` ×2 | **per output pixel** | `floor_i32` — the `my < 0` guard says the author expected negatives |

A1 and A2 were the *same nine lines* in both functions, which is why one edit
covered both — and is itself worth noting as duplication that had already
drifted into two places.

### The gate it passed

| suite | result | what it covers |
|---|---|---|
| `ffai-core` | **47 pass** (43 + 4 new) | `floor_i32` and `floor_frac_nonneg` against libm over dense sweeps, plus the sign trap |
| `ffai-carmenta` lib | **42 pass** | includes `image::tests::resize_u8_matches_plane_path` — the **byte-identity** assertion between A1 and A2 across all three pixel formats and up/down/identity scaling |
| `ffai-carmenta` oracles | **6 pass** | `mobiledet_probability_map_matches_paddle`, `mobiledet_boxes_match_paddle_postprocess`, `craft_maps_match_pytorch_reference`, `crnn_ids_match_pytorch_reference_exactly`, both PARSeq oracles — all run **through** the converted resize |
| `ffai-diana` | **84 pass** | includes `depth_matches_ultralytics` and `depth_matches_ultralytics_across_tiers`, which run A5 and A6 end to end against the PyTorch reference |
| clippy | **0 errors** on all three CI gate commands | the intentional casts carry `#[allow]` with the reason stated in place |
| rustfmt | `ffai-core` clean | see §8 for pre-existing drift elsewhere that is **not** from this work |

Three external references (Paddle, PyTorch, Ultralytics) agree after the
change. That is a stronger statement than the unit tests alone.

---

## 4. The 71 that were left, and why

Every remaining site has a verdict. None was skipped for lack of time; the
appendix lists all of them individually.

| verdict | n | meaning |
|---|---:|---|
| **COLD** | 35 | once per call / image / crop / load. Real frequency too low to measure, and turbocharger §7 is explicit that a change measuring zero should not be made. |
| **SEMANTIC** | 19 | the rounding *mode* is load-bearing — it reproduces a reference's geometry. Changing ties here is a correctness regression wearing a performance change's clothes. |
| **GATED** | 10 | genuine frequency, but the change alters output. Needs a corpus gate, so each is its own experiment (§5). |
| **HARNESS** | 6 | `ffai-bench`. Changing the instrument's arithmetic changes the instrument. |
| **ORACLE** | 1 | `diana/src/silu.rs` `exp_legacy_round()` is **deliberately** the slow arm; turbocharger's follow-up 2 keeps it as the A/B proof that the magic rounding is bit-identical. Speeding it up destroys the test. |

The largest SEMANTIC group is `diana/src/image.rs` `letterbox_with()` /
`letterbox_shape()`, which reproduce Ultralytics' `auto=True` rule including its
`- 0.1` pad bias. The comment already in the file states the stakes: *"a
one-pixel offset shifts every box by one pixel, which is small enough to pass a
smoke test and large enough to cost mAP."* Its two `floor`s are also already
hoisted out of the pixel loop to O(nw) + O(3·nh) by an earlier campaign.

---

## 5. The three GATED experiments

> **Updated after §7.4.** (1) and (2) were run and **reverted** — the exact
> substitution works and the loop still does not widen, because the barrier
> there is the saturating cast, not the rounding. (3) is untouched.

These are the only remaining sites with real per-element frequency. Each
changes output, so none can be gated byte-identically.

1. **[`mercury/src/asr/vocab_int8.rs`](../../crates/ffai-mercury/src/asr/vocab_int8.rs) `cpu_fwd()`** — `round` per element of `d = 384`, **every decode
   step**. The densest remaining site in the workspace. The magic number
   applies, but it rounds ties-to-even where `f32::round` is ties-away, so an
   exact `.5` selects a different `i8`. **Needs corpus WER on both LibriSpeech
   holdouts**, per [docs/whys/OPEN.md](../whys/OPEN.md) §4's warning that the
   argmax-flip instrument is blind to this class.
2. **`vocab_int8.rs` `new()`** — the same substitution over ~20 M weights, but
   **once at load**. A load-time win only, and it changes the quantised weights,
   so it rides on the same gate as (1).
3. **[`carmenta/src/mobiledet.rs`](../../crates/ffai-carmenta/src/mobiledet.rs) `box_score_fast()` L504-505** — `round` per scanline per box. It decides
   which pixels are summed into a box score, so it moves detection output.
   **Needs corpus CER on both Carmenta holdouts.**

---

## 6. What NOT to expect — and what actually happened

The prediction written here before measuring was:

> The win here is **not the op's own cost — it is that the loop can finally
> widen** [...] Anywhere the loop has a gather or a data-dependent branch,
> removing the `floor` will measure zero, and that result should be recorded
> rather than retried.

**The second sentence is the one that came true.** Recorded below rather than
retried.

---

## 7. Measured: the disassembly A/B

Release build (`lto = "thin"`, `codegen-units = 1`), `--emit=asm`, the same
symbol before (`656889e`) and after (`ce93d34`). Counts are over each
function's own body.

### 7.1 The calls are gone — that part is real

| function | body lines | `callq floorf` | ymm refs | packed arithmetic |
|---|---:|---:|---:|---|
| `carmenta resize_bilinear` | 446 → **395** | 2 → **0** | 0 → 0 | none → none |
| `carmenta resize_bilinear_u8` | 415 → **398** | 2 → **0** | 0 → 0 | none → none |
| `diana bilinear2x_align_corners` | 675 → **609** | 2 → **0** | 0 → 0 | none → none |

`f32::floor` was lowering to a real `callq floorf` — confirming §1's baseline
argument rather than assuming it. Six calls across the three functions are now
zero, and each body lost 4–11 % of its instructions.

### 7.2 But the loops did not widen — and `floor` was never why

Zero `ymm` references before **and** after. The only "packed" mnemonics present
are `movaps`/`movups`/`xorps` — register moves and zeroing, not arithmetic.
Every float operation in these loops is still scalar: `addss`, `mulss`,
`subss`, `divss`, `maxss`.

Reading one layer down, the resize loop has **three** independent blockers and
`floor` was not among them:

1. **The tap loads are gathers.** `movss (%rdx,%r10,4)` and
   `movss (%rdx,%r11,4)` — indexed by runtime registers holding the computed
   `x0`/`x1`, not by the loop counter. A resampler reads where the arithmetic
   says to; that is inherent to the operation, not an accident of how it is
   written.
2. **Six `js` branches from `usize as f32`.** u64→f32 has no single SSE2
   instruction, so each `oy as f32` / `ox as f32` compiles to a sign test and
   two paths.
3. **Five `panic_bounds_check` calls** still in the loop.

Removing the `floor` removed a call and left all three standing. **This is a
recorded zero on the vectorisation claim**, not a step toward it, and the
conversions are kept on the narrower ground that six libm calls left a
per-pixel loop with byte-identical output.

### 7.3 ★ The refutation does NOT transfer to `vocab_int8::cpu_fwd`

This is the finding worth having, and it came out of the refutation rather
than in spite of it. The quantiser's loop disassembles to:

```asm
.LBB2548_23:
    cmpq    %r15, %rdi
    je      .LBB2548_24
    jbe     .LBB2548_52
    movss   (%rbx,%r15,4), %xmm0      ; CONTIGUOUS load, indexed by the counter
    divss   %xmm6, %xmm0
    callq   roundf                    ; <- the only thing that cannot widen
    movaps  %xmm7, %xmm1
    maxss   %xmm0, %xmm1              ; clamp lo
    movaps  %xmm8, %xmm0
    minss   %xmm1, %xmm0              ; clamp hi
    cvttss2si %xmm0, %eax
    ...
    movb    %al, (%r14,%r15)          ; CONTIGUOUS store
```

Unit-stride in, unit-stride out, and every operation in the body has a packed
SSE2 twin — `divps`, `maxps`, `minps`, `cvttps2dq`. **`callq roundf` is the
sole barrier**, which is exactly the shape the resize loops turned out not to
have. Only two `roundf` calls exist in all of `ffai-mercury`, and they are
these two sites.

**And it may not need the WER gate after all.** §5 assumed the magic number,
which is ties-to-even and therefore an output change. But `f32::round` is
ties-away-from-zero, and ties-away has its own branchless packed-SSE2 form:
`trunc(|x| + 0.5)` via `cvttps2dq`, with one compare-and-subtract correcting
the single case where `|x| + 0.5` rounds up across the boundary (the classic
`0.49999997` trap). If that lands exact, the gate is `cmp`, not a corpus run.
That is the next experiment, and it is a better one than §5 described.

---

### 7.4 `vocab_int8` — the call came out exactly, and it STILL did not widen

§7.3's plan was run. The exact-ties-away primitive works; the conclusion it
was built to support does not.

**[`round_ties_away_i32`](../../crates/ffai-core/src/fastmath.rs) is exact.**
`trunc(|x| + 0.5)` with one correction, swept against `f32::round` at
one-ulp resolution around every half-boundary in +/-200, densely across the
quantiser's range, and at the edges. Two real bugs surfaced while building
it, and both are now pinned by tests:

1. **The obvious correction cancels.** `(t as f32) - a > 0.5` looks right and
   is wrong: at `a = 0.499_999_97` the true difference is `0.500_000_03`,
   which is exactly halfway between two f32 and **rounds to 0.5** — so `> 0.5`
   is false, the correction never fires, and the function returns 1 where
   `round` returns 0. Reformulated as `(t as f32) - 0.5 > a`, which does not
   subtract two nearly-equal quantities. This is the same shape as the
   `1 - 2/(e^2x + 1)` trap in `rusty-fast-transcendentals` §4.
2. **Negating the integer is wrong at the rail.** `-i32::MAX` is
   `i32::MIN + 1`, but `(-inf).round() as i32` is `i32::MIN`. The sign is now
   applied in float, before a single saturating cast.

**Wired into both `vocab_int8` sites, 136 mercury tests pass and `callq
roundf` across the whole crate goes 2 → 0.** So the barrier named in §7.3 was
removed, exactly as intended.

**And the loop still did not widen.**

| `cpu_fwd` | before | after |
|---|---:|---:|
| body lines | 345 | **371** |
| scalar float ops | 19 | **26** |
| ymm refs | 0 | 0 |
| packed arithmetic | none | none |
| `callq roundf` (crate) | 2 | **0** |

One layer down again, and it is a mechanism neither §7.3 nor the skill
anticipated: **Rust's `as i32` on a float is *saturating*, and LLVM emits two
data-dependent branches for it** — `ucomiss` + `ja` for the range rail and
`ucomiss` + `jp` for NaN. The original had one such cast; the exact
formulation needs **two**. So the change trades one libm call for ~26
instructions and two extra branches, in a loop that vectorises neither way.

**Reverted at the call sites.** "Revert if not faster" is the rule, and this
is not demonstrably faster — it is a coin flip that adds instructions. The
primitive stays in `fastmath`: it is proven, its tests document two traps that
actually bit, and it becomes immediately useful the moment the cast question
below is answered.

**The real barrier, now named.** In this loop the blocker was never `round` —
it is the saturating float→int cast. Removing it needs either
`f32::to_int_unchecked` behind a proven range, or a clamp expressed so LLVM
can prove the rails dead. The first is `unsafe`, and this workspace sets
`unsafe_code = "warn"` at the root, so that is an owner's decision with a
`SAFETY` invariant to write — not a drive-by. **That, not another rounding
substitution, is the next experiment here.**

---

## 8. One thing found on the way that is not ours

`cargo fmt --all --check` reports **212 diffs in `ffai-argus` and 14 in
`ffai-bench`**, all of them in `examples/`, and all of them import-ordering
(`{render, run_ocr, BenchConfig}` → `{BenchConfig, render, run_ocr}`). That is
the **style-edition 2024** sort order. There is no `rustfmt.toml`, the workspace
is `edition = "2024"`, and `rust-toolchain.toml` pins 1.95.0 — so a plain
`cargo fmt` here disagrees with what is committed.

It predates this work (verified by formatting each touched file's pristine
`HEAD` copy: `image.rs` 20 diffs before and after, `svtr.rs` 13/13,
`depth_ops.rs` 5/5, `depth_engine.rs` 7/7 — unchanged). But CI has a Format
gate, so either that gate is currently red on this branch or it runs a
different toolchain, and it is worth knowing which.

---

## Appendix — every one of the 89, with its verdict

Generated 2026-09-19. `FIXED` rows are this change; the rest carry the reason
they were left. Rounding ops only — the 28 slow-transcendental and 23 candle
activation sites are the first campaign's closed territory and are not
relisted.

### FIXED — 18 sites, 6 functions

| crate | site | op | how |
|---|---|---|---|
| `ffai-carmenta` | [`image.rs`](../../crates/ffai-carmenta/src/image.rs) `resize_bilinear()` | `floor`×4 | `floor_frac_nonneg` |
| `ffai-carmenta` | [`image.rs`](../../crates/ffai-carmenta/src/image.rs) `resize_bilinear_u8()` | `floor`×4 | `floor_frac_nonneg` |
| `ffai-carmenta` | [`image.rs`](../../crates/ffai-carmenta/src/image.rs) `kernel()` | `floor`×2 | `floor_i32` (signed) |
| `ffai-carmenta` | [`svtr.rs`](../../crates/ffai-carmenta/src/svtr.rs) `svtr_input()` | `floor`×4 | `floor_frac_nonneg` |
| `ffai-diana` | [`depth_ops.rs`](../../crates/ffai-diana/src/depth_ops.rs) `bilinear2x_align_corners()` | `floor`×2 | `floor_frac_nonneg` |
| `ffai-diana` | [`depth_engine.rs`](../../crates/ffai-diana/src/depth_engine.rs) `to_source_resolution()` | `floor`×2 | `floor_i32` (signed) |

### GATED — 10 sites

| crate | site | ops | lines | why |
|---|---|---|---|---|
| `ffai-carmenta` | [`mobiledet.rs`](../../crates/ffai-carmenta/src/mobiledet.rs) `box_score_fast()` | `floor`×2 `ceil`×2 `trunc`×2 `round`×2 | 466,467,468,469,475,504,505 | L466-475 are per box. L504/505 ARE per scanline, but round decides which pixels are summed into the box score, so it changes detection output: needs the corpus CER gate, not a byte-identical one. |
| `ffai-mercury` | [`vocab_int8.rs`](../../crates/ffai-mercury/src/asr/vocab_int8.rs) `new()` | `round`×1 | 151 | vocab_int8: ~20 M rounds but ONCE at load. A load-time win only, and it changes the quantised weights, so it needs the WER gate. |
| `ffai-mercury` | [`vocab_int8.rs`](../../crates/ffai-mercury/src/asr/vocab_int8.rs) `cpu_fwd()` | `round`×1 | 232 | vocab_int8: per element of d=384, EVERY decode step - the densest remaining site. But round() is ties-away and the magic number is ties-even, so an exact .5 selects a different i8. Bitstream change: needs corpus WER on both holdouts. |

### SEMANTIC — 19 sites

| crate | site | ops | lines | why |
|---|---|---|---|---|
| `ffai-carmenta` | [`engine.rs`](../../crates/ffai-carmenta/src/engine.rs) `recognize()` | `round`×5 | 630,631,632,633,684 | round maps box coords to source pixels; a tie flip moves a box edge one pixel. Per box, not per pixel. |
| `ffai-carmenta` | [`mobiledet.rs`](../../crates/ffai-carmenta/src/mobiledet.rs) `boxes_from_probability()` | `round`×6 | 605,606,607 | per box; round sets the emitted box geometry |
| `ffai-diana` | [`image.rs`](../../crates/ffai-diana/src/image.rs) `letterbox_shape()` | `round`×2 | 54,55 | reproduces Ultralytics rounding; once per image |
| `ffai-diana` | [`image.rs`](../../crates/ffai-diana/src/image.rs) `letterbox_with()` | `round`×4 `floor`×2 | 79,80,94,95,113,136 | reproduces Ultralytics `auto=True` including the -0.1 pad bias. The in-place comment says a one-pixel offset costs mAP. The two floors are already hoisted out of the pixel loop to O(nw)+O(3*nh). |

### ORACLE — 1 sites

| crate | site | ops | lines | why |
|---|---|---|---|---|
| `ffai-diana` | [`silu.rs`](../../crates/ffai-diana/src/silu.rs) `exp_legacy_round()` | `round`×1 | 75 | DELIBERATELY the slow arm. turbocharger follow-up 2 keeps it as the A/B oracle proving the magic rounding is bit-identical. Speeding it up destroys the test. |

### COLD — 35 sites

| crate | site | ops | lines | why |
|---|---|---|---|---|
| `ffai-argus` | [`preprocess.rs`](../../crates/ffai-argus/src/preprocess.rs) `build_taps()` | `floor`×2 `ceil`×1 | 80,88,89 | tap precompute, O(out_n) once per image; preprocessing is a measured 0.4 % of a caption. Also floors BEFORE .max(0.0), so a cast would be wrong. |
| `ffai-argus` | [`preprocess.rs`](../../crates/ffai-argus/src/preprocess.rs) `fit_longest_edge()` | `round`×2 | 197,198 | once per image |
| `ffai-argus` | [`preprocess.rs`](../../crates/ffai-argus/src/preprocess.rs) `build_fixed_taps()` | `ceil`×1 | 248 | once per image |
| `ffai-carmenta` | [`craft.rs`](../../crates/ffai-carmenta/src/craft.rs) `upsample_bilinear()` | `floor`×2 | 243,245 | per-axis precompute, O(out_h+out_w), reused across channels by design |
| `ffai-carmenta` | [`formula.rs`](../../crates/ffai-carmenta/src/formula.rs) `input()` | `round`×2 | 150,151 | once per formula crop |
| `ffai-carmenta` | [`image.rs`](../../crates/ffai-carmenta/src/image.rs) `crnn_input_patch()` | `round`×1 | 349 | once per patch |
| `ffai-carmenta` | [`image.rs`](../../crates/ffai-carmenta/src/image.rs) `mobiledet_input()` | `trunc`×4 `round`×1 | 667,670,671,673 | once per page |
| `ffai-carmenta` | [`live.rs`](../../crates/ffai-carmenta/src/live.rs) `percentile()` | `round`×1 | 113 | once per call |
| `ffai-carmenta` | [`live.rs`](../../crates/ffai-carmenta/src/live.rs) `band_pad()` | `round`×1 | 178 | once per band |
| `ffai-carmenta` | [`live.rs`](../../crates/ffai-carmenta/src/live.rs) `srt_time()` | `round`×1 | 512 | once per subtitle cue |
| `ffai-carmenta` | [`mobiledet.rs`](../../crates/ffai-carmenta/src/mobiledet.rs) `split_internal_gutters()` | `round`×1 | 688 | once per page |
| `ffai-carmenta` | [`onnx_graph.rs`](../../crates/ffai-carmenta/src/onnx_graph.rs) `exec_in()` | `round`×2 | 614,615 | once per Resize node in the ONNX graph |
| `ffai-carmenta` | [`svtr.rs`](../../crates/ffai-carmenta/src/svtr.rs) `svtr_input()` | `ceil`×1 | 414 | the remaining ceil sets the output width, once per crop |
| `ffai-core` | [`types.rs`](../../crates/ffai-core/src/types.rs) `vtt_time()` | `round`×1 | 282 | once per subtitle cue |
| `ffai-core` | [`types.rs`](../../crates/ffai-core/src/types.rs) `srt_time()` | `round`×1 | 296 | once per subtitle cue |
| `ffai-diana` | [`config.rs`](../../crates/ffai-diana/src/config.rs) `ch()` | `ceil`×1 | 229 | model config, once at load |
| `ffai-diana` | [`config.rs`](../../crates/ffai-diana/src/config.rs) `rep()` | `round`×1 | 239 | model config, once at load |
| `ffai-mercury` | [`aligner.rs`](../../crates/ffai-mercury/src/asr/aligner.rs) `align_segments()` | `ceil`×1 | 162 | once per segment |
| `ffai-mercury` | [`decoder.rs`](../../crates/ffai-mercury/src/asr/decoder.rs) `apply_logit_filters()` | `floor`×1 | 973 | once per decode |
| `ffai-mercury` | [`diarize.rs`](../../crates/ffai-mercury/src/asr/diarize.rs) `subsegment_at()` | `ceil`×1 | 155 | once per subsegment |
| `ffai-mercury` | [`diarizer.rs`](../../crates/ffai-mercury/src/asr/diarizer.rs) `diarize_incremental()` | `ceil`×1 | 301 | once per window |
| `ffai-mercury` | [`diarizer.rs`](../../crates/ffai-mercury/src/asr/diarizer.rs) `embed_windows()` | `ceil`×1 | 410 | once per window |
| `ffai-mercury` | [`vad.rs`](../../crates/ffai-mercury/src/asr/vad.rs) `percentile()` | `round`×1 | 125 | once per call |
| `ffai-mercury` | [`whisper_candle.rs`](../../crates/ffai-mercury/src/asr/whisper_candle.rs) `adaptive_ctx_secs()` | `ceil`×1 | 834 | once per window |
| `ffai-mercury` | [`vits.rs`](../../crates/ffai-mercury/src/tts/vits.rs) `durations()` | `ceil`×1 | 609 | once per phoneme, O(phonemes) |

### HARNESS — 6 sites

| crate | site | ops | lines | why |
|---|---|---|---|---|
| `ffai-bench` | [`resample.rs`](../../crates/ffai-bench/src/resample.rs) `resample()` | `floor`×2 | 50,56 | the measurement instrument; changing its arithmetic changes the instrument |
| `ffai-bench` | [`runner.rs`](../../crates/ffai-bench/src/runner.rs) `run_detect_reference()` | `round`×1 | 523 | the measurement instrument; changing its arithmetic changes the instrument |
| `ffai-bench` | [`runner.rs`](../../crates/ffai-bench/src/runner.rs) `run_detect_engine()` | `round`×1 | 832 | the measurement instrument; changing its arithmetic changes the instrument |
| `ffai-bench` | [`runner.rs`](../../crates/ffai-bench/src/runner.rs) `run_reference()` | `round`×1 | 1129 | the measurement instrument; changing its arithmetic changes the instrument |
| `ffai-bench` | [`tts.rs`](../../crates/ffai-bench/src/tts.rs) `run_tts_reference()` | `round`×1 | 279 | the measurement instrument; changing its arithmetic changes the instrument |

**Total: 18 FIXED + 71 recorded = 89.**

