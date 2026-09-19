# Turbocharger 2 — the rounding class

**Status:** census, 2026-09-19. **Nothing here is measured yet.** This page is
an inventory and a ranked set of hypotheses; every entry needs the gate in §6
before it becomes a claim.

The first turbocharger campaign ([docs/finished/turbocharger.md](../finished/turbocharger.md))
is closed and it closed properly: Tier 1 took **all 17 candle activation
sites**, Tier 2 took **every hot scalar `exp`/`log` loop**, and Tier 3 recorded
the one-off setup arithmetic that must not be touched. `ffai-core`'s
[`fastmath`](../../crates/ffai-core/src/fastmath.rs) and
[`fastops`](../../crates/ffai-core/src/fastops.rs) are the consolidated kernel
and its zero-copy delivery.

**What that campaign never targeted is the rounding class.** Its §4 states the
rule — *"`round()` is SSE4.1, not SSE2"* — but every target in its three tiers
was an `exp`, a `log` or an activation. `floor`, `ceil`, `trunc` and `round`
were left where they sat. There are **89 of them on shipping paths**, and a
handful are in the innermost loop of an image resize.

---

## 1. Why this bites in *this* workspace

The rule is not universal — it depends on the compile baseline, so it has to be
checked per repo. Here it holds:

`.cargo/config.toml` sets rustflags for **`wasm32-unknown-unknown`** (`+simd128`)
and for **Linux/GNU linking** (RELRO, noexecstack, frame pointers). It sets **no
`target-cpu` and no `target-feature` for native builds**, and there is none in
`Cargo.toml`, `rust-toolchain.toml` or the CI workflows. So the native x86-64
baseline is the default one:

| op | instruction | at this baseline |
|---|---|---|
| `floor` / `ceil` / `trunc` | `vroundps` / `roundss` | **SSE4.1 — above baseline.** Lowers to a call or a branchy sequence. |
| `round` | *none* | Rust's `f32::round` is ties-**away-from-zero**; `vroundps` is ties-to-**even**. No x86 instruction implements it at any baseline. |
| `sqrt`, `min`, `max`, `abs`, FMA | SSE2 / FMA3 | Already vectorised. **Do not touch.** |

Two qualifications, both load-bearing:

* **This is a native-only win.** wasm32 has `f32.floor` as a real instruction
  and `f32x4.floor` under `+simd128`, which we already enable. The wasm modules
  do not pay this tax; the native `ffai` binary does.
* **Inside an existing `#[target_feature(enable = "avx2")]` kernel, `floor` is
  free** — the attribute lifts the baseline for that function. None of the Tier
  A sites below are inside such a function; they are plain safe Rust.

---

## 2. The fix is usually *deletion*, not the magic number

This is the important difference from the first campaign, and it is why Tier A
is cheap.

`rusty-fast-transcendentals` §2 offers the magic-number round
(`(x + 1.5*2^23) - 1.5*2^23`). That is the right tool when you need
round-to-nearest. **Most sites here do not.** They compute a *floor for a
non-negative value*, and Rust's `as usize` / `as u32` cast already truncates
toward zero — which **is** floor when the value cannot be negative:

```rust
// shipped (carmenta/src/image.rs:44-51) — two floor() per output pixel
let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);   // <- provably >= 0
let x0 = (fx.floor() as usize).min(sw - 1);
let wx = fx - fx.floor();

// candidate — no libm, no SSE4.1, and the loop can widen
let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
let fx_i = fx as u32;                  // truncation == floor for fx >= 0
let x0 = (fx_i as usize).min(sw - 1);
let wx = fx - fx_i as f32;
```

That is turbocharger §5's *"better than accelerating: deleting"* applied to a
different op. **The `.max(0.0)` is the whole licence** — it is what makes the
cast exact, and it must be re-read at each site rather than assumed.

Where a value **can** go negative (`diana/src/depth_engine.rs`, which explicitly
tests `my < 0`), the cast is *not* floor and this shortcut is wrong. Those sites
need a branchless floor or the magic number, and they carry more risk for less
reward — they are Tier B for that reason.

---

## 3. Tier A — verified per-element, in the inner loop

Each of these was read, not inferred. "per output pixel" means the call sits
inside the innermost loop over `ox`.

| # | site | op | frequency | non-negative? |
|---|---|---|---|---|
| A1 | [`carmenta/src/image.rs:49,51`](../../crates/ffai-carmenta/src/image.rs#L49) `resize_bilinear()` | `floor` ×2 | **per output pixel** | yes — `.max(0.0)` |
| A2 | [`carmenta/src/image.rs:94,96`](../../crates/ffai-carmenta/src/image.rs#L94) `resize_bilinear_u8()` | `floor` ×2 | **per output pixel** | yes — `.max(0.0)` |
| A3 | [`carmenta/src/image.rs:131,135`](../../crates/ffai-carmenta/src/image.rs#L131) `kernel()` (bicubic) | `floor` ×2 | **per output pixel** | check at site |
| A4 | [`carmenta/src/svtr.rs:424,428`](../../crates/ffai-carmenta/src/svtr.rs#L424) `svtr_input()` | `floor` ×4 | **per output pixel, per text crop** | yes — `.max(0.0)` |
| A5 | [`diana/src/depth_ops.rs:50,55`](../../crates/ffai-diana/src/depth_ops.rs#L50) `bilinear2x_align_corners()` | `floor` ×2 | **per output pixel**, twice per depth forward | yes — `oy as f32 * sy`, `sy >= 0` |
| A6 | [`mercury/src/asr/vocab_int8.rs:232`](../../crates/ffai-mercury/src/asr/vocab_int8.rs#L232) `cpu_fwd()` | `round` ×1 | **per element of d (384), every decode step** | **no — signed** |

**A1–A5 are the same edit five times** and should be one brick each, gated
byte-identical. A1/A2 are the hottest thing on this list: `mobiledet_input`
calls them three times per page (once per channel), and CHANGELOG 0.10.0 has
just reworked exactly this code, so the context is loaded.

**A6 is different and must not be batched with them.** It is a real `round`
(ties-away-from-zero) inside the int8 activation quantizer, on the decode hot
path. The magic number is the right tool, but it rounds ties-to-**even**, so on
an exact `.5` it selects a different `i8`. That is a *bitstream change*, not a
speed change — it needs corpus WER on both holdouts, not a byte-identical
assertion. Treat it as its own experiment.

---

## 4. Tier B — real but smaller, or riskier

| site | op | why it is lower |
|---|---|---|
| [`carmenta/src/mobiledet.rs:466-475`](../../crates/ffai-carmenta/src/mobiledet.rs#L466) `box_score_fast()` | `floor`×2 `ceil`×2 `trunc`×2 | per **box**, not per pixel — O(boxes), and the scanline loop below it is the real cost |
| [`carmenta/src/mobiledet.rs`](../../crates/ffai-carmenta/src/mobiledet.rs) `boxes_from_probability()` | `round` ×6 | per box |
| [`carmenta/src/engine.rs`](../../crates/ffai-carmenta/src/engine.rs) `recognize()` | `round` ×5 | per crop |
| [`diana/src/depth_engine.rs:217,223`](../../crates/ffai-diana/src/depth_engine.rs#L217) `to_source_resolution()` | `floor` ×2 | per pixel, **but the value can be negative** — the `my < 0` guard proves it. Needs a real floor, not a cast. |
| [`mercury/src/asr/vocab_int8.rs:151`](../../crates/ffai-mercury/src/asr/vocab_int8.rs#L151) `new()` | `round` | ~20 M rounds (51 865 × 384) but **once, at load** — a load-time win only |
| [`argus/src/preprocess.rs:88,89`](../../crates/ffai-argus/src/preprocess.rs#L88) `build_taps()` | `floor` ×2 | tap precompute, once per image. turbocharger measured preprocessing at **54 ms of a 14 561 ms caption (0.4 %)** — bounded above by that. |

---

## 5. Tier C — do NOT touch (recorded so nobody spends a day here)

* **[`diana/src/image.rs:79-95`](../../crates/ffai-diana/src/image.rs#L79) `letterbox_with()`** — `round` ×4, computed **once per image**, and each one
  deliberately reproduces Ultralytics' own rule, including the `- 0.1` bias.
  The comment in place says it: *"a one-pixel offset shifts every box by one
  pixel, which is small enough to pass a smoke test and large enough to cost
  mAP."* Changing the rounding mode here is a correctness regression wearing a
  performance change's clothes.
* **`diana/src/head.rs:278` `decode()`** — the `exp` is already deferred to the
  survivors of top-k. The comment says so and it is right.
* **`diana/src/depth_head.rs:144,155`** — candle tensor `exp`, at the head only.
  Already Tier 3 of the first campaign.
* **`mercury/src/asr/mel.rs`, `fbank.rs` setup** (`hz_to_mel`, `mel_to_hz`,
  `hamming`, `new`) — built once per filterbank. Already Tier 3.
* **`core/src/types.rs` `srt_time`/`vtt_time`, `carmenta/src/live.rs` `srt_time`** —
  subtitle formatting, once per cue.
* **`diana/src/config.rs` `ch()`/`rep()`** — model config, once at load.
* **Everything under `ffai-bench/`** — the harness, not a shipping path.
  Changing its arithmetic changes the instrument.
* **`sqrt`, `min`, `max`, `abs`** anywhere — SSE2 baseline, already vectorised.

---

## 6. The gate

Tier A1–A5 are **integer-index computations**, so the bar is the strict one:

1. **Byte-identical output.** A floor that moves one index by one moves a pixel.
   `cmp`-proven against the pre-change binary on a real corpus page, not a
   tolerance.
2. **The existing oracles must stay green** — `resize_u8_matches_plane_path`
   already asserts A2 against A1 across all three pixel formats and up/down/
   identity scaling, which makes it the cheapest possible regression net.
   `mobiledet_matches_plane_path` covers the same seam.
3. **Corpus CER on both Carmenta holdouts** (`cord`, `doc`) for A1–A4, because
   these feed detection geometry. Unchanged, not "close".
4. **A6 does not get this gate** — it changes the quantizer. It needs WER on
   both LibriSpeech holdouts, per `docs/whys/OPEN.md` §4's warning that the
   argmax-flip instrument is blind to this class.

Measurement discipline is `codec-measurement`: ABBA-interleaved, best-of-N,
pinned, with a null arm. **The counter to read first is not a duration** — it is
whether the loop vectorised. Check the emitted `.s` for packed ops before and
after; if the loop did not widen, the change bought nothing and should revert
regardless of what the clock says.

---

## 7. What NOT to expect

The first campaign's §7 applies unchanged, plus one specific to this class:

**A `floor` is not an `exp`.** The activation campaign's headline was 32x
because candle was doing a scalar `erf` per element. A `floor` at this baseline
is a handful of instructions, not forty nanoseconds. The win here is **not the
op's own cost — it is that the loop can finally widen**, and that is worth
something only where the surrounding loop is otherwise vectorisable arithmetic.
A1–A5 are (multiply, add, four loads, three FMA); A6's quantize loop is
(divide, round, clamp, cast) and should widen well. Anywhere the loop has a
gather or a data-dependent branch, removing the `floor` will measure zero, and
that result should be recorded rather than retried.

---

## Appendix — the complete census

Every transcendental and rounding call on a shipping path (`crates/*/src/`,
excluding `#[cfg(test)]` modules), generated 2026-09-19 at `30e18d7`.

**140 sites in 81 functions**: 89 rounding (`round` 42, `floor` 29, `ceil` 12,
`trunc` 6), 28 slow-transcendental, 23 candle activations.

The 23 `act` and 28 `slow` rows are the first campaign's closed territory and
are listed for completeness — they are either already on `fastmath`/`fastops`
(and so excluded by the scanner), or recorded as Tier 3 setup arithmetic. **The
new work is the `round` column.**

`LP` is loop-nesting depth above the call site; `LP0` in a function whose body
is one expression usually means "once per call".

### `ffai-argus` — 16 sites

| LP | file | function | ops | class |
|---:|---|---|---|---|
| 0 | [`src/preprocess.rs:62`](../../crates/ffai-argus/src/preprocess.rs#L62) | `lanczos3()` | `sin`×2 | slow |
| 1 | [`src/preprocess.rs:80`](../../crates/ffai-argus/src/preprocess.rs#L80) | `build_taps()` | `floor`×2 `ceil`×1 | round |
| 0 | [`src/preprocess.rs:197`](../../crates/ffai-argus/src/preprocess.rs#L197) | `fit_longest_edge()` | `round`×2 | round |
| 0 | [`src/preprocess.rs:248`](../../crates/ffai-argus/src/preprocess.rs#L248) | `build_fixed_taps()` | `ceil`×1 | round |
| 1 | [`src/preprocess.rs:299`](../../crates/ffai-argus/src/preprocess.rs#L299) | `lanczos3_f64()` | `sin`×2 | slow |
| 0 | [`src/siglip.rs:513`](../../crates/ffai-argus/src/siglip.rs#L513) | `forward()` | `softmax`×2 | act |
| 1 | [`src/siglip.rs:1491`](../../crates/ffai-argus/src/siglip.rs#L1491) | `inplace_softmax()` | `softmax`×1 | act |
| 1 | [`src/siglip.rs:1496`](../../crates/ffai-argus/src/siglip.rs#L1496) | `set_inplace_softmax()` | `softmax`×1 | act |
| 0 | [`src/text.rs:1044`](../../crates/ffai-argus/src/text.rs#L1044) | `rope_tables()` | `cos`×1 `sin`×1 | slow |

### `ffai-bench` — 8 sites

| LP | file | function | ops | class |
|---:|---|---|---|---|
| 1 | [`src/resample.rs:50`](../../crates/ffai-bench/src/resample.rs#L50) | `resample()` | `floor`×2 | round |
| 0 | [`src/resample.rs:86`](../../crates/ffai-bench/src/resample.rs#L86) | `windowed_sinc()` | `sin`×1 `cos`×1 | slow |
| 0 | [`src/runner.rs:523`](../../crates/ffai-bench/src/runner.rs#L523) | `run_detect_reference()` | `round`×1 | round |
| 1 | [`src/runner.rs:832`](../../crates/ffai-bench/src/runner.rs#L832) | `run_detect_engine()` | `round`×1 | round |
| 0 | [`src/runner.rs:1129`](../../crates/ffai-bench/src/runner.rs#L1129) | `run_reference()` | `round`×1 | round |
| 0 | [`src/tts.rs:279`](../../crates/ffai-bench/src/tts.rs#L279) | `run_tts_reference()` | `round`×1 | round |

### `ffai-carmenta` — 55 sites

| LP | file | function | ops | class |
|---:|---|---|---|---|
| 0 | [`src/craft.rs:243`](../../crates/ffai-carmenta/src/craft.rs#L243) | `upsample_bilinear()` | `floor`×2 | round |
| 0 | [`src/crnn.rs:226`](../../crates/ffai-carmenta/src/crnn.rs#L226) | `ctc_greedy_with()` | `softmax`×1 | act |
| 0 | [`src/engine.rs:630`](../../crates/ffai-carmenta/src/engine.rs#L630) | `recognize()` | `round`×5 | round |
| 1 | [`src/formula.rs:150`](../../crates/ffai-carmenta/src/formula.rs#L150) | `input()` | `round`×2 | round |
| 2 | [`src/image.rs:44`](../../crates/ffai-carmenta/src/image.rs#L44) | `resize_bilinear()` | `floor`×4 | round |
| 2 | [`src/image.rs:89`](../../crates/ffai-carmenta/src/image.rs#L89) | `resize_bilinear_u8()` | `floor`×4 | round |
| 2 | [`src/image.rs:131`](../../crates/ffai-carmenta/src/image.rs#L131) | `kernel()` | `floor`×2 | round |
| 0 | [`src/image.rs:349`](../../crates/ffai-carmenta/src/image.rs#L349) | `crnn_input_patch()` | `round`×1 | round |
| 0 | [`src/image.rs:667`](../../crates/ffai-carmenta/src/image.rs#L667) | `mobiledet_input()` | `trunc`×4 `round`×1 | round |
| 0 | [`src/live.rs:113`](../../crates/ffai-carmenta/src/live.rs#L113) | `percentile()` | `round`×1 | round |
| 0 | [`src/live.rs:178`](../../crates/ffai-carmenta/src/live.rs#L178) | `band_pad()` | `round`×1 | round |
| 1 | [`src/live.rs:512`](../../crates/ffai-carmenta/src/live.rs#L512) | `srt_time()` | `round`×1 | round |
| 2 | [`src/mobiledet.rs:466`](../../crates/ffai-carmenta/src/mobiledet.rs#L466) | `box_score_fast()` | `floor`×2 `ceil`×2 `trunc`×2 `round`×2 | round |
| 0 | [`src/mobiledet.rs:605`](../../crates/ffai-carmenta/src/mobiledet.rs#L605) | `boxes_from_probability()` | `round`×6 | round |
| 0 | [`src/mobiledet.rs:688`](../../crates/ffai-carmenta/src/mobiledet.rs#L688) | `split_internal_gutters()` | `round`×1 | round |
| 1 | [`src/onnx_graph.rs:517`](../../crates/ffai-carmenta/src/onnx_graph.rs#L517) | `exec_in()` | `round`×2 `softmax`×1 | act,round |
| 0 | [`src/parseq.rs:32`](../../crates/ffai-carmenta/src/parseq.rs#L32) | `softmax_last()` | `softmax`×1 | act |
| 0 | [`src/svtr.rs:318`](../../crates/ffai-carmenta/src/svtr.rs#L318) | `attention()` | `softmax`×1 | act |
| 0 | [`src/svtr.rs:382`](../../crates/ffai-carmenta/src/svtr.rs#L382) | `forward()` | `softmax`×1 | act |
| 2 | [`src/svtr.rs:414`](../../crates/ffai-carmenta/src/svtr.rs#L414) | `svtr_input()` | `floor`×4 `ceil`×1 | round |

### `ffai-core` — 7 sites

| LP | file | function | ops | class |
|---:|---|---|---|---|
| 0 | [`src/fastops.rs:133`](../../crates/ffai-core/src/fastops.rs#L133) | `cpu_fwd()` | `gelu_erf`×1 `gelu`×1 `tanh`×1 `silu`×1 `erf`×1 | act,slow |
| 1 | [`src/types.rs:282`](../../crates/ffai-core/src/types.rs#L282) | `vtt_time()` | `round`×1 | round |
| 0 | [`src/types.rs:296`](../../crates/ffai-core/src/types.rs#L296) | `srt_time()` | `round`×1 | round |

### `ffai-diana` — 19 sites

| LP | file | function | ops | class |
|---:|---|---|---|---|
| 0 | [`src/blocks.rs:528`](../../crates/ffai-diana/src/blocks.rs#L528) | `forward_inner()` | `softmax`×1 | act |
| 0 | [`src/config.rs:229`](../../crates/ffai-diana/src/config.rs#L229) | `ch()` | `ceil`×1 | round |
| 0 | [`src/config.rs:239`](../../crates/ffai-diana/src/config.rs#L239) | `rep()` | `round`×1 | round |
| 2 | [`src/depth_engine.rs:217`](../../crates/ffai-diana/src/depth_engine.rs#L217) | `to_source_resolution()` | `floor`×2 | round |
| 2 | [`src/depth_head.rs:144`](../../crates/ffai-diana/src/depth_head.rs#L144) | `forward()` | `exp`×2 | slow |
| 3 | [`src/depth_ops.rs:50`](../../crates/ffai-diana/src/depth_ops.rs#L50) | `bilinear2x_align_corners()` | `floor`×2 | round |
| 2 | [`src/head.rs:278`](../../crates/ffai-diana/src/head.rs#L278) | `decode()` | `exp`×1 | slow |
| 0 | [`src/image.rs:54`](../../crates/ffai-diana/src/image.rs#L54) | `letterbox_shape()` | `round`×2 | round |
| 0 | [`src/image.rs:79`](../../crates/ffai-diana/src/image.rs#L79) | `letterbox_with()` | `round`×4 `floor`×2 | round |
| 0 | [`src/silu.rs:75`](../../crates/ffai-diana/src/silu.rs#L75) | `exp_legacy_round()` | `round`×1 | round |

### `ffai-mercury` — 35 sites

| LP | file | function | ops | class |
|---:|---|---|---|---|
| 0 | [`src/asr/adaptive.rs:199`](../../crates/ffai-mercury/src/asr/adaptive.rs#L199) | `attention_kv_dtype()` | `softmax`×1 | act |
| 1 | [`src/asr/align.rs:332`](../../crates/ffai-mercury/src/asr/align.rs#L332) | `forced_align()` | `exp`×1 | slow |
| 1 | [`src/asr/aligner.rs:162`](../../crates/ffai-mercury/src/asr/aligner.rs#L162) | `align_segments()` | `ceil`×1 | round |
| 0 | [`src/asr/decoder.rs:973`](../../crates/ffai-mercury/src/asr/decoder.rs#L973) | `apply_logit_filters()` | `floor`×1 | round |
| 0 | [`src/asr/diarize.rs:155`](../../crates/ffai-mercury/src/asr/diarize.rs#L155) | `subsegment_at()` | `ceil`×1 | round |
| 1 | [`src/asr/diarizer.rs:301`](../../crates/ffai-mercury/src/asr/diarizer.rs#L301) | `diarize_incremental()` | `ceil`×1 | round |
| 1 | [`src/asr/diarizer.rs:410`](../../crates/ffai-mercury/src/asr/diarizer.rs#L410) | `embed_windows()` | `ceil`×1 | round |
| 0 | [`src/asr/fbank.rs:42`](../../crates/ffai-mercury/src/asr/fbank.rs#L42) | `to_mel()` | `log10`×1 | slow |
| 0 | [`src/asr/fbank.rs:58`](../../crates/ffai-mercury/src/asr/fbank.rs#L58) | `hamming()` | `cos`×1 | slow |
| 0 | [`src/asr/mel.rs:48`](../../crates/ffai-mercury/src/asr/mel.rs#L48) | `hz_to_mel()` | `ln`×2 | slow |
| 0 | [`src/asr/mel.rs:59`](../../crates/ffai-mercury/src/asr/mel.rs#L59) | `mel_to_hz()` | `ln`×1 `exp`×1 | slow |
| 0 | [`src/asr/mel.rs:127`](../../crates/ffai-mercury/src/asr/mel.rs#L127) | `new()` | `cos`×1 | slow |
| 0 | [`src/asr/speaker.rs:333`](../../crates/ffai-mercury/src/asr/speaker.rs#L333) | `forward()` | `softmax`×1 | act |
| 1 | [`src/asr/text_decoder.rs:510`](../../crates/ffai-mercury/src/asr/text_decoder.rs#L510) | `fast_softmax()` | `softmax`×1 | act |
| 0 | [`src/asr/text_decoder.rs:819`](../../crates/ffai-mercury/src/asr/text_decoder.rs#L819) | `forward_cross()` | `softmax`×1 | act |
| 0 | [`src/asr/text_decoder.rs:875`](../../crates/ffai-mercury/src/asr/text_decoder.rs#L875) | `attend_prepared()` | `softmax`×1 | act |
| 0 | [`src/asr/text_decoder.rs:1139`](../../crates/ffai-mercury/src/asr/text_decoder.rs#L1139) | `forward()` | `softmax`×1 | act |
| 0 | [`src/asr/vad.rs:106`](../../crates/ffai-mercury/src/asr/vad.rs#L106) | `frame_energies()` | `log10`×1 | slow |
| 0 | [`src/asr/vad.rs:125`](../../crates/ffai-mercury/src/asr/vad.rs#L125) | `percentile()` | `round`×1 | round |
| 3 | [`src/asr/vocab_int8.rs:151`](../../crates/ffai-mercury/src/asr/vocab_int8.rs#L151) | `new()` | `round`×1 | round |
| 1 | [`src/asr/vocab_int8.rs:232`](../../crates/ffai-mercury/src/asr/vocab_int8.rs#L232) | `cpu_fwd()` | `round`×1 | round |
| 0 | [`src/asr/wav2vec2.rs:278`](../../crates/ffai-mercury/src/asr/wav2vec2.rs#L278) | `forward()` | `softmax`×1 | act |
| 1 | [`src/asr/wav2vec2.rs:574`](../../crates/ffai-mercury/src/asr/wav2vec2.rs#L574) | `emissions()` | `softmax`×1 | act |
| 0 | [`src/asr/whisper_candle.rs:834`](../../crates/ffai-mercury/src/asr/whisper_candle.rs#L834) | `adaptive_ctx_secs()` | `ceil`×1 | round |
| 0 | [`src/tts/vits.rs:285`](../../crates/ffai-mercury/src/tts/vits.rs#L285) | `synthesize_ids()` | `exp`×1 | slow |
| 1 | [`src/tts/vits.rs:594`](../../crates/ffai-mercury/src/tts/vits.rs#L594) | `durations_raw()` | `exp`×1 | slow |
| 0 | [`src/tts/vits.rs:609`](../../crates/ffai-mercury/src/tts/vits.rs#L609) | `durations()` | `ceil`×1 | round |
| 0 | [`src/tts/vits.rs:1000`](../../crates/ffai-mercury/src/tts/vits.rs#L1000) | `gelu_erf()` | `gelu_erf`×1 | act |
| 0 | [`src/tts/vits.rs:1042`](../../crates/ffai-mercury/src/tts/vits.rs#L1042) | `rq_spline_inverse()` | `exp`×1 | slow |
| 0 | [`src/tts/vits.rs:1087`](../../crates/ffai-mercury/src/tts/vits.rs#L1087) | `softplus()` | `exp`×1 `ln_1p`×1 | slow |
| 0 | [`src/tts/vits.rs:1119`](../../crates/ffai-mercury/src/tts/vits.rs#L1119) | `gauss()` | `ln`×1 `sin_cos`×1 | slow |
