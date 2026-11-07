# Mercury — fifteen veins for deterministic instruction-reduction wins

Survey date: 2026-11-06. Scope: `crates/ffai-mercury` (ASR: Whisper on candle +
the WhisperX layer; TTS: VITS/piper on candle). Read with the house skills
loaded (`instruction-counting`, `rusty-compiler-leverage`, `codec-eliminate-
redundancy`, `codec-memory-copies`, `rusty-curiosity`, `codec-measurement`).

**What "deterministic" means here.** This box has a 28–49 % op-level timing
spread (`examples/noise_floor.rs`) and no callgrind. Every vein below is sized
and gated on a number that cannot move run to run: a **count** (candle ops per
token, allocations per sentence, FFT calls per window, bytes memcpy'd / memset,
distance evaluations per cluster pass) or an **emitted-asm instruction count**.
Wall-clock is taken last, paired and interleaved, by `codec-measurement`'s bar,
and only for the veins whose count-win is large enough to resolve.

**Quality gates per vein** are stated. Byte-identical where the move is pure
plumbing; the existing oracles (`tests/oracle_decoder.rs` token identity,
`examples/vits_oracle.rs` stage tensors, the argmax-flip audit, corpus WER)
where arithmetic reorders.

Ranked by expected return. The ASR decoder is ~50 % of the ASR pipeline and
its framework residue was last measured at **57 %** (`docs/whys/encode-
residual-117.md`), so veins 1–3 lead.

---

## 1. Self-attention K/V cache: four full passes over the cache per layer per token

**Where:** `text_decoder.rs` `Attention::forward_self` (cat) → `attend`
(`split_heads(k).transpose(2,3).contiguous()`, `split_heads(v).contiguous()`).

**Shape of the defect:** the cache is stored as the raw `(1, seq, n_state)`
projection. Every token does `Tensor::cat` on both K and V (full copy of the
cache — the `.drain(..)`/`.remove(0)` O(n²) twin from `codec-memory-copies`),
then `attend` re-gathers the *whole* cache into the kernel's `(heads, HD, keys)`
layout and re-copies V contiguous. Four passes over `n × 1536 B` per layer per
token, O(n²) over the window. Cross-attention got the "cache the finished
operands" treatment (its comment names 63 % of decoder time saved); self-
attention did not.

**Fix:** a flat cache in kernel layout with capacity 448: K as `[heads][HD][cap]`
(appending a key is 64 strided stores), V as `[heads][cap][HD]` (one 256 B
memcpy). Hand `xattn_head` a `keys` count and the base pointers — no candle op
in the loop. `DecoderState::save/restore` becomes a length + a shared `Arc`
snapshot (beam search copies become O(1) refs to a copy-on-write buffer).

**Deterministic instrument:** bytes copied per window (counter), candle ops per
token (count), both before/after. Expected: cat+gather traffic for a 224-token
window ~600 MB → ~0; self-attn ops 10/layer/token → 1.

**Gate:** `oracle_decoder` token identity (byte-identical: same arithmetic,
same order). **Return:** large — the self-attn stage is dispatch-bound ("2 % of
bandwidth, 98 % dispatch", `flash_attn.rs` l.59), and this removes the dispatch.

## 2. Logit post-processing: three full-vocabulary exp passes and two tensor read-outs per token

**Where:** `decoder.rs` `decode_once` (step-0 `to_vec1` + the sampling
closure's second `to_vec1`), `apply_logit_filters` (`logsumexp(&logits[ts..])`
+ `max_f32(&logits[..ts])`), the per-token `values[chosen] - logsumexp(&values)`,
`argmax` (`max_by` with `total_cmp`, loop-carried, unvectorised).

**Count today, per token:** 1–2 copies of 51 865 f32 out of the tensor, 1 full
`max_f32`, 1 full exp-sum (`logsumexp`), 1 partial exp-sum over the 1 501
timestamp logits, 1 partial max, plus a scalar `max_by` argmax. `sample`'s
comment already calls this "the single densest transcendental site in the
workspace".

**Fix:** one pass. Expose the vocabulary projection's output as `&[f32]`
(write straight into a reusable `Vec` the decoder owns, skip `Tensor` for the
logits entirely — `VocabProj::forward_into`). Then: one `max_f32` over all V
(vectorised, exists), one `exp_sub_sum` pass with two accumulators (text mass
and timestamp mass come from the same pass; `logsumexp(ts..)` is
`max + ln(sum_ts)`), argmax as `max` then `position(== max)` (byte-identical to
first-wins; the vorbis argmin lesson). Step 0 reads `no_speech` from the same
pass. The temperature>0 path shares the exp pass with the sampler.

**Deterministic instrument:** full-vocab passes per token (count: ~5 → 2),
heap bytes per token (207 KB × 1–2 → 0), emitted-asm count of the fused
function.

**Gate:** byte-identical token ids. The `lp` value's summation order must be
kept (sum in the same order over the same elements) or pin it to the
`logprob_threshold` behaviour with the corpus. **Return:** ~1 M instructions
per token today in these passes against a ~2.5 ms token; 5–10 % of decode-step
instructions, zero risk.

## 3. Per-token projection plumbing: bias as a separate op, reshape pairs, embedding, `powf` per call

**Where:** `text_decoder.rs` `QLinear::forward` (`Half`/`Int8` arms: reshape →
`gemv_rows` → `apply_op1` → `broadcast_add(bias)` → reshape = 4–5 candle ops
and a fresh output `Vec` per projection; 8 projections per layer → ~36 ops per
token for 4 layers). `Attention::scale` computes `(head_dim as f64).powf(-0.25)`
on **every** attention call (8 libm `powf` per token). `forward_cross` calls
`flash_attn::serves(&k,&v)` twice per layer per token. `TextDecoder::forward`'s
embedding is `Tensor::new` + `unsqueeze` + `index_select` + `narrow` +
`broadcast_add` for a 384-float lookup. `F16GemvOp` clones an `Arc` per call.

**Fix (in order of yield):** fold the bias into `dot_f16`/`dot_i8_blocked`
(initialise the accumulator slot with `bias[i]` — no extra pass, no op); make
`F16Gemv::forward` write into a caller buffer for rank-2 single-row inputs so
the reshape pair disappears; compute `scale` at load and store it on
`Attention`; evaluate `serves` once per cache build (it only depends on shape
and dtype); embedding as a `copy_from_slice` + 384 adds into the flat residual.

**Deterministic instrument:** candle ops per token (count), `powf` calls per
token (8 → 0), allocations per token (counting `GlobalAlloc`).

**Gate:** byte-identical (bias add order unchanged: `acc + bias` is the same
association as `broadcast_add` on the finished dot... verify with the oracle —
if f32 `bias + Σ` vs `Σ + bias` differs in the last bit the oracle says so).
**Return:** this is the rest of the 57 % residue once veins 1–2 land. Together
1–3 are the "flatten the per-token block" campaign TTS already ran on its
decoder (`FlatDecoder`, 82.7 % → kernels), applied to the ASR decoder.

## 4. Mel front end: FFTs of all-zero padding frames, and a full STFT redone on escalation

**Where:** `mel.rs` `MelSpectrogram::compute`: 3 000 complex FFTs per window
regardless of content. The adaptive-context path (`whisper_candle.rs`)
computes the mel at the short context, and on escalation computes it **again**
at 30 s.

**Shape:** "work sized to the container rather than to what is live" (the
third redundancy in `codec-eliminate-redundancy`). A 6 s clip padded to 30 s
runs 2 400 FFTs over exact zeros. The FFT of an all-zero frame is exactly zero
(every op on `0.0` yields `±0.0`; `norm_sqr` is `+0.0`), so skipping it is
byte-identical. Also: `padded` reflect-pad allocates 480 k floats per window;
`Tensor::from_slice(&self.filters.weights)` copies the filterbank into a tensor
on **every** call (hold the `Tensor` in the struct); `from_slice(&power)` copies
2.4 MB and `to_vec1` copies 0.96 MB back.

**Fix:** (a) a per-frame "window is all zero" test (`frame.iter().all(==0)` is
one vectorised compare over 400 floats, or track the last non-zero sample
index once and derive the frame bound — O(1) per frame); (b) keep the power
spectrum of the 30 s-padded window from the short-context attempt: frames are
independent, so the short-context power spectrum is a prefix of the full one
and escalation only recomputes filterbank + log (cheap); (c) hoist the filter
tensor.

**Deterministic instrument:** FFT calls per window (3 000 → ~`ceil(speech
frames)`), FFTs per escalated window (2× → 1×), bytes copied per window.

**Gate:** byte-identical mel (`tests/oracle_mel.rs`). **Return:** mel is
1.6× slower than whisper.cpp (7.0 vs 4.4 ms per 30 s), ~2 % of pipeline; on
short clips (the corpus average is ~6 s) this removes ~80 % of the FFT work
and the double-mel on every escalated window. Small stage, near-free brick.

## 5. Zero-fill-then-overwrite, both engines

**Where (ASR):** `text_decoder.rs` `FastGelu` (`vec![0f32; n]` on 9.2 MB per
encoder layer MLP + the two conv outputs — ~46 MB memset per 30 s window),
`FastSoftmax` (same, plus **two `Vec` allocations per row** in the f16 arm:
1 500 rows × 2), `audio_encoder.rs` `conv1d_gemm` (`col` 6.9 MB zeroed then
fully written), `flash_attn.rs` (`out` zeroed in every op; per-head `scores`/
`acc`/`run_max`/`run_sum` allocated per call). **Where (TTS):**
`decoder_kernels.rs` `FlatFlow::run` (`cols`, `skip`, `acts` per coupling),
`conv_transpose1d_flat` (`Vec<Vec<f32>>` per output channel then a full
`flatten` copy — at stage 3 that is a 6 MB extra copy per upsampler), `FlatConv
::conv` outputs.

**Fix:** `Vec::with_capacity` + `spare_capacity_mut` write + `set_len`, the
exact shape `ffai_core::fastops::ElemOp` already uses (and the only shape that
won in the rusty_aac campaign — the `collect`/`unzip` spellings lost). For
`conv_transpose1d_flat`: `par_chunks_mut(l_out)` into one preallocated flat
output instead of `Vec<Vec>` + flatten.

**Deterministic instrument:** bytes memset per window / per sentence (counter
in the kernels, or `strace`-style allocator tally), allocations per call.

**Gate:** byte-identical (no arithmetic changes). **Return:** ASR ~50 MB of
discarded writes per window against an encoder at ~100 ms; TTS the
`Vec<Vec>`+flatten alone is one full output copy per upsampler. Mechanical,
low risk, and the house rule says do it before any kernel work.

## 6. TTS conv tile: a zeroed 64–512 KB scratch per rayon task, then a merge pass

**Where:** `decoder_kernels.rs` `conv1d_flat_prepacked_into`, parallel arm:
`let mut tile = vec![0f32; (co1 - co0) * bt]` per task, kernel accumulates into
`tile`, then a merge loop writes `out += tile + bias`.

**Arithmetic:** stage 3 (c=32, L=48.6 k): 48 tasks × 128 KB tiles = 6 MB
zeroed + 6 MB merged per conv, 6 convs per stage → ~36 MB of memset and
~36 MB of extra read+write per sentence at that stage alone; stages 1–2 add
~24 MB. That is the same order as the `clone`/`act`/`add` glue the per-op
profile already names at 22 % of the decoder. The comment says tasks "write
DIRECTLY into `out`"; the code still goes through the tile.

**Fix:** the AVX2 kernel already does load-add-store per strip; point it at
the task's disjoint `out` rows (the residual is already there; `conv_into` is
`+=` semantics) and add the bias in the first tap's pass or as a strided
pre-pass over the task's rows. No tile, no merge.

**Deterministic instrument:** bytes memset + bytes merged per sentence; the
kernel's emitted-asm store count per output element (2 → 1).

**Gate:** `kernels_match_candle_on_every_decoder_shape` (tolerance, as today;
summation order unchanged if the first tap initialises instead of adding to
zero — then byte-identical to the current kernel). **Return:** TTS decoder is
82.7 % of synthesis; this is one of the three named glue costs.

## 7. TTS flow: tensor round-trips through `from_slice`/`to_vec1`

**Where:** `decoder_kernels.rs` `FlatFlow::run`, closure `mm`:
`Tensor::from_slice(x)` (**copies** the im2col matrix, 192×5×190 f32 ≈ 730 KB,
16× per sentence ≈ 11.7 MB) → `matmul` → `flatten_all().to_vec1()` (copies the
result out). Same in `vits.rs` `rel_attention` (`qkv.to_vec1`), `layer_norm_
named` (`to_vec1` + gamma/beta `to_vec1` per call, 36× per sentence),
`gelu_erf` (`to_vec1` + `from_vec`), `conv()` depthwise arm (`to_vec1` of x, w,
bias per call).

**Fix:** `Tensor::from_vec` moves (the FFN note in `vits.rs` l.509–520 records
that `from_slice` vs `from_vec` was a measured 6 % CPU-time swing); read the
matmul output through candle's storage (`storage_and_layout`) or a `CustomOp`
that consumes it, instead of `to_vec1`. Keep `cols`/`acts` as `Tensor`s that
are rebuilt by `from_vec` of a recycled `Vec` (take the storage back out after
the matmul to recycle it). Hoist gamma/beta `Vec`s for every LayerNorm at load.

**Deterministic instrument:** bytes copied per sentence through
`from_slice`/`to_vec1` (counter), allocations per sentence.

**Gate:** stage oracles (`dec_in`, `m_p`) — byte-identical, no arithmetic.
**Return:** flow is the stage the campaign fought hardest; the whys say its
v1 "wash" was plumbing. ~20–30 MB of pure copies per sentence.

## 8. Int8 vocabulary GEMV: a horizontal reduction per 32-weight block

**Where:** `vocab_int8.rs` `dot_i8_blocked`: per block (12 per row, 51 865 rows
per token) — `maddubs` + `madd` + `add` (3 instr of work) followed by
`extracti128` + 2 `shuffle` + 3 `add` + `cvtsi` + scalar `mul`+`add` (~9
instructions of reduction). Roughly 70 % of the kernel's instructions are the
per-block reduce.

**Fix:** convert the block's 8 i32 lanes to f32 once (`cvtepi32_ps`) and FMA
with a broadcast `bscale[b]` into a **vector** f32 accumulator; one horizontal
reduce per **row**. Per block: 2 loads + 3 int ops + cvt + fma = 7 instructions
against ~12 today. ~40 % fewer instructions in the hottest single kernel of
decode (27 % of decode time).

**Deterministic instrument:** emitted-asm instruction count per row (exact).
**Gate:** NOT byte-identical (the row sum becomes 8 partial sums) — gate with
the argmax-flip audit (`FFAI_VOCAB_AUDIT`) plus corpus WER on test-clean (the
sensitive corpus, `OPEN.md` §3). **Caveat, stated up front:** the kernel
measures 53.6 GB/s and may already be bandwidth-bound on this box, in which
case the instruction win does not become a wall win here. It will on cores
with less bandwidth per core (E-cores, aarch64), and the asm count is a
verdict either way. Also: `xu` quantisation uses `f32::round` (ties-away,
libm call) 384× per token — `fastmath::round_ties_away_i32` exists.

## 9. Runtime switches and string-keyed lookups on hot paths

**Where (TTS, per sentence):** `vits.rs` reads `std::env::var` inside `conv()`
(×2–3 per conv call), `layer_norm_named` (36×), `gelu_erf` (24×), `ffn` (12×),
`rel_attention` (6×), `conv_flow_reverse`, `flow_reverse`, `decode`;
`decoder_kernels.rs` `FlatDecoder::run`/`FlatFlow::run` read `FFAI_PROFILE`.
Order of 250 `std::env::var` calls per sentence; each takes the process env
lock, two `GetEnvironmentVariableW` calls, UTF-16 conversion and a `String`
allocation. Every weight access is `self.t(&format!("{name}.weight"))` — a
`format!` allocation + `HashMap<String,_>` hash per op (~750 per sentence),
plus `geom.get(name)`. `rel_table` rebuilds and copies the `(2T−1)×hd`
relative tables per layer per call.

**Where (ASR, per window):** `decoder.rs` reads `FFAI_DEBUG_TOKENS` twice,
`FFAI_CANDLE_ENCODER`, `FFAI_CANDLE_DECODER`, `FFAI_ALLOW_NONSPEECH` ×2;
`whisper_candle.rs` reads 6–8 per transcribe. `DecodeConfig::default()` reads
three env vars.

**Fix:** `knobs::Flag`/`Num` (one relaxed atomic load) for every per-call read
that is not an `ab_clips`-style per-transcribe flip (those are documented and
stay). Resolve weight names to a `Vec<Tensor>` index table at load (the
"producer already knew" move); build rel tables per `T` once per call for all
six layers' embeddings rather than once per layer.

**Deterministic instrument:** env reads per sentence/window (count via a
wrapper), allocations per sentence (`format!` + `String` from env).

**Gate:** byte-identical. **Return:** small in wall on a fast box (~0.3 ms per
sentence), but this is the rusty_zstd "cold arm of a runtime knob is a
per-call cost" class — the inlined parse chains bloat the callers' frames —
and it closes a consistency gap (ASR has `knobs.rs`, TTS does not).

## 10. Diarization clustering: pairwise distances recomputed on every merge

**Where:** `diarize.rs` `cluster`: each iteration recomputes
`cosine_distance` for every member pair of every cluster pair, and
`cosine_distance` recomputes both norms (two `sqrt`) each call. Complexity
O(n³)–O(n⁴) distance evaluations in windows `n`, each 192-dim.

**Arithmetic:** 40 windows (one 30 s horizon) is fine — the anatomy says
"clustering ~0". 200 windows (a 150 s file, or `max_speakers` forcing merges
to the end) is ~8 M distance evaluations × ~600 FLOPs ≈ 5 GFLOP, i.e. seconds,
on a stage the model-free design was meant to keep free.

**Fix:** L2-normalise embeddings once (cosine distance becomes `1 − dot`), fill
the `n×n` distance matrix once, and update average-linkage distances with the
Lance–Williams formula on merge (weighted mean of the two merged rows).
Identical merge sequence if ties are broken identically (lowest `(i, j)` —
pin it in a test against the current implementation on random embeddings).

**Deterministic instrument:** distance evaluations per `cluster` call (count:
O(n³) → `n(n−1)/2`), `sqrt` calls per call.

**Gate:** label-identical against the current `cluster` on synthetic and
corpus embeddings (`examples/diarize_gate.rs`, DER unchanged by
construction). **Return:** removes a latent cliff for long-form and
`max_speakers` users; also `StreamState::parts` clones every stored embedding
per tick (hand out slices).

## 11. Kernel reachability: every hand-written kernel is x86_64-only

**Where:** `flash_attn.rs`, `f16_gemv.rs`, `vocab_int8.rs`, `decoder_kernels.rs
::conv_quad_avx2` all have `#[cfg(not(target_arch = "x86_64"))]` stubs that
decline. `ffai_core::fastmath` has NEON and simd128 twins; Mercury's own
kernels have none. On **aarch64-apple-darwin** (in the release matrix per the
`flash_attn.rs` l.510 note) and **wasm32** (the RAG Converter deployment in the
README runs Mercury in the browser) the decoder takes the candle three-op
attention, f32 GEMV through candle's m=1 path (the documented 2.5× cliff),
and the f32 vocabulary matmul (the pre-4.34× state).

**Fix:** portable twins. NEON for `xattn_head` (8 lanes → 2×4), `dot_f16`
(`vcvt_f32_f16`), `dot_i8_blocked` (`vmull_s8`/`vpadalq`); simd128 for the
same three (`i16x8_extmul`, `f32x4_relaxed_madd` or plain FMA emulation). The
scalar twins and `*_matches_scalar` tests already exist as oracles.

**Deterministic instrument:** the kernel-reach census (`REACHABILITY.md`):
bytes down the kernel path vs the fallback path, per target. Today: 0 % on
both non-x86 targets.

**Gate:** `*_matches_scalar` on each target + token identity. **Return:** this
is not an instruction-count win on this box; it is the largest return for the
two shipped targets that currently run the slowest arms. Listed here because
"a fast path nothing calls is slow code with extra maintenance" and the
browser deployment is the one users see.

## 12. Beam search: a full 51 865-element indirect sort per beam per step

**Where:** `decoder.rs` `top_k`: builds a `Vec<u32>` of all indices,
`sort_unstable_by` with indirect `total_cmp` (~830 k compares, cache-hostile
indirection), then takes `k ≤ 16`. `logsumexp` is computed per beam per step
again inside `top_k` (the sampler already computed the max; vein 2's shared
pass applies). `candidates` clones `beam.tokens` for each of the `k`
expansions.

**Fix:** `select_nth_unstable_by` to `k` then sort the head (the nucleus-
sampling lesson: sort only the head you read, reproduce tie order explicitly
— index ascending), or a single pass keeping a `k`-element insertion buffer
(k ≤ 16: branch-light, L1-resident). Share the `max`/`exp` pass with vein 2.

**Deterministic instrument:** compares per `top_k` (count), allocations per
step.

**Gate:** identical `(token, lp)` lists including tie order (test on
constructed ties). **Return:** beam users only (`FFAI_BEAM_SIZE`, "~5× the
cost"), but the sort is pure overhead inside that 5×.

## 13. WhisperX model-free kernels: filterbank sparsity, aligner spans, VAD energies

**Where:** `fbank.rs` `compute`: 80 filters × 201 bins dense dot per frame
where each triangular filter is non-zero over ~10–30 bins; window fill does
`padded.get(start + i).copied().unwrap_or(0.0)` per sample (Option + bounds
check inside a 400-wide loop that is never out of range after the explicit
pad). `align.rs` `forced_align`: span collection is `(0..t_n).filter(|t|
owner[t] == j)` per token — O(T·J) scans after an O(T·J) DP; `emissions.at(t,
blank)` re-read inside the `j` loop; two `Vec<bool>` side tables. `vad.rs`
`frame_energies`: 400-sample windows on a 160 hop re-sum each sample 2.5×, and
the 10th percentile `clone`s + full-sorts the energies.

**Fix:** per-filter `(lo, hi)` support precomputed at construction (summing
`0.0 × p` into the accumulator is exact, so skipping zero weights is
byte-identical for finite `p`); `mel.rs`'s scalar fallback gets the same.
Single-pass span assembly in the aligner; hoist the blank lookup; one `u8`
flag array. Prefix sum of squares for VAD; `select_nth_unstable` for the
percentile.

**Deterministic instrument:** MACs per fbank frame (80×201 → Σ support),
iterations per `forced_align` (T·J·2 → T·J + T), sorts per `detect` call.

**Gate:** byte-identical (fbank: exact zero contributions; aligner: same
spans; VAD: same regions). **Return:** each is a small share of its stage —
fbank is 1.1 % of diarization, the aligner is dominated by the 95 M-parameter
model — but they are free, exact, and the aligner's quadratic scan is a
latent cliff on long segments.

## 14. Encoder front end as a storage-reading op; flash per-head scratch hoisted

**Where:** `audio_encoder.rs` `conv1d_gemm`: `x.flatten_all().to_vec1()`
copies the input out (4.6 MB for conv2), builds `col` (vein 5), then
`Tensor::from_vec` — a `CustomOp1` reading `CpuStorage` directly removes the
copy. `flash_attn.rs` `flash_head_strided`: `scores` (64 KB), `acc`, `run_max`,
`run_sum` allocated per head per layer per window (24+ allocations per
window); `attend_merged` zeroes `out` (vein 5). `EncoderAttention::forward`
`split_heads(q).contiguous()` and `split_heads(v).contiguous()` are two
2.3 MB gathers per layer that the K-side already removed via the transposed
projection — V could be projected straight into `(heads, seq, HD)` the same
way (`Wᵀ` reshaped per head), and q's gather is the one `OPEN.md` §2 refuted
for the *fused* projection but not for a per-tensor relayout.

**Deterministic instrument:** bytes copied per window, allocations per window.

**Gate:** byte-identical encoder features (same arithmetic). **Return:** conv
front end is 7.6 % of the encoder; the q/v prep is a measured
`ea_prep` bucket. Medium.

## 15. TTS text-encoder attention: scale in f64 per element, four transposes, per-call relative tables

**Where:** `vits.rs` `rel_attention`: `(f64::from(qkv_v[cq + t]) * scale) as
f32` per q element (an f64 round trip inside a scatter transpose), three
scatter transposes of q/k/v into `[h][t][d]`, an output transpose back, and
`rel_table` allocates + copies `(2T−1)×hd` for K and V per layer per call;
the row kernel's inner dot runs over three bounds-checked slices of `hd = 96`.

**Fix:** fold `1/sqrt(hd)` into the fused QKV weight's q rows at load — the
identical trick `EncoderAttention::key_wt_scaled` uses, exact (a scalar
multiply on the weight, applied once); emit q/k/v from `conv1d_k1_gemm` in the
layout the kernel wants (the GEMM can produce `[3·192, T]` and the kernel can
read `k[(d)*T + j]` column-wise, or transpose once via the matmul orientation
as the encoder K does); build the two relative tables once per call (they
depend only on `T`) and index them with the proven-in-range arithmetic;
`get_unchecked` is **not** needed — slice the three rows once per `j`.

**Deterministic instrument:** passes over the `3·192·T` activation per layer
(count), allocations per layer, emitted-asm bounds-check count in the row
kernel.

**Gate:** `m_p` stage oracle (reassociation-level deltas; the scale fold is
exact). **Return:** text encoder is at parity with onnxruntime already; this
is quality-of-code plus a modest count win. Last for a reason.

---

## How to run the campaign (the order, and the rig)

1. **Build the deterministic rig first** (`codec-analyzer` instrument #7 and
   `instruction-counting` §3): a counting `GlobalAlloc` behind a feature, a
   candle-op counter (wrap `apply_op*`/`matmul` call sites behind one helper,
   or count `CustomOp` names), FFT-call and bytes-copied counters, printed as
   **work-parity anchors** beside every number: tokens generated, windows,
   FFT calls, bytes. If an anchor moves, the experiment changed.
2. **Veins 1–3 as one campaign** on the ASR decoder, one brick per commit,
   `oracle_decoder` token identity after each. Count ops/token and bytes/token
   before and after each brick; take the paired wall A/B only at the end.
3. **Veins 4–5** are mechanical and byte-identical — land them next; they
   touch both engines and lower the memory traffic every later measurement
   sits on.
4. **Veins 6–7** (TTS) with `dec_in`/`audio` stage oracles.
5. **Vein 8** only with the argmax audit and test-clean WER; record the
   bandwidth caveat with the asm count either way.
6. **Vein 11** when a non-x86 box or CI runner is available to census on.
7. The rest as filler bricks; each is an afternoon and exact.

**Measurement rules from `codec-measurement` that bind this campaign:**

- **Sub-1 % bricks: the counter is the verdict, the clock is confirmation**
  (§15). Veins 9, 12, 13, 14 and 15 are each under 1 % of their pipeline on
  this box. Land them behind ONE switch each engine (`FFAI_VEINS=off` restores
  the old path) so the batch carries the timing verdict where 5–15 % resolves;
  each brick keeps its own counter and its own correctness gate.
- **A static instruction count has a noise floor too**: the register allocator
  moves a path ±1–3 instructions on any edit to the function. Quote an asm
  count (vein 8) only when it moves further than an unrelated edit does, and
  weight it by the path that runs, not the shortest path.
- **A fast path is priced in CALLS, never in static instructions.** Vein 4's
  zero-frame skip ADDS code; its verdict is FFT calls per window. Confirm from
  both sides: on a full 30 s of speech it must read exactly 3 000, so it can
  add work to no content.
- **Hoisting out of a short-circuit turns "sometimes" into "always".** Vein 2's
  fused pass must still skip the timestamp-mass computation when
  `cfg.timestamps` is off, and vein 3's hoisted `serves()` must be re-evaluated
  on every cache rebuild, not once per model.
- **Both arms must link the same dependencies.** Candle and `ffai-core` move
  under this crate; preserve the baseline binary before any bump and price the
  bump separately.
- **Re-run the null arm per session**, print `cpu/wall` beside every timing as
  the validity check (not as the stopwatch: Windows CPU time is 15.625 ms
  ticks), and make the harness refuse to report an effect under three of its
  own clock quanta. `examples/noise_floor.rs` and `examples/interleaved_ab.rs`
  are the existing instruments.
- **The browser and aarch64 arms (vein 11) are a reachability census first**,
  bytes down the kernel path vs the fallback per target, before any kernel is
  written.

**Record refutations with their number** in the source, as this crate already
does. Several of these veins were half-attempted before (the per-call
scratch hoist in `FlatDecoder::run`, the int8 MLP); the notes say *why* each
lost, and the veins above are shaped to avoid the recorded failure (first-touch
cost, marshalling cost, fork-join below the work threshold).
