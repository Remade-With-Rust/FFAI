# Mercury veins — the win ledger

Companion to `mercury-instruction-veins.md`. One row per brick, appended as it
lands. Every row carries the deterministic counter that proves the work was
removed (count, bytes, ops, calls), the gate that proved the output unchanged,
and the arithmetic that sizes it against the stage it sits in. Timing is taken
last, batched, by `codec-measurement`'s bar, and is never the keep/revert
verdict for a brick under ~1 % of the pipeline.

Status words: **landed** (gated, in tree) · **arithmetic** (sized from the
code, counter not yet instrumented) · **refuted** (built, measured, reverted,
number kept) · **blocked** (reason stated).

## How the numbers were taken

* **Identity gate**: `examples/veins_gate.rs` run on the tree BEFORE the
  campaign (a `git worktree` at `2635b6b` + the counters only) and on the
  tree AFTER it, same clips, same voice, same seed; the `# identity` section
  of the two outputs must be byte-identical. Clips: the first 12
  `librispeech-test-clean` utterances + `longform-01` (ASR), the first 8
  Harvard sentences (TTS, seed 7). Model: whisper-tiny.en, piper
  lessac-medium.
* **Work counters**: `asr::profile::Counters`, printed by the same harness
  under `FFAI_PROFILE=1`. Incremented at the sites the bricks remove, so the
  "before" column is a real count from the pre-campaign tree, not a formula.
  Per-token figures divide by generated tokens.
* **Allocation census**: a counting `GlobalAlloc` in the harness —
  allocations and bytes requested over the whole clip set.
* **Unit gates** (bit-identity, no model): `cargo test -p ffai-mercury --lib`
  — `flat_kv_tests`, `flash_attn::veins_tests`, `vocab_int8::reduction_tests`,
  `buf::tests`; `tests/veins_reachability.rs` for vein 11.
* The null arm: both harness binaries run on the same clip set twice; the
  identity text does not move between runs of the same binary (checked once
  per session before any A/B).

Static instruction counts quoted below are from reading the kernel, not from
`objdump`; they are bounded by the ±1–3 noise floor of register allocation
and are labelled "static" where used.

## Running sum

| Vein | Wins landed | Deterministic saving so far |
|---|---:|---|
| 1 Self-attn KV cache | 3 | KV bytes copied 510.0 MB → 14.4 MB (−97.2 %); cache ops 66.3 → 12.3 per token |
| 2 Logit post-processing | 3 | vocabulary heap 142.5 MB → 0; full-vocabulary passes 6.1 → 3.1 per token |
| 3 Per-token projection plumbing | 3 | plumbing ops 128 → 32 per token (4 → 1 per projection); `powf` 8.1 → 0 per token |
| 4 Mel STFT | 3 | FFT calls 31 000 → 18 972 (−38.8 %) over the same 18 STFT builds |
| 5 Zero-fill-then-overwrite | 3 | ASR memset 573.3 MB → 3.3 MB (−99.4 %) |
| 6 TTS conv tile | 3 | TTS memset 502.6 MB → 34.4 MB (−93.1 %); tile stores 3 → 1 instr per 8 outputs (static) |
| 7 TTS flow tensor copies | 3 | TTS tensor copies 232.0 MB → 126.0 MB (−45.7 %) |
| 8 Int8 vocab reduce | 3 | static: block reduce ≈10 → ≈5.25 instr, block MAC 5 → 4.5 instr per row, `roundf` gone; bitwise gate |
| 9 Hot-path env/string lookups | 3 | env reads 1 296 → 0; name lookups 1 488 → 816; TTS allocations 86 944 → 74 075 |
| 10 Diarization clustering | 3 | distance evals O(n³–n⁴) → n(n−1)/2 per `cluster` call (by construction; the harness does not diarize) |
| 11 Kernel reachability | 3 | census (29/13/16) + 5 reachability assertions + cross-target gate (see Gate runs) |
| 12 Beam top_k | 3 | elements sorted 17.9 M → 0; beam heap 143.1 MB → 0; passes 20.7 → 8.9 per token (beam run) |
| 13 WhisperX small kernels | 3 | static: fbank ≈25/201 bins, align O(T·J) → O(T), VAD sort → select; not on the harness path |
| 14 Encoder front end | 3 | encoder copies 158.7 MB → 101.2 MB (−36.2 %); q/v split copies gone (bitwise gate) |
| 15 TTS rel-attention | 3 | static: vectorised row max, 5 zero passes gone, ≈90 % of rel adds skipped; inside the TTS memset total |

Whole-pipeline census, same clips, same binaries: ASR 914 722 → 463 103
allocations (−49.4 %), 4 963 461 169 → 3 784 938 105 B requested (−23.7 %);
TTS 86 944 → 74 075 allocations (−14.8 %), 1 780 386 831 → 1 582 858 995 B
(−11.1 %). All 45 bricks landed; none refuted; the identity section of the
harness is byte-identical before and after (see Gate runs).

(Counter values under each brick are the two harness runs described above:
"before" is the pre-campaign tree, "after" is this tree, 13 clips / 669
generated tokens for ASR, 8 Harvard sentences for TTS. Where several bricks
remove work at the same counter the figure is quoted once and cross-referenced.)

## Bricks

### Vein 1 — self-attention KV cache (`asr/text_decoder.rs`, `asr/flash_attn.rs`)

**1.1 In-place append into a flat preallocated cache.** `FlatKv` replaces
`Option<(Tensor, Tensor)>` + two `Tensor::cat` per layer per token. A token is
written into place (2·n_state floats); the cache grows by doubling up to the
model's 448 positions, so nothing is copied on the steady state and the
allocation is reused across windows (`reset` clears `len`, keeps the buffers).
Counter `kv_bytes_copied`: 510 025 728 B → 14 413 824 B over the clip set
(762 370 → 21 545 B/token, −97.2 %); what remains is the tight prefix copy
that prefill and the first keys under the kernel floor still take. Arithmetic: at key
count *n*, the two cats copied 2·n·384·4 B per layer per token; the append
copies 3 072 B. Over a 100-token window that is Σ 3 072·n ≈ 15.5 MB per layer
→ 0.3 MB. Gate: `flat_kv_tests` (bit-identical to the tensor layout),
identity harness. Status: **landed**.

**1.2 Kernel-layout storage, strided kernel read.** K is held `[heads][HD][cap]`
and V `[heads][cap][HD]` — the operands the fused kernel streams — so the
per-token `split_heads(k).transpose(2,3).contiguous()` and
`split_heads(v).contiguous()` passes (two more whole-cache copies, 11 candle
ops) are gone; `xattn_head` gained a `kstride` parameter and reads the cache
where it lies (`decode_head`). Counter `kv_cache_ops`: 44 328 → 8 220 (66.3 → 12.3 ops/token, −81.5 %;
the legacy arm `FFAI_VEINS=off` reads 66.9/token on the same binary). The
floor is the three-op prefix path prefill and the first keys take. Gate: `veins_tests::decode_head_reads_a_strided_flat_cache_like_the_op_reads_a_tight_one`
(bitwise against the `CustomOp3` path). Status: **landed**.

**1.3 Direct kernel call, one output tensor.** The steady-state step calls
`decode_head` per head on the flat slices and wraps the merged context in ONE
`Tensor::from_vec` for the output projection: no `CustomOp3` dispatch, no
`transpose(1,2).flatten_from(2)` merge copy, no per-call `scratch`
allocation (the score row lives in `FlatKv`). Prefill and the first keys under
the kernel floor take the three-op path on a tight prefix copy — the same
operands the legacy arm built. Beam search's `save/restore` now copies the
used prefix (2·len·n_state floats per layer) where it used to share storage;
noted as the one cost added, a few percent of one decode step's weight
traffic. `FFAI_VEINS=off` keeps the legacy arm. Status: **landed**.

### Vein 2 — logit post-processing (`asr/decoder.rs`, `asr/text_decoder.rs`, `asr/vocab_int8.rs`)

**2.1 Logits land in the caller's buffer.** `TextDecoder::forward_into` +
`Int8Vocab::forward_into`: the int8 GEMV writes straight into a per-window
`Vec<f32>` that is refilled every token; the `(1, 51 865)` tensor built only
to be `to_vec1()`-copied is gone, and so is the SECOND copy step 0 took for
the no-speech read. Counter `vocab_heap_bytes`: 142 522 272 B → 0 (213 038 B/token → 0).
`vocab_passes` (bricks 2.1–2.3 together): 4 068 → 2 043 (6.1 → 3.1 per
token, −49.8 %).
Arithmetic: 207 460 B allocated + one full-vocabulary copy per token (two at
step 0) → 0. Status: **landed**.

**2.2 One maximum, reused.** `argmax` was a scalar `max_by(total_cmp)`
compare chain over the vocabulary and `logsumexp` then re-found the same max
before its exp-sum pass. Now the max is found once (vectorised `max_f32`),
`argmax_at` scans backwards for its last position by bit-equality (stops at
the first hit — the same index the compare chain returned), and
`logsumexp_with_max` takes it. Full-vocabulary passes per greedy token:
to_vec1 1 + argmax 1 + logsumexp 2 = 4 → exp-sum 1 + partial scan. Counter
`vocab_passes` — shared with 2.1 (6.1 → 3.1 per token). Documented divergence: a vocabulary whose maximum
logit is exactly ±0.0 in both signs. Status: **landed**.

**2.3 The timestamp rule hands back the maximum.** `apply_logit_filters`
already computed `max(text logits)` and `max(timestamp logits)` for Whisper's
timestamp-mass rule; it now returns the post-filter maximum (one of those two,
or their max — associativity makes it the bit the full scan would find), so
the sampler makes no max pass at all when timestamps are on. Counter
`vocab_passes` — shared with 2.1 (6.1 → 3.1 per token). Status: **landed**.

### Vein 3 — per-token projection plumbing (`asr/text_decoder.rs`, `asr/f16_gemv.rs`)

**3.1 Bias folded into the GEMV, rank emitted directly.** `F16Gemv::forward_fused`
adds the bias as each output row lands and returns the caller's rank, so the
`reshape` pair and the `broadcast_add` pass (4 ops, two 1.5 KB tensors) per
projection become 1 op. Eight single-row projections per layer. Counter
`plumbing_ops`: 128.0 → 32.0 per token (legacy arm `FFAI_VEINS=off` vs the
fused arm on the same binary; 21 408 ops over 669 tokens), exactly 4 → 1 per
projection. Same f32 dot then the
same f32 add. Status: **landed**.

**3.2 `powf` hoisted to load.** `Attention::scale2` (head_dim^-0.25 squared,
the factor q carries) is a constant of the layer geometry, computed in the
constructor from the same expression; the per-call `scale()` and its `powf`
(2 per layer per token, self + cross) are gone. Counter `powf_calls`:
5 424 → 0 over the clip set (8.1/token → 0). Status: **landed**.

**3.3 `serves()` decided once per cache build.** The cross-attention path
asked `flash_attn::serves(k, v)` twice per token for a cache whose shape
cannot change until the next window rebuilds it; `kernel_serves` is now set
when the cache is built. Arithmetic: 2 dims4() + predicate evaluations per
layer per token → 0 (static). Status: **landed**.

### Vein 4 — mel STFT (`asr/mel.rs`, `asr/whisper_candle.rs`)

**4.1 No FFT of an all-zero frame.** The 30 s context is zero-padded in the
sample domain, so on a 6 s clip 80 % of the 3 000 frames are silence; the
spectrum of an all-zero frame is exactly zero (every butterfly is a sum of
products with 0), so those columns are left as the zeros `power` already
holds. Counter `fft_calls`: 31 000 → 18 972 over the same 18 STFT builds (−38.8 %);
the saving is the padded tail of each window, so a full-speech window keeps
its full frame count. Status: **landed**.

**4.2 Filterbank tensor held, power moved, log fused.** The `(80, 201)`
filter tensor is built once in `new` instead of `from_slice`-copied every
window; `power` (2.4 MB) is MOVED into its tensor instead of copied; the
log10/peak pass is applied as the product is read out of the tensor (one
pass writing the final buffer) instead of a `to_vec1` copy followed by an
in-place pass. Arithmetic per 30 s window: 64 KB + 2.4 MB + 0.96 MB of copies
→ 0.96 MB. Status: **landed** (arithmetic; identical sequence of per-element
operations).

**4.3 Pad and reflect in one allocation.** `compute_window` builds the
zero-extended, reflect-padded frame directly from the audio instead of
`pad_or_trim` (1.9 MB zero-fill + copy) followed by `reflect_pad` (another
1.9 MB copy). Same samples at the same offsets. Arithmetic per window:
3.84 MB copied + 1.92 MB zeroed → 1.92 MB copied. Status: **landed**.

### Vein 5 — zero-fill-then-overwrite (`buf.rs`, `asr/text_decoder.rs`, `asr/audio_encoder.rs`, `asr/flash_attn.rs`)

**5.1 `FastGelu` / `FastSoftmax` write through spare capacity.** Both ops
zero-filled their output then overwrote every element; `buf::filled_f32` /
`filled_f16` (the `fastops::ElemOp` shape, the one form that measured a win
there) publishes the buffer only after the fill. The f16 softmax arm also
drops its second per-row zeroed `tmp` by running the softmax in place on the
widened row. Counter `memset_bytes` (ASR, bricks 5.1–5.3 together):
573 258 512 B → 3 336 192 B over the clip set (856 889 → 4 987 B/token,
−99.4 %). Arithmetic: encoder GELU
4 × 1500 × 1536 × 4 B = 36.9 MB per window; cross-attention softmax rows
per token under the kernel's floor. Status: **landed**.

**5.2 im2col without the zero pass.** `conv1d_gemm`'s `col` matrix is written
into spare capacity; the one edge cell each outer tap leaves is written as an
explicit 0.0. Arithmetic: conv1 (240 × 3000) + conv2 (1152 × 1500) floats =
9.8 MB zeroed per window → 0. Status: **landed**.

**5.3 Encoder attention output without the zero pass.** `MergedAttnOp`'s
`(seq, heads·HD)` output is written by exactly one head per column band;
it is now published through `filled_f32`. Arithmetic: 4 × 2.3 MB per window
→ 0. Status: **landed**.

### Vein 6 — TTS conv tile (`tts/decoder_kernels.rs`)

**6.1 Quad kernel stores instead of load-add-store.** Each interior strip of
`conv_quad_avx2` is the only writer of its tile cells and holds the whole
ci × j reduction in registers, so `tile += acc` was reading zeros:
`0.0 + acc == acc` for every value the accumulator can hold. One `vmovups`
replaces `vmovups`+`vaddps`+`vmovups` per 8 outputs (static). Status: **landed**.

**6.2 Tile zeroed only where the scalar paths accumulate.** With 6.1, a full
quad row's interior strips need no zero; only the edge columns and the <8-wide
tail do. The tile is built in a `filled_f32` closure that zeroes those ranges
and then runs the accumulation, so every cell is written before publish.
Counter `memset_bytes` (TTS, 8 sentences, bricks 6.2, 6.3, 15.2 together):
502 553 600 B → 34 446 336 B (−93.1 %). `tile_merge_bytes` is unchanged at
471 050 240 B: the tile→output merge pass is not claimed by this vein. Arithmetic at k=3, dilation ≤
27: ≤ 108 of 1 024–8 192 columns per row keep the zero. Status: **landed**.

**6.3 Transposed conv rows straight into one buffer.** `conv_transpose1d_flat`
built a `Vec<Vec<f32>>` (one allocation per output channel) and then
flattened it (a second full pass); rows are now written into their final
positions by `par_chunks_mut`. Arithmetic per upsample stage: c_out
allocations + c_out × l_out × 4 B copied → 0. Status: **landed**.

### Vein 7 — TTS flow tensor round-trips (`tts/decoder_kernels.rs`, `tts/vits.rs`)

**7.1 Flow GEMM outputs read where they lie.** `FlatFlow::run`'s `mm` returned
`to_vec1()` copies of every product; the gate pass, the residual/skip adds and
the coupling subtract now read the product tensor through `buf::with_f32`,
with the in-layer bias folded into the read (`t + bias`, the same add the
separate pass made). Counter `tts_copy_bytes` (TTS, bricks 7.1 and 7.3
together): 232 031 232 B → 126 027 264 B (−45.7 %); the survivor is the input
`from_slice` copies named below plus the per-layer relative tables. Arithmetic per
coupling layer: (2c + 2c) × len floats; plus c/2 × len per coupling. The
input `from_slice` copies stay — allocate-and-move measured neutral before
(see `ffn`). Status: **landed**.

**7.2 Layer norm and GELU operands borrowed.** `layer_norm_named` copied x,
gamma and beta out with `to_vec1` on every call (36 calls per sentence);
`gelu_erf` copied x. All four are now borrowed (`flat3`, `with_f32`).
Arithmetic: 3 × 66 KB per layer norm, 66 KB per GELU → 0. Status: **landed**.

**7.3 Depthwise conv and rel-attention operands borrowed.** The dp depthwise
convs copied activation, weight and bias; `rel_attention` copied the fused
QKV output. Borrowed now. Counter `tts_copy_bytes` — shared with 7.1
(232.0 MB → 126.0 MB per 8 sentences). Status: **landed**.

### Vein 8 — int8 vocabulary GEMV (`asr/vocab_int8.rs`)

**8.1 Block reductions grouped four at a time, exactly.** The per-block
horizontal i32 reduce (extract, 2 shuffles, 2 adds, convert, multiply, add
≈ 10 instructions × 12 blocks per row) is replaced by one 4-block `hadd`
tree + one packed `cvt`/`mul` + four sequential adds (≈ 21 per 4 blocks,
static). Integer block sums are exact however the lanes are added; the f32
products and their summation ORDER are the legacy kernel's, so the row is
bit-identical. Gate: `reduction_tests::grouped_and_paired_kernels_match_legacy_bitwise`.
Status: **landed**.

**8.2 Quantiser rounds without libm.** `(v / sx).round()` was a `roundf`
call per activation element (384 per token), the one non-widenable
instruction in the loop; `fastmath::round_ties_away_i32` reproduces
ties-away exactly (gate: `quantiser_rounding_matches_round`). Status: **landed**.

**8.3 Two vocabulary rows per activation load.** `dot_i8_blocked_x2` shares
each 32-byte load of the quantised activation between two weight rows: 5
instructions per block per row → 4.5 (static), per-row arithmetic untouched
(same gate as 8.1). `FFAI_VEINS=off` routes to the legacy kernel. Status: **landed**.

### Vein 9 — hot-path env reads and name lookups (`asr/knobs.rs`, `tts/vits.rs`, `tts/decoder_kernels.rs`)

**9.1 `std::env::var` → `knobs::Present`.** Eleven env reads on the
per-conv / per-layer-norm / per-GELU paths of the VITS synthesizer (and two
in the kernels' `run`) become one relaxed atomic load each; `knobs::reset_all`
lets the in-process A/B harnesses that flip variables keep working. Counter
`env_reads`: 1 296 → 0 (162 per sentence → 0). Status: **landed**.

**9.2 Conv and layer-norm operands resolved once.** `conv_cache` /
`ln_cache` map a canonical name to weight + bias + geometry (gamma + beta),
so a conv is ONE map lookup instead of two `format!`s and three lookups, and
a layer norm one instead of three. Counter `name_lookups`: 1 488 → 816
(186 → 102 per sentence, −45.2 %). Status:
**landed**.

**9.3 Encoder and DDSConv layer names formatted at load.** `EncLayerNames`
and `DdsLayerNames` hold every per-layer string the text encoder and the four
DDSConv stacks touch; the per-sentence `format!` allocations in those loops
(≈ 20 per encoder layer, 12 per DDSConv stack) are gone. Allocation census
(TTS, 8 sentences, every TTS brick together): 86 944 → 74 075 allocations
(−14.8 %), 1 780 386 831 → 1 582 858 995 B requested (−11.1 %). Status: **landed**.

### Vein 10 — diarization clustering (`asr/diarize.rs`, `asr/diarizer.rs`)

**10.1 Pairwise distances once.** `cluster` re-evaluated `cosine_distance`
for every member pair of every cluster pair on every merge round; the n × n
matrix is built once and the average-linkage sums read it in the same member
order, so each merge decision is the same f32. Counter `dist_evals` per
`cluster` call: O(n³–n⁴) → n(n−1)/2. The whisper harness does not run the
diarizer (`dist_evals` reads 0 in both arms), so this vein's figure is by
construction, not from the clip set. Gate: `diarize` unit tests (unchanged)
+ exactness by construction. Status: **landed**.

**10.2 Norms once per embedding.** `cosine_distance_with_norms` takes the two
`sqrt(Σx²)` values, computed once per embedding (n sqrt + n dots) instead of
twice per pair (n² each). Same expression, same bits. Status: **landed**.

**10.3 Stream state borrowed, not cloned.** `StreamState::parts_ref` hands
the incremental diarizer `&[f32]` views; `parts` cloned every stored
embedding (192 floats × windows in the 30 s horizon) on every tick. `cluster`
is generic over `AsRef<[f32]>` so both callers compile unchanged. Status:
**landed**.

### Vein 11 — kernel reachability (`docs/plans/mercury-kernel-reachability.md`, `tests/veins_reachability.rs`)

**11.1 Census.** 29 `cfg(x86_64)` items, 13 non-x86 stubs, 16
`target_feature` kernels across 4 files, with each kernel's gate, the shipped
shape it serves and its closed-gate fallback tabulated. Status: **landed**.

**11.2 Reachability tests.** On an AVX2 box `serves`, `attend_merged`,
`attend_merged_unsplit`, `F16Gemv::new` and `Int8Vocab::new` must ACCEPT the
whisper-tiny shapes; on any other machine they must decline. A predicate
change that routes production onto a fallback now fails a test. Status: **landed**.

**11.3 Cross-target compile gate.** `cargo check -p ffai-mercury --lib
--no-default-features` passes for `aarch64-unknown-linux-gnu` and
`wasm32-unknown-unknown`. The first wasm32 run FAILED: `MergedAttnOp` named
the x86-only `flash_head_strided_io` (brick 14.2) with no non-x86 stub — the
same class of defect the census found (13 stubs, two of them historically
missing) — and the stub is now in place. With default features the aarch64
check stops in the `blake3` / `aws-lc-sys` build scripts for want of a cross
C toolchain on this box; that is the dependency set, not mercury source.
NEON / simd128 twins of the decode kernels are the vein's open work and are
not claimed. Status: **landed** (gate), twins **blocked** (no non-x86 box).

### Vein 12 — beam `top_k` (`asr/decoder.rs`)

**12.1 Selection instead of a full sort.** One pass keeps the k best in a
k-sized insertion buffer; the 51 865-element `sort_unstable` per beam per
step is gone. Counter `topk_sorted` (beam run, `FFAI_BEAM_SIZE=3`, 4 clips,
117 tokens): 17 893 080 → 0 elements sorted (152 932 → 0 per token). Ties now break
by lower index — a DEFINED order where the unstable sort's was arbitrary.
Status: **landed**.

**12.2 No index vector.** The `(0..vocab).collect::<Vec<u32>>()` (207 KB per
beam step) is gone with the sort. Counter `vocab_heap_bytes` (beam) — by
construction 207 460 B → 0 per beam step; measured on the beam run
143 144 640 B → 0. Status: **landed**.

**12.3 `logsumexp` reuses the selection's maximum.** The best candidate IS
the maximum; `logsumexp_with_max` skips the max pass. `vocab_passes` per
beam step: 1 (idx) + 1 (sort) + 2 (logsumexp) → 1 (scan) + 1 (exp-sum);
measured on the beam run 2 423 → 1 043 (20.7 → 8.9 per token).
Beam decoding is opt-in (`FFAI_BEAM_SIZE`); its identity run is listed under
"Gate runs". Status: **landed**.

### Vein 13 — WhisperX small kernels (`asr/fbank.rs`, `asr/align.rs`, `asr/vad.rs`)

**13.1 Fbank dot over the filter's support.** Each triangular filter is
non-zero on ~25 of 201 bins; the band is found once and the dot runs over it.
`0 × p` contributes +0.0, which leaves a partial sum unchanged, so the fold is
the same number. Counter `small_kernel_ops` (fbank share): 80 × 201 → Σ band
widths per frame. The whisper harness does not reach the fbank or the forced
aligner, and at the VAD site the counter counts the frames selected over
(17 603 in both arms, equal by construction), so the vein's saving is static
and from the kernels' own unit tests, not the clip set. Status: **landed**.

**13.2 Forced-align spans in one sweep.** `owner` is non-decreasing in t, so
token j's frames are one contiguous run; the per-token full-frame `filter`
scan (O(T·J)) is a single sweep (O(T)), emitting frames visited in the same
ascending order. Status: **landed**.

**13.3 VAD percentiles by selection.** `select_nth_unstable_by` finds the
k-th smallest energy in O(n) instead of sorting the frame energies (twice,
for `detect` and `energy_contrast_db`); the k-th smallest is the same value
whatever the algorithm. Status: **landed**.

### Vein 14 — encoder front end (`asr/audio_encoder.rs`, `asr/flash_attn.rs`, `asr/text_decoder.rs`)

**14.1 im2col reads the input in place.** `conv1d_gemm` copied its input
with `to_vec1` (0.96 MB conv1, 4.6 MB conv2 per window); it now borrows the
storage. Counter `enc_copy_bytes`: 158 720 000 B → 101 184 000 B over the
clip set (−36.2 %); what remains is the per-window materialisation of the
encoder attention operands on the non-kernel path, which no brick claims. Status: **landed**.

**14.2 q read unsplit by the encoder kernel.** `attend_merged_unsplit` reads
head h of q as the column band `h·HD..` of the `(seq, heads·HD)` projection
with a row stride of `heads·HD` (`scores_tile` gained `q_stride`), so the
`split_heads(q).contiguous()` copy (2.3 MB per layer) never happens. Gate:
`veins_tests::unsplit_merged_attention_matches_the_split_operand_path_bitwise`.
Status: **landed**.

**14.3 v read unsplit.** The same for v (`accum_pv` gained `v_stride`):
another 2.3 MB per layer per window. Arithmetic: 4 layers × 4.6 MB = 18.4 MB
per window of copies removed, and the two `transpose` + `contiguous` op pairs
with them. `FFAI_VEINS=off` keeps the split arm. Status: **landed**.

### Vein 15 — TTS relative attention (`tts/vits.rs`)

**15.1 Row softmax max vectorised.** `fold(NEG_INFINITY, f32::max)` over each
score row (a loop-carried, NaN-aware reduction LLVM will not lane) is
`fastmath::max_f32`. Max is associative on finite scores, so the value is
the fold's to the bit. Status: **landed**.

**15.2 Attention buffers without zero passes.** q/k/v head-major copies, the
per-head output and the transposed output (5 buffers of 192 × T floats) are
each fully assigned; they are published through `filled_f32`. Arithmetic at
T = 88: 5 × 67 KB zeroed per layer × 6 layers → 0. Status: **landed**.

**15.3 Relative term skipped outside the window.** The relative tables are
zero for |i − j| > 4, so for ~90 % of (i, j) pairs at T = 88 the `+ rr[d]`
was an add of +0.0 — which cannot change a value, and whose product folds
into a sum that is never −0.0. Those pairs take the plain dot (one load and
one add fewer per MAC, static). Status: **landed**.

## Gate runs

Both binaries are `cargo build --release -p ffai-mercury --example veins_gate`
(Rust 1.95, x86-64 AVX2/F16C box, Windows). "Before" is the `git worktree` at
`2635b6b` with only the counters patch applied; "after" is this tree. Clip set:
the first 12 `librispeech-test-clean` utterances + `longform-01` (13 clips,
176.3 s, 669 generated tokens), 8 Harvard sentences through piper
lessac-medium at seed 7. The `# identity` section is 38 lines (ASR segment
timestamps + text, TTS sample count + FNV-1a of the sample bits).

| Run | Arms | Result |
|---|---|---|
| Identity, greedy | before vs after | byte-identical (empty diff) |
| Null arm | after vs after, ASR only (`--no-tts`) | byte-identical; TTS identity rests on the before/after match at seed 7 |
| Identity, beam (`FFAI_BEAM_SIZE=3`, 4 clips) | before vs after | byte-identical; `topk_sorted` 17 893 080 → 0, beam heap 143 144 640 B → 0, `vocab_passes` 20.7 → 8.9 per token |
| Legacy arm (`FFAI_VEINS=off`, 4 clips) | before vs after-with-veins-off | byte-identical; the legacy arm reads `plumbing_ops` 128.0 and `kv_cache_ops` 66.9 per token, matching the before tree |
| Unit gates | `cargo test -p ffai-mercury --lib` (diarize, diarizer, vocab_int8::reduction_tests, buf, flash_attn::veins_tests, text_decoder::flat_kv_tests) | 31 passed, 0 failed |
| `tests/veins_reachability.rs` (4 tests) + `tests/oracle_mel.rs` | this box (AVX2 + F16C) | 5 passed: every shipped kernel accepts the whisper-tiny shapes; mel matches the OpenAI oracle |
| `cargo clippy --no-deps -p ffai-mercury --lib -- -D warnings -A unsafe_code` (the blocking CI form) | after | clean; the campaign's 12 findings (const fn, must_use, redundant pub(crate), if-not-else, single-match-else, one f64→f32 cast annotated, one redundant unsafe block) fixed with no arithmetic change |
| `cargo check --lib --no-default-features --target {aarch64-unknown-linux-gnu, wasm32-unknown-unknown}` | after | both compile (see 11.3 for the stub this caught) |
| `cargo fmt -p ffai-mercury -- --check` | — | clean |

Work counters, after vs before, greedy run (full table in the harness output):

| Counter | Before | After | Change |
|---|---:|---:|---:|
| ASR `kv_bytes_copied` | 510 025 728 | 14 413 824 | −97.2 % |
| ASR `kv_cache_ops` | 44 328 | 8 220 | −81.5 % |
| ASR `vocab_passes` | 4 068 | 2 043 | −49.8 % |
| ASR `vocab_heap_bytes` | 142 522 272 | 0 | −100 % |
| ASR `powf_calls` | 5 424 | 0 | −100 % |
| ASR `plumbing_ops` (legacy arm vs fused, per token) | 128.0 | 32.0 | −75 % |
| ASR `fft_calls` | 31 000 | 18 972 | −38.8 % |
| ASR `memset_bytes` | 573 258 512 | 3 336 192 | −99.4 % |
| ASR `enc_copy_bytes` | 158 720 000 | 101 184 000 | −36.2 % |
| ASR allocations / bytes | 914 722 / 4 963 461 169 | 463 103 / 3 784 938 105 | −49.4 % / −23.7 % |
| TTS `memset_bytes` | 502 553 600 | 34 446 336 | −93.1 % |
| TTS `tts_copy_bytes` | 232 031 232 | 126 027 264 | −45.7 % |
| TTS `env_reads` | 1 296 | 0 | −100 % |
| TTS `name_lookups` | 1 488 | 816 | −45.2 % |
| TTS allocations / bytes | 86 944 / 1 780 386 831 | 74 075 / 1 582 858 995 | −14.8 % / −11.1 % |
| unchanged by design: `tile_merge_bytes` 471 050 240, `rel_table_builds` 96, `small_kernel_ops` 17 603, `stft_builds` 18 | | | |

### Final-tree re-gate, and an open TTS determinism finding

After the clippy fixes the release harness was rebuilt on the exact
committed tree. The greedy identity (13 clips + 8 sentences) and the beam
identity (4 clips) are again byte-identical to the pre-campaign binary.

The legacy-arm re-run then showed a diff, and the side that moved was the
**pre-campaign binary**. One of its runs (`--clips 4`) produced different
hashes for all 8 TTS sentences, with identical sample counts and identical
ASR text. Three repeats of the pre-campaign binary, the legacy arm and the
default arm then gave the same 8 hashes in all 9 runs, the hashes every other
run of both binaries produced.

So the pre-campaign tree has a **rare run-to-run nondeterminism in TTS**:
1 run in about 19 diverged. The whole run moves together while the durations
stay put, which points at the waveform path (flow or decoder), not the
duration predictor. Nothing in the TTS code picks a kernel by timing. In
about 10 checked runs the post-campaign tree has not diverged, but at that
rate 10 runs cannot rule it out. This matters beyond the campaign because
the README promises byte-identical output per seed. **Open**, and not caused
by this campaign.

**Rate measured.** Each binary was run 150 times TTS-only (`--no-asr`, 8
sentences, seed 7), interleaved, on the loaded box, and the identity lines of
every run were hashed:

| Binary | Usual output | Alternate output | Alternate rate |
|---|---:|---:|---:|
| pre-campaign | 142 | 8 | 5.3 % |
| post-campaign | 146 | 4 | 2.7 % |

There are exactly **two** outputs, and the alternate is **the same in both
binaries**. So the synthesizer is bimodal: something decided once per process
selects one of two exact waveforms. The campaign reproduces both modes byte
for byte, and the identity claim holds within each mode. The 8-vs-4 split is
within chance (Fisher exact, two-sided, p ≈ 0.38), so no rate change is
claimed in either direction.

Consequence outside this campaign: the README's "byte-identical output per
seed" is true of the code path but **not of the process**, which differs in
about 3–5 % of runs on this box. Next step, not taken here: log which mode a
run lands in alongside per-stage hashes (encoder, duration predictor, flow,
decoder) to find the first stage that differs, then check thread-count- or
dispatch-dependent reductions in that stage.

### Timing pass (2026-11-07): no admissible verdict on this box

Batched after the counts, per `codec-measurement`: the two release harness
binaries run ABBA-interleaved (before, after, after, before) for 6 rounds per
arm after one discarded warm-up pair, pinned to logical CPUs 0–15 (the
P-core threads of the i7-14650HX), wall time by stopwatch and CPU time from
the process. ASR arm = `--no-tts` (13 clips); TTS arm = `--no-asr`
(8 sentences). n = 12 runs per binary per arm.

| Arm | Metric | Before median | After median | Change | After faster, paired rounds | Same-binary gap (median) |
|---|---|---:|---:|---:|---:|---:|
| ASR | wall | 6.07 s | 8.17 s | +35 % | 2 / 6 | 0.9–1.3 s |
| ASR | CPU | 19.63 s | 19.74 s | +0.5 % | 3 / 6 | 1.0–1.9 s |
| TTS | wall | 1.46 s | 1.40 s | −3.8 % | 3 / 6 | 0.5–1.1 s |
| TTS | CPU | 7.50 s | 7.42 s | −1.0 % | 2 / 6 | 0.6–0.8 s |

Verdict: **inadmissible, not a regression and not a win.** The null arm
fails the bar: two runs of the SAME binary inside one round differ by more
than any before/after difference, ASR wall time drifts from 20 s to 4 s
across the session as other jobs come and go, and the paired-round signs are
coin flips. The clock cannot see this campaign here, and the arithmetic says
it should not expect to: the bytes removed from the ASR arm (≈ 1.2 GB of
memset + copies over 20 s of CPU) cost on the order of 0.1 s at memory
bandwidth, below the 1–2 s same-binary spread. The keep verdict for every
brick stays the count plus the identity gate. A clock verdict needs a quiet
box (no concurrent cargo jobs) or the `instruction-counting` route
(callgrind Ir under WSL, which this box has).
