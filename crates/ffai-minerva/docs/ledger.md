# Minerva ledger

One line of evidence per brick. Plan: rusty_sloth `docs/plans/bitnet.md`, Phase 4.
Machine: i7-14650HX (8P+8E, 24 threads, L3 30 MB), 2 x 16 GB DDR5-5600 (dual channel,
89.6 GB/s theoretical), Windows 11, rustc 1.95.0.

## Brick a — roofline (2026-11-23)

Instrument: `minerva roofline` (`src/roofline.rs`).

### Bytes per token — deterministic, from the file's tensor table

`ours-i2_s.gguf` (rusty_sloth's export of bitnet-b1.58-2B-4T):

| term | bytes / token | note |
|---|---:|---|
| ternary projections (210 `I2_S` tensors, 30 layers x 7) | 521.0 MB | codes + per-tensor scale block |
| head (tied `token_embd`, 128,256 x 2,560, f16) | 656.7 MB | **55.7 % of a token** |
| norms (f32) | 1.76 MB | |
| embedding row (input lookup) | 5 KB | |
| KV cache (f16, K+V, 30 layers x 640) | 76.8 KB per cached position | 78.6 MB at position 1,024 |
| **total at position 0** | **1,179.5 MB** | |

With an int8 head screen (one byte an element + an f32 bound a row; the 1-12 survivors'
exact recompute is ignored), the head is 328.9 MB and a token is **851.6 MB, -27.8 %**.
That confirms the brief's "roughly a quarter" from the file's own counts.

### Read bandwidth — CONTENDED, a lower bound, not yet banked

Method: a 512 MiB buffer (17x L3), every page pre-written; N scoped threads each sum a
contiguous share (8 independent lanes); a NULL arm runs the same harness over 0 bytes;
read and null arms run ABBA, 21 passes each, plus one warm-up pass discarded. The clock is
`Instant` (QPC, 100 ns); GB/s = bytes / (arm - null arm). The process ran at High
priority, unpinned (a multithreaded read). **The box was loaded:** `faucet.exe` (another
project) was using ~55 % of all 24 threads, a solidcraft release build was running, and
only 3.6 GB of RAM was free. Read spread was 1.36-1.91x, so treat the medians as a floor.

| threads | best GB/s | median GB/s | null / read |
|---:|---:|---:|---:|
| 1 | 16.2 | 14.2 | 0.5 % |
| 2 | 25.4 | 20.9 | 1.1 % |
| 4 | 37.7 | 28.7 | 2.0 % |
| 6 | 47.6 | 40.7 | 3.8 % |
| 8 | 48.3 | 43.5 | 4.1 % |
| 12 | 58.5 | 58.2 | 7.8 % |
| 16 | 68.4 | 56.7 | 12.1 % |
| 24 | 71.0 | 69.8 | 14.8 % |

### Ceiling (at the 24-thread median, 69.8 GB/s): tok/s <= bandwidth / bytes per token

| head | pos 0 | pos 64 | pos 512 | pos 1024 |
|---|---:|---:|---:|---:|
| f16 | 59.2 | 58.9 | 57.3 | 55.5 |
| int8 screen | 82.0 | 81.5 | 78.4 | 75.0 |

At 8 threads (43.5 GB/s) the f16 ceiling is 36.9 tok/s at position 0.

### What it decides

- **The head is the biggest single term (55.7 %).** That makes brick d (the int8 screen)
  worth up to 1.39x on the ceiling, more than any ternary-kernel change can buy.
- **TL1 does not save bytes; TL2 does.** TL1 is still 2 bits a weight. TL2 packs 3 weights
  in 5 bits (1.67 bits a weight), so it would cut the 521 MB ternary stream by about 87 MB
  per token: -7.4 % of a token with the f16 head, -10.2 % with the int8 screen. Brick e
  prices TL2 on that, plus an Ir count. It cannot be refuted on bytes alone.
- **Brick d requirement (rusty-sloth-05, 2026-11-23):** with the screen, the f16 head is read
  only for its 1-12 survivor rows a token (5 KB each) plus the input embedding row. So keep it
  **memory-mapped from the GGUF, not loaded**: resident becomes ~520 MB of codes + 328 MB of
  int8 head + KV, not +657 MB more, and that matters for a phone's budget. Measure resident set and
  page faults in both arms. Two caveats: mapping is `unsafe` (a fenced seam under
  `unsafe.md`, outside the `forbid` library core), and wasm32 has no mmap, so the browser
  build keeps the f16 head out of memory some other way (fetch survivor rows on demand, or
  ship the int8 screen plus an f16 side file).
- **The ceiling does not go in the README** until the bandwidth is rerun on a quiet box.
- **Open:** a clean rerun (faucet stopped, free RAM restored), then a P-core-only
  affinity sweep to see what the E-cores add.

## Brick b: the scalar engine, end to end (2026-11-23)

`src/model.rs` (GGUF load and checks), `src/engine.rs` (the `build_bitnet_158` graph, f16 KV
cache, greedy), `src/kernel.rs` (the I2_S GEMV, rows split across rayon), `src/numerics.rs`
(the float arithmetic). Run with `minerva generate --ids F --new N`.

### Gate: bitnet.cpp's tokens, teacher-forced, under the near-tie rule — PASS

`tools/conformance.py` (rusty_sloth's `step` / `teacher_forced` / `vocabulary`, imported):
our greedy ids for each bench prompt, then bitnet.cpp's `llama-server` (765741d, Windows MSVC,
CPU, n_probs 256) asked for its distribution at every step of them. The result is in
`docs/conformance-b.json`.

| prompt | in / out | agree | worst gap |
|---|---|---:|---:|
| p64 | 64 / 48 | 48/48 | 0 |
| p256 | 256 / 48 | 48/48 | 0 |
| p512 | 512 / 48 | 47/48 | 0.0012 |
| p1024 | 1024 / 48 | 48/48 | 0 |

The one parting (p512 step 38) is a near-tie on both sides: bitnet.cpp has 0.4538 against
our token's 0.4526, and we have 0.467 against 0.430. **191/192 steps, gate < 0.1.**

- **The gate can fail:** `--poison` (our ids rotated by one) reads 43/48 with a worst gap of
  0.998, exit 1.
- **The quick gate against bitnet.cpp's own greedy output** (`bitnet-golden/cpp-ours.json`,
  3 prompts x 16): 48/48. After renormalising ours over its top-4 (that server softmaxes
  among the top n_probs only), the largest probability difference is 0.050.

### Arithmetic rules adopted here (`numerics.rs`), for "same file, same tokens on every device"

- The integer paths are exact: the I2_S product and `act_quant`.
- Every transcendental (`expf`, `cosf`, `sinf`, `powf`) comes from the pure-Rust `libm` crate,
  never the platform libm (MSVC CRT, glibc and wasm differ in the last bit).
- Float dot products use one canonical order: 16 lanes, multiply then add, no FMA, a fixed
  pairwise reduction. The SIMD twins (brick c) must reproduce it bit for bit.
- Scale-setting sums (RMS norm, softmax denominator) run in f64, sequentially, as `ggml_float`.
- f16 rounding happens where the reference rounds: K and V cached as f16, the query and the
  attention weights rounded to f16 before their products, and the hidden state rounded to f16
  before the head.
- RoPE is NEOX (pairs half a head apart), with the angle advanced by repeated f32
  multiplication, as `ggml_rope`.
- The loader refuses RoPE scaling, `rope_freqs`, partial RoPE, and any tensor of the wrong
  type or shape.

### Speed, for orientation only (contended box, not a measurement)

24 threads, the scalar row product, no SIMD: decode 9.7-12.4 tok/s, which is already 16-21 %
of the contended 59 tok/s ceiling. Prefill runs the GEMV once per token (p1024: 69.6 s,
68 ms a token), so a GEMM prefill that decodes each weight row once for many tokens is open
work. It is not on the decode path the bench gates.

### Open

- `src/kernel.rs` holds a copy of rusty_ternary's private row product, gated `assert_eq!` to
  `kernel::gemv` on whole tensors. Requested from rusty-sloth-05: a public row-range
  `gemv_codes`. Swap to it when it lands.
- Activation rounding: rusty_ternary ties to even (training's rule); bitnet.cpp rounds half
  away from zero. They can part only on exact ties, which the near-tie gate covers.

## Brick c: a SIMD twin for every scalar kernel (2026-11-23)

`src/simd/` (`mod.rs` holds dispatch and the reach census; `x86.rs`, `neon.rs` and `wasm.rs`
hold the twins; `tests.rs` holds the oracle tests). Every kernel on the decode path has a
twin at each tier, **bit-identical** to its scalar oracle:

| kernel | oracle | SSSE3 | AVX2 | NEON | SIMD128 |
|---|---|:-:|:-:|:-:|:-:|
| I2_S GEMV | `rusty_ternary::kernel::gemv_codes` | yes | yes | yes | yes |
| f16 mat-vec (head, attention scores) | `numerics::dot16_f16{,_bytes}` | yes | yes | yes | yes |
| attention value sum | `numerics::weighted_sum_f16` | yes | yes | yes | yes |
| act_quant | `rusty_ternary::quant::act_quant` | yes | yes | yes | yes |
| rms_norm | `numerics::rms_norm` | yes | yes | yes | yes |
| softmax (scale, max, exp, sum) | `numerics::softmax` | yes | yes | yes | yes |
| f16 <-> f32 conversions | `half` | yes | yes (F16C) | yes | yes |
| finish, ReLU²·up, residual add, RoPE | `numerics::*` | yes | yes | yes | yes |

The only per-token code left scalar is the RoPE angle table: 64 `libm` cos/sin values a token,
which is setup, not a kernel.

**Oracle changes, made in scalar first so they could be twinned:** the f64 sums that set a
scale (RMS norm, softmax denominator) moved from a sequential order to 4 f64 lanes (an order
every ISA can express). `exp` moved from libm to `exp_canon`: Cephes' polynomial behind a
ties-to-even range reduction, plain f32 multiplies and adds, within 3e-7 of exp. Softmax now
takes the attention scale, as `soft_max_ext` does.

**Dispatch:** `Isa` is opaque, built only by run-time detection or a checked `MINERVA_ISA`, so
safe code cannot name an unsupported tier. Dispatch happens once per kernel call, with the loop
inside the `#[target_feature]` function. `unsafe` is confined to `src/simd/`; the crate is
`deny(unsafe_code)`.

### Validation

- **Oracle tests:** 22 tests, every kernel at every supported tier, `assert_eq!` on bits.
  Inputs cover every tail length, ±128 saturation, subnormals, ±0, ±inf, and exact rounding
  ties.
  - x86 (scalar, ssse3, avx2) runs natively.
  - **wasm32 (scalar, simd128) runs under wasmtime**, simd128 and scalar-only builds both.
  - The f16->f32 sweep covers all 65,536 halves; **f32->f16 was swept over all 2^32 floats** on
    SSSE3 and AVX2 (55 s, `--ignored`).
- **Planted bugs, each caught:** a dropped `- Σa` in the AVX2 GEMV, a reordered SSE reduction
  (a 1-ulp drift), and an FMA in the AVX2 mat-vec.
- **End to end, five configurations, one hash:** a golden prompt (13 + 16) gives the same 16
  ids and the same FNV-1a over every generated step's 128,256 logit bits (`f2f2db068ec651cb`)
  on x86 scalar, SSSE3 and AVX2, and **in WebAssembly** (`minerva.wasm`, wasmtime) at scalar
  and simd128.
- **Reach census:** each tier's run shows 100.00 % of every kernel's elements at that tier,
  with element counts equal across tiers. The GEMV counts 58,353,254,400 = 28 steps x
  2,084,044,800 weights.
- **Conformance gate rerun (default tier, AVX2):** 190/192, worst gap 0.0154, **PASS**
  (`docs/conformance-c.json`). The new parting (p1024 step 21) is a coin flip on both sides
  (bitnet.cpp 0.4802 / 0.4648, ours 0.4732 / 0.4702). The int8 activation quantiser is
  discontinuous, so a last-bit oracle change can move a probability ~0.02 downstream; that is
  what the near-tie rule absorbs. The golden prompts still give 48/48, and the worst probability
  difference is unchanged at 0.0501.
- **Clippy `-D warnings`** clean on x86_64, aarch64, wasm32+simd128 and wasm32 scalar-only.
- **NEON is compiled and clippy-clean but NOT executed:** this host has no ARM hardware and no
  emulator, and WSL is paused. Its tests need one run on an aarch64 machine.

### Speed (`tools/tier_ab.py`): one binary, `MINERVA_ISA` per arm, 10 rotated rounds, internal clock

24 threads, p64 + 64 new. Every run printed the same logits hash (work parity OK). The box
was lightly loaded (another session's build at ~9 % CPU).

| tier | decode tok/s median (min-max) | vs scalar | prefill p64 median |
|---|---|---|---:|
| scalar | 14.21 (13.09-14.40) | 1.000x | 3.26 s |
| ssse3 | 29.07 (25.46-30.01) | 2.036x, 10/10, z = +3.16 | 1.14 s |
| avx2 | 37.18 (34.87-38.62) | 2.650x, 10/10, z = +3.16 | 1.07 s |
| null (avx2 vs avx2) | median 0.969, range 0.858-1.059 | | |

AVX2 over SSSE3 is 1.18-1.40x per round (10/10), beyond the null range. In wasm, one thread:
simd128 2.90 tok/s against scalar 1.11 (2.6x).

At 37.2 tok/s x 1,179.5 MB a token, decode streams ~44 GB/s: 63 % of the contended 69.8 GB/s
ceiling. The f16 head is still 55.7 % of those bytes, which is brick d's target.

### Open

- Run NEON's tests on aarch64.
- Prefill is still one GEMV per token. `rusty_ternary::kernel::gemm_codes` (pinned) plus a
  GEMM twin that decodes each weight row once for many tokens is the prefill lever.
- Bandwidth rerun on a quiet box (still open from brick a).

## Standing rule: engine comparisons run at EQUAL thread counts (Tim, 2026-11-23)

Every Minerva vs bitnet.cpp (or any other engine) corpus comparison sets the thread count
explicitly on both engines: Minerva's `--threads` and bitnet.cpp's `-t` / `-tb`. The count is
recorded on every row, and each count is its own matched pair (8 vs 8, 24 vs 24). No
cross-thread number goes in a report, this ledger or the README. The one bitnet.cpp CPU number
we hold (25.8 tok/s decode at 8 threads, a 2026-11-21 WSL smoke run) is not comparable to any
Minerva number above, all of which were taken at 24 threads.

## Brick d: the int8 head screen, and the copies around the model (2026-11-23)

Run through `codec-memory-copies`: instruments first, then each hop A/B'd.

### Instruments added

- `memstat` (Windows `K32GetProcessMemoryInfo`) at the end of `generate`: page faults, peak and
  final working set, peak private commit.
- `--features census-alloc`: a counting allocator. It reports allocations per phase (load,
  prefill + first, decode) and is off in every timed build.
- `tools/cold_ab.py`: a cold-cache load A/B. Each run gets a fresh `robocopy /J` (unbuffered)
  copy of the model, then the same copy again warm as the control.
- `tools/tier_ab.py --var X`: any env var as the A/B arm. Ids must be equal across arms, and the
  logits hash equal within an arm.

### Baseline (brick c plumbing, p64 + 64)

- Load 0.41 s.
- 1,220 MB peak private commit and ~301,600 page faults: the whole-file heap copy.
- 409,688 allocations at load. These are the GGUF metadata's vocabulary strings (128,256
  tokens + 280,147 merges) copied by `rusty_ternary::gguf::Gguf::parse`.
- Decode: 3.83 allocations a token (~6 KB). At ~100 ns each that is under 1 us of a ~27 ms
  token, priced as negligible and not chased.

### Copy 1: the model memory-mapped instead of read onto the heap (native; wasm keeps the read)

| | heap copy | map | map + prefetch hint |
|---|---:|---:|---:|
| warm load | 0.41 s | 0.04 s | 0.04-0.11 s |
| peak private commit | 1,220 MB | 33 MB | 33 MB |
| cold load + first token (`cold_ab.py`, 3 rounds, median) | 29.2 s / 32.4 s (two runs) | 34.0 s | **28.7 s** |
| warm load + first token | 2.04 s / 1.79 s | 1.79 s | **1.62 s** |

A plain map is 1.16x slower cold than the streamed read: the first token faults the file in
page by page. `PrefetchVirtualMemory` (Windows) / `madvise(WILLNEED)` (unix) right after
mapping is an async hint that fixes it, so the map now wins cold and warm. Cold numbers are
disk-bound on this drive (~43 MB/s). Logits are unchanged (`f2f2db068ec651cb`).
`MINERVA_LOAD=read` keeps the streamed read as the A/B arm.

### The screen (`src/head.rs`, `simd::i8f_matvec`)

The greedy head reads an int8 copy (328 MB, built at first use) instead of the f16 head
(657 MB), with a proven per-row bound. Only the rows it cannot rule out are recomputed in f16.
The token is the full head's by proof (the bound and its proof are in `head.rs`), and it
follows rusty_sloth's `head_q8_*` design.

- **Refuted first: an int8 activation (an integer screen).** Its bound term
  `s ‖q‖ ‖x - t·a‖` ran 1.1-4.7x the weight term: this model's hidden state has outlier
  channels (`max|x|/rms` 5.5-22.5). That left 98.8 and 38,751 survivors a step on two golden
  prompts. The `simd::i8_gemv` kernel stays in tree, tested, but unused.
- **Kept: int8 weights x f32 activation**, in the canonical 16-lane order. The products are
  exact (8 + 11 bits), so the twins are bit-identical to the oracle at every tier.

**Survivors a step** (rows recomputed exactly, of 128,256):

| prompt | avg | most |
|---|---:|---:|
| golden 0 / 1 / 2 (16 steps) | 1.81 / 33.69 / 2.94 | 7 / 157 / 7 |
| p64 / p256 / p512 / p1024 (48 steps) | 2.19 / 5.35 / 2.69 / 14.73 | 11 / 137 / 10 / 563 |

**Validation:**

- `MINERVA_HEAD=verify` runs the screen and the full head every step and errors on any
  difference: 240 steps, no difference.
- At scalar, ssse3 and avx2 the survivor sets are identical, and so are the ids.
- The conformance gate passes: 190/192, worst gap 0.0154 (`docs/conformance-d.json`). The
  full-head logits hashes are identical to brick c's, and the gate now also requires the
  screened ids to equal the full head's on every bench prompt.
- The census shows `i8f_matvec` at 100 % of its tier.

**Copy 2: the f16 head released from the working set.** Building the int8 copy read the whole
f16 head, and those pages stayed resident: the working set went **1,207 -> 1,538 MB**. The
build now runs chunk by chunk and drops each chunk's f16 pages (`VirtualUnlock` /
`madvise(DONTNEED)`; a later read re-faults them from the file cache):

| p64 + 64 | full head | screen, no release | screen + release |
|---|---:|---:|---:|
| working set | 1,207 MB | 1,538 MB | **884 MB** (peak 897) |
| peak private commit | 33 MB | 352 MB | 351 MB (the int8 copy) |

This is rusty-sloth-05's predicted phone budget: 521 MB of codes + 328 MB int8 + KV. The
screen adds no allocation a token (decode stays at 3.83).

**Speed** (`tier_ab.py --var MINERVA_HEAD --arms full,screen,screen`, 12 rotated rounds, 24
threads, p64 + 64; ids equal across arms). The box was loaded by another session's build
(~11 % CPU):

| head | decode tok/s median (min-max) | vs full |
|---|---|---|
| full | 31.30 (25.72-36.44) | 1.000x |
| screen | 37.71 (29.00-41.31) | **1.168x, 12/12, z = +3.46** |
| null (screen vs screen) | median 1.001, range 0.889-1.251 | |

The screen won every round by 1.10-1.31x. The byte arithmetic (1,179.5 -> 851.6 MB a token)
predicts at most 1.385x if decode were purely bandwidth-bound; at ~63 % of bandwidth, 1.17x
is consistent. The screen's build costs a one-time ~0.2 s on the first token.

### NEON executed (closes brick c's open item)

On a Pixel 9 Pro XL (Android 17, aarch64, adb over Wi-Fi), with the test binary built with the
NDK r27c linker: tiers `["scalar", "neon"]`, all 24 oracle tests pass, and **the exhaustive
f32->f16 sweep (all 2^32) passes on NEON** (67.3 s). The binary was deleted from the phone
afterwards. The CPU also has asimddp, i8mm and sve2, so dot-product and i8mm tiers are now
testable.

### Open

- The 409,688 load-time allocations are in `rusty_ternary`'s metadata parse (the vocabulary).
  Minerva does not need the vocabulary; a parse mode that skips arrays is a rusty_ternary
  question for rusty-sloth.
- A sidecar int8 head file would skip the build and never touch the f16 head at all. That is
  an export-format question (keyed to the model's hash).
- Prefill is still one GEMV a token (GEMM via `gemm_codes` is next).

## Brick g, first matched-thread race vs bitnet.cpp (2026-11-23, run by rusty-sloth)

Minerva FFai `57d74f4` (brick d: head screen on, avx2), built from a `git archive`, against
bitnet.cpp `765741d` (MSVC `llama-server`, `-t T -tb T`):

- Both engines run `ours-i2_s.gguf` with exact prompt ids, greedy, `ignore_eos`; token counts
  are asserted.
- ABBA rounds; each engine is timed on its own clock.
- A background process from another session held ~14 % of the CPU throughout.
- Harness: `F:\coding\bitnet-bench\harness\cpu_race.py`. Logs: `results\cpu_race*.log`. JSON:
  `results\cpu_race_8_24.json`, `results\cpu_race_12_16.json`.

**Equal threads on every row** (the standing rule above):

| threads | prompt -> new | decode Minerva | decode bitnet.cpp | ratio | prefill Minerva | prefill bitnet.cpp |
|---|---|---:|---:|---|---:|---:|
| 8 vs 8 | 64 -> 128 | 45.15 tok/s | 40.46 tok/s | **1.118x, 8/8** | 1,270 ms | 368 ms |
| 8 vs 8 | 512 -> 64 | 43.41 | 38.49 | **1.130x, 8/8** | 8,650 ms | 3,145 ms |
| 12 vs 12 | 64 -> 128 | 41.51 | 35.53 | 1.168x, 5/6 | 1,465 ms | 318 ms |
| 12 vs 12 | 512 -> 64 | 40.75 | 34.22 | 1.186x, 5/6 | 9,480 ms | 2,829 ms |

- **Not admissible; re-run on a quiet box.** 24 vs 24: bitnet.cpp collapsed to 2.15 / 4.07
  tok/s (its spin barrier fails at threads = logical cores under background load); Minerva
  read 52.3 / 49.0. 16 vs 16: Minerva itself swung 20.6 vs 46.0 between rows, so the box was
  noisy.
- **Decode leads by 12-19 % at matched 8 and 12 threads**, most of it the int8 head screen.
- **Prefill trails 3-4x**, as expected: Minerva still runs one GEMV a token. GEMM prefill is
  the next brick.
- Both engines are fastest at 8 threads (the P-cores; bandwidth is the limit).
- The README row waits for a quiet-box re-run with 16 and 24 threads included.

## Next brick (plan): GEMM prefill, from rusty-sloth's prefill audit of 2011d91 (2026-11-23)

The audit was read-only, through the codec-vectorize-kernel and codec-memory-copies lenses.
Ranked inputs:

1. **Root cause, memory traffic:** prefill runs one GEMV per prompt token, so every token
   streams all 521 MB of codes. p512 reads 512 x 521 MB = 267 GB, which at 31-40 GB/s (8
   threads) is 6.7-8.6 s against the measured 8.65 s, so the arithmetic closes. It is the
   front-drain pattern: the same bytes moved N times. bitnet.cpp reads the weights once a
   batch and is compute-bound.
2. **GEMM shape:**
   - Unpack each code vector once and apply it to a tile of tokens.
   - The token tile's activations stay L1-resident (8 tokens x 2,560 B; `down` 6,912 B).
   - Register-tile 2-4 rows x 4 tokens of i32. Loops: rows outer, then k blocks, then tokens.
   - Gate: one token's result must be bit-identical to `gemv_codes`, `assert_eq!` against
     `rusty_ternary::kernel::gemm_codes` (pinned, 03ebc97).
3. **Missing ISA rungs:**
   - This CPU has **AVX-VNNI** (confirmed locally: `avxvnni=true`, no AVX-512).
     `_mm256_dpbusd_avx_epi32` fuses maddubs + madd + add with no i16 step.
   - The Pixel has `asimddp` (sdot) and `i8mm`.
   - New tiers `Avx2Vnni` and `NeonDot`, each `assert_eq!` to the oracle. Decode is
     memory-bound and will barely move; compute-bound prefill should.
   - **Correction to the audit's i16 budget:** the oracle and the twins accept the invalid
     code 3 (it contributes 2a), so a block can reach 4 x 768 = 3,072 per i16 lane. That
     allows at most 10 blocks in i16 before widening, not 15.
4. **Fork-join tax:**
   - ~240 rayon regions a token (7 projections x 30 layers + attention). At 5-15 us each that
     is 1-4 ms of a ~22 ms decode token.
   - GEMM prefill pays it once a batch.
   - For decode, fuse q/k/v and gate/up into one region each (they share one activation).
     Measure it with a par-region census, before and after.
5. **Prefill attention** is O(n²) per-position mat-vecs (p512 ~1e10 f16 MACs). A tiled causal
   kernel (query tile x key blocks, online softmax) follows once the GEMM lands.

Bounds: p512 is 1.2 T MACs, and ideal AVX-VNNI on 8 P-cores is ~2 T MAC/s, so the compute floor
is ~0.6 s. bitnet.cpp does 3.1 s at 8 threads, so beating it on prefill is realistic. Every
measurement runs at equal threads, on p64 and p512.

## Brick e0: code 3 refused at load (2026-11-23)

`rusty_ternary::i2s::check_codes` (pin e68e203) runs once per projection in `Model::load`. A
tensor holding code 3 (not an I2_S weight; no writer emits it) is a load error naming the
tensor, so every kernel may take `code <= 2` as given: the i16 budget is 15 blocks, and the
VNNI / sdot range proofs are tighter.

- Both GGUFs load: ours, and **Microsoft's `ggml-model-i2_s.gguf` (their converter's output)
  holds no code 3**.
- `model::tests::a_tensor_holding_code_3_is_refused` plants `0xff` in `blk.0.attn_q` and
  expects the error.
- **Cost:** warm load 0.04 -> 0.27 s, but load + first token is unchanged (1.61-1.72 s, against
  1.62 s): the scan pages the codes in at load instead of at the first token. Cold
  (`cold_ab.py`, 3 rounds) load + first token is 28.75 s, against 28.7 s.

## Brick e1: GEMM prefill (2026-11-23)

`Session::prefill` runs a prompt in batches of up to 128 tokens.

- Every per-token operation (norms, quantisation, RoPE, KV store, attention, residual) is the
  same as in `step`.
- The seven projections per layer are batched through `simd::ternary_gemm` / `kernel::gemm_par`.
  The output is `[rows, tokens]`; threads split the rows; each code block is unpacked once and
  applied to a tile of 8 tokens held in 8 i32 accumulators (single-token tail tiles).
- So the weights stream once per batch, not once a token. The audit's root cause: p512 used to
  read 267 GB.
- `MINERVA_PREFILL=step` keeps the per-token path as the A/B arm and the oracle.

**Gates:**

- `ternary_gemm` is `assert_eq!` to `rusty_ternary::kernel::gemm_codes` at every tier, x86
  native and wasm simd128 under wasmtime; the cases include ±128/127 saturation and token
  counts with tails.
- End to end, full head, `MINERVA_PREFILL=step` vs the GEMM: the logits hashes are identical on
  p64 (`1c2856725481ae14`) and p512 (`b94e537d411a31eb`).
- Conformance gate PASS, 190/192, worst gap 0.0154 (`docs/conformance-e1.json`). Every bench
  prompt's logits hash is identical to bricks c and d.
- Clippy `-D warnings` on x86_64, aarch64, wasm32 ± simd128.

**Speed** (`tier_ab.py --var MINERVA_PREFILL --arms step,gemm,gemm --threads T --new 8`, 6
rotated rounds; "prefill" is the `prefill + first` median, so it includes the first decoded
token and the one-time ~0.2 s screen build):

| threads | prompt | step | GEMM | speedup | null arm (gemm#2 / gemm), on decode |
|---|---|---:|---:|---:|---|
| 8 | p64 | 1.40 s | 0.56 s | **2.5x** | 0.958 (0.84-1.06) |
| 8 | p512 | 11.18 s | 3.37 s | **3.3x** | 0.992 (0.62-1.04) |
| 12 | p64 | 1.47 s | 0.55 s | **2.7x** | 1.006 (0.93-1.14) |
| 12 | p512 | 10.74 s | 3.27 s | **3.3x** | 0.995 (0.91-1.11) |

Decode is unchanged (within the null arm), as it must be. At 24 threads (gate run), p1024
prefill went from 22.2 s to 6.2 s. For orientation only (not a matched race):
bitnet.cpp's p512 was 3.15 s at 8 threads in rusty-sloth's run, so Minerva is now near parity
before the VNNI tier.

Next: brick e2, the `Avx2Vnni` and `NeonDot` GEMM tiers. Prefill is now compute-bound, so a
fused `vpdpbusd` / `sdot` should show.

## Brick e2: AVX-VNNI and NEON-dot tiers (2026-11-23)

New tiers `avx2vnni` (AVX2 + F16C + AVX-VNNI; `Isa::best` on this CPU) and `neondot` (NEON +
dotprod).

- They change only the ternary GEMV and GEMM. `vpdpbusd` / `sdot` take codes <= 2 times the
  activations straight into i32, in place of `pmaddubsw` + `paddw` + `pmaddwd` + `paddd`.
- Every other kernel at these tiers runs its `avx2` / `neon` twin.
- `sdot` is a one-instruction inline `asm!` helper, because `vdotq_s32` is unstable
  (`stdarch_neon_dotprod`) on the pinned 1.95 toolchain. It is register-only:
  `pure, nomem, nostack`.
- **Census bug caught by its own test:** cells are written by the tier's discriminant but were
  read by its position in `TIERS`, which now differ. `snapshot` now indexes by discriminant.

**Gates:**

- `avx2vnni` passes every oracle test (`assert_eq!` on the GEMV and GEMM).
- Conformance PASS 190/192 at `avx2vnni` (`docs/conformance-e2.json`), screen ids == full ids.
- Clippy `-D warnings` on x86_64, aarch64, wasm32 ± simd128.
- **NEON-dot: compiled for aarch64; its oracle run on the Pixel is pending** (wireless adb
  re-pair).

**Speed:** `tier_ab.py --var MINERVA_ISA --arms avx2,avx2vnni,avx2vnni --threads T --new 32`, 6
rotated rounds, paired prefill speedups. Work parity OK.

| threads | prompt | prefill+first speedup, each vnni arm (median) | wins | decode |
|---|---|---|---|---|
| 8 | p64 | 1.055x / 1.046x | 4/6, 4/6 | 0.95-0.98x (within null 0.88-1.05) |
| 8 | p512 | 1.097x / 1.200x | 6/6, 4/6 | 1.09x |
| 12 | p64 | 1.081x / 1.066x | 4/6, 5/6 | 1.04-1.05x |
| 12 | p512 | 1.204x / 1.087x | 4/6, 6/6 | 1.03-1.05x |

At p512 the gain is real: pooled, 10/12 at both thread counts (z ≈ +2.3). Kept. It is far
short of ~2x. rusty-sloth's reading (V1 below): the GEMM tile is 1 row x 8 tokens, so every
activation load feeds one dot instruction and the loop is load-bound. Attention and the
per-token glue now also share p512's cost.

## Instruction-reduction veins (rusty-sloth's read of d25f871, 2026-11-23): the work queue after e2

Instrument: static instructions per innermost loop / MACs retired, from the LINKED binary's
asm (`cargo rustc --release --bin minerva -- --emit asm`; ThinLTO makes library asm lie), with
census call counts as work-parity anchors. Intel SDE gives dynamic counts if needed.

- **V1:** tile R rows x T tokens in the GEMM (each activation load feeds R dots, each
  unpacked code T); sweep R x T within 16 ymm.
- **V2:** with codes <= 2, keep AVX2 / SSSE3 per-block sums in i16 for up to 15 blocks, then
  widen once. Applies to the decode GEMV too.
- **V3:** grouped-query attention widens each f16 K and V element once per query head (4 per
  KV head). Process a group's 4 heads per KV row: one widen, 4 chains, each in its current
  order, so bit-identical.
- **V4:** `census::add` runs on every dispatcher call. Gate it on a relaxed `AtomicBool` set
  by `--census`.
- **V5:** loop-invariant broadcasts and recomputed constants in the per-token glue (the asm
  INVARIANT / HALF-STORE greps).

Each gets an `assert_eq!` twin gate plus an unchanged logits hash, or a recorded refutation.
Order: e2's NeonDot run, V1, V3, V2, V4, then bricks 3-4.

## Brick e1b: prefill's transpose + finish, parallel and fused (2026-11-23)

**Instrument first:** `src/prof.rs` adds coarse prefill stage timers (`MINERVA_PROF=1`; a few
scopes a layer phase, one relaxed load when off). At `c7ae4c1`, p512 prefill split as:

| threads | total | GEMM | transpose + finish | attention | other |
|---|---:|---:|---:|---:|---:|
| 8 | 3.97 s | 58.6 % | **25.8 %** | 6.5 % | 9.2 % |
| 12 | 3.11 s | 55.7 % | **28.5 %** | 5.7 % | 10.1 % |

A quarter of prefill was the serial gather of each token's column out of the GEMM's
`[rows, tokens]` sums (512-byte strides), plus a separate `finish` pass. It is now parallel over
blocks of 16 tokens (each 64-byte line of the sums read once), with the scalar `finish` (the
oracle, which the twins equal bit for bit) fused into the same pass. The logits hashes are
unchanged (p64 `1c2856725481ae14`, p512 `b94e537d411a31eb`). The stage went 1.03 s -> 0.16 s at
8 threads.

**Paired A/B, old binary (`c7ae4c1`) vs new** (`tier_ab.py --var EXE`, 6 rotated rounds, two
new arms, `--new 8`), prefill speedups:

| threads | p512 | p64 |
|---|---|---|
| 8 | **1.329x / 1.204x, 6/6 + 6/6** | 1.019x / 1.034x |
| 12 | **1.218x / 1.239x, 6/6 + 6/6** | 1.043x / 1.033x |

After: p512 at 8 threads is about 2.3 s (GEMM 71 %, finish 7 %, attention 8 %, other 14 %).
The serial per-token loops in "other" are next.

## Copies audit of the kernels and glue (codec-memory-copies, 2026-11-23)

A grep of every allocation / collect / fill / copy in `engine`, `head`, `kernel`, `numerics`,
`model` and `simd/*`, each classified by how often it fires. **No front-drain and no
runtime-length memcpy on a hot path; no copy inside any kernel** (kernel tails and lane
stores are at most 16 elements).

- **Decode, 3.87 allocations a token: rayon's injector, now fixed.** Traced with the census
  allocator's new backtrace mode (`MINERVA_ALLOC_TRACE`; it now also hooks `alloc_zeroed`):
  `crossbeam_deque::Injector::push` from `kernel::gemv_par` and `Session::forward`, a 1,520 B
  block every 63 jobs. Every one of the ~240 parallel regions a token was injected from the
  main thread, a non-pool thread. `generate` now runs on a pool worker (`rayon::scope`), so
  nested regions fork through the worker's own deque. `MINERVA_POOL=outside` keeps the old arm.
  - Counter: **3.93 -> 0.07 allocations a decode token**. Logits hash unchanged.
  - Clock (`tier_ab.py --var MINERVA_POOL`, 6 rounds, two inside arms), decode p64 + 64:
    **8T 1.224x / 1.187x (12/12)**, **12T 1.291x / 1.251x (12/12)**; null arms 0.97 / 0.98.
    p512 prefill: inside the noisy null arm (0.76-1.71), no claim.
- Prefill batch buffers: 14 fresh zeroed vectors a 128-token batch (~31 MB), +25,808 page
  faults at p512 (415,385 against the step path's 389,577). Measured next.
- `sums` collect in `bitlinear_batch` (210 a batch, ~0.5 KB each) and the session buffers
  (once): negligible.
- `i8f_matvec_scalar` builds a temp row per call: scalar tier only.
- `read_chunked` zero-fill then read: wasm / `MINERVA_LOAD=read` only. It stays: wasmtime
  refuses one 1.2 GB read, which the chunks work around.
- The "3.87 a token" reported in brick d was this injector, not the engine's own buffers;
  the engine allocates nothing a decode step.
- **Refuted (reverted): reusing the prefill batch buffers across batches.** Moving the 14
  buffers into the session (grown, never re-zeroed; every cell written before it is read) is
  bit-identical, including p1024's shorter final batch (logits hash `566af54a37be0f53` both
  ways), and deterministically cuts page faults by **53,963 at p1024 and 19,245 at p512**.
  The clock did not move: old (`c2dd617`) vs new, 6 rounds, two arms:
  - 8T p512: 0.958x / 0.954x (4/12)
  - 12T p512: 1.020x / 1.083x (7/12)
  - 8T p1024: 1.029x / 1.035x (9/12)

  That is within the null arms (0.69-1.42, 0.88-1.11, 0.82-1.08), consistent with the
  estimated 1-2 %. It also cost +27 MB of peak working set (916 -> 943 MB), which counts
  against the phone budget. Reverted because it is not measurably faster, not because it is
  measurably worse.

## Norm / quantise chain: deterministic instruction wins W1-W3 (2026-11-23)

Instrument: `F:\coding\bitnet-bench\harness\asm_count.py` (rusty-sloth's), static instructions
per innermost loop trip from the LINKED binary's asm (AVX2), divided by the elements a trip
retires. The chain runs 4 times a layer per token (2,560 + 2,560 + 2,560 + 6,912 = 14,592
elements).

| step | baseline | after | commit |
|---|---:|---:|---|
| `act_sum` pass (inlined, SSE2 baseline) | 13 per 8 = 1.625/elem | removed (fused into the quantiser: 4 `vpaddd` per 32) | W1 `42d4dba` |
| quantise loop | 36 per 32 = 1.125 | 40 (W1) -> **28 per 32 = 0.875** | W2 `829f76a` |
| rms_norm sum of squares | 16 per 16 = 1.00 | **12 per 16 = 0.75** | W3 `bbff372` |
| rms_norm apply | 15 per 32 = 0.47 | unchanged | |
| max \|x\| | 12 per 32 = 0.375 | unchanged | |
| **chain total** | **4.60/elem** | **2.47/elem (-46 %)** | |

- **W1:** the quantiser holds each code as an i32 lane in -128..=127 (so the saturating pack
  stores exactly it) and adds the lanes as it goes. `act_quant_sum` returns (scale, sum).
  `Session::forward`'s static size went 1,138 -> 970.
- **W2:** `|x·s| <= 127·(1+2u)` because `s = 127/max|x|` (or `|x| < 1e-5` under the floor), so
  the oracle's clamp never acts. `cvtps2dq` / `vcvtnq` round ties-to-even, so the separate round
  goes too. The NaN mask stays.
- **W3:** an f32 squared is exact in f64 (48 of 53 bits), so `fma(v, v, acc)` == `acc + v·v` bit
  for bit (`vfmadd231pd`, NEON `vfmaq_f64`). The avx2 tier now requires `fma`.
- **Trap seen on the way:** a 188-insn "loop" in `act_quant_avx2` is the scalar tail
  (`quant_one`), auto-vectorised 32 wide by LLVM with a scalarised conversion. It never runs
  (the tail is < 32 elements); it is static size only. Read the op mix before trusting a loop's
  identity.
- **Gates for each:** `assert_eq!` vs the oracles at every tier (x86 native, wasm simd128
  under wasmtime), the fused sum against `act_sum`; logits hashes unchanged (p64
  `1c2856725481ae14`, p512 `b94e537d411a31eb`); clippy `-D warnings` on 4 builds.
- **Clock:** pre-W1 `1325dac` vs W1-W3, 6 rounds, two arms. Prefill p512: 8T 1.04x / 1.03x (9/12),
  12T 0.99x / 1.01x. Decode p64 + 64: 8T 0.99x / 0.95x, 12T 1.01x / 0.98x. All within the null
  arms. That matches the counter: ~0.9 M insns a token, ~0.25 % of decode, ~2-4 % of the
  serial share of prefill. The verdict is the counter (codec-measurement §15: sub-1 % bricks);
  the clock shows no regression. **Below clock resolution, not "no effect".**
- **Not built (arithmetic):** W4, fusing max|x| into the norm's apply pass. It saves 0.125/elem
  (~55k insns a token) for two new fused kernels across four tiers.
- The time lever left in this chain is not instructions but its serial execution in prefill
  (it runs token by token on one thread, the "other" ~14 %). Parallelise it across tokens.
