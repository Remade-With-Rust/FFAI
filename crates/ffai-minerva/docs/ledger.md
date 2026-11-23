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
