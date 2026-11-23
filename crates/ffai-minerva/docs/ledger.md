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
