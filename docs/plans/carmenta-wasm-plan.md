# Carmenta on WebAssembly — Plan

**Component:** Carmenta — FFai's OCR component (`ffai-carmenta`)
**Status:** **it runs.** Steps 0, 1 and 2 are landed;
`ffai-carmenta-wasm` loads weights from bytes and reads an image in a wasm
runtime, producing the same text as the native build. Step 3 (the upstream
candle `+simd128` defect) is now the largest outstanding lever, and Step 4
(threads) is unchanged.

*Original status, kept: does not compile for `wasm32-unknown-unknown`; three
Cargo.toml lines fix that with zero source changes; nothing runs, because three
runtime traps sit on the default path.*

**Prime directive inherited from the Diana plan:** no claim without a number.
Every figure below was measured on this machine, read out of the code, or
carried from a cited campaign section. Figures that are **not** measured are
labelled as such.

---

## 0. Where this actually stands

| | native | wasm32 |
|---|---|---|
| compiles | yes | **no** — 3 Cargo.toml lines away |
| compiles C | yes (`onig_sys`) | no, once `onig` is dropped |
| SIMD | AVX2 (8×f32) | **none today**, and see §2 |
| threads | 3 nested rayon levels | **none**, and see §3 |
| weights | `mmap` | **no path** — 2 sites, no bytes constructor |
| conv backbone | `conv3x3_avx2`, **1.65×** candle | `conv3x3_scalar` — now measured, §7.1 |

### The compile side is solved and cheap

Reproduced today, in this order — each line is required and none is enough
alone:

1. `getrandom = { version = "0.3", features = ["wasm_js"] }`, wasm32-gated.
   `.cargo/config.toml` already supplies the `--cfg getrandom_backend` half;
   the crate errors unless it gets both.
2. `tokenizers` with `unstable_wasm` instead of `onig`. `onig` is Oniguruma, a
   C regex engine, and there is no `cc` for wasm32. `unstable_wasm` swaps in
   fancy-regex, which is pure Rust. candle already excludes `tokenizers`
   upstream on wasm32, so this is our own dependency and nobody else's.
3. **A `fetch` feature, which Carmenta does not have.** It declares
   `ffai-models = { workspace = true }`, defaults on, so `hf-hub` 1.0 arrives
   and calls `reqwest::blocking` — a module that does not exist on wasm32.
   **24 × `E0433`.** Mercury and Diana both opted out long ago; Carmenta is
   the only engine that never did.

With those three, `cargo check -p ffai-carmenta --target wasm32-unknown-unknown
--no-default-features` **finishes clean, 4 warnings**, and the native build is
unchanged — same graph, `cargo check` exit 0.

Item 3 is worth landing on its own merits regardless of wasm: today Carmenta
ships the HTTP/TLS stack with no way for an embedder to remove it.

### Three runtime traps, all on the default path

None of these is reachable by configuration in a browser, because **the
switches are environment variables and a browser has no environment.**
`std::env::var` returns `Err` on wasm32, so every one of these defaults *on*:

| trap | site | why it fires |
|---|---|---|
| `Instant::now()` | [`crnn.rs:187`](../../crates/ffai-carmenta/src/crnn.rs), `:205`, `:210` | three per CRNN forward, i.e. **per text line**. Unconditional — `FFAI_PROFILE` gates the *reporting*, not the clock. Panics on wasm32. |
| `rayon` | [`engine.rs:768`](../../crates/ffai-carmenta/src/engine.rs) | `FFAI_REC_SERIAL` is absent → `lines.len() >= 3` → `par_iter` → pool → `std::thread::spawn`. **Any page with 3+ lines.** |
| `mmap` weights | [`engine.rs:193`](../../crates/ffai-carmenta/src/engine.rs), [`svtr.rs:462`](../../crates/ffai-carmenta/src/svtr.rs) | `VarBuilder::from_mmaped_safetensors`. No filesystem, no mmap. |

A fourth is confined to one path: `std::thread::spawn` at
[`live.rs:386`](../../crates/ffai-carmenta/src/live.rs), with its `JoinHandle`
at `:130`. The LIVE reader is architecturally threaded — this is not a shim,
and the honest scope is **one-shot OCR on wasm, live reader stays native.**

And one latent design bug that only wasm exposes: `FFAI_CONV3X3=0` is the
opt-out for our own conv kernel, so **in a browser the kernel cannot be turned
off** — see §1, where that matters more than it sounds.

---

## 1. What wasm costs Carmenta — the shape is known, the number is not

> **Superseded in part by §7.2 — read that first.** The ranking below is built
> on a profile taken from DOCUMENT PAGES. On browser-sized input the shares
> invert: detection is 88.8 % and the recognizer this section is about is
> 10.6 %. The reasoning is kept because the kernel analysis is still correct
> for what it covers, and because the mistake — generalising one corpus class's
> profile to a different workload — is worth leaving visible.

From [§8.100](../Carmenta-mission-plan.md) and §8.97, measured natively:

* **94 % of runtime is `rec_fwd`.**
* Inside it: `.rec_cnn` **84.2 %** (1126 ms/call), `.rec_rnn` 11.6 %,
  `.rec_head` 0.0 %.
* The backbone runs at **29.7 GFLOP/s against ~112 single-core AVX2 peak —
  26 %.** 166 GFLOP per page.

So Carmenta's wasm performance is, to within a rounding error, **the
performance of one 3×3 convolution kernel**. Everything else is noise.

That kernel is [`conv3x3.rs`](../../crates/ffai-carmenta/src/conv3x3.rs)
(§8.101) — a non-materialising direct conv that never builds candle's im2col
buffer. Its own measured table:

| form | vs candle |
|---|---:|
| direct, `Vec` per output tile | 0.41× |
| no allocation, CO-blocked, per-row AXPY | 0.57× |
| register tile indexed by a runtime channel counter | 0.21× |
| padded common stride, one long AXPY per tap | 0.67× |
| **AVX2, 4 channels × 24 columns held in ymm** | **1.65×** |

**Every scalar form of the direct conv measured *below* candle. Only the
vectorised one passed it.** The dispatch is `#[cfg(target_arch = "x86_64")]`
plus `is_x86_feature_detected!`, so on wasm it falls to `conv3x3_scalar` —
and, because the opt-out is an env var, it does so **with no way to fall back
to candle's im2col instead**.

The shipped scalar path's speed against candle is **not in the record**. It is
plausibly the fastest of the scalar forms and still under 1.0×. That single
unknown decides whether the wasm default should be our kernel or candle's, and
it is measurable natively in an afternoon — which is Step 0.

---

## 2. SIMD is blocked globally and **open per-function** — the key finding

The Diana plan established that `-C target-feature=+simd128` cannot be used at
all, because candle-core 0.11 does not compile with it:

```
RUSTFLAGS='-C target-feature=+simd128' cargo check -p candle-core --target wasm32-unknown-unknown
error[E0433]: cannot find type `CurrentCpuF16` in this scope
   --> candle-core-0.11.0/src/cpu/mod.rs:259:15
```

**Reproduced today, still broken.** It concluded that porting kernels (its
Step 5) must wait for the upstream fix (its Step 1).

**That conclusion does not hold for kernels we write ourselves.** The global
flag is not the only way to get SIMD out of LLVM. Probed today:

```rust
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn axpy(a: f32, x: &[f32], y: &mut [f32]) { /* core::arch::wasm32 */ }
```

built with **no** `+simd128` rustflag, `--emit asm`:

```
      6 v128.load
      3 v128.store
      3 f32x4.mul
      3 f32x4.add
      1 f32x4.splat
```

Real SIMD128, unrolled 3×, in a build where candle stays scalar. **The
function-level attribute is enough, and it routes around the upstream
defect entirely.**

Consequences, in order of importance:

* **Our conv3x3 can be vectorised on wasm today.** It is 84 % of runtime, it
  already has the scalar oracle, the `*_matches_scalar` test and the runtime
  dispatch structure. No upstream dependency, no fork, no `[patch.crates-io]`.
* **candle's kernels and `gemm` cannot.** candle already enables `gemm`'s
  `wasm-simd128-enable` feature and gemm defaults `DEFAULT_WASM_SIMD128 =
  true`, so the machinery is armed — but pulp reaches it through
  `feature_detected!("simd128")`, and the whole path needs the module built
  with the target feature. Until candle is fixed, everything *outside* our
  kernel is scalar on wasm. That is the LSTM (11.6 %), CRAFT, and the final
  2×2 valid conv.
* **LLVM auto-vectorisation of our other scalar loops stays off** for the same
  reason. That is free performance sitting behind a one-line upstream fix,
  which is why filing it is still worth an hour.

**One caveat that has no equivalent on x86:** wasm validates an entire module
ahead of time, so a `v128` instruction anywhere makes the *module* require
SIMD support — there is no `is_wasm_feature_detected!` and no per-call
dispatch. A simd128 Carmenta is a **baseline requirement**, not a runtime
upgrade. Practically that is fine (Chrome 91, Firefox 89, Safari 16.4 —
March 2023), but if a non-SIMD fallback is ever needed it is a **second
module**, not a second branch.

---

## 3. Threads: the native finding **inverts** on wasm

Natively, more rayon is the one lever this codebase has proven negative.
Three levels nest — ours over lines, candle's over im2col tiles, and `gemm`,
which candle hands `Parallelism::Rayon(num_cpus::get())` on every matmul. A
one-line band strip measured **177 ms/line under `par_iter` against 82 ms
serial**, which is why the shipped gate is `lines.len() >= 3`. Flattening it
to one level was built, measured ABBA-paired over 16 rounds — z = +1.50,
inside a 17–27 % noise floor — and reverted.

**On wasm that nesting does not exist.** candle's `default_num_threads()`
calls `num_cpus::get_physical()`, whose wasm32 branch returns a literal `1`,
so candle takes `Parallelism::None` and never enters rayon at all. Verified in
the vendored sources.

So on wasm, a line-level fan-out would be the **only** rayon level in the
process — which is precisely the arm that won natively (2.07 s → 1.65 s on a
7-line frame) when it was not fighting two layers beneath it. The thing that
is harmful natively is the clean lever in a browser.

It still needs `wasm-bindgen-rayon`, `+atomics,+bulk-memory`, and
cross-origin isolation headers from whoever serves the page — a **deployment**
constraint on the embedder, not code we can write for them. If the target
cannot set COOP/COEP, single-threaded is the honest answer and the number to
publish is the single-threaded one.

---

## 4. The ports, in dependency order

### Step 0 — measure before building anything *(half a day)*

Diana simulated its two missing capabilities **natively** so the cost was real
rather than projected (`FFAI_DIANA_NO_AVX2=1`, `FFAI_DIANA_THREADS=1`,
min-of-80). Carmenta cannot do this yet: there is no switch that forces
`conv3x3_scalar` while keeping our kernel selected.

* **Build:** `FFAI_CONV3X3=scalar` alongside the existing `0`, forcing the
  scalar arm on x86.
* **Measure:** three arms on the same page set — candle im2col
  (`FFAI_CONV3X3=0`), ours scalar, ours AVX2 — plus `FFAI_REC_SERIAL=1`
  throughout to model the single-threaded browser.
* **Decides:** whether the wasm default is our kernel or candle's, and what
  the simd128 twin has to beat. If scalar-ours is below candle, the *first*
  wasm change is a `cfg` making candle the default there, and that is a
  one-line win available before any kernel work.
* **Gate:** the noise floor is established **before** the A/B, not after a
  confusing one — §8.100's D6 built a whole diagnosis on a sweep that ran
  while other jobs were live.

### Step 1 — make it run at all *(1–2 days)*

Nothing below matters until this lands, and none of it is speed work.

* The three Cargo.toml lines from §0 (already proven).
* **Bytes constructors** for the two `mmap` sites, mirroring Diana's
  `Yolo26::from_bytes` / `from_buffered_safetensors`. Gate the same way Diana
  did: **byte-identical output** from the bytes path and the filesystem path.
  Serves embedded targets too.
* **`Instant`** — a `cfg`-gated clock. Unlike Mercury's adaptive dtype probe
  this is pure instrumentation, so wasm can return a zero-cost stub with no
  behaviour change at all.
* **A `par` shim**, copied from [`ffai-diana/src/par.rs`](../../crates/ffai-diana/src/par.rs).
  Two call sites, and one of them is in `live.rs`. Structural, not guarded:
  on wasm the parallel methods *are* the serial ones, so an unguarded call is
  serial rather than a runtime panic.
* **Exclude the live reader** from the wasm build — `thread::spawn` is not
  shimmable and the one-shot path never touches it.
* **Gate:** an image in, text out, in a wasm runtime. First real number.

### Step 2 — the simd128 conv3x3 twin *(the big one)*

84 % of runtime, unblocked by §2, and the file is already shaped for it.

* Same discipline as the AVX2 twin: scalar stays the oracle,
  `simd128_matches_scalar` beside `avx2_matches_scalar`, f64 reference for
  correctness rather than agreement-with-candle.
* **The tile geometry must be re-swept, not translated.** AVX2's 4×24 shape
  was chosen because twelve accumulators + three inputs + one broadcast is
  sixteen ymm registers *exactly*, lifting the ratio to 12 FMAs per 7 memory
  ops. SIMD128 is 4 lanes, not 8, and wasm exposes no architectural register
  count — the winning shape is an empirical question with a different answer.
* Carry across unchanged: the per-tile `Vec` costs more than the arithmetic; a
  runtime-indexed `dp[i].add(x)` is a pointer gather that vetoes vectorisation
  outright (0.21×); an accumulator living in memory pays a load and a store
  per FMA.
* **Gate:** beats whatever Step 0 named as the wasm baseline, measured in the
  wasm runtime, not simulated.

### Step 3 — file the candle issue *(an hour, not ours to fix)*

Either drop `simd128` from the `cfg(any(...))` on the f16/bf16 paths, or
implement `CurrentCpuF16`/`CurrentCpuBF16` in `simd128.rs`. It unblocks the
global flag, which brings `gemm`'s simd128 kernels and LLVM auto-vectorisation
of every other loop — the LSTM's 11.6 %, CRAFT, preprocessing. Diana wants the
identical fix, so it is one issue for two components.

### Step 4 — threads, if the embedder can set the headers

`wasm-bindgen-rayon` over lines, one level, per §3. Largest remaining lever
after the kernel, gated entirely on a deployment constraint.
**Gate:** cores-busy > 1.0 measured in the runtime, not assumed from flags.

### Step 5 — everything else, only if Step 0 says it is there

Detector choice (mobile-det gives line-level boxes and skips the per-word
path), input resolution, int8 recognition. All of it is competing for the 6 %
that is not `rec_fwd`, so Amdahl caps the lot at ~1.06× — **unless** Steps 2–4
have already moved the backbone enough to change the profile, in which case
re-profile before choosing.

---

## 5. What must not be claimed

* **No wasm performance number until something runs in a wasm runtime.**
  Everything in §1 is native measurement of a native binary.
* **Diana's 2.9× native-to-wasm ratio does not transfer.** It is a different
  network with a different shape, and Diana measured it by simulating missing
  capabilities on native hardware, not by running wasm. Carmenta's ratio is
  unknown and will be worse in at least one respect: Diana's kernels lose
  AVX2 and fall back to scalar loops of the same algorithm, whereas Carmenta
  falls off a kernel that beat candle onto one that measured below it.
* **Nothing about the OCR accuracy gates.** They are ISA-independent and
  unaffected — but a wasm build has not run them, so it has not passed them.
* **"Compiles for wasm" is not "works in a browser."** Three panics sit on the
  default path, all of them defaulted *on* by the absence of an environment.

---

## 6. Order of work

1. **Step 0** — the scalar switch and the three-arm native measurement. Cheap,
   and it may hand back a one-line wasm win before any kernel exists.
2. **Step 1** — make it run. The `fetch` opt-out inside it is worth landing on
   its own merits today.
3. **Step 3** — file the candle issue early; it costs an hour and may land
   without us while Step 2 proceeds.
4. **Step 2** — the simd128 conv twin. The largest lever we control.
5. **Step 4** — threads, if the embedder can isolate.
6. **Step 5** — re-profile, then decide.

---

## 7. What execution found — results, in the order they arrived

Written after running the plan. Every number here was taken interleaved on a
machine with other tenants at 100 % load, min-of-rounds; §8.100's D6 trap
(2.4× spread on identical configurations minutes apart) showed up again in the
first pass and is why nothing single-shot is quoted.

### 7.1 Step 0's question had a clear answer, and it was the predicted one

`FFAI_CONV3X3=scalar` was added so the arm a wasm build takes could be
measured on hardware that has AVX2. Interleaved min-of-4, 620×200 capture,
`FFAI_REC_SERIAL=1`, `craft-crnn`:

| arm | | vs shipped |
|---|---:|---:|
| ours-avx2 (shipped) | 3040 ms | 1.00× |
| **ours-scalar (what wasm takes)** | **5282 ms** | **1.74×** |
| candle im2col | 2771 ms | 0.91× |

The scalar arm is **1.91× candle's**. Sending wasm to candle was one `cfg` and
was the largest single win in the browser build — exactly the one-line win §4
predicted might exist. (`ours-avx2` vs `candle` is inside the noise at this
resolution and is **not** evidence against §8.101's 1.65×, which measured the
kernel rather than a pipeline it is a tenth of.)

### 7.2 The profile inverts on browser-sized input — the plan's model was wrong

§1 of this plan built its ranking on §8.100's "94 % in `rec_fwd`". That figure
is from **document pages**. On a 620×200 capture with 3 lines:

| stage | share | ms/call |
|---|---:|---:|
| **`det_fwd`** | **88.8 %** | 2636 |
| `rec_fwd` | 10.6 % | 105 |

CRAFT runs a fixed canvas regardless of how much text is on it, so on a small
image detection is nearly all of the work and the recognizer — where the conv
kernel lives — is a tenth of it. **The conv kernel was never the browser
lever.** `readLine` (detection skipped) is, and it is now exposed on the wasm
API for exactly this reason.

Corollary, measured: `mobiledet-crnn` is **1.71× faster end to end** than
`craft-crnn` on that capture (1704 ms vs 2910 ms) with a **4.7 MB** detector
against CRAFT's 83 MB.

### 7.3 Wiring the kernel into CRAFT: refuted, 2.48× slower

CRAFT is VGG16-BN — every block is 3×3 / stride 1 / pad 1, exactly
`conv3x3`'s preconditions — and it called candle's `forward`. Wiring the
1.65× kernel into the 88 % stage looked free. Interleaved, three rounds, it
was **2.48× slower** and lost in every round.

The kernel is **shape-specialised**: its 4×24 register tile was swept on CRNN
crops (h=64 collapsing to 3, widths 58..1357), and CRAFT runs a 1280-long
canvas at 64..512 channels, where im2col hands `gemm` a big well-blocked
matmul and wins. Kept as `FFAI_CONV3X3_DET=1` rather than deleted, because the
regime boundary is the finding.

### 7.4 Step 2 built, correct, and inside the noise

The SIMD128 twin exists and works: function-level
`#[target_feature(enable = "simd128")]` compiles with the global flag off (the
candle defect of §2 is **still** live — reproduced again) and emits real
`f32x4`. Measured in Node on the CRNN path, `readLine`, 200×40 crop:

| build | min-of-3, round 1 | round 2 |
|---|---:|---:|
| `wasm-candle-conv` | 222 ms | 222 ms |
| SIMD128 twin | 206 ms | 255 ms |

**Inside the noise.** Both produce identical text, so the twin is correct, and
it ships as the default because it is the arm that will matter once `gemm` is
vectorised — but it must not be advertised as a win.

Why it cannot be one yet: our kernel is reachable from `craft.rs` and
`crnn.rs` only. **SVTR has no convolutions at all** — it is a transformer, so
`mobiledet-svtr` goes through candle's `gemm` end to end and the twin cannot
touch it. Vectorising `gemm` is Step 3, in someone else's repository.

### 7.5 The pair to ship is not the pair the native ranking predicts

The plan and §7.2 both pointed at `mobiledet-svtr` for the browser: a 4.7 MB
detector against CRAFT's 83 MB, 1.71x faster end to end natively, and SVTR is
the document default. Measured in the target instead — Node, 200x40 crop,
`readLine` so this is purely the recognizer:

| recognizer | |
|---|---:|
| CRNN | **206 ms** |
| SVTR | **8905 ms** |

**43x.** SVTR is a transformer, so every matmul lands in candle's `gemm`,
which has no SIMD on wasm; CRNN is convolutions and an LSTM. Natively SVTR is
~1.9x slower than CRNN and worth it for accuracy — the crossing turns that
into two orders of magnitude. **A native benchmark could not have told us
this**, which is the argument for Step 1 (make it run) preceding every
performance decision rather than following them.

The pair to ship is therefore `mobiledet-crnn`: the small fast detector with
the fast recognizer, a combination neither component's own ranking suggests.
It is exposed as `Reader.mobiledetCrnn` and is the demo page's default.

Two more results from the same session:

* **One pair per module instance.** Loading `craft-crnn` and `mobiledet-svtr`
  into a single wasm memory put ~118 MB of weights in 32-bit linear memory and
  did not complete.
* **`readLine` is the browser API.** Detection is 88.8 % of native runtime on a
  620x200 capture and wasm multiplies it by roughly two orders of magnitude: a
  full `read` of that capture did not finish inside ten minutes, while
  `readLine` on a crop returns in ~200 ms. The deployable shape is "the user
  selects a region", not "OCR this page".

### 7.6 What is left

* **Step 3 is now the whole game.** Every non-conv operation on wasm is scalar
  because `-C target-feature=+simd128` cannot be turned on, and that one fact
  explains both remaining problems: detection's two-orders-of-magnitude cost
  and SVTR's 43x. It is a few lines upstream, it unblocks Diana identically,
  and nothing we can write ourselves substitutes for it.
* **Step 4 (threads)** unchanged, and still a deployment constraint on the
  embedder rather than code. Worth more here than natively, because on wasm a
  line-level fan-out would be the ONLY rayon level (§3).
* **No browser measurement exists.** Everything above is Node.
* **No corpus gate has been run on a wasm build.** Text matches native on the
  crops tested; that is not the accuracy gate and must not be quoted as one.
* **The kernel's shape envelope is unmapped.** §7.3 found the CRNN tile loses
  badly on CRAFT's shapes; nobody has swept a tile FOR detector shapes, and
  `FFAI_CONV3X3_DET=1` is the A/B already built for whoever does.
