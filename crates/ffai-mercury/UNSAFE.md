# ffai-mercury — `unsafe` inventory

Satisfies gate **H-16**. Every `unsafe` site in this crate, what it is for, and what makes
it sound. **Last audited: 2026-08-15** (survey depth — no Miri, no TSan; see H-23/H-24).

> **How this list was built, and why the obvious method undercounts — TWICE.** The
> workspace lint `unsafe_code = "warn"` enumerates 25 sites. It reports **nothing** for the
> 3 sites carrying `#[allow(unsafe_code)]`: an attribute that silences the lint also
> silences the inventory. It also reports nothing for code behind a `cfg` for a target you
> are not compiling — the 3 `#[cfg(not(target_arch = "x86_64"))]` fallback stubs were
> invisible on this x86 machine and were missed by the first pass of this file. They were
> caught later by `tools/lint_invariants.py`, which reads the SOURCE rather than the
> compiler's opinion about one target.
>
> The lesson generalises: **an inventory built from compiler output describes one
> configuration, not the crate.** `tools/lint_invariants.py` now enforces R1 (every unsafe
> site has a SAFETY comment) and R2 (every `#[allow(unsafe_code)]` file is named here) in
> CI, so neither blind spot can reopen.

**Totals**: **37 sites** — 18 blocks, 14 `unsafe fn`, 2 `unsafe impl`, 9 allow-suppressed.
**SAFETY comments**: 36, covering all 37 sites — the adjacent `unsafe impl Send`/`Sync`
pair at `decoder_kernels.rs:194-195` shares one comment. Each states the invariant *and*
what upholds it: for Class A the dominating `have_avx2()`/`have_f16c()` check, for Class B
the index arithmetic that makes the regions disjoint, for Class C the fact that the
invariant is **not** enforceable from inside this crate.

## Class A — SIMD kernels (`#[target_feature]`)

> **Drift, recorded 2026-11-07.** The 2026-11 instruction-veins campaign added `unsafe`
> sites without updating this file. By `tools/lint_invariants.py`'s pattern (one hit per
> `unsafe fn`, `unsafe impl` or `unsafe {` line), the counts moved from the pre-campaign
> tree (2635b6b) as follows: `asr/vocab_int8.rs` 4 → 17 (new blocked int8 kernels, their
> x2 pairing and two non-x86 stubs), `asr/f16_gemv.rs` 4 → 5 (the bias-fused GEMV call),
> `asr/flash_attn.rs` 22 → 23 (the strided-I/O kernel). All are Class A, guarded by the
> same `have_avx2()` / `have_f16c()` dispatch, and every one has a SAFETY comment (R1
> passes). The table and totals below predate them except where Classes D and E say
> otherwise; a full re-inventory is owed.


23 sites. `unsafe` here is a *compile-time capability* claim, not pointer arithmetic: a
`#[target_feature(enable = "avx2")]` function is `unsafe` because calling it on a CPU
without AVX2 is UB. Soundness therefore rests entirely on the runtime dispatch check.

| File | Sites | Guard |
|---|---|---|
| [`asr/flash_attn.rs`](src/asr/flash_attn.rs) | 15 — 7 blocks, 8 `fn` | `have_avx2()` / `have_f16c()`; non-x86 builds get an `unreachable!` stub |
| [`tts/decoder_kernels.rs`](src/tts/decoder_kernels.rs) | 4 — 212, 241, 1341 (blocks); 732 (`fn`) | same dispatch pattern |
| [`asr/f16_gemv.rs`](src/asr/f16_gemv.rs) | 2 — 141 (block); 153 (`fn`) | `have_f16c()` |
| non-x86 fallback stubs | 3 — one per kernel file | `unsafe fn` only to match the x86 signature; body is `unreachable!`, so no reachable code exists to be unsound |
| [`asr/vocab_int8.rs`](src/asr/vocab_int8.rs) | 2 — 202 (block); 237 (`fn`) | `have_avx2()` |

**Invariant**: every `#[target_feature]` function is called only from a path that has
already tested for that feature at runtime. Adding a call site without the check is UB on
older CPUs and will not fail any current test — the CI machine has AVX2.

Two of these blocks carry a *second* obligation beyond the feature check: `flash_attn`'s
f16 K/V views reinterpret candle's `F16` storage as `u16`, which is sound only because
`half::f16` is `repr(transparent)` over `u16` and the tensors were checked contiguous.
Both say so at the site.

## Class B — raw-pointer sharing across rayon tasks

| Site | What it does | Argument | Status |
|---|---|---|---|
| [`tts/decoder_kernels.rs:194-195`](src/tts/decoder_kernels.rs#L194) | `unsafe impl Send`/`Sync` for `SendPtr(*mut f32)` | Chunk co-ranges partition `[0, c_out)` and block t-ranges partition `[0, l_out)`, so no two tasks address the same element | Documented at the site, **still unproven by tooling** — no TSan |
| `tts/decoder_kernels.rs:212, 241` | `from_raw_parts_mut` into the output buffer per task | Disjointness as above | SAFETY comment present |
| `asr/flash_attn.rs:186` | `from_raw_parts_mut` per attention head band | Disjoint column bands | SAFETY comment present |

**Invariant**: the index arithmetic that computes each task's region must be *provably*
non-overlapping. This is the crate's highest-risk `unsafe`: a wrong bound is a data race,
not a panic — silently wrong audio, or memory corruption, with no test failing.

**Open**: no TSan run (H-24), no Miri (H-23). The disjointness argument is now written at
the site rather than only in this file, but a comment is not a proof. TSan on the TTS
decode path remains the single highest-value item here.

## Class C — memory-mapped model weights (allow-suppressed)

| Site | What it does |
|---|---|
| [`asr/model.rs:110`](src/asr/model.rs#L110) | `VarBuilder::from_mmaped_safetensors` |
| [`asr/model.rs:118`](src/asr/model.rs#L118) | same, second mapping of the same blob |
| [`asr/aligner.rs:86`](src/asr/aligner.rs#L86) | same, for the alignment model |

**Invariant**: the mapped file must not change for the lifetime of the mapping. This is
**not enforceable from inside this crate** — any process with write access to the model
cache can violate it, and the result is immediate UB rather than a parse error.

This is R-001 in the audit and the primary path in the
[threat model](docs/threat-model.md#4-the-highest-value-attack-path). Deployers must treat
the model cache directory as security-relevant. The code-side mitigation is to assert the
upstream hash verification at this boundary instead of assuming it.

## Class D — OS thread-affinity FFI (allow-suppressed)

| Site | What it does |
|---|---|
| [`gemm_pin.rs`](src/gemm_pin.rs) Windows block | `GetProcessAffinityMask` + `SetThreadAffinityMask` on the current thread |
| [`gemm_pin.rs`](src/gemm_pin.rs) Linux block | `sched_getaffinity` + `sched_setaffinity` on the current thread |

Both pin one short-lived helper thread to the lowest allowed logical CPU, so gemm's
once-per-process CPUID cache detection reads the same core on every run (see the module
docs for the hybrid-CPU bug this fixes). **Invariant**: the handles are the OS
pseudo-handles for the calling process and thread (Windows) or pid 0 (Linux), and every
out-pointer is a live, aligned local of the size passed. Each call's failure return
leaves the thread unpinned, which is a valid state.

## Class E — write-before-read buffers (allow-suppressed)

| Site | What it does |
|---|---|
| [`buf.rs`](src/buf.rs) `filled_f32`, 2 blocks | `&mut [f32]` over a `Vec`'s spare capacity, then `set_len` |
| [`buf.rs`](src/buf.rs) `filled_f16`, 2 blocks | same for `half::f16` |

Added by the 2026-11 instruction-veins campaign and missed from this file until
`tools/lint_invariants.py` R2 flagged it. **Invariant**: the caller's closure writes every
element before `set_len` publishes the length. The element types have no invalid bit
patterns and no drop glue, so nothing is dropped on unwind. **Caveat**: forming a
`&mut [T]` over uninitialized memory is the pattern `ffai_core::fastops::ElemOp` already
uses, but it is not sound under a strict reading of the reference rules; a
`&mut [MaybeUninit<T>]` signature would be. A closure that skips an element would also
make the output depend on stale heap contents, which breaks determinism without
crashing. The identity gate is the check for that.

## Rules for changing anything here

1. Every new `unsafe` gets a `// SAFETY:` comment naming the invariant *and* what upholds
   it. "Coverage is 28/28" is a gate (H-16) — a new site without a comment breaks it.
2. Every new site is added to this file in the same commit, including `#[allow]`-ed ones.
3. Class A: prove the runtime feature check dominates the call site.
4. Class B: state the disjointness arithmetic explicitly; TSan before merge.
5. Class C: do not add another mapping without revisiting R-001.
6. Class D: affinity calls only; never pin a thread that outlives the call.
