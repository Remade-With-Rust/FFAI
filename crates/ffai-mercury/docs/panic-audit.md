# ffai-mercury — panic audit (gate H-18)

Every panic site reachable from a public entry point, and the invariant that makes
it unreachable. **Audited 2026-08-15.**

**Counts.** Non-test library code contains **0 bare `.unwrap()`** and **18
`.expect("...")`**, all carrying a message that states the invariant.
`tools/lint_invariants.py` R5 now enforces the zero, so the discipline cannot erode
quietly.

> **Why a panic is a security finding here.** `ffai-mercury` is a library. A panic
> propagates into whatever process embedded it, so on a path reachable from
> caller-supplied audio or text it is a denial of service, not a debugging aid. The
> audit has already found two: `reflect_pad` indexing an empty slice, and
> `normalize` overflowing `u64` on a long digit run.

## Scope

The untrusted surface is **caller audio and caller text**. Model files are trusted
input by product decision (R-001, `threat-model.md`), so panic sites whose invariant
comes from a model file are in a different risk class and are marked *model-derived*
below — they are reachable only by a corrupt or hostile model, which the threat model
excludes.

## Sites

| Site | Invariant | Why it holds |
|---|---|---|
| `asr/vad.rs:190` | `bridged` is non-empty | Seeded `vec![regions[0]]` behind an `if regions.is_empty() { return }` guard, and the loop only ever pushes. Caller audio cannot empty it. |
| `asr/whisper_candle.rs:87,104,221` | the lazy `state` is initialized | Each site is preceded in the same function by the `if guard.is_none() { … }` block that fills it, under the same mutex guard. |
| `asr/decoder.rs:202` | `Ladder::Full` yields `Some` | Structural: `decode_window_inner` returns `None` only for the fallback rungs, and `Full` is the terminal rung. |
| `asr/decoder.rs:545` | a beam has ≥1 token | Beams are seeded with the SOT token before the loop and only grow. |
| `asr/diarize.rs:262` | every cluster is non-empty | `members` starts as `(0..n).map(|i| vec![i])` — one member each — and merging is `remove(j)` + `extend` into `i`, which never empties a survivor. For `n == 0` the `map` never runs. |
| `asr/speaker.rs:375,379,380` | config vectors are non-empty | *Model-derived*: `channels`, `kernel_sizes`, `dilations` come from the speaker-model config. |
| `tts/vits.rs:805,838`, `tts/decoder_kernels.rs:677` | 4 layers / 3 resblocks | *Model-derived*: fixed VITS architecture constants. |
| `tts/onnx.rs:99,113` | slice is exactly 8 / 4 bytes | Immediately follows `take(8)` / `take(4)`, which returns exactly that length or errors. |
| `tts/onnx.rs:190,215` | slice is exactly 4 bytes | `chunks_exact(4)` yields only full chunks by construction; the `raw.len() % 4` case errors earlier. |

## What this audit does NOT cover

- **Arithmetic panics**, which are a separate class and were the source of every real
  defect so far — see `docs/plans/use-protection-please.md` gate H-17. Five were found
  and fixed; overflow in release is *silent* because this workspace deliberately
  carries no `overflow-checks` (H-05 waived).
- **Allocation failure**, which aborts rather than panics and is out of scope.
- **`ffai-carmenta` and `ffai-diana`**, which parse far more untrusted input and have
  not been audited.
