# WHYS — diarization's per-chunk cost

**Unknown:** the live demo showed Mercury at ~500 ms/chunk against
whisper.cpp's ~274 ms. The arms were not doing equal work (that descent is in
the demo's commit), but stripping that away left a real question: with
speakers on, diarization costs **+621 ms on a 3 s chunk, 6.8× the ASR-only
path** — for a ~20M-parameter model. Why?

Method: `codec-six-whys-unknowns`. Profile before code.

---

## D2/D3 — which stage, and is it compute or overhead?

`examples/diarize_anatomy.rs`, 3 s chunk → 3 windows:

| stage | ms/chunk | share |
|---|---:|---:|
| fbank (all windows) | 5.7 | 1.1 % |
| **embed (all windows)** | **516.9** | **~100 %** |
| cluster + glue | ~0 | ~0 % |

**The embedding forward is the whole cost.** 172 ms per 1.5 s window. Not
the DSP, not the clustering, not glue — so three of the four levers named
from reading the code were aimed at ~1 % of the problem.

Scaling with frames, to separate per-frame compute from fixed per-call cost:

| window s | frames | ms | ms/frame |
|---|---:|---:|---:|
| 0.5 | 51 | 98.3 | 1.93 |
| 1.0 | 101 | 176.6 | 1.75 |
| 1.5 | 151 | 262.0 | 1.74 |
| 3.0 | 301 | 459.8 | 1.53 |

**Linear.** ms/frame is near-constant, so there is no large fixed cost to
amortize — which **prunes batching the windows into one forward**, the
lever that looked most obvious. Batching would buy the ~20 % that larger `m`
gives these GEMMs, not a multiple.

## D4/D5 — why is the forward slow?

Hand-estimated ~34.6 MFLOP/frame for this topology puts the achieved rate at
**~20 GFLOP/s**, against the 200–615 GFLOP/s candle reaches on this box.
That looked like the classic "a conv is a matmul after im2col, and this one
is not going through a GEMM" finding.

**REFUTED before a line was written.** candle's CPU backend has
`USE_IM2COL_CONV1D: bool = true` — conv1d already lowers to im2col + matmul.
The rate is instead explained by the topology: ECAPA's Res2Net blocks split
1024 channels into **8 sequential groups**, each doing its own small conv, so
the model inherently produces many small GEMMs where `m ≈ frames ≈ 150`.
That is the architecture, not our code, and it is not fixable without
changing the model.

*Rule reconfirmed: check whether the framework already does the thing before
building it. Ten minutes of reading the dependency beat a day of writing a
GEMM path that already existed.*

## The lever the profile actually pointed at

If the per-forward cost is inherent, the only remaining lever is **doing
fewer forwards** — and the live path was doing enormously many redundant
ones. `diarize_streaming` embeds every window of the buffer it is handed,
and the demo hands it the trailing 10 s once a second. Consecutive ticks
therefore share almost all of their audio, and every shared window was being
re-embedded from scratch.

**A content-keyed embedding cache.** Keyed on the window's SAMPLES, not its
time bounds, because the buffer slides: `(0.0, 1.5)` names different audio on
consecutive ticks, so a time key would return a confidently wrong embedding.
Content keying makes a hit numerically identical to a recompute — the same
vector, not an approximation — so it cannot move DER by construction. The
hash is ~96 KB per window against a ~172 ms forward, about 0.06 % of what a
hit saves.

### Gate

Paired A/B, both arms on the **same tick's audio** back to back, because
per-tick cost varies with how much speech VAD finds — arm-by-arm across runs
would compare different work, which is the error that started this whole
thread:

| clip | cache won | median cache | median no-cache | speedup |
|---|---|---:|---:|---:|
| longform-01 | 8/10 | 1069 ms | 2000 ms | 1.87× |
| longform-02 | 20/24 | 1489 ms | 2361 ms | 1.59× |
| **pooled** | **28/34, z = +3.77** | | | |

Neither clip's ratio is the headline — the paired win count is, and pooled it
clears |z| > 2 comfortably.

**Quality: DER identical to the digit**, 4.21 % blind / 5.00 % oracle with
the cache on and off, exactly as content-keying guarantees. 126 tests pass.
`FFAI_DIARIZE_CACHE=off` reproduces the old path.

## The 24 % hit rate — measured, and it overturned my correction

I first said the misses were hop/tick incommensurability. Then I read
`subsegment`, saw windows anchored to `region.start` (a VAD-detected speech
onset, which tracks content), and publicly withdrew that: if windows follow
the speech, a sliding buffer should still hit. **The trace says the first
answer was right, by a mechanism neither statement contained.**

`FFAI_DIARIZE_TRACE=1` over the sliding pattern, cache arm only:

| tick | regions | windows | hits |
|---|---|---:|---:|
| 1 | `0.400-10.000` | 12 | 12 (vs warm-up, same buffer) |
| 2 | `0.000-9.250` | 12 | **0** |
| 3 | `0.000-8.250, 9.310-10.000` | 11 | **0** |
| 4 | `0.000-7.250, 8.310-9.720` | 10 | **0** |
| 5 | `0.000-6.250, 7.310-8.720, 9.210-10.000` | 10 | **8** |
| 6 | `0.000-5.250, 6.310-7.720, 8.210-9.810` | 10 | **6** |

Two things fall out. Hits appear only at ticks 5 and 6 — **exactly three
ticks** after 2 and 3, which is `lcm(1 s tick, 0.75 s hop) = 3 s`. And the
leading region is **clipped to `0.000`** from tick 2 onward: the buffer's
edge cuts it, so its windows anchor to the *buffer* while the audio beneath
slides. Regions that are *not* clipped track their content correctly (ends
moving −1.0 s per tick) and hit.

So windows were effectively buffer-relative after all — via **clipping**,
not via the hop arithmetic — and both mechanisms compound. Ten of 34 distinct
window bounds recurred holding *different* audio, `0.000-1.500` alone under
five different keys. That is the entire miss population.

*The lesson is not "I was right all along". It is that I stated a mechanism,
then withdrew it on a second reading, and both moves were guesses. The trace
took ten minutes and settled what two rounds of reasoning could not.*

## The fix: an absolute window grid

`AsrOptions::stream_offset_secs` tells the engine where the buffer sits in
the stream; `subsegment_at` snaps each chain's first window to the absolute
hop grid. The same audio then lands on the same bounds regardless of where
the buffer begins — which is the precondition the content-keyed cache needed
to hit. `0.0` (the default, and every batch caller) reproduces the previous
behaviour for any region already on the grid.

| | median cache | median no-cache | speedup | paired | hit rate |
|---|---:|---:|---:|---|---:|
| cache only | 1069 ms | 2000 ms | 1.87× | 8/10 | 24 % |
| **+ absolute grid** | **641 ms** | 2024 ms | **3.16×** | **10/10** | 39 % |

The 39 % is understated — the counters include the no-cache arm's forced
misses, so the cache arm's own rate is considerably higher.

### RETRACTED: "DER unchanged" was measured on a stale binary

**The first version of this grid more than doubled DER, and shipped in 0.6.0
described as neutral.**

| | blind DER | oracle DER |
|---|---:|---:|
| region-anchored (before) | 4.21 % | 5.00 % |
| absolute grid, v1 | **9.60 %** | **8.11 %** |
| absolute grid, fixed | **4.20 %** | 5.00 % |

The mechanism: v1 snapped the whole window chain *forward* to the grid, which
skipped up to one hop — 0.75 s — of every region's leading audio. That is
precisely the moment a speaker starts talking, and the evidence a cluster most
needs.

**How it got through.** I re-ran `diarize_gate` after the change and read
4.21 %, so I wrote "DER unchanged" into the commit, this document, both
READMEs, and a published crate. The gate binary was **stale**: I had rebuilt
the library (`cargo build -p ffai-mercury`) and not the example, so the
measurement came from code that predated the change. This is the *third*
stale-artifact incident in this campaign — after the beam-search A/B and the
demo's wasm bundle — and the first one where the bad number reached users.

*The rule was already written and I did not follow it: `cargo build -p <lib>`
does not rebuild the examples that measure it. Build the artifact that
produces the number, and check its mtime before believing it.*

**The fix.** Emit the region-start window first, then follow the absolute grid
for everything after it. Coverage and alignment instead of a trade between
them; it costs one extra forward per region. DER returns to 4.20 %, and the
live speedup is 1.75× with 10/10 paired wins (down from the 3.16× that was
partly paid for in accuracy nobody had priced).

Two unit tests pin the invariant: the same audio yields the same absolute
bounds as the buffer slides, and snapping never drops a region.
`FFAI_DIARIZE_ABSGRID=off` restores region-anchored placement.

## Coarser hop — REFUTED

Forwards scale as `span / hop`, so a bigger hop looked like nearly free
compute. Measured (`examples/hop_sweep.rs` for cost, `diarize_gate` under
`FFAI_DIARIZE_HOP` for quality):

| hop | forwards | vs 0.75 | blind DER |
|---|---:|---:|---:|
| **0.75 s** | 442 | 1.00× | **4.20 %** |
| 1.00 s | 350 | 1.26× | 10.71 % |
| 1.50 s | 257 | 1.72× | 17.56 % |
| 2.00 s | 197 | 2.24× | 37.38 % |
| 3.00 s | 140 | 3.16× | 53.05 % |

DER degrades far faster than forwards fall — 2.24× fewer forwards costs about
nine times the error. 0.75 s is not inherited convention after all; it sits
near the knee. Geometry is runtime-settable (`FFAI_DIARIZE_WINDOW` / `_HOP`)
so this stays re-measurable.

## Incremental streaming — the one that worked

The cache removed repeated *forwards*, but the pipeline still **asked** for
windows it had already answered, and the ones it could not reuse were exactly
the boundary windows whose bounds move as the buffer slides even when the
speech does not. So keep the answers rather than re-derive the questions:
`diarize::StreamState` holds `(abs_start, abs_end, embedding)` in absolute
stream time, a tick sub-segments only past `processed_to`, and clustering runs
over the union.

Three load-bearing design points:

- **The settled cut is on the window's START, not its end.** A window merely
  *extending* past the mark was already generated with the audio available
  then; regenerating it would put a second embedding of overlapping speech
  into the clustering input and weigh that speech twice.
- **History is bounded** (`STREAM_HORIZON_SECS`, 30 s). Clustering is O(n²) in
  windows and the registry already carries identity across calls, so older
  windows have nothing left to contribute.
- **Turns return buffer-relative.** The state speaks absolute; the caller's
  transcript timestamps do not. Returning absolute times would shift every
  speaker label by the stream offset.

| path | live median | vs original |
|---|---:|---:|
| original | ~1260 ms | 1.00× |
| + content cache | ~509–535 ms | ~2.4× |
| **+ incremental** | **~377 ms** | **~3.3×** |

With incremental on, the cache's own A/B ratio collapses to **1.00×** — it is
subsumed exactly as predicted, because nothing asks twice. The cache stays for
the non-incremental path (a caller supplying no offset).

**Gates.** Streaming DER **5.55 %** against a 5.68 % baseline — slightly
better. Batch DER unchanged at 4.20 %. 130 tests, two new ones pinning that
settled audio is never re-segmented and that history stays bounded.
`FFAI_DIARIZE_INCREMENTAL=off` reverts.

**A gate that measured nothing, caught before it was believed.** The streaming
gate never passed `stream_offset_secs`, so every chunk claimed to start at 0
and the incremental path never engaged — its first "5.68 %, unchanged" was the
OLD code. The trace counter read **zero `diarize:incr` calls**, which is what
gave it away. Instrument whether the path under test actually ran; a gate that
silently exercises the old code reports a number that is true and irrelevant.

## Hardening: four gaps, and one was a real defect

Gating a change proves it works on the harness's inputs. It says nothing about
callers who behave differently — and asking "what is still untested?" found a
silent failure no DER number could have shown.

**1. A backwards offset swallowed audio, silently.** `processed_to` is the
furthest buffer end seen, so every window of rewound audio sorts before it:
`pending` returned nothing, that audio was never embedded, and the turns were
clipped to a range the stored windows did not overlap. **No speaker labels at
all, and no error.** Reachable in ordinary use — a second recording in one
session without `reset_speakers`, or any caller whose offset bookkeeping
restarts. Probed directly (`pending = 0`) rather than argued about. Fixed by
`note_buffer`: a buffer ending earlier than everything seen is a different
stream, so the history is dropped. Only the history — speaker identity stays
the caller's to keep or clear, since discarding it here would overrule them.

Verified through the demo's HTTP path, not only in a unit test:

```
[diarize:incr] reused=25 stored=25 to=70.44
[diarize:incr] stream rewound to 5.00 — history dropped
[diarize:incr] reused=0  stored=13 to=15.44
[diarize:incr] reused=13 stored=25 to=25.44
```

and the rewound request still returns `SPEAKER_00: …` where before it returned
nothing.

**2. The buffer-relative conversion had no test.** It is the arithmetic that
shifts every speaker label by the stream offset when wrong — a
diarization-looking bug that is not one. Extracted to a model-free
`clip_to_buffer` (this module's split: algorithm here, weights elsewhere) and
pinned with a five-case test covering both straddling boundaries and both
exclusions.

**3. `reset_speakers` clearing the window history was written and unverified.**
Now asserted. It matters twice: the history is the audio those identities were
learned from, and a new session starting at t=0 sits before the old
`processed_to` — gap 1 again, by another route.

**4. The demo's offset wiring had only been reasoned about.** Driven through
HTTP with advancing offsets, the trace shows reuse climbing 0 → 13 → 25 with
the horizon holding the store at 25. After a session where wiring that looked
correct silently was not, "I read the code" is not evidence.

Streaming DER after all four: **5.55 %**, unchanged. 134 tests.

## What is left

The per-forward cost is architectural (Res2Net's 8 sequential groups) and the
geometry sits at its knee, so what remains is structural rather than tuning:
clustering incrementally instead of re-clustering the horizon each tick, and
batching windows into one forward.

Also unexplored, and larger if it works: the per-forward cost itself is
architectural, but nothing forces 1.5 s windows at 0.75 s hop. A coarser hop
is strictly fewer forwards, at a DER cost that is measurable rather than
assumed — `diarize_gate.rs` already exists to price it.
