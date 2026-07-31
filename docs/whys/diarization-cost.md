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

**Gate.** Unlike the cache, this *moves* window placement, so it is not
neutral by construction and the batch path is affected too. DER measured
**4.21 % blind / 5.00 % oracle — identical**. Two new unit tests pin the
invariant it rests on: the same audio must yield the same absolute bounds as
the buffer slides, and snapping must never drop a region.

## What is left

The per-forward cost is architectural (Res2Net's 8 sequential groups), so the
remaining levers are all "ask for fewer forwards":

- **Coarser hop.** Nothing forces 0.75 s. A larger hop is strictly fewer
  windows at a DER cost that `diarize_gate.rs` can price rather than assume.
- **Incremental streaming.** Even with a perfect cache, the pipeline still
  clusters over the whole buffer every tick. Embedding only the new tail and
  keeping prior embeddings with their absolute timestamps subsumes the cache
  entirely — no window can miss if it is never requested twice.

Also unexplored, and larger if it works: the per-forward cost itself is
architectural, but nothing forces 1.5 s windows at 0.75 s hop. A coarser hop
is strictly fewer forwards, at a DER cost that is measurable rather than
assumed — `diarize_gate.rs` already exists to price it.
