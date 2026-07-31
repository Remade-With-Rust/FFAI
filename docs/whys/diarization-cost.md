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

## What is left, honestly

**The hit rate is only ~24 %, and it should be far higher.** Windows are
placed on a 0.75 s hop *relative to the buffer*, while the demo slides the
buffer by 1 s. Those grids are incommensurate, so window boundaries realign
only every 3 s — most ticks re-cut the same audio at new offsets and miss.
Placing windows on an **absolute** grid (aligned to the source's own
timeline, not the buffer's start) would make nearly every shared window a
hit, and is the obvious next brick. The 1.6–1.9× measured here is what the
cache achieves *despite* the misalignment.

Also unexplored, and larger if it works: the per-forward cost itself is
architectural, but nothing forces 1.5 s windows at 0.75 s hop. A coarser hop
is strictly fewer forwards, at a DER cost that is measurable rather than
assumed — `diarize_gate.rs` already exists to price it.
