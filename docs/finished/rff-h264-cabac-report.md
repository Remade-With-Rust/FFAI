# rff-codec-h264 0.1.0 — CABAC streams fail to decode

**Version:** `rff-codec-h264` 0.1.0 (latest on crates.io), `rff-core` 0.1.1,
`rff-format-mp4` 0.1.0. Confirmed against the current published versions, not a
stale lockfile.

## Summary

Any H.264 stream using **CABAC** entropy coding fails to decode. Since CABAC is
the default for every x264 profile above Baseline — and High is x264's default
profile — this means **a normally-encoded MP4 does not decode at all**.

## Reproduction

Source: any clip. Below is `akiyo_cif` from the Xiph derf collection, 352x288,
164 frames, encoded with ffmpeg/libx264.

```bash
# WORKS — CAVLC
ffmpeg -i akiyo_cif.y4m -c:v libx264 -profile:v main -coder 0 -bf 0 \
       -preset medium -crf 23 -g 30 -pix_fmt yuv420p cavlc.mp4

# FAILS — identical settings, CABAC only
ffmpeg -i akiyo_cif.y4m -c:v libx264 -profile:v main -coder 1 -bf 0 \
       -preset medium -crf 23 -g 30 -pix_fmt yuv420p cabac.mp4

# FAILS — x264 defaults (High profile)
ffmpeg -i akiyo_cif.y4m -c:v libx264 -preset medium -crf 23 -g 30 \
       -pix_fmt yuv420p default.mp4
```

Feeding each through `send_packet` / `receive_frame`:

| variant | frames decoded | first error |
|---|---:|---|
| CAVLC, no B-frames | **164 / 164** | — |
| CAVLC + B-frames (`-bf 2`) | **164 / 164** | — |
| High profile 8x8 transform, CAVLC | **164 / 164** | — |
| **CABAC, no B-frames** | **49 / 164** | packet 11: `rusty_h264: bitstream truncated` |
| **x264 default (High)** | **0 / 164** | packet 2: `rusty_h264: unsupported coding tool: P_Skip without reference` |

## What the isolation shows

One variable at a time, everything else held constant:

* **B-frames are fine** — CAVLC + B-frames decodes 164/164.
* **The 8x8 transform is fine** — High profile with `-coder 0` decodes 164/164.
* **CABAC is the only variable that breaks it.**

The two error strings are likely the same root cause seen from different
angles: a CABAC parser that desynchronises produces garbage syntax elements,
which surface either as an over-read (`bitstream truncated`) or as a nonsense
macroblock type (`P_Skip without reference`) depending on what the garbage
decodes to. The `P_Skip without reference` message appears at packet 2 rather
than packet 11, so the High-profile stream desyncs sooner — consistent with
more CABAC-coded syntax per packet.

## Content dependence

Severity scales with entropy — measured across a 7.7x bits/frame span:

| clip | bits/frame | frames decoded (CABAC) |
|---|---:|---:|
| akiyo | 7,407 | 49 / 164 (30 %) |
| container | 15,259 | 14 / 164 (9 %) |
| foreman | 17,605 | 14 / 164 (9 %) |
| stefan | 36,525 | 8 / 90 (9 %) |
| bus | 35,848 | 11 / 150 (7 %) |
| mobile | 56,734 | 13 / 164 (8 %) |

More CABAC-coded data per frame means the parser desynchronises sooner. The
simplest content gets ~30 % of the way through; everything else stops at 7-9 %.

## Note for the consumer side

`send_packet` reports all of this correctly and precisely — the diagnostics are
good. The silent-failure behaviour originally observed was a caller bug
(FFai's `ffai-media` discarded the `Err` and continued), now fixed there. This
report is only about the CABAC decode gap itself.
