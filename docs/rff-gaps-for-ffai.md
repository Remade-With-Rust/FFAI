# What FFai needs from remade_ffmpeg_rs

Measured 2026-08-06 against the published crates, not read off a roadmap. Every
row was produced by running the thing.

FFai's video path is `rff-format-*` for demux + `rusty_h264` for decode. The
gaps below are what stands between that and matching Ultralytics' ingestion
surface (12 video containers, RTSP/webcam/screen, streaming).

**Ordered by what unblocks the most for the least work.**

---

## 1. `rff-codec-h264` pins `rusty_h264 ^0.2` — BLOCKING, one line

`rff-codec-h264` 0.1.0 declares `rusty_h264 = "^0.2"`, which caps resolution at
**0.2.1** no matter what is published. 0.8.0 shipped 2026-08-05. The cost of
that caret, measured on identical files:

| | rusty_h264 0.2.1 | 0.8.0 |
|---|---:|---:|
| CAVLC | 164/164 | 164/164 |
| CABAC | 49/164 | **164/164** |
| x264 default (High) | **0/164** | **164/164** |
| 1080p decode | 47.50 ms/frame | **22.72 ms/frame** |

**x264's default profile is High**, so on 0.2.1 a normal MP4 decoded to nothing.

**Ask:** relax to `rusty_h264 = "0.8"` and release `rff-codec-h264` 0.2.
**FFai's workaround today:** bypasses `rff-codec-h264` and calls `rusty_h264`
directly, losing the `CodecRegistry` seam. We would rather have the seam back.

## 2. `Stream` carries no duration or frame count — BLOCKING a format match

`rff_core::Stream` exposes `codec_id, width, height, pixel_format, time_base,
extradata`. There is no `duration`, `nb_frames` or equivalent; only `Packet` has
`duration`.

Ultralytics prints `video 1/1 (frame 12/164)`. We print `(frame 12)` because we
cannot know the total, and printing a guess is worse than printing nothing.

**Ask:** `Stream::duration: Option<i64>` and/or `nb_frames: Option<u64>`,
populated where the container declares it (MP4 `mvhd`/`stts`, MKV `Duration`).

## 3. MKV/WebM delivers AVCC, and it decodes to ZERO FRAMES SILENTLY

The most serious item, because of how it fails.

| container | demuxer | packets | frames | errors | Annex-B-looking |
|---|---|---:|---:|---:|---:|
| MP4 (H.264) | `mp4` | 164 | **164** | 0 | 164/164 |
| AVI (H.264) | `avi` | 164 | **164** | 0 | 164/164 |
| MPEG-TS (H.264) | `mpegts` | 164 | **164** | 0 | 164/164 |
| **MKV (H.264)** | `matroska` | 164 | **0** | **0** | 6/164 |
| WebM (VP9) | `matroska` | 164 | 0 | 0 | 0/164 |

`rff-format-mp4` normalises to **Annex-B** (start codes) and sets `extradata` to
empty. `rff-format-mkv` passes packets through as **AVCC** (length-prefixed
NALs) with the 41-byte `avcC` in `extradata`. `rusty_h264` accepts Annex-B, so
MKV yields 164 packets, 0 frames and **no error at all**.

**This is the same failure shape as the CABAC bug**: a silent short read that a
caller cannot distinguish from an empty video.

**Ask, in preference order:**
1. `rff-format-mkv` normalises to Annex-B like `rff-format-mp4` already does —
   consistent packet contract across demuxers is worth more than either choice.
2. Failing that, publish the bitstream filter (`h264_mp4toannexb` in ffmpeg
   terms) so consumers do not each write it.
3. Regardless: **a decoder handed a bitstream it cannot parse should ERROR, not
   return `Ok(None)` forever.**

**FFai's workaround today:** we implement the AVCC→Annex-B conversion locally
(`crates/ffai-media/src/annexb.rs`) and will delete it the moment either 1 or 2
lands.

## 4. MPEG-TS reports `0x0` dimensions

`mpegts` decodes correctly but its `Stream` says `width: 0, height: 0` — TS
carries no dimensions in the PMT, they come from the in-band SPS. Anything
sizing a buffer from the stream header gets zero.

**Ask:** parse the first SPS during `read_header`, or document that TS
dimensions are only valid after the first decoded frame.

## 5. Containers not yet published

Of Ultralytics' 12 video formats:

| status | formats |
|---|---|
| **published + verified decoding here** | mp4, mov, m4v, avi, ts |
| **published, blocked by item 3** | mkv, webm |
| published, untested by us | gif |
| **not published** | **mpg, mpeg (MPEG-PS), wmv, asf (ASF)** |

MPEG-PS and ASF are the only container gaps, and both are legacy. Low priority
against items 1-3.

## 6. Protocols — RTSP/RTMP

`rff-io` covers local files and HTTP(S). Ultralytics accepts `rtsp://`,
`rtmp://`, and `.streams` files listing several at once. RTSP is the one that
matters for surveillance, which is Diana's strongest use case.

**Ask:** `rff-io` protocol handlers for RTSP (RFC 2326 + RTP depacketisation)
and RTMP. This is libavformat's protocol layer, and it is the largest genuinely
missing piece.

## 7. Capture devices — webcam and screen

Ultralytics takes a webcam index, a screen region, and `.streams` files. That is
**libavdevice**, not libavformat: dshow/v4l2/avfoundation for cameras,
gdigrab/x11grab for screens.

**Ask:** an `rff-device` crate, or an explicit decision that capture is the
application's job. Either is fine — what does not work is FFai guessing.

## 8. VP9 in WebM

`rff-codec-vp9` is published and we have not wired it. Once item 3 lands, WebM
becomes a registry call plus a codec registration.

---

## Summary: what unblocks what

| item | effort | unblocks |
|---|---|---|
| 1. relax the h264 pin | one line + release | the registry seam, and everyone else on 0.2.1 |
| 2. `Stream` duration | small | exact Ultralytics-format progress lines |
| 3. MKV Annex-B | small | **mkv + webm — 2 of the 4 formats we lack** |
| 4. TS dimensions | small | correct sizing from headers |
| 5. MPEG-PS / ASF | new crates | 4 legacy formats |
| 6. RTSP/RTMP | large | **live surveillance ingest, the real prize** |
| 7. capture devices | medium | webcam/screen demos |
| 8. wire VP9 | ours, trivial | webm once 3 lands |

**Items 1-4 are small and together take FFai from 3 of 12 containers to 7 of
12**, with the two we still lack after that being legacy. Item 6 is the one
worth planning properly, because RTSP is what "deploy this on a camera" means.

---

## What FFai built locally, expecting these to land

Written so it can be deleted rather than maintained:

* `crates/ffai-media/src/annexb.rs` — AVCC→Annex-B conversion (item 3).
* `sample_frames`/`stream_frames` call `rusty_h264` directly rather than through
  `rff-codec-h264` (item 1).
* `frame_count_hint()` returns `None` and the CLI prints `(frame N)` (item 2).

Each is a workaround with a named owner upstream, not a fork.
