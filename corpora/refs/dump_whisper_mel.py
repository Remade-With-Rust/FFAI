#!/usr/bin/env python3
"""Dump openai-whisper's own log-mel spectrogram as the oracle for Mercury's
mel front-end.

The input signal is generated deterministically from a formula (not from
audio), so the fixture is tiny, license-free, and regenerable by anyone:

    python dump_whisper_mel.py --out crates/ffai-mercury/tests/fixtures/mel_oracle_80.f32

Output layout: raw little-endian f32, (n_mels, n_frames) row-major — the
same layout ffai_mercury::asr::mel::MelChunk uses.
"""

import argparse
import math
import struct
import sys


def reference_signal(n: int, sample_rate: int) -> list:
    """A deterministic chirp + tone mix, mirrored exactly in the Rust test.

    Sweeping frequency exercises every mel band, which a single tone would
    not: a filterbank bug in one band would hide behind silence elsewhere.
    """
    out = []
    for i in range(n):
        t = i / sample_rate
        chirp = math.sin(2.0 * math.pi * (200.0 + 1500.0 * t) * t)
        tone = 0.5 * math.sin(2.0 * math.pi * 3000.0 * t)
        out.append(0.6 * (chirp + tone))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-mels", type=int, default=80)
    ap.add_argument("--seconds", type=float, default=1.0)
    args = ap.parse_args()

    import numpy as np
    import torch
    import whisper

    sample_rate = whisper.audio.SAMPLE_RATE
    n = int(args.seconds * sample_rate)
    audio = torch.tensor(reference_signal(n, sample_rate), dtype=torch.float32)

    mel = whisper.log_mel_spectrogram(audio, n_mels=args.n_mels)
    arr = mel.numpy().astype(np.float32)
    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<II", arr.shape[0], arr.shape[1]))
        fh.write(arr.tobytes(order="C"))
    print(f"wrote {args.out}: {arr.shape[0]}x{arr.shape[1]} f32", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
