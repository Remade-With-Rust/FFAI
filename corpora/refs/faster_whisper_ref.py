#!/usr/bin/env python3
"""Reference adapter: faster-whisper (Python + CTranslate2/C++).

Batch contract (see crates/ffai-bench/src/reference.rs): read a file list,
emit one JSON object per line to stdout.

    python faster_whisper_ref.py --batch files.txt --model tiny

    {"load_secs": 1.83}
    {"path": "clip1.wav", "text": "...", "transcribe_secs": 0.42}

`transcribe_secs` EXCLUDES model load, which is reported once as `load_secs`.
That split is what lets ffai bench report warm and end-to-end throughput
separately instead of quoting whichever number flatters someone.
"""

import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="file with one audio path per line")
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--language", default=None)
    args = ap.parse_args()

    # utf-8-sig tolerates a BOM, which some editors and shells prepend.
    with open(args.batch, encoding="utf-8-sig") as fh:
        paths = [line.strip() for line in fh if line.strip()]

    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print(json.dumps({"load_secs": time.perf_counter() - t0}), flush=True)

    for path in paths:
        t0 = time.perf_counter()
        segments, _info = model.transcribe(
            path, beam_size=args.beam_size, language=args.language
        )
        # faster-whisper is lazy: the generator must be drained inside the
        # timed region or we would report ~0 s and claim infinite speed.
        text = " ".join(seg.text.strip() for seg in segments)
        elapsed = time.perf_counter() - t0
        print(
            json.dumps({"path": path, "text": text, "transcribe_secs": elapsed}),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
