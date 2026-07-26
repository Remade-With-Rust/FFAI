#!/usr/bin/env python3
"""Reference adapter: openai-whisper (Python/PyTorch) — the accuracy definition.

Batch contract identical to faster_whisper_ref.py:

    python openai_whisper_ref.py --batch files.txt --model tiny
"""

import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="file with one audio path per line")
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--language", default=None)
    # Explicit, because the defaults differ between implementations:
    # openai-whisper defaults to greedy (beam_size=None) while faster-whisper
    # defaults to beam_size=5. Benchmarking those against each other compares
    # decoding strategies, not implementations. ffai bench pins both.
    ap.add_argument("--beam-size", type=int, default=5)
    args = ap.parse_args()

    # utf-8-sig tolerates a BOM, which some editors and shells prepend.
    with open(args.batch, encoding="utf-8-sig") as fh:
        paths = [line.strip() for line in fh if line.strip()]

    import whisper

    t0 = time.perf_counter()
    model = whisper.load_model(args.model)
    print(json.dumps({"load_secs": time.perf_counter() - t0}), flush=True)

    for path in paths:
        t0 = time.perf_counter()
        result = model.transcribe(
            path, language=args.language, beam_size=args.beam_size, fp16=False
        )
        elapsed = time.perf_counter() - t0
        print(
            json.dumps(
                {"path": path, "text": result["text"].strip(), "transcribe_secs": elapsed}
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
