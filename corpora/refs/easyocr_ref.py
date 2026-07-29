"""Batch adapter for EasyOCR (Python/PyTorch) — the scene-text breadth bar.

Contract (crates/ffai-bench/src/reference.rs): read a filelist, emit JSONL —
one {"load_secs"} line, then {"path", "text", "transcribe_secs"} per image.

The Reader is constructed once (that's the model load) and held in memory;
per-image time is recognition only — the same warm contract as the ASR
references, and the fair in-memory comparison for LIVE per-frame latency.

detail=0/paragraph=False returns one string per detected text line in
EasyOCR's own reading order; lines are joined with newlines. Mode::Ocr
scoring collapses whitespace, so line-break placement doesn't cost anyone.
"""

import argparse
import json
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    with open(args.batch, encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]

    t0 = time.perf_counter()
    import easyocr  # noqa: PLC0415 — import inside the timed load on purpose

    reader = easyocr.Reader([args.lang], gpu=False, verbose=False)
    # One untimed warm inference: the first readtext pays one-off lazy init
    # (thread pools, conv algo selection) that steady-state frames never see.
    if paths:
        reader.readtext(paths[0], detail=0)
    print(json.dumps({"load_secs": time.perf_counter() - t0}), flush=True)

    for path in paths:
        t1 = time.perf_counter()
        lines = reader.readtext(path, detail=0, paragraph=False)
        elapsed = time.perf_counter() - t1
        print(
            json.dumps({"path": path, "text": "\n".join(lines), "transcribe_secs": elapsed}),
            flush=True,
        )


if __name__ == "__main__":
    main()
