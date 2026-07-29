"""Batch adapter for Tesseract (C++) — the native-CPU OCR bar.

Contract (crates/ffai-bench/src/reference.rs): read a filelist, emit JSONL —
one {"load_secs"} line, then {"path", "text", "transcribe_secs"} per image.

HONESTY NOTE on per-image timing: tesseract.exe has no batch/server mode, so
each image is one subprocess, and `transcribe_secs` includes ~10-30 ms of
process spawn that the in-memory references (EasyOCR/PaddleOCR, held loaded
in Python) do not pay. That IS how Tesseract is deployed per-frame from a
CLI, and per-invocation is its real LIVE cost — but when reading the
per-page latency notes, know the spawn tax is inside Tesseract's number.
Recorded here rather than discovered later.

Configuration is pinned by the caller (references.toml): --psm and --oem are
required arguments, not defaults, so the ledger's argv says exactly what ran.
"""

import argparse
import json
import os
import subprocess
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="file with one image path per line")
    ap.add_argument("--bin", required=True, help="path to tesseract executable")
    ap.add_argument("--tessdata", required=True, help="TESSDATA_PREFIX directory")
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--psm", required=True, help="page segmentation mode (pinned)")
    ap.add_argument("--oem", required=True, help="OCR engine mode (pinned)")
    args = ap.parse_args()

    with open(args.batch, encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]

    # CreateProcess wants a normalized absolute path; a forward-slash relative
    # exe path raises WinError 2 even when the file exists.
    args.bin = os.path.abspath(args.bin)
    args.tessdata = os.path.abspath(args.tessdata)
    env = dict(os.environ, TESSDATA_PREFIX=args.tessdata)

    # No model preload exists for a subprocess-per-image tool; one untimed
    # warm run heats the OS file cache for the binary + traineddata so the
    # first timed image doesn't pay cold-disk costs no other image pays.
    if paths:
        subprocess.run(
            [args.bin, paths[0], "stdout", "-l", args.lang, "--psm", args.psm, "--oem", args.oem],
            capture_output=True,
            env=env,
            check=False,
        )
    print(json.dumps({"load_secs": 0.0}), flush=True)

    for path in paths:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [args.bin, path, "stdout", "-l", args.lang, "--psm", args.psm, "--oem", args.oem],
            capture_output=True,
            env=env,
            check=False,
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"tesseract failed on {path}: {proc.stderr.decode(errors='replace')}", file=sys.stderr)
            continue
        text = proc.stdout.decode("utf-8", errors="replace")
        print(json.dumps({"path": path, "text": text, "transcribe_secs": elapsed}), flush=True)


if __name__ == "__main__":
    main()
