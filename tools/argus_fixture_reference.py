#!/usr/bin/env python3
"""Harness fixture standing in for a VLM reference adapter. NOT A MODEL.

Emits the batch-adapter contract (crates/ffai-bench/src/reference.rs) with a
canned answer per image, so `ffai bench vlm` can be exercised end to end
without VLMEvalKit, model weights, or a network. It looks at the file name and
nothing else.

    argv:   <this> <filelist>
    stdout: one JSON object per line
              {"path": "...", "text": "...", "transcribe_secs": 0.0}
            plus one {"load_secs": 0.0}

The real Arm-1 and Arm-2 adapters are step 0b/0c of
docs/plans/argus-launch-plan.md. This file is how we know the pipe is
connected before they exist — nothing it prints is a measurement.
"""

import json
import os
import sys
import time

VERSION = "argus-fixture-reference 1.0 (NOT A MODEL)"


def main(argv: list[str]) -> int:
    if "--version" in argv:
        print(VERSION)
        return 0
    if len(argv) != 2:
        print(f"usage: {argv[0]} <filelist>", file=sys.stderr)
        return 2

    started = time.perf_counter()
    try:
        with open(argv[1], "r", encoding="utf-8") as fh:
            paths = [ln.strip() for ln in fh if ln.strip()]
    except OSError as exc:
        print(f"cannot read filelist: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"load_secs": time.perf_counter() - started}))
    for path in paths:
        t0 = time.perf_counter()
        stem = os.path.splitext(os.path.basename(path))[0]
        # A canned, deterministic answer. Deterministic on purpose: the plan's
        # Gate 2 makes byte-stability a v1 requirement, and a fixture that
        # varied run to run would make the harness's own repeatability
        # untestable.
        text = f"a fixture answer about {stem}"
        print(
            json.dumps(
                {
                    "path": path,
                    "text": text,
                    "transcribe_secs": time.perf_counter() - t0,
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
