#!/usr/bin/env python3
"""Reference adapter: whisper.cpp (C++/ggml) — the native, no-Python bar.

This is the comparison FFai's pure-Rust claim is ultimately judged against.
faster-whisper measures us against CTranslate2, a specialised inference
engine; whisper.cpp measures us against a straightforward native
implementation, which is the closer analogue of what we are building.

`whisper-cli` accepts many files in one invocation and loads the model once,
so the batch contract (see crates/ffai-bench/src/reference.rs) maps cleanly:
we run it once, read each clip's `-otxt` output, and parse whisper.cpp's own
per-file `total time` for the warm number.

    python whisper_cpp_ref.py --batch files.txt \
        --bin .../whisper-cli.exe --model .../ggml-tiny.en.bin --threads 4
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

TOTAL_TIME = re.compile(r"total time\s*=\s*([0-9.]+)\s*ms")
LOAD_TIME = re.compile(r"load time\s*=\s*([0-9.]+)\s*ms")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--bin", required=True, help="path to whisper-cli")
    ap.add_argument("--model", required=True, help="path to a ggml-*.bin model")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--beam-size", type=int, default=1)
    args = ap.parse_args()

    with open(args.batch, encoding="utf-8-sig") as fh:
        paths = [line.strip() for line in fh if line.strip()]

    # Windows CreateProcess does not resolve a relative executable path
    # against the cwd the way a shell does; resolve it ourselves so
    # references.toml can stay repo-relative.
    cmd = [
        os.path.abspath(args.bin),
        "-m", args.model,
        "-t", str(args.threads),
        "-bs", str(args.beam_size),
        # best-of must track beam size, or greedy still samples 5 candidates.
        "-bo", str(args.beam_size),
        # Deliberately NOT -nt: that flag suppresses timestamp *generation*,
        # not just printing (43 vs 50 decode runs on a sample clip), which
        # would measure whisper.cpp doing less work than our engine. -otxt
        # already writes timestamp-free text for scoring.
        "-otxt",        # write <input>.txt per clip
        *paths,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = (proc.stderr or "") + (proc.stdout or "")
    if proc.returncode != 0:
        print(log[-2000:], file=sys.stderr)
        return proc.returncode

    # whisper.cpp prints ONE timing block per run, not per file, so only an
    # aggregate is available. Report it as such rather than dividing it into
    # per-clip numbers it never measured. Its "total time" excludes model
    # load, which it reports separately — the warm/cold split we want.
    totals = [float(x) for x in TOTAL_TIME.findall(log)]
    loads = [float(x) for x in LOAD_TIME.findall(log)]
    if loads:
        print(json.dumps({"load_secs": max(loads) / 1000.0}), flush=True)
    if totals:
        print(json.dumps({"batch_transcribe_secs": sum(totals) / 1000.0}), flush=True)

    for path in paths:
        txt = path + ".txt"
        text = ""
        if os.path.exists(txt):
            with open(txt, encoding="utf-8", errors="replace") as fh:
                text = " ".join(fh.read().split())
            # Best-effort cleanup ONLY. On Windows this raised
            # PermissionError(WinError 32) partway through a 134-clip TTS judge
            # run -- a transient lock from an AV scanner or a not-yet-released
            # whisper-cli handle -- which aborted the whole reference and made
            # the bench report 0/134 clips and three SKIPPED gates. The text is
            # already in memory by this point, so failing to unlink a temp file
            # must never cost the run its results.
            for attempt in range(3):
                try:
                    os.remove(txt)
                    break
                except OSError:
                    if attempt == 2:
                        break
                    time.sleep(0.1)
        print(json.dumps({"path": path, "text": text}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
