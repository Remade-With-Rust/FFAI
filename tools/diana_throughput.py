"""Per-tier BATCH THROUGHPUT: our threads against their batched tensor.

Seven axes of this campaign measured per-image LATENCY and found ~1.9x
behind at every tier with no tier-dependence. That is one question. This is
a different one, and it is the question a server actually asks: given N
images and a whole machine, how many per second?

The two implementations answer it by different mechanisms, and that is the
point rather than a flaw:

  * Diana runs `detect_batch` — rayon across images, kernels serial inside
    each (see crates/ffai-diana/src/parallel.rs), so every core carries a
    whole image and there are no per-layer barriers.
  * Ultralytics takes a LIST and batches it into one tensor, so the win
    comes from bigger GEMMs rather than from threads. Python's GIL is why it
    cannot do what we do; a batched tensor is what it does instead.

Both are each implementation's best throughput path, which is what makes the
comparison fair. Work parity is checked, not assumed: total detections must
match.

Discipline carried from the latency work — three warm calls, min-of-N (load
only ever ADDS time), arms alternated per tier, and a deterministic count
compared before any duration is believed.

Usage:
    .venv-diana/Scripts/python.exe tools/diana_throughput.py [--images 24] [--reps 3]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCH_AB = ROOT / "target" / "release" / "examples" / "batch_ab.exe"
TIERS = ["n", "s", "m", "l", "x"]


def ours(tier: str, images: int) -> tuple[float, int]:
    """(wall ms for the whole batch, total detections)."""
    r = subprocess.run(
        [str(BATCH_AB), "--child", tier, str(images), "batch"],
        capture_output=True, text=True, cwd=ROOT,
    )
    wall = float(r.stdout.strip().splitlines()[-1].split()[0])
    m = re.search(r"dets (\d+)", r.stderr)
    return wall, int(m.group(1)) if m else -1


def theirs(tier: str, images: int, clips: list) -> tuple[float, int]:
    import cv2
    from ultralytics import YOLO

    arrays = [cv2.imread(str(p)) for p in clips[:images]]
    model = YOLO(str(ROOT / "corpora" / "cache" / f"yolo26{tier}.pt"))
    kw = dict(imgsz=640, conf=0.25, max_det=100, rect=True, device="cpu", verbose=False)
    for _ in range(3):
        model.predict(arrays[0], **kw)
    # Give the reference its BEST configuration, not its default.
    #
    # `predict(list)` defaults to batch=1 and loops: 15.76 img/s at n. An
    # explicit batch=4 gives 18.79 — 19% better. Measuring the default and
    # calling the difference our win would have published a 1.58x advantage
    # that is really ~1.1x. The reference's defaults are configuration; read
    # them before claiming anything.
    best_wall, best_res = None, None
    for batch in (1, 4, 8):
        t = time.perf_counter()
        try:
            res = model.predict(arrays, batch=batch, **kw)
        except TypeError:
            res = model.predict(arrays, **kw)
        w = (time.perf_counter() - t) * 1e3
        if best_wall is None or w < best_wall:
            best_wall, best_res = w, res
    wall, res = best_wall, best_res
    dets = sum(0 if r.boxes is None else len(r.boxes) for r in res)
    return wall, dets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=24)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--clips", default=os.environ.get("FFAI_DIANA_CLIPS", "corpora/clips/diana-coco"))
    args = ap.parse_args()

    clips = sorted((ROOT / args.clips).glob("coco-*.png"))
    if len(clips) < args.images:
        clips = (clips * ((args.images // max(len(clips), 1)) + 1))[: args.images]

    print(f"batch throughput · {args.images} images · best of {args.reps} · {args.clips}")
    print("reference gets its best of batch=1/4/8; we get detect_batch")
    print(f"{'tier':<5} {'Diana img/s':>12} {'ref img/s':>11} {'ratio':>8}  {'dets ok':>8}")
    for i, tier in enumerate(TIERS):
        try:
            a = b = None
            for r in range(args.reps):
                if (i + r) % 2 == 0:
                    x = ours(tier, args.images)
                    y = theirs(tier, args.images, clips)
                else:
                    y = theirs(tier, args.images, clips)
                    x = ours(tier, args.images)
                a = x if a is None or x[0] < a[0] else a
                b = y if b is None or y[0] < b[0] else b
        except Exception as e:  # noqa: BLE001
            print(f"{tier:<5} skipped: {e}")
            continue
        ours_ips = args.images / (a[0] / 1e3)
        ref_ips = args.images / (b[0] / 1e3)
        # A count both sides report. Divergent counts void the comparison.
        ok = "yes" if a[1] == b[1] else f"NO {a[1]}/{b[1]}"
        print(f"{tier:<5} {ours_ips:>12.2f} {ref_ips:>11.2f} {ours_ips / ref_ips:>7.2f}x  {ok:>8}",
              flush=True)


if __name__ == "__main__":
    main()
