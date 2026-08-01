"""D6 probe: does the reference's timed span include work ours does not?

The bench times our engine on a **pre-decoded** `ImageBuffer`
(`crates/ffai-bench/src/runner.rs`, the `decoded` loop) and times the
reference on `model.predict(path)` — which reads the file and decodes it
INSIDE the timed region. That is not a like-for-like comparison, and the
bias runs against the reference: it is doing strictly more work than we
are, so the published latency gap UNDERSTATES how far behind we sit.

Six-whys depth 6, question 1: "do both arms do identical work?" This
answers it with a number instead of an argument.

**Interleaved, not sequential.** The box is routinely saturated by another
benchmark, so "measure A for a while, then B" samples two different
machines. Each round runs A then B then B then A (ABBA) so any
warm-up/cool-down trend cancels rather than accumulating, and the verdict
is the paired win rate — under the null that is a fair coin, so
`z = (wins - N/2) / (0.5*sqrt(N))` and |z| > 2 is real regardless of how
far the medians drifted.

Usage:
    .venv-diana/Scripts/python.exe tools/diana_d6_span.py \
        --model corpora/cache/yolo26n.pt --rounds 24
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="corpora/cache/yolo26n.pt")
    ap.add_argument("--clips", default="corpora/clips/diana-coco")
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--images", type=int, default=8, help="images per round")
    ap.add_argument("--rect", default="on")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    clips = sorted((ROOT / args.clips).glob("coco-*.png"))[: args.images]
    if not clips:
        raise SystemExit(f"no clips in {args.clips}")
    paths = [str(p) for p in clips]
    # Decoded ONCE, outside every timed region — the same courtesy our
    # engine receives from the harness.
    arrays = [cv2.imread(p) for p in paths]
    assert all(a is not None for a in arrays)

    rect = args.rect == "on"
    model = YOLO(args.model)
    kw = dict(imgsz=640, conf=0.001, max_det=100, rect=rect, device="cpu", verbose=False)
    model.predict(paths[0], **kw)  # warm

    def with_decode() -> float:
        t = time.perf_counter()
        for p in paths:
            model.predict(p, **kw)
        return time.perf_counter() - t

    def without_decode() -> float:
        t = time.perf_counter()
        for a in arrays:
            model.predict(a, **kw)
        return time.perf_counter() - t

    pairs = []
    for r in range(args.rounds):
        # ABBA: the arm that runs first alternates, so "the second one is
        # warmer" cancels across rounds instead of accumulating into one arm.
        if r % 2 == 0:
            a = with_decode()
            b = without_decode()
        else:
            b = without_decode()
            a = with_decode()
        pairs.append((a, b))
        print(f"  round {r + 1:2d}/{args.rounds}: path {a * 1e3:7.1f} ms  "
              f"array {b * 1e3:7.1f} ms  decode share {100 * (a - b) / a:5.1f}%",
              flush=True)

    ratios = [a / b for a, b in pairs]
    shares = [100.0 * (a - b) / a for a, b in pairs]
    wins = sum(1 for a, b in pairs if a > b)  # path slower == decode costs
    n = len(pairs)
    z = (wins - n / 2) / (0.5 * math.sqrt(n))

    per_img = statistics.median(b for _, b in pairs) / len(paths)
    print()
    print(f"images/round      : {len(paths)}")
    print(f"paired rounds     : {n}")
    print(f"path slower in    : {wins}/{n}  z = {z:+.2f}")
    print(f"median ratio      : {statistics.median(ratios):.4f}x")
    print(f"median decode share of the reference's timed span: "
          f"{statistics.median(shares):.1f}%")
    print(f"reference inference-only, per image: {per_img * 1e3:.1f} ms")
    print(json.dumps({
        "rounds": n,
        "wins": wins,
        "z": z,
        "median_ratio": statistics.median(ratios),
        "median_decode_share_pct": statistics.median(shares),
        "ref_inference_only_ms": per_img * 1e3,
    }))


if __name__ == "__main__":
    main()
