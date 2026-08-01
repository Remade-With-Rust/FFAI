"""D6/D3: how much CPU WORK does the reference do per image?

Wall time says who finishes first; it cannot say whether the loser is doing
more work or spreading the same work over fewer cores, and those two want
opposite fixes. CPU time (summed over threads) is the work; `cpu / wall` is
the mean occupancy.

This is also the only instrument that survives this campaign's box, which
is routinely saturated by another benchmark: CPU time does not accrue while
descheduled, so it stays comparable when wall does not. A wall-clock probe
run during the v3 sweep read the reference at 252 ms/image against the 67 ms
its own ledger row records — a 3.7x measurement artifact, and exactly the
kind of number that gets quoted.

Pair with `cargo run -p ffai-diana --example cpu_vs_wall`, which reports the
same two quantities for our engine over the same pre-decoded inputs.

Usage:
    .venv-diana/Scripts/python.exe tools/diana_d6_cpu.py --model corpora/cache/yolo26n.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="corpora/cache/yolo26n.pt")
    ap.add_argument("--clips", default="corpora/clips/diana-coco")
    ap.add_argument("--images", type=int, default=12)
    ap.add_argument("--rect", default="on")
    args = ap.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO

    clips = sorted((ROOT / args.clips).glob("coco-*.png"))[: args.images]
    if not clips:
        raise SystemExit(f"no clips in {args.clips}")
    # Decoded once, outside the timed region — the same courtesy the bench
    # harness gives our engine, so the two probes time the same span.
    arrays = [cv2.imread(str(p)) for p in clips]
    assert all(a is not None for a in arrays)

    model = YOLO(args.model)
    kw = dict(imgsz=640, conf=0.001, max_det=100, rect=args.rect == "on",
              device="cpu", verbose=False)
    model.predict(arrays[0], **kw)  # warm

    print(f"torch intra-op threads: {torch.get_num_threads()} · "
          f"interop {torch.get_num_interop_threads()} · {len(arrays)} images")
    print(f"{'img':>4}  {'wall ms':>10} {'cpu ms':>10} {'occupancy':>10}  dets")

    tw = tc = 0.0
    for i, a in enumerate(arrays):
        c0 = time.process_time()
        w0 = time.perf_counter()
        r = model.predict(a, **kw)[0]
        wall = time.perf_counter() - w0
        cpu = time.process_time() - c0
        tw += wall
        tc += cpu
        n = 0 if r.boxes is None else len(r.boxes)
        print(f"{i:>4}  {wall * 1e3:>10.1f} {cpu * 1e3:>10.1f} {cpu / wall:>9.2f}x  {n}")

    occ = tc / tw
    print()
    print(f"total wall {tw * 1e3:.1f} ms · total cpu {tc * 1e3:.1f} ms")
    print(f"mean per image: wall {tw * 1e3 / len(arrays):.1f} ms · "
          f"cpu {tc * 1e3 / len(arrays):.1f} ms")
    print(f"OCCUPANCY {occ:.2f}x")
    print(json.dumps({
        "images": len(arrays),
        "wall_ms_per_image": tw * 1e3 / len(arrays),
        "cpu_ms_per_image": tc * 1e3 / len(arrays),
        "occupancy": occ,
        "torch_threads": torch.get_num_threads(),
    }))


if __name__ == "__main__":
    main()
