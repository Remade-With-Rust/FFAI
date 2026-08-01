"""Per-tier engine-vs-reference cost, measured in CPU time.

The bench's speed gate compares WALL throughput, and within one run the
engine and the reference are timed minutes apart — so on a box another
session is using, the comparison samples two different machines. Every
attempt to wait for a quiet box this session failed: the reference that the
ledger records at 67 ms/image has measured 252 ms and 19372 ms in probes
taken hours apart.

CPU time does not accrue while descheduled. It answers a slightly different
question — total WORK rather than latency — but it answers it the same way
whatever else the box is doing, which is the trade this campaign needs. Use
it to corroborate the SHAPE of the wall result (does the gap narrow as the
model grows?), not to replace it.

Arms are alternated per tier so neither systematically samples a busier
minute, and both are handed pre-decoded images so the timed spans match.

Usage:
    .venv-diana/Scripts/python.exe tools/diana_cpu_ratio.py [--images 8]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OURS = ROOT / "target" / "release" / "examples" / "cpu_vs_wall.exe"
TIERS = ["n", "s", "m", "l", "x"]


def ours_cpu_ms(tier: str, images: int) -> tuple[float, float]:
    """(cpu ms/image, wall ms/image) for Diana at this tier."""
    out = subprocess.run(
        [str(OURS), tier, str(images)], capture_output=True, text=True, cwd=ROOT
    ).stdout
    m = re.search(r"mean per image: wall ([\d.]+) ms .+ cpu ([\d.]+) ms", out)
    if not m:
        raise SystemExit(f"could not parse cpu_vs_wall output for {tier}:\n{out[-400:]}")
    return float(m.group(2)), float(m.group(1))


def ref_cpu_ms(tier: str, images: int) -> tuple[float, float]:
    """(cpu ms/image, wall ms/image) for Ultralytics at this tier."""
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "diana_d6_cpu.py"),
            "--model",
            f"corpora/cache/yolo26{tier}.pt",
            "--images",
            str(images),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout
    line = [l for l in out.splitlines() if l.startswith("{")]
    if not line:
        raise SystemExit(f"could not parse diana_d6_cpu output for {tier}:\n{out[-400:]}")
    d = json.loads(line[-1])
    return d["cpu_ms_per_image"], d["wall_ms_per_image"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=8)
    args = ap.parse_args()

    print(f"CPU work per image, rectangular geometry, {args.images} images/tier")
    print(f"{'tier':<5} {'Diana cpu':>10} {'ref cpu':>10} {'cpu x':>7}   {'Diana wall':>11} {'ref wall':>10} {'wall x':>7}")
    rows = []
    for i, tier in enumerate(TIERS):
        try:
            # Alternate which arm goes first so neither systematically lands
            # on the busier half of a minute.
            if i % 2 == 0:
                a = ours_cpu_ms(tier, args.images)
                b = ref_cpu_ms(tier, args.images)
            else:
                b = ref_cpu_ms(tier, args.images)
                a = ours_cpu_ms(tier, args.images)
        except SystemExit as e:
            print(f"{tier:<5} skipped: {e}")
            continue
        rows.append((tier, a[0], b[0], a[1], b[1]))
        print(
            f"{tier:<5} {a[0]:>10.1f} {b[0]:>10.1f} {a[0] / b[0]:>6.2f}x   "
            f"{a[1]:>11.1f} {b[1]:>10.1f} {a[1] / b[1]:>6.2f}x",
            flush=True,
        )
        time.sleep(1)

    print()
    print("cpu x > 1 means Diana does MORE total work; < 1 means less.")
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        print(
            f"shape: cpu ratio goes {first[1] / first[2]:.2f}x at {first[0]} -> "
            f"{last[1] / last[2]:.2f}x at {last[0]}"
        )


if __name__ == "__main__":
    main()
