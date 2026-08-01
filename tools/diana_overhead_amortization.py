"""Is the per-tier latency gap narrowing because fixed overhead amortizes?

The same-run gate table shows the gap against Ultralytics shrinking as the
model grows — ~2.2x at n, ~1.2x at m, ~0.9x at x. The proposed reason is
structural: Diana pays ~120 fork-joins per image (one per convolution, one
per activation), and a fork-join's cost is set by the thread count and the
barrier, NOT by how big the tensors are. Bigger tiers do more arithmetic per
layer, so the same fixed cost is spread over more work.

That is a story until it makes a prediction. It makes this one:

    overhead(tier) = cpu_at_24_threads - cpu_at_1_thread

should be roughly CONSTANT in absolute ms across tiers, while the serial
work itself grows several-fold. If instead the overhead scales with the
work, the explanation is wrong and the gap is narrowing for some other
reason.

CPU time is the right instrument here for two reasons: it is what the
overhead is actually spent as (threads spinning and waking at barriers, and
work landing on slower E-cores), and it does not accrue while descheduled,
so it survives the other benchmarks this box runs.

Usage:
    python tools/diana_overhead_amortization.py [--images 6]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "target" / "release" / "examples" / "cpu_vs_wall.exe"
TIERS = ["n", "s", "m", "l", "x"]


def cpu_ms(tier: str, threads: int, images: int) -> float:
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    out = subprocess.run(
        [str(EXE), tier, str(images)], capture_output=True, text=True, cwd=ROOT, env=env
    ).stdout
    m = re.search(r"mean per image: wall [\d.]+ ms .+ cpu ([\d.]+) ms", out)
    if not m:
        raise SystemExit(f"unparseable output for {tier}@{threads}:\n{out[-300:]}")
    return float(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=6)
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()

    print(f"fork-join overhead by tier · {args.images} images · CPU ms/image")
    print(f"{'tier':<5} {'serial (1t)':>12} {f'parallel ({args.threads}t)':>14} {'overhead':>10} {'x work':>8}")
    base = None
    rows = []
    for tier in TIERS:
        try:
            serial = cpu_ms(tier, 1, args.images)
            par = cpu_ms(tier, args.threads, args.images)
        except SystemExit as e:
            print(f"{tier:<5} skipped ({e})")
            continue
        over = par - serial
        base = base or serial
        rows.append((tier, serial, par, over))
        print(f"{tier:<5} {serial:>12.1f} {par:>14.1f} {over:>10.1f} {serial / base:>7.2f}x", flush=True)

    if len(rows) >= 2:
        overs = [r[3] for r in rows]
        works = [r[1] for r in rows]
        print()
        print(
            f"work grew {works[-1] / works[0]:.2f}x from {rows[0][0]} to {rows[-1][0]}; "
            f"overhead went {overs[0]:.0f} -> {overs[-1]:.0f} ms "
            f"({overs[-1] / overs[0]:.2f}x)"
        )
        print(
            "PREDICTION HOLDS if the overhead ratio is far below the work ratio — "
            "a fixed cost amortising. It FAILS if they track each other."
        )


if __name__ == "__main__":
    main()
