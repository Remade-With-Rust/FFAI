#!/usr/bin/env python3
"""Ultralytics behind Diana's `--serve` protocol, so the two can be measured alike.

Reads frame paths on stdin, one per line; writes one JSON line per frame. Prints
`{"ready":true}` once the model is loaded, so a driver can start measuring after
load rather than including it.

This exists for symmetry, and the symmetry is the point. Measuring Diana in a
child process while Ultralytics ran inside the driver would compare a process
that does nothing else against a process also running the harness, the image
decode for display, and the JSON. Same protocol, same isolation, one variable.

`predict(path)` is given the PATH, not a decoded array, because that is what
Diana's serve loop is given and what the ledger's reference adapter uses — both
engines then pay for their own decode inside their own timed region.
"""

import json
import sys
import time


def main():
    weights = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

    from ultralytics import YOLO

    model = YOLO(weights)
    out = sys.stdout
    print(json.dumps({"ready": True}), file=out, flush=True)

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        t = time.perf_counter()
        res = model.predict(path, verbose=False, conf=conf)[0]
        ms = (time.perf_counter() - t) * 1000.0
        n = 0 if res.boxes is None else len(res.boxes)
        print(json.dumps({"ms": round(ms, 2), "n": n}), file=out, flush=True)


if __name__ == "__main__":
    main()
