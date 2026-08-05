"""Function-level comparison against Ultralytics, via torch.profiler.

This project spent a long campaign saying "we have no function-vs-function
benchmark against torch". We do — PyTorch ships an operator profiler — and the
reason it took so long is that constructing it soundly needs three D6 checks,
two of which fail on the obvious approach.

**D6.1 — profile the path that actually runs.** `YOLO(...).model(x)` returns a
TUPLE of (Tensor, dict): the training one2many head runs alongside one2one.
`predict()` is 0.75x of it (65.07 ms vs 86.21). Diana drops the one2many head
at conversion, so comparing against the raw module credits Ultralytics with
work it does not do at inference.

**D6.2 — check the units, per profile.** `self_cpu_time_total` summed over ops
came to 1.04x wall under `net(x)` and **3.59x wall under `predict()`** — the
latter is summing the 4 worker threads. Absolute op milliseconds are therefore
NOT comparable between the two profiles, nor against Diana's wall-clock stage
table. SHARES are, because each is a fraction of its own total.

**D6.3 — same input.** Both engines are given the identical letterboxed
tensor, so preprocessing is out of both arms.

A fourth thing the obvious approach gets wrong: the raw module reports
`native_batch_norm` at 6.2 ms, which reads as "Ultralytics does not fold BN".
`predict()` reports **0.00 ms** — it fuses at load, exactly as Diana folds at
conversion. That finding was an artefact of profiling the unfused module.
"""

import argparse
import numpy as np
import time
import torch
from torch.profiler import ProfilerActivity, profile
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="corpora/cache/yolo26n.pt")
    ap.add_argument("--input", default="corpora/refs/fixtures/diana_depth_input.bin")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    x = torch.from_numpy(
        np.fromfile(args.input, dtype=np.float32).reshape(1, 3, 640, 640)
    )
    m = YOLO(args.model)
    torch.set_num_threads(args.threads)

    for _ in range(4):
        m.predict(x, verbose=False, device="cpu")
    wall = 1e9
    for _ in range(15):
        t = time.perf_counter()
        m.predict(x, verbose=False, device="cpu")
        wall = min(wall, time.perf_counter() - t)

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(args.iters):
            m.predict(x, verbose=False, device="cpu")
    ka = prof.key_averages()
    total = sum(e.self_cpu_time_total for e in ka)

    print(f"ultralytics predict(), {args.threads} threads, 640x640")
    print(f"  wall min-of-15      {wall*1e3:8.2f} ms")
    print(f"  profiler self-CPU   {total/1e3/args.iters:8.2f} ms")
    print(f"  ratio               {total/1e3/args.iters/(wall*1e3):8.2f}x  "
          f"(>1 means the profiler sums worker threads; compare SHARES, not ms)\n")
    print(f"  {'op':<30} {'share':>8} {'calls':>7}")
    for e in sorted(ka, key=lambda e: -e.self_cpu_time_total)[:10]:
        print(f"  {e.key[:30]:<30} {e.self_cpu_time_total/total*100:7.1f}% "
              f"{e.count//args.iters:>7}")

    def share(sub):
        return sum(e.self_cpu_time_total for e in ka if sub in e.key) / total * 100

    print(f"\n  convolution {share('convolution'):5.1f}%   "
          f"activation {share('silu'):5.1f}%   batch_norm {share('batch_norm'):5.1f}%")
    print("\n  Diana, same input, wall-clock stages: convolution 75.4%, "
          "activation 0.7%,\n  batch_norm 0% (folded at conversion).")
    print("  The activation share is the one function-level difference large "
          "enough to see:\n  87 separate silu_ ops against an in-place "
          "convolution epilogue.")


if __name__ == "__main__":
    main()
