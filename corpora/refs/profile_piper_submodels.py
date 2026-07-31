"""A TRUSTWORTHY per-stage split for the piper reference — no profiler.

`profile_piper_stages_matched.py` had to be discarded: enabling ORT's profiler
makes the run 1.75-1.93x slower, the tax is not uniform per node, and
correcting for it drove the duration predictor to a negative time. So this
takes the per-stage numbers a different way entirely.

Rather than instrumenting the graph, cut it. `onnx.utils.extract_model` yields
CUMULATIVE PREFIXES of the network, each a valid standalone model driven by the
real graph inputs:

    A = text encoder                    -> /enc_p/Split_output_{0,1}
    B = A + duration predictor          -> /Ceil_output_0
    C = B + flow                        -> /Mul_7_output_0
    D = C + decoder                     -> output   (the whole model)

Each prefix is timed UNPROFILED at a pinned thread count, and the stages come
out by differencing: dp = B - A, flow = C - B, decoder = D - C. No intermediate
tensor ever has to be synthesized, which is what makes the cut safe.

The design carries its own falsifier: prefix D is the entire model, so its time
must match the standalone full-model measurement. If it does not, extraction is
perturbing ORT's graph optimization and the whole split is void — so that check
is printed first and loudly.

    python corpora/refs/profile_piper_submodels.py \
        --model <voice.onnx> --ids ids.txt --threads 8 --full-wall 1811.1
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime

# Stage boundaries, taken from dump_vits_stages.py (which pins the oracle).
PREFIXES = [
    ("enc_p", ["/enc_p/Split_output_0", "/enc_p/Split_output_1"]),
    ("+dp", ["/Ceil_output_0"]),
    ("+flow", ["/Mul_7_output_0"]),
    ("+dec", ["output"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3, help="full prefix sweeps; stages differenced within each")
    ap.add_argument(
        "--full-wall",
        type=float,
        default=0.0,
        help="ms/pass for the UNMODIFIED model at this thread count; the falsifier",
    )
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.ids).read_text().splitlines() if line.strip()]
    graph_inputs = ["input", "input_lengths", "scales"]
    tmp = Path(tempfile.mkdtemp(prefix="piper_sub_"))

    def feeds_for(ids):
        return {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
        }

    def time_model(path, out_names):
        so = onnxruntime.SessionOptions()
        if args.threads:
            so.intra_op_num_threads = args.threads
            so.inter_op_num_threads = 1
        sess = onnxruntime.InferenceSession(
            str(path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        real_outs = [o.name for o in sess.get_outputs()]
        for ids in rows:  # warm
            sess.run(real_outs, feeds_for(ids))
        best = float("inf")
        for _ in range(args.passes):
            t0 = time.perf_counter()
            for ids in rows:
                sess.run(real_outs, feeds_for(ids))
            best = min(best, time.perf_counter() - t0)
        return best * 1000.0

    # Differencing AMPLIFIES noise: flow = (+flow) - (+dp) compounds the error
    # of two large prefixes, and a first cut of this script had ORT's flow swing
    # 74% between runs while its text encoder held to 2%. So sweep all prefixes
    # together, REPEATEDLY, and difference WITHIN each rep — drift then hits
    # both terms of a difference alike instead of being attributed to a stage.
    # Report the spread so an unresolved stage announces itself.
    sessions = []
    for name, outs in PREFIXES:
        sub = tmp / f"{name.strip('+')}.onnx"
        onnx.utils.extract_model(args.model, str(sub), graph_inputs, outs)
        sessions.append((name, sub, outs))
    # The unmodified model rides ALONG IN EVERY REP as a fifth entry, not once
    # at the end. Measured last, it absorbs all the drift accumulated over the
    # sweep and the falsifier then fires on machine load rather than on
    # extraction fidelity -- which is exactly what happened twice.
    sessions.append(("=full", args.model, ["output"]))

    reps = []
    for _ in range(args.reps):
        reps.append([(name, time_model(sub, outs)) for name, sub, outs in sessions])

    results = [
        (name, sorted(r[i][1] for r in reps)[len(reps) // 2])
        for i, (name, _, _) in enumerate(sessions)
    ]
    for name, ms in results:
        print(f"  {'model' if name == '=full' else 'prefix'} {name:<8} {ms:8.1f} ms (median of {args.reps})")

    # Falsifier WITHIN each rep, so drift cancels in the ratio.
    print()
    ratios = sorted(rep[-2][1] / rep[-1][1] for rep in reps)
    ratio = ratios[len(ratios) // 2]
    ok = 0.9 < ratio < 1.1
    print(
        f"FALSIFIER (per-rep, full prefix / unmodified): median {ratio:.3f}x "
        f"[{ratios[0]:.3f} .. {ratios[-1]:.3f}]  "
        f"[{'OK — extraction is faithful' if ok else 'VOID — extraction perturbed the graph, split is meaningless'}]"
    )
    if not ok:
        return

    print("\nper-stage, differenced WITHIN each rep (median [min..max] over reps):")
    names = {"enc_p": "text_encoder", "+dp": "duration_pred", "+flow": "flow", "+dec": "decoder"}
    for i, (name, _, _) in enumerate(sessions[:-1]):
        vals = sorted(rep[i][1] - (rep[i - 1][1] if i else 0.0) for rep in reps)
        med = vals[len(vals) // 2]
        spread = (vals[-1] - vals[0]) / med * 100 if med else 0.0
        flag = "" if spread < 25 else "   <- UNRESOLVED, spread too wide to compare"
        share = 100.0 * med / results[-1][1]
        print(
            f"  {names[name]:<15} {med:8.1f} ms  [{vals[0]:7.1f} .. {vals[-1]:7.1f}]"
            f"  spread {spread:4.0f}%  share {share:5.1f}%{flag}"
        )

    # SHARE is the statistic that survives machine load. Across three runs of
    # this script the reference's absolute times swung up to 66% while its
    # per-stage SHARES held to about a point (enc ~8.5%, dp ~4.8%, flow ~15%,
    # dec ~65%). Comparing shares against ours, then scaling by the overall
    # ratio, gives a per-stage verdict that a loaded box cannot corrupt --
    # whereas comparing raw ms across sessions is exactly how this campaign
    # got its per-stage picture backwards twice.
    print(
        "\n  NOTE: compare SHARES, not raw ms, unless both engines were measured\n"
        "  in the same window. Shares are load-invariant here; absolute ms are not."
    )


if __name__ == "__main__":
    main()
