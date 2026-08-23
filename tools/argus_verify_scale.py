#!/usr/bin/env python3
"""Oracle probe: verify a VLM corpus's declared `scale` is actually right.

    .venv-argus/Scripts/python.exe tools/argus_verify_scale.py \
        --corpus corpora/argus-ocrbench-lite-v1.toml

Feeds the benchmark's OWN ground-truth answers back through the corpus's
declared scorer. A correct scale makes that normalise to exactly 1.0.

WHY THIS EXISTS
---------------
It caught a real defect in this repo on the day it was written.

OCRBench's scale was declared as a flat `1000.0` — the full benchmark's
maximum — on the reasonable-sounding basis that "OCRBench is scored out of
1000". But its evaluator returns a **count of correct answers**, so on a
50-item corpus a perfect run scores 50, not 1000. A genuinely good 40-of-50
run normalised to **0.04**, and the quality gate would have read that as near-
total failure. Nothing errored. The raw number in the ledger was correct the
whole time; only the normaliser was wrong, and the normaliser is what the gate
verdicts on.

THE ASYMMETRY THAT MAKES THIS NECESSARY
---------------------------------------
The obvious check — feed garbage, expect 0 — is worth almost nothing on its
own, because **a broken join also scores 0**. Constant-garbage predictions and
predictions the evaluator could not match produce the identical number. You
need the other end: only a correct join with a correct scale can reach 1.0.

Run the two together and they bracket the instrument:

    garbage -> 0.0   (the metric is not crediting nonsense)
    truth   -> 1.0   (the join lands and the scale is right)

Neither alone is evidence.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import tomllib


def first_answer(a):
    """OCRBench-style answers may be a list, or a str holding a list repr."""
    if isinstance(a, str) and a.startswith("["):
        try:
            a = json.loads(a.replace("'", '"'))
        except (ValueError, TypeError):
            return a
    if isinstance(a, list):
        return str(a[0]) if a else ""
    return str(a)


def run_scorer(cmd, predictions):
    argv = [c.replace("{predictions}", predictions) for c in cmd]
    # Normalise the executable's separators. The corpus declares it with
    # forward slashes (`.venv-argus/Scripts/python.exe`) because that is what
    # the TOML and the Rust harness use — Rust's `Command` resolves it happily,
    # but Windows `CreateProcess` via Python's subprocess raises
    # `FileNotFoundError [WinError 2]` on a relative forward-slash path. Same
    # declaration, two runners, one of which needs it spelled the local way.
    if argv and os.sep != "/":
        local = os.path.normpath(argv[0])
        if os.path.exists(local):
            argv[0] = os.path.abspath(local)
    out = subprocess.run(argv, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        print(f"scorer exited {out.returncode}:\n{out.stderr[-1200:]}", file=sys.stderr)
        return None
    for ln in reversed([l for l in out.stdout.splitlines() if l.strip()]):
        try:
            return json.loads(ln.strip())
        except ValueError:
            continue
    print(f"scorer printed no contract line:\n{out.stdout[-600:]}", file=sys.stderr)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--dataset", default=None,
                    help="defaults to the --dataset in the corpus's scorer command")
    ap.add_argument("--tolerance", type=float, default=1e-6)
    args = ap.parse_args()

    with open(args.corpus, "rb") as fh:
        man = tomllib.load(fh)
    scorer = man.get("scorer")
    if not scorer:
        print(f"{args.corpus} declares no [scorer]", file=sys.stderr)
        return 2
    scale = float(scorer["scale"])
    cmd = scorer["command"]

    dataset = args.dataset
    if dataset is None and "--dataset" in cmd:
        dataset = cmd[cmd.index("--dataset") + 1]
    if not dataset:
        print("could not infer --dataset from the scorer command; pass it", file=sys.stderr)
        return 2

    from vlmeval.dataset import build_dataset
    ds = build_dataset(dataset)
    if ds is None:
        print(f"VLMEvalKit does not know dataset '{dataset}'", file=sys.stderr)
        return 2
    d = ds.data.copy()
    d["index"] = d["index"].astype(str)
    if "answer" not in d.columns:
        print(f"{dataset} has no `answer` column — this probe cannot verify it "
              f"(judge-scored benchmarks need a different check)", file=sys.stderr)
        return 2
    answers = dict(zip(d["index"], d["answer"]))

    holdout = [c for c in man.get("clips", []) if c.get("split") == "holdout"]
    if not holdout:
        print("corpus has no holdout clips", file=sys.stderr)
        return 2

    tmp = tempfile.mkdtemp(prefix="ffai-scale-probe-")
    truth_p = os.path.join(tmp, "truth.jsonl")
    junk_p = os.path.join(tmp, "junk.jsonl")
    missing = []
    with open(truth_p, "w", encoding="utf-8") as t, open(junk_p, "w", encoding="utf-8") as j:
        for c in holdout:
            cid = str(c["id"])
            if cid not in answers:
                missing.append(cid)
                continue
            base = {"id": cid, "path": c["path"], "prompt": c.get("prompt")}
            t.write(json.dumps({**base, "prediction": first_answer(answers[cid])}) + "\n")
            j.write(json.dumps({**base, "prediction": "zzzz-not-an-answer-zzzz"}) + "\n")
    if missing:
        print(f"{len(missing)} clip id(s) are not {dataset} indices: {missing[:5]}",
              file=sys.stderr)
        return 1

    print(f"corpus   {man['name']}  ({len(holdout)} holdout items)")
    print(f"declared scale = {scale}  metric = {scorer.get('metric')}\n")

    fails = 0
    truth = run_scorer(cmd, truth_p)
    if truth is None:
        return 1
    tn = truth["score"] / scale
    ok = abs(tn - 1.0) <= args.tolerance
    fails += 0 if ok else 1
    print(f"  ground truth -> raw {truth['score']:>10.4f}  normalised {tn:.6f}   "
          f"{'OK' if ok else 'WRONG SCALE'}")
    if not ok:
        implied = truth["score"]
        print(f"       a perfect run should normalise to 1.0; declared scale {scale} gives "
              f"{tn:.4f}.\n"
              f"       the scale this corpus should declare is {implied} "
              f"(= the evaluator's perfect score for these {len(holdout)} items).")

    junk = run_scorer(cmd, junk_p)
    if junk is not None:
        jn = junk["score"] / scale
        jok = jn < 0.5
        fails += 0 if jok else 1
        print(f"  garbage      -> raw {junk['score']:>10.4f}  normalised {jn:.6f}   "
              f"{'OK' if jok else 'SUSPECT — nonsense should not score'}")

    print(f"\n  {'PASS' if fails == 0 else 'FAIL'} — "
          f"a scale is only verified when BOTH ends land: truth 1.0 and garbage ~0.")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
