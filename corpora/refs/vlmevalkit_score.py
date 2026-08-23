#!/usr/bin/env python3
"""Score FFai predictions with VLMEvalKit's OWN evaluator. Arm 1's metric.

Implements the `ScorerSpec` contract (crates/ffai-bench/src/corpus.rs):

    argv:   --dataset <NAME> --predictions <predictions.jsonl>
    stdout: exactly one JSON object -> {"score": <float>, "metric": <str>, "n": <int>}
    exit:   0 on success; non-zero means the run is VOID, never score 0

What this file does and does not do
-----------------------------------
It **converts** our predictions into the frame VLMEvalKit's evaluator expects,
calls `dataset.evaluate(...)`, and **reads the headline number out of the
result**. That is all.

It does NOT compare an answer to a ground truth, does not extract a choice
letter from free text, does not normalise casing or punctuation, and does not
decide what counts as correct. Every one of those is answer extraction, every
one of them is part of a VLM metric, and every one of them written here would
be a scorer tuned to our own model's output — the 2.8x-biased metric that cost
the Carmenta campaign a year (docs/plans/argus-launch-plan.md §5).

If you find yourself adding an `if pred.lower() == gt.lower()` to this file,
the answer is that the dataset's evaluator already does it, correctly, and
differently from how you were about to.

The join key
------------
Predictions carry the corpus clip id, which `tools/argus_build_corpus.py` set
to VLMEvalKit's own `index` column precisely so this join is theirs and not a
fuzzy match on text. An index we cannot join is a hard error: silently dropping
it would shrink the scored population and produce a real number over the wrong
items.
"""

import argparse
import json
import os
import sys
import tempfile

# Keys a VLMEvalKit evaluator may use for its headline figure, in the order we
# will accept them. Ordered deliberately: the most explicitly-named "this is
# the overall score" keys first, generic ones last.
#
# We never average several candidates together and never fall back to "the
# first numeric thing we found" — an unrecognised result shape is an error,
# because picking the wrong column here would misreport a benchmark and look
# entirely normal doing it.
HEADLINE_KEYS = [
    "Final Score", "Final Score Norm", "final score",
    "Overall", "overall", "OVERALL",
    "Average", "avg", "acc", "Acc", "accuracy",
    "score", "Score",
]


def die(msg: str, code: int = 1) -> int:
    print(msg, file=sys.stderr)
    return code


def read_predictions(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            if "id" not in rec or "prediction" not in rec:
                raise ValueError(f"prediction line missing id/prediction: {ln[:120]}")
            out.append((str(rec["id"]), rec["prediction"]))
    return out


def headline_from(result, dataset: str):
    """Pull one number out of whatever `evaluate` returned.

    Returns (value, key_used). Raises if the shape is unrecognised — see the
    note on HEADLINE_KEYS for why guessing is not allowed.
    """
    import pandas as pd

    # dict-like
    if isinstance(result, dict):
        for k in HEADLINE_KEYS:
            if k in result:
                v = result[k]
                if isinstance(v, (list, tuple)) and v:
                    v = v[0]
                return float(v), k
        raise ValueError(f"no headline key in dict result; keys were {list(result)[:20]}")

    if isinstance(result, pd.DataFrame):
        df = result
        # Common VLMEvalKit shape: one row, metrics as columns.
        if len(df) == 1:
            row = df.iloc[0]
            for k in HEADLINE_KEYS:
                if k in df.columns:
                    return float(row[k]), k
        # Alternative shape: a 'split'/'category' column plus a value column,
        # with an explicit Overall row.
        for label_col in ("split", "category", "Category", "Task", "index"):
            if label_col in df.columns:
                for val_col in df.columns:
                    if val_col == label_col:
                        continue
                    hits = df[df[label_col].astype(str).str.lower().isin(
                        ("overall", "final score", "average", "all"))]
                    if len(hits) == 1:
                        try:
                            return float(hits.iloc[0][val_col]), f"{label_col}=overall/{val_col}"
                        except (TypeError, ValueError):
                            continue
        # Single-column frames with a named index.
        if df.shape[1] == 1:
            col = df.columns[0]
            for k in HEADLINE_KEYS:
                if k in df.index:
                    return float(df.loc[k, col]), f"index:{k}"
        raise ValueError(
            f"unrecognised DataFrame result for {dataset}: "
            f"columns={list(df.columns)[:20]} index={list(df.index)[:20]}"
        )

    if isinstance(result, (int, float)):
        return float(result), "scalar"

    raise ValueError(f"unrecognised result type {type(result).__name__} for {dataset}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--judge", default="exact_matching",
                    help="VLMEvalKit judge; exact_matching keeps this offline "
                         "and reproducible, which is a house property")
    ap.add_argument("--keep-eval-file", default=None,
                    help="write the intermediate xlsx here instead of a temp dir")
    args = ap.parse_args()

    try:
        preds = read_predictions(args.predictions)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return die(f"cannot read predictions {args.predictions}: {exc}")
    if not preds:
        return die("predictions file is empty — nothing to score")

    try:
        from vlmeval.dataset import build_dataset
        from vlmeval.smp import dump
    except ImportError as exc:
        return die(
            f"VLMEvalKit is not importable ({exc}).\n"
            f"  .venv-argus/Scripts/pip install -e .tools-bench/VLMEvalKit"
        )

    ds = build_dataset(args.dataset)
    if ds is None:
        return die(f"VLMEvalKit does not know dataset '{args.dataset}'")

    data = ds.data.copy()
    data["index"] = data["index"].astype(str)
    by_index = {str(v): i for i, v in enumerate(data["index"].tolist())}

    missing = [pid for pid, _ in preds if pid not in by_index]
    if missing:
        return die(
            f"{len(missing)} prediction id(s) are not {args.dataset} indices, "
            f"first few: {missing[:5]}.\n"
            f"  The corpus must be built by tools/argus_build_corpus.py so clip ids ARE "
            f"VLMEvalKit's index column; a fuzzy join here would silently score the wrong rows."
        )

    # Score exactly the rows we predicted, and no others. Handing the evaluator
    # rows with no prediction would let it count them as wrong and report a
    # number over a population the harness never ran.
    wanted = [by_index[pid] for pid, _ in preds]
    subset = data.iloc[wanted].copy()
    subset["prediction"] = [text for _, text in preds]

    # Drop the base64 image column before dumping.
    #
    # Not an optimisation — a correctness fix. openpyxl caps a cell at 32767
    # characters, so writing a 71 KB base64 image emits
    # "Cell contents too long ... truncated" and silently stores a corrupted
    # image. Text evaluators (OCRBench, DocVQA, ChartQA, TextVQA) never read it,
    # so the right move is to not write it: a column that is not there fails
    # loudly if some evaluator does want it, while a truncated one fails
    # silently and scores whatever the mangled bytes decode to.
    for col in ("image",):
        if col in subset.columns:
            subset = subset.drop(columns=[col])

    tmpdir = tempfile.mkdtemp(prefix="ffai-vlmeval-")
    eval_file = args.keep_eval_file or os.path.join(tmpdir, f"{args.dataset}_ffai.xlsx")
    os.makedirs(os.path.dirname(os.path.abspath(eval_file)), exist_ok=True)
    # VLMEvalKit's own dumper, so the format is by definition the one its
    # loader expects.
    dump(subset, eval_file)

    judge_kwargs = {"judge": args.judge, "nproc": 1, "verbose": False}
    try:
        result = ds.evaluate(eval_file, **judge_kwargs)
    except TypeError:
        # Older/newer signatures differ in which kwargs they accept.
        result = ds.evaluate(eval_file)
    except Exception as exc:  # noqa: BLE001
        return die(f"{args.dataset}.evaluate failed: {type(exc).__name__}: {exc}")

    try:
        value, key = headline_from(result, args.dataset)
    except (ValueError, TypeError) as exc:
        return die(
            f"could not read a headline score from {args.dataset}.evaluate: {exc}\n"
            f"  raw result:\n{result}"
        )

    json.dump(
        {"score": value, "metric": f"{args.dataset}:{key}", "n": len(preds)},
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
