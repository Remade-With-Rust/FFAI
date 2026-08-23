#!/usr/bin/env python3
"""Harness fixture for the Argus VLM bench plumbing. THIS IS NOT A METRIC.

    ┌────────────────────────────────────────────────────────────────────┐
    │  DO NOT GROW THIS INTO A SCORER.                                    │
    │                                                                     │
    │  It exists to prove one thing: that `ffai bench vlm` writes a       │
    │  predictions file, invokes a declared external evaluator, and       │
    │  records what comes back. It deliberately does NOT read ground      │
    │  truth, does NOT extract answers, and does NOT compare anything.    │
    │  Its "score" is a function of prediction length.                     │
    │                                                                     │
    │  A VLM metric includes answer extraction, and an extractor written  │
    │  here would be tuned — however unconsciously — to our own model's   │
    │  output. That is exactly the 2.8x-biased scorer that cost the       │
    │  Carmenta campaign a year and shipped four mechanisms that turned   │
    │  out to be artifacts of it. See docs/plans/argus-launch-plan.md §5. │
    │                                                                     │
    │  Real corpora declare VLMEvalKit. If you are here because you want  │
    │  a number, you want step 0b, not this file.                          │
    └────────────────────────────────────────────────────────────────────┘

Contract (crates/ffai-bench/src/corpus.rs::ScorerSpec):

    argv:   <this> <predictions.jsonl>
    stdin:  unused
    stdout: exactly one JSON object -> {"score": <float>, "metric": <str>, "n": <int>}
    exit:   0 on success; non-zero is treated as a VOID run, never as score 0
"""

import json
import sys

VERSION = "argus-fixture-scorer 1.0 (NOT A METRIC)"


def main(argv: list[str]) -> int:
    if "--version" in argv:
        print(VERSION)
        return 0
    if len(argv) != 2:
        print(f"usage: {argv[0]} <predictions.jsonl>", file=sys.stderr)
        return 2

    path = argv[1]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln for ln in (l.strip() for l in fh) if ln]
    except OSError as exc:
        # Fail loudly. A scorer that cannot read its input has not scored 0 —
        # it has not scored, and the harness must see the difference.
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 1

    n = 0
    answered = 0
    for ln in lines:
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError as exc:
            print(f"malformed prediction line: {exc}", file=sys.stderr)
            return 1
        if "id" not in rec or "prediction" not in rec:
            print(f"prediction line missing id/prediction: {ln[:120]}", file=sys.stderr)
            return 1
        n += 1
        if rec["prediction"].strip():
            answered += 1

    # A placeholder number with no claim attached: the share of items that
    # produced any answer at all, as a percentage. It measures liveness, not
    # correctness, and the corpus declares scale = 100 so the harness's
    # normalisation path is exercised.
    score = (100.0 * answered / n) if n else 0.0
    json.dump({"score": score, "metric": "fixture", "n": n}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
