#!/usr/bin/env python3
"""ARM 1 — the model run by VLMEvalKit's own wrapper. Prices the MODEL.

Batch-adapter contract (crates/ffai-bench/src/reference.rs):

    argv:   --batch <filelist> --corpus <corpora/argus-*.toml> --model <NAME>
    stdout: {"load_secs": <f>} then one {"path","text","transcribe_secs"} per clip

Why this arm exists
-------------------
It is the arm that answers Gate 1.1: run a published model through the
published harness and check the published row comes back. If it does not, the
scoreboard is broken and every Argus number after it would be too.

It is deliberately VLMEvalKit's *own* model wrapper — its preprocessing, its
chat template, its generation config — because a row is only comparable to a
leaderboard if it was produced the way the leaderboard produced it. The moment
we substitute our own inference for theirs, this stops being Arm 1 and becomes
Arm 2 (corpora/refs/smolvlm_hf_ref.py).

The prompts come from the corpus, which got them from VLMEvalKit's own
`build_prompt` at export time — so they are theirs end to end, and they are
hash-pinned, which the live dataset object is not.
"""

import argparse
import json
import os
import sys
import time
import tomllib


def load_prompts(corpus_path: str):
    """Map absolute clip path -> prompt, from the pinned manifest."""
    with open(corpus_path, "rb") as fh:
        man = tomllib.load(fh)
    base = os.path.dirname(os.path.abspath(corpus_path))
    out = {}
    for clip in man.get("clips", []):
        p = os.path.normcase(os.path.abspath(os.path.join(base, clip["path"])))
        out[p] = clip.get("prompt")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="filelist path")
    ap.add_argument("--corpus", required=True, help="corpora/argus-*.toml")
    ap.add_argument("--model", required=True, help="VLMEvalKit model key, e.g. SmolVLM2-256M")
    ap.add_argument("--dataset", required=True,
                    help="VLMEvalKit dataset name, e.g. OCRBench. REQUIRED, and not "
                         "cosmetic — see the note in main()")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override the wrapper's own default; leave unset to keep "
                         "the published configuration, which is the point of Arm 1")
    args = ap.parse_args()

    with open(args.batch, "r", encoding="utf-8") as fh:
        paths = [ln.strip() for ln in fh if ln.strip()]
    prompts = load_prompts(args.corpus)

    t0 = time.perf_counter()
    try:
        from vlmeval.config import supported_VLM
    except ImportError as exc:
        print(f"VLMEvalKit not importable: {exc}", file=sys.stderr)
        return 1
    if args.model not in supported_VLM:
        near = [k for k in supported_VLM if args.model.lower().split("-")[0] in k.lower()][:12]
        print(f"unknown VLMEvalKit model '{args.model}'. near matches: {near}", file=sys.stderr)
        return 1
    model = supported_VLM[args.model]()
    if args.max_new_tokens is not None and hasattr(model, "kwargs"):
        model.kwargs["max_new_tokens"] = args.max_new_tokens
    load_secs = time.perf_counter() - t0
    print(json.dumps({"load_secs": load_secs}), flush=True)

    for path in paths:
        key = os.path.normcase(os.path.abspath(path))
        prompt = prompts.get(key)
        msg = [{"type": "image", "value": path}]
        if prompt:
            msg.append({"type": "text", "value": prompt})
        t = time.perf_counter()
        try:
            # `dataset` is LOAD-BEARING and passing None silently produces a
            # different row.
            #
            # VLMEvalKit models format the prompt per dataset:
            # `SmolVLM.generate_inner` dispatches to `build_prompt_docvqa`,
            # `build_prompt_chartqa`, `build_prompt_mmbench`, … on the dataset
            # name, and each supplies its OWN instruction preamble and chat
            # markers around the question — e.g. DocVQA prepends
            # "<|im_start|>User:<image>Give a short and terse answer …".
            # With `dataset=None` the wrapper falls through to
            # `build_prompt_default` with no brief/yes-no suffix, which is a
            # materially different prompt from the one the leaderboard row was
            # produced with.
            #
            # Arm 1 exists to reproduce a published row. A silently different
            # prompt makes it reproduce nothing, while still returning
            # perfectly plausible text — the exact failure Gate 1.1 is meant to
            # catch, arriving through the arm that is supposed to be the check.
            #
            # (The dataset-level prompt is separate and already pinned: it is
            # in the corpus, put there by `dataset.build_prompt` at export
            # time. The model layer wraps it; it does not replace it, because
            # SmolVLM inherits `use_custom_prompt` -> False.)
            text = model.generate(message=msg, dataset=args.dataset)
        except Exception as exc:  # noqa: BLE001
            print(f"{path}: generate failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            text = ""
        print(
            json.dumps({
                "path": path,
                "text": text if isinstance(text, str) else str(text),
                "transcribe_secs": time.perf_counter() - t,
            }),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
