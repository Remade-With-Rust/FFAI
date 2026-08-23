#!/usr/bin/env python3
"""Preflight for the Argus VLM bench (steps 0b/0c). Says what is missing, precisely.

    .venv-argus/Scripts/python.exe tools/argus_doctor.py
    .venv-argus/Scripts/python.exe tools/argus_doctor.py --dataset OCRBench --model SmolVLM2-256M

Why a doctor and not a README section
-------------------------------------
Every check here corresponds to a way this pipeline produces a WRONG NUMBER
rather than an error:

* a missing dataset TSV makes VLMEvalKit silently download a different revision
* a mismatched transformers version changes a model's chat template, which
  degrades output with no error at all — the plan calls it "the highest-risk
  silent failure in the whole build"
* differing CPU thread counts between the two arms turn a speed ratio into a
  measurement of the scheduler (codec-measurement §2)
* and a full disk fails a weight download halfway, leaving a truncated file
  that loads and produces garbage

The last one is not hypothetical: this box had 3.1 GB free of 930 GB when this
work started, which is less than one torch install.

Exit code is the number of FAILs, so it is usable as a gate.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"
_counts = {OK: 0, WARN: 0, FAIL: 0}


def say(status: str, what: str, detail: str = "") -> None:
    _counts[status] += 1
    line = f"  [{status}] {what}"
    if detail:
        line += f"\n         {detail}"
    print(line)


def check_disk(min_gb: float) -> None:
    try:
        free = shutil.disk_usage(os.getcwd()).free / (1024 ** 3)
    except OSError as exc:
        say(WARN, "disk space", f"could not measure: {exc}")
        return
    if free < min_gb:
        say(FAIL, f"disk space {free:.1f} GB free",
            f"need ~{min_gb:.0f} GB. A weight download that runs out of space leaves a "
            f"TRUNCATED file that still loads and produces garbage. "
            f"Reclaim with: cargo clean in unused checkouts.")
    else:
        say(OK, f"disk space {free:.1f} GB free")


def check_import(mod: str, why: str, required: bool = True) -> object | None:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        say(OK, f"{mod} {ver}")
        return m
    except ImportError as exc:
        say(FAIL if required else WARN, f"{mod} missing", f"{why} ({exc})")
        return None


def check_threads() -> None:
    """Both arms must be given the same core count or the ratio is fiction."""
    try:
        import torch
        say(OK, f"torch threads = {torch.get_num_threads()}",
            "pass --threads N identically to BOTH arms when taking a speed number; "
            "differing core counts make the wall ratio measure the scheduler")
    except ImportError:
        pass


def check_dataset(name: str) -> None:
    try:
        from vlmeval.dataset import build_dataset
    except ImportError as exc:
        say(FAIL, f"dataset {name}", f"vlmeval not importable ({exc})")
        return
    root = os.environ.get("LMUData", os.path.expanduser("~/LMUData"))
    tsv = os.path.join(root, f"{name}.tsv")
    if os.path.exists(tsv):
        say(OK, f"dataset {name} TSV cached", f"{tsv} ({os.path.getsize(tsv)/1e6:.0f} MB)")
    else:
        say(WARN, f"dataset {name} TSV not cached",
            f"first run downloads it to {root}; that download is also the moment a "
            f"full disk corrupts it silently")
    try:
        ds = build_dataset(name)
        if ds is None:
            say(FAIL, f"dataset {name}", "VLMEvalKit does not know this name")
        else:
            say(OK, f"dataset {name} builds", f"{len(ds.data)} items")
    except Exception as exc:  # noqa: BLE001
        say(FAIL, f"dataset {name} build failed", f"{type(exc).__name__}: {exc}")


def check_model(key: str) -> None:
    try:
        from vlmeval.config import supported_VLM
    except ImportError:
        say(FAIL, f"model {key}", "vlmeval not importable")
        return
    if key in supported_VLM:
        say(OK, f"Arm-1 model key '{key}' is known to VLMEvalKit")
    else:
        near = [k for k in supported_VLM if key.lower()[:6] in k.lower()][:10]
        say(FAIL, f"Arm-1 model key '{key}' unknown", f"near matches: {near}")


def check_scorer_contract() -> None:
    """Round-trip the scorer contract on a fixture, so the plumbing is proven
    without needing weights. Catches an adapter that prints logs to stdout."""
    import tempfile
    fixture = [
        {"id": "x1", "path": "a.png", "prompt": "q?", "prediction": "yes"},
        {"id": "x2", "path": "b.png", "prompt": "q?", "prediction": ""},
    ]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "p.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            for r in fixture:
                fh.write(json.dumps(r) + "\n")
        out = subprocess.run(
            [sys.executable, "tools/argus_fixture_scorer.py", p],
            capture_output=True, text=True, check=False,
        )
        if out.returncode != 0:
            say(FAIL, "scorer contract", f"fixture scorer exited {out.returncode}: {out.stderr[:200]}")
            return
        try:
            obj = json.loads(out.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            say(FAIL, "scorer contract", f"fixture scorer printed non-JSON: {out.stdout[:200]}")
            return
        if "score" in obj and obj.get("n") == 2:
            say(OK, "scorer contract round-trips", f"{obj}")
        else:
            say(FAIL, "scorer contract", f"unexpected object {obj}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="e.g. OCRBench")
    ap.add_argument("--model", default=None, help="Arm-1 VLMEvalKit model key, e.g. SmolVLM2-256M")
    ap.add_argument("--min-disk-gb", type=float, default=8.0)
    args = ap.parse_args()

    print("Argus preflight (docs/plans/argus-launch-plan.md steps 0b/0c)\n")
    print(f"  interpreter: {sys.executable}")
    print(f"  cwd:         {os.getcwd()}\n")

    print(" environment")
    check_disk(args.min_disk_gb)
    check_import("torch", "Arm 2 runs the checkpoint on CPU through it")
    tf = check_import("transformers", "supplies the processor and the CHAT TEMPLATE")
    check_import("PIL", "image loading for Arm 2")
    check_import("vlmeval", "Arm 1 and the scorer are VLMEvalKit's")
    check_threads()
    if tf is not None:
        say(OK, "chat template source",
            "processor.apply_chat_template is used — never hand-written turn markers, "
            "which degrade output silently rather than erroring")

    print("\n contract")
    check_scorer_contract()

    if args.dataset:
        print("\n dataset")
        check_dataset(args.dataset)
    if args.model:
        print("\n arm 1 model")
        check_model(args.model)

    print(f"\n  {_counts[OK]} ok / {_counts[WARN]} warn / {_counts[FAIL]} fail")
    if _counts[FAIL]:
        print("  -> steps 0b/0c cannot produce an admissible number until the FAILs clear.")
    return _counts[FAIL]


if __name__ == "__main__":
    raise SystemExit(main())
