#!/usr/bin/env python3
"""Export a VLMEvalKit dataset into an FFai `corpora/argus-*.toml`.

Step 0b of docs/plans/argus-launch-plan.md. Turns a benchmark VLMEvalKit
already knows about into a corpus with FFai's discipline attached: every image
hash-pinned, every prompt inline (and therefore inside the manifest
fingerprint), the benchmark's own evaluator declared as the `[scorer]`.

    .venv-argus/Scripts/python.exe tools/argus_build_corpus.py \
        --dataset OCRBench --limit 200 --out corpora/argus-ocrbench-v1.toml

THE PROMPT COMES FROM VLMEvalKit, NOT FROM US
---------------------------------------------
`dataset.build_prompt(row)` is what VLMEvalKit feeds its own models: question
text, multiple-choice options, and any dataset-specific instruction suffix
("Answer the question using a single word or phrase."). Composing that string
ourselves would be a home-grown prompt, and a prompt is part of a benchmark
exactly the way answer extraction is — reword it and the score moves. So we
take theirs verbatim and pin it.

The `index` column is preserved as the clip id, because that is the key
VLMEvalKit's evaluator joins predictions on. Lose it and no score can be
computed.
"""

import argparse
import base64
import hashlib
import io
import os
import sys

# Scale of the raw score each benchmark reports, so `ScorerSpec::scale` is a
# DECLARED fact rather than something inferred from an observed value. A wrong
# guess here is invisible: the normalised number still looks plausible.
#
# Anything not listed must be added deliberately after reading that dataset's
# evaluator, never defaulted — hence the hard failure below.
# `PER_ITEM` means the evaluator returns a COUNT of correct answers, so the
# normaliser is the number of items scored, not a constant. Getting this wrong
# is silent: OCRBench was first declared at a flat 1000.0 (its full-benchmark
# maximum) and a genuinely good 40-of-50 subset run normalised to 0.04, which
# the quality gate would have read as a catastrophic failure.
#
# HOW TO VERIFY A SCALE — the oracle probe, and it takes one minute:
# feed the dataset's OWN ground-truth answers back as predictions. A correct
# scale makes that score exactly 1.0 normalised. A garbage-prediction run
# scoring 0 proves nothing on its own, because a BROKEN JOIN also scores 0 —
# you need both ends.
PER_ITEM = "PER_ITEM"

KNOWN_SCALES = {
    # verified by oracle probe 2026-08-21: 50/50 ground truth -> 50.0
    "OCRBench": (PER_ITEM, "OCRBench"),
    "OCRBench_v2": (100.0, "OCRBench_v2"),
    "DocVQA_VAL": (100.0, "ANLS"),
    "DocVQA_TEST": (100.0, "ANLS"),
    "ChartQA_TEST": (100.0, "acc"),
    "InfoVQA_VAL": (100.0, "ANLS"),
    "TextVQA_VAL": (100.0, "acc"),
    "MMStar": (100.0, "acc"),
    "MMBench_V11": (100.0, "acc"),
    "AI2D_TEST": (100.0, "acc"),
    "MMMU_DEV_VAL": (100.0, "acc"),
    "MathVista_MINI": (100.0, "acc"),
    "HallusionBench": (100.0, "aAcc"),
    "COCO_VAL": (100.0, "CIDEr"),
}

# Scales below OCRBench are DECLARED FROM DOCUMENTATION AND NOT YET ORACLE-
# PROBED. Run the probe before quoting a number from any of them.
UNVERIFIED_SCALES = {k for k in KNOWN_SCALES if k != "OCRBench"}

# Which FFai ContentClass each benchmark's imagery is. Reporting is stratified
# by class, so a wrong label here makes a per-class table lie.
CLASS_OF = {
    "OCRBench": "scene_text",
    "OCRBench_v2": "scene_text",
    "DocVQA_VAL": "document_scan",
    "DocVQA_TEST": "document_scan",
    "InfoVQA_VAL": "diagram",
    "ChartQA_TEST": "diagram",
    "AI2D_TEST": "diagram",
    "TextVQA_VAL": "scene_text",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def toml_escape(s: str) -> str:
    """Escape for a TOML basic string. Prompts contain quotes and newlines."""
    out = s.replace("\\", "\\\\").replace('"', '\\"')
    out = out.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return out


def text_of_prompt(msgs) -> str:
    """Pull the text out of VLMEvalKit's message list, in order.

    `build_prompt` returns a list of {'type': 'image'|'text', 'value': ...}.
    We keep only the text parts: the image is already a corpus clip, and the
    engine is handed it separately.
    """
    if isinstance(msgs, str):
        return msgs
    parts = []
    for m in msgs:
        if isinstance(m, dict) and m.get("type") == "text":
            parts.append(str(m.get("value", "")))
    return "\n".join(p for p in parts if p).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="VLMEvalKit dataset name, e.g. OCRBench")
    ap.add_argument("--out", required=True, help="output corpora/argus-*.toml")
    ap.add_argument("--clips-dir", default=None, help="default corpora/clips/<slug>")
    ap.add_argument("--limit", type=int, default=0, help="first N items (0 = all)")
    ap.add_argument("--holdout-every", type=int, default=1,
                    help="1 = every item is holdout; N>1 marks 1-in-N as train")
    ap.add_argument("--license", default=None,
                    help="license string recorded per clip; defaults to a string "
                         "naming the dataset's own terms")
    args = ap.parse_args()

    if args.dataset not in KNOWN_SCALES:
        print(
            f"error: no declared score scale for dataset '{args.dataset}'.\n"
            f"       Add it to KNOWN_SCALES after reading that dataset's evaluator.\n"
            f"       Guessing a scale silently rescales someone's benchmark.\n"
            f"       known: {', '.join(sorted(KNOWN_SCALES))}",
            file=sys.stderr,
        )
        return 2

    scale, metric = KNOWN_SCALES[args.dataset]
    slug = f"argus-{args.dataset.lower().replace('_', '-')}"
    # The corpus NAME comes from the output file stem, not from the dataset.
    #
    # It used to be derived from the dataset alone, which meant a 50-item
    # subset and the full 1000-item benchmark both wrote `argus-ocrbench-v1`
    # into the ledger. Their manifest hashes differ, so they are technically
    # distinguishable — but a reader scanning the `corpus` column would see one
    # name and reasonably assume one benchmark. A subset is a different
    # measurement and its name has to say so.
    corpus_name = os.path.splitext(os.path.basename(args.out))[0]
    clips_dir = args.clips_dir or os.path.join("corpora", "clips", slug)
    os.makedirs(clips_dir, exist_ok=True)

    from vlmeval.dataset import build_dataset  # noqa: E402

    print(f"building dataset {args.dataset} (downloads its TSV on first run) ...", file=sys.stderr)
    ds = build_dataset(args.dataset)
    if ds is None:
        print(f"error: VLMEvalKit does not know dataset '{args.dataset}'", file=sys.stderr)
        return 1
    data = ds.data
    total = len(data)
    n = total if args.limit <= 0 else min(args.limit, total)
    print(f"  {total} items, exporting {n}", file=sys.stderr)

    from PIL import Image  # noqa: E402

    rows = []
    for i in range(n):
        line = data.iloc[i]
        index = str(line["index"])

        # Image: VLMEvalKit stores base64 in `image`, or a path in `image_path`.
        raw = None
        if "image" in data.columns and line.get("image") is not None:
            val = line["image"]
            if isinstance(val, list):
                val = val[0]
            if isinstance(val, str) and val:
                raw = base64.b64decode(val)
        if raw is None and "image_path" in data.columns:
            p = line.get("image_path")
            if isinstance(p, list):
                p = p[0]
            if p and os.path.exists(p):
                raw = open(p, "rb").read()
        if raw is None:
            print(f"  skip {index}: no image", file=sys.stderr)
            continue

        # Normalise to PNG so the pinned bytes are a format our decoders read
        # and the hash is stable across re-exports of the same source.
        try:
            im = Image.open(io.BytesIO(raw))
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=False)
            raw = buf.getvalue()
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {index}: undecodable image ({exc})", file=sys.stderr)
            continue

        rel = os.path.join("clips", slug, f"{index}.png").replace("\\", "/")
        dest = os.path.join("corpora", "clips", slug, f"{index}.png")
        with open(dest, "wb") as fh:
            fh.write(raw)

        prompt = text_of_prompt(ds.build_prompt(line))
        split = "train" if (args.holdout_every > 1 and i % args.holdout_every == 0) else "holdout"
        rows.append((index, rel, prompt, split, sha256_bytes(raw)))

    if not rows:
        print("error: exported nothing", file=sys.stderr)
        return 1

    holdout = sum(1 for r in rows if r[3] == "holdout")
    lic = args.license or (
        f"{args.dataset} — see the dataset's own terms; images are NOT redistributed "
        f"by this repo (corpora/clips/ is gitignored, the manifest is the committed artifact)"
    )
    subset_note = ""
    if len(rows) < total:
        subset_note = "\n".join([
            "#",
            f"# !! SUBSET: {len(rows)} of {total} items. A score from this corpus is NOT",
            f"#    comparable to a published {args.dataset} row, which is computed over all",
            f"#    {total} items. Use it to exercise the pipeline end to end; use the FULL",
            "#    export for Gate 1.1, which is the gate that asks whether a published row",
            "#    reproduces. A subset answers a different question and must not be quoted",
            "#    as if it answered that one.",
        ]) + "\n"
    # Resolve a per-item scale now that the holdout count is known. The scorer
    # scores exactly the holdout, so that count is the normaliser.
    if scale == PER_ITEM:
        scale = float(holdout)
        scale_note = "\n".join([
            f"# scale = {holdout} because {args.dataset}'s evaluator returns a COUNT of",
            "# correct answers, not a percentage - so the normaliser is the number of",
            "# holdout items, and it CHANGES if the holdout changes. Re-export rather",
            "# than editing the split by hand.",
            "#",
            "# Verified by oracle probe: feeding this dataset's own ground-truth answers",
            "# back as predictions scores exactly the item count, i.e. 1.0 normalised.",
        ]) + "\n"
    elif args.dataset in UNVERIFIED_SCALES:
        scale_note = "\n".join([
            f"# !! scale {scale} is declared from documentation and has NOT been",
            "# oracle-probed. Feed the dataset's own answers back as predictions and",
            "# check the result normalises to 1.0 before quoting any score from it.",
            "# A garbage-prediction run scoring 0 does NOT verify a scale, because a",
            "# broken join scores 0 too.",
        ]) + "\n"
    else:
        scale_note = ""

    ver_cmd = '["' + '", "'.join([
        ".venv-argus/Scripts/python.exe", "-c",
        "import importlib.metadata as m; print('vlmevalkit', m.version('vlmeval'))",
    ]) + '"]'

    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"""# {args.dataset}, exported from VLMEvalKit by tools/argus_build_corpus.py.
#
# GENERATED — re-export rather than hand-edit. Every prompt below is
# VLMEvalKit's own `build_prompt` output, verbatim: the question, its options
# and its instruction suffix are part of the benchmark, and rewording any of
# them would measure something else. They are inline so they fall inside
# `Manifest::manifest_hash`.
#
# The clip id IS VLMEvalKit's `index` column. That is the key its evaluator
# joins predictions on — renaming it makes the corpus unscoreable.
#
# Scoring is VLMEvalKit's, never ours (plan §5). See corpora/refs/vlmevalkit_score.py.
{subset_note}
name = "{corpus_name}"
version = 1
task = "vlm"

# The scorer is SELECTED by name, not defined here. Its argv lives in
# corpora/references.toml under [[scorer]], which is read and reviewed as
# executable input — a corpus is data, and data must not carry commands.
[scorer]
name = "vlmevalkit-{args.dataset.lower()}"
metric = "{metric}"
{scale_note}scale = {scale}
""")
        cls = CLASS_OF.get(args.dataset, "other")
        for index, rel, prompt, split, sha in rows:
            fh.write("\n[[clips]]\n")
            fh.write(f'id = "{toml_escape(index)}"\n')
            fh.write(f'path = "{rel}"\n')
            fh.write(f'prompt = "{toml_escape(prompt)}"\n')
            fh.write(f'class = "{cls}"\n')
            fh.write(f'split = "{split}"\n')
            fh.write(f'license = "{toml_escape(lic)}"\n')
            fh.write(f'sha256 = "{sha}"\n')

    print(f"wrote {args.out}: {len(rows)} clips ({holdout} holdout) -> {clips_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
