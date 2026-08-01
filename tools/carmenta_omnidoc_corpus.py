"""Pin the OmniDocBench text subset — the first Carmenta document corpus we did
not build ourselves.

Every document number this campaign has is on `carmenta-doc-v1`, which we
render. That is right for engineering — it is how §8.28 attributed 55.75 pp to
reading order — and useless for a comparable claim: nobody can place 0.32 %
against anything they know. OmniDocBench is the board Baidu's Unlimited-OCR
states its headline on (93.23 % v1.5), and it is Apache-2.0.

## What this scores, and what it does not

OmniDocBench's headline is a COMPOSITE that blends TEDS (table structure) and
CDM (formula) with text metrics. Carmenta produces neither tables nor formulas,
so quoting that number would be false. What is directly ours to measure is the
END-TO-END TEXT metric, and their annotations hand it over: every region
carries `order` and `text`, so ground truth in reading order is regions sorted
by `order` with their text concatenated — exactly the format our harness
already scores.

So the subset is filtered to what we can honestly be measured on:

  * `language == english` (our recognizers are English-only),
  * no `table` and no `equation_isolated` region on the page,
  * `ignore` regions and the `abandon` class dropped, as the benchmark intends.

316 pages survive (regions with a null `order` are dropped, not defaulted —
see below). The page attribute `layout` is carried into the sidecar so
results split by single / double / three column — the axis that made
`carmenta-doc` informative, and the one that tests whether our column code
generalises past the two columns it was measured on.

Clips are not committed (`/corpora/clips/` is gitignored); the manifest's
SHA-256s are what travel.
"""

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = Path("C:/odb")
OUT = REPO / "corpora" / "clips" / "carmenta-omnidoc"
MANIFEST = REPO / "corpora" / "carmenta-omnidoc-v1.toml"
LICENSE = "Apache-2.0 (OmniDocBench, opendatalab)"

# Regions that carry prose we are expected to emit. `abandon` is the
# benchmark's own bucket for content excluded from scoring.
TEXT_CATS = {"text_block", "title", "header", "footer", "figure_caption", "page_number"}
EXCLUDE_PAGE_IF = {"table", "equation_isolated"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pages = json.loads((SRC / "OmniDocBench.json").read_text(encoding="utf-8"))

    kept = []
    for p in pages:
        info = p["page_info"]
        attr = info.get("page_attribute", {})
        if attr.get("language") != "english":
            continue
        cats = {d["category_type"] for d in p["layout_dets"]}
        if cats & EXCLUDE_PAGE_IF:
            continue
        regions = [
            d for d in p["layout_dets"]
            if d["category_type"] in TEXT_CATS
            and str(d.get("ignore", "False")) != "True"
            and (d.get("text") or "").strip()
            # A region with a null `order` has no place in the reading
            # sequence, and inventing one would be scoring our guess against
            # their silence. Dropped, not defaulted.
            and d.get("order") is not None
        ]
        if len(regions) < 3:
            continue  # too little text for a CER to mean anything
        regions.sort(key=lambda d: int(d.get("order", 0)))
        img = SRC / "images" / info["image_path"]
        if not img.exists():
            continue
        kept.append((info, attr, regions, img))

    # Stratify the split by layout so both halves carry every column count.
    by_layout = defaultdict(list)
    for row in kept:
        by_layout[row[1].get("layout", "unknown")].append(row)

    lines = ['name = "carmenta-omnidoc"', "version = 1", 'task = "ocr"', ""]
    counts, n = Counter(), 0
    for layout in sorted(by_layout):
        for i, (info, attr, regions, img) in enumerate(by_layout[layout]):
            stem = f"omni-{n:04}"
            shutil.copyfile(img, OUT / f"{stem}.png")
            (OUT / f"{stem}.txt").write_text(
                "\n".join(r["text"].strip() for r in regions), encoding="utf-8")
            (OUT / f"{stem}.json").write_text(json.dumps({
                "layout": layout,
                "data_source": attr.get("data_source"),
                "page_size": [info.get("width"), info.get("height")],
                "source_image": info["image_path"],
                "regions": [
                    {"order": int(r.get("order", 0)), "class": r["category_type"],
                     "poly": r.get("poly"), "text": r["text"]}
                    for r in regions
                ],
            }, indent=1), encoding="utf-8")

            split = "train" if i % 4 == 0 else "holdout"
            digest = hashlib.sha256((OUT / f"{stem}.png").read_bytes()).hexdigest()
            lines += [
                "[[clips]]",
                f'id = "{stem}"',
                f'path = "clips/carmenta-omnidoc/{stem}.png"',
                f'ground_truth = "clips/carmenta-omnidoc/{stem}.txt"',
                'class = "document_scan"',
                f'split = "{split}"',
                f'license = "{LICENSE}"',
                f'sha256 = "{digest}"',
                "",
            ]
            counts[(layout, split)] += 1
            n += 1

    MANIFEST.write_text("\n".join(lines), encoding="utf-8")
    print(f"carmenta-omnidoc: {n} pages")
    for k in sorted(counts):
        print(f"   {k[0]:16s} {k[1]:8s} {counts[k]}")
    print(f"manifest: {MANIFEST.relative_to(REPO)}")


if __name__ == "__main__":
    main()
