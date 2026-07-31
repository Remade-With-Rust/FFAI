"""Pin the DOCUMENT holdout for M-C3 from DocLayNet-v1.1.

M-C3 (DOCUMENT) gates on reading-order accuracy AND end-to-end CER against
PP-Structure. Nothing in the existing corpora can serve that: render and frames
are synthetic single-column, CORD is receipts, screencast is a UI. There was no
document corpus, which meant M-C3 could not have been *claimed* even if it had
been built.

## Why DocLayNet and not the obvious alternatives

- **DocLayNet-v1.1** — 80k human-annotated pages, 11 layout classes, six
  document categories, and `pdf_cells` carrying per-cell TEXT. Licence
  **CDLA-Permissive-1.0**, stated explicitly on the dataset card.
- **OmniDocBench** — purpose-built for document parsing and a better fit on
  paper (reading order, formulas, tables). **Disqualified on audit**: neither
  the HF card data nor the README states any licence. §7.1's rule is the one
  that disqualified CRAFT's Google Drive weights — an unclear licence is a no,
  regardless of how good the data is.
- **PubLayNet** — layout only, no text, and the HF mirrors 401.

DocLayNet also seeds M-C4 (LONG) for free: `original_filename` and `num_pages`
group pages back into documents, which is what a multi-page holdout needs.

## What this writes, per page

- `doclaynet-NNN.png`   the page image
- `doclaynet-NNN.txt`   plain text, `pdf_cells` in annotation order — the
                        end-to-end CER target
- `doclaynet-NNN.json`  layout regions (bbox + class) and the document
                        metadata, for the reading-order and layout gates

Streaming, because the two test shards are 1.9 GB and we want ~60 pages.
"""

import hashlib
import io
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "corpora" / "clips" / "carmenta-doclaynet"
MANIFEST = REPO / "corpora" / "carmenta-doclaynet-v1.toml"
LICENSE = "CDLA-Permissive-1.0 (DocLayNet-v1.1, ds4sd/IBM)"

# Both shards are read because DocLayNet is stored grouped by category, so a
# prefix of one shard is not a sample of the dataset — the first attempt at
# this pinned 39 pages covering four of six categories, with every financial
# report in train and every government tender in holdout.
TRAIN_FRACTION = 0.25
# DocLayNet's own category ids (1-based, COCO style).
CLASSES = [
    "caption", "footnote", "formula", "list-item", "page-footer", "page-header",
    "picture", "section-header", "table", "text", "title",
]
# Cap per document category so one collection cannot dominate the holdout, and
# split WITHIN each category so train and holdout see the same mix.
PER_CATEGORY = 10
CATEGORIES = [
    "financial_reports", "government_tenders", "laws_and_regulations",
    "manuals", "patents", "scientific_articles",
]


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from PIL import Image

    OUT.mkdir(parents=True, exist_ok=True)
    # Ranged reads over the hub rather than a 1.9 GB download: the two test
    # shards are parquet, so pyarrow can pull one row group at a time.
    fs = HfFileSystem()
    shards = [
        "datasets/ds4sd/DocLayNet-v1.1/data/test-00000-of-00002-635b47e9044a436c.parquet",
        "datasets/ds4sd/DocLayNet-v1.1/data/test-00001-of-00002-b2396409212624ed.parquet",
    ]

    def stream():
        for shard in shards:
            with fs.open(shard, "rb") as fh:
                pf = pq.ParquetFile(fh)
                for g in range(pf.num_row_groups):
                    for rec in pf.read_row_group(g).to_pylist():
                        yield rec

    rows, seen = [], {}
    for ex in stream():
        meta = ex["metadata"]
        cat = meta.get("doc_category") or "unknown"
        if seen.get(cat, 0) >= PER_CATEGORY:
            continue
        # `pdf_cells` is a list-per-region; flatten while keeping region order.
        text_lines = []
        for region in ex["pdf_cells"] or []:
            for cell in region or []:
                t = (cell.get("text") or "").strip()
                if t:
                    text_lines.append(t)
        if len(text_lines) < 5:
            continue  # a page with almost no text cannot score CER meaningfully
        seen[cat] = seen.get(cat, 0) + 1
        rows.append((cat, ex, "\n".join(text_lines)))
        if all(seen.get(c, 0) >= PER_CATEGORY for c in CATEGORIES):
            break

    # Stratify: the first TRAIN_FRACTION of EACH category goes to train, so the
    # two splits carry the same document mix. Splitting by position instead put
    # every financial report in train and every government tender in holdout.
    rows.sort(key=lambda r: r[0])
    split_of, per = {}, {}
    for idx, (cat, _, _) in enumerate(rows):
        n = per.get(cat, 0)
        per[cat] = n + 1
        split_of[idx] = "train" if n < round(PER_CATEGORY * TRAIN_FRACTION) else "holdout"

    entries = []
    for i, (cat, ex, text) in enumerate(rows):
        stem = f"doclaynet-{i:03}"
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img["bytes"]))
        img.convert("RGB").save(OUT / f"{stem}.png")
        (OUT / f"{stem}.txt").write_text(text, encoding="utf-8")

        meta = ex["metadata"]
        regions = [
            {"bbox": [round(float(v), 2) for v in b],
             "class": CLASSES[c - 1] if 1 <= c <= len(CLASSES) else f"id{c}"}
            for b, c in zip(ex["bboxes"], ex["category_id"])
        ]
        (OUT / f"{stem}.json").write_text(json.dumps({
            "regions": regions,
            "doc_category": meta.get("doc_category"),
            # LONG (M-C4) groups pages back into documents by these two.
            "original_filename": meta.get("original_filename"),
            "num_pages": meta.get("num_pages"),
            "page_size": [meta.get("coco_width"), meta.get("coco_height")],
        }, indent=1), encoding="utf-8")

        digest = hashlib.sha256((OUT / f"{stem}.png").read_bytes()).hexdigest()
        entries.append((stem, cat, split_of[i], digest))

    lines = ['name = "carmenta-doclaynet"', "version = 1", 'task = "ocr"', ""]
    for stem, cat, split, digest in entries:
        lines += [
            "[[clips]]",
            f'id = "{stem}"',
            f'path = "clips/carmenta-doclaynet/{stem}.png"',
            f'ground_truth = "clips/carmenta-doclaynet/{stem}.txt"',
            f'class = "{cat}"',
            f'split = "{split}"',
            f'license = "{LICENSE}"',
            f'sha256 = "{digest}"',
            "",
        ]
    MANIFEST.write_text("\n".join(lines), encoding="utf-8")

    counts = {}
    for _, cat, split, _ in entries:
        counts[(cat, split)] = counts.get((cat, split), 0) + 1
    print(f"wrote {len(entries)} pages to {OUT}")
    for k in sorted(counts):
        print(f"   {k[0]:24s} {k[1]:8s} {counts[k]}")
    print(f"manifest: {MANIFEST.relative_to(REPO)}")


if __name__ == "__main__":
    main()
