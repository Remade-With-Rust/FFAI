"""Build the diana-coco detection corpus from COCO val2017 (Diana M-D0).

Deterministic, audit-first corpus construction — the pyannote lesson applied
to images, and every rule below is a corpus-design decision recorded here
rather than in anyone's memory:

- **Source**: COCO 2017 validation split. Annotations come from the pinned
  `corpora/cache/annotations_trainval2017.zip` (instances_val2017.json,
  annotations licensed CC-BY-4.0); images are fetched one by one, ungated,
  from `images.cocodataset.org/val2017/` (verified no-auth 2026-07-30).
- **License filter**: COCO images carry individual Flickr licenses (ids 1-8
  in the annotation file's `licenses` table). Only images under
  Attribution (4), Attribution-ShareAlike (5), no-known-copyright (7), or
  US-Government-Work (8) are eligible — the NC/ND classes are excluded so
  the corpus can back public claims. The license NAME is recorded per clip
  in the manifest.
- **Crowd exclusion**: images containing any `iscrowd=1` annotation are
  ineligible. pycocotools treats crowd regions as ignore-zones during
  matching; the ffai-bench proxy scorer (crates/ffai-bench/src/detect.rs)
  deliberately does not implement ignore logic, so the corpus excludes the
  cases where it would matter instead of silently mis-scoring them.
- **Selection**: eligible images sorted by id, then a fixed stride walk
  selects COUNT of them — deterministic, spread across the whole split, no
  RNG anywhere.
- **Source cache keyed by COCO file name, not by clip index.** The clip id
  `coco-NNN` is a *position in the stride walk*, so it names a different
  COCO image at every COUNT. Caching the downloaded JPEG under the clip id
  (as this script first did) means raising COUNT silently rebuilds the
  corpus from whatever images the previous COUNT had left on disk, pairing
  image N with image M's ground truth. Nothing would fail — mAP would just
  quietly be wrong. The cache lives in `corpora/cache/coco-val2017/` under
  COCO's own file name, which is stable by construction and shared across
  every corpus version.
- **Split**: `i % 4 == 3` -> train, else holdout (the prepare_carmenta_synth
  rule), giving 45 holdout / 15 train at COUNT=60.
- **Stored as PNG, not the source JPEG.** COCO ships JPEG; the corpus
  decodes it once with Pillow and re-encodes losslessly to PNG. Two
  reasons, and the second is the load-bearing one: FFai's image ingest is
  PNG-only until the rff decoders land (ROADMAP Phase 3), and — more
  importantly — every implementation then reads *identical pixels*. Left
  as JPEG, our decoder and PaddleOCR's and Ultralytics' would each produce
  slightly different arrays, and the resulting mAP delta would be a
  decoder comparison wearing a detector's clothes. The pixels are the
  corpus; the container is not.
- **Ground truth**: one JSON per image:
  `{"width": W, "height": H, "objects": [[x0,y0,x1,y1,cls], ...]}` with
  boxes xyxy in original pixels (COCO xywh converted, 2dp) and `cls` the
  contiguous 0-79 index of category ids sorted ascending — the Ultralytics
  convention. The mapping is written to `classes.json` beside the clips.
- **Provenance**: `provenance.json` records each image's coco_url,
  flickr_url and license so the manifest's license strings are checkable.

Usage (from the repo root):
    python tools/diana_coco_corpus.py                    # v2, 60 clips
    python tools/diana_coco_corpus.py --count 600 --version 3

Each version gets its own clips directory and manifest, so an earlier
corpus — and every ledger row that names its hash — stays reproducible.

Idempotent: already-downloaded images are kept if their bytes hash the same.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_ZIP = ROOT / "corpora" / "cache" / "annotations_trainval2017.zip"
# Shared across corpus versions and keyed by COCO's own file name — see the
# "source cache" note in the module docstring for why the clip id must not
# be used here.
SRC_CACHE = ROOT / "corpora" / "cache" / "coco-val2017"
IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"

ALLOWED_LICENSE_IDS = {4, 5, 7, 8}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(im: dict) -> Path:
    """Download one val2017 image into the shared cache, keyed by its name."""
    dst = SRC_CACHE / im["file_name"]
    if not dst.exists():
        url = IMAGE_URL.format(file_name=im["file_name"])
        with urllib.request.urlopen(url, timeout=60) as resp:
            body = resp.read()
        # Write via a temp name so an interrupted run cannot leave a
        # truncated file that the next run happily treats as cached.
        tmp = dst.with_suffix(dst.suffix + ".part")
        tmp.write_bytes(body)
        tmp.replace(dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=60, help="clips to select")
    ap.add_argument("--version", type=int, default=2, help="corpus version")
    ap.add_argument("--jobs", type=int, default=16, help="parallel downloads")
    args = ap.parse_args()
    count: int = args.count

    # v2 predates versioned directories and its manifest points at the
    # unsuffixed path; leave it exactly where it is.
    clips_dir = ROOT / "corpora" / "clips" / (
        "diana-coco" if args.version == 2 else f"diana-coco-v{args.version}"
    )
    manifest = ROOT / "corpora" / f"diana-coco-v{args.version}.toml"

    if not CACHE_ZIP.exists():
        sys.exit(
            f"missing {CACHE_ZIP} — download it first:\n"
            "  curl -o corpora/cache/annotations_trainval2017.zip "
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        )
    print(f"reading instances_val2017.json from {CACHE_ZIP.name} ...")
    with zipfile.ZipFile(CACHE_ZIP) as zf:
        with zf.open("annotations/instances_val2017.json") as f:
            inst = json.load(io.TextIOWrapper(f, encoding="utf-8"))

    licenses = {l["id"]: l["name"] for l in inst["licenses"]}
    cat_ids = sorted(c["id"] for c in inst["categories"])
    cat_to_contig = {cid: i for i, cid in enumerate(cat_ids)}
    cat_names = {c["id"]: c["name"] for c in inst["categories"]}

    by_image: dict[int, list[dict]] = {}
    for ann in inst["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    images = {im["id"]: im for im in inst["images"]}
    eligible = []
    for iid in sorted(images):
        im = images[iid]
        anns = by_image.get(iid, [])
        if not anns:
            continue
        if im["license"] not in ALLOWED_LICENSE_IDS:
            continue
        if any(a.get("iscrowd", 0) == 1 for a in anns):
            continue
        eligible.append(iid)
    print(
        f"{len(images)} val2017 images -> {len(eligible)} eligible "
        f"(license in {sorted(ALLOWED_LICENSE_IDS)}, has objects, no crowd regions)"
    )
    if len(eligible) < count:
        sys.exit(f"only {len(eligible)} eligible images for --count {count}")

    stride = len(eligible) // count
    selected = [eligible[i * stride] for i in range(count)]

    # Fetch first, in parallel: the network is the slow part and it is the
    # part with nothing to serialise on.
    SRC_CACHE.mkdir(parents=True, exist_ok=True)
    todo = [images[iid] for iid in selected if not (SRC_CACHE / images[iid]["file_name"]).exists()]
    if todo:
        print(f"fetching {len(todo)} images ({len(selected) - len(todo)} already cached) ...")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for i, _ in enumerate(pool.map(fetch, todo), 1):
                if i % 25 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}]")
    else:
        print(f"all {len(selected)} images already cached")

    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / "classes.json").write_text(
        json.dumps(
            {str(cat_to_contig[cid]): cat_names[cid] for cid in cat_ids}, indent=1
        ),
        encoding="utf-8",
    )

    manifest_lines = [
        f"# diana-coco v{args.version} — COCO val2017 subset for `ffai bench detect`.",
        "# Generated by tools/diana_coco_corpus.py — selection, license filter,",
        "# crowd exclusion and split rule are documented (and enforced) there.",
        "# Annotations: CC-BY-4.0 (COCO Consortium). Images: individual Flickr",
        "# licenses, filtered to Attribution/Attribution-ShareAlike/no-known-",
        "# copyright/US-Gov — the per-clip `license` field records which.",
        'name = "diana-coco"',
        f"version = {args.version}",
        'task = "detect"',
    ]
    provenance = {}
    for i, iid in enumerate(selected):
        im = images[iid]
        clip_id = f"coco-{i:03d}"
        src_path = SRC_CACHE / im["file_name"]
        img_path = clips_dir / f"{clip_id}.png"
        gt_path = clips_dir / f"{clip_id}.json"

        if not img_path.exists():
            from PIL import Image

            with Image.open(src_path) as jpg:
                jpg.convert("RGB").save(img_path, format="PNG", optimize=True)
            if (i + 1) % 50 == 0 or i + 1 == count:
                print(f"  encoded [{i + 1}/{count}]")

        objects = []
        for ann in by_image[iid]:
            x, y, w, h = ann["bbox"]
            objects.append(
                [
                    round(x, 2),
                    round(y, 2),
                    round(x + w, 2),
                    round(y + h, 2),
                    cat_to_contig[ann["category_id"]],
                ]
            )
        gt_path.write_text(
            json.dumps(
                {"width": im["width"], "height": im["height"], "objects": objects}
            ),
            encoding="utf-8",
        )

        provenance[clip_id] = {
            "coco_image_id": iid,
            "file_name": im["file_name"],
            "coco_url": im.get("coco_url", ""),
            "flickr_url": im.get("flickr_url", ""),
            "license": licenses[im["license"]],
        }
        split = "train" if i % 4 == 3 else "holdout"
        rel = clips_dir.relative_to(ROOT / "corpora").as_posix()
        manifest_lines += [
            "",
            "[[clips]]",
            f'id = "{clip_id}"',
            f'path = "{rel}/{clip_id}.png"',
            f'ground_truth = "{rel}/{clip_id}.json"',
            'class = "photo"',
            f'split = "{split}"',
            f'license = "{licenses[im["license"]]} (Flickr; COCO 2017 annotations CC-BY-4.0)"',
            f'sha256 = "{sha256_file(img_path)}"',
        ]

    (clips_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=1), encoding="utf-8"
    )
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    n_hold = sum(1 for i in range(count) if i % 4 != 3)
    print(f"wrote {manifest.name}: {count} clips ({n_hold} holdout / {count - n_hold} train)")
    print(f"clips + ground truth + classes.json + provenance.json in {clips_dir}")


if __name__ == "__main__":
    main()
