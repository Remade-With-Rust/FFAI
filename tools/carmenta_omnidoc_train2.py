"""§8.164: the TRAIN-ONLY corpus extension — curing the 80-page fitting famine.

Every rule this campaign fitted died the same death: 80 train pages cannot
select against. The 236-page holdout can never donate (every page has been read
by judged experiments), so new train material comes from UPSTREAM: the 362
English OmniDocBench pages v1 EXCLUDED for containing tables or isolated
equations. Those exclusions were right for SCORING (we emit neither) and are
irrelevant for FITTING - table innards are precisely the orphan junk the
suppression rules starve for, and the pages arrive with regions, order and text
already annotated. 70 of them are exam_paper against the 12 we have today.

Same sidecar format as v1, ids `omnx-*` so nothing collides, split=train on
every clip, separate manifest so the scored corpus is untouched. NOTHING here
may ever be quoted as a holdout number.
"""
import hashlib, json, shutil
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = Path("C:/odb")
OUT = REPO / "corpora" / "clips" / "carmenta-omnidoc-train2"
MANIFEST = REPO / "corpora" / "carmenta-omnidoc-train2-v1.toml"
TEXT_CATS = {"text_block", "title", "header", "footer", "figure_caption", "page_number"}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pages = json.loads((SRC / "OmniDocBench.json").read_text(encoding="utf-8"))
    kept = []
    for p in pages:
        info = p["page_info"]; attr = info.get("page_attribute", {})
        if attr.get("language") != "english": continue
        cats = {d["category_type"] for d in p["layout_dets"]}
        if not (cats & {"table", "equation_isolated"}): continue   # v1 has the rest
        regions = [d for d in p["layout_dets"]
                   if d["category_type"] in TEXT_CATS
                   and str(d.get("ignore", "False")) != "True"
                   and (d.get("text") or "").strip() and d.get("order") is not None]
        if len(regions) < 3: continue
        regions.sort(key=lambda d: int(d.get("order", 0)))
        img = SRC / "images" / info["image_path"]
        if img.exists(): kept.append((info, attr, regions, img))
    lines = ['name = "carmenta-omnidoc-train2"', "version = 1", 'task = "ocr"', ""]
    counts, n = Counter(), 0
    for info, attr, regions, img in kept:
        stem = f"omnx-{n:04}"
        shutil.copyfile(img, OUT / f"{stem}.png")
        (OUT / f"{stem}.txt").write_text("\n".join(r["text"].strip() for r in regions), encoding="utf-8")
        (OUT / f"{stem}.json").write_text(json.dumps({
            "layout": attr.get("layout", "unknown"), "data_source": attr.get("data_source"),
            "page_size": [info.get("width"), info.get("height")],
            "source_image": info["image_path"],
            "regions": [{"order": int(r.get("order", 0)), "class": r["category_type"],
                         "poly": r.get("poly"), "text": r["text"]} for r in regions],
        }, indent=1), encoding="utf-8")
        digest = hashlib.sha256((OUT / f"{stem}.png").read_bytes()).hexdigest()
        lines += ["[[clips]]", f'id = "{stem}"',
                  f'path = "clips/carmenta-omnidoc-train2/{stem}.png"',
                  f'ground_truth = "clips/carmenta-omnidoc-train2/{stem}.txt"',
                  'class = "document_scan"', 'split = "train"',
                  'license = "Apache-2.0 (OmniDocBench, opendatalab)"',
                  f'sha256 = "{digest}"', ""]
        counts[attr.get("data_source", "?")] += 1
        n += 1
    MANIFEST.write_text("\n".join(lines), encoding="utf-8")
    print(f"train2: {n} pages -> {OUT}")
    for k, v in counts.most_common(): print(f"   {k:22s} {v}")

if __name__ == "__main__":
    main()
