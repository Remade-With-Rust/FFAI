"""Dump YOLO26n reference activations for the candle port's oracles (M-D1).

TWO fixtures, because one cannot do both jobs:

- **synth** — the input is a FORMULA, not an image (the mel-fixture trick
  from Mercury M1). Both sides generate the same deterministic 640x640x3
  tensor from a closed form, so it carries no license, needs no download,
  and anyone can regenerate it byte-identically. It exercises every layer's
  arithmetic. What it CANNOT do is exercise the decode: a formula contains
  no objects, so every one of the 300 output rows is a junk low-score
  detection (measured: 0 detections over 0.25 confidence).
- **photo** — a corpus image (CC-licensed, already pinned in
  `corpora/diana-coco-v1.toml`), letterboxed square exactly as the engine
  will. This one produces real detections, so a decode or anchor-grid error
  shows up as a wrong BOX rather than as a small numeric delta.

**Shipping vs size, resolved the way the gitignore already resolves it.**
The full activation dump is ~43 MB per fixture — far too large to commit,
and `*.npz` is gitignored for exactly this reason (mobiledet_stages.npz).
But Carmenta paid for the opposite lesson too: three oracles that could not
run on a fresh clone, and "for a project that gates everything on oracles,
an oracle that cannot run is worse than one that fails." So this script
emits BOTH:

  yolo26n_oracle_{synth,photo}.npz     full tensors — gitignored, regenerable
  yolo26n_oracle_digest.json           TRACKED, ~200 KB: per-tensor shape and
                                       mean/std/min/max, plus 256 exact values
                                       at fixed deterministic indices

The digest is a real gate, not a placeholder: 256 exact values per layer at
1e-4 catches any wiring, ordering, or channel-count error, because such an
error moves all of them. The Rust oracle runs the digest always and upgrades
to full-tensor max-abs when the npz is present.

Every tier gets its own digest. The n port was checked layer by layer
during the port itself; for a tier VARIANT the thing under test is the
scale derivation, and most of the ways that can be wrong are shape errors
the strict load already catches. What it does not catch is the handful of
scale-dependent choices that are shape-invisible — a `shortcut` flag, a
missing activation, a C3k where a Bottleneck belongs but happens to have
matching widths. Those move every sampled value, so the digest is the gate
that closes them.

Usage:
    .venv-diana/Scripts/python.exe tools/diana_oracle_dump.py [--model yolo26n]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "corpora" / "refs" / "fixtures"

_ap = argparse.ArgumentParser(description="dump a YOLO26 per-layer oracle")
_ap.add_argument("--model", default="yolo26n", help="checkpoint stem in corpora/cache")
_args = _ap.parse_args()
MODEL = _args.model
CKPT = ROOT / "corpora" / "cache" / f"{MODEL}.pt"
DIGEST = FIX / f"{MODEL}_oracle_digest.json"
# coco-032 is the busiest clip in the corpus — 23 objects across 6 classes.
# Chosen deliberately: the end2end top-k is TWO-STAGE (top-k anchors by best
# class score, then top-k over the flattened anchor x class scores), so one
# anchor can emit several detections under different classes. A single-object
# image cannot fail that; a crowded multi-class one can.
PHOTO = ROOT / "corpora" / "clips" / "diana-coco" / "coco-032.jpg"

IMGSZ = 640
SAMPLES = 256


def fixture_input() -> torch.Tensor:
    """The pinned input, from a formula both implementations can reproduce.

    Deterministic, license-free, and structured rather than noise: smooth
    gradients plus hard edges at several scales, so every stride level sees
    real signal and a wiring error in the FPN/PAN skips cannot hide behind a
    flat field.

        r = |sin(3x)cos(2y)|,  g = ((x*7)%1 + (y*11)%1)/2,  b = 1 if
        (floor(8x)+floor(8y)) even else 0.25,  with x,y in [0,1)
    """
    idx = np.arange(IMGSZ, dtype=np.float64)
    x = (idx / IMGSZ)[None, :].repeat(IMGSZ, axis=0)
    y = (idx / IMGSZ)[:, None].repeat(IMGSZ, axis=1)
    r = np.abs(np.sin(3.0 * x) * np.cos(2.0 * y))
    g = (((x * 7.0) % 1.0) + ((y * 11.0) % 1.0)) / 2.0
    checker = (np.floor(x * 8.0) + np.floor(y * 8.0)) % 2.0
    b = np.where(checker == 0.0, 1.0, 0.25)
    arr = np.stack([r, g, b], axis=0)[None].astype(np.float32)
    return torch.from_numpy(arr)


def photo_input() -> torch.Tensor:
    """A pinned corpus image, letterboxed SQUARE — the geometry M-D0 pinned.

    Reproduces Ultralytics' LetterBox: scale to fit, centre-pad with 114,
    /255, CHW. Identical to `corpora/refs/yolo_ort_ref.py` on purpose; if
    these two ever drift the oracle stops testing what the engine runs.
    """
    from PIL import Image

    img = Image.open(PHOTO).convert("RGB")
    w0, h0 = img.size
    r = min(IMGSZ / w0, IMGSZ / h0)
    nw, nh = round(w0 * r), round(h0 * r)
    left, top = round((IMGSZ - nw) / 2 - 0.1), round((IMGSZ - nh) / 2 - 0.1)
    canvas = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (left, top))
    # Ship the letterboxed canvas itself, TRACKED and tier-independent.
    #
    # The Rust oracle cannot recompute this: it is a corpus JPEG through
    # PIL's BILINEAR, and our letterbox does not reproduce that bit for bit
    # (nor should it — that would be a resampler comparison wearing a
    # detector's clothes). But the canvas is uint8, so a PNG round-trips it
    # exactly, and 500 KB buys the ONLY fixture on which the two-stage
    # top-k's row order is defined. The synth input produces no detection
    # above 0.008; its 300 rows are a ranking of noise.
    canvas.save(FIX / "diana_photo_input.png", format="PNG", optimize=True)
    arr = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
    return torch.from_numpy(np.ascontiguousarray(arr))


def sample_indices(n: int, count: int) -> np.ndarray:
    """Deterministic spread of `count` flat indices over `n` elements.

    A fixed stride rather than an RNG: reproducible from the numbers alone,
    in any language, with no shared generator to match.
    """
    if n <= count:
        return np.arange(n, dtype=np.int64)
    return (np.arange(count, dtype=np.int64) * (n // count)) % n


def run_fixture(net, head, x: torch.Tensor) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"input": x.numpy()}
    layer_out: dict[int, torch.Tensor] = {}
    hooks = []

    def mk(i):
        def hook(mod, inp, o):
            if isinstance(o, torch.Tensor):
                layer_out[i] = o.detach()

        return hook

    for i, layer in enumerate(net.model):
        hooks.append(layer.register_forward_hook(mk(i)))
    with torch.no_grad():
        final, _raw = net(x)
    for h in hooks:
        h.remove()

    for i, t in layer_out.items():
        out[f"layer_{i:02d}"] = t.numpy()

    # The one2one branch explicitly. THIS is the branch inference uses
    # (head.py: `preds["one2one"] if self.end2end`); the one2many cv2/cv3
    # tensors are training-only and porting them would be silently wrong.
    feats = [layer_out[i] for i in (16, 19, 22)]
    with torch.no_grad():
        o2o = head.forward_head(feats, **head.one2one)
    for k, v in o2o.items():
        if isinstance(v, torch.Tensor):
            out[f"head_{k}"] = v.detach().numpy()
        elif isinstance(v, (list, tuple)):
            for j, t in enumerate(v):
                if isinstance(t, torch.Tensor):
                    out[f"head_{k}_L{j}"] = t.detach().numpy()
    with torch.no_grad():
        decoded = head._inference(o2o)
    out["decoded"] = decoded.detach().numpy()
    out["anchors"] = head.anchors.detach().numpy()
    out["strides"] = head.strides.detach().numpy()
    out["final"] = final.detach().numpy()
    return out


model = YOLO(str(CKPT))
net = model.model
net.eval()
head = net.model[-1]

digest: dict = {
    "checkpoint": CKPT.name,
    "imgsz": IMGSZ,
    "samples_per_tensor": SAMPLES,
    "input_formula": "r=|sin(3x)cos(2y)|; g=((7x mod 1)+(11y mod 1))/2; b=checker(8) on x,y in [0,1)",
    "photo_source": str(PHOTO.relative_to(ROOT)).replace("\\", "/"),
    "nc": int(head.nc),
    "reg_max": int(head.reg_max),
    "end2end": bool(head.end2end),
    "max_det": int(head.max_det),
    "strides": [float(s) for s in head.stride],
    "inference_branch": "one2one",
    "fixtures": {},
}

for kind, maker in (("synth", fixture_input), ("photo", photo_input)):
    if kind == "photo" and not PHOTO.exists():
        print(f"skipping photo fixture: {PHOTO} absent (run tools/diana_coco_corpus.py)")
        continue
    out = run_fixture(net, head, maker())
    npz = FIX / f"{MODEL}_oracle_{kind}.npz"
    np.savez_compressed(npz, **out)

    entry = {}
    # The ENTIRE [300, 6] result, not a sample of it.
    #
    # 256 scattered indices cannot express "compare this row only if its
    # confidence is unambiguous", and that qualifier is the whole game: the
    # two-stage top-k ranks 672k candidates, so on any input without real
    # detections the rows near the cut are separated by less than f32
    # reassociation noise and their ORDER is undefined. A positional
    # comparison of those rows tests luck. 1800 floats is ~20 KB — cheap
    # enough that the gate should just have the real thing.
    entry["final_full"] = [float(f) for f in out["final"].reshape(-1)]
    for k, v in out.items():
        flat = v.reshape(-1)
        idx = sample_indices(flat.size, SAMPLES)
        entry[k] = {
            "shape": list(v.shape),
            "mean": float(v.mean()),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
            "sample_idx": idx.tolist(),
            "sample_val": [float(f) for f in flat[idx]],
        }
    digest["fixtures"][kind] = entry

    conf = out["final"][0][:, 4]
    print(f"wrote {npz.relative_to(ROOT)} ({npz.stat().st_size / 1e6:.1f} MB)")
    print(f"  {kind}: {len(out)} tensors · dets>0.25 = {int((conf > 0.25).sum())}")

DIGEST.write_text(json.dumps(digest, indent=1), encoding="utf-8")
print(f"wrote {DIGEST.relative_to(ROOT)} ({DIGEST.stat().st_size / 1e3:.0f} KB, TRACKED)")
