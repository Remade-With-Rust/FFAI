"""Ground truth for the PP-OCRv5 mobile-det port: run the REAL exported
inference model and dump its probability map on a pinned input.

Why this exists: the rep-branch fusion's own self-check compares the branches
it *collected* against the fused conv, so a branch it never collected passes
silently. That is exactly what happened — 19 identity BatchNorm branches were
dropped and the check still read 4.88e-04. A self-check that cannot fail on a
missing input is not an instrument.

So the oracle is external: paddle's own exported program, fed a deterministic
input, fetched at the output. Everything downstream — the corrected fusion,
the numpy reference, the candle port — is gated on reproducing THIS array.

Writes `corpora/refs/fixtures/mobiledet_probmap.npz` (input + prob map).
"""

import json
import os
from pathlib import Path

import numpy as np

MODEL = Path.home() / ".paddlex" / "official_models" / "PP-OCRv5_mobile_det"
OUT = Path(__file__).resolve().parent.parent / "corpora" / "refs" / "fixtures"

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "corpora" / "clips" / "carmenta-render" / "page-00.png"

# TWO pinned inputs, because either alone is a weak instrument.
#
#   `noise` exercises every channel of every stage in the post-normalisation
#   range, which a photograph does not — but it drives the head's sigmoid to
#   ~1e-6 everywhere, where an error of a whole logit is invisible in probability
#   space. So agreement is scored on the LOGIT, not the probability.
#
#   `page` produces a real bimodal map with values across the full range, which
#   is what the postprocess actually consumes — and it pins the preprocessing
#   (BGR channel order, ImageNet stats) that the Rust side must reproduce.
H, W = 256, 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def noise_input():
    rng = np.random.RandomState(20260730)
    return np.clip(rng.randn(1, 3, H, W).astype(np.float32), -2.5, 2.5)


def page_input():
    """A real page through PaddleOCR's documented det preprocessing.

    `inference.yml` decodes BGR and then normalises with ImageNet's *RGB*
    statistics — mean[0]=0.485 lands on the BLUE channel. That looks like a bug
    and is not one: it is what the weights were trained against, so the port
    reproduces it deliberately. Sides are rounded to a multiple of 32.
    """
    from PIL import Image

    img = Image.open(PAGE).convert("RGB")
    long = max(img.size)
    scale = 960 / long if long > 960 else 1.0
    w = max(32, int(round(img.width * scale / 32)) * 32)
    h = max(32, int(round(img.height * scale / 32)) * 32)
    a = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32)[:, :, ::-1]
    a = (a / 255.0 - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)


def main():
    import paddle

    paddle.set_device("cpu")
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    prog, feed, fetch = paddle.static.load_inference_model(str(MODEL / "inference"), exe)
    print(f"loaded program: {len(feed)} feed / {len(fetch)} fetch")

    OUT.mkdir(parents=True, exist_ok=True)
    arrays, summary = {}, {}
    for tag, x in (("noise", noise_input()), ("page", page_input())):
        (prob,) = exe.run(prog, feed={feed[0]: x}, fetch_list=fetch)
        prob = np.asarray(prob, dtype=np.float32)
        # Logit space: the head is a sigmoid, and on `noise` every probability
        # sits at ~1e-6 where a full logit of error rounds away to nothing.
        logit = np.log(np.clip(prob, 1e-12, 1 - 1e-7) / np.clip(1 - prob, 1e-12, None))
        arrays[f"{tag}_input"], arrays[f"{tag}_prob"] = x, prob
        summary[tag] = {
            "input_shape": list(x.shape), "prob_shape": list(prob.shape),
            "prob_min": float(prob.min()), "prob_max": float(prob.max()),
            "prob_mean": float(prob.mean()),
            "logit_min": float(logit.min()), "logit_max": float(logit.max()),
            "fg_frac_at_0.3": float((prob > 0.3).mean()),
        }
        print(f"{tag:6s} in {x.shape} prob {prob.shape} "
              f"p[{prob.min():.4g},{prob.max():.4g}] "
              f"logit[{logit.min():.3f},{logit.max():.3f}] "
              f"fg@0.3 {(prob > 0.3).mean() * 100:.2f}%")

    # ---- the fixture the Rust oracle test consumes.
    #
    # A 256x256 crop of a real page at NATIVE resolution: no resize, so the
    # python and Rust sides cannot disagree about interpolation, and the test
    # covers the preprocessing (BGR order, ImageNet stats) as well as the
    # network. Stored as PNG + probabilities rather than as a float input,
    # because a fixture you can look at is a fixture you can debug.
    from PIL import Image
    from safetensors.numpy import save_file

    crop = Image.open(PAGE).convert("RGB").crop((300, 300, 300 + 256, 300 + 256))
    crop.save(OUT / "mobiledet_oracle_crop.png")
    a = np.asarray(crop, dtype=np.float32)[:, :, ::-1]
    xc = np.ascontiguousarray(((a / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None], np.float32)
    (pc,) = exe.run(prog, feed={feed[0]: xc}, fetch_list=fetch)
    pc = np.asarray(pc, np.float32)
    save_file({"prob": pc}, str(OUT / "mobiledet_oracle_prob.safetensors"))
    summary["crop"] = {
        "box": [300, 300, 556, 556], "prob_shape": list(pc.shape),
        "fg_frac_at_0.3": float((pc > 0.3).mean()),
        "prob_mean": float(pc.mean()),
    }
    print(f"crop   fixture 256x256  fg@0.3 {(pc > 0.3).mean() * 100:.2f}%  mean {pc.mean():.4f}")

    # Paddle's own DBPostProcess is run by `carmenta_mobiledet_dbref.py`, in a
    # separate process: importing paddlex pulls in torch, whose DLLs collide
    # with the already-loaded paddle inference runtime on this box.

    np.savez_compressed(OUT / "mobiledet_probmap.npz", **arrays)
    summary["seed"] = 20260730
    summary["page"]["source"] = str(PAGE.relative_to(REPO)).replace("\\", "/")
    (OUT / "mobiledet_probmap.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
