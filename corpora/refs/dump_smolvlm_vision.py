#!/usr/bin/env python3
"""Dump SmolVLM's vision tower stage-by-stage as the oracle for the candle port.

Step 3 of docs/plans/argus-launch-plan.md:

    "vision tower only, oracle-matched against Arm 2's runtime on fixed
     input — tensor match to tolerance. A mismatched tower cannot be debugged
     later through generated text."

That last sentence is the whole reason this file exists. A vision tower that
is subtly wrong still produces fluent, plausible captions; §7 of the plan
measured the same failure mode from a prompt difference alone — 43 of 50
answers changed with nothing raising an error. Numerical error hides behind
plausibility, so the tower is compared as TENSORS, before a single token is
generated.

    .venv-argus/Scripts/python.exe corpora/refs/dump_smolvlm_vision.py \
        --out .oracle/smolvlm-vision

THE INPUT IS A FORMULA, NOT A FILE
----------------------------------
Following `dump_whisper_mel.py`: the input image is generated from a
deterministic formula mirrored exactly in the Rust test, so no image fixture
has to be committed, nothing is license-encumbered, and anyone can regenerate
the oracle from scratch. A flat or noise-only image would let whole classes of
bug hide (a broken position encoding is invisible on a constant image), so the
pattern deliberately varies along BOTH axes and across channels.

WHAT IS WRITTEN
---------------
Two things, because they answer different questions:

* `<out>/<stage>.f32` — raw little-endian f32, row-major, full tensors. These
  are what a tensor-match test actually compares. They are large (a 1024x768
  hidden state is 3 MB) and therefore NOT committed; the directory is
  gitignored and the dump is regenerable by re-running this script.

* `<out>/summary.json` — shapes, dtypes, and per-stage mean/std/min/max plus
  the first and last 8 values. Small enough to commit, so a test can detect
  gross drift on a machine that has never run this script. An oracle that
  cannot run is worse than one that fails.

STAGES
------
Chosen so a mismatch can be BISECTED rather than merely detected: if
`patch_embed` matches and `layer_00` does not, the bug is in the first encoder
block and not in the preprocessing.
"""

import argparse
import hashlib
import json
import math
import os
import struct
import sys

# Kept in sync with the Rust test. Any change here is a change to the oracle.
IMG = 512
CHANNELS = 3


def reference_image(size: int = IMG):
    """A deterministic RGB pattern, mirrored exactly in the Rust test.

    Varies along x, along y, and between channels, so that a transposed axis,
    a swapped channel order, or a dropped position encoding all change the
    output. A constant or single-axis pattern would hide every one of them.
    """
    px = bytearray(size * size * CHANNELS)
    i = 0
    for y in range(size):
        fy = y / size
        for x in range(size):
            fx = x / size
            r = 0.5 + 0.5 * math.sin(6.0 * math.pi * fx)
            g = 0.5 + 0.5 * math.sin(6.0 * math.pi * fy + 1.0)
            b = 0.5 + 0.5 * math.sin(6.0 * math.pi * (fx + fy) + 2.0)
            px[i] = int(max(0.0, min(1.0, r)) * 255.0 + 0.5)
            px[i + 1] = int(max(0.0, min(1.0, g)) * 255.0 + 0.5)
            px[i + 2] = int(max(0.0, min(1.0, b)) * 255.0 + 0.5)
            i += 3
    return bytes(px)


def write_f32(path, arr):
    flat = arr.reshape(-1).astype("float32")
    with open(path, "wb") as fh:
        fh.write(flat.tobytes())
    return flat


def stats(flat):
    import numpy as np
    f = np.asarray(flat, dtype="float64")
    return {
        "mean": float(f.mean()),
        "std": float(f.std()),
        "min": float(f.min()),
        "max": float(f.max()),
        "sha256_f32le": hashlib.sha256(
            np.asarray(flat, dtype="float32").tobytes()
        ).hexdigest(),
        "head8": [float(v) for v in f[:8]],
        "tail8": [float(v) for v in f[-8:]],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--out", default=".oracle/smolvlm-vision")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        print(f"Arm-2 stack not importable: {exc}", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    raw = reference_image()
    img = Image.frombytes("RGB", (IMG, IMG), raw)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    # Preprocess exactly the way the engine will: the processor's own pipeline,
    # never a hand-rolled resize/normalise. Getting this wrong shifts every
    # downstream tensor and would read as a tower bug.
    enc = processor(text="<image>", images=[img], return_tensors="pt")
    pixel_values = enc["pixel_values"]
    print(f"pixel_values: {tuple(pixel_values.shape)}", file=sys.stderr)

    # SmolVLM tiles: (batch, tiles, C, H, W). Take ONE tile — the tower is
    # applied per tile, so one is enough to gate the tower itself, and tiling
    # is gated separately by the tile COUNT and positions.
    pv = pixel_values
    if pv.dim() == 5:
        tiles = pv.shape[1]
        pv = pv[:, 0]
    else:
        tiles = 1
    print(f"tiles={tiles}, single-tile input {tuple(pv.shape)}", file=sys.stderr)

    vision = model.model.vision_model
    captured = {}

    # Dump the INPUT too. Step 3 gates the TOWER, and feeding it our own
    # preprocessing would test two bricks at once — a mismatch could then be
    # the resize, the normalisation, the channel order or the tower, and the
    # oracle could not say which. `codec-bringup-decoder`'s per-stage isolation
    # law: feed each stage the reference's previous-stage output. Preprocessing
    # gets its own gate, separately.
    captured["pixel_values_tile0"] = pv.detach().to(torch.float32).cpu().numpy()

    def grab(name):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            captured[name] = t.detach().to(torch.float32).cpu().numpy()
        return hook

    handles = [vision.embeddings.register_forward_hook(grab("embeddings"))]
    layers = vision.encoder.layers
    for idx in {0, len(layers) // 2, len(layers) - 1}:
        handles.append(layers[idx].register_forward_hook(grab(f"layer_{idx:02d}")))
    if hasattr(vision, "post_layernorm"):
        handles.append(vision.post_layernorm.register_forward_hook(grab("post_layernorm")))

    with torch.inference_mode():
        out = vision(pixel_values=pv)
        captured["vision_out"] = (
            out.last_hidden_state.detach().to(torch.float32).cpu().numpy()
        )
        # The connector is what turns tower output into LLM embedding space —
        # §3.1.6, the pixel-shuffle variant for SmolVLM. Dumped because it is
        # the other half of "the tower matched but the model is still wrong".
        conn = getattr(model.model, "connector", None)
        if conn is not None:
            captured["connector"] = (
                conn(out.last_hidden_state).detach().to(torch.float32).cpu().numpy()
            )
    for h in handles:
        h.remove()

    summary = {
        "model": args.model,
        "image": {"size": IMG, "channels": CHANNELS, "formula": "reference_image()"},
        "tiles": int(tiles),
        "pixel_values_shape": list(pixel_values.shape),
        "single_tile_shape": list(pv.shape),
        "stages": {},
    }
    for name, arr in captured.items():
        flat = write_f32(os.path.join(args.out, f"{name}.f32"), np.asarray(arr))
        summary["stages"][name] = {"shape": list(np.asarray(arr).shape), **stats(flat)}
        print(f"  {name:16s} {tuple(np.asarray(arr).shape)}", file=sys.stderr)

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"wrote {len(captured)} stages -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
