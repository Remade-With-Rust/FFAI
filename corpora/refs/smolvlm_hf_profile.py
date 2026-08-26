"""Where does PyTorch spend a caption? The stage split, to compare against ours.

NOT the oracle. `smolvlm_hf_ref.py` is pinned — it defines the decode config the
ledger's quality gate is measured under, and it must not drift. This file is a
separate instrument that answers one question the oracle cannot: **which stage
are we losing to?**

`ffai-demo` reports ours as (decode, preprocess, vision, assemble, prefill,
generate). PyTorch reports one number. Without the same split on both sides,
"we are 1.19x slower" names no stage and cannot be acted on.

Same checkpoint, same image, same greedy config, same thread count as the
default runtime. Run:

    .venv-argus/Scripts/python.exe corpora/refs/smolvlm_hf_profile.py <image> [--max 32]
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--max", type=int, default=32)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32, device_map="cpu")
    model.eval()
    load_secs = time.perf_counter() - t0

    # --- preprocess ------------------------------------------------------
    image = Image.open(args.image).convert("RGB")
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "What is written in this image?"}],
    }]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    t = time.perf_counter()
    inputs = processor(text=chat, images=[image], return_tensors="pt")
    preprocess_ms = (time.perf_counter() - t) * 1e3

    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]
    prompt_tokens = int(input_ids.shape[1])
    # (batch, tiles, 3, H, W)
    tiles = int(pixel_values.shape[1]) if pixel_values.dim() == 5 else 1

    inner = model.model  # Idefics3Model
    vision = inner.vision_model
    connector = getattr(inner, "connector", None)

    def best(fn, reps=3):
        """Min of `reps`. A transient inflates a mean; it cannot deflate a min."""
        b = float("inf")
        out = None
        for _ in range(reps):
            t = time.perf_counter()
            out = fn()
            b = min(b, (time.perf_counter() - t) * 1e3)
        return b, out

    with torch.inference_mode():
        # WARM FIRST — the demo holds this process open across clicks, so its
        # reference arm is warm. Timing a cold first inference against that
        # would flatter us by whatever lazy init costs.
        for _ in range(2):
            _ = model.generate(**inputs, max_new_tokens=2, do_sample=False)

        # --- vision tower, ALL TILES AS ONE BATCH ------------------------
        # The structural difference worth measuring: transformers runs one
        # batch-N forward where we run N passes.
        pv = pixel_values
        if pv.dim() == 5:
            b_, n_, c_, h_, w_ = pv.shape
            pv = pv.view(b_ * n_, c_, h_, w_)
        vision_ms, vout = best(lambda: vision(pixel_values=pv).last_hidden_state)
        connector_ms, _ = best(lambda: connector(vout) if connector is not None else None)

        # --- the WHOLE caption, exactly the pinned reference's kwargs -----
        # Nothing derived by subtracting two separately-timed calls: an earlier
        # version of this file did that and produced 402 ms/token on a quiet
        # box against 95 ms/token on a loaded one, which is impossible and was
        # the harness, not the machine.
        total_ms, out = best(
            lambda: model.generate(**inputs, max_new_tokens=args.max, do_sample=False)
        )
        produced = int(out.shape[1] - input_ids.shape[1])

    # generate() runs vision THEN the text side, so the text side is what is
    # left. Both terms come from the same min-of-3 regime.
    text_side_ms = total_ms - vision_ms

    text = processor.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    print(json.dumps({
        "engine": "PyTorch + transformers",
        "threads": torch.get_num_threads(),
        "load_secs": load_secs,
        "tiles": tiles,
        "prompt_tokens": prompt_tokens,
        "generated": produced,
        "preprocess_ms": preprocess_ms,
        "vision_ms": vision_ms,
        "connector_ms": connector_ms,
        "caption_total_ms": total_ms,
        "text_side_ms": text_side_ms,
        "text_per_token_ms": text_side_ms / max(produced, 1),
        "text": text,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
