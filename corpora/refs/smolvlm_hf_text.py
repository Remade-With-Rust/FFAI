"""PyTorch's TEXT tower alone: prefill and decode, timed DIRECTLY.

`smolvlm_hf_profile.py` reports a text side of `min(total) - min(vision)`. Two
independently-taken minima do not subtract: `min(a+b) >= min(a) + min(b)`, so
that figure UNDERSTATES PyTorch's text time and flatters it against ours. It was
good enough to say "the deficit is vision"; it is not good enough to conclude
"we lose the text side by 1.59x", which is what it was then used for.

This measures the two text phases the way `examples/stage_split` measures ours:

  * **prefill** — one forward over the whole prompt, no cache, timed alone.
  * **decode**  — N single-token steps with the cache, timed alone.

Same hidden size, same prompt length, same token budget, min-of-N on both.

    .venv-argus/Scripts/python.exe corpora/refs/smolvlm_hf_text.py [--seq 1142] [--max 32]
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoModelForImageTextToText

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1142)
    ap.add_argument("--max", type=int, default=32)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    text = model.model.text_model
    lm_head = model.lm_head
    hidden = text.config.hidden_size

    def best(fn, reps):
        b = float("inf")
        out = None
        for _ in range(reps):
            t = time.perf_counter()
            out = fn()
            b = min(b, (time.perf_counter() - t) * 1e3)
        return b, out

    # Deterministic input, same shape our tower is measured on.
    embeds = torch.full((1, args.seq, hidden), 0.02, dtype=torch.float32)
    step = torch.full((1, 1, hidden), 0.02, dtype=torch.float32)

    with torch.inference_mode():
        # warm
        for _ in range(2):
            _ = text(inputs_embeds=embeds[:, :8], use_cache=True)

        # ---- PREFILL: one pass over the whole prompt --------------------
        def prefill():
            return text(inputs_embeds=embeds, use_cache=True)

        prefill_ms, out = best(prefill, args.reps)

        # ---- DECODE: N single-token steps reusing that cache ------------
        def decode():
            o = text(inputs_embeds=embeds, use_cache=True)
            past = o.past_key_values
            for _ in range(args.max):
                o = text(inputs_embeds=step, past_key_values=past, use_cache=True)
                past = o.past_key_values
                _ = lm_head(o.last_hidden_state[:, -1:])
            return o

        both_ms, _ = best(decode, args.reps)
        decode_ms = both_ms - prefill_ms

    print(json.dumps({
        "engine": "PyTorch + transformers (text tower only)",
        "threads": torch.get_num_threads(),
        "seq": args.seq,
        "generated": args.max,
        "hidden": hidden,
        "prefill_ms": prefill_ms,
        "prefill_plus_decode_ms": both_ms,
        "decode_ms": decode_ms,
        "decode_per_token_ms": decode_ms / max(args.max, 1),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
