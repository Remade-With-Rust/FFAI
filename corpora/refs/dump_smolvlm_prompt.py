#!/usr/bin/env python3
"""Dump SmolVLM's assembled prompt — chat template, token ids, image spans.

Step 4 of docs/plans/argus-launch-plan.md. §3.3 names sequence assembly as

    "the actual 'multimodal' step and the one most likely to be silently
     wrong (wrong offset = plausible but degraded output)"

and §2.2 calls the chat template "the highest-risk silent failure in the whole
build". §7 then MEASURED that risk on this very model: **43 of 50 answers
changed on identical weights**, from prompt formatting alone, with nothing
raising an error anywhere.

So step 4 is gated on something discrete rather than on a tolerance. Token ids
are integers: they either match the reference exactly or they do not. A tensor
comparison can be argued about; `input_ids[i] != input_ids[i]` cannot.

    .venv-argus/Scripts/python.exe corpora/refs/dump_smolvlm_prompt.py \
        --out .oracle/smolvlm-prompt

WHAT IS DUMPED
--------------
* `prompt.json` — the templated string, the full id sequence, the special-token
  vocabulary, and the image-span structure (where image tokens sit, how many
  per tile, which markers separate tiles). Committed-size, and it is the
  oracle the Rust side is checked against.
* `input_ids.i64` / `attention_mask.i64` — raw little-endian i64, for a
  byte-exact comparison without going through JSON.

The point of dumping the STRUCTURE and not just the ids is that a mismatch
should say *which* part is wrong: a wrong template shows up as different text,
a wrong tile count as a different number of image spans, and a wrong
placeholder expansion as a right-length sequence with the ids in the wrong
places.
"""

import argparse
import json
import math
import os
import sys

IMG = 512
CHANNELS = 3


def reference_image(size: int = IMG):
    """The same deterministic pattern the vision dumper uses, so both oracles
    describe one input and a discrepancy cannot be blamed on the image."""
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


def spans_of(ids, image_token_id):
    """Contiguous runs of the image token — the blocks assembly must splice."""
    out, start = [], None
    for i, t in enumerate(ids):
        if t == image_token_id and start is None:
            start = i
        elif t != image_token_id and start is not None:
            out.append({"start": start, "len": i - start})
            start = None
    if start is not None:
        out.append({"start": start, "len": len(ids) - start})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--out", default=".oracle/smolvlm-prompt")
    ap.add_argument("--question", default="What is written in this image?")
    ap.add_argument("--embeds", action="store_true",
                    help="also load the model and dump the embedding-level merge "
                         "(text_embeds, image_hidden, inputs_embeds). Slower, and "
                         "it is what step 4's splice is gated against.")
    args = ap.parse_args()

    try:
        from PIL import Image
        from transformers import AutoProcessor
    except ImportError as exc:
        print(f"Arm-2 stack not importable: {exc}", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    img = Image.frombytes("RGB", (IMG, IMG), reference_image())
    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer

    # The processor's OWN chat template. Hand-writing the turn markers is the
    # failure this whole step exists to prevent.
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": args.question}],
    }]
    templated = processor.apply_chat_template(messages, add_generation_prompt=True)
    enc = processor(text=templated, images=[img], return_tensors="pt")

    ids = enc["input_ids"][0].tolist()
    mask = enc["attention_mask"][0].tolist()

    image_token = getattr(processor, "image_token", None)
    image_token_str = getattr(image_token, "content", image_token)
    if image_token_str is None:
        image_token_str = "<image>"
    image_token_id = tok.convert_tokens_to_ids(image_token_str)

    spans = spans_of(ids, image_token_id)

    # Every special token that actually appears, with its id and count. A
    # missing `<fake_token_around_image>` or a wrong `<row_i_col_j>` is exactly
    # the "plausible but degraded" failure, and it is invisible in a length
    # check alone.
    specials = {}
    for t in set(ids):
        s = tok.convert_ids_to_tokens(t)
        if s and s.startswith("<") and s.endswith(">"):
            specials[s] = {"id": int(t), "count": int(sum(1 for x in ids if x == t))}

    with open(os.path.join(args.out, "input_ids.i64"), "wb") as fh:
        for t in ids:
            fh.write(int(t).to_bytes(8, "little", signed=True))
    with open(os.path.join(args.out, "attention_mask.i64"), "wb") as fh:
        for m in mask:
            fh.write(int(m).to_bytes(8, "little", signed=True))

    doc = {
        "model": args.model,
        "question": args.question,
        "image": {"size": IMG, "formula": "reference_image()"},
        "templated_text": templated,
        "n_tokens": len(ids),
        "image_token": {"str": image_token_str, "id": int(image_token_id)},
        "image_spans": spans,
        "n_image_tokens": int(sum(s["len"] for s in spans)),
        "pixel_values_shape": list(enc["pixel_values"].shape),
        "specials": dict(sorted(specials.items(), key=lambda kv: -kv[1]["count"])),
        "first_64_ids": [int(t) for t in ids[:64]],
        "last_32_ids": [int(t) for t in ids[-32:]],
    }
    with open(os.path.join(args.out, "prompt.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)

    print(f"tokens        : {len(ids)}", file=sys.stderr)
    print(f"image spans   : {len(spans)}  totalling {doc['n_image_tokens']} image tokens",
          file=sys.stderr)
    print(f"pixel_values  : {tuple(enc['pixel_values'].shape)}", file=sys.stderr)
    print(f"specials      : {list(doc['specials'])[:8]}", file=sys.stderr)
    if args.embeds:
        dump_merge(args, enc, ids, doc)

    print(f"wrote -> {args.out}", file=sys.stderr)
    return 0


def dump_merge(args, enc, ids, doc):
    """Dump the three tensors of the embedding-level splice.

    All three, not just the result, because that is what lets the splice be
    tested IN ISOLATION: given the reference's own text embeddings and its own
    image hidden states, does our merge produce its `inputs_embeds`? Feeding
    our tower's output instead would test the tower and the splice together,
    and a mismatch could not be attributed to either
    (`codec-bringup-decoder`'s per-stage isolation law).
    """
    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    inner = model.model

    with torch.inference_mode():
        input_ids = enc["input_ids"]
        text_embeds = inner.get_input_embeddings()(input_ids)
        # The connector's output for ALL tiles — (n_tiles, tokens_per_tile, dim).
        image_hidden = inner.get_image_features(
            enc["pixel_values"], enc.get("pixel_attention_mask"), return_dict=True
        ).pooler_output
        merged = inner.inputs_merger(
            input_ids=input_ids,
            inputs_embeds=text_embeds,
            image_hidden_states=image_hidden,
        )

    # The reference's GREEDY output for this exact image+prompt — step 4's
    # headline gate. Greedy so it is deterministic and so the comparison is
    # token equality rather than a distribution.
    with torch.inference_mode():
        gen = model.generate(
            **{k: v for k, v in enc.items()},
            do_sample=False,
            max_new_tokens=32,
        )
    out_ids = gen[0][enc["input_ids"].shape[1]:].tolist()
    doc["reference_output_ids"] = [int(t) for t in out_ids]
    print(f"  reference output: {len(out_ids)} tokens", file=sys.stderr)

    for name, t in (
        ("text_embeds", text_embeds),
        ("image_hidden", image_hidden),
        ("inputs_embeds", merged),
        # ALL tiles, so the Rust side can run its own tower over the same input
        # the reference used and the comparison isolates preprocessing out.
        ("pixel_values", enc["pixel_values"]),
    ):
        a = t.detach().to(torch.float32).cpu().numpy()
        with open(os.path.join(args.out, f"{name}.f32"), "wb") as fh:
            fh.write(a.reshape(-1).astype("float32").tobytes())
        doc.setdefault("embeds", {})[name] = {"shape": list(a.shape)}
        print(f"  {name:14s} {tuple(a.shape)}", file=sys.stderr)

    doc["image_token_id"] = int(doc["image_token"]["id"])
    with open(os.path.join(args.out, "prompt.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
