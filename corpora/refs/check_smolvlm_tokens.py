#!/usr/bin/env python3
"""Step 4's headline gate: do OUR embeddings decode to the reference's tokens?

    cargo run --release -p ffai-argus --example build_inputs_embeds
    .venv-argus/Scripts/python.exe corpora/refs/check_smolvlm_tokens.py

The plan's condition for step 4 is

    "a known image+prompt reproduces the reference implementation's output
     tokens"

and note it says *output* tokens. Everything proved so far is on the input
side: the assembled ids match exactly and the splice is bit-exact. Those are
necessary and they are not this.

WHY THIS IS NOT A TAUTOLOGY
---------------------------
Our vision tower carries ~1e-4 of float error against the reference (step 3,
104.8 dB SNR — reassociation, not a defect). Composed through the connector and
the splice, our `inputs_embeds` sits 2.1e-4 from the reference's.

**Greedy decoding is an argmax.** A small perturbation can flip a token, and a
flipped token changes the answer. That failure would arrive through accumulated
numerics rather than through a structural mistake, and it would look exactly
like the "plausible but degraded output" §3.3 warns about — fluent, confident,
different.

So this feeds OUR embeddings to the REFERENCE's own decoder. If the tokens come
out identical, our half of the pipeline is good enough that the decoder cannot
tell the difference. That is the only standard worth holding it to, and it is
strictly stronger than any tensor tolerance we could argue for.

Isolation is preserved throughout: preprocessing comes from the reference's
`pixel_values`, and the decoder is the reference's, so a mismatch can only be
our tower, connector or assembly.
"""

import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = os.path.join(root, ".oracle", "smolvlm-prompt")
    doc = json.load(open(os.path.join(d, "prompt.json"), encoding="utf-8"))

    ours_path = os.path.join(d, "ours_inputs_embeds.f32")
    if not os.path.exists(ours_path):
        print("run the Rust side first:\n"
              "  cargo run --release -p ffai-argus --example build_inputs_embeds",
              file=sys.stderr)
        return 2

    shape = tuple(doc["embeds"]["inputs_embeds"]["shape"])
    ours = np.fromfile(ours_path, dtype="<f4").reshape(shape)
    ref = np.fromfile(os.path.join(d, "inputs_embeds.f32"), dtype="<f4").reshape(shape)
    print(f"inputs_embeds {shape}  max_abs(ours vs ref) = {np.abs(ours - ref).max():.3e}")

    model = AutoModelForImageTextToText.from_pretrained(
        doc["model"], dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    tok = AutoProcessor.from_pretrained(doc["model"]).tokenizer

    n = shape[1]
    mask = torch.ones((1, n), dtype=torch.long)
    want = [int(t) for t in doc["reference_output_ids"]]

    with torch.inference_mode():
        got = model.generate(
            inputs_embeds=torch.from_numpy(ours.copy()),
            attention_mask=mask,
            do_sample=False,
            max_new_tokens=len(want),
        )[0].tolist()

    # `generate` with inputs_embeds returns only the NEW tokens (there are no
    # input ids to echo), so no prefix has to be stripped. Guard it rather than
    # assume: a silent prefix would make a mismatch look like a shift.
    if len(got) > len(want) and got[: len(want)] != want:
        got = got[-len(want):]

    print()
    print("reference:", want)
    print("ours     :", got)
    print()
    print("reference text:", repr(tok.decode(want, skip_special_tokens=True)))
    print("ours      text:", repr(tok.decode(got, skip_special_tokens=True)))
    print()

    if got == want:
        print(f"PASS — {len(want)}/{len(want)} output tokens identical.")
        print("Our tower + connector + assembly are numerically close enough that")
        print("the reference decoder produces the same argmax at every step.")
        return 0

    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    same = sum(1 for a, b in zip(got, want) if a == b)
    print(f"FAIL — {same}/{len(want)} tokens match; first divergence at step {first}.")
    print(f"  reference[{first}] = {want[first] if first < len(want) else None}"
          f" ({tok.convert_ids_to_tokens([want[first]]) if first < len(want) else None})")
    print(f"  ours     [{first}] = {got[first] if first < len(got) else None}"
          f" ({tok.convert_ids_to_tokens([got[first]]) if first < len(got) else None})")
    print()
    print("A late divergence is greedy decode amplifying float error and is a")
    print("tolerance question. A divergence at step 0 is a structural one.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
