#!/usr/bin/env python3
"""ARM 2 — the SAME checkpoint, an independent CPU runtime. Prices the PORT.

Two modes, ONE pinned decode configuration — which is the reason both live
here rather than in two scripts that would drift apart.

Batch-adapter contract (crates/ffai-bench/src/reference.rs):

    argv:   --batch <filelist> --corpus <corpora/argus-*.toml> --model <hf-repo>
    stdout: {"load_secs": <f>} then one {"path","text","transcribe_secs"} per clip

Serve contract (crates/ffai-demo), for the side-by-side demo:

    argv:   --serve --model <hf-repo>
    stdout: {"load_secs": <f>} then one {"text","secs"} per request line
    stdin:  one {"path","prompt","max_new_tokens"} JSON object per line

`--serve` exists so the demo can compare WARM against WARM. Spawning this
script per click would reload ~1 GB of weights every time and put a ~15 s load
inside a latency reading, which would make the reference look absurdly slow for
a reason that has nothing to do with the reference. The demo already warms its
own engines before serving for exactly this reason; this extends the same
courtesy to the arm it is being compared against.

Why this arm exists, and why it is not Arm 1
--------------------------------------------
Arm 1 answers "how good is this model". **Only this arm answers "how good is
our implementation of it"** — and that is the question every other FFai
headline is an answer to. The README says why in as many words:

    "Comparing our small.en against their tiny.en would price the model rather
     than the implementation, which is the error the reference file exists to
     prevent."

So this is plain Transformers on CPU, running the identical checkpoint FFai's
engine will run, under the identical decode configuration, on the identical
clips. It feeds the correctness, speed and footprint gates. Arm 1 cannot: it
carries VLMEvalKit's batching, its resizing policy and its own generation
defaults, none of which FFai's engine has, so a speed or memory number taken
against it would price the harness.

DECODE CONFIG IS PINNED, NOT DEFAULTED
--------------------------------------
`--greedy` is the default and `do_sample=False` is passed explicitly, because
a model's `generation_config.json` can ship `do_sample=True` with a
temperature, and an unpinned reference compares decoding strategies while
pretending to compare implementations. The seed is set anyway so that a
deliberate `--sample` run is still reproducible — the plan's §2 Gate 2 makes
byte-stability a v1 requirement, and a reference that cannot repeat itself
cannot hold a v1 engine to that standard.

Whatever is pinned here must be echoed in this reference's `config` string in
corpora/references.toml, since that string is the key `fill_gates` matches an
engine against.
"""

import argparse
import json
import os
import sys
import time
import tomllib


def load_prompts(corpus_path: str):
    with open(corpus_path, "rb") as fh:
        man = tomllib.load(fh)
    base = os.path.dirname(os.path.abspath(corpus_path))
    out = {}
    for clip in man.get("clips", []):
        p = os.path.normcase(os.path.abspath(os.path.join(base, clip["path"])))
        out[p] = clip.get("prompt")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", help="filelist; batch mode (bench harness)")
    ap.add_argument("--corpus", help="corpus toml supplying per-clip prompts")
    ap.add_argument("--serve", action="store_true",
                    help="persistent mode: read request lines from stdin, one "
                         "JSON result per line. Loads the model ONCE.")
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--sample", action="store_true",
                    help="stochastic decode (seeded). Default is greedy; the "
                         "engine this arm is matched against decodes greedily.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads; 0 leaves torch's own default. Set it "
                         "when comparing speed, or the two arms get different core counts")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args()

    if not args.serve:
        if not args.batch or not args.corpus:
            ap.error("--batch and --corpus are required unless --serve is given")
        with open(args.batch, "r", encoding="utf-8") as fh:
            paths = [ln.strip() for ln in fh if ln.strip()]
        prompts = load_prompts(args.corpus)
    else:
        paths, prompts = [], {}

    t0 = time.perf_counter()
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError as exc:
        print(f"Arm-2 stack not importable: {exc}\n"
              f"  .venv-argus/Scripts/pip install torch transformers pillow",
              file=sys.stderr)
        return 1

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, device_map="cpu"
    )
    model.eval()
    load_secs = time.perf_counter() - t0
    print(json.dumps({"load_secs": load_secs}), flush=True)

    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": bool(args.sample)}
    if args.sample:
        gen_kwargs["temperature"] = args.temperature

    def caption(path, prompt, kwargs):
        """One image, one prompt. The ONLY generation path in this file.

        Both modes call it, so `--serve` and `--batch` cannot drift in the
        chat template, the tail-only decode, or the pinned generation kwargs.
        """
        image = Image.open(path).convert("RGB")
        # The processor's OWN chat template. Hand-writing the turn markers is
        # the highest-risk silent failure in a VLM build: get them wrong and
        # output degrades with no error at all.
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        chat = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=chat, images=[image], return_tensors="pt")
        with torch.inference_mode():
            out = model.generate(**inputs, **kwargs)
        # Decode ONLY the newly generated tail; decoding the whole sequence
        # returns the prompt back and every metric would score the question.
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return (
            processor.decode(new_tokens, skip_special_tokens=True).strip(),
            int(inputs["input_ids"].shape[1]),
        )

    if args.serve:
        # One request per line, one result per line. A blank line or EOF ends
        # the session, so closing the pipe is a clean shutdown rather than a
        # broken one.
        for line in sys.stdin:
            line = line.strip()
            if not line:
                break
            t = time.perf_counter()
            try:
                req = json.loads(line)
                kwargs = dict(gen_kwargs)
                if req.get("max_new_tokens"):
                    kwargs["max_new_tokens"] = int(req["max_new_tokens"])
                text, prompt_tokens = caption(
                    req["path"], req.get("prompt") or "Describe this image.", kwargs
                )
                print(
                    json.dumps({
                        "text": text,
                        "secs": time.perf_counter() - t,
                        "prompt_tokens": prompt_tokens,
                    }),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                # Report and KEEP SERVING. One bad image must not take the
                # worker down; the demo would then silently lose its reference
                # pane for the rest of the session.
                print(
                    json.dumps({
                        "error": f"{type(exc).__name__}: {exc}",
                        "secs": time.perf_counter() - t,
                    }),
                    flush=True,
                )
        return 0

    for path in paths:
        key = os.path.normcase(os.path.abspath(path))
        prompt = prompts.get(key) or "Describe this image."
        t = time.perf_counter()
        text = ""
        try:
            text, _ = caption(path, prompt, gen_kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"{path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            json.dumps({
                "path": path,
                "text": text,
                "transcribe_secs": time.perf_counter() - t,
            }),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
