"""Extract a piper voice's VITS weights from its .onnx into safetensors, with
canonical VITS module names, plus a JSON of every conv's geometry.

Why this exists (M-T2): Mercury's `piper-candle` engine runs the SAME voice
files piper does — weights are data, fetched not vendored — but candle wants
named tensors, and ONNX export mangles some names (weight-norm folding left
32 weights called `onnx::Conv_*`; the phoneme embedding is literally named
`sid`). The NODE names still carry the module path, so each initializer is
renamed by the op that consumes it:

    /flow/flows.6/enc/in_layers.0/Conv  ->  flow.flows.6.enc.in_layers.0.weight

Conv attributes (dilations, strides, pads, groups) and LeakyRelu slopes are
extracted into the JSON so the Rust side reproduces geometry from data
instead of assumptions.

    dump_piper_weights.py --model .piper-voices/en_US-lessac-medium.onnx \
        --out <cache>/models/piper-vits-lessac-medium
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


def canonical_name(node_name: str, op: str, which: int) -> str:
    # "/flow/flows.6/enc/in_layers.0/Conv" -> "flow.flows.6.enc.in_layers.0"
    parts = [p for p in node_name.strip("/").split("/") if p]
    if parts and parts[-1].startswith(op):
        parts = parts[:-1]
    base = ".".join(parts)
    suffix = "weight" if which == 0 else "bias"
    return f"{base}.{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = onnx.load(args.model)
    g = model.graph
    init = {t.name: t for t in g.initializer}
    tensors: dict[str, np.ndarray] = {}
    conv_attrs: dict[str, dict] = {}
    leaky: dict[str, float] = {}
    renames: dict[str, str] = {}

    for node in g.node:
        if node.op_type in ("Conv", "ConvTranspose"):
            weight_inputs = [i for i in node.input if i in init]
            attrs = {a.name: list(a.ints) for a in node.attribute if a.ints}
            cname = canonical_name(node.name, node.op_type, 0).rsplit(".", 1)[0]
            conv_attrs[cname] = {"op": node.op_type, **attrs}
            for k, w in enumerate(weight_inputs):
                canon = f"{cname}.{'weight' if k == 0 else 'bias'}"
                renames[w] = canon
        elif node.op_type == "Mul" and node.name == "/dp/flows.0/Mul":
            # ElementwiseAffine's logs was constant-folded to exp(-logs).
            for i in node.input:
                if i in init and i.endswith("Exp_output_0"):
                    renames[i] = "dp.flows.0.exp_neg_logs"
        elif node.op_type == "Gather" and node.name == "/enc_p/emb/Gather":
            # The phoneme embedding table, exported under the name `sid`.
            for i in node.input:
                if i in init:
                    renames[i] = "enc_p.emb.weight"
        elif node.op_type == "LeakyRelu":
            for a in node.attribute:
                if a.name == "alpha":
                    leaky[node.name] = round(a.f, 6)

    kept, skipped = 0, 0
    for name, t in init.items():
        arr = numpy_helper.to_array(t)
        # Weights only: float tensors with real shape. Scalars and int shape
        # constants belong to the graph, not the model. The filter is ndim,
        # NOT size — dp.flows.0.m is a real [2,1] parameter, and a size
        # threshold silently ate it on the first pass.
        if arr.dtype not in (np.float32, np.float16) or arr.ndim == 0:
            skipped += 1
            continue
        canon = renames.get(name, name)
        if canon.startswith(("onnx::", "/")):
            # A float constant that is not consumed by a Conv — keep it under
            # a stable name so nothing is silently dropped.
            canon = "graph_const." + canon.replace("/", "_").replace("::", "_")
        if canon in tensors:
            raise SystemExit(f"name collision: {canon} (from {name})")
        tensors[canon] = arr.astype(np.float32)
        kept += 1

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(outdir / "vits.safetensors"))
    (outdir / "vits-graph.json").write_text(
        json.dumps({"conv_attrs": conv_attrs, "leaky_relu": leaky}, indent=1),
        encoding="utf-8",
    )
    print(f"kept {kept} tensors ({skipped} non-weight constants skipped)")
    total = sum(t.size for t in tensors.values())
    print(f"total params: {total/1e6:.2f} M")
    for prefix in ("enc_p", "dp", "flow", "dec"):
        n = sum(t.size for k, t in tensors.items() if k.startswith(prefix))
        print(f"  {prefix:<6} {n/1e6:.2f} M")


if __name__ == "__main__":
    main()
