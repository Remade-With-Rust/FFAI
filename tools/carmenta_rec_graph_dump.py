"""Record a PP-OCR recognizer's inference program as a DAG fixture (R.2).

`ppocrv5_mobile_rec_graph.json` (the record `svtr.rs` was generated from) is
an op SEQUENCE: the mobile backbone is a chain, so recorded order was enough.
The server tier's HGNetV2 backbone is not a chain — it has pooled shortcuts and
eight concats — so this record carries, for every op, the ops its operands come
from. A port written against this file cannot mis-wire a branch the way a
sequence read could; the R.1 lesson (a port drifts where its record is silent)
applies to structure as much as to constants.

Usage (paddle venv):
    python tools/carmenta_rec_graph_dump.py PP-OCRv5_server_rec
    python tools/carmenta_rec_graph_dump.py PP-OCRv6_medium_rec

Writes corpora/refs/fixtures/ppocrv5_server_rec_graph.json (the mobile fixture's naming): nodes in program order with
op name, input node ids (parameters by NAME with shape), result shapes where
paddle can state them, and the attributes a port needs. Parameters are listed
once, by name, with shape and a sha256 of their bytes. Nothing here touches
the shipped engine.
"""
import hashlib
import json
import sys
from pathlib import Path

ATTRS = ("strides", "paddings", "dilations", "groups", "epsilon", "axis", "perm", "shape",
         "value", "kernel_size", "pooling_type", "adaptive", "ceil_mode", "global_pooling",
         "exclusive", "padding_algorithm", "approximate", "begin_norm_axis", "scale", "bias",
         "num_or_sections", "axes", "starts", "ends", "start", "stop", "step", "transpose_x",
         "transpose_y", "keepdim", "dim", "slope", "offset", "threshold")


def main(model):
    import numpy as np
    import paddle

    root = Path.home() / ".paddlex" / "official_models" / model
    assert root.exists(), f"{root} missing; instantiate paddleocr.TextRecognition(model_name={model!r}) once"
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    prog, feeds, fetches = paddle.static.load_inference_model(str(root / "inference"), exe)
    block = prog.global_block()
    ops = list(block.ops)
    scope = paddle.static.global_scope()

    # Parameters: name -> shape + digest, once.
    params = {}
    for p in block.all_parameters():
        arr = np.array(scope.find_var(p.name).get_tensor())
        params[p.name] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                          "sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]}

    # Map each op to a stable id in program order; parameter ops carry their name.
    ids = {}
    for i, op in enumerate(ops):
        ids[op.id()] = i
    pname = {}
    for op in ops:
        if op.name() == "builtin.parameter":
            pname[op.id()] = op.attrs().get("parameter_name")

    nodes = []
    for i, op in enumerate(ops):
        name = op.name()
        if name in ("builtin.parameter", "pd_op.fetch"):
            continue
        ins = []
        for j in range(op.num_operands()):
            src = op.operand_source(j)
            d = src.get_defining_op() if src is not None else None
            if d is None:
                ins.append(None)
            elif d.name() == "builtin.parameter":
                ins.append({"param": pname[d.id()], "shape": params[pname[d.id()]]["shape"]})
            elif d.name() == "builtin.combine":
                # combine packs a list of values; flatten to their producers
                subs = []
                for jj in range(d.num_operands()):
                    s2 = d.operand_source(jj)
                    d2 = s2.get_defining_op() if s2 is not None else None
                    if d2 is None:
                        subs.append(None)
                    elif d2.name() == "builtin.parameter":
                        subs.append({"param": pname[d2.id()], "shape": params[pname[d2.id()]]["shape"]})
                    else:
                        subs.append(ids[d2.id()])
                ins.append({"combine": subs})
            else:
                ins.append(ids[d.id()])
        outs = []
        for r in op.results():
            try:
                outs.append([int(v) for v in r.shape])
            except Exception:
                outs.append(None)
        attrs = {k: v for k, v in op.attrs().items() if k in ATTRS}
        nodes.append({"i": i, "op": name.split(".")[-1], "in": ins, "out": outs, "attr": attrs})

    fix = Path("corpora/refs/fixtures")
    fix.mkdir(parents=True, exist_ok=True)
    out = fix / f"{model.lower().replace('-', '')}_graph.json"
    out.write_text(json.dumps({"model": model, "feeds": list(feeds), "n_params": len(params),
                               "params": params, "nodes": nodes}, ensure_ascii=False, indent=0),
                   encoding="utf-8")
    kinds = {}
    for n in nodes:
        kinds[n["op"]] = kinds.get(n["op"], 0) + 1
    print(f"  {model}: {len(nodes)} nodes, {len(params)} params -> {out}")
    print("  " + ", ".join(f"{k}:{v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "PP-OCRv5_server_rec")
