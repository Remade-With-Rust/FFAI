"""Dump per-stage VITS fixtures from piper's own onnxruntime, at zero noise
(deterministic), for Mercury's stage oracles (M-T2).

Promotes the stage-boundary tensors to graph outputs and runs fixed phoneme
sequences through the ORIGINAL runtime:

    m_p, logs_p   /enc_p/Split_output_0, _1     (text encoder)
    w_ceil        /Ceil_output_0                (duration predictor, integer)
    z_p           <input of the first flow op>  (expanded prior, zero noise)
    dec_in        /Mul_7_output_0               (flow output * mask)
    audio         output

Each requested sentence becomes fixtures/<id>.safetensors with all six
tensors. A WER regression in candle must be attributable to ONE stage in
minutes — the mel-oracle discipline, applied to synthesis.

    dump_vits_stages.py --model <voice.onnx> --fixtures <espeak jsonl> \
        --ids hvd-01-01,hvd-01-02 --out crates/ffai-mercury/tests/fixtures/vits
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from safetensors.numpy import save_file

BOUNDARIES = {
    "m_p": "/enc_p/Split_output_0",
    "logs_p": "/enc_p/Split_output_1",
    "w_ceil": "/Ceil_output_0",
    "dec_in": "/Mul_7_output_0",
}


def flow_input(g) -> str:
    """The tensor entering the first /flow/ node (z_p, post noise-add)."""
    init = {t.name for t in g.initializer}
    for node in g.node:
        if node.name.startswith("/flow/"):
            for i in node.input:
                if i not in init and not i.startswith("/flow/"):
                    return i
    raise SystemExit("no /flow/ input found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixtures", required=True, help="espeak phoneme jsonl (has phoneme_ids)")
    ap.add_argument("--ids", required=True, help="comma-separated sentence ids")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = onnx.load(args.model)
    g = model.graph
    boundaries = dict(BOUNDARIES)
    boundaries["z_p"] = flow_input(g)
    print("z_p tensor:", boundaries["z_p"])

    existing = {o.name for o in g.output}
    for tensor in boundaries.values():
        if tensor not in existing:
            g.output.append(onnx.ValueInfoProto(name=tensor))
    sess = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out_names = [o.name for o in sess.get_outputs()]

    wanted = set(args.ids.split(","))
    rows = {}
    for line in Path(args.fixtures).read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if obj["id"] in wanted:
                rows[obj["id"]] = obj["phoneme_ids"]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for sid in sorted(wanted):
        sentences = rows[sid]
        assert len(sentences) == 1, f"{sid}: multi-sentence fixture not supported"
        ids = np.array([sentences[0]], dtype=np.int64)
        feeds = {
            "input": ids,
            "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
            # zero noise, unit length: fully deterministic.
            "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }
        results = dict(zip(out_names, sess.run(out_names, feeds)))
        tensors = {"ids": ids}
        tensors["audio"] = results["output"].astype(np.float32)
        for key, tensor_name in boundaries.items():
            tensors[key] = results[tensor_name].astype(np.float32)
        save_file(tensors, str(outdir / f"{sid}.safetensors"))
        shapes = {k: list(v.shape) for k, v in tensors.items()}
        print(f"{sid}: {shapes}")


if __name__ == "__main__":
    main()
