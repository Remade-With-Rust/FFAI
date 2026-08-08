"""NumPy reference for PP-OCRv5_mobile_rec, driven BY the recorded graph (8.167).

Written to be wrong fast: numpy iterates in seconds where a candle port iterates
in minutes, so every structural misunderstanding surfaces here, against the
oracle fixture, before a line of Rust exists. When this matches paddle, the Rust
becomes a transliteration with a known-good target.

THE BACKBONE IS NOT TRANSCRIBED. Hand-copying 30 layers of stride/group/padding
is exactly the error class that produces fluent-but-wrong text, so the conv chain
is READ from `ppocrv5_mobile_rec_graph.json` and executed in recorded order. Only
the structural shapes (where SE branches sit, where the encoder begins) are
expressed in code, and each is a single recorded landmark.

Layout, as recorded:
  stem      conv2d_0 s2 + the one real batch_norm
  backbone  conv -> +bias -> LAB -> hardswish -> LAB, repeated; BN folded into
            the weights at export. Two squeeze-excite branches, marked in the
            graph by a pool2d whose kernel is [1,1] (adaptive -> 1x1 = global).
  neck      avgpool k=(3,2) s=(3,2): H 3->1, W 80->40 (the 40 timesteps)
            conv131+BN+swish -> conv132+BN+swish -> flatten -> transpose
  encoder   2 pre-norm blocks, dim 120, 8 heads x 15, scale 1/sqrt(15),
            MLP 120->240->120 swish. Dropout 0.1 is identity at inference.
  head      LN -> (B,1,N,120) -> conv133+BN+swish -> concat -> conv134 (pad 0,1)
            +BN+swish -> conv135+BN+swish -> squeeze -> linear_8 -> softmax
"""
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

REPO = Path(__file__).resolve().parent.parent
W = load_file(str(Path.home() / "AppData/Local/ffai/models/ppocrv5-mobile-rec/rec.safetensors"))
FIX = REPO / "corpora/refs/fixtures"
GRAPH = json.loads((FIX / "ppocrv5_mobile_rec_graph.json").read_text(encoding="utf-8"))
EPS = 1e-5


def conv(x, w, stride=(1, 1), pad=(0, 0), groups=1):
    """NCHW cross-correlation via sliding windows; reference clarity over speed."""
    if pad != (0, 0):
        x = np.pad(x, ((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])))
    n = x.shape[0]
    oc, icg, kh, kw = w.shape
    cols = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
    cols = cols[:, :, ::stride[0], ::stride[1]]           # n, c, oh, ow, kh, kw
    oh, ow = cols.shape[2], cols.shape[3]
    ocg = oc // groups
    out = np.empty((n, oc, oh, ow), dtype=np.float32)
    for g in range(groups):
        c = cols[:, g * icg:(g + 1) * icg]
        c = c.transpose(0, 2, 3, 1, 4, 5).reshape(n, oh * ow, -1)
        ww = w[g * ocg:(g + 1) * ocg].reshape(ocg, -1).T
        out[:, g * ocg:(g + 1) * ocg] = (c @ ww).reshape(n, oh, ow, ocg).transpose(0, 3, 1, 2)
    return out


def bn(x, p):
    mean, var, gamma, beta = W[f"{p}.w_1"], W[f"{p}.w_2"], W[f"{p}.w_0"], W[f"{p}.b_0"]
    s = gamma / np.sqrt(var + EPS)
    return x * s[None, :, None, None] + (beta - mean * s)[None, :, None, None]


def hardswish(x):
    return x * np.clip(x + 3.0, 0.0, 6.0) / 6.0


def hardsigmoid(x):
    return np.clip(x / 6.0 + 0.5, 0.0, 1.0)


def swish(x):
    return x / (1.0 + np.exp(-x))


def ln(x, p, eps=EPS):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * W[f"{p}.w_0"] + W[f"{p}.b_0"]


def avgpool(x, k, s):
    n, c, h, w = x.shape
    oh = (h - k[0]) // s[0] + 1
    ow = (w - k[1]) // s[1] + 1
    v = np.lib.stride_tricks.sliding_window_view(x, (k[0], k[1]), axis=(2, 3))
    return v[:, :, ::s[0], ::s[1]].mean(axis=(4, 5))[:, :, :oh, :ow]


def run_backbone(x, verbose):
    """Execute the recorded conv chain up to the encoder's neck pool."""
    lab = [0]

    def apply_lab(t):
        i = lab[0]
        lab[0] += 1
        return (t * W[f"learnable_affine_block_{i}.w_0"].reshape(1, -1, 1, 1)
                + W[f"learnable_affine_block_{i}.w_1"].reshape(1, -1, 1, 1))

    i = 0
    n_blocks = 0
    while i < len(GRAPH):
        r = GRAPH[i]
        op, at = r["op"], r.get("attr", {})
        if op == "pool2d" and at.get("strides") == [3, 2]:
            break                                   # the neck pool: backbone done
        if op == "pool2d":                          # squeeze-excite branch
            red, exp = GRAPH[i + 1], GRAPH[i + 6]
            s = x.mean(axis=(2, 3), keepdims=True)  # adaptive 1x1 == global avg
            rn, en = red["params"][0][:-4], exp["params"][0][:-4]
            s = conv(s, W[f"{rn}.w_0"]) + W[f"{rn}.b_0"].reshape(1, -1, 1, 1)
            s = np.maximum(s, 0.0)
            s = conv(s, W[f"{en}.w_0"]) + W[f"{en}.b_0"].reshape(1, -1, 1, 1)
            x = x * hardsigmoid(s)
            while GRAPH[i]["op"] != "multiply" or "learnable" in str(GRAPH[i].get("params")):
                i += 1
            i += 1
            continue
        if op in ("conv2d", "depthwise_conv2d"):
            name = r["params"][0][:-4]
            x = conv(x, W[f"{name}.w_0"], tuple(at["strides"]), tuple(at["paddings"]),
                     at.get("groups", 1))
            if GRAPH[i + 1]["op"] == "batch_norm_":         # stem only
                x = bn(x, GRAPH[i + 1]["params"][0][:-4])
                i += 2
                continue
            x = x + W[f"{name}.b_0"].reshape(1, -1, 1, 1)
            x = apply_lab(x)
            x = hardswish(x)
            x = apply_lab(x)
            n_blocks += 1
            i += 8                                          # skip the recorded unit
            continue
        i += 1
    if verbose:
        print(f"    backbone: {n_blocks} blocks, {lab[0]} LABs, out {x.shape}")
    return x


def attn(x, qkv, proj):
    b, n, c = x.shape
    t = x @ W[f"{qkv}.w_0"] + W[f"{qkv}.b_0"]
    t = t.reshape(b, n, 3, 8, 15).transpose(2, 0, 3, 1, 4)
    q, k, v = t[0] * (15 ** -0.5), t[1], t[2]
    a = q @ k.transpose(0, 1, 3, 2)
    a = np.exp(a - a.max(-1, keepdims=True))
    a = a / a.sum(-1, keepdims=True)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(b, n, c)
    return o @ W[f"{proj}.w_0"] + W[f"{proj}.b_0"]


def enc(x, l1, qkv, proj, l2, fc1, fc2):
    x = x + attn(ln(x, l1), qkv, proj)
    h = swish(ln(x, l2) @ W[f"{fc1}.w_0"] + W[f"{fc1}.b_0"])
    return x + (h @ W[f"{fc2}.w_0"] + W[f"{fc2}.b_0"])


def forward(x, verbose=True):
    x = run_backbone(x, verbose)
    x = avgpool(x, (3, 2), (3, 2))
    # EncoderWithSVTR keeps the pre-reduction tensor as a SHORTCUT and
    # concatenates it with the encoder output later (`cat(h, z)`). The dropout
    # op's struct_name gave this away: /MultiHead/SequenceEncoder/EncoderWithSVTR/.
    # 960 = 480 shortcut + 480 encoder, not a duplicated branch.
    shortcut = x
    # Neck convs carry NO bias: they are followed by a real BN, which supplies
    # it. Only the backbone's BN-folded convs have a `.b_0`.
    x = swish(bn(conv(x, W["conv2d_131.w_0"], pad=(0, 1)), "batch_norm2d_146"))
    x = swish(bn(conv(x, W["conv2d_132.w_0"]), "batch_norm2d_147"))
    b, c, h, w = x.shape
    if verbose:
        print(f"    neck out {x.shape}")
    x = x.reshape(b, c, h * w).transpose(0, 2, 1)
    x = enc(x, "layer_norm_0", "linear_0", "linear_1", "layer_norm_1", "linear_2", "linear_3")
    x = enc(x, "layer_norm_2", "linear_4", "linear_5", "layer_norm_3", "linear_6", "linear_7")
    x = ln(x, "layer_norm_4", 1e-6)
    if verbose:
        print(f"    encoder out {x.shape}")
    # HEAD, as recorded (ops 381-404). The first cut jumped straight to linear_8
    # and every shape still matched — which is exactly why shape agreement is not
    # correctness. conv134 takes 960 = 480+480 in-channels, so the recorded
    # `concat` doubles conv133's output along the channel axis.
    b2, n2, c2 = x.shape
    h = x.reshape(b2, 1, n2, c2).transpose(0, 3, 1, 2)          # (B,120,1,N)
    h = swish(bn(conv(h, W["conv2d_133.w_0"]), "batch_norm2d_148"))
    h = np.concatenate([shortcut, h], axis=1)                    # 480 + 480 -> 960
    h = swish(bn(conv(h, W["conv2d_134.w_0"], pad=(0, 1)), "batch_norm2d_149"))
    h = swish(bn(conv(h, W["conv2d_135.w_0"]), "batch_norm2d_150"))
    x = h.squeeze(2).transpose(0, 2, 1)                          # (B,N,120)
    if verbose:
        print(f"    head out {x.shape}")
    y = x @ W["linear_8.w_0"] + W["linear_8.b_0"]
    e = np.exp(y - y.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def main():
    meta = json.loads((FIX / "svtr_fixture.json").read_text(encoding="utf-8"))
    x = np.fromfile(FIX / meta["input"], dtype=np.float32).reshape(meta["input_shape"])
    want = np.fromfile(FIX / meta["output"], dtype=np.float32).reshape(meta["output_shape"])
    got = forward(x)
    print(f"  got {got.shape}   want {want.shape}")
    if got.shape != want.shape:
        print("  SHAPE MISMATCH — structure is wrong, not just weights")
        return
    d = np.abs(got - want)
    print(f"  max abs diff {d.max():.3e}   mean {d.mean():.3e}")
    print(f"  argmax agreement {(got.argmax(-1) == want.argmax(-1)).mean() * 100:.1f} %")
    print("  MATCH" if d.max() < 1e-4 else "  NOT YET")


if __name__ == "__main__":
    main()
