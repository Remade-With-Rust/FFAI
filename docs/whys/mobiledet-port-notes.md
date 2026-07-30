# PP-OCRv5 mobile-det port notes (pre-port, verified against source + checkpoint)

Weights: `ppocrv5-mobile-det` manifest — `det-train.safetensors` (905 tensors,
STRUCTURAL names; the export-fused program has op-indexed names only, blocking).
Oracle: `corpora/refs/fixtures/mobiledet_fixture.json` (reference polys+texts on
a pinned frame, mkldnn-off path). Shape maps: both dumped in fixtures/.

## LearnableRepLayer (verified in paddlex pplcnetv3.py, lines 88-152)

Branches, all SUMMED: optional identity `BatchNorm2D(in)` (iff in==out and
stride==1) + optional 1x1 conv-BN "small" branch (iff k>1) + `num_conv_branches`
x (kxk conv-BN). Each branch's conv is BIAS-FREE + BN. After the sum:
`lab` = per-tensor scalar affine (scale*x + bias, shapes (1,)), then IFF
stride != 2: `hardswish` followed by ANOTHER scalar affine.

**Fold at load** (numpy or Rust, gated by the fixture):
1. conv+BN -> W' = W * g/sqrt(v+eps) per out-ch, b' = beta - mu*g/sqrt(v+eps)
2. pad 1x1 kernels to kxk (center); identity BN -> folded identity kernel
3. sum all (W', b'); fold the FIRST lab into (W*s, b*s + t)
4. keep [hardswish + second affine] as the post-op (stride!=2 only)

Fused block == the export program's shape: one conv + scalars, which is why the
fused dump shows 104 anonymous `learnable_affine_block_N.w_0` scalars.

## Remaining structure (needs PaddleOCR repo, not in venv)

- det-variant PPLCNetV3: stage strides + which stages feed the neck
  (4 taps), det scale 0.75 with make_divisible channel rounding —
  read `ppocr/modeling/backbones/rep_*/det_pplcnet_v3` in a shallow clone.
- Neck RSEFPN(96): per-level 1x1 `neck.inp_conv.N` + 3x3 `neck.ins_conv.N`
  (shapes in train map), SE per level, top-down upsample+add.
- DBHead `head.binarize`/`head.thresh`: conv3x3-BN-relu -> convT 2x2-BN-relu
  -> convT 2x2 -> sigmoid; inference uses the binarize (probability) branch
  only. Postprocess: threshold 0.3, CC boxes, unclip ~1.5 (reuse boxes.rs
  machinery).

## Trap log
- `paddle.load` on `.pdiparams` returns a bare ndarray — use
  `paddle.static.load_inference_model` (fused) or the TRAIN checkpoint.
- paddlex `create_model` predictor hits the oneDNN PIR crash on this box;
  the mkldnn-off PaddleOCR() constructor path works (fixture came from it).
- Oracle fixtures must include LONG outputs (the parseq sqrt(d) lesson).
