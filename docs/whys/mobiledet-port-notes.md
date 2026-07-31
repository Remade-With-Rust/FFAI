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


---

## PORT DE-RISKED (2026-07-30): the rep branches are fused offline

`tools/carmenta_mobiledet_fuse.py` collapses every LearnableRepLayer into a
single conv weight+bias, **905 tensors -> 275**, with an algebraic self-check
(branch-sum vs fused conv on random input, max abs diff **4.88e-04**). The
candle port therefore never implements rep branches at all — it loads
`det-fused.safetensors` and sees plain convolutions.

What fused: 4x `conv_kxk` + optional `conv_1x1` + their BatchNorms + the
pre-activation `lab` affine. What did NOT (and must be implemented): the
POST-activation `act.lab` scalar affine, because it sits after hardswish.
Its per-layer scale/bias are in
`corpora/refs/fixtures/ppocrv5_mobile_det_fused.json`.

### Backbone (PP-LCNetV3), derived from the checkpoint

Stem: `backbone.conv1` conv3x3 3->16 stride 2 (+BN, fused).

| stage | idx | depthwise | pointwise |
|---|---|---|---|
| 2 | 0 | k3 ch16 g16 | 16 -> 32 |
| 3 | 0,1 | k3 ch32/48 | 32->48, 48->48 |
| 4 | 0,1 | k3 ch48/96 | 48->96, 96->96 |
| 5 | 0-4 | k3 ch96, then k5 ch192 x4 | 96->192, then 192->192 x4 |
| 6 | 0-3 | k5 ch192, then k5 ch384 x3 | 192->384, then 384->384 x3 |

Every block: depthwise conv -> hardswish -> `act.lab` affine -> pointwise
conv -> hardswish -> `act.lab` affine. Strides live at each stage's index 0.

Taps: `backbone.layer_list.{0..3}` are 1x1 convs WITH bias emitting
**12 / 18 / 42 / 360** channels — exactly the neck's four inputs.

### Neck (RSEFPN, 96 wide, 24 out per level)

- `ins_conv.{0..3}`: 1x1 [12,18,42,360] -> 96, each followed by an SE block
  (`se_block.conv1` 96->24, `se_block.conv2` 24->96).
- `inp_conv.{0..3}`: 3x3 96 -> 24, each with SE (24->6->24).
- Top-down: upsample+add across levels, then all four upsampled to 1/4 and
  concatenated -> 96 channels.

### Head (DBHead) — inference needs `binarize` only

`conv1` 3x3 96->24 + `conv_bn1` + relu -> `conv2` **transposed** 2x2 24->24 +
`conv_bn2` + relu -> `conv3` **transposed** 2x2 24->1 -> sigmoid. The two
transposed convs restore the 1/4 map to input resolution.

`thresh` has the identical shape and is training-only; do not load it.

### Postprocess

Binarize the probability map at 0.3, connected components, then unclip each
region by ~1.5x its area/perimeter ratio (Vatti/offset). `boxes.rs` already
has the component walk; only the unclip is new.

**Why this port is the highest-value brick left:** it closes BOTH open
fronts at once. Detection is 62-89 % of frame time (CRAFT is a VGG16;
this is 4.7 MB), and §8.13 measured crop geometry as a 16-point CER cause
that exists only because CRAFT components have to be reconstructed into word
boxes — DBNet emits the polygons directly.
