//! CRAFT text detection (Baek et al. 2019) on candle — VGG16-BN backbone,
//! U-Net-style decoder, two output maps (character region + affinity).
//!
//! Ported layer-for-layer from clovaai/CRAFT-pytorch (MIT), weights
//! `craft_mlt_25k` converted to safetensors by `tools/carmenta_m_c1_prepare.py`.
//! Every shape below was verified against the checkpoint under
//! `load_state_dict(strict=True)` — the constructor widths in the original
//! paper's figure are NOT what the shipped weights use (conv0 of each upconv
//! is (in+mid)→mid, and the upconv ladder is 512/256/128/64 mid-widths).
//!
//! Two faithful quirks, kept because the weights were trained with them:
//! - Skip connections tap the `BatchNorm` output BEFORE its `ReLU` (the slice
//!   boundaries land on BN layers; the next slice starts with the `ReLU`).
//! - `slice4` ends at BN(features.38): the "`relu5_3`" input to the decoder is
//!   also pre-activation.
//!
//! Oracle: region/affinity maps match the `PyTorch` reference on the pinned
//! fixture crop — see `tests/craft_oracle.rs`.

use candle_core::{Result, Tensor};
use candle_nn::{batch_norm, conv2d, BatchNorm, Conv2d, Conv2dConfig, Module, ModuleT, VarBuilder};

/// One VGG conv block: Conv3x3(pad 1) + `BatchNorm`, indexed like torchvision's
/// `features` so `VarBuilder` paths match the checkpoint keys exactly.
struct ConvBn {
    conv: Conv2d,
    bn: BatchNorm,
}

impl ConvBn {
    fn new(vb: &VarBuilder, prefix: &str, conv_idx: usize, cin: usize, cout: usize) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            conv: conv2d(cin, cout, 3, cfg, vb.pp(format!("{prefix}.{conv_idx}")))?,
            bn: batch_norm(cout, 1e-5, vb.pp(format!("{prefix}.{}", conv_idx + 1)))?,
        })
    }

    /// Conv + BN, NO `ReLU` — the caller decides, because slice boundaries tap
    /// pre-activation outputs.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.bn.forward_t(&det_conv(x, &self.conv)?, false)
    }
}

fn relu(x: &Tensor) -> Result<Tensor> {
    x.relu()
}

/// Route a detector convolution through our kernel or candle's.
///
/// **Off by default, and the default is the measured answer.** CRAFT is
/// VGG16-BN, so every 3x3 here matches `crate::conv3x3`'s preconditions
/// exactly, and that kernel measured 1.65x candle on the CRNN (§8.101) — which
/// makes wiring it into the detector look free, and is why it is worth writing
/// down that it is not. Measured here, interleaved, min-of-rounds: the kernel
/// is **slower** on CRAFT's shapes.
///
/// The kernel is SHAPE-SPECIALISED. Its register tile was swept on CRNN crops:
/// h=64 collapsing to 3, widths 58..1357, channels 64..256 — tall-thin strips
/// where holding 4 channels x 24 columns in sixteen ymm registers beats
/// im2col's 9x memory expansion. CRAFT runs a 1280-long canvas at 64..512
/// channels: large square maps, where im2col hands `gemm` a big well-blocked
/// matmul and wins. Same arithmetic, different regime.
///
/// Kept as a switch rather than deleted because the regime boundary is the
/// finding — a future tile sweep for detector shapes has its A/B already
/// built, and `FFAI_CONV3X3_DET=1` is how the number above was obtained.
fn det_conv(x: &Tensor, conv: &Conv2d) -> Result<Tensor> {
    if std::env::var("FFAI_CONV3X3_DET").as_deref() == Ok("1") {
        crate::conv3x3::apply(x, conv)
    } else {
        conv.forward(x)
    }
}

fn max_pool2(x: &Tensor) -> Result<Tensor> {
    x.max_pool2d_with_stride(2, 2)
}

/// `double_conv`: 1x1 (in+mid)->mid + BN + `ReLU`, 3x3 mid->out + BN + `ReLU`.
struct DoubleConv {
    conv0: Conv2d,
    bn1: BatchNorm,
    conv3: Conv2d,
    bn4: BatchNorm,
}

impl DoubleConv {
    fn new(vb: &VarBuilder, name: &str, cat_in: usize, mid: usize, out: usize) -> Result<Self> {
        let vb = vb.pp(name).pp("conv");
        Ok(Self {
            conv0: conv2d(cat_in, mid, 1, Conv2dConfig::default(), vb.pp("0"))?,
            bn1: batch_norm(mid, 1e-5, vb.pp("1"))?,
            conv3: conv2d(mid, out, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("3"))?,
            bn4: batch_norm(out, 1e-5, vb.pp("4"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.bn1.forward_t(&det_conv(x, &self.conv0)?, false)?.relu()?;
        self.bn4.forward_t(&det_conv(&x, &self.conv3)?, false)?.relu()
    }
}

pub struct Craft {
    // basenet slices; fields hold the ConvBn blocks in order.
    s1: [ConvBn; 2],           // features 0/1, 3/4
    s1b: [ConvBn; 2],          // features 7/8, 10/11 (after pool 6)
    s2: [ConvBn; 2],           // features 14/15, 17/18 (after pool 13)
    s3: [ConvBn; 3],           // features 20/21, 24/25 (pool 23 between), 27/28
    s4: [ConvBn; 3],           // features 30/31, 34/35 (pool 33 between), 37/38
    conv6: Conv2d,             // slice5.1: 512->1024, 3x3, pad 6, dilation 6
    conv7: Conv2d,             // slice5.2: 1024->1024, 1x1
    up1: DoubleConv,
    up2: DoubleConv,
    up3: DoubleConv,
    up4: DoubleConv,
    cls: [Conv2d; 5],
}

impl Craft {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let b = |p: &str, i, cin, cout| ConvBn::new(&vb, &format!("basenet.{p}"), i, cin, cout);
        Ok(Self {
            s1: [b("slice1", 0, 3, 64)?, b("slice1", 3, 64, 64)?],
            s1b: [b("slice1", 7, 64, 128)?, b("slice1", 10, 128, 128)?],
            s2: [b("slice2", 14, 128, 256)?, b("slice2", 17, 256, 256)?],
            s3: [b("slice3", 20, 256, 256)?, b("slice3", 24, 256, 512)?, b("slice3", 27, 512, 512)?],
            s4: [b("slice4", 30, 512, 512)?, b("slice4", 34, 512, 512)?, b("slice4", 37, 512, 512)?],
            conv6: conv2d(
                512,
                1024,
                3,
                Conv2dConfig { padding: 6, dilation: 6, ..Default::default() },
                vb.pp("basenet.slice5.1"),
            )?,
            conv7: conv2d(1024, 1024, 1, Conv2dConfig::default(), vb.pp("basenet.slice5.2"))?,
            up1: DoubleConv::new(&vb, "upconv1", 1536, 512, 256)?,
            up2: DoubleConv::new(&vb, "upconv2", 768, 256, 128)?,
            up3: DoubleConv::new(&vb, "upconv3", 384, 128, 64)?,
            up4: DoubleConv::new(&vb, "upconv4", 192, 64, 32)?,
            cls: [
                conv2d(32, 32, 3, pad1(), vb.pp("conv_cls.0"))?,
                conv2d(32, 32, 3, pad1(), vb.pp("conv_cls.2"))?,
                conv2d(32, 16, 3, pad1(), vb.pp("conv_cls.4"))?,
                conv2d(16, 16, 1, Conv2dConfig::default(), vb.pp("conv_cls.6"))?,
                conv2d(16, 2, 1, Conv2dConfig::default(), vb.pp("conv_cls.8"))?,
            ],
        })
    }

    /// Input: (1, 3, H, W), CRAFT-normalized RGB. Output: (H/2, W/2, 2) —
    /// channel 0 = character region score, channel 1 = affinity score.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // ---- backbone ----
        //
        // SKIP-TAP SEMANTICS, measured not assumed: the reference saves each
        // slice output *before* the next slice runs, but torchvision's ReLUs
        // are `inplace=True`, and slices 2-4 BEGIN with the ReLU belonging to
        // the previous BN — which mutates the saved tensor retroactively. So
        // the effective skips are POST-ReLU for relu2_2/relu3_2/relu4_3, and
        // genuinely PRE-ReLU for relu5_3 only, because slice5 begins with a
        // MaxPool instead (its reference stats go negative: mean -0.32).
        // Getting this wrong was a 0.96 max-abs oracle failure.
        let x = relu(&self.s1[0].forward(x)?)?;
        let x = relu(&self.s1[1].forward(&x)?)?;
        let x = max_pool2(&x)?;
        let x = relu(&self.s1b[0].forward(&x)?)?;
        let relu2_2 = relu(&self.s1b[1].forward(&x)?)?; // post-ReLU (inplace aliasing)

        let x = max_pool2(&relu2_2)?;
        let x = relu(&self.s2[0].forward(&x)?)?;
        let relu3_2 = relu(&self.s2[1].forward(&x)?)?; // post-ReLU

        let x = relu(&self.s3[0].forward(&relu3_2)?)?;
        let x = max_pool2(&x)?;
        let x = relu(&self.s3[1].forward(&x)?)?;
        let relu4_3 = relu(&self.s3[2].forward(&x)?)?; // post-ReLU

        let x = relu(&self.s4[0].forward(&relu4_3)?)?;
        let x = max_pool2(&x)?;
        let x = relu(&self.s4[1].forward(&x)?)?;
        let relu5_3 = self.s4[2].forward(&x)?; // PRE-ReLU: slice5 starts with a pool

        // slice5: maxpool 3x3 s1 p1 (zero-pad legal: input is a BN output fed
        // straight to pooling in the original too — values may be negative,
        // so pad with the tensor's own replicated edge instead of zeros).
        let x = pad_same_2d(&relu5_3)?.max_pool2d_with_stride(3, 1)?;
        let x = self.conv6.forward(&x)?;
        let fc7 = self.conv7.forward(&x)?;

        // ---- decoder ----
        let y = Tensor::cat(&[&fc7, &relu5_3], 1)?;
        let y = self.up1.forward(&y)?;
        let y = upsample_bilinear(&y, relu4_3.dim(2)?, relu4_3.dim(3)?)?;
        let y = Tensor::cat(&[&y, &relu4_3], 1)?;
        let y = self.up2.forward(&y)?;
        let y = upsample_bilinear(&y, relu3_2.dim(2)?, relu3_2.dim(3)?)?;
        let y = Tensor::cat(&[&y, &relu3_2], 1)?;
        let y = self.up3.forward(&y)?;
        let y = upsample_bilinear(&y, relu2_2.dim(2)?, relu2_2.dim(3)?)?;
        let y = Tensor::cat(&[&y, &relu2_2], 1)?;
        let y = self.up4.forward(&y)?;

        let y = relu(&det_conv(&y, &self.cls[0])?)?;
        let y = relu(&det_conv(&y, &self.cls[1])?)?;
        let y = relu(&det_conv(&y, &self.cls[2])?)?;
        let y = relu(&det_conv(&y, &self.cls[3])?)?;
        let y = det_conv(&y, &self.cls[4])?;
        // (1, 2, h, w) -> (h, w, 2), matching the reference's permute.
        y.squeeze(0)?.permute((1, 2, 0))
    }
}

fn pad1() -> Conv2dConfig {
    Conv2dConfig { padding: 1, ..Default::default() }
}

/// Edge-replicate pad by 1 on both spatial dims (torch `MaxPool2d` padding
/// semantics use -inf, but replicate >= -inf gives identical maxima on the
/// interior and edge windows alike, and candle's pooling has no pad param).
fn pad_same_2d(x: &Tensor) -> Result<Tensor> {
    x.pad_with_same(2, 1, 1)?.pad_with_same(3, 1, 1)
}

/// Bilinear upsample to an exact (`out_h`, `out_w`), `align_corners=false` —
/// `PyTorch` `F.interpolate(mode="bilinear")` semantics, computed on CPU.
/// candle 0.11 ships only nearest-neighbour; nearest here fails the oracle.
pub fn upsample_bilinear(x: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    let (b, c, in_h, in_w) = x.dims4()?;
    debug_assert_eq!(b, 1);
    let data = x.flatten_all()?.to_vec1::<f32>()?;
    let scale_h = in_h as f32 / out_h as f32;
    let scale_w = in_w as f32 / out_w as f32;

    let mut out = vec![0f32; c * out_h * out_w];
    // Precompute per-axis source indices/weights once; reuse across channels.
    let axis = |out_n: usize, in_n: usize, scale: f32| -> Vec<(usize, usize, f32)> {
        (0..out_n)
            .map(|o| {
                let src = ((o as f32 + 0.5) * scale - 0.5).max(0.0);
                let i0 = (src.floor() as usize).min(in_n - 1);
                let i1 = (i0 + 1).min(in_n - 1);
                (i0, i1, src - src.floor())
            })
            .collect()
    };
    let ys = axis(out_h, in_h, scale_h);
    let xs = axis(out_w, in_w, scale_w);

    for ch in 0..c {
        let src = &data[ch * in_h * in_w..(ch + 1) * in_h * in_w];
        let dst = &mut out[ch * out_h * out_w..(ch + 1) * out_h * out_w];
        for (oy, &(y0, y1, fy)) in ys.iter().enumerate() {
            for (ox, &(x0, x1, fx)) in xs.iter().enumerate() {
                let top = src[y0 * in_w + x0] * (1.0 - fx) + src[y0 * in_w + x1] * fx;
                let bot = src[y1 * in_w + x0] * (1.0 - fx) + src[y1 * in_w + x1] * fx;
                dst[oy * out_w + ox] = top * (1.0 - fy) + bot * fy;
            }
        }
    }
    Tensor::from_vec(out, (1, c, out_h, out_w), x.device())
}
