//! PP-OCRv5 mobile recognition (SVTR) — the matched half of our detector.
//!
//! We pair a 2024 PP-OCRv5 DETECTOR with EasyOCR's 2019 VGG-CRNN recognizer.
//! §8.165 measured the mismatch: with detection held constant and only the
//! recognizer swapped, SVTR reads our hardest stylized pages at 40.12 % against
//! CRNN's 44.13 % (-4.01 pp) AND the clean controls at 2.66 % against 3.20 %
//! (-0.53 pp). It wins on both halves, so the 18,383-class multilingual head's
//! confusability risk did not materialise.
//!
//! ## Written against the checkpoint, not a paper
//!
//! Structure and constants come from the recorded inference program
//! (`corpora/refs/fixtures/ppocrv5_mobile_rec_graph.json`, 407 ops with their
//! weights, strides, groups and paddings), and the backbone table below is
//! GENERATED from it rather than transcribed — hand-copying 30 layers is the
//! error class that yields fluent-but-wrong text.
//!
//! §8.167 wrote a numpy reference first and matched the paddle oracle at
//! **1.3e-5**. Three hypotheses died there, two of which looked correct:
//! skipping the head convs still matched EVERY SHAPE, and `concat([h, h])`
//! scored 100 % argmax agreement while being structurally wrong. What fixed it
//! was an op's `struct_name` — `/MultiHead/SequenceEncoder/EncoderWithSVTR/` —
//! revealing that the encoder keeps a SHORTCUT concatenated with its own output
//! (960 = 480 shortcut + 480 encoder). This port carries that structure.
//!
//! `examples/svtr_oracle.rs` asserts this module against the same fixture.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{ops::softmax, Module, VarBuilder};

/// One backbone unit: conv -> +bias -> LAB -> hardswish -> LAB.
/// BatchNorm is folded into the weights at export, so only the stem has a real
/// one; every other block's `.b_0` IS the folded bias.
struct Blk(&'static str, usize, (usize, usize), (usize, usize));
/// Squeeze-excite: global avg -> conv -> relu -> conv -> hardsigmoid -> scale.
struct Se(&'static str, &'static str);

enum Stage {
    Blk(Blk),
    Se(Se),
}

/// Generated from the recorded graph; 28 blocks and 2 SE branches, in order.
fn backbone_spec() -> Vec<Stage> {
    use Stage::{Blk as B, Se as S};
    vec![
        B(Blk("conv2d_136", 16, (1, 1), (1, 1))),
        B(Blk("conv2d_137", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_138", 32, (1, 1), (1, 1))),
        B(Blk("conv2d_139", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_140", 64, (1, 1), (1, 1))),
        B(Blk("conv2d_141", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_142", 64, (2, 1), (1, 1))),
        B(Blk("conv2d_143", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_144", 128, (1, 1), (1, 1))),
        B(Blk("conv2d_145", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_146", 128, (1, 2), (1, 1))),
        B(Blk("conv2d_147", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_148", 240, (1, 1), (2, 2))),
        B(Blk("conv2d_149", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_150", 240, (1, 1), (2, 2))),
        B(Blk("conv2d_151", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_152", 240, (1, 1), (2, 2))),
        B(Blk("conv2d_153", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_154", 240, (1, 1), (2, 2))),
        B(Blk("conv2d_155", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_156", 240, (2, 1), (2, 2))),
        S(Se("conv2d_96", "conv2d_97")),
        B(Blk("conv2d_157", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_158", 480, (1, 1), (2, 2))),
        S(Se("conv2d_107", "conv2d_108")),
        B(Blk("conv2d_159", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_160", 480, (2, 1), (2, 2))),
        B(Blk("conv2d_161", 1, (1, 1), (0, 0))),
        B(Blk("conv2d_162", 480, (1, 1), (2, 2))),
        B(Blk("conv2d_163", 1, (1, 1), (0, 0))),
    ]
}

const BN_EPS: f64 = 1e-5;
/// The final encoder norm uses a tighter epsilon than the rest (recorded).
const LN_EPS: f64 = 1e-5;
const LN_EPS_FINAL: f64 = 1e-6;
const HEADS: usize = 8;
const HEAD_DIM: usize = 15;

pub struct Svtr {
    vb: VarBuilder<'static>,
    n_class: usize,
}

fn hardswish(x: &Tensor) -> Result<Tensor> {
    ((x + 3.0)?.clamp(0.0, 6.0)? * x)? / 6.0
}

fn hardsigmoid(x: &Tensor) -> Result<Tensor> {
    ((x / 6.0)? + 0.5)?.clamp(0.0, 1.0)
}

fn swish(x: &Tensor) -> Result<Tensor> {
    x * candle_nn::ops::sigmoid(x)?
}

impl Svtr {
    pub fn new(vb: VarBuilder<'static>, n_class: usize) -> Result<Self> {
        Ok(Self { vb, n_class })
    }

    fn w(&self, name: &str, shape: (usize, usize, usize, usize)) -> Result<Tensor> {
        self.vb.get(shape, name)
    }

    /// Conv with the recorded stride/pad/groups, then the folded bias.
    fn conv(
        &self,
        x: &Tensor,
        name: &str,
        stride: (usize, usize),
        pad: (usize, usize),
        groups: usize,
        bias: bool,
    ) -> Result<Tensor> {
        let w = self.vb.get_unchecked(&format!("{name}.w_0"))?;
        let (oc, icg, kh, kw) = w.dims4()?;
        let _ = (oc, icg, kh, kw);
        // candle's conv2d takes ONE stride and ONE padding; the recorded model
        // uses asymmetric values (e.g. (2,1) to collapse height faster than
        // width). Emulate by padding manually and striding per axis.
        let x = x.pad_with_zeros(2, pad.0, pad.0)?.pad_with_zeros(3, pad.1, pad.1)?;
        let cfg = candle_nn::Conv2dConfig { padding: 0, stride: 1, dilation: 1, groups,
            ..Default::default() };
        let y = candle_nn::Conv2d::new(w, None, cfg).forward(&x)?;
        let y = if stride.0 > 1 {
            let n = (y.dim(2)? + stride.0 - 1) / stride.0;
            y.index_select(
                &Tensor::from_vec(
                    (0..n).map(|i| (i * stride.0) as u32).collect::<Vec<_>>(),
                    n,
                    y.device(),
                )?,
                2,
            )?
        } else {
            y
        };
        let y = if stride.1 > 1 {
            let n = (y.dim(3)? + stride.1 - 1) / stride.1;
            y.index_select(
                &Tensor::from_vec(
                    (0..n).map(|i| (i * stride.1) as u32).collect::<Vec<_>>(),
                    n,
                    y.device(),
                )?,
                3,
            )?
        } else {
            y
        };
        if bias {
            let b = self.vb.get_unchecked(&format!("{name}.b_0"))?;
            y.broadcast_add(&b.reshape((1, b.elem_count(), 1, 1))?)
        } else {
            Ok(y)
        }
    }

    fn bn(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let mean = self.vb.get_unchecked(&format!("{name}.w_1"))?;
        let var = self.vb.get_unchecked(&format!("{name}.w_2"))?;
        let gamma = self.vb.get_unchecked(&format!("{name}.w_0"))?;
        let beta = self.vb.get_unchecked(&format!("{name}.b_0"))?;
        let s = (gamma / (var + BN_EPS)?.sqrt()?)?;
        let shift = (beta - (mean * &s)?)?;
        let c = s.elem_count();
        x.broadcast_mul(&s.reshape((1, c, 1, 1))?)?
            .broadcast_add(&shift.reshape((1, c, 1, 1))?)
    }

    /// Learnable affine block: per-channel scale then shift, consumed in order.
    fn lab(&self, x: &Tensor, i: &mut usize) -> Result<Tensor> {
        let w = self.vb.get_unchecked(&format!("learnable_affine_block_{i}.w_0"))?;
        let b = self.vb.get_unchecked(&format!("learnable_affine_block_{i}.w_1"))?;
        *i += 1;
        let c = w.elem_count();
        x.broadcast_mul(&w.reshape((1, c, 1, 1))?)?
            .broadcast_add(&b.reshape((1, c, 1, 1))?)
    }

    fn ln(&self, x: &Tensor, name: &str, eps: f64) -> Result<Tensor> {
        let w = self.vb.get_unchecked(&format!("{name}.w_0"))?;
        let b = self.vb.get_unchecked(&format!("{name}.b_0"))?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let d = x.broadcast_sub(&mean)?;
        let var = d.sqr()?.mean_keepdim(D::Minus1)?;
        d.broadcast_div(&(var + eps)?.sqrt()?)?
            .broadcast_mul(&w)?
            .broadcast_add(&b)
    }

    fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let w = self.vb.get_unchecked(&format!("{name}.w_0"))?;
        let b = self.vb.get_unchecked(&format!("{name}.b_0"))?;
        x.broadcast_matmul(&w)?.broadcast_add(&b)
    }

    fn attention(&self, x: &Tensor, qkv: &str, proj: &str) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        let t = self
            .linear(x, qkv)?
            .reshape((b, n, 3, HEADS, HEAD_DIM))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let q = (t.i(0)? * (HEAD_DIM as f64).powf(-0.5))?;
        let k = t.i(1)?;
        let v = t.i(2)?;
        let a = q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let a = softmax(&a, D::Minus1)?;
        let o = a.matmul(&v.contiguous()?)?.transpose(1, 2)?.reshape((b, n, c))?;
        self.linear(&o, proj)
    }

    fn enc_block(
        &self,
        x: &Tensor,
        l1: &str,
        qkv: &str,
        proj: &str,
        l2: &str,
        fc1: &str,
        fc2: &str,
    ) -> Result<Tensor> {
        let x = (x + self.attention(&self.ln(x, l1, LN_EPS)?, qkv, proj)?)?;
        let h = swish(&self.linear(&self.ln(&x, l2, LN_EPS)?, fc1)?)?;
        &x + self.linear(&h, fc2)?
    }

    /// (1, 3, 48, W) normalised to [-1, 1] -> (1, T, n_class) probabilities.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.bn(&self.conv(x, "conv2d_0", (2, 2), (1, 1), 1, false)?, "batch_norm2d_0")?;
        let mut lab = 0usize;
        for st in backbone_spec() {
            match st {
                Stage::Blk(Blk(name, groups, stride, pad)) => {
                    x = self.conv(&x, name, stride, pad, groups, true)?;
                    x = self.lab(&x, &mut lab)?;
                    x = hardswish(&x)?;
                    x = self.lab(&x, &mut lab)?;
                }
                Stage::Se(Se(reduce, expand)) => {
                    let s = x.mean_keepdim(2)?.mean_keepdim(3)?;
                    let s = self.conv(&s, reduce, (1, 1), (0, 0), 1, true)?.relu()?;
                    let s = self.conv(&s, expand, (1, 1), (0, 0), 1, true)?;
                    x = x.broadcast_mul(&hardsigmoid(&s)?)?;
                }
            }
        }
        // Neck pool k=(3,2) s=(3,2): H 3->1, W 80->40 (the timesteps).
        let (_, _, h, w) = x.dims4()?;
        let x = x.avg_pool2d_with_stride((3, 2), (3, 2))?;
        let _ = (h, w);
        // SHORTCUT: EncoderWithSVTR keeps the pre-reduction tensor and
        // concatenates it with the encoder output later. 960 = 480 + 480.
        let shortcut = x.clone();
        let x = swish(&self.bn(&self.conv(&x, "conv2d_131", (1, 1), (0, 1), 1, false)?, "batch_norm2d_146")?)?;
        let x = swish(&self.bn(&self.conv(&x, "conv2d_132", (1, 1), (0, 0), 1, false)?, "batch_norm2d_147")?)?;
        let (b, c, h, w) = x.dims4()?;
        let x = x.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?;
        let x = self.enc_block(&x, "layer_norm_0", "linear_0", "linear_1", "layer_norm_1", "linear_2", "linear_3")?;
        let x = self.enc_block(&x, "layer_norm_2", "linear_4", "linear_5", "layer_norm_3", "linear_6", "linear_7")?;
        let x = self.ln(&x, "layer_norm_4", LN_EPS_FINAL)?;
        // Head: (B,N,C) -> (B,C,1,N) -> conv133 -> cat(shortcut) -> conv134 -> conv135
        let (b, n, c) = x.dims3()?;
        let hd = x.reshape((b, 1, n, c))?.permute((0, 3, 1, 2))?.contiguous()?;
        let hd = swish(&self.bn(&self.conv(&hd, "conv2d_133", (1, 1), (0, 0), 1, false)?, "batch_norm2d_148")?)?;
        let hd = Tensor::cat(&[&shortcut, &hd], 1)?;
        let hd = swish(&self.bn(&self.conv(&hd, "conv2d_134", (1, 1), (0, 1), 1, false)?, "batch_norm2d_149")?)?;
        let hd = swish(&self.bn(&self.conv(&hd, "conv2d_135", (1, 1), (0, 0), 1, false)?, "batch_norm2d_150")?)?;
        let x = hd.squeeze(2)?.transpose(1, 2)?.contiguous()?;
        let y = self.linear(&x, "linear_8")?;
        let _ = self.n_class;
        softmax(&y, D::Minus1)
    }
}


/// CRNN-path crops are 1x64xW grayscale; SVTR wants 3x48xW **BGR**, normalised
/// (x/255 - 0.5)/0.5, height fixed at 48 with the width scaled by aspect
/// (PaddleOCR's `RecResizeImg`, image_shape [3,48,320]).
///
/// The model's own input signature is `[-1, 3, 48, -1]` — width is dynamic — so
/// a single crop is sized to its true aspect instead of being padded to 320.
/// That is what PaddleOCR does for a one-image batch (it sets `imgW` from the
/// batch's max width/height ratio), and it is also what the §8.165 ceiling probe
/// measured, so the accuracy result carries over.
///
/// BGR is not cosmetic: the checkpoint was trained on cv2-decoded images, and
/// feeding RGB silently swaps two channels of every crop.
pub fn svtr_input(
    img: &ffai_core::types::ImageBuffer,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    device: &Device,
) -> Result<Tensor> {
    const H: usize = 48;
    let (iw, ih) = (img.width as usize, img.height as usize);
    let (x1, y1) = (x1.min(iw), y1.min(ih));
    if x1 <= x0 + 1 || y1 <= y0 + 1 {
        return Err(candle_core::Error::Msg("degenerate crop".into()));
    }
    let (cw, ch) = (x1 - x0, y1 - y0);
    let w = (((H as f32) * cw as f32 / ch as f32).ceil() as usize).clamp(8, 2048);
    let stride = match img.format {
        ffai_core::types::PixelFormat::Rgb8 => 3,
        ffai_core::types::PixelFormat::Rgba8 => 4,
        ffai_core::types::PixelFormat::Gray8 => 1,
    };
    // planar BGR, bilinear sample from the crop
    let mut planes = vec![0f32; 3 * H * w];
    for oy in 0..H {
        let sy = ((oy as f32 + 0.5) * ch as f32 / H as f32 - 0.5).max(0.0);
        let (y0i, fy) = (sy.floor() as usize, sy - sy.floor());
        let y1i = (y0i + 1).min(ch - 1);
        for ox in 0..w {
            let sx = ((ox as f32 + 0.5) * cw as f32 / w as f32 - 0.5).max(0.0);
            let (x0i, fx) = (sx.floor() as usize, sx - sx.floor());
            let x1i = (x0i + 1).min(cw - 1);
            let at = |yy: usize, xx: usize, c: usize| -> f32 {
                let i = ((y0 + yy) * iw + (x0 + xx)) * stride;
                if stride == 1 { img.data[i] as f32 } else { img.data[i + c] as f32 }
            };
            for c in 0..3 {
                // source channel: BGR output from RGB input
                let sc = if stride == 1 { 0 } else { 2 - c };
                let top = at(y0i, x0i, sc) * (1.0 - fx) + at(y0i, x1i, sc) * fx;
                let bot = at(y1i, x0i, sc) * (1.0 - fx) + at(y1i, x1i, sc) * fx;
                let v = top * (1.0 - fy) + bot * fy;
                planes[c * H * w + oy * w + ox] = (v / 255.0 - 0.5) / 0.5;
            }
        }
    }
    // R.1 ORACLE HAND-OFF. `FFAI_SVTR_DUMP=<dir>` writes the exact tensor this
    // recognizer is about to read, so the reference weights can be given
    // BYTE-IDENTICAL input under onnxruntime.
    //
    // Handing the reference our IMAGE instead would re-run its own
    // preprocessing and compare two different things — §48 learned that the
    // hard way, where a token-perfect ORT match validated the executor while a
    // white-padding bug sat untouched in the preprocessing the oracle never saw.
    //
    // Pair with `FFAI_REC_SERIAL=1`: lines are recognised in parallel above a
    // threshold, and a shared counter would not match the emitted line order.
    if let Ok(dir) = std::env::var("FFAI_SVTR_DUMP") {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::SeqCst);
        let _ = std::fs::create_dir_all(&dir);
        let mut b = Vec::with_capacity(8 + planes.len() * 4);
        b.extend_from_slice(&(w as u32).to_le_bytes());
        b.extend_from_slice(&(H as u32).to_le_bytes());
        for v in &planes {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{n:05}.bin")), b);
    }
    Tensor::from_vec(planes, (1, 3, H, w), device)
}

/// Greedy CTC over the head's label order: index 0 is the blank, 1..N are the
/// charset lines. Returns the string and the mean probability of the kept
/// (non-blank, non-repeat) steps, matching what `crnn::decode` reports.
pub fn ctc_greedy(probs: &Tensor, charset: &[String]) -> Result<(String, Option<f32>)> {
    let (_, t, n) = probs.dims3()?;
    let best = probs.argmax(D::Minus1)?.flatten_all()?.to_vec1::<u32>()?;
    let conf = probs.max(D::Minus1)?.flatten_all()?.to_vec1::<f32>()?;
    let mut out = String::new();
    let (mut sum, mut kept) = (0f32, 0usize);
    let mut prev = u32::MAX;
    for i in 0..t {
        let k = best[i];
        if k != 0 && k != prev {
            // class k -> charset[k - 1]; the blank occupies index 0
            if let Some(s) = charset.get(k as usize - 1) {
                out.push_str(s);
            }
            sum += conf[i];
            kept += 1;
        }
        prev = k;
    }
    let _ = n;
    Ok((out, if kept == 0 { None } else { Some(sum / kept as f32) }))
}

/// One emitted run of characters, with WHERE ON THE LINE it came from.
#[derive(Debug, Clone)]
pub struct CharSpan {
    /// byte offset of this run within the decoded string
    pub byte: usize,
    /// fraction along the crop, 0..1, of the timestep that emitted it
    pub x: f32,
}

/// `ctc_greedy`, plus the x-position of every emitted character run.
///
/// F.2 of the recognition audit. Splicing a formula's LaTeX into a line needs to
/// know WHERE in the decoded string a pixel span falls, and CTC already knows:
/// the decoder walks T timesteps left to right across the resized crop, so a
/// kept timestep IS an x-position. Timestep `i` of `t` sits at `(i + 0.5) / t`
/// along the crop, which the caller maps back to page pixels through the crop
/// rectangle it built.
///
/// No model change and no second forward pass — the information was always in
/// the loop, simply discarded. Kept as a SEPARATE function so `ctc_greedy` stays
/// byte-identical: F.2's gate is that the emitted text does not move, and the
/// cheapest way to pass a byte-identity gate is to not touch the code that
/// produces the bytes.
pub fn ctc_greedy_spans(
    probs: &Tensor,
    charset: &[String],
) -> Result<(String, Option<f32>, Vec<CharSpan>)> {
    let (_, t, _) = probs.dims3()?;
    let best = probs.argmax(D::Minus1)?.flatten_all()?.to_vec1::<u32>()?;
    let conf = probs.max(D::Minus1)?.flatten_all()?.to_vec1::<f32>()?;
    let mut out = String::new();
    let mut spans = Vec::new();
    let (mut sum, mut kept) = (0f32, 0usize);
    let mut prev = u32::MAX;
    for i in 0..t {
        let k = best[i];
        if k != 0 && k != prev {
            if let Some(s) = charset.get(k as usize - 1) {
                spans.push(CharSpan {
                    byte: out.len(),
                    x: (i as f32 + 0.5) / t as f32,
                });
                out.push_str(s);
            }
            sum += conf[i];
            kept += 1;
        }
        prev = k;
    }
    Ok((out, if kept == 0 { None } else { Some(sum / kept as f32) }, spans))
}

/// Charset in head order from the file `carmenta_svtr_prepare.py` writes.
/// Entries can be multi-character (CJK ligatures are single lines), so this is
/// `Vec<String>`, not `Vec<char>` — indexing chars here would desynchronise the
/// table from the head.
pub fn load_charset(path: &std::path::Path) -> Result<Vec<String>> {
    let txt = std::fs::read_to_string(path)
        .map_err(|e| candle_core::Error::Msg(format!("charset {}: {e}", path.display())))?;
    Ok(txt.lines().map(|s| s.to_string()).collect())
}

use candle_core::IndexOp;

/// Load from a safetensors file produced by `tools/carmenta_svtr_prepare.py`.
pub fn load(path: &std::path::Path, device: &Device, n_class: usize) -> Result<Svtr> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
    Svtr::new(vb, n_class)
}
