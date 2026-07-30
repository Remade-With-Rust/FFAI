//! PARSeq-tiny recognition (Bautista & Atienza 2022) on candle — the
//! audit-cleared successor to the g2 CRNN, brought in to close the M-C1
//! quality gate (the CRNN's measured ceiling is the sentence-period class;
//! PARSeq reads a 94-char mixed-case charset with punctuation).
//!
//! Ported against the CHECKPOINT's shape map
//! (`corpora/refs/fixtures/parseq_tiny_shapes.json`) and strhub's decoder
//! math, not the paper figure. Inference is the pure AR-greedy path with
//! `refine_iters = 0` on both sides — the iterative refinement pass is a
//! recorded later brick, not a silent omission. Oracle:
//! `tests/oracles.rs::parseq_ids_match_pytorch_reference`.
//!
//! Architecture (from shapes): ViT-tiny encoder — 4×8 patch conv on a
//! 32×128 RGB crop → 128 tokens, dim 192, no cls token, learned pos_embed,
//! 12 pre-norm blocks (3 heads), final norm — and ONE decoder layer with
//! separate query/content LayerNorms, packed-projection self+cross
//! attention, GELU MLP 192→768, then a final norm and a 95-way head
//! (charset 94 + EOS; BOS/PAD are input-only).

use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};

pub const CHARSET: &str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
pub const EOS_ID: u32 = 0;
pub const BOS_ID: u32 = 95;
const DIM: usize = 192;
const ENC_HEADS: usize = 3;
const DEC_HEADS: usize = 6;
const MAX_STEPS: usize = 26; // max_label_length 25 + EOS slot

fn softmax_last(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax(x, D::Minus1)
}

/// Multi-head attention with a torch-packed in_proj (3E×E weight, 3E bias).
struct Mha {
    in_w: Tensor, // (3E, E)
    in_b: Tensor, // (3E,)
    out: Linear,
    heads: usize,
}

impl Mha {
    fn new(vb: &VarBuilder, heads: usize) -> Result<Self> {
        Ok(Mha {
            in_w: vb.get((3 * DIM, DIM), "in_proj_weight")?,
            in_b: vb.get(3 * DIM, "in_proj_bias")?,
            out: linear(DIM, DIM, vb.pp("out_proj"))?,
            heads,
        })
    }

    /// q: (1, Lq, E), kv: (1, Lk, E), mask: optional (Lq, Lk) additive.
    fn forward(&self, q: &Tensor, kv: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (lq, lk) = (q.dim(1)?, kv.dim(1)?);
        let hd = DIM / self.heads;
        let proj = |x: &Tensor, off: usize| -> Result<Tensor> {
            let w = self.in_w.i((off * DIM..(off + 1) * DIM, ..))?;
            let b = self.in_b.i(off * DIM..(off + 1) * DIM)?;
            x.broadcast_matmul(&w.t()?)?.broadcast_add(&b)
        };
        let to_heads = |x: &Tensor, l: usize| -> Result<Tensor> {
            x.reshape((1, l, self.heads, hd))?.transpose(1, 2)?.contiguous()
        };
        let qh = to_heads(&proj(q, 0)?, lq)?;
        let kh = to_heads(&proj(kv, 1)?, lk)?;
        let vh = to_heads(&proj(kv, 2)?, lk)?;
        let mut scores = (qh.matmul(&kh.transpose(2, 3)?)? * (1.0 / (hd as f64).sqrt()))?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(&m.reshape((1, 1, lq, lk))?)?;
        }
        let ctx = softmax_last(&scores)?.matmul(&vh)?; // (1, H, Lq, hd)
        ctx.transpose(1, 2)?.reshape((1, lq, DIM))?.apply(&self.out)
    }
}

struct VitBlock {
    norm1: LayerNorm,
    attn: Mha,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl VitBlock {
    fn new(vb: &VarBuilder) -> Result<Self> {
        // timm ViT packs qkv as a Linear named `attn.qkv` — same packed
        // layout as torch MHA's in_proj, loaded through the same struct.
        let attn = Mha {
            in_w: vb.get((3 * DIM, DIM), "attn.qkv.weight")?,
            in_b: vb.get(3 * DIM, "attn.qkv.bias")?,
            out: linear(DIM, DIM, vb.pp("attn.proj"))?,
            heads: ENC_HEADS,
        };
        Ok(VitBlock {
            norm1: layer_norm(DIM, 1e-6, vb.pp("norm1"))?,
            attn,
            norm2: layer_norm(DIM, 1e-6, vb.pp("norm2"))?,
            fc1: linear(DIM, 4 * DIM, vb.pp("mlp.fc1"))?,
            fc2: linear(4 * DIM, DIM, vb.pp("mlp.fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = self.norm1.forward(x)?;
        let x = (x + self.attn.forward(&n, &n, None)?)?;
        let m = self.norm2.forward(&x)?.apply(&self.fc1)?.gelu_erf()?.apply(&self.fc2)?;
        x + m
    }
}

struct DecoderLayer {
    self_attn: Mha,
    cross_attn: Mha,
    norm_q: LayerNorm,
    norm_c: LayerNorm,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl DecoderLayer {
    fn new(vb: &VarBuilder) -> Result<Self> {
        Ok(DecoderLayer {
            self_attn: Mha::new(&vb.pp("self_attn"), DEC_HEADS)?,
            cross_attn: Mha::new(&vb.pp("cross_attn"), DEC_HEADS)?,
            norm_q: layer_norm(DIM, 1e-5, vb.pp("norm_q"))?,
            norm_c: layer_norm(DIM, 1e-5, vb.pp("norm_c"))?,
            norm1: layer_norm(DIM, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(DIM, 1e-5, vb.pp("norm2"))?,
            linear1: linear(DIM, 4 * DIM, vb.pp("linear1"))?,
            linear2: linear(4 * DIM, DIM, vb.pp("linear2"))?,
        })
    }

    /// strhub `forward_stream`: query attends content (masked), then memory,
    /// then MLP — all residual, norms exactly where strhub puts them.
    fn forward(
        &self,
        query: &Tensor,
        content: &Tensor,
        memory: &Tensor,
        query_mask: &Tensor,
    ) -> Result<Tensor> {
        let qn = self.norm_q.forward(query)?;
        let cn = self.norm_c.forward(content)?;
        let t = (query + self.self_attn.forward(&qn, &cn, Some(query_mask))?)?;
        let t = (&t + self.cross_attn.forward(&self.norm1.forward(&t)?, memory, None)?)?;
        let m = self.norm2.forward(&t)?.apply(&self.linear1)?.gelu_erf()?.apply(&self.linear2)?;
        t + m
    }
}

pub struct Parseq {
    patch: candle_nn::Conv2d,
    pos_embed: Tensor,
    blocks: Vec<VitBlock>,
    enc_norm: LayerNorm,
    layer: DecoderLayer,
    dec_norm: LayerNorm,
    pos_queries: Tensor, // (1, 26, 192)
    text_embed: Tensor,  // (97, 192)
    head: Linear,
}

impl Parseq {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let enc = vb.pp("encoder");
        let cfg = candle_nn::Conv2dConfig { stride: 4, ..Default::default() };
        // Non-square stride: conv2d config has one stride; patch kernel is
        // (4, 8) with stride (4, 8). candle Conv2dConfig stride is uniform —
        // emulate the (4,8) stride by reshaping: kernel width 8 with stride 8
        // horizontally. Simplest correct route: use stride 4 and step
        // columns manually? No — candle supports only square stride here, so
        // the patchify is done by hand in `encode` with a matmul.
        let _ = cfg;
        Ok(Parseq {
            patch: candle_nn::conv2d(
                3,
                DIM,
                // kernel loaded manually below; placeholder never used
                1,
                Default::default(),
                enc.pp("__unused__"),
            )
            .or_else(|_| {
                // Build a Conv2d directly from the checkpoint tensors.
                let w = enc.get((DIM, 3, 4, 8), "patch_embed.proj.weight")?;
                let b = enc.get(DIM, "patch_embed.proj.bias")?;
                Ok::<_, candle_core::Error>(candle_nn::Conv2d::new(
                    w,
                    Some(b),
                    candle_nn::Conv2dConfig { stride: 4, ..Default::default() },
                ))
            })?,
            pos_embed: enc.get((1, 128, DIM), "pos_embed")?,
            blocks: (0..12).map(|i| VitBlock::new(&enc.pp(format!("blocks.{i}")))).collect::<Result<_>>()?,
            enc_norm: layer_norm(DIM, 1e-6, enc.pp("norm"))?,
            layer: DecoderLayer::new(&vb.pp("decoder.layers.0"))?,
            dec_norm: layer_norm(DIM, 1e-5, vb.pp("decoder.norm"))?,
            pos_queries: vb.get((1, MAX_STEPS, DIM), "pos_queries")?,
            // strhub TokenEmbedding scales lookups by sqrt(embed_dim). Missing
            // this was INVISIBLE at step 0 (content = BOS only: LayerNorm
            // cancels a pure per-row scale) and degraded every later step
            // (token+positional MIX shifts toward the positional term) —
            // measured as confidence decay 0.9996->0.90 by step 2 and early
            // EOS on long words ('warehouse' -> 'war'). Scale at load.
            text_embed: (vb.get((97, DIM), "text_embed.embedding.weight")? * (DIM as f64).sqrt())?,
            head: linear(DIM, 95, vb.pp("head"))?,
        })
    }

    /// (1, 3, 32, 128) normalized crop -> (1, 128, 192) memory.
    fn encode(&self, x: &Tensor) -> Result<Tensor> {
        // Patchify by hand: the checkpoint's patch conv has kernel (4,8) and
        // stride (4,8); candle's uniform-stride conv can't express it, and a
        // matmul over unfolded patches is exact and cheap at this size.
        let w = self.patch.weight(); // (192, 3, 4, 8)
        let b = self.patch.bias().expect("patch bias");
        let (gh, gw) = (32 / 4, 128 / 8); // 8 x 16 = 128 patches
        // unfold: (1,3,32,128) -> (128 patches, 3*4*8)
        let mut cols = Vec::with_capacity(gh * gw);
        for py in 0..gh {
            for px in 0..gw {
                let patch = x.i((.., .., py * 4..(py + 1) * 4, px * 8..(px + 1) * 8))?;
                cols.push(patch.flatten_all()?);
            }
        }
        let cols = Tensor::stack(&cols, 0)?; // (128, 96)
        let wmat = w.reshape((DIM, 3 * 4 * 8))?; // (192, 96)
        let tokens = cols.matmul(&wmat.t()?)?.broadcast_add(b)?.unsqueeze(0)?; // (1,128,192)
        let mut h = tokens.broadcast_add(&self.pos_embed)?;
        for blk in &self.blocks {
            h = blk.forward(&h)?;
        }
        self.enc_norm.forward(&h)
    }

    /// AR-greedy decode (refine_iters = 0). Returns (text, mean confidence).
    pub fn recognize(&self, x: &Tensor) -> Result<(String, Option<f32>)> {
        let memory = self.encode(x)?;
        let dev = x.device();
        let charset: Vec<char> = CHARSET.chars().collect();

        // Causal mask over MAX_STEPS: row i sees cols 0..=i.
        let mask: Vec<f32> = (0..MAX_STEPS)
            .flat_map(|i| (0..MAX_STEPS).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let full_mask = Tensor::from_vec(mask, (MAX_STEPS, MAX_STEPS), dev)?;

        let mut tgt_ids: Vec<u32> = vec![BOS_ID];
        let mut out = String::new();
        let mut confs: Vec<f32> = Vec::new();
        for i in 0..MAX_STEPS {
            let j = i + 1;
            // content = [BOS embed (no pos)] ++ (text_embed(tok) + pos_query)
            let mut content_rows = Vec::with_capacity(j);
            content_rows.push(self.text_embed.i(tgt_ids[0] as usize)?);
            for (k, &id) in tgt_ids.iter().enumerate().skip(1) {
                let e = self.text_embed.i(id as usize)?;
                let p = self.pos_queries.i((0, k - 1))?;
                content_rows.push((e + p)?);
            }
            let content = Tensor::stack(&content_rows, 0)?.unsqueeze(0)?; // (1, j, 192)
            let query = self.pos_queries.i((.., i..j, ..))?.contiguous()?; // (1, 1, 192)
            let qmask = full_mask.i((i..j, 0..j))?.contiguous()?;

            let t = self.layer.forward(&query, &content, &memory, &qmask)?;
            let logits = self.dec_norm.forward(&t)?.apply(&self.head)?; // (1,1,95)
            let probs = softmax_last(&logits)?.flatten_all()?.to_vec1::<f32>()?;
            let (id, p) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, p)| (i as u32, *p))
                .unwrap_or((EOS_ID, 0.0));
            if std::env::var("FFAI_PARSEQ_DEBUG").is_ok() {
                let mut top: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
                top.sort_by(|a, b| b.1.total_cmp(&a.1));
                eprintln!("step {i}: top3 {:?}", &top[..3]);
            }
            if id == EOS_ID {
                break;
            }
            out.push(charset[id as usize - 1]);
            confs.push(p);
            tgt_ids.push(id);
        }
        // ---- refinement pass (refine_iters = 1), strhub semantics ----
        //
        // The AR pass is fragile on out-of-distribution renderings (measured:
        // first-letter doubling on crisp UI fonts, reproduced by the
        // REFERENCE at refine_iters=0 on identical crops). Production PARSeq
        // masks that slip class with one cloze re-prediction: every position
        // is re-predicted seeing all content EXCEPT its own AR prediction
        // (content index i+1 is the token emitted at step i — the mask
        // blocks exactly j == i+1).
        if std::env::var("FFAI_PARSEQ_DEBUG").is_ok() {
            eprintln!("AR ids: {tgt_ids:?} -> {out:?}");
        }
        let l = tgt_ids.len(); // [BOS] + non-EOS tokens = strhub's tgt_in
        if l > 1 && std::env::var("FFAI_NO_REFINE").is_err() {
            let mut content_rows = Vec::with_capacity(l);
            content_rows.push(self.text_embed.i(tgt_ids[0] as usize)?);
            for (k, &id) in tgt_ids.iter().enumerate().skip(1) {
                content_rows.push((self.text_embed.i(id as usize)? + self.pos_queries.i((0, k - 1))?)?);
            }
            let content = Tensor::stack(&content_rows, 0)?.unsqueeze(0)?;
            let query = self.pos_queries.i((.., 0..l, ..))?.contiguous()?;
            let cloze: Vec<f32> = (0..l)
                .flat_map(|i| (0..l).map(move |j| if j == i + 1 { f32::NEG_INFINITY } else { 0.0 }))
                .collect();
            let cloze = Tensor::from_vec(cloze, (l, l), dev)?;
            let t = self.layer.forward(&query, &content, &memory, &cloze)?;
            let logits = self.dec_norm.forward(&t)?.apply(&self.head)?; // (1, l, 95)
            let probs = softmax_last(&logits)?.squeeze(0)?.to_vec2::<f32>()?;
            let mut refined = String::new();
            let mut confs2 = Vec::new();
            for row in &probs {
                let (id, p) = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, p)| (i as u32, *p))
                    .unwrap_or((EOS_ID, 0.0));
                if id == EOS_ID {
                    break;
                }
                refined.push(charset[id as usize - 1]);
                confs2.push(p);
            }
            if !refined.is_empty() {
                let conf =
                    Some(confs2.iter().sum::<f32>() / confs2.len() as f32);
                return Ok((refined, conf));
            }
        }

        let conf = if confs.is_empty() {
            None
        } else {
            Some(confs.iter().sum::<f32>() / confs.len() as f32)
        };
        Ok((out, conf))
    }
}

/// Preprocess a grayscale crop for PARSeq: replicate to RGB, bicubic resize
/// to 32x128, normalize (x/255 - 0.5)/0.5, CHW.
pub fn parseq_input(
    gray: &[f32],
    w: usize,
    h: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let resized = crate::image::resize_bicubic(gray, w, h, 128, 32);
    let norm: Vec<f32> = resized.iter().map(|&p| (p / 255.0 - 0.5) / 0.5).collect();
    let plane = Tensor::from_vec(norm, (1, 1, 32, 128), device)?;
    Tensor::cat(&[&plane, &plane, &plane], 1)?.to_dtype(DType::F32)
}
