//! The forward pass: one token at a time through the `BitNet` b1.58 graph, as bitnet.cpp's
//! `build_bitnet_158` (pinned `765741d`) builds it, with a KV cache and greedy decoding.
//!
//! Per block:
//!
//! ```text
//! h  = rms_norm(x, attn_norm)
//! q, k, v = BitLinear(h)                 (one activation quantisation for all three)
//! q, k = RoPE_neox(q), RoPE_neox(k)
//! a  = attention(q, K, V)                (K, V cached as f16; q and the weights rounded to f16)
//! x += BitLinear_o(rms_norm(a, attn_sub_norm))
//! h  = rms_norm(x, ffn_norm)
//! f  = relu(BitLinear_gate(h))^2 * BitLinear_up(h)
//! x += BitLinear_down(rms_norm(f, ffn_sub_norm))
//! ```
//!
//! then `logits = head · f16(rms_norm(x, output_norm))`. A `BitLinear` is the activation's
//! 8-bit quantisation (`rusty_ternary::quant::act_quant`), the exact integer product, and
//! `rusty_ternary::kernel::finish`.

use rayon::prelude::*;
use rusty_ternary::kernel::finish;
use rusty_ternary::quant::act_quant;

use crate::kernel::gemv_par;
use crate::model::{Model, Ternary};
use crate::numerics::{
    dot16_f16, dot16_f16_bytes, f16_table, rms_norm, rope_neox, rope_table, round_f16, softmax, to_f16_bits,
};

/// Vocabulary rows a thread takes at a time in the head.
const HEAD_CHUNK: usize = 256;

/// Errors a session can hit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepError {
    /// The context is full.
    ContextFull(usize),
    /// A token id outside the vocabulary.
    BadToken(u32),
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ContextFull(n) => write!(f, "context of {n} tokens is full"),
            Self::BadToken(t) => write!(f, "token {t} is outside the vocabulary"),
        }
    }
}

impl std::error::Error for StepError {}

/// One sequence's state: the KV cache and the scratch a step needs, allocated once.
pub struct Session<'m> {
    m: &'m Model,
    ctx: usize,
    pos: usize,
    /// `[layer][position][kv_dim]` as `f16` bits.
    k: Vec<u16>,
    v: Vec<u16>,
    x: Vec<f32>,
    h: Vec<f32>,
    q: Vec<f32>,
    kc: Vec<f32>,
    vc: Vec<f32>,
    att: Vec<f32>,
    scores: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    out: Vec<f32>,
    qa: Vec<i8>,
    isum: Vec<i32>,
    rope: Vec<(f32, f32)>,
    logits: Vec<f32>,
}

impl std::fmt::Debug for Session<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").field("ctx", &self.ctx).field("pos", &self.pos).finish_non_exhaustive()
    }
}

/// `out = finish(W · quant(input))` for one projection, the activation already quantised.
fn bitlinear(m: &Model, t: &Ternary, qa: &[i8], act_scale: f32, isum: &mut [i32], out: &mut [f32]) {
    let isum = &mut isum[..t.rows];
    gemv_par(m.codes(t), &qa[..t.cols], isum);
    for (o, &s) in out[..t.rows].iter_mut().zip(isum.iter()) {
        *o = finish(s, act_scale, t.scale);
    }
}

impl<'m> Session<'m> {
    /// A session over `m` holding up to `ctx` positions.
    #[must_use]
    pub fn new(m: &'m Model, ctx: usize) -> Self {
        let c = &m.cfg;
        let kv = c.layers * ctx * c.kv_dim();
        let q_dim = c.heads * c.head_dim;
        let widest = c.embd.max(q_dim).max(c.ffn);
        Self {
            m,
            ctx,
            pos: 0,
            k: vec![0; kv],
            v: vec![0; kv],
            x: vec![0.0; c.embd],
            h: vec![0.0; widest],
            q: vec![0.0; q_dim],
            kc: vec![0.0; c.kv_dim()],
            vc: vec![0.0; c.kv_dim()],
            att: vec![0.0; q_dim],
            scores: vec![0.0; c.heads * ctx],
            gate: vec![0.0; c.ffn],
            up: vec![0.0; c.ffn],
            out: vec![0.0; widest],
            qa: vec![0; widest],
            isum: vec![0; widest],
            rope: vec![(0.0, 0.0); c.head_dim / 2],
            logits: vec![0.0; c.vocab],
        }
    }

    /// Tokens consumed so far.
    #[must_use]
    pub const fn pos(&self) -> usize {
        self.pos
    }

    /// Feed one token. With `logits`, compute and return the next-token logits.
    ///
    /// # Errors
    /// When the context is full or the token is outside the vocabulary.
    pub fn step(&mut self, token: u32, logits: bool) -> Result<Option<&[f32]>, StepError> {
        let m = self.m;
        let c = &m.cfg;
        if self.pos >= self.ctx {
            return Err(StepError::ContextFull(self.ctx));
        }
        let tok = usize::try_from(token).ok().filter(|&t| t < c.vocab).ok_or(StepError::BadToken(token))?;
        let (embd, kv_dim, hd, ffn) = (c.embd, c.kv_dim(), c.head_dim, c.ffn);
        let q_dim = c.heads * hd;
        let group = c.heads / c.kv_heads;
        let pos = self.pos;
        #[expect(clippy::cast_precision_loss, reason = "head width, small")]
        let kq_scale = 1.0 / (hd as f32).sqrt();

        let t16 = f16_table();
        for (x, b) in self.x.iter_mut().zip(m.f16_row(m.embd, embd, tok).chunks_exact(2)) {
            *x = t16[usize::from(u16::from_le_bytes([b[0], b[1]]))];
        }
        rope_table(pos, hd, c.rope_base, &mut self.rope);

        for (li, l) in m.layers.iter().enumerate() {
            // --- attention ---
            rms_norm(&self.x, &l.attn_norm, c.rms_eps, &mut self.h[..embd]);
            let s = act_quant(&self.h[..embd], &mut self.qa[..embd]);
            bitlinear(m, &l.q, &self.qa, s, &mut self.isum, &mut self.q);
            bitlinear(m, &l.k, &self.qa, s, &mut self.isum, &mut self.kc);
            bitlinear(m, &l.v, &self.qa, s, &mut self.isum, &mut self.vc);
            for head in self.q.chunks_exact_mut(hd) {
                rope_neox(head, &self.rope);
            }
            for head in self.kc.chunks_exact_mut(hd) {
                rope_neox(head, &self.rope);
            }
            let layer_base = li * self.ctx * kv_dim;
            let at = layer_base + pos * kv_dim;
            for (d, &v) in self.k[at..at + kv_dim].iter_mut().zip(&self.kc) {
                *d = to_f16_bits(v);
            }
            for (d, &v) in self.v[at..at + kv_dim].iter_mut().zip(&self.vc) {
                *d = to_f16_bits(v);
            }
            for v in &mut self.q {
                *v = round_f16(*v);
            }
            let n = pos + 1;
            let (kcache, vcache) = (&self.k[layer_base..layer_base + n * kv_dim], &self.v[layer_base..at + kv_dim]);
            let q = &self.q;
            let ctx = self.ctx;
            self.att.par_chunks_mut(hd).zip(self.scores.par_chunks_mut(ctx)).enumerate().for_each(
                |(head, (out, sc))| {
                    let g = head / group;
                    let qh = &q[head * hd..(head + 1) * hd];
                    let sc = &mut sc[..n];
                    for (t, s) in sc.iter_mut().enumerate() {
                        let kt = &kcache[t * kv_dim + g * hd..t * kv_dim + (g + 1) * hd];
                        *s = dot16_f16(kt, qh) * kq_scale;
                    }
                    softmax(sc);
                    out.fill(0.0);
                    for (t, &p) in sc.iter().enumerate() {
                        let p = round_f16(p);
                        let vt = &vcache[t * kv_dim + g * hd..t * kv_dim + (g + 1) * hd];
                        for (o, &v) in out.iter_mut().zip(vt) {
                            *o += p * t16[usize::from(v)];
                        }
                    }
                },
            );
            rms_norm(&self.att, &l.attn_sub_norm, c.rms_eps, &mut self.h[..q_dim]);
            let s = act_quant(&self.h[..q_dim], &mut self.qa[..q_dim]);
            bitlinear(m, &l.o, &self.qa, s, &mut self.isum, &mut self.out);
            for (x, &o) in self.x.iter_mut().zip(&self.out[..embd]) {
                *x += o;
            }

            // --- feed-forward ---
            rms_norm(&self.x, &l.ffn_norm, c.rms_eps, &mut self.h[..embd]);
            let s = act_quant(&self.h[..embd], &mut self.qa[..embd]);
            bitlinear(m, &l.gate, &self.qa, s, &mut self.isum, &mut self.gate);
            bitlinear(m, &l.up, &self.qa, s, &mut self.isum, &mut self.up);
            for (g, &u) in self.gate.iter_mut().zip(&self.up) {
                let r = if *g > 0.0 { *g } else { 0.0 };
                *g = r * r * u;
            }
            rms_norm(&self.gate, &l.ffn_sub_norm, c.rms_eps, &mut self.h[..ffn]);
            let s = act_quant(&self.h[..ffn], &mut self.qa[..ffn]);
            bitlinear(m, &l.down, &self.qa, s, &mut self.isum, &mut self.out);
            for (x, &o) in self.x.iter_mut().zip(&self.out[..embd]) {
                *x += o;
            }
        }
        self.pos += 1;
        if !logits {
            return Ok(None);
        }
        rms_norm(&self.x, &m.output_norm, c.rms_eps, &mut self.h[..embd]);
        for v in &mut self.h[..embd] {
            *v = round_f16(*v);
        }
        let h = &self.h[..embd];
        let head = m.head_bytes();
        let row = embd * 2;
        self.logits.par_chunks_mut(HEAD_CHUNK).enumerate().for_each(|(ci, ls)| {
            let base = ci * HEAD_CHUNK;
            for (j, l) in ls.iter_mut().enumerate() {
                let r = base + j;
                *l = dot16_f16_bytes(&head[r * row..(r + 1) * row], h);
            }
        });
        Ok(Some(&self.logits))
    }
}

/// The greedy choice: the largest logit, the lowest id on an exact tie.
#[must_use]
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    u32::try_from(best).unwrap_or(u32::MAX)
}

/// The `k` likeliest tokens with their probabilities (softmax in `f64`, for reporting only).
#[must_use]
pub fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f64)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let z: f64 = logits.iter().map(|&l| (f64::from(l) - f64::from(max)).exp()).sum();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let k = k.min(idx.len());
    idx.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    idx.truncate(k);
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    idx.into_iter()
        .map(|i| (u32::try_from(i).unwrap_or(u32::MAX), (f64::from(logits[i]) - f64::from(max)).exp() / z))
        .collect()
}

/// Prefill `prompt`, then decode `new` tokens greedily. EOS is not a stop (matched token
/// counts against the reference). `on_step` sees each generated step's logits.
///
/// # Errors
/// When the context overflows or a token is out of range.
pub fn generate(
    s: &mut Session<'_>,
    prompt: &[u32],
    new: usize,
    mut on_step: impl FnMut(usize, &[f32]),
) -> Result<Vec<u32>, StepError> {
    let (last, head) = prompt.split_last().ok_or(StepError::BadToken(u32::MAX))?;
    for &t in head {
        s.step(t, false)?;
    }
    let mut out = Vec::with_capacity(new);
    let mut next = *last;
    for i in 0..new {
        let logits = s.step(next, true)?.unwrap_or_default();
        on_step(i, logits);
        next = argmax(logits);
        out.push(next);
    }
    Ok(out)
}
