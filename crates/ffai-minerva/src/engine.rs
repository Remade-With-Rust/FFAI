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
//! 8-bit quantisation, the exact integer product, and `finish`.
//!
//! Every kernel goes through [`crate::simd`] at the session's [`Isa`], chosen once; every tier
//! is bit-identical to the scalar oracles, so the tier changes the speed and nothing else.

use rayon::prelude::*;

use crate::head;
use crate::kernel::gemv_par;
use crate::model::{Model, Ternary};
use crate::numerics::rope_table;
use crate::simd::{self, Isa};

/// Vocabulary rows a thread takes at a time in the head.
const HEAD_CHUNK: usize = 256;

/// Errors a session can hit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepError {
    /// The context is full.
    ContextFull(usize),
    /// A token id outside the vocabulary.
    BadToken(u32),
    /// `HeadMode::Verify` found the screen's token differing from the full head's (a defect).
    ScreenMismatch {
        /// The full head's token.
        full: u32,
        /// The screen's token.
        screen: u32,
    },
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ContextFull(n) => write!(f, "context of {n} tokens is full"),
            Self::BadToken(t) => write!(f, "token {t} is outside the vocabulary"),
            Self::ScreenMismatch { full, screen } => {
                write!(f, "head screen chose {screen}, the full head {full}")
            }
        }
    }
}

impl std::error::Error for StepError {}

/// One sequence's state: the KV cache and the scratch a step needs, allocated once.
pub struct Session<'m> {
    m: &'m Model,
    isa: Isa,
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
    cos: Vec<f32>,
    sin: Vec<f32>,
    logits: Vec<f32>,
    screen: head::Scratch,
    /// Screened steps, rows recomputed exactly in total, and the most in one step.
    pub screen_stats: (u64, u64, usize),
}

/// How a decode step picks its token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadMode {
    /// The full `f16` head: every logit (what `--probs` and the conformance harness need).
    Full,
    /// The `int8` screen plus the survivors' exact logits ([`crate::head`]): the same token.
    Screen,
    /// Both, and an error if they ever disagree.
    Verify,
}

impl HeadMode {
    /// `MINERVA_HEAD=full|screen|verify`, else [`HeadMode::Screen`].
    ///
    /// # Panics
    /// On any other value.
    #[must_use]
    pub fn from_env() -> Self {
        match std::env::var("MINERVA_HEAD").as_deref() {
            Ok("full") => Self::Full,
            Ok("verify") => Self::Verify,
            Ok("screen") | Err(_) => Self::Screen,
            Ok(o) => panic!("MINERVA_HEAD={o}: want full, screen or verify"),
        }
    }
}

impl std::fmt::Debug for Session<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("isa", &self.isa)
            .field("ctx", &self.ctx)
            .field("pos", &self.pos)
            .finish_non_exhaustive()
    }
}

/// `out = finish(W · a)` for one projection, the activation already quantised (`qa`, its scale
/// and its sum).
fn bitlinear(
    isa: Isa,
    m: &Model,
    t: &Ternary,
    qa: &[i8],
    act: (f32, i32),
    isum: &mut [i32],
    out: &mut [f32],
) {
    let isum = &mut isum[..t.rows];
    gemv_par(isa, m.codes(t), &qa[..t.cols], act.1, isum);
    simd::finish_rows(isa, isum, act.0, t.scale, &mut out[..t.rows]);
}

/// Quantise `x` into `qa`: the scale and the codes' sum.
fn quantise(isa: Isa, x: &[f32], qa: &mut [i8]) -> (f32, i32) {
    let qa = &mut qa[..x.len()];
    let s = simd::act_quant(isa, x, qa);
    (s, simd::act_sum(qa))
}

impl<'m> Session<'m> {
    /// A session over `m` holding up to `ctx` positions, at [`Isa::from_env`].
    #[must_use]
    pub fn new(m: &'m Model, ctx: usize) -> Self {
        Self::with_isa(m, ctx, Isa::from_env())
    }

    /// A session at a chosen instruction set.
    #[must_use]
    pub fn with_isa(m: &'m Model, ctx: usize, isa: Isa) -> Self {
        let c = &m.cfg;
        let kv = c.layers * ctx * c.kv_dim();
        let q_dim = c.heads * c.head_dim;
        let widest = c.embd.max(q_dim).max(c.ffn);
        Self {
            m,
            isa,
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
            cos: vec![0.0; c.head_dim / 2],
            sin: vec![0.0; c.head_dim / 2],
            logits: vec![0.0; c.vocab],
            screen: head::Scratch::new(c.vocab, c.embd),
            screen_stats: (0, 0, 0),
        }
    }

    /// The instruction set this session runs.
    #[must_use]
    pub const fn isa(&self) -> Isa {
        self.isa
    }

    /// Tokens consumed so far.
    #[must_use]
    pub const fn pos(&self) -> usize {
        self.pos
    }

    /// Feed one token. With `logits`, compute and return the next-token logits (the full head).
    ///
    /// # Errors
    /// When the context is full or the token is outside the vocabulary.
    pub fn step(&mut self, token: u32, logits: bool) -> Result<Option<&[f32]>, StepError> {
        self.forward(token)?;
        if !logits {
            return Ok(None);
        }
        self.final_hidden();
        self.head_full();
        Ok(Some(&self.logits))
    }

    /// Feed one token and return the greedy next token through the head screen (the full head's
    /// token, by proof: [`crate::head`]).
    ///
    /// # Errors
    /// When the context is full or the token is outside the vocabulary.
    pub fn step_argmax(&mut self, token: u32) -> Result<u32, StepError> {
        self.forward(token)?;
        self.final_hidden();
        Ok(self.screen_argmax())
    }

    fn screen_argmax(&mut self) -> u32 {
        let m = self.m;
        let h8 = m.screen.get(m);
        let embd = m.cfg.embd;
        if let Some((t, n)) = head::argmax(self.isa, m, h8, &self.h[..embd], &mut self.screen) {
            self.screen_stats.0 += 1;
            self.screen_stats.1 += n as u64;
            self.screen_stats.2 = self.screen_stats.2.max(n);
            t
        } else {
            self.head_full();
            argmax(&self.logits)
        }
    }

    /// The final norm, rounded to `f16` as the reference does before the head, into `h`.
    fn final_hidden(&mut self) {
        let (isa, m) = (self.isa, self.m);
        let embd = m.cfg.embd;
        simd::rms_norm(
            isa,
            &self.x,
            &m.output_norm,
            m.cfg.rms_eps,
            &mut self.h[..embd],
        );
        simd::round_f16_slice(isa, &mut self.h[..embd]);
    }

    /// Every logit from the `f16` head.
    fn head_full(&mut self) {
        let (isa, m) = (self.isa, self.m);
        let embd = m.cfg.embd;
        let h = &self.h[..embd];
        let head = m.head_bytes();
        let row = embd * 2;
        self.logits
            .par_chunks_mut(HEAD_CHUNK)
            .zip(head.par_chunks(HEAD_CHUNK * row))
            .for_each(|(ls, rows)| {
                simd::f16_matvec_bytes(isa, rows, h, ls);
            });
    }

    /// The blocks: one token into the KV cache and the residual stream `x`.
    fn forward(&mut self, token: u32) -> Result<(), StepError> {
        let m = self.m;
        let isa = self.isa;
        let c = &m.cfg;
        if self.pos >= self.ctx {
            return Err(StepError::ContextFull(self.ctx));
        }
        let tok = usize::try_from(token)
            .ok()
            .filter(|&t| t < c.vocab)
            .ok_or(StepError::BadToken(token))?;
        let (embd, kv_dim, hd, ffn) = (c.embd, c.kv_dim(), c.head_dim, c.ffn);
        let q_dim = c.heads * hd;
        let group = c.heads / c.kv_heads;
        let pos = self.pos;
        #[expect(clippy::cast_precision_loss, reason = "head width, small")]
        let kq_scale = 1.0 / (hd as f32).sqrt();

        simd::f16_bytes_to_f32(isa, m.f16_row(m.embd, embd, tok), &mut self.x);
        rope_table(pos, hd, c.rope_base, &mut self.cos, &mut self.sin);

        for (li, l) in m.layers.iter().enumerate() {
            // --- attention ---
            simd::rms_norm(isa, &self.x, &l.attn_norm, c.rms_eps, &mut self.h[..embd]);
            let act = quantise(isa, &self.h[..embd], &mut self.qa);
            bitlinear(isa, m, &l.q, &self.qa, act, &mut self.isum, &mut self.q);
            bitlinear(isa, m, &l.k, &self.qa, act, &mut self.isum, &mut self.kc);
            bitlinear(isa, m, &l.v, &self.qa, act, &mut self.isum, &mut self.vc);
            for head in self.q.chunks_exact_mut(hd) {
                simd::rope_neox(isa, head, &self.cos, &self.sin);
            }
            for head in self.kc.chunks_exact_mut(hd) {
                simd::rope_neox(isa, head, &self.cos, &self.sin);
            }
            let layer_base = li * self.ctx * kv_dim;
            let at = layer_base + pos * kv_dim;
            simd::f32_to_f16(isa, &self.kc, &mut self.k[at..at + kv_dim]);
            simd::f32_to_f16(isa, &self.vc, &mut self.v[at..at + kv_dim]);
            simd::round_f16_slice(isa, &mut self.q);
            let n = pos + 1;
            let kcache = &self.k[layer_base..layer_base + n * kv_dim];
            let vcache = &self.v[layer_base..layer_base + n * kv_dim];
            let q = &self.q;
            let ctx = self.ctx;
            self.att
                .par_chunks_mut(hd)
                .zip(self.scores.par_chunks_mut(ctx))
                .enumerate()
                .for_each(|(head, (out, sc))| {
                    let g = head / group;
                    let sc = &mut sc[..n];
                    simd::f16_matvec_strided(
                        isa,
                        kcache,
                        g * hd,
                        kv_dim,
                        &q[head * hd..(head + 1) * hd],
                        sc,
                    );
                    simd::softmax(isa, sc, kq_scale);
                    simd::round_f16_slice(isa, sc);
                    simd::weighted_sum_f16(isa, sc, vcache, g * hd, kv_dim, out);
                });
            simd::rms_norm(
                isa,
                &self.att,
                &l.attn_sub_norm,
                c.rms_eps,
                &mut self.h[..q_dim],
            );
            let act = quantise(isa, &self.h[..q_dim], &mut self.qa);
            bitlinear(isa, m, &l.o, &self.qa, act, &mut self.isum, &mut self.out);
            simd::add_assign(isa, &mut self.x, &self.out[..embd]);

            // --- feed-forward ---
            simd::rms_norm(isa, &self.x, &l.ffn_norm, c.rms_eps, &mut self.h[..embd]);
            let act = quantise(isa, &self.h[..embd], &mut self.qa);
            bitlinear(
                isa,
                m,
                &l.gate,
                &self.qa,
                act,
                &mut self.isum,
                &mut self.gate,
            );
            bitlinear(isa, m, &l.up, &self.qa, act, &mut self.isum, &mut self.up);
            simd::relu2_mul(isa, &mut self.gate, &self.up);
            simd::rms_norm(
                isa,
                &self.gate,
                &l.ffn_sub_norm,
                c.rms_eps,
                &mut self.h[..ffn],
            );
            let act = quantise(isa, &self.h[..ffn], &mut self.qa);
            bitlinear(
                isa,
                m,
                &l.down,
                &self.qa,
                act,
                &mut self.isum,
                &mut self.out,
            );
            simd::add_assign(isa, &mut self.x, &self.out[..embd]);
        }
        self.pos += 1;
        Ok(())
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
    let z: f64 = logits
        .iter()
        .map(|&l| (f64::from(l) - f64::from(max)).exp())
        .sum();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let k = k.min(idx.len());
    idx.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
        logits[b].total_cmp(&logits[a]).then(a.cmp(&b))
    });
    idx.truncate(k);
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    idx.into_iter()
        .map(|i| {
            (
                u32::try_from(i).unwrap_or(u32::MAX),
                (f64::from(logits[i]) - f64::from(max)).exp() / z,
            )
        })
        .collect()
}

/// Prefill `prompt`, then decode `new` tokens greedily. EOS is not a stop (matched token
/// counts against the reference). `on_step` sees each generated step's logits (with
/// [`HeadMode::Full`] and [`HeadMode::Verify`]; `None` with the screen) and its token.
///
/// # Errors
/// When the context overflows, a token is out of range, or [`HeadMode::Verify`] finds a mismatch.
pub fn generate(
    s: &mut Session<'_>,
    prompt: &[u32],
    new: usize,
    mode: HeadMode,
    mut on_step: impl FnMut(usize, Option<&[f32]>, u32),
) -> Result<Vec<u32>, StepError> {
    let (last, head) = prompt.split_last().ok_or(StepError::BadToken(u32::MAX))?;
    for &t in head {
        s.step(t, false)?;
    }
    let mut out = Vec::with_capacity(new);
    let mut next = *last;
    for i in 0..new {
        next = match mode {
            HeadMode::Full => {
                let logits = s.step(next, true)?.unwrap_or_default();
                let t = argmax(logits);
                on_step(i, Some(logits), t);
                t
            }
            HeadMode::Screen => {
                let t = s.step_argmax(next)?;
                on_step(i, None, t);
                t
            }
            HeadMode::Verify => {
                let screen = s.step_argmax(next)?;
                s.head_full();
                let full = argmax(&s.logits);
                if full != screen {
                    return Err(StepError::ScreenMismatch { full, screen });
                }
                on_step(i, Some(&s.logits), full);
                full
            }
        };
        out.push(next);
    }
    Ok(out)
}
