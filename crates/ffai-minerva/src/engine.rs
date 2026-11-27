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
use crate::kernel::{gemm_par, gemv_par};
use crate::lora::{self, Adapter, Proj};
use crate::model::{Model, Ternary};
use crate::numerics::rope_table;
use crate::simd::{self, Isa};

/// Tokens a thread finishes at a time after a prefill GEMM (16 i32 = one cache line of a row).
const FINISH_BLOCK: usize = 16;

/// Prompt tokens a prefill batch holds: the weights stream once a batch, not once a token.
const PREFILL_BATCH: usize = 128;

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
    /// Grouped decode attention's staging: scores by (KV head, position chunk), values by
    /// (KV head, output slice).
    att_tmp: Vec<f32>,
    att_otmp: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    out: Vec<f32>,
    qa: Vec<i8>,
    isum: Vec<i32>,
    isum2: Vec<i32>,
    isum3: Vec<i32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    logits: Vec<f32>,
    screen: head::Scratch,
    /// The LoRA adapter on top of the base, if any ([`Self::with_adapter`]).
    lora: Option<&'m Adapter>,
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

/// A batch's projection: `out[t·stride ..][..rows] = finish(W · a_t)` for every token `t`, the
/// activations already quantised (`qa`, `bm` tokens of `t.cols`; their scales and sums in `acts`).
#[expect(
    clippy::too_many_arguments,
    reason = "the batch's buffers, passed explicitly"
)]
fn bitlinear_batch(
    isa: Isa,
    m: &Model,
    t: &Ternary,
    qa: &[i8],
    acts: &[(f32, i32)],
    ysum: &mut [i32],
    out: &mut [f32],
    stride: usize,
) {
    let bm = acts.len();
    let sums: Vec<i32> = acts.iter().map(|a| a.1).collect();
    let ysum = &mut ysum[..t.rows * bm];
    let g = crate::prof::scope(crate::prof::Stage::Gemm);
    gemm_par(isa, m.codes(t), t.cols, bm, &qa[..bm * t.cols], &sums, ysum);
    drop(g);
    let _f = crate::prof::scope(crate::prof::Stage::Finish);
    // The `[rows, tokens]` sums to each token's floats: in parallel over blocks of 16 tokens, so
    // a 64-byte line of `ysum` (one row's 16 tokens) is read once, with `finish` fused in (the
    // scalar oracle, which the `finish_rows` twins equal bit for bit).
    let (rows, scale) = (t.rows, t.scale);
    let ysum = &*ysum;
    out[..bm * stride]
        .par_chunks_mut(FINISH_BLOCK * stride)
        .zip(acts.par_chunks(FINISH_BLOCK))
        .enumerate()
        .for_each(|(blk, (o, acts))| {
            let k0 = blk * FINISH_BLOCK;
            for r in 0..rows {
                let line = &ysum[r * bm + k0..r * bm + k0 + acts.len()];
                for (j, (&s, act)) in line.iter().zip(acts).enumerate() {
                    o[j * stride + r] = rusty_ternary::kernel::finish(s, act.0, scale);
                }
            }
        });
}

/// Positions a grouped decode scores task covers.
const DEC_CHUNK: usize = 128;

/// Output values a grouped decode values task covers (64 bytes of `f16` a row).
const DEC_SLICE: usize = 32;

/// Positions from which decode takes the grouped attention by default (x86 only).
const DEC_GROUP_MIN: usize = 1024;

/// Whether decode takes the grouped attention (it needs groups of 4 and head widths of a
/// multiple of [`DEC_SLICE`]): `MINERVA_DECATT=group` always, `head` never, unset on x86 from
/// [`DEC_GROUP_MIN`] positions. Laptop, 8 threads, under load (median 67%): attention 1.28x at
/// 1,856 positions (7/8 rounds, null 0.98) and 1.27x at 1,024 (null 1.10), but 0.70x at 256 and
/// 0.67x at 64, where three phases cost more than the reads they save. aarch64 waits for its
/// grouped kernels (today it would loop per head).
fn dec_grouped(n: usize, group: usize, hd: usize) -> bool {
    static MODE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let mode = *MODE.get_or_init(|| match std::env::var("MINERVA_DECATT").as_deref() {
        Ok("group") => Some(true),
        Ok("head") => Some(false),
        _ => None,
    });
    let on = mode.unwrap_or(cfg!(target_arch = "x86_64") && n >= DEC_GROUP_MIN);
    on && n > 0 && group == 4 && hd.is_multiple_of(DEC_SLICE)
}

/// One decode step's attention, each KV head's `K` and `V` read once for its whole group (the
/// per-head path reads them once a query head). Three phases, each with enough tasks to fill
/// the threads: scores by (KV head, chunk of positions) (each score is independent of the
/// others), softmax and `f16` rounding by query head, values by (KV head, slice of outputs)
/// (each output still sums over positions in order from zero). The same bits as the per-head
/// path: the grouped kernels equal the per-head ones.
#[expect(
    clippy::too_many_arguments,
    reason = "the attention's buffers, passed explicitly"
)]
fn attend_decode_grouped(
    isa: Isa,
    kcache: &[u16],
    vcache: &[u16],
    q: &[f32],
    n: usize,
    (kv_heads, group, hd, kv_dim, ctx): (usize, usize, usize, usize, usize),
    kq_scale: f32,
    (scores, tmp, otmp): (&mut [f32], &mut [f32], &mut [f32]),
    att: &mut [f32],
) {
    // Scores: KV head `g`'s region of `tmp` holds its chunks back to back, chunk `c` as `group`
    // runs of its length.
    tmp[..kv_heads * group * ctx]
        .par_chunks_mut(group * ctx)
        .enumerate()
        .for_each(|(g, reg)| {
            let qg = &q[g * group * hd..(g + 1) * group * hd];
            reg[..group * n]
                .par_chunks_mut(group * DEC_CHUNK)
                .enumerate()
                .for_each(|(c, o)| {
                    let len = o.len() / group;
                    let t0 = c * DEC_CHUNK;
                    simd::f16_matvec_strided_group(
                        isa,
                        kcache,
                        g * hd + t0 * kv_dim,
                        kv_dim,
                        hd,
                        qg,
                        len,
                        o,
                        len,
                    );
                });
        });
    // Softmax and rounding, a query head a task, after gathering its row.
    let tmp = &*tmp;
    scores[..kv_heads * group * ctx]
        .par_chunks_mut(ctx)
        .enumerate()
        .for_each(|(h, row)| {
            let (g, j) = (h / group, h % group);
            let reg = &tmp[g * group * ctx..];
            for t0 in (0..n).step_by(DEC_CHUNK) {
                let len = DEC_CHUNK.min(n - t0);
                let src = &reg[t0 * group + j * len..t0 * group + (j + 1) * len];
                row[t0..t0 + len].copy_from_slice(src);
            }
            let row = &mut row[..n];
            simd::softmax(isa, row, kq_scale);
            simd::round_f16_slice(isa, row);
        });
    // Values: (KV head, slice of `DEC_SLICE` outputs) a task, staged then placed.
    let scores = &*scores;
    let slices = hd / DEC_SLICE;
    otmp[..kv_heads * group * hd]
        .par_chunks_mut(group * DEC_SLICE)
        .enumerate()
        .for_each(|(i, o)| {
            let (g, b) = (i / slices, i % slices);
            simd::weighted_sum_f16_group(
                isa,
                &scores[g * group * ctx..(g + 1) * group * ctx],
                ctx,
                n,
                vcache,
                g * hd + b * DEC_SLICE,
                kv_dim,
                o,
                DEC_SLICE,
            );
        });
    for (i, o) in otmp[..kv_heads * group * hd]
        .chunks_exact(group * DEC_SLICE)
        .enumerate()
    {
        let (g, b) = (i / slices, i % slices);
        for (j, src) in o.chunks_exact(DEC_SLICE).enumerate() {
            let at = (g * group + j) * hd + b * DEC_SLICE;
            att[at..at + DEC_SLICE].copy_from_slice(src);
        }
    }
}

/// One KV head's attention for its group of query heads (`q` the group's heads, `hd` each):
/// scores over positions `0..n` into `sc` (one row of `ctx` a head), softmax and `f16` rounding
/// per head, then the value sums into `out` (`hd` a head). Each head's arithmetic is the
/// per-head kernels' exactly; the grouped kernels widen each `f16` K / V element once.
#[expect(
    clippy::too_many_arguments,
    reason = "the attention's buffers, passed explicitly"
)]
fn attend_group(
    isa: Isa,
    kcache: &[u16],
    vcache: &[u16],
    offset: usize,
    kv_dim: usize,
    hd: usize,
    q: &[f32],
    n: usize,
    sc: &mut [f32],
    ctx: usize,
    kq_scale: f32,
    out: &mut [f32],
) {
    simd::f16_matvec_strided_group(isa, kcache, offset, kv_dim, hd, q, n, sc, ctx);
    for row in sc.chunks_exact_mut(ctx) {
        let row = &mut row[..n];
        simd::softmax(isa, row, kq_scale);
        simd::round_f16_slice(isa, row);
    }
    simd::weighted_sum_f16_group(isa, sc, ctx, n, vcache, offset, kv_dim, out, hd);
}

/// Prefill's per-token RMS norm then quantisation, in parallel across the batch's tokens:
/// token `k` reads `src[k·dim ..][..dim]` and writes its own slices of `h` and `qa` and
/// `acts[k]`, exactly as the per-token step does.
fn norm_quant_tokens(
    isa: Isa,
    src: &[f32],
    w: &[f32],
    eps: f32,
    h: &mut [f32],
    qa: &mut [i8],
    acts: &mut [(f32, i32)],
) {
    let dim = w.len();
    let n = acts.len() * dim;
    src.par_chunks(dim)
        .zip(h[..n].par_chunks_mut(dim))
        .zip(qa[..n].par_chunks_mut(dim))
        .zip(acts.par_iter_mut())
        .for_each(|(((x, h), qa), act)| {
            simd::rms_norm(isa, x, w, eps, h);
            *act = quantise(isa, h, qa);
        });
}

/// Quantise `x` into `qa`: the scale and the codes' sum.
fn quantise(isa: Isa, x: &[f32], qa: &mut [i8]) -> (f32, i32) {
    let qa = &mut qa[..x.len()];
    simd::act_quant_sum(isa, x, qa)
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
            att_tmp: vec![0.0; c.heads * ctx],
            att_otmp: vec![0.0; q_dim],
            gate: vec![0.0; c.ffn],
            up: vec![0.0; c.ffn],
            out: vec![0.0; widest],
            qa: vec![0; widest],
            isum: vec![0; widest],
            isum2: vec![0; widest],
            isum3: vec![0; widest],
            cos: vec![0.0; c.head_dim / 2],
            sin: vec![0.0; c.head_dim / 2],
            logits: vec![0.0; c.vocab],
            screen: head::Scratch::new(c.vocab, c.embd),
            lora: None,
            screen_stats: (0, 0, 0),
        }
    }

    /// Run with `a`'s updates on top of the base model, from the first token.
    ///
    /// # Errors
    /// When the session has already consumed tokens (its cache holds the base's keys and values)
    /// or the adapter does not fit this model's projections.
    pub fn with_adapter(mut self, a: &'m Adapter) -> Result<Self, String> {
        if self.pos != 0 {
            return Err("an adapter goes on before the first token".into());
        }
        if !a.fits(self.m) {
            return Err("the adapter's shapes do not fit this model".into());
        }
        self.lora = Some(a);
        Ok(self)
    }

    /// Layer `li`'s update of projection `p`, when there is an adapter and it updates it.
    fn lora(&self, li: usize, p: Proj) -> Option<&'m lora::Pair> {
        self.lora.and_then(|a| a.get(li, p))
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

    /// The context size in positions.
    #[must_use]
    pub const fn ctx(&self) -> usize {
        self.ctx
    }

    /// Forget every position from `pos` on, keeping the cache before it (a shared prompt
    /// prefix); the next token is fed at `pos`. No-op when `pos` is not before the current one.
    pub fn rewind(&mut self, pos: usize) {
        self.pos = self.pos.min(pos);
    }

    /// Feed one token. With `logits`, compute and return the next-token logits (the full head).
    ///
    /// # Errors
    /// When the context is full or the token is outside the vocabulary.
    pub fn step(&mut self, token: u32, logits: bool) -> Result<Option<&[f32]>, StepError> {
        let _total = logits.then(|| crate::prof::scope(crate::prof::Stage::DecTotal));
        self.forward(token)?;
        if !logits {
            return Ok(None);
        }
        let _head = crate::prof::scope(crate::prof::Stage::DecHead);
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
        let _total = crate::prof::scope(crate::prof::Stage::DecTotal);
        self.forward(token)?;
        let _head = crate::prof::scope(crate::prof::Stage::DecHead);
        self.final_hidden();
        Ok(self.screen_argmax())
    }

    /// Feed one token and return the next token's top `k` (the full head's, exact values, largest
    /// first, lowest id on a tie) through the head screen ([`head::top_k`]), and the rows it
    /// recomputed. Falls back to the full head when the screen cannot serve `k`.
    ///
    /// # Errors
    /// When the context is full or the token is outside the vocabulary.
    pub fn step_top_k(
        &mut self,
        token: u32,
        k: usize,
    ) -> Result<(Vec<(u32, f32)>, usize), StepError> {
        let _total = crate::prof::scope(crate::prof::Stage::DecTotal);
        self.forward(token)?;
        let _head = crate::prof::scope(crate::prof::Stage::DecHead);
        self.final_hidden();
        let m = self.m;
        let embd = m.cfg.embd;
        if let Some(r) = head::top_k(
            self.isa,
            m,
            m.screen.get(m),
            &self.h[..embd],
            &mut self.screen,
            k,
        ) {
            return Ok(r);
        }
        self.head_full();
        let mut ids: Vec<u32> = (0..u32::try_from(self.logits.len()).unwrap_or(u32::MAX)).collect();
        let l = &self.logits;
        ids.sort_unstable_by(|a, b| l[*b as usize].total_cmp(&l[*a as usize]).then(a.cmp(b)));
        ids.truncate(k.max(1));
        Ok((
            ids.into_iter().map(|i| (i, l[i as usize])).collect(),
            l.len(),
        ))
    }

    /// After [`Self::step`] with logits: the screen's top `k` against the full head's (the same
    /// ids and the same value bits, in order), and the rows the screen recomputed. `None` when the
    /// screen cannot serve `k`. For the census and the oracle test.
    pub fn check_top_k(&mut self, k: usize) -> Option<(bool, usize)> {
        let m = self.m;
        let embd = m.cfg.embd;
        let (got, n) = head::top_k(
            self.isa,
            m,
            m.screen.get(m),
            &self.h[..embd],
            &mut self.screen,
            k,
        )?;
        let l = &self.logits;
        let mut ids: Vec<u32> = (0..u32::try_from(l.len()).unwrap_or(u32::MAX)).collect();
        ids.select_nth_unstable_by(k - 1, |a, b| {
            l[*b as usize].total_cmp(&l[*a as usize]).then(a.cmp(b))
        });
        ids.truncate(k);
        ids.sort_unstable_by(|a, b| l[*b as usize].total_cmp(&l[*a as usize]).then(a.cmp(b)));
        let same = got.len() == ids.len()
            && got
                .iter()
                .zip(&ids)
                .all(|(g, &i)| g.0 == i && g.1.to_bits() == l[i as usize].to_bits());
        Some((same, n))
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

    /// Feed a run of prompt tokens. Bit-identical to feeding them one at a time with
    /// [`Self::step`] (every per-token operation is the same; the projections are exact integer
    /// GEMMs), but the weights are read once per batch of up to 128 tokens instead of once a
    /// token. `MINERVA_PREFILL=step` takes the one-at-a-time path (the A/B arm and the oracle).
    ///
    /// # Errors
    /// When the context would overflow or a token is outside the vocabulary.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<(), StepError> {
        if std::env::var("MINERVA_PREFILL").as_deref() == Ok("step") {
            for &t in tokens {
                self.forward(t)?;
            }
            return Ok(());
        }
        for batch in tokens.chunks(PREFILL_BATCH) {
            self.prefill_batch(batch)?;
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the block, as `forward`, over a batch"
    )]
    fn prefill_batch(&mut self, tokens: &[u32]) -> Result<(), StepError> {
        let _total = crate::prof::scope(crate::prof::Stage::Total);
        let m = self.m;
        let isa = self.isa;
        let c = &m.cfg;
        let bm = tokens.len();
        if self.pos + bm > self.ctx {
            return Err(StepError::ContextFull(self.ctx));
        }
        for &t in tokens {
            if usize::try_from(t).ok().is_none_or(|t| t >= c.vocab) {
                return Err(StepError::BadToken(t));
            }
        }
        let (embd, kv_dim, hd, ffn) = (c.embd, c.kv_dim(), c.head_dim, c.ffn);
        let q_dim = c.heads * hd;
        let group = c.heads / c.kv_heads;
        let widest = embd.max(q_dim).max(ffn);
        let rows_max = m.layers.first().map_or(0, |l| {
            [l.q.rows, l.k.rows, l.o.rows, l.gate.rows, l.down.rows]
                .into_iter()
                .max()
                .unwrap_or(0)
        });
        let (pos0, ctx) = (self.pos, self.ctx);
        #[expect(clippy::cast_precision_loss, reason = "head width, small")]
        let kq_scale = 1.0 / (hd as f32).sqrt();

        // The batch's buffers, once per batch (a prompt is one to a few batches).
        let mut x = vec![0f32; bm * embd];
        let mut h = vec![0f32; bm * widest];
        let mut q = vec![0f32; bm * q_dim];
        let mut kc = vec![0f32; bm * kv_dim];
        let mut vc = vec![0f32; bm * kv_dim];
        let mut att = vec![0f32; bm * q_dim];
        let mut gate = vec![0f32; bm * ffn];
        let mut up = vec![0f32; bm * ffn];
        let mut out = vec![0f32; bm * embd];
        let mut qa = vec![0i8; bm * widest];
        let mut acts = vec![(0f32, 0i32); bm];
        let mut ysum = vec![0i32; rows_max.max(widest) * bm];
        let mut scores = vec![0f32; bm * c.heads * ctx];
        let (mut cos, mut sin) = (vec![0f32; bm * hd / 2], vec![0f32; bm * hd / 2]);

        for (k, &tok) in tokens.iter().enumerate() {
            simd::f16_bytes_to_f32(
                isa,
                m.f16_row(m.embd, embd, tok as usize),
                &mut x[k * embd..(k + 1) * embd],
            );
            let half = hd / 2;
            rope_table(
                pos0 + k,
                hd,
                c.rope_base,
                &mut cos[k * half..(k + 1) * half],
                &mut sin[k * half..(k + 1) * half],
            );
        }

        for (li, l) in m.layers.iter().enumerate() {
            // --- attention ---
            norm_quant_tokens(
                isa,
                &x[..bm * embd],
                &l.attn_norm,
                c.rms_eps,
                &mut h,
                &mut qa,
                &mut acts[..bm],
            );
            bitlinear_batch(isa, m, &l.q, &qa, &acts, &mut ysum, &mut q, q_dim);
            bitlinear_batch(isa, m, &l.k, &qa, &acts, &mut ysum, &mut kc, kv_dim);
            bitlinear_batch(isa, m, &l.v, &qa, &acts, &mut ysum, &mut vc, kv_dim);
            for (p, o, w) in [
                (Proj::Q, &mut q, q_dim),
                (Proj::K, &mut kc, kv_dim),
                (Proj::V, &mut vc, kv_dim),
            ] {
                if let Some(p) = self.lora(li, p) {
                    lora::apply_batch(p, &h, embd, o, w, bm);
                }
            }
            let half = hd / 2;
            let layer_base = li * ctx * kv_dim;
            let at = layer_base + pos0 * kv_dim;
            q[..bm * q_dim]
                .par_chunks_mut(q_dim)
                .zip(kc[..bm * kv_dim].par_chunks_mut(kv_dim))
                .zip(vc[..bm * kv_dim].par_chunks(kv_dim))
                .zip(self.k[at..at + bm * kv_dim].par_chunks_mut(kv_dim))
                .zip(self.v[at..at + bm * kv_dim].par_chunks_mut(kv_dim))
                .zip(cos[..bm * half].par_chunks(half))
                .zip(sin[..bm * half].par_chunks(half))
                .for_each(|((((((q, kc), vc), kd), vd), cs), sn)| {
                    for head in q.chunks_exact_mut(hd) {
                        simd::rope_neox(isa, head, cs, sn);
                    }
                    for head in kc.chunks_exact_mut(hd) {
                        simd::rope_neox(isa, head, cs, sn);
                    }
                    simd::f32_to_f16(isa, kc, kd);
                    simd::f32_to_f16(isa, vc, vd);
                    simd::round_f16_slice(isa, q);
                });
            let kcache = &self.k[layer_base..layer_base + (pos0 + bm) * kv_dim];
            let vcache = &self.v[layer_base..layer_base + (pos0 + bm) * kv_dim];
            let qr = &q;
            let att_scope = crate::prof::scope(crate::prof::Stage::Attention);
            // One task a (token, KV head): the group's `group` query heads share each widened
            // K and V element (bit-identical to the per-head kernels).
            att.par_chunks_mut(group * hd)
                .zip(scores.par_chunks_mut(group * ctx))
                .enumerate()
                .for_each(|(i, (o, sc))| {
                    let (k, g) = (i / c.kv_heads, i % c.kv_heads);
                    let n = pos0 + k + 1;
                    let qg = &qr[k * q_dim + g * group * hd..k * q_dim + (g + 1) * group * hd];
                    attend_group(
                        isa,
                        kcache,
                        vcache,
                        g * hd,
                        kv_dim,
                        hd,
                        qg,
                        n,
                        sc,
                        ctx,
                        kq_scale,
                        o,
                    );
                });
            drop(att_scope);
            norm_quant_tokens(
                isa,
                &att[..bm * q_dim],
                &l.attn_sub_norm,
                c.rms_eps,
                &mut h,
                &mut qa,
                &mut acts[..bm],
            );
            bitlinear_batch(isa, m, &l.o, &qa, &acts, &mut ysum, &mut out, embd);
            if let Some(p) = self.lora(li, Proj::O) {
                lora::apply_batch(p, &h, q_dim, &mut out, embd, bm);
            }
            x[..bm * embd]
                .par_chunks_mut(embd)
                .zip(out[..bm * embd].par_chunks(embd))
                .for_each(|(x, o)| simd::add_assign(isa, x, o));

            // --- feed-forward ---
            norm_quant_tokens(
                isa,
                &x[..bm * embd],
                &l.ffn_norm,
                c.rms_eps,
                &mut h,
                &mut qa,
                &mut acts[..bm],
            );
            bitlinear_batch(isa, m, &l.gate, &qa, &acts, &mut ysum, &mut gate, ffn);
            bitlinear_batch(isa, m, &l.up, &qa, &acts, &mut ysum, &mut up, ffn);
            for (p, o) in [(Proj::Gate, &mut gate), (Proj::Up, &mut up)] {
                if let Some(p) = self.lora(li, p) {
                    lora::apply_batch(p, &h, embd, o, ffn, bm);
                }
            }
            gate[..bm * ffn]
                .par_chunks_mut(ffn)
                .zip(up[..bm * ffn].par_chunks(ffn))
                .zip(h[..bm * ffn].par_chunks_mut(ffn))
                .zip(qa[..bm * ffn].par_chunks_mut(ffn))
                .zip(acts[..bm].par_iter_mut())
                .for_each(|((((g, u), h), qa), act)| {
                    simd::relu2_mul(isa, g, u);
                    simd::rms_norm(isa, g, &l.ffn_sub_norm, c.rms_eps, h);
                    *act = quantise(isa, h, qa);
                });
            bitlinear_batch(isa, m, &l.down, &qa, &acts, &mut ysum, &mut out, embd);
            if let Some(p) = self.lora(li, Proj::Down) {
                lora::apply_batch(p, &h, ffn, &mut out, embd, bm);
            }
            x[..bm * embd]
                .par_chunks_mut(embd)
                .zip(out[..bm * embd].par_chunks(embd))
                .for_each(|(x, o)| simd::add_assign(isa, x, o));
        }
        self.x.copy_from_slice(&x[(bm - 1) * embd..bm * embd]);
        self.pos += bm;
        Ok(())
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
            // q, k and v share one activation: one join tree, one sync point, and the small k / v
            // fill the threads q leaves idle (each has its own sum scratch).
            {
                let _t = crate::prof::scope(crate::prof::Stage::DecTernary);
                let (qa, isum, isum2, isum3) =
                    (&self.qa, &mut self.isum, &mut self.isum2, &mut self.isum3);
                let (qo, ko, vo) = (&mut self.q, &mut self.kc, &mut self.vc);
                rayon::join(
                    || bitlinear(isa, m, &l.q, qa, act, isum, qo),
                    || {
                        rayon::join(
                            || bitlinear(isa, m, &l.k, qa, act, isum2, ko),
                            || bitlinear(isa, m, &l.v, qa, act, isum3, vo),
                        )
                    },
                );
                if self.lora.is_some() {
                    let (q, k, v) = (
                        self.lora(li, Proj::Q),
                        self.lora(li, Proj::K),
                        self.lora(li, Proj::V),
                    );
                    lora::apply_group(
                        &mut [
                            (q, &mut self.q[..]),
                            (k, &mut self.kc[..]),
                            (v, &mut self.vc[..]),
                        ],
                        &self.h[..embd],
                    );
                }
            }
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
            let att_scope = crate::prof::scope(crate::prof::Stage::DecAttention);
            if dec_grouped(n, group, hd) {
                attend_decode_grouped(
                    isa,
                    kcache,
                    vcache,
                    q,
                    n,
                    (c.kv_heads, group, hd, kv_dim, ctx),
                    kq_scale,
                    (&mut self.scores, &mut self.att_tmp, &mut self.att_otmp),
                    &mut self.att,
                );
            } else {
                // One task a query head: decode has only 5 KV-head tasks, which idles threads (measured:
                // the grouped path lost at 64-1,088 positions); prefill uses the grouped kernels.
                self.att
                    .par_chunks_mut(hd)
                    .zip(self.scores.par_chunks_mut(ctx))
                    .enumerate()
                    .for_each(|(head, (out, sc))| {
                        let g = head / group;
                        let sc = &mut sc[..n];
                        let qh = &q[head * hd..(head + 1) * hd];
                        simd::f16_matvec_strided(isa, kcache, g * hd, kv_dim, qh, sc);
                        simd::softmax(isa, sc, kq_scale);
                        simd::round_f16_slice(isa, sc);
                        simd::weighted_sum_f16(isa, sc, vcache, g * hd, kv_dim, out);
                    });
            }
            drop(att_scope);
            simd::rms_norm(
                isa,
                &self.att,
                &l.attn_sub_norm,
                c.rms_eps,
                &mut self.h[..q_dim],
            );
            let act = quantise(isa, &self.h[..q_dim], &mut self.qa);
            {
                let _t = crate::prof::scope(crate::prof::Stage::DecTernary);
                bitlinear(isa, m, &l.o, &self.qa, act, &mut self.isum, &mut self.out);
                if let Some(p) = self.lora(li, Proj::O) {
                    lora::apply_group(&mut [(Some(p), &mut self.out[..])], &self.h[..q_dim]);
                }
            }
            simd::add_assign(isa, &mut self.x, &self.out[..embd]);

            // --- feed-forward ---
            simd::rms_norm(isa, &self.x, &l.ffn_norm, c.rms_eps, &mut self.h[..embd]);
            let act = quantise(isa, &self.h[..embd], &mut self.qa);
            // gate and up share one activation: one join, one sync point.
            {
                let _t = crate::prof::scope(crate::prof::Stage::DecTernary);
                let (qa, isum, isum2) = (&self.qa, &mut self.isum, &mut self.isum2);
                let (go, uo) = (&mut self.gate, &mut self.up);
                rayon::join(
                    || bitlinear(isa, m, &l.gate, qa, act, isum, go),
                    || bitlinear(isa, m, &l.up, qa, act, isum2, uo),
                );
                if self.lora.is_some() {
                    let (g, u) = (self.lora(li, Proj::Gate), self.lora(li, Proj::Up));
                    lora::apply_group(
                        &mut [(g, &mut self.gate[..]), (u, &mut self.up[..])],
                        &self.h[..embd],
                    );
                }
            }
            simd::relu2_mul(isa, &mut self.gate, &self.up);
            simd::rms_norm(
                isa,
                &self.gate,
                &l.ffn_sub_norm,
                c.rms_eps,
                &mut self.h[..ffn],
            );
            let act = quantise(isa, &self.h[..ffn], &mut self.qa);
            let _t = crate::prof::scope(crate::prof::Stage::DecTernary);
            bitlinear(
                isa,
                m,
                &l.down,
                &self.qa,
                act,
                &mut self.isum,
                &mut self.out,
            );
            if let Some(p) = self.lora(li, Proj::Down) {
                lora::apply_group(&mut [(Some(p), &mut self.out[..])], &self.h[..ffn]);
            }
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

/// The log of the softmax's denominator over every logit: `max + ln Σ exp(l - max)`, in `f64`,
/// summed in vocabulary order with `libm`'s `exp` and `log` (pure Rust, so the same bits on
/// every device, unlike the platform's maths library).
#[must_use]
pub fn log_norm(logits: &[f32]) -> f64 {
    let max = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let z: f64 = logits.iter().map(|&l| libm::exp(f64::from(l) - max)).sum();
    max + libm::log(z)
}

/// Each token's log-probability under `logits` (a log-softmax over the whole vocabulary),
/// given [`log_norm`] of them.
#[must_use]
pub fn logprob(logits: &[f32], norm: f64, token: u32) -> f64 {
    logits
        .get(token as usize)
        .map_or(f64::NEG_INFINITY, |&l| f64::from(l) - norm)
}

/// The `k` likeliest tokens and their log-probabilities, likeliest first (lowest id on a tie).
#[must_use]
pub fn top_k_logprobs(logits: &[f32], k: usize) -> Vec<(u32, f64)> {
    top_k_logprobs_norm(logits, k, log_norm(logits))
}

/// [`top_k_logprobs`] given [`log_norm`] of the logits already (a caller that also needs the
/// chosen token's log-probability computes the norm, 128 K `exp`s, once).
#[must_use]
pub fn top_k_logprobs_norm(logits: &[f32], k: usize, norm: f64) -> Vec<(u32, f64)> {
    top_ids(logits, k)
        .into_iter()
        .map(|i| (i, logprob(logits, norm, i)))
        .collect()
}

/// The `k` largest logits' ids, largest first, the lowest id on a tie.
fn top_ids(logits: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let k = k.min(idx.len());
    if k == 0 {
        return Vec::new();
    }
    idx.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b].total_cmp(&logits[a]).then(a.cmp(&b))
    });
    idx.truncate(k);
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    idx.into_iter()
        .map(|i| u32::try_from(i).unwrap_or(u32::MAX))
        .collect()
}

/// The `k` likeliest tokens with their probabilities (softmax in `f64` with `libm`, for
/// reporting only).
#[must_use]
pub fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f64)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let z: f64 = logits
        .iter()
        .map(|&l| libm::exp(f64::from(l) - f64::from(max)))
        .sum();
    top_ids(logits, k)
        .into_iter()
        .map(|i| {
            (
                i,
                libm::exp(f64::from(logits[i as usize]) - f64::from(max)) / z,
            )
        })
        .collect()
}

/// Teacher-forced scoring: the log-probability of each of `target` after `prompt` (the
/// prompt's tokens, then each target token in turn, each step through the full head). The
/// sequence's log-likelihood is their sum.
///
/// # Errors
/// When the context overflows, a token is out of range, or `prompt` is empty.
pub fn score(s: &mut Session<'_>, prompt: &[u32], target: &[u32]) -> Result<Vec<f64>, StepError> {
    rayon::scope(|_| {
        let (last, head) = prompt.split_last().ok_or(StepError::BadToken(u32::MAX))?;
        s.prefill(head)?;
        let mut out = Vec::with_capacity(target.len());
        let mut next = *last;
        for &t in target {
            if usize::try_from(t).ok().is_none_or(|t| t >= s.m.cfg.vocab) {
                return Err(StepError::BadToken(t));
            }
            let logits = s.step(next, true)?.unwrap_or_default();
            out.push(logprob(logits, log_norm(logits), t));
            next = t;
        }
        Ok(out)
    })
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
    on_step: impl FnMut(usize, Option<&[f32]>, u32) + Send,
) -> Result<Vec<u32>, StepError> {
    // Run on a pool worker: every parallel region below then forks through that worker's own
    // deque instead of the pool's global injector (from a non-pool thread each of the ~240
    // regions a token was injected, and the injector allocated a block every 63 jobs).
    // (A scope also on a worker already: keep-awake jobs need one.)
    if rayon::current_thread_index().is_some()
        || std::env::var("MINERVA_POOL").as_deref() != Ok("outside")
    {
        return rayon::scope(|sc| generate_in(s, prompt, new, mode, on_step, Some(sc)));
    }
    generate_in(s, prompt, new, mode, on_step, None)
}

/// Lets the kept-awake workers go when dropped (also on an error or a panic).
struct Awake(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Drop for Awake {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
}

/// Prompts up to this many tokens keep the workers awake by default (on aarch64).
const SPIN_MAX_PROMPT: usize = 64;

/// Whether to keep the pool's other workers awake through the prompt and its first token:
/// `MINERVA_SPIN=0` never, `prompt` always, unset for prompts up to [`SPIN_MAX_PROMPT`] tokens on
/// aarch64. Pixel 9 Pro XL, 4 threads: first token 1.31x at 64 tokens with decode unchanged, about
/// 1.1x at 256 and 512, and decode after a 512-token prompt 0.93x (the spinning cores warm the
/// phone); spinning through the whole answer cost decode 0.87x. Not measured elsewhere.
fn spin_prompt(prompt_len: usize) -> bool {
    match std::env::var("MINERVA_SPIN").as_deref() {
        Ok("0") => false,
        Ok("prompt") => true,
        _ => cfg!(target_arch = "aarch64") && prompt_len <= SPIN_MAX_PROMPT,
    }
}

/// Keeps the pool's other workers looking for work until the returned guard drops, instead of
/// parking between a token's many short regions (a parked worker wakes slowly, and on a phone its
/// core clocks down). A broadcast job cannot be stolen, so a worker waiting in a join never picks
/// up another's; the caller's own copy returns at once.
fn keep_awake(sc: &rayon::Scope<'_>) -> Option<Awake> {
    let me = rayon::current_thread_index()?;
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = std::sync::Arc::clone(&stop);
    sc.spawn_broadcast(move |_, ctx| {
        if ctx.index() == me {
            return;
        }
        while !flag.load(std::sync::atomic::Ordering::Acquire) {
            if rayon::yield_now() != Some(rayon::Yield::Executed) {
                std::hint::spin_loop();
            }
        }
    });
    Some(Awake(stop))
}

/// [`generate`] on a pool worker; `pool` is the scope keep-awake workers run in.
fn generate_in<'s, 'm: 's>(
    s: &mut Session<'m>,
    prompt: &[u32],
    new: usize,
    mode: HeadMode,
    mut on_step: impl FnMut(usize, Option<&[f32]>, u32),
    pool: Option<&rayon::Scope<'s>>,
) -> Result<Vec<u32>, StepError> {
    let mut awake = pool
        .filter(|_| spin_prompt(prompt.len()))
        .and_then(keep_awake);
    let (last, head) = prompt.split_last().ok_or(StepError::BadToken(u32::MAX))?;
    s.prefill(head)?;
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
        // The prompt and its first token are done: let the workers park for the answer.
        drop(awake.take());
    }
    Ok(out)
}

#[cfg(test)]
mod logprob_tests {
    use super::*;

    #[test]
    fn logprobs_are_a_log_softmax() {
        let logits = [2.0f32, -1.0, 0.5, 2.0, -30.0];
        let norm = log_norm(&logits);
        let total: f64 = (0..5).map(|t| libm::exp(logprob(&logits, norm, t))).sum();
        assert!((total - 1.0).abs() < 1e-12, "{total}");
        // Shifting every logit moves the norm, not the log-probabilities.
        let shifted: Vec<f32> = logits.iter().map(|l| l + 100.0).collect();
        let n2 = log_norm(&shifted);
        assert!((logprob(&shifted, n2, 2) - logprob(&logits, norm, 2)).abs() < 1e-9);
        // Ties go to the lowest id; an id outside the vocabulary has no probability.
        let top = top_k_logprobs(&logits, 3);
        assert_eq!(top.iter().map(|t| t.0).collect::<Vec<_>>(), vec![0, 3, 2]);
        assert_eq!(logprob(&logits, norm, 9), f64::NEG_INFINITY);
        assert!(top_k_logprobs(&logits, 0).is_empty());
    }
}
