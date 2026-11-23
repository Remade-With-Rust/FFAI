//! The greedy head's screen: greedy decoding needs only the argmax of the `vocab x embd` `f16`
//! head times the hidden state, so screen every row with an `int8` copy (half the head's bytes)
//! and a proven per-row bound, then recompute exactly only the rows the screen cannot rule out.
//! The token is, by proof, the full head's: the same `f32` value, the same lowest-id tie-break.
//! (rusty_sloth measured this on the GPU: 1.27x end to end, commit `04c07b1`; the bound follows
//! its `head_q8_prepare` / `head_q8_verify` with the activation quantised too, for an integer
//! screen.)
//!
//! ## The bound
//!
//! For row `r` with `f16` weights `w`, the hidden state `x` (already rounded to `f16`):
//!
//! - `F = w·x` (real), `f` = [`crate::simd::f16_matvec_bytes`]'s `f32` value, the one the full
//!   head computes. Its products are exact (`f16 x f16` fits an `f32`); a lane adds `embd / 16`
//!   of them and the reduction 4 more, under 164 roundings in series for `embd <= 2560`, so
//!   `|f - F| <= γ_164 Σ|w_j x_j| <= 9.8e-6 ‖w‖₂ ‖x‖₂` (Cauchy-Schwarz).
//! - The copy: `s = max|w| / 127` (`f32`), `q = round(w / s)` clamped to ±127. `A = s·(q·x)` and
//!   `F - A = (w - s q)·x`, so `|F - A| <= ‖w - s q‖₂ ‖x‖₂`.
//! - `a_r`: [`crate::simd::i8f_matvec`]'s `f32` dot (exact products, under 164 roundings:
//!   `<= 9.8e-6 ‖q‖₂ ‖x‖₂` off `q·x`), times `s` (one rounding): `|a_r - A| <= 1e-5 s ‖q‖₂ ‖x‖₂ + u|a_r|`.
//!
//! Per row, made once at load: `c = (1 + 1e-4)(‖w - s q‖₂ + 1e-5 ‖w‖₂ + 1e-5 s ‖q‖₂)` (norms in
//! `f64`, rounded up to `f32`). Per token: `X >= ‖x‖₂`. Then `B = (1 + 1e-4) c X + 1e-6 |a_r|`
//! covers every term and the roundings of `B` and `a_r ± B` themselves, so
//! **`f ∈ [lo, hi] = [a_r - B, a_r + B]`**.
//!
//! The activation stays `f32`. An `int8` activation (an integer screen) was tried first and
//! refuted: its term `s ‖q‖₂ ‖x - t·a‖₂` ran 1.1-4.7x the weight term, because this model's hidden
//! state has outlier channels (`max|x| / rms` 5-22), leaving 99 and 38,751 survivors a step on two
//! golden prompts.
//!
//! ## Why the survivors hold the winner
//!
//! Let `T = max_r lo_r` (at row `p`) and `r*` the full head's winner (largest `f`, lowest id).
//! `f_{r*} >= f_p >= lo_p = T`. A row with `hi_r < T` has `f_r <= hi_r < T <= f_{r*}`: neither
//! the winner nor tied with it. So every row with the winning value survives `hi_r >= T`, and
//! the lowest id among the survivors' exact maxima is `r*`.

use std::sync::OnceLock;

use rayon::prelude::*;

use crate::model::Model;
use crate::numerics::f16_table;
use crate::simd::{self, Isa};

/// Vocabulary rows a thread screens at a time.
const CHUNK: usize = 256;

/// `x` (non-negative) as the smallest `f32` not below it.
fn round_up(x: f64) -> f32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rounded, then nudged up if it went down"
    )]
    let f = x as f32;
    if f64::from(f) < x {
        f32::from_bits(f.to_bits() + 1)
    } else {
        f
    }
}

/// The `int8` copy of a head and each row's bound coefficient.
pub struct HeadQ8 {
    cols: usize,
    q: Vec<i8>,
    scale: Vec<f32>,
    c: Vec<f32>,
}

impl std::fmt::Debug for HeadQ8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeadQ8")
            .field("rows", &self.scale.len())
            .field("cols", &self.cols)
            .finish_non_exhaustive()
    }
}

impl HeadQ8 {
    /// Quantise `m`'s head in chunks of rows (in parallel), releasing each chunk's `f16` pages from
    /// the resident set once it is quantised: the head is read once, and never held whole.
    #[must_use]
    pub fn build(m: &Model) -> Self {
        let (rows, cols) = (m.cfg.vocab, m.cfg.embd);
        let mut q = vec![0i8; rows * cols];
        let mut scale = vec![0f32; rows];
        let mut c = vec![0f32; rows];
        let t16 = f16_table();
        q.par_chunks_mut(CHUNK * cols)
            .zip(scale.par_chunks_mut(CHUNK))
            .zip(c.par_chunks_mut(CHUNK))
            .enumerate()
            .for_each_init(
                || vec![0f32; cols],
                |w, (ci, ((qc, sc), bc))| {
                    let r0 = ci * CHUNK;
                    for (j, ((qr, s), b)) in qc
                        .chunks_exact_mut(cols)
                        .zip(sc.iter_mut())
                        .zip(bc.iter_mut())
                        .enumerate()
                    {
                        quantise_row(m.f16_row(m.head, cols, r0 + j), t16, w, qr, s, b);
                    }
                    m.release_head_rows(r0..r0 + sc.len());
                },
            );
        Self { cols, q, scale, c }
    }

    /// The bytes the screen reads a token.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.q.len() + 8 * self.scale.len()
    }
}

/// One row: `w` (scratch) its `f32` values, `qr` its codes, `s` its scale, `b` its bound
/// coefficient.
fn quantise_row(bytes: &[u8], t16: &[f32], w: &mut [f32], qr: &mut [i8], s: &mut f32, b: &mut f32) {
    for (v, x) in w.iter_mut().zip(bytes.chunks_exact(2)) {
        *v = t16[usize::from(u16::from_le_bytes([x[0], x[1]]))];
    }
    let amax = w.iter().fold(0f32, |a, v| a.max(v.abs()));
    if !amax.is_finite() {
        // A non-finite weight: the row can never be ruled out.
        (*s, *b) = (0.0, f32::INFINITY);
        return;
    }
    let sc = amax / 127.0;
    let (mut e2, mut w2, mut q2) = (0f64, 0f64, 0f64);
    for (qv, &wv) in qr.iter_mut().zip(w.iter()) {
        #[expect(clippy::cast_possible_truncation, reason = "clamped to ±127")]
        let k = if sc > 0.0 {
            (wv / sc).round_ties_even().clamp(-127.0, 127.0) as i8
        } else {
            0
        };
        *qv = k;
        let e = f64::from(wv) - f64::from(sc) * f64::from(k);
        e2 += e * e;
        w2 += f64::from(wv) * f64::from(wv);
        q2 += f64::from(k) * f64::from(k);
    }
    *s = sc;
    *b = round_up((e2.sqrt() + 1e-5 * w2.sqrt() + 1e-5 * f64::from(sc) * q2.sqrt()) * (1.0 + 1e-4));
}

/// A screen's per-session scratch, allocated once.
#[derive(Debug, Default)]
pub struct Scratch {
    dots: Vec<f32>,
    hi: Vec<f32>,
    survivors: Vec<u32>,
    one: [f32; 1],
}

impl Scratch {
    /// Scratch for a head of `rows x cols`.
    #[must_use]
    pub fn new(rows: usize, cols: usize) -> Self {
        let _ = cols;
        Self {
            dots: vec![0.0; rows],
            hi: vec![0.0; rows],
            survivors: Vec::with_capacity(64),
            one: [0.0],
        }
    }
}

/// The screened argmax of `m`'s head times `x` (the hidden state rounded to `f16`), or `None`
/// when `x` is all zero or not finite (the caller then runs the full head). Returns the token
/// and the number of rows recomputed exactly.
pub fn argmax(
    isa: Isa,
    m: &Model,
    h8: &HeadQ8,
    x: &[f32],
    sc: &mut Scratch,
) -> Option<(u32, usize)> {
    let cols = h8.cols;
    assert_eq!(x.len(), cols);
    let x2: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    if !(x2.is_finite() && x2 > 0.0) {
        return None;
    }
    let xn = round_up(x2.sqrt() * (1.0 + 1e-4));

    let lo_max = sc
        .dots
        .par_chunks_mut(CHUNK)
        .zip(sc.hi.par_chunks_mut(CHUNK))
        .zip(h8.q.par_chunks(CHUNK * cols))
        .enumerate()
        .map(|(ci, ((dots, hi), q))| {
            simd::i8f_matvec(isa, q, x, dots);
            let base = ci * CHUNK;
            let mut lo_max = f32::NEG_INFINITY;
            for (j, (&d, h)) in dots.iter().zip(hi.iter_mut()).enumerate() {
                let r = base + j;
                let av = d * h8.scale[r];
                let b = h8.c[r] * xn * 1.0001 + 1e-6 * av.abs();
                *h = av + b;
                lo_max = lo_max.max(av - b);
            }
            lo_max
        })
        .reduce(|| f32::NEG_INFINITY, f32::max);

    sc.survivors.clear();
    for (r, &h) in sc.hi.iter().enumerate() {
        // A NaN bound keeps its row.
        if h >= lo_max || h.is_nan() {
            sc.survivors.push(u32::try_from(r).unwrap_or(u32::MAX));
        }
    }
    let mut best: Option<(u32, f32)> = None;
    for &r in &sc.survivors {
        simd::f16_matvec_bytes(isa, m.f16_row(m.head, cols, r as usize), x, &mut sc.one);
        let f = sc.one[0];
        if best.is_none_or(|(_, bf)| f > bf) {
            best = Some((r, f));
        }
    }
    best.map(|(r, _)| (r, sc.survivors.len()))
}

/// A model's screen, built on first use.
#[derive(Debug, Default)]
pub struct Lazy(OnceLock<HeadQ8>);

impl Lazy {
    /// The screen of `m`, building it the first time.
    pub fn get(&self, m: &Model) -> &HeadQ8 {
        self.0.get_or_init(|| HeadQ8::build(m))
    }
}
