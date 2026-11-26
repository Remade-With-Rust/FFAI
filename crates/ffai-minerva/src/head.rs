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
//! The default screen's activation is 16 bits: `a = round(x / t)`, `x = t·a + e`. Its integer dot
//! `D = q·a` is exact, `a_r = s·t·D` (computed in `f64`, then one rounding to `f32`), so
//! `|s·(q·x) - a_r| <= s ‖q‖₂ ‖e‖₂ + 2u|a_r|`, and the same `B` plus `s ‖q‖₂ ‖e‖₂` covers it. `‖e‖`
//! is measured, not estimated, each token. `MINERVA_SCREEN=f32` keeps the float dot above.
//!
//! An `int8` activation (an integer screen) was tried first and
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

/// Lanes of the quantiser's folds.
const W: usize = 8;

/// `1.5 * 2^23`: adding then subtracting it rounds an `f32` below `2^22` to an integer, ties even.
const MAGIC: f32 = 12_582_912.0;

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

/// A screen's codes: built in memory, or mapped from a head file ([`HeadQ8::open`]).
enum Codes {
    Owned(Vec<i8>),
    #[cfg(not(target_arch = "wasm32"))]
    Mapped {
        map: memmap2::Mmap,
        at: usize,
        len: usize,
    },
}

impl Codes {
    #[allow(unsafe_code)]
    fn as_slice(&self) -> &[i8] {
        match self {
            Self::Owned(v) => v,
            #[cfg(not(target_arch = "wasm32"))]
            Self::Mapped { map, at, len } => {
                let b = &map[*at..*at + *len];
                // SAFETY: `i8` and `u8` have the same size and alignment and every bit pattern is
                // valid for both; the slice borrows the map, which lives as long as `self`.
                unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<i8>(), b.len()) }
            }
        }
    }
}

/// The `int8` copy of a head and each row's bound coefficient.
pub struct HeadQ8 {
    cols: usize,
    q: Codes,
    scale: Vec<f32>,
    c: Vec<f32>,
    /// `s ‖q‖₂`, rounded up: what a row multiplies the activation's error by.
    sq: Vec<f32>,
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
        let mut sq = vec![0f32; rows];
        let t16 = f16_table();
        q.par_chunks_mut(CHUNK * cols)
            .zip(scale.par_chunks_mut(CHUNK))
            .zip(c.par_chunks_mut(CHUNK))
            .zip(sq.par_chunks_mut(CHUNK))
            .enumerate()
            .for_each_init(
                || vec![0f32; cols],
                |w, (ci, (((qc, sc), bc), nc))| {
                    let r0 = ci * CHUNK;
                    for (j, (((qr, s), b), n)) in qc
                        .chunks_exact_mut(cols)
                        .zip(sc.iter_mut())
                        .zip(bc.iter_mut())
                        .zip(nc.iter_mut())
                        .enumerate()
                    {
                        (*s, *b, *n) = quantise_row(m.f16_row(m.head, cols, r0 + j), t16, w, qr);
                    }
                    m.release_head_rows(r0..r0 + sc.len());
                },
            );
        Self {
            cols,
            q: Codes::Owned(q),
            scale,
            c,
            sq,
        }
    }

    /// The bytes the screen reads a token.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.q.as_slice().len() + 12 * self.scale.len()
    }

    /// Write this screen as a head file for `m` (layout at [`HEAD_FILE`]), so a later process
    /// can [`Self::open`] it instead of building it.
    ///
    /// # Errors
    /// When the file cannot be written.
    pub fn write(&self, m: &Model, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let rows = self.scale.len();
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        let header = head_file_header(rows, self.cols, fingerprint(m));
        f.write_all(&header)?;
        for v in [&self.scale, &self.c, &self.sq] {
            for x in v {
                f.write_all(&x.to_le_bytes())?;
            }
        }
        let at = header.len() + 12 * rows;
        f.write_all(&vec![0u8; at.next_multiple_of(PAGE) - at])?;
        for chunk in self.q.as_slice().chunks(1 << 20) {
            let bytes: Vec<u8> = chunk.iter().map(|v| v.cast_unsigned()).collect();
            f.write_all(&bytes)?;
        }
        f.flush()
    }

    /// Open a head file written by [`Self::write`] for `m`, mapping its codes (on wasm32,
    /// reading them). It is accepted only if it was made from this model by this quantiser:
    /// every [`SAMPLE`]th row is quantised again from the model and must equal the file's codes,
    /// scale and bound coefficients bit for bit, and those rows' bytes must give the file's
    /// fingerprint. The screen's proof is checked for those rows; for the rest it rests on the
    /// file being the one [`Self::write`] wrote for this model, which the explicit `--head` flag
    /// asserts.
    ///
    /// # Errors
    /// When the file cannot be read, is not a version-1 head file of this shape, or does not
    /// match the model.
    pub fn open(m: &Model, path: &std::path::Path) -> Result<Self, String> {
        let (rows, cols) = (m.cfg.vocab, m.cfg.embd);
        let at_codes = (HEADER + 12 * rows).next_multiple_of(PAGE);
        let total = at_codes + rows * cols;
        #[cfg(not(target_arch = "wasm32"))]
        let map = {
            let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
            map_file(&f).map_err(|e| e.to_string())?
        };
        #[cfg(target_arch = "wasm32")]
        let map = std::fs::read(path).map_err(|e| e.to_string())?;
        let b: &[u8] = &map;
        if b.len() != total || b[..8] != HEAD_FILE[..] {
            return Err(format!(
                "not a head file for this model ({} bytes; want {total}, starting MNVHEAD)",
                b.len()
            ));
        }
        let word = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        if word(8) != 1 || word(12) as usize != rows || word(16) as usize != cols {
            return Err(format!(
                "head file version {} for {} x {}; want version 1 for {rows} x {cols}",
                word(8),
                word(12),
                word(16)
            ));
        }
        let mut fp = [0u8; 8];
        fp.copy_from_slice(&b[24..32]);
        if u64::from_le_bytes(fp) != fingerprint(m) {
            return Err("head file made from a different model (fingerprint differs)".into());
        }
        let floats = |k: usize| -> Vec<f32> {
            let at = HEADER + 4 * k * rows;
            b[at..at + 4 * rows]
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                .collect()
        };
        let (scale, c, sq) = (floats(0), floats(1), floats(2));
        let codes = &b[at_codes..];
        let t16 = f16_table();
        let (mut w, mut qr) = (vec![0f32; cols], vec![0i8; cols]);
        for r in (0..rows).step_by(SAMPLE) {
            let got = quantise_row(m.f16_row(m.head, cols, r), t16, &mut w, &mut qr);
            let same = got.0.to_bits() == scale[r].to_bits()
                && got.1.to_bits() == c[r].to_bits()
                && got.2.to_bits() == sq[r].to_bits()
                && qr
                    .iter()
                    .zip(&codes[r * cols..(r + 1) * cols])
                    .all(|(a, &b)| a.cast_unsigned() == b);
            if !same {
                return Err(format!(
                    "head file row {r} differs from this model quantised by this build"
                ));
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let q = Codes::Mapped {
            map,
            at: at_codes,
            len: rows * cols,
        };
        #[cfg(target_arch = "wasm32")]
        let q = Codes::Owned(map[at_codes..].iter().map(|v| v.cast_signed()).collect());
        Ok(Self {
            cols,
            q,
            scale,
            c,
            sq,
        })
    }
}

/// A head file's magic. Layout, little-endian: the magic (8 bytes); version `1`, rows, cols
/// and a zero (`u32` each); the model's [`fingerprint`] (`u64`); then every row's scale, then
/// every row's bound coefficient `c`, then every row's `s ‖q‖₂` (three `f32` arrays of `rows`);
/// zeros to the next 4,096-byte boundary; then the codes, `rows x cols` `i8`, row-major.
pub const HEAD_FILE: [u8; 8] = *b"MNVHEAD\0";

/// The header's bytes.
const HEADER: usize = 32;

/// The codes start on a page boundary.
const PAGE: usize = 4096;

/// Every this many rows is quantised again when a head file is opened (2,004 of 128,256 rows:
/// about 10 MB of the head read).
const SAMPLE: usize = 64;

fn head_file_header(rows: usize, cols: usize, fp: u64) -> Vec<u8> {
    let mut h = HEAD_FILE.to_vec();
    for v in [1, rows, cols, 0] {
        h.extend_from_slice(&u32::try_from(v).unwrap_or(u32::MAX).to_le_bytes());
    }
    h.extend_from_slice(&fp.to_le_bytes());
    h
}

/// FNV-1a over the head's shape and the bytes of every [`SAMPLE`]th row: which model a head
/// file belongs to.
fn fingerprint(m: &Model) -> u64 {
    let (rows, cols) = (m.cfg.vocab, m.cfg.embd);
    let mut f = 0xcbf2_9ce4_8422_2325_u64;
    let mut eat = |b: &[u8]| {
        for &x in b {
            f = (f ^ u64::from(x)).wrapping_mul(0x0100_0000_01b3);
        }
    };
    eat(&(rows as u64).to_le_bytes());
    eat(&(cols as u64).to_le_bytes());
    for r in (0..rows).step_by(SAMPLE) {
        eat(m.f16_row(m.head, cols, r));
    }
    f
}

/// Map a head file read-only, asking the OS to read it ahead.
#[cfg(not(target_arch = "wasm32"))]
#[allow(unsafe_code)]
fn map_file(f: &std::fs::File) -> std::io::Result<memmap2::Mmap> {
    // SAFETY: a read-only map of a file no one modifies while it is mapped (a head file is an
    // immutable artifact, as the model file is); every access is bounds-checked through slices.
    let m = unsafe { memmap2::Mmap::map(f)? };
    #[cfg(unix)]
    {
        let _ = m.advise(memmap2::Advice::WillNeed);
    }
    Ok(m)
}

/// One row: `w` (scratch) its `f32` values, `qr` its codes. Returns its scale, its bound
/// coefficient and `s ‖q‖₂`.
fn quantise_row(bytes: &[u8], t16: &[f32], w: &mut [f32], qr: &mut [i8]) -> (f32, f32, f32) {
    for (v, x) in w.iter_mut().zip(bytes.chunks_exact(2)) {
        *v = t16[usize::from(u16::from_le_bytes([x[0], x[1]]))];
    }
    // Fixed-width lanes, so the folds vectorise on every target (a strict-order fold cannot). The
    // norms' summation order is not part of the bound: its 1e-4 slack covers any.
    let mut m8 = [0f32; W];
    let mut bad = [0u32; W];
    let (wc, wt) = w.as_chunks::<W>();
    for c in wc {
        for ((m, b), v) in m8.iter_mut().zip(&mut bad).zip(c) {
            let a = v.abs();
            *m = if a > *m { a } else { *m };
            *b |= u32::from(!a.is_finite());
        }
    }
    let mut amax = m8.iter().fold(0f32, |a, &v| a.max(v));
    let mut finite = bad.iter().all(|&b| b == 0);
    for v in wt {
        amax = amax.max(v.abs());
        finite &= v.is_finite();
    }
    if !finite {
        // A non-finite weight: the row can never be ruled out.
        return (0.0, f32::INFINITY, f32::INFINITY);
    }
    let sc = amax / 127.0;
    if sc <= 0.0 {
        qr.fill(0);
        return (0.0, 0.0, 0.0);
    }
    let (mut e8, mut w8, mut q8) = ([0f64; W], [0f64; W], [0f64; W]);
    let mut lane = |q: &mut i8, v: f32, l: usize| {
        // Round to nearest, ties to even, by the magic number (`round_ties_even` is a libm call
        // on a target without SSE4.1); `|v / sc| <= 127.00001`.
        // Its low mantissa byte is the code.
        let m = v / sc + MAGIC;
        *q = m.to_bits().to_le_bytes()[0].cast_signed();
        let k = f64::from(m - MAGIC);
        let v = f64::from(v);
        let e = v - f64::from(sc) * k;
        e8[l] += e * e;
        w8[l] += v * v;
        q8[l] += k * k;
    };
    let (qc, qt) = qr.as_chunks_mut::<W>();
    for (q, c) in qc.iter_mut().zip(wc) {
        for (l, (q, &v)) in q.iter_mut().zip(c).enumerate() {
            lane(q, v, l);
        }
    }
    for (l, (q, &v)) in qt.iter_mut().zip(wt).enumerate() {
        lane(q, v, l);
    }
    let (e2, w2, q2) = (
        e8.iter().sum::<f64>(),
        w8.iter().sum::<f64>(),
        q8.iter().sum::<f64>(),
    );
    let sq = f64::from(sc) * q2.sqrt();
    (
        sc,
        round_up((e2.sqrt() + 1e-5 * w2.sqrt() + 1e-5 * sq) * (1.0 + 1e-4)),
        round_up(sq * (1.0 + 1e-4)),
    )
}

/// Whether the 16-bit screen runs: unless `MINERVA_SCREEN=f32` (the float screen, for the A/B).
fn float_screen_off() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MINERVA_SCREEN").as_deref() != Ok("f32"))
}

/// A screen's per-session scratch, allocated once.
#[derive(Debug, Default)]
pub struct Scratch {
    dots: Vec<f32>,
    act: simd::ActI16,
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
            act: simd::ActI16::default(),
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
    let int = float_screen_off().then(|| sc.act.set(x)).flatten();
    // `‖e‖₂`, with room for the `f64` roundings in it and in `t·a`.
    let en = int.map_or(0.0, |(_, e)| round_up(e * (1.0 + 1e-4) + 1e-12 * x2.sqrt()));
    let act = &sc.act;

    let lo_max = sc
        .dots
        .par_chunks_mut(CHUNK)
        .zip(sc.hi.par_chunks_mut(CHUNK))
        .zip(h8.q.as_slice().par_chunks(CHUNK * cols))
        .enumerate()
        .map(|(ci, ((dots, hi), q))| {
            if let Some((t, _)) = int {
                let mut d = [0i64; CHUNK];
                let d = &mut d[..dots.len()];
                simd::i8i16_matvec(isa, q, act, d);
                for (o, &d) in dots.iter_mut().zip(d.iter()) {
                    // `|D| < 2^53`: the conversion is exact.
                    #[expect(
                        clippy::cast_precision_loss,
                        clippy::cast_possible_truncation,
                        reason = "exact below 2^53; one rounding to f32, in the bound"
                    )]
                    {
                        *o = (d as f64 * t) as f32;
                    }
                }
            } else {
                simd::i8f_matvec(isa, q, x, dots);
            }
            let base = ci * CHUNK;
            let mut lo_max = f32::NEG_INFINITY;
            for (j, (&d, h)) in dots.iter().zip(hi.iter_mut()).enumerate() {
                let r = base + j;
                let av = d * h8.scale[r];
                let b = (h8.c[r] * xn + h8.sq[r] * en) * 1.0001 + 1e-6 * av.abs();
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

/// A model's screen: built on first use, or attached from a head file.
#[derive(Debug, Default)]
pub struct Lazy {
    h: OnceLock<HeadQ8>,
    sidecar: OnceLock<()>,
}

impl Lazy {
    /// Use `h` (from [`HeadQ8::open`]) as the screen. `false` when a screen is already in place.
    pub fn attach(&self, h: HeadQ8) -> bool {
        let ok = self.h.set(h).is_ok();
        if ok {
            let _ = self.sidecar.set(());
        }
        ok
    }

    /// Where the screen came from: `"sidecar"` (a head file), `"built"`, or `"none"` yet.
    #[must_use]
    pub fn source(&self) -> &'static str {
        match (self.h.get(), self.sidecar.get()) {
            (None, _) => "none",
            (Some(_), Some(())) => "sidecar",
            (Some(_), None) => "built",
        }
    }

    /// The screen of `m`, building it the first time.
    pub fn get(&self, m: &Model) -> &HeadQ8 {
        self.h.get_or_init(|| {
            let t = std::time::Instant::now();
            let h = HeadQ8::build(m);
            let t = t.elapsed();
            if std::env::var_os("MINERVA_PROF").is_some() {
                let f =
                    h.q.as_slice()
                        .iter()
                        .fold(0xcbf2_9ce4_8422_2325_u64, |f, &q| {
                            (f ^ u64::from(q.cast_unsigned())).wrapping_mul(0x0100_0000_01b3)
                        });
                eprintln!("head build: {:.3}s codes-fnv {f:016x}", t.as_secs_f64());
            }
            h
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OURS: &str = "F:/coding/bitnet-golden/ours-i2_s.gguf";

    /// A head file written for the real model opens to the same screen; one changed byte in a
    /// checked row, or the fingerprint, is refused.
    #[test]
    #[ignore = "needs F:/coding/bitnet-golden/ours-i2_s.gguf"]
    fn a_head_file_round_trips_and_a_changed_one_is_refused() {
        let m = Model::load(std::path::Path::new(OURS)).expect("model");
        let built = HeadQ8::build(&m);
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
        let path = dir.join("test-head.mnvh");
        built.write(&m, &path).expect("write");
        let opened = HeadQ8::open(&m, &path).expect("open");
        assert!(opened.q.as_slice() == built.q.as_slice());
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&opened.scale), bits(&built.scale));
        assert_eq!(bits(&opened.c), bits(&built.c));
        assert_eq!(bits(&opened.sq), bits(&built.sq));
        drop(opened);
        let bytes = std::fs::read(&path).expect("read");
        let at_codes = (HEADER + 12 * m.cfg.vocab).next_multiple_of(PAGE);
        for at in [24, HEADER, at_codes + SAMPLE * m.cfg.embd + 7] {
            let mut bad = bytes.clone();
            bad[at] ^= 1;
            let p = dir.join("test-head-bad.mnvh");
            std::fs::write(&p, &bad).expect("write bad");
            assert!(
                HeadQ8::open(&m, &p).is_err(),
                "byte {at} changed, still opened"
            );
            let _ = std::fs::remove_file(p);
        }
        let _ = std::fs::remove_file(path);
    }
}
