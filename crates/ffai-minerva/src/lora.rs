//! LoRA adapters: a low-rank update on top of the ternary base, read from a llama.cpp
//! adapter `GGUF` (what `convert_lora_to_gguf.py` writes and bitnet.cpp's `--lora` reads).
//!
//! Each adapted projection gains `y += s · B (A x)`, with `A` the `rank x cols` `lora_a` and `B`
//! the `rows x rank` `lora_b`, both `f32`, and `s = user_scale · alpha / rank` (or
//! `user_scale` when the file's alpha is zero), as bitnet.cpp's `llm_build_lora_mm` adds
//! it: onto the projection's finished output, `x` being the projection's `f32` input before
//! its 8-bit quantisation.
//!
//! The arithmetic is plain `f32` in one fixed order (every dot product is eight running sums
//! combined as one tree, then the tail), so it vectorises on every target and gives the same
//! bits on every device. It cannot equal ggml's order bit for bit, so the gate against
//! bitnet.cpp is the near-tie gate, as for the base model.

use rayon::prelude::*;
use rusty_ternary::gguf::{Gguf, TYPE_F32, Value};

use crate::model::{Model, Ternary};

/// The projections an adapter may update, in a layer's order.
const PROJ: [&str; 7] = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];

/// Which projection of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proj {
    /// Query.
    Q = 0,
    /// Key.
    K = 1,
    /// Value.
    V = 2,
    /// Attention output.
    O = 3,
    /// FFN gate.
    Gate = 4,
    /// FFN up.
    Up = 5,
    /// FFN down.
    Down = 6,
}

fn ternary(m: &Model, layer: usize, p: usize) -> &Ternary {
    let l = &m.layers[layer];
    [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down][p]
}

/// One projection's update.
#[derive(Clone, Debug, PartialEq)]
pub struct Pair {
    rank: usize,
    cols: usize,
    rows: usize,
    /// `rank x cols`, row-major.
    a: Vec<f32>,
    /// `rows x rank`, row-major.
    b: Vec<f32>,
    scale: f32,
}

/// Ranks up to this many keep `A x` on the stack.
const STACK_RANK: usize = 128;

/// `B` rows a task takes.
const B_CHUNK: usize = 256;

impl Pair {
    /// `y[..rows] += scale · B (A x)` for one token (`x` holds `cols`).
    pub(crate) fn apply(&self, x: &[f32], y: &mut [f32]) {
        let mut stack = [0f32; STACK_RANK];
        let mut heap = Vec::new();
        let t: &mut [f32] = if self.rank <= STACK_RANK {
            &mut stack[..self.rank]
        } else {
            heap.resize(self.rank, 0.0);
            &mut heap
        };
        let x = &x[..self.cols];
        t.par_iter_mut()
            .zip(self.a.par_chunks_exact(self.cols))
            .for_each(|(t, a)| *t = dot(a, x));
        let (t, rank, scale) = (&*t, self.rank, self.scale);
        y[..self.rows]
            .par_chunks_mut(B_CHUNK)
            .zip(self.b.par_chunks(B_CHUNK * rank))
            .for_each(|(y, b)| {
                for (y, b) in y.iter_mut().zip(b.chunks_exact(rank)) {
                    *y += dot(b, t) * scale;
                }
            });
    }

    /// The rank.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// The factor on `B (A x)`.
    #[must_use]
    pub const fn scale(&self) -> f32 {
        self.scale
    }
}

/// `Σ a_i b_i` in one fixed order: eight running sums (lane `j` takes `i ≡ j mod 8`), combined
/// as `((s0+s4)+(s1+s5)) + ((s2+s6)+(s3+s7))`, then the tail in order. No fused multiply-add
/// (Rust never contracts), so every target computes the same bits.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let body = n / 8 * 8;
    let mut s = [0f32; 8];
    for (x, y) in a[..body].chunks_exact(8).zip(b[..body].chunks_exact(8)) {
        for j in 0..8 {
            s[j] += x[j] * y[j];
        }
    }
    let mut tail = 0f32;
    for (x, y) in a[body..n].iter().zip(&b[body..n]) {
        tail += x * y;
    }
    (((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]))) + tail
}

/// A loaded adapter: per layer, each projection's update or none.
#[derive(Clone, Debug, PartialEq)]
pub struct Adapter {
    layers: Vec<[Option<Pair>; 7]>,
    /// The file's `adapter.lora.alpha`.
    pub alpha: f32,
}

impl Adapter {
    /// Read an adapter file for `m`. `user_scale` multiplies the update (1.0 is the file's own
    /// strength, 0.0 turns it off).
    ///
    /// # Errors
    /// When the file cannot be read, is not a LoRA adapter `GGUF` for this architecture, names
    /// a tensor this engine does not adapt, or a pair's shapes do not fit the model.
    pub fn load(m: &Model, path: &std::path::Path, user_scale: f32) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        Self::from_bytes(m, &bytes, user_scale)
    }

    /// As [`Self::load`], from the file's bytes.
    ///
    /// # Errors
    /// As [`Self::load`].
    pub fn from_bytes(m: &Model, bytes: &[u8], user_scale: f32) -> Result<Self, String> {
        let g = Gguf::parse_lazy(bytes).map_err(|e| format!("not a readable GGUF: {e:?}"))?;
        let text = |k: &str| g.get(k).and_then(Value::as_str).unwrap_or("");
        if text("general.type") != "adapter" || text("adapter.type") != "lora" {
            return Err(
                "not a LoRA adapter (want general.type = adapter, adapter.type = lora)".into(),
            );
        }
        if text("general.architecture") != rusty_ternary::gguf::ARCHITECTURE {
            return Err(format!(
                "adapter for `{}`, not `{}`",
                text("general.architecture"),
                rusty_ternary::gguf::ARCHITECTURE
            ));
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "alpha is an f32 in the file"
        )]
        let alpha = match g.get("adapter.lora.alpha") {
            Some(Value::Float(v)) => *v as f32,
            Some(v) => v.as_u64().map_or(0.0, |v| v as f32),
            None => 0.0,
        };
        let mut layers: Vec<[Option<Pair>; 7]> =
            (0..m.cfg.layers).map(|_| Default::default()).collect();
        // (layer, projection) to its lora_a and lora_b: dims and values.
        type Half = Option<(Vec<u64>, Vec<f32>)>;
        let mut halves: std::collections::BTreeMap<(usize, usize), [Half; 2]> =
            std::collections::BTreeMap::new();
        for t in &g.tensors {
            let (base, half) = if let Some(b) = t.name.strip_suffix(".lora_a") {
                (b, 0)
            } else if let Some(b) = t.name.strip_suffix(".lora_b") {
                (b, 1)
            } else {
                return Err(format!("tensor `{}`: not a .lora_a or .lora_b", t.name));
            };
            let key = parse_name(base)
                .filter(|&(l, _)| l < m.cfg.layers)
                .ok_or_else(|| {
                    format!("tensor `{}`: not a projection this engine adapts", t.name)
                })?;
            if t.ty != TYPE_F32 || t.dims.len() != 2 {
                return Err(format!("tensor `{}`: want a 2-d f32 tensor", t.name));
            }
            let raw = g
                .bytes_of(t)
                .map_err(|e| format!("tensor `{}`: {e:?}", t.name))?;
            let v = raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let slot = &mut halves.entry(key).or_default()[half];
            if slot.replace((t.dims.clone(), v)).is_some() {
                return Err(format!("tensor `{}` appears twice", t.name));
            }
        }
        for ((l, p), [a, b]) in halves {
            let name = format!("blk.{l}.{}.weight", PROJ[p]);
            let (Some((da, a)), Some((db, b))) = (a, b) else {
                return Err(format!("`{name}`: lora_a or lora_b is missing"));
            };
            let w = ternary(m, l, p);
            // GGUF dims are innermost first: lora_a [cols, rank], lora_b [rank, rows].
            let as_usize = |d: u64| usize::try_from(d).unwrap_or(usize::MAX);
            let (cols, rank, rank_b, rows) = (
                as_usize(da[0]),
                as_usize(da[1]),
                as_usize(db[0]),
                as_usize(db[1]),
            );
            if cols != w.cols || rows != w.rows || rank != rank_b || rank == 0 {
                return Err(format!(
                    "`{name}`: lora_a {da:?} and lora_b {db:?} do not fit a {} x {} projection",
                    w.rows, w.cols
                ));
            }
            #[expect(clippy::cast_precision_loss, reason = "a rank, small")]
            let scale = if alpha == 0.0 {
                user_scale
            } else {
                user_scale * alpha / rank as f32
            };
            layers[l][p] = Some(Pair {
                rank,
                cols,
                rows,
                a,
                b,
                scale,
            });
        }
        Ok(Self { layers, alpha })
    }

    /// Layer `layer`'s update of `p`, if the adapter has one.
    #[must_use]
    pub fn get(&self, layer: usize, p: Proj) -> Option<&Pair> {
        self.layers.get(layer)?[p as usize].as_ref()
    }

    /// How many projections the adapter updates, and the largest rank.
    #[must_use]
    pub fn summary(&self) -> (usize, usize) {
        let pairs = self.layers.iter().flatten().flatten();
        (
            pairs.clone().count(),
            pairs.map(|p| p.rank).max().unwrap_or(0),
        )
    }

    /// Whether every update fits `m`'s projections (an adapter read for another model may not).
    #[must_use]
    pub fn fits(&self, m: &Model) -> bool {
        self.layers.len() == m.cfg.layers
            && self.layers.iter().enumerate().all(|(l, ps)| {
                ps.iter().enumerate().all(|(p, x)| {
                    x.as_ref().is_none_or(|x| {
                        let w = ternary(m, l, p);
                        x.rows == w.rows && x.cols == w.cols
                    })
                })
            })
    }
}

/// `blk.N.attn_q.weight` to `(N, index in PROJ)`.
fn parse_name(base: &str) -> Option<(usize, usize)> {
    let rest = base.strip_prefix("blk.")?.strip_suffix(".weight")?;
    let (n, proj) = rest.split_once('.')?;
    Some((n.parse().ok()?, PROJ.iter().position(|&p| p == proj)?))
}

/// A batch's update: token `k` reads `h[k·dim ..][..dim]` and adds to `out[k·stride ..]`.
pub(crate) fn apply_batch(
    p: &Pair,
    h: &[f32],
    dim: usize,
    out: &mut [f32],
    stride: usize,
    bm: usize,
) {
    out[..bm * stride]
        .par_chunks_mut(stride)
        .zip(h[..bm * dim].par_chunks(dim))
        .for_each(|(o, x)| p.apply(x, o));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_is_the_fixed_order() {
        let a: Vec<f32> = (0..21).map(|i| 1.0 + i as f32 * 0.37).collect();
        let b: Vec<f32> = (0..21).map(|i| 0.5 - i as f32 * 0.11).collect();
        let mut s = [0f32; 8];
        for i in 0..16 {
            s[i % 8] += a[i] * b[i];
        }
        let mut tail = 0f32;
        for i in 16..21 {
            tail += a[i] * b[i];
        }
        let want = (((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]))) + tail;
        assert_eq!(dot(&a, &b).to_bits(), want.to_bits());
    }

    #[test]
    fn names() {
        assert_eq!(parse_name("blk.12.ffn_down.weight"), Some((12, 6)));
        assert_eq!(parse_name("blk.0.attn_output.weight"), Some((0, 3)));
        assert_eq!(parse_name("output.weight"), None);
        assert_eq!(parse_name("blk.0.attn_norm.weight"), None);
    }

    #[test]
    fn apply_is_b_a_x() {
        let p = Pair {
            rank: 2,
            cols: 3,
            rows: 2,
            a: vec![1.0, 0.0, 2.0, 0.0, 1.0, 0.0],
            b: vec![1.0, 1.0, 0.0, 3.0],
            scale: 0.5,
        };
        let mut y = vec![10.0, 20.0];
        p.apply(&[1.0, 2.0, 3.0], &mut y);
        // A x = [7, 2]; B (A x) = [9, 6]; y += 0.5 · that.
        assert_eq!(y, vec![14.5, 23.0]);
    }
}
