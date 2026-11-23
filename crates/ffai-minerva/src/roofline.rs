//! The roofline: what one decoded token must stream from memory, and how fast this machine can
//! stream it. `tok/s <= read bandwidth / bytes per token` bounds every kernel built after it.
//!
//! Two instruments, both deterministic in what they count:
//!
//! - [`census`] reads a `GGUF`'s tensor table and sorts every byte by how often a greedy decode
//!   step reads it: the ternary projections and the head once per token, the embedding one row
//!   per token, the KV cache once per cached position per token.
//! - [`bandwidth`] streams a buffer far larger than the last-level cache from N threads and
//!   times it, interleaved (ABBA) with a NULL arm that runs the identical harness over zero
//!   bytes, so thread spawn and join are measured rather than assumed away.

use std::hint::black_box;
use std::time::Instant;

use rusty_ternary::gguf::{Gguf, TYPE_F16, TYPE_F32, TYPE_I2_S};

/// Where a model's bytes go per decoded token.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ByteCensus {
    /// `I2_S` projection bytes (codes plus each tensor's scale block): read once per token.
    pub ternary: u64,
    /// Number of `I2_S` tensors.
    pub ternary_tensors: usize,
    /// The output head's bytes: read once per token for the logits.
    pub head: u64,
    /// The head's element type (`TYPE_F16` / `TYPE_F32`).
    pub head_type: u32,
    /// Rows and columns of the head (`vocab`, `hidden`).
    pub head_shape: (u64, u64),
    /// Whether the head is the token embedding (tied: no `output.weight`).
    pub tied: bool,
    /// One embedding row: what the input lookup reads per token.
    pub embed_row: u64,
    /// `f32` tensors (the norms): read once per token.
    pub norms: u64,
    /// Bytes of tensors that fit none of the above (should be 0 for `BitNet` b1.58).
    pub other: u64,
    /// Number of transformer blocks.
    pub layers: u64,
    /// The KV width per layer (`hidden / heads * kv_heads`).
    pub kv_dim: u64,
}

impl ByteCensus {
    /// Bytes of the KV cache read per token per cached position, at `elem` bytes an element
    /// (2 for `f16`, which is bitnet.cpp's default cache type): `K` and `V`, every layer.
    #[must_use]
    pub const fn kv_per_position(&self, elem: u64) -> u64 {
        2 * self.kv_dim * self.layers * elem
    }

    /// Bytes one greedy decode step streams with `positions` tokens already cached, with the
    /// head read at `head_bytes` (pass [`Self::head`] for the plain head).
    #[must_use]
    pub const fn per_token(&self, head_bytes: u64, positions: u64, kv_elem: u64) -> u64 {
        self.ternary
            + head_bytes
            + self.embed_row
            + self.norms
            + self.other
            + positions * self.kv_per_position(kv_elem)
    }

    /// The head as an int8 screen copy reads it: one byte an element plus one `f32` bound a row
    /// (the survivors' exact recompute is 1-12 rows, a few tens of KB, ignored here).
    #[must_use]
    pub const fn head_q8(&self) -> u64 {
        let (rows, cols) = self.head_shape;
        rows * cols + rows * 4
    }
}

fn meta_u64(g: &Gguf<'_>, key: &str) -> Result<u64, String> {
    g.get(key)
        .and_then(rusty_ternary::gguf::Value::as_u64)
        .ok_or_else(|| format!("missing metadata `{key}`"))
}

/// Sort a parsed file's tensor bytes by how often a decode step reads them.
///
/// # Errors
/// When the file lacks the architecture metadata or holds a tensor type a ternary model does
/// not use.
pub fn census(g: &Gguf<'_>) -> Result<ByteCensus, String> {
    let arch = g
        .get("general.architecture")
        .and_then(rusty_ternary::gguf::Value::as_str)
        .ok_or("missing `general.architecture`")?
        .to_owned();
    let layers = meta_u64(g, &format!("{arch}.block_count"))?;
    let hidden = meta_u64(g, &format!("{arch}.embedding_length"))?;
    let heads = meta_u64(g, &format!("{arch}.attention.head_count"))?;
    let kv_heads = meta_u64(g, &format!("{arch}.attention.head_count_kv")).unwrap_or(heads);
    if heads == 0 || hidden % heads != 0 {
        return Err(format!("hidden {hidden} not divisible by heads {heads}"));
    }

    let mut c = ByteCensus {
        layers,
        kv_dim: hidden / heads * kv_heads,
        ..ByteCensus::default()
    };
    let has_output = g.tensor("output.weight").is_some();
    c.tied = !has_output;
    for t in &g.tensors {
        let len = t
            .byte_len()
            .ok_or_else(|| format!("{}: type {} has no size rule", t.name, t.ty))?
            as u64;
        let is_head = t.name == "output.weight" || (!has_output && t.name == "token_embd.weight");
        if t.name == "token_embd.weight" {
            let cols = *t.dims.first().ok_or("token_embd has no dims")?;
            c.embed_row = cols * if t.ty == TYPE_F16 { 2 } else { 4 };
        }
        if is_head {
            c.head = len;
            c.head_type = t.ty;
            c.head_shape = (t.dims.get(1).copied().unwrap_or(1), t.dims[0]);
        } else if t.name == "token_embd.weight" {
            // Untied: only one row per token is read, counted in `embed_row`.
        } else if t.ty == TYPE_I2_S {
            c.ternary += len;
            c.ternary_tensors += 1;
        } else if t.ty == TYPE_F32 {
            c.norms += len;
        } else {
            c.other += len;
        }
    }
    if c.head == 0 {
        return Err("no head tensor (output.weight or token_embd.weight)".into());
    }
    Ok(c)
}

/// One timed pass: `threads` scoped threads each sum their share of `buf`. Returns seconds,
/// measured with `Instant` (`QueryPerformanceCounter` on Windows: sub-microsecond, no tick
/// lattice), spawn and join included — which is what the null arm prices.
fn pass(buf: &[u64], threads: usize) -> f64 {
    let chunk = buf.len().div_ceil(threads.max(1)).max(1);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|i| {
                let lo = (i * chunk).min(buf.len());
                let hi = ((i + 1) * chunk).min(buf.len());
                let part = &buf[lo..hi];
                s.spawn(move || black_box(sum(black_box(part))))
            })
            .collect();
        for h in handles {
            black_box(h.join().unwrap_or(0));
        }
    });
    t0.elapsed().as_secs_f64()
}

/// Eight independent lanes so the loads are not serialised behind one add chain; the adds are
/// `paddq` on baseline `x86-64`, far below the load rate, so this is a READ test.
fn sum(part: &[u64]) -> u64 {
    let mut acc = [0u64; 8];
    let blocks = part.chunks_exact(8);
    let tail = blocks.remainder();
    for b in blocks {
        for (a, &x) in acc.iter_mut().zip(b) {
            *a = a.wrapping_add(x);
        }
    }
    acc.iter().chain(tail).fold(0u64, |s, &x| s.wrapping_add(x))
}

/// The result of one thread count.
#[derive(Clone, Debug)]
pub struct BandwidthRow {
    /// Threads used.
    pub threads: usize,
    /// Read-arm seconds per pass, in run order.
    pub read_s: Vec<f64>,
    /// Null-arm seconds per pass (same harness, zero bytes), in run order.
    pub null_s: Vec<f64>,
    /// Bytes read per pass.
    pub bytes: u64,
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    if s.is_empty() {
        f64::NAN
    } else {
        s[s.len() / 2]
    }
}

fn min(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

impl BandwidthRow {
    /// Best pass, null-arm minimum subtracted: GB/s (1e9 bytes).
    #[must_use]
    pub fn best_gbs(&self) -> f64 {
        self.bytes as f64 / (min(&self.read_s) - min(&self.null_s)) / 1e9
    }
    /// Median pass, null-arm median subtracted: GB/s.
    #[must_use]
    pub fn median_gbs(&self) -> f64 {
        self.bytes as f64 / (median(&self.read_s) - median(&self.null_s)) / 1e9
    }
    /// Spread of the read arm: max / min.
    #[must_use]
    pub fn spread(&self) -> f64 {
        self.read_s.iter().copied().fold(0.0, f64::max) / min(&self.read_s)
    }
    /// The null arm's minimum as a share of the read arm's minimum.
    #[must_use]
    pub fn null_share(&self) -> f64 {
        min(&self.null_s) / min(&self.read_s)
    }
    /// The clock's resolution as seen by this row: the smallest non-zero gap between any two
    /// readings of either arm.
    #[must_use]
    pub fn quantum(&self) -> f64 {
        let mut all: Vec<f64> = self.read_s.iter().chain(&self.null_s).copied().collect();
        all.sort_by(f64::total_cmp);
        all.windows(2)
            .map(|w| w[1] - w[0])
            .filter(|&d| d > 0.0)
            .fold(f64::INFINITY, f64::min)
    }
}

/// Stream `buf` from each thread count in `threads`, `reps` passes per arm, the read and null
/// arms alternating ABBA (the leading arm swaps every round). One warm-up pass per count is
/// discarded.
#[must_use]
pub fn bandwidth(buf: &[u64], threads: &[usize], reps: usize) -> Vec<BandwidthRow> {
    let empty: &[u64] = &[];
    threads
        .iter()
        .map(|&t| {
            let _ = pass(buf, t);
            let mut row = BandwidthRow {
                threads: t,
                read_s: Vec::new(),
                null_s: Vec::new(),
                bytes: (buf.len() * 8) as u64,
            };
            for r in 0..reps {
                if r % 2 == 0 {
                    row.read_s.push(pass(buf, t));
                    row.null_s.push(pass(empty, t));
                } else {
                    row.null_s.push(pass(empty, t));
                    row.read_s.push(pass(buf, t));
                }
            }
            row
        })
        .collect()
}

/// A buffer of `mib` MiB with every page written (so the read arm never faults), filled with
/// values the summing loop cannot predict away.
#[must_use]
pub fn stream_buffer(mib: usize) -> Vec<u64> {
    let n = mib * (1 << 20) / 8;
    (0..n as u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_covers_every_element_including_the_tail() {
        let v: Vec<u64> = (1..=21).collect();
        assert_eq!(sum(&v), 231);
        assert_eq!(sum(&[]), 0);
    }

    #[test]
    fn per_token_arithmetic() {
        let c = ByteCensus {
            ternary: 1000,
            head: 500,
            head_shape: (10, 20),
            embed_row: 40,
            norms: 8,
            layers: 3,
            kv_dim: 4,
            ..ByteCensus::default()
        };
        assert_eq!(c.kv_per_position(2), 2 * 4 * 3 * 2);
        assert_eq!(c.per_token(c.head, 0, 2), 1548);
        assert_eq!(c.per_token(c.head, 10, 2), 1548 + 480);
        assert_eq!(c.head_q8(), 200 + 40);
    }

    /// The real model, when this machine has it.
    #[test]
    #[ignore = "needs F:/coding/bitnet-golden/ours-i2_s.gguf"]
    fn census_of_ours() {
        let bytes = std::fs::read("F:/coding/bitnet-golden/ours-i2_s.gguf").expect("model file");
        let g = Gguf::parse(&bytes).expect("parse");
        let c = census(&g).expect("census");
        assert!(c.tied, "2B-4T ties its head to the embedding");
        assert_eq!(c.other, 0, "{c:?}");
        assert_eq!(c.ternary_tensors, 7 * c.layers as usize);
        assert_eq!(c.head_shape, (128_256, 2560));
    }
}
