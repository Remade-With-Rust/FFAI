//! The `I2_S` GEMV Minerva runs, split across threads by rows.
//!
//! The row product is `rusty_ternary`'s scalar definition. It is copied here only because that
//! crate does not yet expose a row range (`kernel::gemv` takes a whole tensor, scale tail
//! included); a test holds this copy `assert_eq!` to `rusty_ternary::kernel::gemv` on whole
//! tensors. It goes when the oracle's row-range entry point lands.

use rayon::prelude::*;
use rusty_ternary::i2s::{BLOCK, GROUP};

/// Rows a thread takes at a time.
const CHUNK: usize = 32;

/// One packed row (`cols / 4` bytes of codes) times `a`: `Σ trit · a`.
#[must_use]
pub fn dot_row(codes: &[u8], a: &[i8]) -> i32 {
    let mut sum = 0i32;
    for (bytes, acts) in codes.as_chunks::<GROUP>().0.iter().zip(a.as_chunks::<BLOCK>().0) {
        for (group, run) in acts.as_chunks::<GROUP>().0.iter().enumerate() {
            let shift = 6 - 2 * group;
            for (&byte, &v) in bytes.iter().zip(run) {
                sum += (i32::from((byte >> shift) & 3) - 1) * i32::from(v);
            }
        }
    }
    sum
}

/// `y[r] = Σ_j W[r, j] · a[j]` over the codes of a `y.len() x a.len()` projection, rows split
/// across the thread pool. Exact integers, so the split cannot change a result.
pub fn gemv_par(codes: &[u8], a: &[i8], y: &mut [i32]) {
    let rpc = a.len() / 4;
    assert_eq!(codes.len(), y.len() * rpc);
    y.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, ys)| {
        let base = ci * CHUNK;
        for (j, out) in ys.iter_mut().enumerate() {
            let r = base + j;
            *out = dot_row(&codes[r * rpc..(r + 1) * rpc], a);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_ternary::i2s::pack;
    use rusty_ternary::kernel::gemv;

    fn stream(n: usize, mut seed: u64, modulo: u64, offset: i64) -> Vec<i8> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                i8::try_from(i64::try_from((seed >> 33) % modulo).unwrap() + offset).unwrap()
            })
            .collect()
    }

    #[test]
    fn equals_the_oracle_on_whole_tensors() {
        for (rows, cols) in [(1usize, 128usize), (33, 256), (65, 640), (100, 2560), (7, 6912)] {
            let w = stream(rows * cols, rows as u64, 3, -1);
            let a = stream(cols, cols as u64, 256, -128);
            let p = pack(&w, rows, cols, 0.75).unwrap();
            let mut want = vec![0i32; rows];
            gemv(&p, rows, cols, &a, &mut want).unwrap();
            let mut got = vec![0i32; rows];
            gemv_par(&p[..rows * cols / 4], &a, &mut got);
            assert_eq!(got, want, "{rows}x{cols}");
        }
    }

    #[test]
    fn extremes_do_not_wrap() {
        let cols = 6912;
        let p = pack(&vec![1i8; cols], 1, cols, 1.0).unwrap();
        let mut y = [0i32];
        gemv_par(&p[..cols / 4], &vec![-128i8; cols], &mut y);
        assert_eq!(y[0], -128 * 6912);
    }
}
