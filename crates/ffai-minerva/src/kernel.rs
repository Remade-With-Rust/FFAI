//! The I2_S GEMV Minerva runs: rows split across the thread pool, each chunk through the
//! dispatched twin ([`crate::simd::ternary_gemv`]), whose scalar arm is the oracle itself
//! (`rusty_ternary::kernel::gemv_codes`). Exact integers, so neither the split nor the tier
//! can change a result.

use rayon::prelude::*;

use crate::simd::{self, Isa};

/// Rows a thread takes at a time.
const CHUNK: usize = 64;

/// `y[r] = Σ_j W[r, j] · a[j]` over the codes of a `y.len() x a.len()` projection; `a_sum` is
/// [`simd::act_sum`]`(a)`.
///
/// # Panics
/// When `codes` is not `y.len()` whole rows of `a.len()` weights.
pub fn gemv_par(isa: Isa, codes: &[u8], a: &[i8], a_sum: i32, y: &mut [i32]) {
    let rpc = a.len() / 4;
    assert_eq!(codes.len(), y.len() * rpc);
    y.par_chunks_mut(CHUNK)
        .zip(codes.par_chunks(CHUNK * rpc))
        .for_each(|(ys, cs)| {
            simd::ternary_gemv(isa, cs, a, a_sum, ys);
        });
}

/// Prefill's product over `m` tokens: `y[r·m + t] = Σ_j W[r, j] · a[t·cols + j]` (`[rows, m]`),
/// rows split across the pool, each chunk through [`simd::ternary_gemm`]: the weights are read
/// once for all `m` tokens.
///
/// # Panics
/// When the shapes disagree.
pub fn gemm_par(
    isa: Isa,
    codes: &[u8],
    cols: usize,
    m: usize,
    a: &[i8],
    a_sums: &[i32],
    y: &mut [i32],
) {
    let rpc = cols / 4;
    assert_eq!(y.len() * rpc, codes.len() * m);
    y.par_chunks_mut(CHUNK * m)
        .zip(codes.par_chunks(CHUNK * rpc))
        .for_each(|(ys, cs)| simd::ternary_gemm(isa, cs, cols, m, a, a_sums, ys));
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_ternary::i2s::pack;
    use rusty_ternary::kernel::gemv;

    #[test]
    fn the_threaded_gemv_equals_the_whole_tensor_oracle_at_every_tier() {
        let mut seed = 9u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            seed >> 33
        };
        for (rows, cols) in [
            (1usize, 128usize),
            (63, 256),
            (65, 640),
            (200, 2560),
            (130, 6912),
        ] {
            let w: Vec<i8> = (0..rows * cols).map(|_| (next() % 3) as i8 - 1).collect();
            let a: Vec<i8> = (0..cols).map(|_| (next() % 256) as i8).collect();
            let p = pack(&w, rows, cols, 0.75).unwrap();
            let mut want = vec![0i32; rows];
            gemv(&p, rows, cols, &a, &mut want).unwrap();
            for isa in Isa::all() {
                let mut got = vec![0i32; rows];
                gemv_par(isa, &p[..rows * cols / 4], &a, simd::act_sum(&a), &mut got);
                assert_eq!(got, want, "{} {rows}x{cols}", isa.name());
            }
        }
    }
}
