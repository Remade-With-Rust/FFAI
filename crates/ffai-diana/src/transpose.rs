//! A blocked transpose of the last two dimensions.
//!
//! # Why this exists when the win is 0.3 %
//!
//! Candle's `t().contiguous()` measured **4.3x slower than a naive blocked
//! loop** on the shapes this graph uses. In the one place the graph calls it
//! — twice inside the attention block — that is worth 0.23 % of the
//! pipeline, which is far below anything this box can resolve and would
//! normally be pruned on arithmetic.
//!
//! It is written anyway, on principle: **being 4.3x slower than a loop at
//! anything is a defect, whether or not it is currently on a hot path.**
//! Fifty of those is 1.005^50 = 1.28x and not one is individually
//! measurable, which is precisely the class of gain
//! [`crate::smallgains`] exists to accumulate.
//!
//! # Why it is byte-identical, not approximately equal
//!
//! A transpose is a PERMUTATION. It moves floats; it does not compute with
//! them. So unlike every kernel in this crate the gate is not a tolerance —
//! it is `assert_eq!` on the bytes, and anything less would mean a bug.
//!
//! # The mechanism
//!
//! Tile 32x32 so a tile's source rows and destination columns both stay in
//! L1 while it is copied, and run the INNER loop over the destination's
//! contiguous axis. That last detail is the whole difference: with the
//! column index innermost the destination writes stride by `rows` and touch
//! a fresh cache line every iteration. The first blocked version written
//! here had it backwards and measured only 1.1x better than candle — same
//! defect, same 4x.

use candle_core::{Result, Tensor};

/// Tile edge. 32 f32 is 128 bytes — two cache lines per row, and a 32x32
/// tile is 4 KiB, comfortably inside L1 alongside its destination.
const BLOCK: usize = 32;

/// Blocked transpose of `[.., rows, cols]` into `[.., cols, rows]`.
///
/// Operates on the flat slice for one 2-D plane. Split out so the tests can
/// drive it directly and so the batch loop stays legible.
pub(crate) fn transpose_plane(src: &[f32], rows: usize, cols: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);
    for r0 in (0..rows).step_by(BLOCK) {
        let r1 = (r0 + BLOCK).min(rows);
        for c0 in (0..cols).step_by(BLOCK) {
            let c1 = (c0 + BLOCK).min(cols);
            // Inner loop over `r`: `dst[c * rows + r]` is then contiguous.
            for c in c0..c1 {
                let out = &mut dst[c * rows + r0..c * rows + r1];
                for (i, o) in out.iter_mut().enumerate() {
                    *o = src[(r0 + i) * cols + c];
                }
            }
        }
    }
}

/// `x.transpose(-2, -1).contiguous()`, blocked.
///
/// Falls back to candle's own path for anything this does not handle, so a
/// shape it was not designed for is slow rather than wrong.
pub fn transpose_last2(x: &Tensor) -> Result<Tensor> {
    use candle_core::D;
    let dims = x.dims().to_vec();
    // Reachable from the one small-gains switch, per that register's rule 3:
    // a gain too small to measure alone must be measurable as part of the
    // accumulated stack.
    if crate::smallgains::disabled() || dims.len() < 2 || x.dtype() != candle_core::DType::F32 {
        return x.transpose(D::Minus2, D::Minus1)?.contiguous();
    }
    let (rows, cols) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let planes: usize = dims[..dims.len() - 2].iter().product();
    let src = x.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0f32; src.len()];
    let plane = rows * cols;
    for p in 0..planes {
        transpose_plane(
            &src[p * plane..(p + 1) * plane],
            rows,
            cols,
            &mut out[p * plane..(p + 1) * plane],
        );
    }
    let mut shape = dims;
    let n = shape.len();
    shape.swap(n - 2, n - 1);
    Tensor::from_vec(out, shape, x.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, D};

    /// BYTE-IDENTICAL to candle, not merely close.
    ///
    /// A transpose permutes floats rather than computing with them, so any
    /// difference at all is a bug — there is no reassociation to forgive.
    #[test]
    fn matches_candle_exactly() {
        let dev = Device::Cpu;
        for &(b, h, r, c) in &[
            (1usize, 2usize, 400usize, 32usize), // the attention shape
            (1, 2, 32, 400),                     // and its transpose
            (1, 1, 33, 65),                      // non-multiples of the tile
            (1, 1, 1, 7),
            (2, 3, 64, 64),
        ] {
            let n = b * h * r * c;
            let src: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
            let x = Tensor::from_vec(src, (b, h, r, c), &dev).unwrap();
            let want = x.transpose(D::Minus2, D::Minus1).unwrap().contiguous().unwrap();
            let got = transpose_last2(&x).unwrap();
            assert_eq!(got.dims(), want.dims(), "shape for {b}x{h}x{r}x{c}");
            assert_eq!(
                got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                "bytes differ for {b}x{h}x{r}x{c}"
            );
        }
    }

    /// Non-f32 and rank-1 fall back rather than misbehaving.
    #[test]
    fn falls_back_for_shapes_it_does_not_handle() {
        let dev = Device::Cpu;
        let x = Tensor::zeros((2usize, 3usize), DType::F64, &dev).unwrap();
        let got = transpose_last2(&x).unwrap();
        assert_eq!(got.dims(), &[3, 2]);
    }
}
