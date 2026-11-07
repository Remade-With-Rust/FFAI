//! The neck's `cat(upsample_nearest2d(a, 2x), b)` as ONE pass into ONE buffer.
//!
//! # Why (measured)
//!
//! candle's `upsample_nearest2d` is the generic `interpolate2d` map — per
//! output element it recomputes the source index from strides — and then
//! `cat` copies both the upsampled tensor and `b` again into the result.
//! callgrind: 15.0 M instructions for the two upsamples (1.23 M elements,
//! 12 per element to copy a float) plus the concatenation's copies.
//!
//! Nearest 2x is `dst[y][x] = src[y / 2][x / 2]`: each source row is written
//! once with every element doubled, then that output row is duplicated with
//! one `memcpy`. `b`'s planes are one `memcpy` each into the tail of the same
//! buffer. Pure copies — there is no arithmetic to reorder — so the result is
//! byte-identical to candle's by construction, and the test asserts it.

use candle_core::{Result, Tensor};

/// `cat([upsample_nearest2d(a, 2h, 2w), b], dim 1)` for `a: (1, ca, h, w)`,
/// `b: (1, cb, 2h, 2w)`.
pub fn upsample2x_cat(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (_, ca, h, w) = a.dims4()?;
    let (_, cb, hb, wb) = b.dims4()?;
    if hb != 2 * h || wb != 2 * w {
        candle_core::bail!("upsample2x_cat: b is {hb}x{wb}, expected {}x{}", 2 * h, 2 * w);
    }
    let (oh, ow) = (2 * h, 2 * w);
    crate::cpuop::SliceOp2::new("ffai-upsample2x-cat", move |xa, _, xb, _| {
        let plane = oh * ow;
        let mut out: Vec<f32> = Vec::with_capacity((ca + cb) * plane);
        let spare = &mut out.spare_capacity_mut()[..(ca + cb) * plane];
        // SAFETY: `f32` is plain data; every element of `spare` is written
        // below (each of `ca` planes fully, then `cb` planes by `memcpy`)
        // before `set_len`. Writing through `MaybeUninit` first would be the
        // same bytes with a second pass of `assume_init`.
        #[allow(unsafe_code)]
        let dst: &mut [f32] =
            unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), spare.len()) };
        for (c, dplane) in dst[..ca * plane].chunks_exact_mut(plane).enumerate() {
            let splane = &xa[c * h * w..(c + 1) * h * w];
            for (sy, drows) in dplane.chunks_exact_mut(2 * ow).enumerate() {
                let (r0, r1) = drows.split_at_mut(ow);
                let srow = &splane[sy * w..(sy + 1) * w];
                for (d, &s) in r0.chunks_exact_mut(2).zip(srow) {
                    d[0] = s;
                    d[1] = s;
                }
                r1.copy_from_slice(r0);
            }
        }
        dst[ca * plane..].copy_from_slice(&xb[..cb * plane]);
        // SAFETY: all `(ca + cb) * plane` elements written above.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len((ca + cb) * plane);
        }
        Ok((out, (1, ca + cb, oh, ow).into()))
    })
    .run(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn matches_candle_bit_for_bit() {
        let dev = Device::Cpu;
        for &(ca, cb, h, w) in &[(256, 128, 20, 20), (3, 2, 5, 7), (1, 1, 1, 1)] {
            let a = Tensor::randn(0f32, 1.0, (1, ca, h, w), &dev).unwrap();
            let b = Tensor::randn(0f32, 1.0, (1, cb, 2 * h, 2 * w), &dev).unwrap();
            let want = Tensor::cat(&[&a.upsample_nearest2d(2 * h, 2 * w).unwrap(), &b], 1).unwrap();
            let got = upsample2x_cat(&a, &b).unwrap();
            assert_eq!(got.dims(), want.dims());
            let (g, wv) = (
                got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            );
            for (i, (x, y)) in g.iter().zip(&wv).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "ca={ca} cb={cb} {h}x{w} element {i}");
            }
        }
    }
}
