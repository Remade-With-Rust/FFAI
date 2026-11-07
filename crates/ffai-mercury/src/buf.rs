//! Zero-copy reads of candle CPU tensors and write-before-read buffers.
//!
//! Two moves the instruction-veins campaign (`docs/plans/mercury-instruction-veins.md`)
//! makes everywhere, kept in one place so every site states the same contract:
//!
//! * [`with_f32`] borrows a contiguous f32 tensor's elements instead of
//!   `to_vec1()`-copying them — the copy was the whole cost of several
//!   "glue" sites.
//! * [`filled_f32`] allocates a buffer and hands it to a closure that writes
//!   EVERY element, so the `vec![0.0; n]` zero pass — a full discarded write
//!   over the buffer — never happens. It is the `ffai_core::fastops::ElemOp`
//!   shape, which is the only zero-fill-then-overwrite fix that measured a
//!   win there.

use ffai_core::candle::{CpuStorage, Result as CandleResult, Storage, Tensor};

/// Borrow a contiguous f32 CPU tensor's elements without copying them.
pub fn with_f32<R>(t: &Tensor, f: impl FnOnce(&[f32]) -> R) -> CandleResult<R> {
    let (storage, layout) = t.storage_and_layout();
    let Some((s, e)) = layout.contiguous_offsets() else {
        ffai_core::candle::bail!("expected a contiguous tensor")
    };
    match &*storage {
        Storage::Cpu(CpuStorage::F32(d)) => Ok(f(&d[s..e])),
        _ => ffai_core::candle::bail!("expected f32 CPU storage"),
    }
}

/// `n` f32 written by `fill`, which MUST write every element — there is no
/// zero pass to fall back on.
///
/// The slice handed to `fill` is created over the spare capacity, exactly as
/// `fastops::ElemOp` does: `f32` has no invalid bit patterns and no drop
/// glue, and the length is published with `set_len` only after `fill`
/// returns, so nothing observes an unwritten element and nothing is dropped
/// on unwind. The obligation on the CALLER is the one in the first sentence.
pub fn filled_f32(n: usize, fill: impl FnOnce(&mut [f32])) -> Vec<f32> {
    let mut v: Vec<f32> = Vec::with_capacity(n);
    {
        let spare = v.spare_capacity_mut();
        // SAFETY: `spare` is exactly `n` contiguous `MaybeUninit<f32>`; see the
        // doc comment for why the `&mut [f32]` over it is sound here.
        #[allow(unsafe_code)]
        let dst: &mut [f32] =
            unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n) };
        fill(dst);
    }
    // SAFETY: `fill` wrote all `n` elements (its contract), and `n` is within
    // the capacity reserved above.
    #[allow(unsafe_code)]
    unsafe {
        v.set_len(n);
    }
    v
}

/// [`filled_f32`] for `half::f16`, same contract.
pub fn filled_f16(n: usize, fill: impl FnOnce(&mut [half::f16])) -> Vec<half::f16> {
    let mut v: Vec<half::f16> = Vec::with_capacity(n);
    {
        let spare = v.spare_capacity_mut();
        // SAFETY: as in `filled_f32`; `half::f16` is `repr(transparent)` over
        // `u16`, with no invalid bit patterns and no drop glue.
        #[allow(unsafe_code)]
        let dst: &mut [half::f16] =
            unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<half::f16>(), n) };
        fill(dst);
    }
    // SAFETY: as in `filled_f32`.
    #[allow(unsafe_code)]
    unsafe {
        v.set_len(n);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filled_matches_a_zeroed_then_written_buffer() {
        let a = filled_f32(1000, |d| {
            for (i, x) in d.iter_mut().enumerate() {
                *x = i as f32 * 0.5;
            }
        });
        let mut b = vec![0f32; 1000];
        for (i, x) in b.iter_mut().enumerate() {
            *x = i as f32 * 0.5;
        }
        assert_eq!(a, b);
        assert!(filled_f32(0, |_| {}).is_empty());
    }

    #[test]
    fn with_f32_sees_the_tensor_elements() {
        let t = Tensor::from_vec(
            vec![1f32, 2.0, 3.0],
            (1, 3),
            &ffai_core::candle::Device::Cpu,
        )
        .unwrap();
        let sum = with_f32(&t, |d| d.iter().sum::<f32>()).unwrap();
        assert_eq!(sum, 6.0);
    }
}
