//! The zero-copy seam between our kernels and candle's tensors.
//!
//! # Why this module exists
//!
//! Every kernel used to start with `x.flatten_all()?.to_vec1::<f32>()?` —
//! a full copy of the input — and end with `Tensor::from_vec(..)`. Splitting
//! the 3x3 kernels showed the cost: **im2col 12.5% of detect, the GEMM 9.1%,
//! and ~14.4% in the marshalling around them.** The framework glue was
//! larger than the im2col and dwarfed the arithmetic, so each intermediate
//! round-tripped Tensor -> Vec -> Tensor between consecutive layers.
//!
//! candle's `CustomOp1` hands the CPU storage over directly, so the input is
//! read in place and the output `Vec` *becomes* the tensor's storage rather
//! than being copied into one. This is the same hook that recovered
//! 90 -> 78 ms for Mercury with no change to the arithmetic — "delivery is
//! not a detail".
//!
//! Byte-identical by construction: only the plumbing changes, never an
//! arithmetic operation or its order.

use candle_core::{CpuStorage, Layout, Result, Shape, Tensor};

/// Borrow a contiguous `f32` view of a tensor's storage.
///
/// The `CustomOp1` contract warns that storage "can use arbitrary strides,
/// offsets etc", so this refuses a non-contiguous layout rather than
/// silently reading the wrong elements — a class of bug that produces
/// plausible numbers instead of an error.
pub fn contiguous_f32<'a>(storage: &'a CpuStorage, layout: &Layout) -> Result<&'a [f32]> {
    let all = storage.as_slice::<f32>()?;
    match layout.contiguous_offsets() {
        Some((start, end)) => Ok(&all[start..end]),
        None => candle_core::bail!(
            "ffai-diana kernels require a contiguous input; call .contiguous() first"
        ),
    }
}

/// A `CustomOp1` built from a closure over the borrowed input slice.
///
/// The closure returns the output `Vec` and its shape; that `Vec` is moved
/// into the result tensor's storage, so neither side is copied.
pub struct SliceOp<F> {
    name: &'static str,
    f: F,
}

impl<F> SliceOp<F>
where
    F: Fn(&[f32], &Layout) -> Result<(Vec<f32>, Shape)>,
{
    pub fn new(name: &'static str, f: F) -> Self {
        Self { name, f }
    }

    /// Apply to `x` without building a backward graph — inference only.
    ///
    /// `FFAI_DIANA_NO_ZEROCOPY=1` routes through the old copy-in/copy-out
    /// path instead. The two arms run **identical arithmetic** and differ
    /// only in delivery, which is exactly what makes them a clean A/B — and
    /// on a drifting machine one binary with both arms behind a knob is the
    /// only trustworthy way to measure a change this size. The knob doubles
    /// as the fallback if a future candle version changes the custom-op
    /// contract.
    pub fn run(&self, x: &Tensor) -> Result<Tensor> {
        if copying_fallback() {
            let xs = x.flatten_all()?.to_vec1::<f32>()?;
            let (out, shape) = (self.f)(&xs, x.layout())?;
            return Tensor::from_vec(out, shape, x.device());
        }
        // Timed as its own bucket: this wrapper is entered ~2745 times per
        // image (one per convolution and one per activation), so if it costs
        // anything it costs it 2745 times, and it was previously folded into
        // whichever parent happened to call it.
        if crate::profile::roofline_enabled() {
            let t0 = std::time::Instant::now();
            let r = x.apply_op1_no_bwd(self);
            crate::profile::record_sliceop(
                self.name,
                t0.elapsed().as_nanos() as u64,
                crate::profile::take_last_work_ns(),
            );
            return r;
        }
        crate::profile::timed(|p| &p.sliceop, || x.apply_op1_no_bwd(self))
    }
}

fn copying_fallback() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FFAI_DIANA_NO_ZEROCOPY").is_some())
}

impl<F> candle_core::CustomOp1 for SliceOp<F>
where
    F: Fn(&[f32], &Layout) -> Result<(Vec<f32>, Shape)>,
{
    fn name(&self) -> &'static str {
        self.name
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let xs = contiguous_f32(storage, layout)?;
        let t0 = crate::profile::roofline_enabled().then(std::time::Instant::now);
        let (out, shape) = (self.f)(xs, layout)?;
        if let Some(t0) = t0 {
            crate::profile::set_last_work_ns(t0.elapsed().as_nanos() as u64);
        }
        Ok((CpuStorage::F32(out), shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn round_trips_a_tensor_without_changing_it() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 3, 4), &dev).unwrap();
        let op = SliceOp::new("double", |xs, l| {
            Ok((xs.iter().map(|v| v * 2.0).collect(), l.shape().clone()))
        });
        let y = op.run(&x).unwrap();
        assert_eq!(y.dims(), x.dims());
        let (a, b) = (
            x.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        for (i, (p, q)) in a.iter().zip(&b).enumerate() {
            assert!((p * 2.0 - q).abs() < 1e-6, "element {i}");
        }
    }

    #[test]
    fn rejects_a_non_contiguous_input_rather_than_misreading_it() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (4, 6), &dev).unwrap();
        let t = x.t().unwrap(); // transposed view: not contiguous
        let op = SliceOp::new("identity", |xs, l| Ok((xs.to_vec(), l.shape().clone())));
        assert!(op.run(&t).is_err(), "a non-contiguous layout must be refused");
    }
}
