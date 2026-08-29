//! Tensor-level activations on [`crate::fastmath`] — the drop-in replacements
//! for candle's, which compute a scalar libm call per element.
//!
//! # What this is worth
//!
//! Measured in `ffai-argus` on `(1, 1024, 3072)`, the shape a `SigLIP` MLP
//! actually runs: candle's `.gelu()` took **44.01 ms**, this shape of kernel
//! **1.22 ms** — 32x, with the caption it feeds byte-identical to the reference
//! implementation's. candle's activations are elementwise ops in a backend that
//! uses rayon for `conv2d` and nothing else, evaluating `tanhf`/`erf` per
//! element on one core.
//!
//! # Two things that are NOT interchangeable
//!
//! `gelu_erf` is `0.5x(1 + erf(x/sqrt 2))`; `gelu_tanh` is the tanh
//! approximation of it. **They differ by up to ~1e-3** — far above the
//! tolerance these engines gate at — so a site calling one must be given that
//! one. Six of the seventeen activation sites in this workspace are `gelu_erf`
//! and three are `gelu_tanh`; swapping them silently would be a quality
//! regression that no test here would catch, because both are "a GELU".
//!
//! # Delivery
//!
//! Each op is a [`candle_core::CustomOp1`], which hands the kernel candle's own
//! `CpuStorage`. That matters more than the arithmetic: routing a tensor
//! through `to_vec1()` and `Tensor::from_vec` is a fixed per-call tax, and in
//! the Argus campaign more than half the total win came from removing the glue
//! rather than from the polynomial (3.03 ms -> 2.30 -> 1.08 across three
//! delivery fixes with no change to the inner loop).

use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

/// Elements per rayon task.
///
/// Large enough that scheduling is noise against the work, small enough that a
/// typical activation tensor still makes hundreds of tasks for the pool to
/// balance.
const CHUNK: usize = 8192;

/// One elementwise activation, applied through candle's zero-copy hook.
struct ElemOp {
    name: &'static str,
    f: fn(f32) -> f32,
}

impl CustomOp1 for ElemOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(src) = storage else {
            candle_core::bail!("{}: expects f32", self.name)
        };
        // `contiguous_offsets` is `Some` only for a genuinely flat run. A
        // strided view read as if it were dense is silent corruption, not an
        // error, so this refuses rather than guesses.
        let Some((start, end)) = layout.contiguous_offsets() else {
            candle_core::bail!("{}: expects a contiguous input", self.name)
        };
        let src = &src[start..end];
        let n = src.len();

        let mut out: Vec<f32> = Vec::with_capacity(n);
        {
            // Written through the SPARE capacity, not `vec![0.0; n]`.
            //
            // Every element is overwritten below, so zero-initialising first is
            // an entire discarded pass over the buffer — 2.6 GB per image in
            // the Argus vision tower. `set_len` before filling would be the
            // other way to avoid it and is UB-adjacent (clippy's `uninit_vec`
            // is right); this creates the `&mut [f32]` over the spare region
            // and publishes it only after every element is written.
            let spare = out.spare_capacity_mut();
            // SAFETY: `spare` is exactly `n` contiguous `MaybeUninit<f32>`.
            // `f32` has no invalid bit patterns and no drop glue, and the
            // partitioned loop below writes all n before `set_len` publishes
            // them, so nothing observes an uninitialised value and nothing is
            // dropped on unwind.
            #[allow(unsafe_code)]
            let dst: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n) };
            let f = self.f;
            dst.par_chunks_mut(CHUNK)
                .zip(src.par_chunks(CHUNK))
                .for_each(|(d, s)| {
                    for (o, &i) in d.iter_mut().zip(s) {
                        *o = f(i);
                    }
                });
        }
        // SAFETY: the loop above wrote all n elements.
        #[allow(unsafe_code)]
        unsafe {
            out.set_len(n);
        }
        Ok((CpuStorage::F32(out), layout.shape().clone()))
    }
}

macro_rules! tensor_op {
    ($fn_name:ident, $kernel:path, $tag:literal, $doc:literal) => {
        #[doc = $doc]
        ///
        /// # Errors
        /// If the input is not contiguous `f32`.
        pub fn $fn_name(xs: &Tensor) -> Result<Tensor> {
            // Materialise a strided view before handing it to the kernel.
            //
            // The kernel itself REFUSES non-contiguous storage, and that
            // strictness is right — reading a strided tensor as if it were
            // dense is silent corruption, not an error. But refusing at the
            // API boundary makes every caller responsible for a detail candle's
            // own ops handle, and `wav2vec2`'s `pos_conv` passes exactly such a
            // view. Its test caught this as "expects a contiguous input" the
            // moment the migration landed.
            //
            // `contiguous()` on an already-contiguous tensor is a cheap clone,
            // so the common path pays nothing.
            xs.contiguous()?.apply_op1_no_bwd(&ElemOp {
                name: $tag,
                f: $kernel,
            })
        }
    };
}

tensor_op!(
    gelu_erf,
    crate::fastmath::gelu_erf,
    "ffai-gelu-erf",
    "`0.5x(1 + erf(x/sqrt 2))` — the drop-in for candle's `.gelu_erf()`."
);
tensor_op!(
    gelu_tanh,
    crate::fastmath::gelu_tanh,
    "ffai-gelu-tanh",
    "`gelu_pytorch_tanh` — the drop-in for candle's `.gelu()`."
);
tensor_op!(
    tanh,
    crate::fastmath::tanh,
    "ffai-tanh",
    "`tanh(x)` — the drop-in for candle's `.tanh()`."
);
tensor_op!(
    silu,
    crate::fastmath::silu,
    "ffai-silu",
    "`x * sigmoid(x)` — the drop-in for candle's `.silu()`."
);
tensor_op!(
    erf,
    crate::fastmath::erf,
    "ffai-erf",
    "`erf(x)` — the drop-in for candle's `.erf()`."
);

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Every op against the candle op it replaces, on a shape with an awkward
    /// tail so the chunked loop's remainder is exercised rather than assumed.
    #[test]
    fn every_op_tracks_candles() {
        let d = Device::Cpu;
        let xs = Tensor::rand(-8.0f32, 8.0, (3, 4099), &d).expect("xs");
        assert_ne!(4099 % CHUNK, 0, "the tail must not be a whole chunk");

        let cases: Vec<(&str, Tensor, Tensor)> = vec![
            (
                "gelu_erf",
                gelu_erf(&xs).expect("ours"),
                xs.gelu_erf().expect("candle"),
            ),
            (
                "gelu_tanh",
                gelu_tanh(&xs).expect("ours"),
                xs.gelu().expect("candle"),
            ),
            ("tanh", tanh(&xs).expect("ours"), xs.tanh().expect("candle")),
            ("silu", silu(&xs).expect("ours"), xs.silu().expect("candle")),
            ("erf", erf(&xs).expect("ours"), xs.erf().expect("candle")),
        ];
        for (name, a, b) in cases {
            assert_eq!(a.dims(), b.dims(), "{name}: shape");
            let (av, bv) = (
                a.flatten_all().expect("a").to_vec1::<f32>().expect("av"),
                b.flatten_all().expect("b").to_vec1::<f32>().expect("bv"),
            );
            let worst = av
                .iter()
                .zip(&bv)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            eprintln!("  {name:<10} worst abs {worst:.3e}");
            assert!(worst < 1e-4, "{name} differs from candle by {worst:.3e}");
        }
    }

    /// The distinction that a "both are a GELU" mistake would erase.
    #[test]
    fn the_two_gelus_stay_different() {
        let d = Device::Cpu;
        let xs = Tensor::rand(-4.0f32, 4.0, 2048, &d).expect("xs");
        let a = gelu_erf(&xs)
            .expect("erf")
            .flatten_all()
            .expect("f")
            .to_vec1::<f32>()
            .expect("v");
        let b = gelu_tanh(&xs)
            .expect("tanh")
            .flatten_all()
            .expect("f")
            .to_vec1::<f32>()
            .expect("v");
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst > 1e-4,
            "gelu_erf and gelu_tanh differ by only {worst:.3e} — one has been aliased to the other"
        );
    }

    /// A strided view must give the RIGHT answer, not a plausible wrong one.
    ///
    /// The kernel refuses non-contiguous storage — reading it as dense would
    /// silently transpose the data — so the wrapper materialises first. This
    /// pins the outcome rather than the mechanism: whatever the wrapper does,
    /// a transposed input must produce the transposed result.
    #[test]
    fn a_strided_view_is_handled_not_misread() {
        let d = Device::Cpu;
        let xs = Tensor::rand(-3.0f32, 3.0, (7, 5), &d).expect("xs");
        let strided = xs.t().expect("transpose");
        assert!(!strided.is_contiguous(), "the test needs a strided view");

        let ours = tanh(&strided).expect("strided must be handled");
        let theirs = strided.tanh().expect("candle");
        assert_eq!(ours.dims(), theirs.dims());
        let (a, b) = (
            ours.flatten_all().expect("a").to_vec1::<f32>().expect("av"),
            theirs
                .flatten_all()
                .expect("b")
                .to_vec1::<f32>()
                .expect("bv"),
        );
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-5,
            "strided result differs by {worst:.3e} — misread as dense?"
        );
    }

    #[test]
    fn an_empty_tensor_does_not_panic() {
        let d = Device::Cpu;
        let xs = Tensor::zeros((0, 8), DType::F32, &d).expect("xs");
        assert_eq!(gelu_erf(&xs).expect("gelu").elem_count(), 0);
        assert_eq!(tanh(&xs).expect("tanh").elem_count(), 0);
    }
}
