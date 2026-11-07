//! Explicit AVX2 `SiLU` — the last unpruned lever on the latency gate.
//!
//! # Why this one and not the others
//!
//! The descent priced every alternative first, and each came back closed:
//!
//! * the convolution WRAPPER (bias + reshape) is 1.8 % — nothing there;
//! * the `SliceOp`/`CustomOp1` round trip costs essentially nothing across
//!   2010 calls per image, measured by splitting it into its own bucket;
//! * candle's GEMM already reaches 69 % of peak at the balanced shapes and
//!   is memory-bound at ours, which no microkernel of mine fixes;
//! * a direct convolution replacing im2col lost 9/9 rounds, z = +3.00;
//! * silu's DIVISION has no measurable cost, so `vrcpps` + Newton is pruned.
//!
//! What survives is this: silu is **30.7 % of serial detect**, the largest
//! single line item after two rounds of fixes, and its scalar form runs at
//! 14.0 GB/s against a 32.6 GB/s copy ceiling. The remaining ~2.3x is the
//! polynomial and the bit manipulation, and the only way to take it is to
//! stop hoping LLVM vectorises them and write the vectors.
//!
//! # The contract
//!
//! The scalar path in [`crate::silu`] stays forever as the oracle and the
//! fallback. This is dispatched at runtime on `avx2` + `fma`, so a machine
//! without them runs the scalar path rather than an illegal instruction, and
//! a test asserts the two agree.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// True when this CPU has the features the kernel below requires.
///
/// Checked at RUNTIME, not compile time: the crate is built for the x86-64
/// baseline so that a published binary runs everywhere, which is exactly why
/// `-C target-cpu=native` was measured and pruned as a global answer.
#[inline]
#[must_use] 
pub fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `SiLU` over `src` into `dst`, eight lanes at a time.
///
/// # Safety
///
/// Caller must have checked [`available`]. `dst.len() == src.len()` is
/// asserted rather than assumed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
pub unsafe fn silu_into(src: &[f32], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "silu_avx2: length mismatch");
    // SAFETY: equal lengths, both valid for `len` floats.
    unsafe { silu_ptr::<false>(src.as_ptr(), dst.as_mut_ptr(), src.len(), 0.0) }
}

/// `SiLU` over a buffer IN PLACE — what a fused epilogue needs.
///
/// The kernel loads lane `i` before storing lane `i`, so `src == dst` is
/// correct. It cannot be written as `silu_into(x, x)` because that would hold
/// a shared and a unique reference to one buffer, so both wrappers go through
/// the raw-pointer core and there remains exactly ONE copy of the polynomial.
///
/// # Safety
///
/// Caller must have checked [`available`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
pub unsafe fn silu_in_place(x: &mut [f32]) {
    // SAFETY: one buffer, valid for its length; load-before-store per lane.
    unsafe { silu_ptr::<false>(x.as_ptr(), x.as_mut_ptr(), x.len(), 0.0) }
}

/// `silu(x + bias)` IN PLACE — the whole convolution epilogue in one pass.
///
/// The epilogue used to walk every channel twice: once to add the bias,
/// once for the activation. On the Ir census the bias pass alone was 9.1 M
/// instructions per image across 13.4 M elements. Folding it in here is one
/// broadcast add per eight lanes inside a loop that already holds them.
///
/// Per element this computes exactly `silu(x + bias)` in that order, with the
/// same `mul`/`add` sequence as [`silu_in_place`], so the result is
/// bit-identical to the two-pass form.
///
/// # Safety
///
/// Caller must have checked [`available`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
pub unsafe fn silu_bias_in_place(x: &mut [f32], bias: f32) {
    // SAFETY: one buffer, valid for its length; load-before-store per lane.
    unsafe { silu_ptr::<true>(x.as_ptr(), x.as_mut_ptr(), x.len(), bias) }
}

/// # Safety
///
/// `src`/`dst` valid for `n` floats. They MAY alias exactly. `BIAS` selects
/// whether `bias` is added before the activation; monomorphised so the
/// bias-free kernel carries no add.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn silu_ptr<const BIAS: bool>(src: *const f32, dst: *mut f32, n: usize, bias: f32) { unsafe {
    silu_ptr_res::<BIAS, false>(src, dst, n, bias, std::ptr::null())
}}

/// `silu(x + bias) + res[i]` in place: the residual connection folded into the
/// activation pass. One extra load and add per eight lanes, against a whole
/// second traversal (load, add, store, loop) as its own op.
///
/// # Safety
///
/// `x` and `res` the same length; avx2+fma on the host.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
pub unsafe fn silu_bias_res_in_place(x: &mut [f32], bias: f32, res: &[f32]) {
    assert_eq!(x.len(), res.len(), "silu_avx2: residual length mismatch");
    // SAFETY: one buffer plus a same-length residual; load-before-store per lane.
    unsafe { silu_ptr_res::<true, true>(x.as_ptr(), x.as_mut_ptr(), x.len(), bias, res.as_ptr()) }
}

/// # Safety
///
/// As `silu_ptr`; `res` valid for `n` floats when `RES`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn silu_ptr_res<const BIAS: bool, const RES: bool>(src: *const f32, dst: *mut f32, n: usize, bias: f32, res: *const f32) { unsafe {
    // The same constants the scalar kernel derives, for the same reason:
    // hand-typed decimals here once silently selected a different f32 and
    // broke the oracle, so the compiler computes them.
    const L1: f32 = std::f32::consts::LN_2;
    const L2: f32 = L1 * L1 / 2.0;
    const L3: f32 = L1 * L1 * L1 / 6.0;
    const L4: f32 = L1 * L1 * L1 * L1 / 24.0;
    const L5: f32 = L1 * L1 * L1 * L1 * L1 / 120.0;
    const MAGIC: f32 = 12582912.0; // 1.5 * 2^23

    // `-LOG2_E`, so `t = x * (-log2e)` replaces `(0 - x) * log2e`: IEEE
    // multiplication is sign-symmetric, so the product is bit-identical, and
    // the zero-sign difference it leaves in `f` dies at `f * c5 + c4`. Two
    // instructions per eight lanes (the xor+sub negate) gone.
    let (nlog2e, magic) = (_mm256_set1_ps(-std::f32::consts::LOG2_E), _mm256_set1_ps(MAGIC));
    let (lo, hi) = (_mm256_set1_ps(-125.0), _mm256_set1_ps(125.0));
    let one = _mm256_set1_ps(1.0);
    let (c1, c2, c3) = (_mm256_set1_ps(L1), _mm256_set1_ps(L2), _mm256_set1_ps(L3));
    let (c4, c5) = (_mm256_set1_ps(L4), _mm256_set1_ps(L5));
    // The exponent bias `127 << 23` IS the bit pattern of `1.0f32`, so it
    // shares `one`'s register: this loop holds twelve constants in sixteen
    // ymm registers and the asm showed it spilling to `vmovaps` shuffles and
    // re-broadcasting `log2e` every iteration. One register back is three
    // instructions per eight lanes.
    let ebias = _mm256_castps_si256(one);
    let bvec = _mm256_set1_ps(bias);

    // One eight-lane step of the kernel at offset `$off`, as a macro so the
    // loop below can take two steps per iteration: the loop control was
    // three of the twenty-eight instructions per eight lanes.
    macro_rules! lane8 {
        ($i:expr, $off:expr) => {{
        let x = _mm256_loadu_ps(src.add($i + $off));
            let x = if BIAS { _mm256_add_ps(x, bvec) } else { x };
            // t = clamp(-x * log2e)
            let t = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(x, nlog2e), lo), hi);
            // Round by float addition: z = t + MAGIC, n = z - MAGIC, f = t - n.
            let z = _mm256_add_ps(t, magic);
            let nn = _mm256_sub_ps(z, magic);
            let f = _mm256_sub_ps(t, nn);
            // Horner with SEPARATE mul and add — deliberately NOT `fmadd`.
            //
            // FMA fuses multiply-add into a single rounding where the scalar
            // path rounds twice, so an fmadd version computes a very slightly
            // DIFFERENT function. It passed a 1e-6 unit test against the scalar
            // twin and still failed the full-graph oracle at `head_boxes`:
            // 1.023e-4 against a 1.000e-4 bound, 2 % over, after accumulating
            // through sixty layers.
            //
            // The vector kernel's job is to REPRODUCE the scalar oracle, not to
            // improve its arithmetic. A faster kernel that computes something
            // else is not a faster kernel — and the 2 % overshoot is exactly how
            // quietly that shows up.
            let mut p = _mm256_add_ps(_mm256_mul_ps(f, c5), c4);
            p = _mm256_add_ps(_mm256_mul_ps(f, p), c3);
            p = _mm256_add_ps(_mm256_mul_ps(f, p), c2);
            p = _mm256_add_ps(_mm256_mul_ps(f, p), c1);
            p = _mm256_add_ps(_mm256_mul_ps(f, p), one);
            // 2^n straight out of z's mantissa, exactly as the scalar path does.
            let scale = _mm256_castsi256_ps(_mm256_add_epi32(
                _mm256_slli_epi32(_mm256_castps_si256(z), 23),
                ebias,
            ));
            // Same reasoning: mul then add, matching `1.0 + exp_fast(-x)`.
            let y = _mm256_div_ps(x, _mm256_add_ps(_mm256_mul_ps(p, scale), one));
            let y = if RES { _mm256_add_ps(y, _mm256_loadu_ps(res.add($i + $off))) } else { y };
            _mm256_storeu_ps(dst.add($i + $off), y);
        }};
    }
    let mut i = 0usize;
    while i + 16 <= n {
        lane8!(i, 0);
        lane8!(i, 8);
        i += 16;
    }
    while i + 8 <= n {
        lane8!(i, 0);
        i += 8;
    }
    // Scalar tail — the same function the oracle uses, so the tail cannot
    // drift from the body.
    for j in i..n {
        let x = *src.add(j);
        let y = crate::silu::silu_scalar_pub(if BIAS { x + bias } else { x });
        *dst.add(j) = if RES { y + *res.add(j) } else { y };
    }
}}

#[cfg(not(target_arch = "x86_64"))]
#[allow(unsafe_code)]
pub unsafe fn silu_in_place(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = crate::silu::silu_scalar_pub(*v);
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(unsafe_code)]
pub unsafe fn silu_bias_in_place(x: &mut [f32], bias: f32) {
    for v in x.iter_mut() {
        *v = crate::silu::silu_scalar_pub(*v + bias);
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(unsafe_code)]
pub unsafe fn silu_into(src: &[f32], dst: &mut [f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = crate::silu::silu_scalar_pub(*s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vector kernel must agree with the scalar twin that gates the
    /// whole graph. Checked across the activation range AND the saturating
    /// tails, where the exponent-field write is most likely to go wrong.
    #[test]
    #[allow(unsafe_code)]
    fn matches_the_scalar_twin() {
        if !available() {
            eprintln!("SKIP: no avx2+fma on this host");
            return;
        }
        // Deliberately not a multiple of 8, so the scalar tail is exercised.
        let src: Vec<f32> = (0..1003).map(|i| (i as f32 - 500.0) / 7.3).collect();
        let mut got = vec![0f32; src.len()];
        // SAFETY: `available()` checked above; lengths are equal.
        unsafe { silu_into(&src, &mut got) };
        let mut worst = 0f32;
        for (i, &x) in src.iter().enumerate() {
            let want = crate::silu::silu_scalar_pub(x);
            let scale = want.abs().max(1e-6);
            worst = worst.max((got[i] - want).abs() / scale);
        }
        assert!(worst < 1e-6, "avx2 silu diverges from scalar by {worst:.3e}");
    }

    /// The vector kernel is the scalar kernel, BIT FOR BIT, over a dense sweep
    /// of the activation range and the saturating tails. The 1e-6 test above
    /// is what the kernel was first admitted on; this is the stronger claim
    /// the depthwise fusion rests on, because that fusion REPLACES a scalar
    /// pass with this kernel and the full-graph oracle anchors on exact bytes.
    #[test]
    #[allow(unsafe_code)]
    fn is_bit_identical_to_the_scalar_kernel_over_a_dense_sweep() {
        if !available() {
            eprintln!("SKIP: no avx2+fma on this host");
            return;
        }
        // 4 M points in [-40, 40], plus every float in [-1, 1] at a 2^-20
        // step, plus the clamp edges and the zeros.
        let mut src: Vec<f32> = (0..4_000_000).map(|i| -40.0 + i as f32 * (80.0 / 4_000_000.0)).collect();
        src.extend((0..2_097_152).map(|i| -1.0 + i as f32 * (2.0 / 2_097_152.0)));
        src.extend_from_slice(&[0.0, -0.0, 86.0, -86.0, 87.3, -87.3, 100.0, -100.0, 1e-30, -1e-30]);
        let mut got = vec![0f32; src.len()];
        // SAFETY: `available()` checked above; lengths are equal.
        unsafe { silu_into(&src, &mut got) };
        let mut bad = 0usize;
        for (i, &x) in src.iter().enumerate() {
            let want = crate::silu::silu_scalar_pub(x);
            if got[i].to_bits() != want.to_bits() {
                if bad < 5 {
                    eprintln!("x={x:e}: avx2 {:e} ({:#x}) scalar {:e} ({:#x})", got[i], got[i].to_bits(), want, want.to_bits());
                }
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "{bad} of {} lanes differ from the scalar kernel", src.len());
    }

    /// `silu_bias_in_place` must equal bias-then-`silu_in_place` BIT FOR BIT:
    /// the epilogue fusion rests on the two being the same arithmetic in the
    /// same order, and a tolerance here would let a drift through.
    #[test]
    #[allow(unsafe_code)]
    fn fused_bias_is_bit_identical_to_two_passes() {
        if !available() {
            return;
        }
        for &bias in &[-1.75f32, 0.0, 0.3125, 4.0] {
            // Not a multiple of 8, so the scalar tail is exercised too.
            let src: Vec<f32> = (0..1003).map(|i| (i as f32 - 500.0) / 7.3).collect();
            let mut two = src.clone();
            for e in &mut two {
                *e += bias;
            }
            let mut one = src;
            // SAFETY: `available()` checked above.
            unsafe {
                silu_in_place(&mut two);
                silu_bias_in_place(&mut one, bias);
            }
            for (i, (a, b)) in one.iter().zip(&two).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "bias {bias} element {i}: {a} vs {b}");
            }
        }
    }

    /// Saturating inputs are where an exponent-field write goes wrong
    /// silently — the clamp exists for exactly this and deserves its own case.
    #[test]
    #[allow(unsafe_code)]
    fn survives_the_saturating_tails() {
        if !available() {
            return;
        }
        let src: Vec<f32> = vec![-1e30, -200.0, -88.0, -1.0, 0.0, 1.0, 88.0, 200.0, 1e30, 0.0];
        let mut got = vec![0f32; src.len()];
        // SAFETY: `available()` checked above; lengths are equal.
        unsafe { silu_into(&src, &mut got) };
        for (i, &x) in src.iter().enumerate() {
            let want = crate::silu::silu_scalar_pub(x);
            assert!(
                got[i].is_finite() == want.is_finite(),
                "finiteness differs at x={x}: {} vs {}",
                got[i],
                want
            );
            if want.is_finite() {
                let scale = want.abs().max(1e-6);
                assert!(
                    (got[i] - want).abs() / scale < 1e-6,
                    "x={x}: {} vs {}",
                    got[i],
                    want
                );
            }
        }
    }
}
