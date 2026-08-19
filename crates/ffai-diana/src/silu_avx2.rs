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
    unsafe { silu_ptr(src.as_ptr(), dst.as_mut_ptr(), src.len()) }
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
    unsafe { silu_ptr(x.as_ptr(), x.as_mut_ptr(), x.len()) }
}

/// # Safety
///
/// `src`/`dst` valid for `n` floats. They MAY alias exactly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn silu_ptr(src: *const f32, dst: *mut f32, n: usize) { unsafe {
    // The same constants the scalar kernel derives, for the same reason:
    // hand-typed decimals here once silently selected a different f32 and
    // broke the oracle, so the compiler computes them.
    const L1: f32 = std::f32::consts::LN_2;
    const L2: f32 = L1 * L1 / 2.0;
    const L3: f32 = L1 * L1 * L1 / 6.0;
    const L4: f32 = L1 * L1 * L1 * L1 / 24.0;
    const L5: f32 = L1 * L1 * L1 * L1 * L1 / 120.0;
    const MAGIC: f32 = 12582912.0; // 1.5 * 2^23

    let (log2e, magic) = (_mm256_set1_ps(std::f32::consts::LOG2_E), _mm256_set1_ps(MAGIC));
    let (lo, hi) = (_mm256_set1_ps(-125.0), _mm256_set1_ps(125.0));
    let one = _mm256_set1_ps(1.0);
    let (c1, c2, c3) = (_mm256_set1_ps(L1), _mm256_set1_ps(L2), _mm256_set1_ps(L3));
    let (c4, c5) = (_mm256_set1_ps(L4), _mm256_set1_ps(L5));
    let bias = _mm256_set1_epi32(0x3f80_0000u32 as i32);

    let mut i = 0usize;
    while i + 8 <= n {
        let x = _mm256_loadu_ps(src.add(i));
        // t = clamp(-x * log2e)
        let t = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), x), log2e), lo), hi);
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
            bias,
        ));
        // Same reasoning: mul then add, matching `1.0 + exp_fast(-x)`.
        let y = _mm256_div_ps(x, _mm256_add_ps(_mm256_mul_ps(p, scale), one));
        _mm256_storeu_ps(dst.add(i), y);
        i += 8;
    }
    // Scalar tail — the same function the oracle uses, so the tail cannot
    // drift from the body.
    for j in i..n {
        *dst.add(j) = crate::silu::silu_scalar_pub(*src.add(j));
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
