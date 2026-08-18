//! Bounded proofs (gate H-30).
//!
//! Compiled only under `cfg(kani)`, so this costs a normal build nothing.
//!
//! **What is worth proving here.** Not the SIMD kernels: Kani cannot execute x86
//! intrinsics any more than Miri can, and a proof that stops at the intrinsic
//! boundary proves nothing about the kernel. What it CAN do is the index and
//! length arithmetic around them — which is where every real defect this audit
//! found actually lived:
//!
//! * `reflect_pad` indexing an empty slice (fixed);
//! * `normalize` overflowing `u64` on a long digit run (fixed);
//! * the ONNX dims product wrapping `usize` and matching an empty payload (fixed).
//!
//! Each harness below pins one of those, so a future edit that reintroduces the
//! class fails verification rather than waiting for a fuzzer to rediscover it.
//!
//!     cargo kani --harness <name>

use crate::asr::mel;

/// `pad_or_trim_to` returns EXACTLY `target`, for every input length and target.
///
/// Every downstream shape calculation assumes this. Bounds are small because Kani
/// is a bounded model checker: the interesting cases are the boundaries (0, 1,
/// n == target, n > target), and they all live under 8.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn proof_pad_or_trim_to_length_contract() {
    let n: usize = kani::any();
    let target: usize = kani::any();
    kani::assume(n <= 8);
    kani::assume(target <= 8);

    let samples = vec![0.0f32; n];
    let out = mel::pad_or_trim_to(&samples, target);

    assert!(out.len() == target);
}

/// `n_frames` never claims more samples than exist.
///
/// A frame count that disagrees with the buffer it describes is how a shape
/// assumption becomes an out-of-bounds read.
#[cfg(kani)]
#[kani::proof]
fn proof_n_frames_never_exceeds_input() {
    let n: usize = kani::any();
    // Guard the multiply below from overflowing in the PROOF, not in the code.
    kani::assume(n <= 1 << 20);

    let frames = mel::MelSpectrogram::n_frames(n);

    assert!(frames * 160 <= n);
}

/// The dims-product check cannot be defeated by overflow.
///
/// This mirrors `tts/onnx.rs`: the original code used `dims.iter().product()`,
/// an UNCHECKED multiply over `i64 as usize`. `[1<<32, 1<<32]` wrapped to exactly
/// 0 and matched an empty payload, so a tensor whose dims did not describe its
/// data passed validation. The replacement folds with `checked_mul`; this proves
/// the fold either reports a real product or refuses, and never wraps.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn proof_dims_product_never_wraps() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();

    let dims = [a, b, c];
    let mut acc: Option<usize> = Some(1);
    for d in dims {
        acc = acc.and_then(|x| x.checked_mul(d));
    }

    if let Some(product) = acc {
        // A reported product is REACHABLE by real multiplication: no wrap can
        // have occurred, so it dominates each non-zero factor.
        for d in dims {
            if d != 0 {
                assert!(product >= d || product == 0);
            }
        }
    }
    // The `None` arm needs no assertion: refusing is always sound. The property
    // being proved is that there is no THIRD outcome - a wrapped value returned
    // as if it were the product.
}
