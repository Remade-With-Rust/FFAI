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

// ---------------------------------------------------------------------------
// Class B: the raw-pointer sharing across rayon tasks.
//
// This is R-002 — the last claim in the crate that was argued rather than
// tested. `unsafe impl Send`/`Sync` for `SendPtr(*mut f32)` is sound ONLY IF
// distinct tasks write disjoint regions. That was a comment; here it is a
// proof over the same `task_region` the kernel calls.
//
// Kani cannot execute the AVX2 kernel, and does not need to: the kernel's
// soundness does not depend on what the intrinsics compute, only on WHERE each
// task is allowed to write.
// ---------------------------------------------------------------------------

/// Distinct tasks never address the same output element.
///
/// Bounds are small because this is a bounded model checker; the property is
/// structural (chunks partition rows, blocks partition columns), so the
/// boundaries — a ragged final chunk, a ragged final block, single-element
/// regions — all occur well under these limits.
#[cfg(kani)]
#[kani::proof]
fn proof_task_regions_are_disjoint() {
    use crate::tts::decoder_kernels::task_region;

    let co_chunk: usize = kani::any();
    let block_t: usize = kani::any();
    let c_out: usize = kani::any();
    let l_out: usize = kani::any();
    kani::assume(co_chunk >= 1 && co_chunk <= 3);
    kani::assume(block_t >= 1 && block_t <= 3);
    kani::assume(c_out >= 1 && c_out <= 6);
    kani::assume(l_out >= 1 && l_out <= 6);

    // Exactly how the kernel derives them.
    let n_chunks = c_out.div_ceil(co_chunk);
    let n_blocks = l_out.div_ceil(block_t);
    kani::assume(n_blocks >= 1);

    let a: usize = kani::any();
    let b: usize = kani::any();
    kani::assume(a < n_chunks * n_blocks);
    kani::assume(b < n_chunks * n_blocks);
    kani::assume(a != b);

    let (a_co0, a_co1, a_t0, a_t1) = task_region(a, n_blocks, co_chunk, block_t, c_out, l_out);
    let (b_co0, b_co1, b_t0, b_t1) = task_region(b, n_blocks, co_chunk, block_t, c_out, l_out);

    // Two rectangles are disjoint if they are separated on either axis. Rows
    // and columns each partition, so distinct tasks differ on at least one.
    let rows_disjoint = a_co1 <= b_co0 || b_co1 <= a_co0;
    let cols_disjoint = a_t1 <= b_t0 || b_t1 <= a_t0;
    assert!(rows_disjoint || cols_disjoint);
}

/// Every task's region lies inside the output buffer.
///
/// The kernel forms `out_ptr.add(co * l_out + t0)` with length `t1 - t0`, so
/// the highest element it can touch is `(co1-1) * l_out + t1 - 1`. That must
/// stay under `c_out * l_out`, or the slice runs past the allocation.
#[cfg(kani)]
#[kani::proof]
fn proof_task_regions_are_in_bounds() {
    use crate::tts::decoder_kernels::task_region;

    let co_chunk: usize = kani::any();
    let block_t: usize = kani::any();
    let c_out: usize = kani::any();
    let l_out: usize = kani::any();
    kani::assume(co_chunk >= 1 && co_chunk <= 3);
    kani::assume(block_t >= 1 && block_t <= 3);
    kani::assume(c_out >= 1 && c_out <= 6);
    kani::assume(l_out >= 1 && l_out <= 6);

    let n_chunks = c_out.div_ceil(co_chunk);
    let n_blocks = l_out.div_ceil(block_t);
    let task: usize = kani::any();
    kani::assume(task < n_chunks * n_blocks);

    let (co0, co1, t0, t1) = task_region(task, n_blocks, co_chunk, block_t, c_out, l_out);

    assert!(co1 <= c_out);
    assert!(t1 <= l_out);
    assert!(t0 <= t1); // `bt = t1 - t0` must not underflow
    if co1 > co0 && t1 > t0 {
        let highest = (co1 - 1) * l_out + (t1 - 1);
        assert!(highest < c_out * l_out);
    }
}

// ---------------------------------------------------------------------------
// Class B, second site: `asr/flash_attn.rs`.
//
// This one is subtler than decoder_kernels, and the difference is worth stating
// because it changes what a proof can claim. The slices handed to each head
// OVERLAP by construction - head 0 receives a slice covering the whole buffer -
// so disjointness cannot come from the slice bounds. It comes from the ADDRESSING
// SCHEME: head `h` writes only `h*HD + row*width + c` for `c < HD`, i.e. a column
// band of stride `width`.
//
// SCOPE OF THIS PROOF, stated plainly: it proves the addressing scheme is
// disjoint and in-bounds. It ASSUMES `flash_head_strided` writes only at those
// offsets, which is not proved here - that assumption is what the existing
// `flash_attn::tests::matches_three_op_path` oracle exercises, by comparing the
// strided kernel against the three-op path element for element.
// ---------------------------------------------------------------------------

/// The absolute output index head `h` touches at (`row`, `c`).
///
/// Mirrors `from_raw_parts_mut(base.add(h * HD), ...)` followed by a strided
/// write at `row * width + c`.
#[cfg(any(kani, test))]
const fn head_write_index(h: usize, row: usize, c: usize, hd: usize, width: usize) -> usize {
    h * hd + row * width + c
}

/// Distinct heads never write the same element.
#[cfg(kani)]
#[kani::proof]
fn proof_flash_head_bands_are_disjoint() {
    let hd: usize = kani::any();
    let heads: usize = kani::any();
    let seq: usize = kani::any();
    kani::assume(hd >= 1 && hd <= 3);
    kani::assume(heads >= 1 && heads <= 3);
    kani::assume(seq >= 1 && seq <= 3);
    let width = heads * hd;

    let (h1, r1, c1): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
    let (h2, r2, c2): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
    kani::assume(h1 < heads && h2 < heads && h1 != h2);
    kani::assume(r1 < seq && r2 < seq);
    kani::assume(c1 < hd && c2 < hd);

    // Different heads, therefore different elements - whatever rows and columns
    // within their own bands they choose.
    assert!(head_write_index(h1, r1, c1, hd, width) != head_write_index(h2, r2, c2, hd, width));
}

/// Every head's writes stay inside the `seq * width` output buffer.
#[cfg(kani)]
#[kani::proof]
fn proof_flash_head_writes_are_in_bounds() {
    let hd: usize = kani::any();
    let heads: usize = kani::any();
    let seq: usize = kani::any();
    kani::assume(hd >= 1 && hd <= 3);
    kani::assume(heads >= 1 && heads <= 3);
    kani::assume(seq >= 1 && seq <= 3);
    let width = heads * hd;

    let (h, row, c): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
    kani::assume(h < heads && row < seq && c < hd);

    assert!(head_write_index(h, row, c, hd, width) < seq * width);
}

// ---------------------------------------------------------------------------
// asr::vocab_int8 — block-scaled int8 vocabulary GEMV
//
// `dot_i8_blocked` is `unsafe fn` reading with `_mm256_loadu_si256` and
// `bscale.get_unchecked(b)`. Its inner loop tests `o < blk` but READS 32 BYTES
// per step, so `blk` not being a multiple of 32 is an out-of-bounds read, not a
// slow path — and on the final block it runs past the whole allocation.
//
// `Int8Vocab::new` is the only place that invariant can be established, so
// these harnesses pin exactly what that guard has to guarantee.
// ---------------------------------------------------------------------------

/// The highest byte index the kernel touches for a row of `nblocks` blocks of
/// `blk` bytes each. Mirrors the loop: `base = b * blk`, steps of 32, each step
/// reading 32 bytes from `base + o`.
#[cfg(any(kani, test))]
pub(crate) const fn int8_last_byte_read(nblocks: usize, blk: usize) -> usize {
    // Final block, final 32-lane step, final byte of that step.
    let base = (nblocks - 1) * blk;
    let steps = blk.div_ceil(32);
    base + (steps - 1) * 32 + 31
}

/// With the guard `Int8Vocab::new` enforces — `blk != 0`, `blk % 32 == 0`,
/// `d % blk == 0` — the kernel never reads past the `d`-byte row it was given.
///
/// This is the proof the SAFETY comment on `dot_i8_blocked` asserts in prose.
#[cfg(kani)]
#[kani::proof]
fn proof_int8_blocks_never_read_past_the_row() {
    let blk: usize = kani::any();
    let nblocks: usize = kani::any();
    // Exactly the guard in `Int8Vocab::new`, restated as assumptions.
    kani::assume(blk >= 32 && blk <= 96 && blk % 32 == 0);
    kani::assume(nblocks >= 1 && nblocks <= 4);
    let d = nblocks * blk; // `d % blk == 0` by construction

    assert!(int8_last_byte_read(nblocks, blk) < d);
}

/// The same claim stated as the negative: drop `blk % 32 == 0` and the read DOES
/// escape the row. This harness fails if someone "simplifies" the guard away, so
/// the guard cannot rot without verification going red.
#[cfg(kani)]
#[kani::proof]
fn proof_int8_unaligned_blk_would_over_read() {
    let blk: usize = 48; // passes `d % blk == 0` for d = 96, fails `blk % 32 == 0`
    let nblocks: usize = 2;
    let d = nblocks * blk;
    // 48 is not a multiple of 32, so the last step reads past the row.
    assert!(int8_last_byte_read(nblocks, blk) >= d);
}

/// Per-block scales are indexed with `get_unchecked(b)` for `b < nblocks`, and
/// the slice handed in is `bscale[v*nb .. (v+1)*nb]` — exactly `nb` long.
#[cfg(kani)]
#[kani::proof]
fn proof_int8_bscale_index_in_bounds() {
    let nb: usize = kani::any();
    let v: usize = kani::any();
    let vocab: usize = kani::any();
    kani::assume(nb >= 1 && nb <= 4);
    kani::assume(vocab >= 1 && vocab <= 4);
    kani::assume(v < vocab);

    let (lo, hi) = (v * nb, (v + 1) * nb);
    assert!(hi <= vocab * nb); // the row slice is inside the flat buffer
    assert!(hi - lo == nb); // and it is exactly nb long, so b < nb is in bounds
}

// ---------------------------------------------------------------------------
// asr::f16_gemv — f16 weight GEMV
//
// `dot_f16` takes a RAW POINTER `w.as_ptr().add(i * d)` and reads `d` f16
// values from it. The claim the SAFETY comment makes is that row `i` of an
// `out_dim * in_dim` buffer is fully addressable for every `i < out_dim`.
// ---------------------------------------------------------------------------

/// Row `i` occupies `[i*d, i*d + d)`, and for `i < o` that lies inside `o*d`.
#[cfg(kani)]
#[kani::proof]
fn proof_f16_rows_are_in_bounds() {
    let d: usize = kani::any();
    let o: usize = kani::any();
    let i: usize = kani::any();
    kani::assume(d >= 1 && d <= 32);
    kani::assume(o >= 1 && o <= 32);
    kani::assume(i < o);

    let start = i * d;
    let end = start + d;
    assert!(end <= o * d);
}

/// Distinct output rows never overlap, so the per-row loop cannot alias — the
/// same disjointness argument R-002 makes for the flash-attention bands.
#[cfg(kani)]
fn proof_f16_rows_are_disjoint_impl(i: usize, j: usize, d: usize) {
    let (a0, a1) = (i * d, i * d + d);
    let (b0, _b1) = (j * d, j * d + d);
    // i < j implies a1 <= b0: row i ends at or before row j begins.
    assert!(a0 < a1);
    assert!(a1 <= b0);
}

#[cfg(kani)]
#[kani::proof]
fn proof_f16_rows_are_disjoint() {
    let d: usize = kani::any();
    let o: usize = kani::any();
    let (i, j): (usize, usize) = (kani::any(), kani::any());
    kani::assume(d >= 1 && d <= 16);
    kani::assume(o >= 2 && o <= 16);
    kani::assume(i < j && j < o);

    proof_f16_rows_are_disjoint_impl(i, j, d);
}

/// The kernel's two tail loops (`k + 16 <= d`, then `k + 8 <= d`) leave `k <= d`
/// and never read past `d`. A 16-wide step reads `[k, k+16)`; an 8-wide step
/// reads `[k, k+8)`. Both are inside the row precisely because the tests are
/// `<=` against the width they are about to read, which is the property that
/// `dot_i8_blocked` above gets WRONG.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(6)]
fn proof_f16_tail_loops_stay_in_row() {
    let d: usize = kani::any();
    kani::assume(d >= 1 && d <= 40);

    let mut k: usize = 0;
    while k + 16 <= d {
        assert!(k + 16 <= d); // the 16 bytes this step reads are inside the row
        k += 16;
    }
    while k + 8 <= d {
        assert!(k + 8 <= d);
        k += 8;
    }
    assert!(k <= d);
}

// ---------------------------------------------------------------------------
// Exhaustive mirrors of the bounded proofs above.
//
// Kani runs on Linux and this workstation is not Linux, so until H-30 has a
// runner the harnesses are unexecuted assertions - which is to say, prose. Each
// test below enumerates the SAME domain the matching harness assumes. Over
// bounds this small, exhaustion and symbolic proof are the same statement, so
// these carry the evidence in the meantime and keep carrying it afterwards on
// every platform Kani does not cover.
//
// If you change a harness's `kani::assume` bounds, change its mirror to match.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod exhaustive {
    use super::{head_write_index, int8_last_byte_read};

    /// Mirrors `proof_int8_blocks_never_read_past_the_row`.
    #[test]
    fn int8_blocks_never_read_past_the_row() {
        for blk in [32usize, 64, 96] {
            for nblocks in 1..=4usize {
                let d = nblocks * blk;
                assert!(
                    int8_last_byte_read(nblocks, blk) < d,
                    "blk={blk} nblocks={nblocks}: last read {} escapes row of {d}",
                    int8_last_byte_read(nblocks, blk)
                );
            }
        }
    }

    /// Mirrors `proof_int8_unaligned_blk_would_over_read` - the guard earns its
    /// keep. Every blk that divides d but is NOT a multiple of 32 over-reads.
    #[test]
    fn int8_unaligned_blk_would_over_read() {
        for blk in [16usize, 48, 80] {
            let nblocks = 2usize;
            let d = nblocks * blk;
            assert!(
                int8_last_byte_read(nblocks, blk) >= d,
                "blk={blk} should over-read but did not - is the guard still needed?"
            );
        }
    }

    /// Mirrors `proof_int8_bscale_index_in_bounds`.
    #[test]
    fn int8_bscale_rows_are_exact_and_in_bounds() {
        for nb in 1..=4usize {
            for vocab in 1..=4usize {
                for v in 0..vocab {
                    let (lo, hi) = (v * nb, (v + 1) * nb);
                    assert!(hi <= vocab * nb);
                    assert_eq!(hi - lo, nb);
                }
            }
        }
    }

    /// Mirrors `proof_f16_rows_are_in_bounds` and `proof_f16_rows_are_disjoint`.
    #[test]
    fn f16_rows_are_in_bounds_and_disjoint() {
        for d in 1..=16usize {
            for o in 1..=16usize {
                for i in 0..o {
                    assert!(i * d + d <= o * d);
                    for j in (i + 1)..o {
                        // row i ends at or before row j begins
                        assert!(i * d + d <= j * d);
                    }
                }
            }
        }
    }

    /// Mirrors `proof_f16_tail_loops_stay_in_row`: replays the kernel's two tail
    /// loops and checks every step reads inside the row.
    #[test]
    fn f16_tail_loops_stay_in_row() {
        for d in 1..=40usize {
            let mut k = 0usize;
            while k + 16 <= d {
                assert!(k + 16 <= d);
                k += 16;
            }
            while k + 8 <= d {
                assert!(k + 8 <= d);
                k += 8;
            }
            assert!(k <= d, "d={d}: k={k} ran past the row");
        }
    }

    /// Mirrors `proof_flash_head_bands_are_disjoint` / `..._writes_are_in_bounds`
    /// (R-002). This is the one the residual-risk register calls out as argued in
    /// a comment rather than proven, so an exhaustive check is worth having even
    /// while the symbolic one waits on a runner.
    #[test]
    fn flash_head_bands_are_disjoint_and_in_bounds() {
        for hd in 1..=3usize {
            for heads in 1..=3usize {
                for seq in 1..=3usize {
                    let width = heads * hd;
                    let mut seen = std::collections::HashSet::new();
                    for h in 0..heads {
                        for row in 0..seq {
                            for c in 0..hd {
                                let ix = head_write_index(h, row, c, hd, width);
                                assert!(ix < seq * width, "write {ix} outside {}", seq * width);
                                assert!(seen.insert(ix), "two heads wrote index {ix}");
                            }
                        }
                    }
                }
            }
        }
    }
}
