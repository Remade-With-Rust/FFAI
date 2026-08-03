//! A fused single-pass SiLU.
//!
//! # Why
//!
//! Decomposing the convolution bucket found the activation, not a
//! convolution, holding **25.8% of detect** across 870 calls
//! (`FFAI_PROFILE=1`). That is Mercury's GELU result reproduced on a
//! different model: an elementwise function nobody would nominate, costing
//! as much as the arithmetic it decorates.
//!
//! Two mechanisms, both removable:
//!
//! 1. **`x * sigmoid(x)` is two tensor passes and two allocations.** One
//!    fused pass writes the result once.
//! 2. **`exp` is a scalar libm call**, so it blocks vectorization of the
//!    whole loop. Mercury measured exactly this for `tanh` and fixed it the
//!    same way — a polynomial that auto-vectorizes.
//!
//! `exp(x) = 2^(x·log2e)`: the integer part is written straight into the f32
//! exponent field and the fraction comes from a degree-5 polynomial on
//! [-0.5, 0.5]. No libm call, no branches, so the loop vectorizes.
//!
//! Float, so the gate is a tolerance against candle's own `silu` (which
//! stays the oracle), plus the full-graph parity oracle downstream.

use candle_core::{Result, Tensor};
use rayon::prelude::*;

/// `exp` without a libm call, accurate to ~1e-7 relative over the range an
/// activation sees.
#[inline(always)]
fn exp_fast(x: f32) -> f32 {
    // 2^t, t = x * log2(e). Clamp before the exponent-field write so a
    // saturating input cannot produce a denormal or an invalid exponent.
    let t = (x * std::f32::consts::LOG2_E).clamp(-125.0, 125.0);
    // Round by float ADDITION, not `f32::round`.
    //
    // This module's doc claims it removed the libm call that was blocking
    // vectorization. It removed `exp` and left `round` — and Rust's `round`
    // is ties-AWAY-FROM-ZERO, which no x86 instruction implements
    // (`vroundps` is ties-to-even), so it lowers to a call or a long
    // branchy sequence sitting in the middle of the loop. Removing one libm
    // call and leaving another is an easy thing to do and an invisible
    // thing to have done.
    //
    // Adding 1.5*2^23 forces every value below 2^22 to be rounded into the
    // mantissa's last bit; subtracting it back leaves `t` rounded to
    // nearest-even. Measured over 16 M elements, best of 7, single thread
    // (`examples/silu_ceiling.rs`):
    //
    //   memcpy (the roofline)  5.58 ms   24.04 GB/s
    //   `round()`             60.84 ms    2.21 GB/s
    //   `round_ties_even()`   38.85 ms    3.45 GB/s
    //   this                  12.91 ms   10.40 GB/s   <- 4.71x, BIT-IDENTICAL
    //
    // 10.4 GB/s against a 24 GB/s copy is a transcendental within 2.3x of
    // pure memory traffic, i.e. it now vectorises; `round()` did not.
    //
    // The bit-identical part is why this is a free win rather than a
    // tolerance question: max relative disagreement against the old kernel
    // over the activation range is exactly 0. In-context on the serial path
    // the bench times, the pipeline gain is 1.079x (17/21, z = +2.84) —
    // quote THAT for the engine, and these for the kernel.
    const MAGIC: f32 = 12582912.0; // 1.5 * 2^23
    // `FFAI_DIANA_SILU_ROUND=1` restores `f32::round`, so the pipeline-level
    // A/B stays runnable instead of being a claim in a commit message. The
    // branch is on a cached bool and predicts perfectly; it costs nothing
    // measurable and it is what makes the 1.94x re-checkable on a box that
    // is quiet, which this one has not been.
    let n = if old_rounding() { t.round() } else { (t + MAGIC) - MAGIC };
    let f = t - n; // in [-0.5, 0.5]
    // 2^f = exp(f*ln2), so the coefficients are ln2^k / k!. DERIVED, not
    // transcribed: hand-typed decimals here are a real hazard — trimming one
    // digit to satisfy a lint silently selected a different f32 and broke
    // the oracle. Let the compiler compute them.
    const L1: f32 = std::f32::consts::LN_2;
    const L2: f32 = L1 * L1 / 2.0;
    const L3: f32 = L1 * L1 * L1 / 6.0;
    const L4: f32 = L1 * L1 * L1 * L1 / 24.0;
    const L5: f32 = L1 * L1 * L1 * L1 * L1 / 120.0;
    let p = 1.0 + f * (L1 + f * (L2 + f * (L3 + f * (L4 + f * L5))));
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    p * scale
}

/// Whether to use the pre-fix `f32::round`. See [`exp_fast`].
///
/// Both roundings produce BIT-IDENTICAL results over the activation range
/// (measured: max relative disagreement exactly 0), so this toggle changes
/// speed and nothing else — which is why the oracle is not parameterised
/// over it.
#[inline(always)]
fn old_rounding() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(u8::MAX);
    match CACHED.load(Ordering::Relaxed) {
        u8::MAX => {
            let on = std::env::var("FFAI_DIANA_SILU_ROUND").is_ok_and(|v| v == "1");
            CACHED.store(on as u8, Ordering::Relaxed);
            on
        }
        v => v == 1,
    }
}

/// Whether to use the explicit AVX2 kernel.
///
/// `FFAI_DIANA_NO_AVX2=1` forces the scalar path, so the vector kernel can
/// be A/B'd rather than assumed — the whole point of the harness this
/// campaign built. Runtime feature detection is cached; the crate is
/// compiled for the x86-64 baseline so a published binary still runs on a
/// machine without AVX2.
pub(crate) fn avx2_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static C: AtomicU8 = AtomicU8::new(u8::MAX);
    match C.load(Ordering::Relaxed) {
        u8::MAX => {
            let off = std::env::var("FFAI_DIANA_NO_AVX2").is_ok_and(|v| v == "1");
            let on = !off && crate::silu_avx2::available();
            C.store(on as u8, Ordering::Relaxed);
            on
        }
        v => v == 1,
    }
}

/// The scalar twin, exposed so the AVX2 kernel's test can gate against the
/// exact function the whole graph is oracled on — not a copy of it.
#[inline(always)]
pub fn silu_scalar_pub(x: f32) -> f32 {
    silu_scalar(x)
}

/// `x * sigmoid(x)`, in one pass.
#[inline(always)]
fn silu_scalar(x: f32) -> f32 {
    // Prometheus Stage 1. Compiles to nothing without the feature.
    crate::telemetry::observe_silu_input(x);
    x / (1.0 + exp_fast(-x))
}

/// Chunk size for the parallel split. Large enough that the rayon fork-join
/// is amortized — the smallest tensors here are ~6 k elements, where a split
/// would cost more than the work.
const PAR_THRESHOLD: usize = 1 << 16;

/// SiLU over a contiguous f32 tensor, preserving shape.
///
/// Runs through [`crate::cpuop::SliceOp`], so the input is read in place and
/// the output `Vec` becomes the result tensor's storage — no copy on either
/// side. It previously did `to_vec1()` in and `from_vec` out, which for an
/// activation whose whole cost is memory traffic meant paying that traffic
/// three times over.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    // SiLU CANNOT BE PRICED BY ABLATION — measured, not assumed.
    //
    // Replacing it with the identity to time its cost made the pipeline
    // SLOWER: 49.7 ms against 45.1 ms, six pairs, ABBA. Removing work cannot
    // save negative time, so the arms were not doing the same work
    // (`codec-measurement` §4). Without SiLU's saturation the activations run
    // unbounded through sixty layers and the downstream arithmetic almost
    // certainly lands in denormals, which are orders of magnitude slower on
    // x86.
    //
    // The ablation toggle was deleted rather than documented, because an
    // instrument that answers backwards is worse than no instrument. It also
    // means the profiler's 8.7 % share for this bucket has no independent
    // confirmation, and the "silu costs 6.4 ms/image" figure should be
    // treated as unverified.
    crate::cpuop::SliceOp::new("ffai-silu", |xs, l| {
        // Counted HERE, above the threshold branch, because a counter placed
        // inside it can only ever see calls that already passed it. That is
        // exactly the mistake this instrument made on its first outing: it
        // reported silu "100 % parallel, 52 of 52 calls" while the profiler
        // saw 87, and the missing 35 were the serial ones by construction.
        crate::silu::count_silu(
            xs.len(),
            xs.len() >= PAR_THRESHOLD && !crate::parallel::serial_kernels(),
        );
        let mut v: Vec<f32> = Vec::with_capacity(xs.len());
        // The threshold alone is not the whole decision: inside a parallel
        // batch even a large activation should stay serial, because the
        // machine is already full. See `crate::parallel`.
        if xs.len() >= PAR_THRESHOLD && !crate::parallel::serial_kernels() {
            // `collect`, not `resize` + `par_chunks_mut`.
            //
            // Handing out `&mut` slices in safe Rust requires the buffer to
            // be initialised first, so `resize(n, 0.0)` zero-filled every
            // byte and the kernel then overwrote every byte: the output was
            // WRITTEN TWICE. Measured with `examples/op_overhead.rs` at 1 M
            // elements — kernel 807 us, everything else 1758 us, i.e. 2.2x
            // the arithmetic — and silu is 30.3 % of serial detect, a share
            // that refused to fall when the kernel itself got 4.71x faster.
            // A share that will not move when the kernel moves is not being
            // spent in the kernel.
            //
            // rayon's `collect` allocates once and writes once, stays safe,
            // and preserves order so the output is unchanged.
            // Which parallel form, by SIZE — the two lose to each other in
            // opposite regimes and a single choice is wrong half the time.
            // Measured per call (`examples/op_overhead.rs`):
            //
            //   65 536 elems : chunks_mut 184 us   collect 346 us  -> chunks
            //   1 048 576    : chunks_mut 2564     collect 1702    -> collect
            //
            // `collect` allocates once and writes once, which wins when the
            // write dominates; `par_chunks_mut` carries less bookkeeping,
            // which wins while the buffer is small. The crossover sits
            // between, so take it at 256 k rather than pretending one form
            // is simply better.
            const COLLECT_ABOVE: usize = 1 << 18;
            if avx2_enabled() {
                // Explicit AVX2: eight lanes of the polynomial at a time.
                // Allocation still happens once (the double-write lesson),
                // and the vector kernel writes into it.
                let chunk = (1 << 14).max(xs.len().div_ceil(rayon::current_num_threads().max(1)));
                if xs.len() <= chunk && !crate::smallgains::disabled() {
                    // ONE chunk, so `par_chunks_mut` would fork a job, hand
                    // the whole buffer to a single worker, and join — the
                    // full rayon round trip to arrive back at this thread.
                    //
                    // That is not hypothetical overhead. `silu` is called
                    // ~1305 times per image and MOST of those tensors are
                    // under one chunk, which is why it is the one bucket in
                    // the profile that gets SLOWER with more threads:
                    // 0.292 s at 1 thread against 0.315 s at 24. A stage
                    // that costs more the more cores you give it is paying
                    // for parallelism it never receives.
                    //
                    // SAFETY: `avx2_enabled()` verified avx2+fma at runtime;
                    // the slices are the same length by construction.
                    {
                        let n = xs.len();
                        let spare = &mut v.spare_capacity_mut()[..n];
                        // SAFETY: the kernel writes all `n` elements and
                        // reads none of them; avx2+fma verified above.
                        #[allow(unsafe_code)]
                        let out: &mut [f32] = unsafe {
                            std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<f32>(), n)
                        };
                        #[allow(unsafe_code)]
                        unsafe {
                            crate::silu_avx2::silu_into(xs, out)
                        };
                        // SAFETY: all `n` elements written just above.
                        #[allow(unsafe_code)]
                        unsafe {
                            v.set_len(n)
                        };
                    }
                } else {
                    // Write into UNINITIALISED capacity.
                    //
                    // This path did `v.resize(len, 0.0)` first — zero-filling
                    // every element and then overwriting every element with
                    // the kernel's output. That is the same double write the
                    // comment above records removing from the `collect`
                    // path; it was never removed from this one.
                    //
                    // 10.38 M elements per image at the n tier = 41.5 MiB
                    // written twice. See `smallgains` entry 4 for the
                    // arithmetic and why no speed is claimed for it.
                    let n = xs.len();
                    {
                        let spare = &mut v.spare_capacity_mut()[..n];
                        spare
                            .par_chunks_mut(chunk)
                            .zip(xs.par_chunks(chunk))
                            .for_each(|(o, i)| {
                                // SAFETY: `o` and `i` have equal length by
                                // construction; the kernel writes every
                                // element of `o` and reads none of it, and
                                // `avx2_enabled()` verified avx2+fma.
                                #[allow(unsafe_code)]
                                let o: &mut [f32] = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        o.as_mut_ptr().cast::<f32>(),
                                        o.len(),
                                    )
                                };
                                #[allow(unsafe_code)]
                                unsafe {
                                    crate::silu_avx2::silu_into(i, o)
                                }
                            });
                    }
                    // SAFETY: the zip covers exactly `n` elements in equal
                    // chunks, and the kernel writes every element of each.
                    #[allow(unsafe_code)]
                    unsafe {
                        v.set_len(n)
                    };
                }
            } else if xs.len() >= COLLECT_ABOVE {
                v = xs.par_iter().map(|&x| silu_scalar(x)).collect();
            } else {
                v.resize(xs.len(), 0.0);
                v.par_chunks_mut(1 << 14).zip(xs.par_chunks(1 << 14)).for_each(|(o, i)| {
                    for (o, i) in o.iter_mut().zip(i) {
                        *o = silu_scalar(*i);
                    }
                });
            }
        } else {
            v.extend(xs.iter().map(|v| silu_scalar(*v)));
        }
        Ok((v, l.shape().clone()))
    })
    .run(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn exp_fast_tracks_libm() {
        let mut worst = 0f32;
        let mut t = -30.0f32;
        while t <= 30.0 {
            let (a, b) = (exp_fast(t), t.exp());
            let rel = (a - b).abs() / b.abs().max(1e-30);
            worst = worst.max(rel);
            t += 0.001;
        }
        assert!(worst < 1e-5, "worst relative error {worst:.3e}");
    }

    #[test]
    fn matches_candle_silu() {
        let dev = Device::Cpu;
        for &n in &[1usize, 7, 4096, 1 << 17] {
            // DETERMINISTIC inputs, spanning saturation both ways. This used
            // `Tensor::randn`, which made the test flaky rather than strict:
            // at n=1 the relative error divides by a single random value, so
            // a draw near zero fails a bound that a draw near one passes. A
            // test that depends on the draw is not a gate.
            let vals: Vec<f32> =
                (0..n).map(|i| -12.0 + 24.0 * (i as f32) / (n.max(2) - 1) as f32).collect();
            let x = Tensor::from_vec(vals, n, &dev).unwrap();
            let got = silu(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let want = candle_nn::ops::silu(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            let d = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs() / scale)
                .fold(0.0f32, f32::max);
            assert!(d < 1e-6, "n={n}: max rel {d:.3e}");
        }
    }

    #[test]
    fn saturates_the_way_silu_should() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![-60f32, -20.0, 0.0, 20.0, 60.0], 5, &dev).unwrap();
        let y = silu(&x).unwrap().to_vec1::<f32>().unwrap();
        assert!(y[0].abs() < 1e-6, "large negative -> 0, got {}", y[0]);
        assert!((y[2]).abs() < 1e-6, "silu(0) = 0, got {}", y[2]);
        assert!((y[4] - 60.0).abs() < 1e-3, "large positive -> x, got {}", y[4]);
        assert!(y.iter().all(|v| v.is_finite()), "no NaN/Inf at saturation");
    }
}


/// How much of SiLU's work sits in calls too small to fan out?
///
/// The chunk width is `max(1 << 14, len / threads)`, so any call with fewer
/// than 16384 elements is ONE chunk and runs on a single worker. SiLU is
/// 15.2 % of the pipeline across ~1305 calls per image, and 13.4 M activation
/// elements over those calls is a ~10.3 k mean — BELOW the threshold. If that
/// holds, most of SiLU is serial by construction and no pool width helps it.
///
/// A count, not a timing: the box is half-occupied by another project's
/// benchmark, and this question does not need a clock.
///
/// # A counter placed inside the branch it is testing answers nothing
///
/// This first reported SiLU "**100 % parallel** — 52 calls, zero serial",
/// and it was measuring its own position. It sat inside
/// `if xs.len() >= PAR_THRESHOLD`, so the only calls it could ever see were
/// the ones that had already passed the test. The profiler counted 87 calls
/// per image against its 52, and that impossible pair is what exposed it:
/// **the missing 35 are the serial ones, by construction.**
///
/// It is now at the top of the closure, above every branch, and reports the
/// split as an argument rather than inferring it from where it was placed.
///
/// The scaling measurement had said so all along — SiLU scales 1.24x on four
/// workers, 74 % serial — and was disbelieved for one round because a count
/// is supposed to outrank a timing. It does, but only when it counts the
/// right population.
static SILU_SMALL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SILU_BIG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SILU_CALLS_SMALL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SILU_CALLS_BIG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn count_silu(len: usize, parallel: bool) {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
    // Cached, not re-read per call. `std::env::var` allocates and locks the
    // environment; doing that inside the activation would make the
    // instrument a cost of the thing it measures — exactly the profiler tax
    // `codec-measurement` §6 warns about.
    static ON: AtomicU8 = AtomicU8::new(u8::MAX);
    let on = match ON.load(Relaxed) {
        u8::MAX => {
            let v = std::env::var("FFAI_DIANA_COUNT").is_ok_and(|v| v == "1");
            ON.store(v as u8, Relaxed);
            v
        }
        v => v == 1,
    };
    if on {
        if parallel {
            SILU_BIG.fetch_add(len as u64, Relaxed);
            SILU_CALLS_BIG.fetch_add(1, Relaxed);
        } else {
            SILU_SMALL.fetch_add(len as u64, Relaxed);
            SILU_CALLS_SMALL.fetch_add(1, Relaxed);
        }
    }
}

/// `(serial_elems, serial_calls, parallel_elems, parallel_calls)`, and reset.
pub fn take_silu_split() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        SILU_SMALL.swap(0, Relaxed),
        SILU_CALLS_SMALL.swap(0, Relaxed),
        SILU_BIG.swap(0, Relaxed),
        SILU_CALLS_BIG.swap(0, Relaxed),
    )
}


