//! SIMD twins of every scalar kernel on the decode path, each one bit-identical to its oracle.
//!
//! | kernel | scalar oracle | gate |
//! |---|---|---|
//! | [`ternary_gemv`] | `rusty_ternary::kernel::gemv_codes` | `assert_eq!` (integers) |
//! | [`ternary_gemm`] (prefill) | `rusty_ternary::kernel::gemm_codes` | `assert_eq!` |
//! | [`i8f_matvec`] (head screen) | [`i8f_matvec_scalar`] | bits |
//! | [`i8_gemv`] | [`i8_gemv_scalar`] | `assert_eq!` |
//! | [`f16_matvec_bytes`], [`f16_matvec_strided`] | [`num::dot16_f16_bytes`], [`num::dot16_f16`] | bits |
//! | [`weighted_sum_f16`] | [`num::weighted_sum_f16`] | bits |
//! | [`act_quant`] | `rusty_ternary::quant::act_quant` | `assert_eq!` |
//! | [`rms_norm`] | [`num::rms_norm`] | bits |
//! | [`softmax`] | [`num::softmax`] | bits |
//! | [`f16_bytes_to_f32`], [`f32_to_f16`], [`round_f16_slice`] | `half` | bits (exhaustive) |
//! | [`finish_rows`], [`relu2_mul`], [`add_assign`], [`rope_neox`] | [`num`]'s elementwise fns | bits |
//!
//! Tiers: `ssse3`, `avx2` (+`f16c`) and `avx2vnni` on x86-64, chosen at run time; `neon` and
//! `neondot` on aarch64 (the `vnni` / `dot` tiers change only the ternary GEMV and GEMM, every
//! other kernel runs its `avx2` / `neon` twin);
//! `simd128` on wasm32 when the module is built with it (wasm has no run-time detection). The
//! scalar oracle is always available and is what any other target runs.
//!
//! Bit-identity is by construction: each twin performs the oracle's operations, per element, in
//! the oracle's order — multiplies and adds kept separate (no fused multiply-add anywhere), the
//! 16-lane dot product and 4-lane `f64` sums laid out exactly as [`num::dot16`] and
//! [`num::sum4_f64`] define them, rounding by round-to-nearest-even. The one exception is a NaN's
//! payload, which is not specified and never reaches a token decision.
//!
//! [`Isa`] can only be made by run-time detection (or a checked override), so a twin is only
//! ever called on a CPU that has its instructions; that is the safety argument for every
//! `unsafe` call below, together with the bounds each entry point asserts first.

use std::sync::OnceLock;

use crate::numerics as num;

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;
#[cfg(target_arch = "x86_64")]
mod x86;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Tier {
    Scalar = 0,
    Ssse3 = 1,
    Avx2 = 2,
    Neon = 3,
    Simd128 = 4,
    Avx2Vnni = 5,
    NeonDot = 6,
}

/// Every tier, narrowest first within an architecture (`Isa::best` takes the last supported).
const TIERS: [Tier; 7] = [
    Tier::Scalar,
    Tier::Ssse3,
    Tier::Avx2,
    Tier::Avx2Vnni,
    Tier::Neon,
    Tier::NeonDot,
    Tier::Simd128,
];

impl Tier {
    const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Ssse3 => "ssse3",
            Self::Avx2 => "avx2",
            Self::Neon => "neon",
            Self::Simd128 => "simd128",
            Self::Avx2Vnni => "avx2vnni",
            Self::NeonDot => "neondot",
        }
    }

    fn supported(self) -> bool {
        match self {
            Self::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Self::Ssse3 => std::is_x86_feature_detected!("ssse3"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => {
                std::is_x86_feature_detected!("ssse3")
                    && std::is_x86_feature_detected!("avx2")
                    && std::is_x86_feature_detected!("f16c")
                    && std::is_x86_feature_detected!("fma")
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx2Vnni => Self::Avx2.supported() && std::is_x86_feature_detected!("avxvnni"),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Self::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Self::NeonDot => {
                Self::Neon.supported() && std::arch::is_aarch64_feature_detected!("dotprod")
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Self::Simd128 => true,
            #[allow(
                unreachable_patterns,
                reason = "which arms exist depends on the target"
            )]
            _ => false,
        }
    }
}

/// An instruction set this CPU supports. Only run-time detection makes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Isa(Tier);

impl Isa {
    /// The scalar oracle: supported everywhere.
    pub const SCALAR: Self = Self(Tier::Scalar);

    /// Every tier this CPU supports, scalar first, widest last.
    #[must_use]
    pub fn all() -> Vec<Self> {
        TIERS
            .iter()
            .copied()
            .filter(|t| t.supported())
            .map(Self)
            .collect()
    }

    /// The widest supported tier.
    #[must_use]
    pub fn best() -> Self {
        Self::all().last().copied().unwrap_or(Self::SCALAR)
    }

    /// The tier called `name`, when this CPU supports it.
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        TIERS
            .iter()
            .copied()
            .find(|t| t.name() == name)
            .filter(|t| t.supported())
            .map(Self)
    }

    /// `MINERVA_ISA` when set (and supported), else [`Self::best`]; read once per process.
    ///
    /// # Panics
    /// When `MINERVA_ISA` names a tier this CPU does not support (an A/B arm silently running
    /// another tier would be a measurement of nothing).
    #[must_use]
    pub fn from_env() -> Self {
        static I: OnceLock<Isa> = OnceLock::new();
        *I.get_or_init(|| match std::env::var("MINERVA_ISA") {
            Ok(n) => Self::named(&n).unwrap_or_else(|| {
                panic!("MINERVA_ISA={n}: not supported here ({:?})", Self::all())
            }),
            Err(_) => Self::best(),
        })
    }

    /// Its name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.0.name()
    }
}

/// The kernel-reach census: per kernel and tier, the elements each call processed, counted once
/// per call into per-thread counters (no shared cache line, no contention), summed on read.
pub mod census {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::{Arc, Mutex};

    use super::{Isa, TIERS};

    /// The kernels counted.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Kernel {
        /// I2_S GEMV (weights).
        TernaryGemv = 0,
        /// f16 matrix-vector (head, attention scores; MACs).
        F16Matvec = 1,
        /// Attention value sum (MACs).
        WeightedSum = 2,
        /// Activation quantisation (elements).
        ActQuant = 3,
        /// RMS norm (elements).
        RmsNorm = 4,
        /// Softmax (elements).
        Softmax = 5,
        /// f16 conversions (elements).
        F16Convert = 6,
        /// finish, relu2_mul, add_assign, rope (elements).
        Elementwise = 7,
        /// int8 x int8 GEMV (MACs).
        I8Gemv = 8,
        /// int8 x f32 mat-vec, the head screen (MACs).
        I8fMatvec = 9,
        /// I2_S GEMM, prefill (weights x tokens).
        TernaryGemm = 10,
    }

    /// Kernel names, in [`Kernel`] order.
    pub const NAMES: [&str; 11] = [
        "ternary_gemv",
        "f16_matvec",
        "weighted_sum",
        "act_quant",
        "rms_norm",
        "softmax",
        "f16_convert",
        "elementwise",
        "i8_gemv",
        "i8f_matvec",
        "ternary_gemm",
    ];
    const N: usize = NAMES.len() * TIERS.len();

    struct Slot([AtomicU64; N]);
    static REGISTRY: Mutex<Vec<Arc<Slot>>> = Mutex::new(Vec::new());
    thread_local! {
        static MINE: Arc<Slot> = {
            let s = Arc::new(Slot(std::array::from_fn(|_| AtomicU64::new(0))));
            REGISTRY.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(Arc::clone(&s));
            s
        };
    }

    /// Whether counting is on: off by default, so a dispatch pays one relaxed load and a
    /// predictable branch instead of a thread-local lookup and an atomic add.
    static ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    /// Turn counting on or off (`minerva generate --census`, the census tests).
    pub fn enable(on: bool) {
        ON.store(on, Relaxed);
    }

    /// Count `n` elements of `k` run at `isa` (when counting is on).
    #[inline]
    pub fn add(k: Kernel, isa: Isa, n: usize) {
        if ON.load(Relaxed) {
            add_slow(k, isa, n);
        }
    }

    #[cold]
    #[inline(never)]
    fn add_slow(k: Kernel, isa: Isa, n: usize) {
        MINE.with(|s| s.0[k as usize * TIERS.len() + isa.0 as usize].fetch_add(n as u64, Relaxed));
    }

    /// Totals so far: `(kernel, tier, elements)` for every non-zero cell.
    #[must_use]
    pub fn snapshot() -> Vec<(&'static str, &'static str, u64)> {
        let reg = REGISTRY
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut out = Vec::new();
        for (k, name) in NAMES.iter().enumerate() {
            // Cells are indexed by the tier's discriminant (as `add` writes them), not by its
            // position in `TIERS`.
            for tier in &TIERS {
                let t = *tier as usize;
                let n: u64 = reg
                    .iter()
                    .map(|s| s.0[k * TIERS.len() + t].load(Relaxed))
                    .sum();
                if n > 0 {
                    out.push((*name, tier.name(), n));
                }
            }
        }
        out
    }

    /// Zero every counter.
    pub fn reset() {
        for s in REGISTRY
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
        {
            for c in &s.0 {
                c.store(0, Relaxed);
            }
        }
    }
}

use census::Kernel;

/// `Σ a`, which the SIMD GEMVs subtract once a row (they multiply codes 0/1/2, not trits).
#[must_use]
pub fn act_sum(a: &[i8]) -> i32 {
    a.iter().map(|&v| i32::from(v)).sum()
}

/// `y[r] = Σ_j W[r, j] · a[j]` over `y.len()` whole rows of I2_S codes; `a_sum` is
/// [`act_sum`]`(a)`.
///
/// # Panics
/// When `a.len()` is not a whole number of 128-weight blocks or `codes` is not `y.len()` rows.
pub fn ternary_gemv(isa: Isa, codes: &[u8], a: &[i8], a_sum: i32, y: &mut [i32]) {
    let cols = a.len();
    assert!(cols.is_multiple_of(rusty_ternary::i2s::BLOCK) && codes.len() == y.len() * cols / 4);
    debug_assert_eq!(a_sum, act_sum(a));
    census::add(Kernel::TernaryGemv, isa, y.len() * cols);
    match isa.0 {
        // SAFETY: `Isa` holds `Avx2` / `Ssse3` only when detected; the shapes are asserted above.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2Vnni => unsafe { x86::gemv_vnni(codes, a, a_sum, y) },
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 => unsafe { x86::gemv_avx2(codes, a, a_sum, y) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::gemv_ssse3(codes, a, a_sum, y) },
        // SAFETY: NEON detected; shapes asserted.
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::NeonDot => unsafe { neon::gemv_dot(codes, a, a_sum, y) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon => unsafe { neon::gemv(codes, a, a_sum, y) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::gemv(codes, a, a_sum, y) },
        _ => rusty_ternary::kernel::gemv_codes(codes, cols, a, y).expect("shapes asserted"),
    }
}

/// `y[r] = Σ_j q[r·n + j] · a[j]` (`n = a.len()`), int8 by int8, exact. Every value must lie in
/// `-127..=127` (never -128): the x86 twins multiply `|q|` by `a·sign(q)` in `pmaddubsw`, whose
/// pair sum `2 · 127²` then stays inside an `i16`.
///
/// # Panics
/// When `q` is not `y.len()` rows of `a.len()`.
pub fn i8_gemv(isa: Isa, q: &[i8], a: &[i8], y: &mut [i32]) {
    let n = a.len();
    assert_eq!(q.len(), y.len() * n);
    debug_assert!(!a.contains(&i8::MIN) && !q.contains(&i8::MIN));
    census::add(Kernel::I8Gemv, isa, q.len());
    match isa.0 {
        // SAFETY: tier detected; shapes asserted.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni => unsafe { x86::i8_gemv_avx2(q, a, y) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::i8_gemv_ssse3(q, a, y) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon | Tier::NeonDot => unsafe { neon::i8_gemv(q, a, y) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::i8_gemv(q, a, y) },
        _ => i8_gemv_scalar(q, a, y),
    }
}

/// `out[r] = dot16(q row r, x)`: int8 weights times `f32`, in the canonical 16-lane order
/// ([`num::dot16`]). With `x` holding `f16` values every product is exact (8 + 11 bits), so the
/// twins equal the oracle bit for bit.
///
/// # Panics
/// When `q` is not `out.len()` rows of `x.len()`.
pub fn i8f_matvec(isa: Isa, q: &[i8], x: &[f32], out: &mut [f32]) {
    assert_eq!(q.len(), out.len() * x.len());
    census::add(Kernel::I8fMatvec, isa, q.len());
    match isa.0 {
        // SAFETY: tier detected; shapes asserted.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni => unsafe { x86::i8f_matvec_avx2(q, x, out) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::i8f_matvec_ssse3(q, x, out) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon | Tier::NeonDot => unsafe { neon::i8f_matvec(q, x, out) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::i8f_matvec(q, x, out) },
        _ => i8f_matvec_scalar(q, x, out),
    }
}

/// The oracle of [`i8f_matvec`].
pub fn i8f_matvec_scalar(q: &[i8], x: &[f32], out: &mut [f32]) {
    let mut row = vec![0f32; x.len()];
    for (o, qr) in out.iter_mut().zip(q.chunks_exact(x.len())) {
        for (r, &v) in row.iter_mut().zip(qr) {
            *r = f32::from(v);
        }
        *o = num::dot16(&row, x);
    }
}

/// The tail of an int8 16-lane dot product: lanes stored, the rest in the oracle's order.
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_endian = "little"),
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    expect(
        dead_code,
        reason = "only the SIMD tiers call it; this target has none"
    )
)]
pub(crate) fn i8f_finish(mut acc: [f32; num::LANES], q: &[i8], x: &[f32], from: usize) -> f32 {
    for (j, i) in (from..x.len()).enumerate() {
        acc[j] += f32::from(q[i]) * x[i];
    }
    num::reduce16(acc)
}

/// The oracle of [`i8_gemv`].
pub fn i8_gemv_scalar(q: &[i8], a: &[i8], y: &mut [i32]) {
    for (o, row) in y.iter_mut().zip(q.chunks_exact(a.len())) {
        *o = row
            .iter()
            .zip(a)
            .map(|(&w, &v)| i32::from(w) * i32::from(v))
            .sum();
    }
}

/// Prefill's product: `y[r·m + t] = Σ_j W[r, j] · a[t·cols + j]` for the `codes.len() / (cols/4)`
/// whole rows of I2_S codes and `m` tokens of activations; `a_sums[t]` is [`act_sum`] of token
/// `t`. The output is row-major by ROW (`[rows, m]`), so threads split the rows and each writes
/// one contiguous block. Every code block is unpacked once and applied to a tile of tokens.
///
/// # Panics
/// When the shapes disagree or `cols` is not a whole number of 128-weight blocks.
pub fn ternary_gemm(
    isa: Isa,
    codes: &[u8],
    cols: usize,
    m: usize,
    a: &[i8],
    a_sums: &[i32],
    y: &mut [i32],
) {
    assert!(cols.is_multiple_of(rusty_ternary::i2s::BLOCK) && codes.len().is_multiple_of(cols / 4));
    let rows = codes.len() / (cols / 4);
    assert!(a.len() == m * cols && a_sums.len() == m && y.len() == rows * m);
    census::add(Kernel::TernaryGemm, isa, rows * cols * m);
    match isa.0 {
        // SAFETY: tier detected; shapes asserted.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2Vnni => unsafe { x86::gemm_vnni(codes, cols, m, a, a_sums, y) },
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 => unsafe { x86::gemm_avx2(codes, cols, m, a, a_sums, y) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::gemm_ssse3(codes, cols, m, a, a_sums, y) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::NeonDot => unsafe { neon::gemm_dot(codes, cols, m, a, a_sums, y) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon => unsafe { neon::gemm(codes, cols, m, a, a_sums, y) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::gemm(codes, cols, m, a, a_sums, y) },
        _ => {
            let rpc = cols / 4;
            for (r, yr) in y.chunks_exact_mut(m).enumerate() {
                for (t, o) in yr.iter_mut().enumerate() {
                    let mut one = [0i32];
                    rusty_ternary::kernel::gemv_codes(
                        &codes[r * rpc..(r + 1) * rpc],
                        cols,
                        &a[t * cols..(t + 1) * cols],
                        &mut one,
                    )
                    .expect("shapes asserted");
                    *o = one[0];
                }
            }
        }
    }
}

/// `out[r] = dot16(f16 row r, x)` for `out.len()` contiguous rows of `x.len()` little-endian
/// `f16` values in `mat`.
///
/// # Panics
/// When `mat` holds fewer than `out.len()` rows.
pub fn f16_matvec_bytes(isa: Isa, mat: &[u8], x: &[f32], out: &mut [f32]) {
    let row = 2 * x.len();
    assert!(mat.len() >= out.len() * row);
    census::add(Kernel::F16Matvec, isa, out.len() * x.len());
    match isa.0 {
        // SAFETY: tier detected; `out.len()` rows of `row` bytes lie inside `mat` (asserted).
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni => unsafe { x86::f16_matvec_avx2(mat.as_ptr(), row, x, out) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::f16_matvec_ssse3(mat.as_ptr(), row, x, out) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon | Tier::NeonDot => unsafe { neon::f16_matvec(mat.as_ptr(), row, x, out) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::f16_matvec(mat.as_ptr(), row, x, out) },
        _ => {
            for (r, o) in out.iter_mut().enumerate() {
                *o = num::dot16_f16_bytes(&mat[r * row..(r + 1) * row], x);
            }
        }
    }
}

/// `out[r] = dot16(v[offset + r·stride ..][..x.len()], x)`, `v` as `f16` bits (the attention
/// scores over a KV cache laid out `[position][kv_dim]`).
///
/// # Panics
/// When the last row runs past `v`.
pub fn f16_matvec_strided(
    isa: Isa,
    v: &[u16],
    offset: usize,
    stride: usize,
    x: &[f32],
    out: &mut [f32],
) {
    if out.is_empty() {
        return;
    }
    assert!(offset + (out.len() - 1) * stride + x.len() <= v.len());
    census::add(Kernel::F16Matvec, isa, out.len() * x.len());
    let base = v[offset..].as_ptr().cast::<u8>();
    match isa.0 {
        // SAFETY: tier detected; every row `[offset + r·stride, + x.len())` is inside `v`.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni => unsafe { x86::f16_matvec_avx2(base, 2 * stride, x, out) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::f16_matvec_ssse3(base, 2 * stride, x, out) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon | Tier::NeonDot => unsafe { neon::f16_matvec(base, 2 * stride, x, out) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::f16_matvec(base, 2 * stride, x, out) },
        _ => {
            let _ = base;
            for (r, o) in out.iter_mut().enumerate() {
                let s = offset + r * stride;
                *o = num::dot16_f16(&v[s..s + x.len()], x);
            }
        }
    }
}

/// Grouped-query attention scores: for each of the `x.len() / len` query heads `j` sharing one
/// KV head, `out[j·out_stride + t] = dot16(v[offset + t·stride ..][..len], x[j·len ..][..len])`
/// for `t < n`. Bit-identical to [`f16_matvec_strided`] per head (each head keeps its own lanes
/// in their order); the grouped twins widen each `f16` element once for the whole group.
///
/// # Panics
/// When a row or an output runs past its slice.
#[expect(clippy::too_many_arguments, reason = "a strided group of heads")]
pub fn f16_matvec_strided_group(
    isa: Isa,
    v: &[u16],
    offset: usize,
    stride: usize,
    len: usize,
    x: &[f32],
    n: usize,
    out: &mut [f32],
    out_stride: usize,
) {
    let g = x.len() / len;
    if n == 0 || g == 0 {
        return;
    }
    assert!(x.len() == g * len && offset + (n - 1) * stride + len <= v.len());
    assert!((g - 1) * out_stride + n <= out.len());
    match isa.0 {
        // SAFETY: tier detected; rows and outputs asserted in bounds; `len % 16 == 0`, `g == 4`.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni if g == 4 && len.is_multiple_of(16) => {
            census::add(Kernel::F16Matvec, isa, g * n * len);
            unsafe {
                x86::f16_matvec_group4_avx2(
                    v[offset..].as_ptr(),
                    stride,
                    len,
                    x,
                    n,
                    out,
                    out_stride,
                )
            }
        }
        _ => {
            for j in 0..g {
                let o = &mut out[j * out_stride..j * out_stride + n];
                f16_matvec_strided(isa, v, offset, stride, &x[j * len..(j + 1) * len], o);
            }
        }
    }
}

/// Grouped-query attention values: for each head `j` of the group,
/// `out[j·len + d] = Σ_t p[j·p_stride + t] · v[offset + t·stride + d]` for `t < n`.
/// Bit-identical to [`weighted_sum_f16`] per head; the grouped twins widen each `f16` once.
///
/// # Panics
/// When a row or an input runs past its slice.
#[expect(clippy::too_many_arguments, reason = "a strided group of heads")]
pub fn weighted_sum_f16_group(
    isa: Isa,
    p: &[f32],
    p_stride: usize,
    n: usize,
    v: &[u16],
    offset: usize,
    stride: usize,
    out: &mut [f32],
    len: usize,
) {
    let g = out.len() / len;
    if g == 0 {
        return;
    }
    assert!(out.len() == g * len && (g - 1) * p_stride + n <= p.len());
    if n == 0 {
        out.fill(0.0);
        return;
    }
    assert!(offset + (n - 1) * stride + len <= v.len());
    match isa.0 {
        // SAFETY: tier detected; rows and inputs asserted in bounds; `len % 16 == 0`, `g == 4`.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni if g == 4 && len.is_multiple_of(16) => {
            census::add(Kernel::WeightedSum, isa, g * n * len);
            unsafe {
                x86::weighted_sum_group4_avx2(
                    p,
                    p_stride,
                    n,
                    v[offset..].as_ptr(),
                    stride,
                    out,
                    len,
                )
            }
        }
        _ => {
            for j in 0..g {
                weighted_sum_f16(
                    isa,
                    &p[j * p_stride..j * p_stride + n],
                    v,
                    offset,
                    stride,
                    &mut out[j * len..(j + 1) * len],
                );
            }
        }
    }
}

/// `out[d] = Σ_t p[t] · v[offset + t·stride + d]` (`v` as `f16` bits), each element summed in
/// `t` order from zero.
///
/// # Panics
/// When the last row runs past `v`.
pub fn weighted_sum_f16(
    isa: Isa,
    p: &[f32],
    v: &[u16],
    offset: usize,
    stride: usize,
    out: &mut [f32],
) {
    if p.is_empty() {
        out.fill(0.0);
        return;
    }
    assert!(offset + (p.len() - 1) * stride + out.len() <= v.len());
    census::add(Kernel::WeightedSum, isa, p.len() * out.len());
    let base = v[offset..].as_ptr();
    match isa.0 {
        // SAFETY: tier detected; every row is inside `v` (asserted).
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni => unsafe { x86::weighted_sum_avx2(p, base, stride, out) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::weighted_sum_ssse3(p, base, stride, out) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon | Tier::NeonDot => unsafe { neon::weighted_sum(p, base, stride, out) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::weighted_sum(p, base, stride, out) },
        _ => {
            let _ = base;
            num::weighted_sum_f16(p, &v[offset..], stride, out);
        }
    }
}

/// `rusty_ternary::quant::act_quant`: `out = clamp(round_ties_even(x · s), -128, 127)` with
/// `s = 127 / max(max|x|, 1e-5)`; returns `s`. A NaN quantises to 0.
///
/// # Panics
/// When the lengths differ.
pub fn act_quant(isa: Isa, x: &[f32], out: &mut [i8]) -> f32 {
    act_quant_sum(isa, x, out).0
}

/// [`act_quant`] and [`act_sum`] of its codes, in one pass (the twins add the code lanes they
/// already hold; the scalar arm is the two oracles in turn).
///
/// # Panics
/// When the lengths differ.
pub fn act_quant_sum(isa: Isa, x: &[f32], out: &mut [i8]) -> (f32, i32) {
    assert_eq!(x.len(), out.len());
    census::add(Kernel::ActQuant, isa, x.len());
    match isa.0 {
        // SAFETY: tier detected; equal lengths asserted.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni => unsafe { x86::act_quant_avx2(x, out) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::act_quant_ssse3(x, out) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon | Tier::NeonDot => unsafe { neon::act_quant(x, out) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::act_quant(x, out) },
        _ => {
            let s = rusty_ternary::quant::act_quant(x, out);
            (s, act_sum(out))
        }
    }
}

/// The quantisation scale [`act_quant`] uses, from the largest magnitude.
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_endian = "little"),
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    expect(
        dead_code,
        reason = "only the SIMD tiers call it; this target has none"
    )
)]
#[inline]
#[must_use]
pub(crate) fn act_scale(amax: f32) -> f32 {
    127.0 / amax.max(1e-5)
}

/// One activation, quantised (the tails of every twin).
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_endian = "little"),
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    expect(
        dead_code,
        reason = "only the SIMD tiers call it; this target has none"
    )
)]
#[inline]
#[must_use]
pub(crate) fn quant_one(v: f32, s: f32) -> i8 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped to an i8's range; a NaN casts to 0"
    )]
    let q = (v * s).round_ties_even().clamp(-128.0, 127.0) as i8;
    q
}

/// [`num::rms_norm`].
///
/// # Panics
/// When the lengths differ.
pub fn rms_norm(isa: Isa, x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    assert!(x.len() == w.len() && x.len() == out.len());
    census::add(Kernel::RmsNorm, isa, x.len());
    match isa.0 {
        // SAFETY: tier detected; equal lengths asserted.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni => unsafe { x86::rms_norm_avx2(x, w, eps, out) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::rms_norm_ssse3(x, w, eps, out) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon | Tier::NeonDot => unsafe { neon::rms_norm(x, w, eps, out) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::rms_norm(x, w, eps, out) },
        _ => num::rms_norm(x, w, eps, out),
    }
}

/// [`num::softmax`].
pub fn softmax(isa: Isa, x: &mut [f32], scale: f32) {
    census::add(Kernel::Softmax, isa, x.len());
    match isa.0 {
        // SAFETY: tier detected.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni => unsafe { x86::softmax_avx2(x, scale) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::softmax_ssse3(x, scale) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon | Tier::NeonDot => unsafe { neon::softmax(x, scale) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::softmax(x, scale) },
        _ => num::softmax(x, scale),
    }
}

/// `dst[i] = f32(f16 src[2i..2i+2])` (little-endian half bytes), exact.
///
/// # Panics
/// When `src` is not twice `dst`'s length.
pub fn f16_bytes_to_f32(isa: Isa, src: &[u8], dst: &mut [f32]) {
    assert_eq!(src.len(), 2 * dst.len());
    census::add(Kernel::F16Convert, isa, dst.len());
    match isa.0 {
        // SAFETY: tier detected; lengths asserted.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni => unsafe { x86::h2f_avx2(src, dst) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::h2f_ssse3(src, dst) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon | Tier::NeonDot => unsafe { neon::h2f(src, dst) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::h2f(src, dst) },
        _ => {
            let t = num::f16_table();
            for (d, b) in dst.iter_mut().zip(src.chunks_exact(2)) {
                *d = t[usize::from(u16::from_le_bytes([b[0], b[1]]))];
            }
        }
    }
}

/// `dst[i] = f16 bits of src[i]`, round to nearest even.
///
/// # Panics
/// When the lengths differ.
pub fn f32_to_f16(isa: Isa, src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    census::add(Kernel::F16Convert, isa, src.len());
    match isa.0 {
        // SAFETY: tier detected; lengths asserted.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni => unsafe { x86::f2h_avx2(src, dst) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::f2h_ssse3(src, dst) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon | Tier::NeonDot => unsafe { neon::f2h(src, dst) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::f2h(src, dst) },
        _ => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = num::to_f16_bits(s);
            }
        }
    }
}

/// `x[i] = round_f16(x[i])` in place.
pub fn round_f16_slice(isa: Isa, x: &mut [f32]) {
    census::add(Kernel::F16Convert, isa, x.len());
    match isa.0 {
        // SAFETY: tier detected.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni => unsafe { x86::round_f16_avx2(x) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::round_f16_ssse3(x) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon | Tier::NeonDot => unsafe { neon::round_f16(x) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::round_f16(x) },
        _ => {
            for v in x.iter_mut() {
                *v = num::round_f16(*v);
            }
        }
    }
}

/// [`num::finish_rows`].
///
/// # Panics
/// When the lengths differ.
pub fn finish_rows(isa: Isa, sum: &[i32], act_scale: f32, weight_scale: f32, out: &mut [f32]) {
    assert_eq!(sum.len(), out.len());
    census::add(Kernel::Elementwise, isa, sum.len());
    match isa.0 {
        // SAFETY: tier detected; lengths asserted.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni => unsafe {
            x86::finish_avx2(sum, act_scale, weight_scale, out)
        },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::finish_ssse3(sum, act_scale, weight_scale, out) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon | Tier::NeonDot => unsafe { neon::finish(sum, act_scale, weight_scale, out) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::finish(sum, act_scale, weight_scale, out) },
        _ => num::finish_rows(sum, act_scale, weight_scale, out),
    }
}

/// [`num::relu2_mul`].
///
/// # Panics
/// When the lengths differ.
pub fn relu2_mul(isa: Isa, g: &mut [f32], u: &[f32]) {
    assert_eq!(g.len(), u.len());
    census::add(Kernel::Elementwise, isa, g.len());
    match isa.0 {
        // SAFETY: tier detected; lengths asserted.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni => unsafe { x86::relu2_mul_avx2(g, u) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::relu2_mul_ssse3(g, u) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon | Tier::NeonDot => unsafe { neon::relu2_mul(g, u) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::relu2_mul(g, u) },
        _ => num::relu2_mul(g, u),
    }
}

/// [`num::add_assign`].
///
/// # Panics
/// When the lengths differ.
pub fn add_assign(isa: Isa, x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    census::add(Kernel::Elementwise, isa, x.len());
    match isa.0 {
        // SAFETY: tier detected; lengths asserted.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni => unsafe { x86::add_assign_avx2(x, y) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::add_assign_ssse3(x, y) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon | Tier::NeonDot => unsafe { neon::add_assign(x, y) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::add_assign(x, y) },
        _ => num::add_assign(x, y),
    }
}

/// [`num::rope_neox`].
///
/// # Panics
/// When `cos` / `sin` are not half of `x`.
pub fn rope_neox(isa: Isa, x: &mut [f32], cos: &[f32], sin: &[f32]) {
    assert!(x.len() == 2 * cos.len() && cos.len() == sin.len());
    census::add(Kernel::Elementwise, isa, x.len());
    match isa.0 {
        // SAFETY: tier detected; lengths asserted.
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx2Vnni => unsafe { x86::rope_avx2(x, cos, sin) },
        #[cfg(target_arch = "x86_64")]
        Tier::Ssse3 => unsafe { x86::rope_ssse3(x, cos, sin) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Tier::Neon | Tier::NeonDot => unsafe { neon::rope(x, cos, sin) },
        // SAFETY: the module was built with simd128 (else this arm does not exist); bounds as above.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Tier::Simd128 => unsafe { wasm::rope(x, cos, sin) },
        _ => num::rope_neox(x, cos, sin),
    }
}

#[cfg(test)]
mod tests;
