//! Parallel iterators on native, the identical serial ones on wasm.
//!
//! Modelled on `ffai-diana`'s `par` module, and for the same reason: rayon
//! COMPILES for `wasm32-unknown-unknown`, but building a pool needs
//! `std::thread::spawn`, which the target does not provide. A parallel call in
//! a browser is a panic, not a slowdown.
//!
//! Carmenta reaches it by default. [`crate::engine`] fans lines out whenever a
//! page has three or more of them, and the escape hatch is `FFAI_REC_SERIAL` —
//! an ENVIRONMENT VARIABLE, which a browser does not have. `std::env::var`
//! returns `Err` there, so the serial arm is unreachable and the parallel arm
//! is taken on any page with three lines.
//!
//! **Serial is not a concession here, it is the arm that was already winning
//! in this shape.** Three rayon levels nest natively — ours over lines,
//! candle's over im2col tiles, and `gemm`, which candle hands
//! `Parallelism::Rayon(num_cpus::get())` on every matmul — and a one-line band
//! strip measured **177 ms/line under `par_iter` against 82 ms serial**
//! (§8.100). On wasm the nesting cannot happen at all: candle's
//! `default_num_threads()` calls `num_cpus::get_physical()`, whose wasm32
//! branch returns a literal `1`, so candle takes `Parallelism::None`.
//!
//! The shim is structural rather than guarded. On wasm the parallel methods
//! ARE the serial ones, so an unguarded call site is simply serial instead of
//! a runtime crash — and adding a fifth method natively without adding it here
//! is a wasm BUILD error rather than a wasm panic, which is the point.

/// `use crate::par::prelude::*` in place of `use rayon::prelude::*`.
pub mod prelude {
    #[cfg(not(target_arch = "wasm32"))]
    pub use rayon::prelude::*;

    #[cfg(target_arch = "wasm32")]
    pub use super::serial::*;
}

/// Serial stand-ins carrying rayon's method names.
///
/// Every returned type is a std iterator, which already provides the
/// `map` / `filter` / `collect` / `enumerate` these call sites chain — so the
/// bodies are untouched and the arithmetic is identical. Only the order of
/// evaluation stops being arbitrary, and `engine.rs` collects into a `Vec`
/// indexed by line, so order was never load-bearing.
#[cfg(target_arch = "wasm32")]
pub mod serial {
    /// The `par_iter` of `rayon::prelude`, serially.
    pub trait ParallelSlice<T> {
        fn par_iter(&self) -> core::slice::Iter<'_, T>;
    }

    impl<T> ParallelSlice<T> for [T] {
        #[inline]
        fn par_iter(&self) -> core::slice::Iter<'_, T> {
            self.iter()
        }
    }

    impl<T> ParallelSlice<T> for Vec<T> {
        #[inline]
        fn par_iter(&self) -> core::slice::Iter<'_, T> {
            self.iter()
        }
    }

    /// `(0..n).into_par_iter()`, serially.
    pub trait IntoParallelIterator {
        type Iter;
        fn into_par_iter(self) -> Self::Iter;
    }

    impl IntoParallelIterator for core::ops::Range<usize> {
        type Iter = Self;
        #[inline]
        fn into_par_iter(self) -> Self {
            self
        }
    }
}

/// Threads available for intra-image work. Always 1 on wasm.
#[inline]
#[must_use]
pub fn current_num_threads() -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        rayon::current_num_threads()
    }
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
}
