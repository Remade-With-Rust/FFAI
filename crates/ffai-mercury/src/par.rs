//! Parallel iterators on native, the identical serial ones on wasm.
//!
//! Modelled on `ffai-carmenta`'s module of the same name, and for the same
//! reason: rayon COMPILES for `wasm32-unknown-unknown`, but building a pool
//! needs `std::thread::spawn`, which the target does not provide. A parallel
//! call in a browser is a panic, not a slowdown.
//!
//! Mercury reaches rayon on the hot ASR path — `flash_attn` fans out over
//! attention heads, `text_decoder` over logit chunks, `vocab_int8` over vocab
//! blocks — so an unshimmed browser build crashes inside the first forward
//! pass rather than at load.
//!
//! The shim is structural rather than guarded. On wasm the parallel methods
//! ARE the serial ones, so an unguarded call site is simply serial instead of
//! a runtime crash — and adding a method natively without adding it here is a
//! wasm BUILD error rather than a wasm panic, which is the point.
//!
//! Serial costs less here than it would elsewhere: candle's
//! `default_num_threads()` calls `num_cpus::get_physical()`, whose wasm32
//! branch returns a literal `1`, so every matmul beneath these call sites is
//! already `Parallelism::None` on this target.

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
/// `zip` / `enumerate` / `for_each` these call sites chain — so the bodies are
/// untouched and the arithmetic is identical. Only the order of evaluation
/// stops being arbitrary, and every call site here writes into a disjoint
/// output chunk, so order was never load-bearing.
#[cfg(target_arch = "wasm32")]
pub mod serial {
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

    /// `par_chunks` / `par_chunks_mut`, serially.
    pub trait ParallelSlice<T> {
        fn par_chunks(&self, n: usize) -> core::slice::Chunks<'_, T>;
    }

    impl<T> ParallelSlice<T> for [T] {
        #[inline]
        fn par_chunks(&self, n: usize) -> core::slice::Chunks<'_, T> {
            self.chunks(n)
        }
    }

    /// `for_each_init(init, op)`, serially.
    ///
    /// rayon builds one `init` value per worker thread so an expensive
    /// scratch buffer is not reallocated per item. With one thread that is one
    /// init for the whole loop, which is exactly what this does — and the
    /// blanket impl over `Iterator` means it lands on `par_chunks_mut(..)
    /// .enumerate()` and every other adapter chain a call site builds.
    pub trait ForEachInit: Iterator + Sized {
        #[inline]
        fn for_each_init<T, INIT, OP>(self, init: INIT, mut op: OP)
        where
            INIT: Fn() -> T,
            OP: FnMut(&mut T, Self::Item),
        {
            let mut state = init();
            for item in self {
                op(&mut state, item);
            }
        }
    }

    impl<I: Iterator> ForEachInit for I {}

    pub trait ParallelSliceMut<T> {
        fn par_chunks_mut(&mut self, n: usize) -> core::slice::ChunksMut<'_, T>;
    }

    impl<T> ParallelSliceMut<T> for [T] {
        #[inline]
        fn par_chunks_mut(&mut self, n: usize) -> core::slice::ChunksMut<'_, T> {
            self.chunks_mut(n)
        }
    }
}

/// Index of the rayon worker running this call, or `None` on the caller's own
/// thread — and always `None` on wasm, where there are no workers.
///
/// Call sites use it to avoid NESTING a fan-out inside a worker that is
/// already parallel. On wasm both arms of that choice are serial anyway (the
/// prelude above is the serial shim), so the answer only has to be honest, and
/// "not on a worker" is the honest one.
#[inline]
#[must_use]
pub fn current_thread_index() -> Option<usize> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        rayon::current_thread_index()
    }
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
}

/// Threads available for intra-utterance work. Always 1 on wasm.
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
