//! Parallel iterators on native, the identical serial ones on wasm.
//!
//! `wasm32-unknown-unknown` has no threads to spawn. rayon COMPILES there —
//! `rayon-core` even ships a `web_spin_lock` feature — but building a pool
//! needs `std::thread::spawn`, which the target does not provide. So the wasm
//! build must never construct a pool or call a parallel iterator.
//!
//! Diana already had a serial path: 22 kernels branch on
//! [`crate::parallel::serial_kernels`], because a measured finding said the
//! fan-out is a bad trade for a single image —
//!
//! | rayon threads | CPU ms/image |
//! |---:|---:|
//! | 1 | **363** |
//! | 24 | 844 |
//!
//! — the work is 363 ms and 24 threads spend 844 ms doing it. Serial is not a
//! wasm concession here; it is 2.32x LESS CPU than the fan-out, and wasm gets
//! the good arm by default.
//!
//! That path was not quite complete, though: conv3x3 has five parallel calls
//! behind three guards, silu four behind two. Rather than audit every site and
//! hope the next one is remembered, this module makes the choice structural —
//! on wasm the parallel methods ARE the serial ones, so an unguarded call is
//! simply serial rather than a runtime panic.
//!
//! The shimmed surface is exactly what Diana uses, which a grep says is four
//! methods and nothing else. Adding a fifth on native without adding it here
//! is a wasm build error, not a wasm crash, which is the point.

/// `use crate::par::prelude::*` in place of `use rayon::prelude::*`.
pub mod prelude {
    #[cfg(not(target_arch = "wasm32"))]
    pub use rayon::prelude::*;

    #[cfg(target_arch = "wasm32")]
    pub use super::serial::*;
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

/// Serial stand-ins with rayon's method names.
///
/// Every returned type is a std iterator, which already provides the
/// `enumerate` / `for_each` / `map` / `zip` these call sites chain — so the
/// bodies are untouched and the arithmetic is identical, only the order of
/// evaluation is fixed rather than arbitrary.
#[cfg(target_arch = "wasm32")]
pub mod serial {
    pub trait ParChunksMut<T> {
        fn par_chunks_mut(&mut self, n: usize) -> core::slice::ChunksMut<'_, T>;
    }
    impl<T> ParChunksMut<T> for [T] {
        #[inline]
        fn par_chunks_mut(&mut self, n: usize) -> core::slice::ChunksMut<'_, T> {
            self.chunks_mut(n)
        }
    }

    pub trait ParChunks<T> {
        fn par_chunks(&self, n: usize) -> core::slice::Chunks<'_, T>;
    }
    impl<T> ParChunks<T> for [T] {
        #[inline]
        fn par_chunks(&self, n: usize) -> core::slice::Chunks<'_, T> {
            self.chunks(n)
        }
    }

    // The trait carries the lifetime: an associated type cannot name `'_`,
    // and the borrow has to outlive the returned iterator.
    pub trait ParIter<'a> {
        type Iter;
        fn par_iter(&'a self) -> Self::Iter;
    }
    impl<'a, T: 'a> ParIter<'a> for [T] {
        type Iter = core::slice::Iter<'a, T>;
        #[inline]
        fn par_iter(&'a self) -> Self::Iter {
            self.iter()
        }
    }
    impl<'a, T: 'a> ParIter<'a> for Vec<T> {
        type Iter = core::slice::Iter<'a, T>;
        #[inline]
        fn par_iter(&'a self) -> Self::Iter {
            self.as_slice().iter()
        }
    }

    pub trait IntoParIter {
        type Iter;
        fn into_par_iter(self) -> Self::Iter;
    }
    impl<T> IntoParIter for Vec<T> {
        type Iter = std::vec::IntoIter<T>;
        #[inline]
        fn into_par_iter(self) -> Self::Iter {
            self.into_iter()
        }
    }
    impl IntoParIter for core::ops::Range<usize> {
        type Iter = core::ops::Range<usize>;
        #[inline]
        fn into_par_iter(self) -> Self::Iter {
            self
        }
    }
}
