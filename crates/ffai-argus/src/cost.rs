//! Re-export of [`ffai_core::cost`].
//!
//! The counters moved to `ffai-core` so Mercury and Carmenta can rank their
//! own targets the same way (`docs/plans/turbocharger.md` step 1). This alias
//! keeps `crate::cost::*` working at the ~30 instrumented call sites in
//! `siglip.rs` rather than churning them.
pub use ffai_core::cost::*;
