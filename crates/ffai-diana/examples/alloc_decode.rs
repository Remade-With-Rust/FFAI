//! Localise the allocator gap: DECODE only, no model.
//!
//! `alloc_ab.rs` re-detects ONE preloaded image and measures rusty_alloc at
//! parity with mimalloc. The CLI, decoding 525 different JPEGs and detecting
//! each, measured rusty_alloc **24 % slower on wall and 37 % more CPU**. Same
//! two allocators, opposite verdicts — which is codec-measurement §11's
//! "isolated and in-context mislead in BOTH directions", and means one of the
//! two stages carries the whole difference.
//!
//! The stages differ in exactly the way allocators care about. Detect on a
//! fixed image is a repetitive, steady-state pattern over identical shapes:
//! every allocation the second frame makes, the first frame already freed at
//! the same size. Decode is a stream of DIFFERENT, large, varied-size buffers
//! that are never reused at the same size twice.
//!
//! So this probe runs decode alone, and the pair of numbers localises the gap
//! to a stage before anyone theorises about a mechanism.
//!
//! ```text
//! cargo build --release --example alloc_decode                        # mimalloc
//! cargo build --release --example alloc_decode --features alloc-rusty # rusty_alloc
//! ```
use std::time::Instant;

#[cfg(all(not(feature = "alloc-rusty"), not(feature = "alloc-sys")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
#[cfg(all(not(feature = "alloc-rusty"), not(feature = "alloc-sys")))]
const ARM: &str = "mimalloc";

#[cfg(feature = "alloc-rusty")]
#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;
#[cfg(feature = "alloc-rusty")]
const ARM: &str = "rusty_alloc";

#[cfg(feature = "alloc-sys")]
const ARM: &str = "system";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "corpora/clips/mot17-09/img1".into());
    let reps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let mut frames: Vec<_> = std::fs::read_dir(root.join(&dir))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(p.extension().and_then(|e| e.to_str()), Some("png" | "jpg" | "jpeg"))
        })
        .collect();
    frames.sort();

    // Work count: total decoded pixels. Identical across arms or the
    // comparison is void — an allocator cannot change how many pixels a JPEG
    // has, so a mismatch means one arm silently failed a decode.
    let mut px: u64 = 0;
    let t = Instant::now();
    for _ in 0..reps {
        for f in &frames {
            let img = ffai_media::load_image(f)?;
            px += (img.width as u64) * (img.height as u64);
        }
    }
    let ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "ARM={ARM} frames={} reps={reps} total_ms={ms:.1} per_frame_ms={:.3} px={px}",
        frames.len(),
        ms / (frames.len() * reps) as f64
    );
    Ok(())
}
