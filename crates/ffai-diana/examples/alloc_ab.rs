//! Which allocator, measured — system vs mimalloc vs rusty_alloc.
//!
//! Measured on this box: **58,634 page faults per image**. At 4 KiB each that
//! is ~234 MiB of freshly-faulted pages, against 293.7 MiB/image of
//! allocation — so essentially every byte allocated is arriving on a page the
//! process has to fault in, which means the system allocator is returning
//! memory to the OS and re-faulting it rather than keeping it mapped.
//!
//! That made the allocator the largest single effect in the whole Diana
//! latency campaign: mimalloc measured **1.64x** over the system allocator,
//! more than every kernel-level lever found before it combined.
//!
//! Which is exactly why replacing it is delicate. `mimalloc` is a C library,
//! and FFai exists to not depend on those; `rusty_alloc` is the pure-Rust
//! remake of the same design (mimalloc v2.4.5). But a 1.64x lever is also the
//! most expensive thing in the tree to get wrong, so the swap is a
//! MEASUREMENT, not a preference.
//!
//! Three arms, selected at compile time so the binaries differ by exactly one
//! type in one `#[global_allocator]` line:
//!
//! ```text
//! cargo build --release --example alloc_ab                        # mimalloc
//! cargo build --release --example alloc_ab --features alloc-rusty # rusty_alloc
//! cargo build --release --example alloc_ab --features alloc-sys   # system
//! ```
//!
//! Reports a WORK COUNT (detections + a box checksum) alongside the timing.
//! Allocators must not change results, so a checksum that moves between arms
//! voids the comparison before any duration is compared — and would be a far
//! more important finding than the speed.
use ffai_core::engine::{DetectEngine, DetectOptions};
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

// No `#[global_allocator]` at all — Rust's default, which on Windows is the
// system heap. This is the arm the 1.64x was measured against.
#[cfg(feature = "alloc-sys")]
const ARM: &str = "system";

/// Background page reclaim, so the trim can be A/B'd inside ONE program.
///
/// The first attempt compared a trimmed run of `scaling.rs` against an
/// untrimmed run of this file and called the difference the trim's cost. Two
/// different programs, one variable claimed — the arms have to be the same
/// binary or the number is a build comparison (codec-measurement §4).
#[cfg(feature = "alloc-rusty")]
fn spawn_trim() {
    let ms = std::env::var("FFAI_TRIM_MS").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    if ms == 0 {
        return;
    }
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(ms));
        rusty_alloc::alloc::collect(false);
    });
}

#[cfg(not(feature = "alloc-rusty"))]
fn spawn_trim() {}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    spawn_trim();
    let tier = std::env::args().nth(1).unwrap_or_else(|| "n".into());
    let reps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let img = ffai_media::load_image(&root.join("corpora/clips/diana-coco/coco-032.png"))?;
    let engine =
        ffai_diana::engine::Yolo26::build(&tier, ffai_diana::image::Geometry::Rect, root.join("models"));
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };

    // Warm: first call pays lazy weight materialisation and the allocator's
    // own arena bring-up, neither of which is what we are comparing.
    let warm = engine.detect(&img, &opts)?;

    // Work parity (codec-measurement §4). An allocator changes WHERE bytes
    // live, never what they contain — so this checksum is expected to be
    // bit-identical across arms, and a difference is a correctness bug in the
    // allocator rather than a benchmarking nuisance.
    let n_det = warm.detections.len();
    let mut sum = 0f64;
    for d in &warm.detections {
        sum += f64::from(d.x0) + f64::from(d.y0) + f64::from(d.x1) + f64::from(d.y1) + f64::from(d.confidence);
    }

    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        engine.detect(&img, &opts)?;
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times[0];
    let med = times[times.len() / 2];

    // One machine-readable line so the ABBA harness never has to parse prose.
    println!(
        "ARM={ARM} tier={tier} reps={reps} min={min:.3} med={med:.3} ndet={n_det} sum={sum:.6}"
    );
    Ok(())
}
