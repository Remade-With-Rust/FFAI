//! How many bytes does one image ALLOCATE?
//!
//! The roofline says convolution arithmetic is ~20 ms of a 142 ms image and
//! that ~120 ms is elsewhere. "Elsewhere" was a list of suspects — im2col
//! traffic, silu, framework overhead — and a list of suspects is not a
//! measurement. This is the measurement for the largest one.
//!
//! An eager tensor framework allocates a fresh buffer for every op output.
//! Each fresh buffer is a `malloc`, a first-touch page fault per 4 KiB, and a
//! full write of the result — none of which appear in a FLOP count and all of
//! which are DRAM traffic. A counting `GlobalAlloc` sees all of it, exactly,
//! with no timer and no noise.
//!
//! The number to compare against is this box's ~24 GB/s: bytes / 24 GB/s is
//! the floor that allocation traffic alone puts under the per-image latency.
use ffai_core::engine::{DetectEngine, DetectOptions};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

static BYTES: AtomicU64 = AtomicU64::new(0);
static COUNT: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static ON: AtomicU64 = AtomicU64::new(0);

struct Counting;

// SAFETY: every method forwards to `System` unchanged; the counters are
// relaxed atomics and cannot affect the allocation itself.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if ON.load(Ordering::Relaxed) == 1 {
            BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
            COUNT.fetch_add(1, Ordering::Relaxed);
            let live = LIVE.fetch_add(l.size() as u64, Ordering::Relaxed) + l.size() as u64;
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if ON.load(Ordering::Relaxed) == 1 {
            LIVE.fetch_sub(l.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tier = std::env::args().nth(1).unwrap_or_else(|| "n".into());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(|p| p.parent()).unwrap();
    let img = ffai_media::load_image(&root.join("corpora/clips/diana-coco/coco-032.png"))?;
    let engine = ffai_diana::engine::Yolo26::build(&tier, ffai_diana::image::Geometry::Rect, root.join("models"));
    let opts = DetectOptions { confidence: 0.25, ..Default::default() };
    engine.detect(&img, &opts)?; // warm: weights loaded, pools grown

    ON.store(1, Ordering::Relaxed);
    engine.detect(&img, &opts)?;
    ON.store(0, Ordering::Relaxed);

    let bytes = BYTES.load(Ordering::Relaxed);
    let count = COUNT.load(Ordering::Relaxed);
    println!("tier {tier}: ONE image allocates {count} times, {:.1} MiB total", bytes as f64 / 1048576.0);
    println!("  peak live during the image: {:.1} MiB", PEAK.load(Ordering::Relaxed) as f64 / 1048576.0);
    println!("  mean allocation: {:.1} KiB", bytes as f64 / count.max(1) as f64 / 1024.0);
    // Allocated memory is written at least once, and first touch faults a page
    // per 4 KiB. Both are DRAM traffic that no FLOP count sees.
    for bw in [24.0f64] {
        println!("  writing {:.1} MiB once at {bw} GB/s = {:.1} ms floor", bytes as f64 / 1048576.0, bytes as f64 / (bw * 1e9) * 1e3);
    }
    println!("  page faults implied (4 KiB first touch): {:.0}k", bytes as f64 / 4096.0 / 1e3);
    Ok(())
}
