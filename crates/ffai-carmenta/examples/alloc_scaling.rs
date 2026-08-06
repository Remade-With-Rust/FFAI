//! Does the system allocator serialise concurrent ~1 MB requests? (§8.104)
//!
//! §8.103 ruled out arithmetic, input traffic, load imbalance and scheduling as
//! the cause of the CRNN path's 5.78x-on-16-cores ceiling, and left one
//! candidate: the allocator. Every conv call allocates a padded input (up to
//! ~1 MB) and an output tensor, ~511 allocations per page, and 16 threads do it
//! at once. Windows' default heap backs large requests with `VirtualAlloc`,
//! which takes a process-wide lock — that would produce exactly the observed
//! efficiency curve (89 % -> 65 % -> 44 % as threads grow).
//!
//! This tests the hypothesis WITHOUT touching the engine or adding a
//! dependency: N threads each allocate, touch and free a buffer in a loop. If
//! the allocator is the shared resource, throughput per thread collapses as N
//! rises; if it scales, the hypothesis is dead and §8.103's residue is
//! something else.
//!
//! Two sizes are measured because they take different paths in most allocators:
//! ~1 MB (the real conv buffer — `VirtualAlloc`/mmap territory) and 64 KB
//! (heap-arena territory). The contrast localises the cost.
//!
//! `black_box` guards the touch loop; without it the whole allocation is dead
//! code and the compiler deletes the measurement.
//!
//! Usage: cargo run -p ffai-carmenta --release --example alloc_scaling

// CEILING PROBE (§8.104): `FFAI_MIMALLOC=1` is not a runtime switch — a global
// allocator is chosen at link time — so this example is built twice by the
// harness, once with the feature and once without. Here it is unconditional;
// the system-allocator numbers come from the committed version.
#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Allocate a zeroed `f32` buffer, touch one element per 4 KiB page so the
/// pages are really committed, then drop it. Touching matters: a zero-filled
/// allocation that is never read may never be faulted in, and then the
/// benchmark measures address-space bookkeeping rather than the real cost the
/// conv path pays.
fn alloc_touch_free(n_floats: usize) {
    let v = vec![0f32; n_floats];
    let mut acc = 0f32;
    let mut i = 0;
    while i < v.len() {
        acc += v[i];
        i += 1024; // one f32 per 4 KiB page
    }
    black_box(acc);
    black_box(&v);
}

fn run(n_threads: usize, n_floats: usize, iters: u64) -> f64 {
    let done = Arc::new(AtomicU64::new(0));
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..n_threads {
            let done = done.clone();
            s.spawn(move || {
                for _ in 0..iters {
                    alloc_touch_free(n_floats);
                }
                done.fetch_add(iters, Ordering::Relaxed);
            });
        }
    });
    let secs = t0.elapsed().as_secs_f64();
    done.load(Ordering::Relaxed) as f64 / secs
}

fn main() {
    // 262144 f32 = 1 MiB, the scale of a padded conv5 input; 16384 f32 = 64 KiB.
    for (label, n_floats, iters) in [("1 MiB", 262_144usize, 400u64), ("64 KiB", 16_384, 4_000)] {
        println!("\n  {label} allocate + touch + free\n");
        println!("  {:>8} {:>14} {:>10} {:>12}", "threads", "allocs/s", "speedup", "efficiency");
        let mut base = 0f64;
        for n in [1usize, 2, 4, 8, 16] {
            let rate = run(n, n_floats, iters);
            if n == 1 {
                base = rate;
            }
            println!(
                "  {n:8} {rate:14.0} {:9.2}x {:11.0} %",
                rate / base,
                100.0 * rate / base / n as f64
            );
        }
    }
    println!(
        "\n  Reading: if efficiency holds near 100 % the allocator is NOT the shared\n  \
         resource and §8.103's residue is elsewhere. If it collapses the way the\n  \
         engine's does (89 -> 65 -> 44 %), the hypothesis is confirmed and a\n  \
         different global allocator is worth its dependency."
    );
}
