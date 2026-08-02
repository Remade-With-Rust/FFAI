//! What does `vec![0f32; k * b]` cost across one image's 585 im2col calls?
//!
//! D3 prize probe. The im2col buffer is allocated zeroed and then almost
//! entirely overwritten by the fill — only the padding rows and edge columns
//! keep their zeros. 216.9 MiB of zero-writes per image, to preserve on the
//! order of one percent of them.
//!
//! Whether that costs anything is NOT obvious and must not be reasoned about:
//! `alloc_zeroed` is free when the allocator hands back fresh pages from the
//! OS (already zero) and a full memset when it recycles a freed block. Our
//! allocation counter says 293.7 MiB allocated against a 26.3 MiB peak live,
//! which means heavy recycling — but "means" is a mechanism, not a number.
//!
//! So measure it, at the real size distribution, against the same allocation
//! WITHOUT the zeroing. `codec-measurement` §11: prune on arithmetic before
//! building. If this lands under the noise floor there is nothing here.
//!
//! # OUTCOME: this probe is right, and its implication was wrong
//!
//! It reports ~4.3 ms per image's worth of allocations at the ~380 KiB mean.
//! The change it motivated — filling uninitialised capacity instead — was
//! built, gated with an allocator-poisoning test, and measured in context at
//! **median 0.9997x, 10/21 rounds, z = -0.22**. Exactly chance.
//!
//! The reason is in this file's own shape: it allocates and frees in a tight
//! loop, so the allocator hands back the SAME recycled block every time and
//! `alloc_zeroed` must genuinely `memset` it. The pipeline does not allocate
//! that way; its zeroed allocations largely come from fresh OS pages, which
//! arrive zero at no cost.
//!
//! Kept, because the measurement is correct about what it measures. See
//! `docs/whys/diana-latency.md` — the microbench and the in-context number
//! disagreed, and the in-context number wins.
use std::time::Instant;

fn main() {
    // 585 calls, 216.9 MiB total => ~380 KiB mean. Sweep around it, because a
    // single size would land on one side of the allocator's large-block
    // threshold and answer a different question than the pipeline asks.
    let reps = 585usize;
    for kib in [64usize, 256, 380, 1024] {
        let n = kib * 1024 / 4;
        let mut sink = 0f32;

        let t = Instant::now();
        for _ in 0..reps {
            let v = vec![0f32; n];
            sink += v[n / 2];
            std::hint::black_box(&v);
        }
        let zeroed = t.elapsed().as_secs_f64() * 1e3;

        // Same allocation, same touch, no zero-fill: capacity only, then
        // write the one element the zeroed arm reads.
        let t = Instant::now();
        for _ in 0..reps {
            let mut v: Vec<f32> = Vec::with_capacity(n);
            // SAFETY: measurement only — we write the single element read
            // below and never observe the rest, which is exactly the
            // arithmetic being priced, not a pattern to ship.
            #[allow(unsafe_code)]
            unsafe {
                v.set_len(n);
                *v.get_unchecked_mut(n / 2) = 1.0;
            }
            sink += v[n / 2];
            std::hint::black_box(&v);
        }
        let raw = t.elapsed().as_secs_f64() * 1e3;

        println!("{kib:>5} KiB x {reps}: zeroed {zeroed:7.1} ms · uninit {raw:7.1} ms · zeroing costs {:6.1} ms", zeroed - raw);
        std::hint::black_box(sink);
    }
}
