//! What is im2col losing to SHORT copies?
//!
//! It moves 217 MiB per image (108.4 written, read back by the GEMM) in
//! ~15.1 ms = **14 GB/s**, against 24 GB/s for a flat memcpy on this box. The
//! bytes are not the problem; the call shape is. At an 80x80 feature map the
//! inner `copy_from_slice` is **320 bytes**, issued ~184k times per image.
//!
//! One tap in three can avoid that entirely. For stride 1 the centre
//! horizontal tap (kx = 1) copies `dst[oy*ow .. (oy+1)*ow] = src[(sy-1)*w ..
//! sy*w]` with `w == ow` — so BOTH sides advance contiguously across `oy`,
//! and the whole tap is one large memcpy instead of `oh` small ones. The
//! kx = 0 and kx = 2 taps shift by one element per row and cannot be merged.
//!
//! This prices that before any of it is written.
use std::time::Instant;

fn main() {
    // Representative n-tier layer: 80x80 plane, 64 channels.
    let (w, h, c) = (80usize, 80usize, 64usize);
    let plane = w * h;
    let src: Vec<f32> = (0..c * plane).map(|i| (i % 251) as f32).collect();
    let mut dst = vec![0f32; c * 9 * plane];
    let reps = 300;

    // ROW-AT-A-TIME, exactly what the current fill does for one tap.
    let mut best_rows = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        for ch in 0..c {
            let s = &src[ch * plane..(ch + 1) * plane];
            let d = &mut dst[ch * 9 * plane..ch * 9 * plane + plane];
            for oy in 0..h {
                d[oy * w..(oy + 1) * w].copy_from_slice(&s[oy * w..(oy + 1) * w]);
            }
        }
        best_rows = best_rows.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&dst);
    }

    // WHOLE-TAP, what the centre tap could be.
    let mut best_bulk = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        for ch in 0..c {
            let s = &src[ch * plane..(ch + 1) * plane];
            let d = &mut dst[ch * 9 * plane..ch * 9 * plane + plane];
            d.copy_from_slice(s);
        }
        best_bulk = best_bulk.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&dst);
    }

    let bytes = (c * plane * 4) as f64;
    println!("{c} channels x {h}x{w}, one tap, min-of-{reps}");
    println!("  row-at-a-time ({} B per copy, {} copies): {best_rows:.3} ms  {:.1} GB/s",
             w * 4, c * h, bytes * 2.0 / (best_rows / 1e3) / 1e9);
    println!("  whole-tap memcpy                        : {best_bulk:.3} ms  {:.1} GB/s",
             bytes * 2.0 / (best_bulk / 1e3) / 1e9);
    println!("  ratio {:.2}x", best_rows / best_bulk);
    println!();
    println!("  3 of 9 taps are mergeable (kx = 1, all three ky).");
    println!("  im2col is 15.1 ms/image; a third of it at this ratio saves ~{:.1} ms",
             15.1 / 3.0 * (1.0 - best_bulk / best_rows));
}
