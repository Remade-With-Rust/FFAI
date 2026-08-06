//! Decode cost against content entropy — is the H.264 path linear in the work
//! it is given, or does it degrade on hard content?
//!
//! ```text
//! cargo run --release -p ffai-media --example entropy_decode -- clip.mp4
//! ```
//!
//! The question is NOT "is busy content slower" — it must be, and that is
//! uninteresting. It is whether **cost per bit** is flat across an entropy
//! ladder. A stage that scales with the information it is handed has no gate to
//! build; one whose cost/bit RISES is degrading, and where it turns is where a
//! dispatch threshold would sit.
//!
//! Reports total wall, frames, and per-frame timings so the driver can compute
//! cost/bit and look for a knee. Frame count is printed because a clip that
//! decodes fewer frames than the encoder wrote voids its row (work parity).

use std::path::Path;

fn main() -> ffai_core::error::Result<()> {
    let path = std::env::args().nth(1).expect("usage: entropy_decode <clip.mp4>");
    let reps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    // fps <= 0 keeps EVERY frame - decimation would measure the decimator.
    let fps: f64 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    // Warm: first decode pays lazy init and first-touch faults.
    let _ = ffai_media::sample_frames(Path::new(&path), fps)?;

    let mut best = f64::MAX;
    let mut frames = 0usize;
    let mut px = 0usize;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        let fr = ffai_media::sample_frames(Path::new(&path), fps)?;
        let e = t.elapsed().as_secs_f64();
        if e < best {
            best = e;
        }
        frames = fr.len();
        if let Some(f0) = fr.first() {
            px = f0.image.width as usize * f0.image.height as usize;
        }
        std::hint::black_box(fr);
    }
    println!(
        "{{\"clip\":\"{}\",\"frames\":{},\"px\":{},\"decode_s\":{:.6},\"ms_per_frame\":{:.4}}}",
        Path::new(&path).file_stem().unwrap().to_string_lossy(),
        frames,
        px,
        best,
        best * 1000.0 / frames.max(1) as f64
    );
    Ok(())
}
