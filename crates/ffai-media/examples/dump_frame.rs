//! Dump one decoded frame as raw RGB, so our decode can be diffed against OpenCV's.
//!
//! ```text
//! cargo run --release -p ffai-media --example dump_frame -- clip.mp4 2 out.rgb
//! ```
//!
//! WORK PARITY for video. On still frames both engines are handed identical
//! pixels and agree exactly (41/41 detections, 0.0000 px). On VIDEO each side
//! decodes for itself — different H.264 decoder, different YUV->RGB — so any
//! detection difference could be the model or could be the pixels. This tells
//! them apart.
fn main() -> ffai_core::error::Result<()> {
    let path = std::env::args().nth(1).expect("clip");
    let want: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let out = std::env::args().nth(3).expect("out.rgb");

    for (i, f) in ffai_media::stream_frames(std::path::Path::new(&path), 0.0)?.enumerate() {
        let f = f?;
        if i == want {
            println!("{}x{} {} bytes", f.image.width, f.image.height, f.image.data.len());
            std::fs::write(&out, &f.image.data)?;
            return Ok(());
        }
    }
    Ok(())
}
