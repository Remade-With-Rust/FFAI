//! Print a file's transcript segments with timestamps — the five-minute
//! debugging view that keeps getting rebuilt inline. Uses the library, so it
//! reflects the current code even when the CLI binary is locked by another
//! process (a recurring hazard on this machine).
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example dump_segments -- file.wav [from_s] [to_s]
//! ```
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: dump_segments <wav> [from_s] [to_s]");
    let from: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let to: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(f64::MAX);
    let audio = ffai_media::load_audio(std::path::Path::new(&path)).expect("audio loads");
    let t = WhisperCandle::new().transcribe(&audio, &AsrOptions::default()).expect("transcribes");
    for s in &t.segments {
        if s.end >= from && s.start <= to {
            println!("{:8.2}-{:8.2}  {}", s.start, s.end, s.value.trim());
        }
    }
}
