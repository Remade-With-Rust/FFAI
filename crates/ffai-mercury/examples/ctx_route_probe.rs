//! Throwaway: does per-clip energy contrast predict adaptive-context
//! escalation? If low-contrast clips are the escalating ones, a pre-decode
//! contrast floor removes the doomed-attempt tax without a quality risk.
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::whisper_candle::ADAPTIVE_ESCALATIONS;
use ffai_mercury::asr::WhisperCandle;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

fn main() {
    let dir = std::env::args().nth(1).expect("dir");
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(60);
    let mut clips: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "wav"))
        .collect();
    clips.sort();
    clips.truncate(n);
    unsafe { std::env::set_var("FFAI_ADAPTIVE_CTX", "on") };
    let engine = WhisperCandle::new();
    let mut esc = Vec::new();
    let mut kept = Vec::new();
    for p in &clips {
        let audio = ffai_media::load_audio(p).expect("audio");
        let mono = audio.to_mono();
        let c = ffai_mercury::asr::vad::energy_contrast_db(&mono.samples);
        let before = ADAPTIVE_ESCALATIONS.load(Ordering::Relaxed);
        let _ = engine.transcribe(&audio, &AsrOptions::default());
        let e = ADAPTIVE_ESCALATIONS.load(Ordering::Relaxed) - before;
        if e > 0 { esc.push(c) } else { kept.push(c) }
    }
    let stats = |v: &mut Vec<f32>| {
        v.sort_by(|a, b| a.total_cmp(b));
        if v.is_empty() { return (0.0, 0.0, 0.0); }
        let p = |q: f64| v[((v.len() - 1) as f64 * q) as usize];
        (p(0.1), p(0.5), p(0.9))
    };
    let (a, b, c) = stats(&mut esc);
    println!("escalated n={:>3}  contrast p10/p50/p90 = {a:.1}/{b:.1}/{c:.1}", esc.len());
    let (a, b, c) = stats(&mut kept);
    println!("kept      n={:>3}  contrast p10/p50/p90 = {a:.1}/{b:.1}/{c:.1}", kept.len());
}
