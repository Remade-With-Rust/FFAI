//! Throwaway probe: does VAD-frame energy contrast (p90 − p10, dB) separate
//! clean from noisy corpora? If it does, it can route adaptive-context
//! windows BEFORE any decode is paid for.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example vad_snr_probe
//! ```

use std::path::PathBuf;

fn stats(dir: &str, n: usize) -> Vec<(String, f32)> {
    let mut clips: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "wav"))
        .collect();
    clips.sort();
    clips.truncate(n);
    clips
        .iter()
        .map(|p| {
            let audio = ffai_media::load_audio(p).expect("audio");
            let mono = audio.to_mono();
            let c = ffai_mercury::asr::vad::energy_contrast_db(&mono.samples);
            (p.file_stem().unwrap().to_string_lossy().into_owned(), c)
        })
        .collect()
}

fn main() {
    for (name, dir) in [
        ("clean", "corpora/clips/librispeech-test-clean/audio"),
        ("other", "corpora/clips/librispeech-test-other/audio"),
    ] {
        let mut vals = stats(dir, 60);
        vals.sort_by(|a, b| a.1.total_cmp(&b.1));
        let v: Vec<f32> = vals.iter().map(|x| x.1).collect();
        let pct = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
        println!(
            "{name:>6}: n={} min={:.1} p10={:.1} p50={:.1} p90={:.1} max={:.1}",
            v.len(),
            v[0],
            pct(0.1),
            pct(0.5),
            pct(0.9),
            v[v.len() - 1]
        );
        println!("  lowest: {:?}", &vals[..4.min(vals.len())]);
    }
}
