//! Where does diarization's per-chunk cost actually go?
//!
//! Measured symptom: +621 ms on a 3 s live chunk, 6.8x the ASR-only path,
//! for a ~20M-parameter model. That is a symptom, not a diagnosis — the
//! discipline is to localize before touching code, because the obvious
//! suspect has been wrong repeatedly in this project.
//!
//! Three candidate levers were named from reading `embed_windows`: batch the
//! per-window forwards, parallelize the loop, cache across chunks. This
//! prices the pieces so the profile picks one instead of intuition:
//!
//!   - fbank      : filterbank features per window (pure CPU DSP)
//!   - embed      : the ECAPA-TDNN forward per window
//!   - cluster    : agglomerative clustering over the embeddings
//!   - windows    : how many the geometry produces for a given chunk
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example diarize_anatomy
//! ```
use std::time::Instant;

use ffai_mercury::asr::diarize;
use ffai_mercury::asr::diarizer::Diarizer;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or("corpora/clips/librispeech-test-clean/audio/1089-134686-0000.wav".into());
    let secs: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let rounds: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(9);

    let audio = ffai_media::load_audio(std::path::Path::new(&path))?;
    let mono = audio.to_mono();
    let n = ((secs * mono.sample_rate as f64) as usize).min(mono.samples.len());
    let samples = &mono.samples[..n];

    let diarizer = Diarizer::from_manifest_source(
        None,
        ffai_mercury::asr::diarizer::DEFAULT_MODEL,
        ffai_core::best_device(),
    )?;

    // The geometry the live path actually produces for this chunk.
    let regions = vec![ffai_core::types::TimedSegment {
        start: 0.0,
        end: secs,
        value: (),
        confidence: None,
    }];
    let windows = diarize::subsegment(&regions, diarize::WINDOW_SECS, diarize::HOP_SECS);
    println!(
        "{secs} s chunk -> {} windows ({} s @ {} s hop), {rounds} rounds\n",
        windows.len(),
        diarize::WINDOW_SECS,
        diarize::HOP_SECS
    );

    // Warm: first call pays lazy init we do not want inside a median.
    let _ = diarizer.diarize(samples, &regions, 0.80, None);

    let sr = diarizer.sample_rate() as f64;
    let (mut t_fbank, mut t_embed, mut t_total) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..rounds {
        // Per-stage, over the same windows the real path walks.
        let mut fb = 0.0;
        let mut em = 0.0;
        for &(start, end) in &windows {
            let a = ((start * sr) as usize).min(samples.len());
            let b = ((end * sr).ceil() as usize).clamp(a, samples.len());
            let t = Instant::now();
            let (feats, frames) = diarizer.fbank_for(&samples[a..b]);
            fb += t.elapsed().as_secs_f64() * 1e3;
            if frames == 0 {
                continue;
            }
            let t = Instant::now();
            let _ = diarizer.embed_for(&feats, frames);
            em += t.elapsed().as_secs_f64() * 1e3;
        }
        t_fbank.push(fb);
        t_embed.push(em);

        let t = Instant::now();
        let _ = diarizer.diarize(samples, &regions, 0.80, None);
        t_total.push(t.elapsed().as_secs_f64() * 1e3);
    }

    let (fb, em, tot) = (median(t_fbank), median(t_embed), median(t_total));
    println!("{:<30} {:>9}  {:>7}", "STAGE", "ms/chunk", "share");
    let row = |name: &str, v: f64| println!("{name:<30} {v:>9.1}  {:>6.1}%", v / tot * 100.0);
    row("fbank (all windows)", fb);
    row("embed (all windows)", em);
    row("residue (cluster + glue)", tot - fb - em);
    println!("{:<30} {tot:>9.1}", "TOTAL diarize()");
    println!(
        "\nper window: fbank {:.1} ms · embed {:.1} ms  (n = {})",
        fb / windows.len().max(1) as f64,
        em / windows.len().max(1) as f64,
        windows.len()
    );

    // Does embed cost scale with FRAMES, or is it fixed per call? Linear =>
    // per-frame compute, so the lever is the conv path (or fewer frames).
    // Sublinear => fixed overhead dominates, so the lever is batching the
    // windows into one call. This decides which optimization to build, and
    // it is two minutes of measurement against a day of the wrong one.
    println!("\n{:<12} {:>8} {:>10} {:>12} {:>11}", "WINDOW s", "frames", "ms", "ms/frame", "GFLOP/s*");
    for w in [0.5f64, 1.0, 1.5, 3.0, 6.0] {
        let k = ((w * sr) as usize).min(samples.len());
        if k == 0 {
            continue;
        }
        let (feats, frames) = diarizer.fbank_for(&samples[..k]);
        if frames == 0 {
            continue;
        }
        let mut ts = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            let _ = diarizer.embed_for(&feats, frames);
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let ms = median(ts);
        // ~34.6 MFLOP/frame for this topology (see the module header): the
        // 1024-channel stack plus the 3072-wide MFA dominate.
        let gflops = (frames as f64 * 34.6e6) / (ms / 1e3) / 1e9;
        println!("{w:<12.1} {frames:>8} {ms:>10.1} {:>12.3} {gflops:>11.1}", ms / frames as f64);
    }
    println!("* rough: 34.6 MFLOP/frame from the channel/kernel config, for scale only");
    Ok(())
}
