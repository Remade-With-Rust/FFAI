//! Why did VAD improve WER? — the six-whys descent, instrumented.
//!
//! VAD was built to save encoder passes on silence. It was predicted to make
//! no difference on LibriSpeech, where every clip is continuous read speech
//! with no silence to skip. It moved test-clean WER 7.99 -> 6.79 and CER
//! 3.27 -> 2.74, which means the model of *why* is wrong.
//!
//! This probe does not test a hypothesis. It gets the per-clip facts first,
//! because the last three times this campaign reasoned before measuring it
//! invented a mechanism for an effect that was noise (open-campaign.md).
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example vad_why -- corpora/librispeech-test-clean-v2.toml
//! ```

use std::path::PathBuf;

use ffai_bench::corpus::Manifest;
use ffai_bench::metrics::{cer, wer};
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::{vad, WhisperCandle};

struct Row {
    id: String,
    wer_off: f64,
    wer_on: f64,
    /// Seconds of audio the VAD dropped from the front.
    lead_trimmed: f64,
    /// Seconds dropped from the end.
    tail_trimmed: f64,
    /// Speech regions found.
    regions: usize,
    /// Encoder windows in each arm.
    windows_off: usize,
    windows_on: usize,
    text_off: String,
    text_on: String,
    duration: f64,
}

fn main() {
    let corpus = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "corpora/librispeech-test-clean-v2.toml".to_string());
    let manifest = Manifest::load(&PathBuf::from(&corpus)).expect("corpus loads");
    let limit: usize =
        std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    let engine = WhisperCandle::new();
    let off = AsrOptions { vad: false, ..Default::default() };
    let on = AsrOptions { vad: true, ..Default::default() };

    let mut rows = Vec::new();
    for clip in manifest.clips.iter().take(limit) {
        let path = manifest.clip_path(clip);
        let Ok(audio) = ffai_media::load_audio(&path) else { continue };
        let Ok(Some(truth)) = manifest.ground_truth(clip) else { continue };

        let Ok(t_off) = engine.transcribe(&audio, &off) else { continue };
        let Ok(t_on) = engine.transcribe(&audio, &on) else { continue };

        let mono = audio.to_mono();
        let duration = mono.samples.len() as f64 / 16_000.0;
        let regions = vad::detect(&mono.samples, on.vad_threshold);
        let packed = vad::pack(&regions, on.vad_chunk_secs as f64);

        let lead = regions.first().map(|r| r.start).unwrap_or(0.0);
        let tail = regions.last().map(|r| duration - r.end).unwrap_or(0.0);
        // Without VAD the window count is ceil(duration / 30).
        let windows_off = (mono.samples.len().div_ceil(16_000 * 30)).max(1);

        rows.push(Row {
            id: clip.id.clone(),
            wer_off: wer(&truth, &t_off.text()),
            wer_on: wer(&truth, &t_on.text()),
            lead_trimmed: lead,
            tail_trimmed: tail,
            regions: regions.len(),
            windows_off,
            windows_on: packed.len(),
            text_off: t_off.text(),
            text_on: t_on.text(),
            duration,
        });
        if rows.len() % 20 == 0 {
            eprintln!("  {} clips ...", rows.len());
        }
    }

    let n = rows.len() as f64;
    let mean_off: f64 = rows.iter().map(|r| r.wer_off).sum::<f64>() / n;
    let mean_on: f64 = rows.iter().map(|r| r.wer_on).sum::<f64>() / n;

    println!("\n=== WHY 1: is the gain spread, or a few clips? ===");
    println!("clips {n}   mean per-clip WER  off {:.4}  on {:.4}", mean_off, mean_on);
    let improved = rows.iter().filter(|r| r.wer_on < r.wer_off - 1e-9).count();
    let worsened = rows.iter().filter(|r| r.wer_on > r.wer_off + 1e-9).count();
    let same = rows.len() - improved - worsened;
    println!("improved {improved}   worsened {worsened}   unchanged {same}");
    // If a handful of clips carry it, this is a tail effect, not a broad one.
    let mut deltas: Vec<f64> = rows.iter().map(|r| r.wer_off - r.wer_on).collect();
    deltas.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let total: f64 = deltas.iter().sum();
    let top5: f64 = deltas.iter().take(5).sum();
    println!(
        "total WER improvement {:.3}; top 5 clips carry {:.3} ({:.0}%)",
        total,
        top5,
        100.0 * top5 / total.max(1e-9)
    );

    println!("\n=== WHY 2: does the gain track how much silence was trimmed? ===");
    // If trimming silence is the mechanism, clips with more trimmed should
    // improve more. Correlation is the discriminator.
    let trims: Vec<f64> = rows.iter().map(|r| r.lead_trimmed + r.tail_trimmed).collect();
    let gains: Vec<f64> = rows.iter().map(|r| r.wer_off - r.wer_on).collect();
    println!("corr(silence trimmed, WER gain) = {:.3}", pearson(&trims, &gains));
    let lead: Vec<f64> = rows.iter().map(|r| r.lead_trimmed).collect();
    let tail: Vec<f64> = rows.iter().map(|r| r.tail_trimmed).collect();
    println!("  corr(LEAD trimmed,  WER gain) = {:.3}", pearson(&lead, &gains));
    println!("  corr(TAIL trimmed,  WER gain) = {:.3}", pearson(&tail, &gains));
    println!(
        "  mean lead {:.2}s  mean tail {:.2}s  mean duration {:.2}s",
        lead.iter().sum::<f64>() / n,
        tail.iter().sum::<f64>() / n,
        rows.iter().map(|r| r.duration).sum::<f64>() / n
    );

    println!("\n=== WHY 3: did the window COUNT change? ===");
    // A different number of encoder passes is a different decode entirely,
    // and would be a confound rather than a mechanism.
    let split_differently =
        rows.iter().filter(|r| r.windows_on != r.windows_off).count();
    println!("clips where window count differs: {split_differently} / {}", rows.len());
    let multi = rows.iter().filter(|r| r.windows_off > 1).count();
    println!("clips longer than one 30 s window: {multi}");

    println!("\n=== WHY 4: what actually changed in the text? ===");
    let mut by_gain: Vec<&Row> = rows.iter().collect();
    by_gain.sort_by(|a, b| {
        (b.wer_off - b.wer_on).partial_cmp(&(a.wer_off - a.wer_on)).unwrap()
    });
    for r in by_gain.iter().take(6) {
        println!(
            "\n[{}] dur {:.1}s  lead {:.2}s tail {:.2}s  regions {}  WER {:.3} -> {:.3}",
            r.id, r.duration, r.lead_trimmed, r.tail_trimmed, r.regions, r.wer_off, r.wer_on
        );
        println!("  off: {}", first_line(&r.text_off, 150));
        println!("  on : {}", first_line(&r.text_on, 150));
    }
    println!("\n--- worst regressions ---");
    for r in by_gain.iter().rev().take(3) {
        println!(
            "\n[{}] dur {:.1}s  lead {:.2}s tail {:.2}s  WER {:.3} -> {:.3}",
            r.id, r.duration, r.lead_trimmed, r.tail_trimmed, r.wer_off, r.wer_on
        );
        println!("  off: {}", first_line(&r.text_off, 150));
        println!("  on : {}", first_line(&r.text_on, 150));
    }

    println!("\n=== WHY 5: length effects ===");
    // Whisper's known failure is hallucinating a continuation. If that is the
    // mechanism, the OFF arm should be producing MORE text than the truth.
    let mut over_off = 0usize;
    let mut over_on = 0usize;
    for (r, clip) in rows.iter().zip(manifest.clips.iter()) {
        if let Ok(Some(truth)) = manifest.ground_truth(clip) {
            let t = truth.split_whitespace().count();
            if r.text_off.split_whitespace().count() > t + 2 {
                over_off += 1;
            }
            if r.text_on.split_whitespace().count() > t + 2 {
                over_on += 1;
            }
        }
    }
    println!("clips emitting >2 words MORE than ground truth:  off {over_off}   on {over_on}");

    println!("\n=== corpus totals (what the ledger reports) ===");
    let joined_truth: Vec<String> = manifest
        .clips
        .iter()
        .take(rows.len())
        .filter_map(|c| manifest.ground_truth(c).ok().flatten())
        .collect();
    let truth_all = joined_truth.join(" ");
    let off_all: Vec<&str> = rows.iter().map(|r| r.text_off.as_str()).collect();
    let on_all: Vec<&str> = rows.iter().map(|r| r.text_on.as_str()).collect();
    println!(
        "WER  off {:.4}  on {:.4}      CER  off {:.4}  on {:.4}",
        wer(&truth_all, &off_all.join(" ")),
        wer(&truth_all, &on_all.join(" ")),
        cer(&truth_all, &off_all.join(" ")),
        cer(&truth_all, &on_all.join(" "))
    );
}

fn first_line(s: &str, max: usize) -> String {
    let one = s.replace('\n', " / ");
    if one.chars().count() > max {
        one.chars().take(max).collect::<String>() + " ..."
    } else {
        one
    }
}

/// Pearson correlation — the discriminator for WHY 2. Returns 0 when either
/// side has no variance, which is the honest answer rather than a NaN.
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for (a, b) in x.iter().zip(y.iter()) {
        cov += (a - mx) * (b - my);
        vx += (a - mx) * (a - mx);
        vy += (b - my) * (b - my);
    }
    if vx <= 0.0 || vy <= 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}
