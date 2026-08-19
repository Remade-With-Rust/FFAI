//! Interleaved paired A/B at the **pipeline** level, with a NULL ARM.
//!
//! `codec-six-whys-unknowns` requires that of the three probes behind any
//! refutation, **at least one measures the level above the change**. Every
//! open item in `docs/whys/OPEN.md` is an op- or stage-level change, so the
//! level above is total transcription time — and until now this repo had no
//! harness that could measure it with both arms alive at once. The knobs were
//! `OnceLock`s resolved at process start, which forced arm-by-arm A/B across
//! two processes on a box whose op-level spread is 28–49 %.
//!
//! Two things make this harness trustworthy, and both were learned expensively
//! elsewhere in the suite:
//!
//! 1. **The NULL ARM** (`--test null`). Both arms are configured identically,
//!    so the measured difference is definitionally zero and whatever the
//!    harness reports is its own resolution limit. A VP9 campaign discovered
//!    its A/B read +0.2 % to +10.8 % for two IDENTICAL encoders — wider than
//!    every delta it had been reporting. **Run the null first, every session;
//!    it drifts.** Any result inside the null's band is a coin flip.
//! 2. **ABBA ordering.** The arm that runs first in every round accumulates a
//!    systematic advantage (one null read a mean +1.4 % for arm 0 purely from
//!    position). Alternating the order per round cancels it.
//!
//! The verdict is the **paired win rate**: under the null hypothesis each
//! round is a fair coin, so `z = (wins − N/2) / (0.5·√N)` and `|z| > 2` is a
//! real result *regardless of how far the medians drifted*. Medians and ranges
//! are printed as supporting detail, never as the verdict.
//!
//! ```sh
//! cargo run --release -p ffai-mercury --example pipeline_ab -- --test null
//! cargo run --release -p ffai-mercury --example pipeline_ab -- --test mlp_int8
//! ```
//!
//! Tests: `null`, `mlp_int8`, `par_heads`, `kv_f16`, `min_keys`, `gemv_pad`.

use std::path::PathBuf;
use std::time::Instant;

use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;
use ffai_mercury::asr::knobs;

/// What an arm does to the knobs immediately before each timed call.
/// Load-time knobs are applied once, before the engine is warmed; per-call and
/// prepare-time knobs have to be re-applied every round because the other arm
/// has since changed them.
type Apply = Box<dyn Fn()>;

struct Arm {
    label: &'static str,
    engine: WhisperCandle,
    apply: Apply,
}

impl Arm {
    /// Build an engine under this arm's settings and warm it, so weight
    /// loading and the precision calibration sit outside every timed region.
    fn new(label: &'static str, apply: Apply, warm: &ffai_core::types::AudioBuffer) -> Self {
        (apply)();
        let engine = WhisperCandle::new();
        engine
            .transcribe(warm, &AsrOptions::default())
            .expect("warm-up transcribe");
        Arm {
            label,
            engine,
            apply,
        }
    }

    /// One timed round: the whole clip set through this arm.
    fn round(&self, clips: &[ffai_core::types::AudioBuffer]) -> (f64, String) {
        (self.apply)();
        let opts = AsrOptions::default();
        let t = Instant::now();
        let mut text = String::new();
        for audio in clips {
            let tr = self.engine.transcribe(audio, &opts).expect("transcribe");
            for seg in &tr.segments {
                text.push_str(&seg.value);
            }
        }
        (t.elapsed().as_secs_f64() * 1e3, text)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| default.to_string())
    };
    let test = arg("--test", "null");
    let rounds: usize = arg("--rounds", "41").parse()?;
    let n_clips: usize = arg("--clips", "4").parse()?;

    let dir = PathBuf::from("corpora/clips/librispeech-test-clean/audio");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "wav"))
        .collect();
    paths.sort();
    paths.truncate(n_clips);
    if paths.is_empty() {
        return Err(
            "no clips — run `cargo run -p ffai-bench --example prepare_librispeech`".into(),
        );
    }
    let clips: Vec<_> = paths
        .iter()
        .map(|p| ffai_media::load_audio(p))
        .collect::<Result<Vec<_>, _>>()?;
    let media: f64 = clips.iter().map(|a| a.duration_secs()).sum();

    // Two arms. `null` deliberately configures both identically — the only
    // test here whose true answer is known in advance, which is what makes it
    // a calibration of the harness rather than a measurement of the code.
    let (a_label, a_apply, b_label, b_apply): (&str, Apply, &str, Apply) = match test.as_str() {
        "null" => ("null-A", Box::new(|| {}), "null-B", Box::new(|| {})),
        "mlp_int8" => (
            "MLP int8",
            Box::new(|| knobs::MLP_INT8_ENABLED.set(true)),
            "MLP f16 (shipped)",
            Box::new(|| knobs::MLP_INT8_ENABLED.set(false)),
        ),
        "par_heads" => (
            "parallel heads",
            Box::new(|| knobs::PAR_HEADS.set(true)),
            "serial heads (shipped)",
            Box::new(|| knobs::PAR_HEADS.set(false)),
        ),
        "kv_f16" => (
            "f16 KV cache (shipped)",
            Box::new(|| knobs::KV_F16_DISABLED.set(false)),
            "f32 KV cache",
            Box::new(|| knobs::KV_F16_DISABLED.set(true)),
        ),
        "min_keys" => (
            "fused self-attn (min 8)",
            Box::new(|| knobs::DECODE_MIN_KEYS.set(8)),
            "three-op self-attn (min 1e9)",
            Box::new(|| knobs::DECODE_MIN_KEYS.set(1_000_000_000)),
        ),
        "enc_kt" => (
            "K transposed at projection",
            Box::new(|| knobs::ENC_KT_DISABLED.set(false)),
            "K gathered in prep (old)",
            Box::new(|| knobs::ENC_KT_DISABLED.set(true)),
        ),
        "gemv_pad" => (
            "GEMV padding (shipped)",
            Box::new(|| knobs::GEMV_PAD_DISABLED.set(false)),
            "no GEMV padding",
            Box::new(|| knobs::GEMV_PAD_DISABLED.set(true)),
        ),
        other => return Err(format!("unknown --test `{other}`").into()),
    };

    println!(
        "pipeline A/B · test={test} · {} clips, {media:.1}s audio · {rounds} paired rounds",
        clips.len()
    );
    println!("  A = {a_label}\n  B = {b_label}\n");

    let a = Arm::new(a_label, a_apply, &clips[0]);
    let b = Arm::new(b_label, b_apply, &clips[0]);

    let (mut ta, mut tb, mut a_wins) = (Vec::new(), Vec::new(), 0usize);
    let (mut text_a, mut text_b) = (String::new(), String::new());
    for i in 0..rounds {
        // ABBA: alternate which arm runs first so a "second one is warmer"
        // effect cancels instead of accumulating into one arm.
        let (ra, rb) = if i % 2 == 0 {
            let ra = a.round(&clips);
            (ra, b.round(&clips))
        } else {
            let rb = b.round(&clips);
            (a.round(&clips), rb)
        };
        if ra.0 < rb.0 {
            a_wins += 1;
        }
        ta.push(ra.0);
        tb.push(rb.0);
        text_a = ra.1;
        text_b = rb.1;
        print!("\r  round {}/{rounds}", i + 1);
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!("\r{:32}\r", "");

    ta.sort_by(f64::total_cmp);
    tb.sort_by(f64::total_cmp);
    let (ma, mb) = (ta[rounds / 2], tb[rounds / 2]);
    let overlap = ta[0] <= *tb.last().unwrap() && tb[0] <= *ta.last().unwrap();
    let z = (a_wins as f64 - 0.5 * rounds as f64) / (0.5 * (rounds as f64).sqrt());

    println!(
        "  {a_label:<26} med {ma:8.1} ms  [{:.1} .. {:.1}]",
        ta[0],
        ta.last().unwrap()
    );
    println!(
        "  {b_label:<26} med {mb:8.1} ms  [{:.1} .. {:.1}]",
        tb[0],
        tb.last().unwrap()
    );
    println!(
        "  paired: A won {a_wins}/{rounds} ({:.0} %) · median ratio {:.3}x · ranges {}",
        a_wins as f64 / rounds as f64 * 100.0,
        mb / ma,
        if overlap { "OVERLAP" } else { "DISJOINT" }
    );

    // Prove the arms are actually doing different work. A change that alters
    // numerics (int8, a dtype swap) MUST move some token somewhere; identical
    // transcripts mean the knob never reached the code path, which has
    // silently wasted a day in this campaign before.
    let same = text_a == text_b;
    println!(
        "  transcripts: {} — {}",
        if same { "IDENTICAL" } else { "DIFFER" },
        match (test.as_str(), same) {
            ("null", true) => "expected (same config both arms)",
            ("null", false) => "!! the null arm is NOT deterministic — fix before any A/B",
            ("min_keys", true) | ("par_heads", true) | ("gemv_pad", true) =>
                "expected (numerically exact restructure)",
            (_, true) => "!! suspicious for a numerics-changing knob — did it reach the path?",
            (_, false) => "expected (this knob changes arithmetic)",
        }
    );

    println!(
        "\n  VERDICT: {}",
        if test == "null" {
            format!(
                "harness floor is +/-{:.1} % (|z|={:.1}). Nothing smaller is measurable.",
                ((mb / ma) - 1.0).abs() * 100.0,
                z.abs()
            )
        } else if z.abs() < 2.0 {
            format!("INCONCLUSIVE (z={z:+.1}) — no reliable difference at the pipeline level")
        } else if z > 0.0 {
            format!("{a_label} FASTER (z={z:+.1}, {:.3}x)", mb / ma)
        } else {
            format!("{b_label} FASTER (z={z:+.1}, {:.3}x)", ma / mb)
        }
    );
    Ok(())
}
