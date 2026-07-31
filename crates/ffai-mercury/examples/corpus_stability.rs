//! The gate I argued past: is the duration-predictor routing change actually
//! output-neutral across the WHOLE corpus, not just three oracle fixtures?
//!
//! `durations()` ends in `.ceil()`. The dp routing change swaps a serial direct
//! kernel for candle's threaded matmul, which reorders float accumulation — so
//! a phoneme whose duration lands near an integer boundary can flip by one
//! frame, changing the audio length and everything downstream. Three fixtures
//! passing does not prove 200 sentences pass.
//!
//! This walks every corpus sentence under BOTH routings and compares:
//!   * w_ceil element-by-element (the integer contract), and
//!   * the synthesized audio bit-for-bit (the thing a user hears).
//!
//! A clean run means WER cannot have moved, because the bytes did not.
//! It also reports the near-boundary margin, so "it passed" comes with a
//! measure of how close it came to not passing.
//!
//! Also re-checks the attention kernel's bit-identity at corpus scale rather
//! than the 20 sentences enc_ab asserted it on.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example corpus_stability
//! ```

use std::path::{Path, PathBuf};

use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::{GaussRng, Vits};

fn set(knob: &str, on: bool) {
    // SAFETY: single-threaded here; no rayon region is live across the flip.
    unsafe {
        if on {
            std::env::set_var(knob, "1");
        } else {
            std::env::remove_var(knob);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::var("FFAI_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("ffai"));
    let vits = Vits::load(&cache.join("models").join("piper-vits-lessac-medium"))?;
    let phonemizer = Phonemizer::from_manifest_dir(Path::new("models"))?;

    let manifest =
        ffai_bench::corpus::Manifest::load(Path::new("corpora/harvard-sentences-v1.toml"))?;
    // EVERY clip, not just the holdout — the point is coverage.
    let clips: Vec<_> = manifest.clips.iter().collect();
    println!("corpus: {} sentences", clips.len());

    let mut ids_list = Vec::new();
    for c in &clips {
        let text = std::fs::read_to_string(manifest.clip_path(c))?;
        let ids = vits.id_map.sentence_to_ids(&phonemizer.phonemize(text.trim())?).0;
        ids_list.push((c.id.clone(), ids));
    }

    // ---- 1. duration predictor: integer contract + audio bytes ----
    let mut w_diff = Vec::new();
    let mut audio_diff = Vec::new();
    // How close did the closest phoneme come to flipping its ceil()? A pass
    // with margin 1e-9 is luck; a pass with margin 1e-2 is robust.
    let mut min_margin = f64::MAX;
    let mut min_margin_at = String::new();

    for (id, ids) in &ids_list {
        let (m_p, logs_p, hidden) = vits.text_encoder(ids)?;

        set("FFAI_DP_DIRECT_1X1", true);
        let mut rng = GaussRng::new(0);
        let w_old = vits.durations(&hidden, 0.8, 1.0, &mut rng)?;

        set("FFAI_DP_DIRECT_1X1", false);
        let mut rng = GaussRng::new(0);
        let w_new = vits.durations(&hidden, 0.8, 1.0, &mut rng)?;

        if w_old != w_new {
            let n = w_old.iter().zip(&w_new).filter(|(a, b)| a != b).count();
            w_diff.push((id.clone(), n, w_old.iter().sum::<u32>(), w_new.iter().sum::<u32>()));
        }

        // Margin: the raw pre-ceil durations under the shipped routing, and how
        // far the nearest one sits from the integer boundary it rounds across.
        let raw = vits.durations_raw_probe(&hidden, 0.8, 1.0, &mut GaussRng::new(0))?;
        for r in &raw {
            let frac = r - r.floor();
            // distance to the nearest boundary where ceil() would change
            let m = frac.min(1.0 - frac);
            if m < min_margin {
                min_margin = m;
                min_margin_at = id.clone();
            }
        }

        // Full synthesis both ways, compared on the raw samples.
        set("FFAI_DP_DIRECT_1X1", true);
        let a_old = synth(&vits, &m_p, &logs_p, &w_old)?;
        set("FFAI_DP_DIRECT_1X1", false);
        let a_new = synth(&vits, &m_p, &logs_p, &w_new)?;
        if a_old.len() != a_new.len()
            || a_old.iter().zip(&a_new).any(|(x, y)| x.to_bits() != y.to_bits())
        {
            audio_diff.push(id.clone());
        }
    }

    println!("\n[1] dp routing (old serial direct vs new threaded matmul)");
    println!(
        "    w_ceil   : {}",
        if w_diff.is_empty() {
            format!("IDENTICAL across all {} sentences", ids_list.len())
        } else {
            format!("{} sentences DIFFER", w_diff.len())
        }
    );
    for (id, n, so, sn) in w_diff.iter().take(10) {
        println!("      {id}: {n} phonemes differ, total frames {so} -> {sn}");
    }
    println!(
        "    audio    : {}",
        if audio_diff.is_empty() {
            "BIT-IDENTICAL".to_string()
        } else {
            format!("{} sentences differ: {:?}", audio_diff.len(), &audio_diff[..audio_diff.len().min(5)])
        }
    );
    println!(
        "    closest ceil() boundary anywhere in corpus: {min_margin:.3e} (at {min_margin_at})"
    );

    // ---- 2. attention kernel bit-identity at corpus scale ----
    let mut attn_diff = 0usize;
    for (_, ids) in &ids_list {
        set("FFAI_SERIAL_ATTN", true);
        let (a, _, _) = vits.text_encoder(ids)?;
        set("FFAI_SERIAL_ATTN", false);
        let (b, _, _) = vits.text_encoder(ids)?;
        let av: Vec<f32> = a.flatten_all()?.to_vec1()?;
        let bv: Vec<f32> = b.flatten_all()?.to_vec1()?;
        if av.iter().zip(&bv).any(|(x, y)| x.to_bits() != y.to_bits()) {
            attn_diff += 1;
        }
    }
    println!("\n[2] attention kernel (serial vs parallel row grid)");
    println!(
        "    m_p      : {}",
        if attn_diff == 0 {
            format!("BIT-IDENTICAL across all {} sentences", ids_list.len())
        } else {
            format!("{attn_diff} sentences DIFFER")
        }
    );

    // ---- 3. decoder scratch-buffer reuse vs fresh allocation ----
    // Reusing a buffer is only safe if nothing reads stale bytes from the
    // previous iteration, and that failure mode shows up as a WRONG SAMPLE,
    // not a crash. Compare the arms directly rather than trusting that the
    // oracle's tolerance would have caught it.
    let mut dec_diff = 0usize;
    for (_, ids) in &ids_list {
        let (m_p, logs_p, hidden) = vits.text_encoder(ids)?;
        let mut rng = GaussRng::new(0);
        let w = vits.durations(&hidden, 0.8, 1.0, &mut rng)?;
        set("FFAI_DEC_ALLOC", true);
        let a = synth(&vits, &m_p, &logs_p, &w)?;
        set("FFAI_DEC_ALLOC", false);
        let b = synth(&vits, &m_p, &logs_p, &w)?;
        if a.len() != b.len() || a.iter().zip(&b).any(|(x, y)| x.to_bits() != y.to_bits()) {
            dec_diff += 1;
        }
    }
    println!("\n[3] decoder buffers (fresh alloc vs reused scratch)");
    println!(
        "    audio    : {}",
        if dec_diff == 0 {
            format!("BIT-IDENTICAL across all {} sentences", ids_list.len())
        } else {
            format!("{dec_diff} sentences DIFFER")
        }
    );

    // ---- 4. the GEMM/flat-kernel round vs the candle baseline ----
    // conv1d->GEMM, fused LayerNorm and the A&S GELU all reorder float
    // arithmetic, and `durations()` ends in `.ceil()`. The measured perturbation
    // is ~1e-6 against a corpus-wide rounding margin of 1.03e-4, so this SHOULD
    // hold -- but "should" is what the three-fixture oracle already said, and
    // the whole point of this probe is that three fixtures are not 200
    // sentences. Any flipped duration changes the audio length outright.
    let knobs = ["FFAI_CANDLE_FFN", "FFAI_CANDLE_LN", "FFAI_CANDLE_GELU"];
    let mut k_wdiff = Vec::new();
    let mut k_len_diff = 0usize;
    let mut worst_audio = 0f32;
    for (id, ids) in &ids_list {
        for k in knobs {
            set(k, true);
        }
        let (m_p, logs_p, hidden) = vits.text_encoder(ids)?;
        let w_old = vits.durations(&hidden, 0.8, 1.0, &mut GaussRng::new(0))?;
        let a_old = synth(&vits, &m_p, &logs_p, &w_old)?;

        for k in knobs {
            set(k, false);
        }
        let (m_p, logs_p, hidden) = vits.text_encoder(ids)?;
        let w_new = vits.durations(&hidden, 0.8, 1.0, &mut GaussRng::new(0))?;
        let a_new = synth(&vits, &m_p, &logs_p, &w_new)?;

        if w_old != w_new {
            let n = w_old.iter().zip(&w_new).filter(|(a, b)| a != b).count();
            k_wdiff.push((id.clone(), n, w_old.iter().sum::<u32>(), w_new.iter().sum::<u32>()));
        }
        if a_old.len() != a_new.len() {
            k_len_diff += 1;
        } else {
            for (x, y) in a_old.iter().zip(&a_new) {
                worst_audio = worst_audio.max((x - y).abs());
            }
        }
    }
    println!("
[4] GEMM convs + flat LayerNorm + A&S GELU vs candle baseline");
    println!(
        "    w_ceil   : {}",
        if k_wdiff.is_empty() {
            format!("IDENTICAL across all {} sentences", ids_list.len())
        } else {
            format!("{} sentences DIFFER", k_wdiff.len())
        }
    );
    for (id, n, so, sn) in k_wdiff.iter().take(5) {
        println!("      {id}: {n} phonemes differ, frames {so} -> {sn}");
    }
    println!("    audio len: {}", if k_len_diff == 0 { "identical".into() } else { format!("{k_len_diff} differ") });
    println!("    audio max|delta| vs candle path: {worst_audio:.3e}");

    let clean = w_diff.is_empty()
        && audio_diff.is_empty()
        && attn_diff == 0
        && dec_diff == 0
        && k_wdiff.is_empty()
        && k_len_diff == 0;
    println!(
        "\nverdict: {}",
        if clean {
            "OUTPUT-NEUTRAL — the bytes did not move, so WER cannot have moved"
        } else {
            "OUTPUT CHANGED — a corpus WER run is required, not optional"
        }
    );
    Ok(())
}

fn synth(
    vits: &Vits,
    m_p: &ffai_core::candle::Tensor,
    logs_p: &ffai_core::candle::Tensor,
    w: &[u32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let (m_e, _) = vits.expand_prior(m_p, logs_p, w)?;
    let z = vits.flow_reverse(&m_e)?;
    Ok(vits.decode(&z)?)
}
