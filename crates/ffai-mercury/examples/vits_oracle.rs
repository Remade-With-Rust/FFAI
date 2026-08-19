//! M-T2 stage oracles: Mercury's candle VITS against piper's own onnxruntime
//! intermediates, stage by stage, at zero noise (both sides deterministic).
//!
//! ```text
//! cargo run --release -p ffai-mercury --example vits_oracle
//! ```
//!
//! Fixtures come from `corpora/refs/dump_vits_stages.py` (regenerable by
//! anyone with the voice file; not committed — *.safetensors is data).
//! Stage isolation is the point: each stage consumes the REFERENCE'S input,
//! so an error in one stage cannot hide or compound in the next. `w_ceil`
//! is integer frames and must match EXACTLY; float stages report max-abs
//! and relative-RMS deltas.

use std::path::{Path, PathBuf};

use ffai_core::candle::{IndexOp, Tensor};
use ffai_mercury::tts::vits::{GaussRng, SynthesisOptions, Vits};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::var("FFAI_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dirs_cache().join("ffai"));
    let model_dir = cache.join("models").join("piper-vits-lessac-medium");
    // `FFAI_VOICE_SAFETENSORS=1` loads the Python converter's output instead.
    // The default is the pure-Rust ONNX reader — the path a consumer takes —
    // so the stage oracles gate what actually ships.
    let onnx = std::path::PathBuf::from(".piper-voices/en_US-lessac-medium.onnx");
    let vits = if std::env::var("FFAI_VOICE_SAFETENSORS").is_ok() || !onnx.exists() {
        println!("loading via safetensors (converted)");
        Vits::load(&model_dir)?
    } else {
        println!("loading via ONNX (pure Rust)");
        Vits::load_onnx(&onnx, &onnx.with_extension("onnx.json"))?
    };
    println!(
        "voice loaded: {} Hz, defaults noise {:.3} / len {:.1} / w {:.1}",
        vits.sample_rate,
        vits.defaults.noise_scale,
        vits.defaults.length_scale,
        vits.defaults.noise_w
    );

    let fixture_dir = Path::new("crates/ffai-mercury/tests/fixtures/vits");
    let mut all_ok = true;
    for entry in std::fs::read_dir(fixture_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let fx = ffai_core::candle::safetensors::load(&path, &ffai_core::candle::Device::Cpu)?;
        let ids: Vec<i64> = fx["ids"]
            .i(0)?
            .to_dtype(ffai_core::candle::DType::I64)?
            .to_vec1()?;

        println!("\n=== {name} ({} ids) ===", ids.len());

        // Stage 1: text encoder.
        let (m_p, logs_p, hidden) = vits.text_encoder(&ids)?;
        let ok1a = report("m_p   ", &m_p, &fx["m_p"], 2e-3)?;
        let ok1b = report("logs_p", &logs_p, &fx["logs_p"], 2e-3)?;

        // Stage 2: durations (noise_w = 0: fully deterministic).
        let mut rng = GaussRng::new(0);
        let w_ceil = vits.durations(&hidden, 0.0, 1.0, &mut rng)?;
        let ref_w: Vec<f32> = fx["w_ceil"].i((0, 0))?.to_vec1()?;
        let ref_w: Vec<u32> = ref_w.iter().map(|w| *w as u32).collect();
        let ok2 = w_ceil == ref_w;
        let diff_count = w_ceil.iter().zip(&ref_w).filter(|(a, b)| a != b).count();
        println!(
            "w_ceil  {} ({} phonemes, {} differ; total {} vs {})",
            if ok2 { "EXACT" } else { "MISMATCH" },
            ref_w.len(),
            diff_count,
            w_ceil.iter().sum::<u32>(),
            ref_w.iter().sum::<u32>()
        );

        // Stage 3: flow, from the REFERENCE z_p.
        let z = vits.flow_reverse(&fx["z_p"])?;
        let ok3 = report("dec_in", &z, &fx["dec_in"], 2e-3)?;

        // Stage 4: generator, from the REFERENCE dec_in.
        let audio = vits.decode(&fx["dec_in"])?;
        let ref_audio: Vec<f32> = fx["audio"].flatten_all()?.to_vec1()?;
        let ok4 = report_vec("audio ", &audio, &ref_audio, 2e-2)?;

        // End to end at zero noise: our whole chain vs the reference's.
        let opts = SynthesisOptions {
            noise_scale: 0.0,
            length_scale: 1.0,
            noise_w: 0.0,
            seed: 0,
        };
        let ours = vits.synthesize_ids(&ids, &opts)?;
        let e2e = if ours.len() == ref_audio.len() {
            report_vec("e2e   ", &ours, &ref_audio, 2e-2)?
        } else {
            println!(
                "e2e    LENGTH MISMATCH: {} vs {}",
                ours.len(),
                ref_audio.len()
            );
            false
        };
        all_ok &= ok1a && ok1b && ok2 && ok3 && ok4 && e2e;
    }
    // Byte-stability: the M-T3 determinism gate. At the voice's own noisy
    // defaults, the same seed must reproduce the audio EXACTLY; a different
    // seed must not. (piper cannot make either guarantee.)
    let ids: Vec<i64> = vits.id_map.sentence_to_ids("ðə bˈɜːtʃ kənˈuː slˈɪd.").0;
    let noisy = SynthesisOptions {
        noise_scale: vits.defaults.noise_scale,
        length_scale: 1.0,
        noise_w: vits.defaults.noise_w,
        seed: 42,
    };
    let a = vits.synthesize_ids(&ids, &noisy)?;
    let b = vits.synthesize_ids(&ids, &noisy)?;
    let c = vits.synthesize_ids(&ids, &SynthesisOptions { seed: 43, ..noisy })?;
    let stable = a == b;
    let varies = a != c;
    println!(
        "\nbyte-stability: same seed {} ({} samples), different seed {}",
        if stable { "IDENTICAL" } else { "DIVERGED" },
        a.len(),
        if varies {
            "differs (as it must)"
        } else {
            "IDENTICAL (rng broken)"
        }
    );
    all_ok &= stable && varies;

    println!(
        "\nverdict: {}",
        if all_ok {
            "ALL STAGES PASS"
        } else {
            "STAGE FAILURES ABOVE"
        }
    );
    std::process::exit(if all_ok { 0 } else { 1 });
}

fn report(
    label: &str,
    ours: &Tensor,
    reference: &Tensor,
    tol: f32,
) -> Result<bool, Box<dyn std::error::Error>> {
    let a: Vec<f32> = ours.flatten_all()?.to_vec1()?;
    let b: Vec<f32> = reference.flatten_all()?.to_vec1()?;
    report_vec(label, &a, &b, tol)
}

fn report_vec(
    label: &str,
    a: &[f32],
    b: &[f32],
    tol: f32,
) -> Result<bool, Box<dyn std::error::Error>> {
    if a.len() != b.len() {
        println!("{label} SHAPE MISMATCH: {} vs {}", a.len(), b.len());
        return Ok(false);
    }
    let mut max_abs = 0f32;
    let (mut se, mut ref_sq) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        max_abs = max_abs.max((x - y).abs());
        se += ((x - y) as f64).powi(2);
        ref_sq += (*y as f64).powi(2);
    }
    let rel_rms = (se / ref_sq.max(1e-12)).sqrt();
    let ok = max_abs <= tol;
    println!(
        "{label} max|Δ| {max_abs:.2e}  rel-RMS {rel_rms:.2e}  ({} values) {}",
        a.len(),
        if ok { "OK" } else { "FAIL" }
    );
    Ok(ok)
}

fn dirs_cache() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        PathBuf::from(std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".into()))
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache")
            })
    }
}
