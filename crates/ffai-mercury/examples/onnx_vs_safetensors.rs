//! The gate for the pure-Rust ONNX loader: read a voice **both ways** and
//! prove they agree.
//!
//! ```text
//! cargo run --release -p ffai-mercury --example onnx_vs_safetensors
//! ```
//!
//! Arm A is `corpora/refs/dump_piper_weights.py` (Python + onnx), whose
//! output already passes the M-T2 acoustic oracle against piper's own
//! runtime. Arm B is `Vits::load_onnx`, which reads the `.onnx` directly.
//! If every tensor matches byte-for-byte and every conv geometry matches
//! exactly, arm B inherits arm A's oracle — which is the whole argument for
//! deleting the Python from the consumer's path.
//!
//! Byte equality, not a tolerance: both arms read the same IEEE-754 floats
//! out of the same file. Anything less than exact would mean one of them is
//! transforming values, and that is precisely what must not happen.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use ffai_mercury::tts::onnx;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let voice = PathBuf::from(".piper-voices/en_US-lessac-medium.onnx");
    if !voice.exists() {
        eprintln!(
            "missing {} — fetch it with:\n  .venv-bench/Scripts/python -m piper.download_voices \
             en_US-lessac-medium --data-dir .piper-voices",
            voice.display()
        );
        std::process::exit(2);
    }
    let cache = std::env::var("FFAI_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("ffai"));
    let converted = cache.join("models").join("piper-vits-lessac-medium");

    // ---- arm B: pure Rust, straight from the .onnx ----
    let t0 = std::time::Instant::now();
    let bytes = std::fs::read(&voice)?;
    let ours = onnx::recover(onnx::parse(&bytes)?)?;
    let rust_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "rust  : {} tensors, {} convs, {:.0} ms ({:.1} MB file)",
        ours.tensors.len(),
        ours.geometry.len(),
        rust_ms,
        bytes.len() as f64 / 1e6
    );

    // ---- arm A: the Python converter's safetensors ----
    let st = converted.join("vits.safetensors");
    if !st.exists() {
        eprintln!(
            "missing {} — run corpora/refs/dump_piper_weights.py",
            st.display()
        );
        std::process::exit(2);
    }
    let theirs = ffai_core::candle::safetensors::load(&st, &ffai_core::candle::Device::Cpu)?;
    println!("python: {} tensors", theirs.len());

    // ---- names ----
    let ours_names: BTreeSet<&str> = ours.tensors.keys().map(String::as_str).collect();
    let theirs_names: BTreeSet<&str> = theirs.keys().map(String::as_str).collect();
    let only_ours: Vec<&&str> = ours_names.difference(&theirs_names).collect();
    let only_theirs: Vec<&&str> = theirs_names.difference(&ours_names).collect();
    if !only_ours.is_empty() || !only_theirs.is_empty() {
        println!("\nNAME MISMATCH");
        if !only_ours.is_empty() {
            println!(
                "  only in rust  ({}): {:?}",
                only_ours.len(),
                &only_ours[..only_ours.len().min(8)]
            );
        }
        if !only_theirs.is_empty() {
            println!(
                "  only in python ({}): {:?}",
                only_theirs.len(),
                &only_theirs[..only_theirs.len().min(8)]
            );
        }
        std::process::exit(1);
    }
    println!("names : {} identical", ours_names.len());

    // ---- values, byte for byte ----
    let mut worst: Option<(String, usize)> = None;
    let mut checked = 0usize;
    for name in &ours_names {
        let a = &ours.tensors[*name];
        let b: Vec<f32> = theirs[*name].flatten_all()?.to_vec1()?;
        let b_dims = theirs[*name].dims().to_vec();
        if a.dims != b_dims {
            println!(
                "\nSHAPE MISMATCH {name}: rust {:?} vs python {:?}",
                a.dims, b_dims
            );
            std::process::exit(1);
        }
        let differing = a
            .data
            .iter()
            .zip(&b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        if differing > 0 && worst.as_ref().is_none_or(|(_, n)| differing > *n) {
            worst = Some(((*name).to_string(), differing));
        }
        checked += a.data.len();
    }
    match &worst {
        Some((name, n)) => {
            println!("\nVALUE MISMATCH: `{name}` differs in {n} of its values");
            std::process::exit(1);
        }
        None => println!("values: {checked} floats, byte-identical"),
    }

    // ---- geometry ----
    let graph_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(converted.join("vits-graph.json"))?)?;
    let attrs = graph_json
        .get("conv_attrs")
        .and_then(|v| v.as_object())
        .unwrap();
    let mut geom_bad = 0;
    for (name, g) in &ours.geometry {
        let Some(a) = attrs.get(name) else {
            println!("  geometry: `{name}` missing from the python side");
            geom_bad += 1;
            continue;
        };
        let first = |k: &str, d: usize| {
            a.get(k)
                .and_then(|v| v.as_array())
                .and_then(|v| v.first())
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(d)
        };
        let same = g.transpose == (a.get("op").and_then(|v| v.as_str()) == Some("ConvTranspose"))
            && g.stride == first("strides", 1)
            && g.pad == first("pads", 0)
            && g.dilation == first("dilations", 1);
        if !same {
            println!("  geometry: `{name}` rust {g:?} vs python {a}");
            geom_bad += 1;
        }
    }
    if geom_bad > 0 || ours.geometry.len() != attrs.len() {
        println!(
            "\nGEOMETRY MISMATCH ({geom_bad} bad, {} vs {})",
            ours.geometry.len(),
            attrs.len()
        );
        std::process::exit(1);
    }
    println!("geom  : {} convolutions identical", ours.geometry.len());

    // ---- and the whole engine, end to end, with load cost for both ----
    let config = voice.with_extension("onnx.json");
    let t = std::time::Instant::now();
    let a = ffai_mercury::tts::vits::Vits::load_onnx(&voice, &config)?;
    let load_onnx_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = std::time::Instant::now();
    let b = ffai_mercury::tts::vits::Vits::load(&converted)?;
    let load_st_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "load  : onnx {load_onnx_ms:.0} ms vs safetensors {load_st_ms:.0} ms ({:+.0} ms)",
        load_onnx_ms - load_st_ms
    );

    // Synthesis throughput must be unaffected: the tensors are identical, so
    // any difference would mean the LOADER changed memory layout.
    let ids_bench = a
        .id_map
        .sentence_to_ids("ðə bˈɜːtʃ kənˈuː slˈɪd ɔnðə smˈuːð plˈæŋks.")
        .0;
    let bench_opts = ffai_mercury::tts::vits::SynthesisOptions {
        noise_scale: 0.667,
        length_scale: 1.0,
        noise_w: 0.8,
        seed: 3,
    };
    let mut best = [f64::MAX; 2];
    for _ in 0..5 {
        for (i, v) in [&a, &b].iter().enumerate() {
            let t = std::time::Instant::now();
            v.synthesize_ids(&ids_bench, &bench_opts)?;
            best[i] = best[i].min(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    println!(
        "synth : onnx {:.0} ms vs safetensors {:.0} ms (best of 5, interleaved) — {:.2}x",
        best[0],
        best[1],
        best[1] / best[0]
    );
    let ids = a.id_map.sentence_to_ids("ðə bˈɜːtʃ kənˈuː slˈɪd.").0;
    let opts = ffai_mercury::tts::vits::SynthesisOptions {
        noise_scale: 0.667,
        length_scale: 1.0,
        noise_w: 0.8,
        seed: 7,
    };
    let wav_a = a.synthesize_ids(&ids, &opts)?;
    let wav_b = b.synthesize_ids(&ids, &opts)?;
    let identical = wav_a == wav_b;
    println!(
        "audio : {} samples each, {}",
        wav_a.len(),
        if identical {
            "byte-identical"
        } else {
            "DIFFERENT"
        }
    );

    println!(
        "\nverdict: {}",
        if identical {
            "ONNX loader matches the Python converter"
        } else {
            "MISMATCH"
        }
    );
    let _ = Path::new("");
    std::process::exit(if identical { 0 } else { 1 });
}
