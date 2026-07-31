//! D2/D3 for the decoder — the stage I optimized AROUND rather than into.
//!
//! It is 1024 ms of a ~1958 ms pipeline (52%) at 9.90 cores, and every earlier
//! iteration attacked stages worth 10-24%. The skill's rule is rank by ABSOLUTE
//! cost, so this is overdue.
//!
//! HiFi-GAN upsamples 256x in three stages, so the three stages work on wildly
//! different tensor lengths and a lumped total cannot say which owns the cost.
//! Run with FFAI_PROFILE=1 to get the per-stage split from inside FlatDecoder;
//! this harness supplies the aggregate wall/cpu/GFLOP-per-second context.
//!
//! ```text
//! FFAI_PROFILE=1 cargo run --release -p ffai-mercury --example dec_anatomy
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::{GaussRng, Vits};

#[cfg(windows)]
unsafe extern "system" {
    fn GetCurrentProcess() -> isize;
    fn GetProcessTimes(h: isize, c: *mut u64, e: *mut u64, k: *mut u64, u: *mut u64) -> i32;
}

fn cpu_secs() -> f64 {
    #[cfg(windows)]
    unsafe {
        let (mut c, mut e, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
        if GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) == 0 {
            return 0.0;
        }
        (k + u) as f64 * 1e-7
    }
    #[cfg(not(windows))]
    0.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::var("FFAI_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("ffai"));
    let vits = Vits::load(&cache.join("models").join("piper-vits-lessac-medium"))?;
    let phonemizer = Phonemizer::from_manifest_dir(Path::new("models"))?;

    let manifest =
        ffai_bench::corpus::Manifest::load(Path::new("corpora/harvard-sentences-v1.toml"))?;
    let ids_list: Vec<Vec<i64>> = manifest
        .holdout()
        .take(20)
        .map(|c| {
            let t = std::fs::read_to_string(manifest.clip_path(c)).unwrap();
            vits.id_map.sentence_to_ids(&phonemizer.phonemize(t.trim()).unwrap()).0
        })
        .collect();

    // The decoder's inputs, computed once so the upstream stages are excluded.
    let mut zs = Vec::new();
    for ids in &ids_list {
        let (m_p, logs_p, hidden) = vits.text_encoder(ids)?;
        let mut rng = GaussRng::new(0);
        let w = vits.durations(&hidden, 0.8, 1.0, &mut rng)?;
        let (m_e, _) = vits.expand_prior(&m_p, &logs_p, &w)?;
        zs.push(vits.flow_reverse(&m_e)?);
    }
    let frames: usize = zs.iter().map(|z| z.dim(2).unwrap()).sum();
    let samples = frames * 256; // three upsampling stages, 256x overall
    println!("20 sentences: {frames} latent frames -> {samples} samples ({:.1}s audio)", samples as f64 / 22050.0);

    // Warm.
    for z in &zs {
        vits.decode(z)?;
    }

    let (mut best, mut best_cpu) = (f64::MAX, f64::MAX);
    for _ in 0..5 {
        let (t0, c0) = (Instant::now(), cpu_secs());
        for z in &zs {
            std::hint::black_box(vits.decode(z)?);
        }
        let w = t0.elapsed().as_secs_f64();
        if w < best {
            best = w;
            best_cpu = cpu_secs() - c0;
        }
    }

    // Rough MAC count for the whole decoder, dominated by the resblock convs
    // and the transposed upsamplers. Channels halve as length grows: the
    // lessac-medium generator is 512 -> 256 -> 128 -> 64 over ups of 8, 8, 4.
    let f = frames as f64;
    let mut gflop = 0.0;
    let mut len = f;
    let mut ch = 512.0;
    for (up, k) in [(8.0, 16.0), (8.0, 16.0), (4.0, 8.0)] {
        // transposed conv: ch -> ch/2 over len*up outputs, kernel k
        gflop += len * up * ch * (ch / 2.0) * k / (up) * 2.0;
        len *= up;
        ch /= 2.0;
        // three resblocks, two convs each, kernels 3/7/11 -> ~7 average
        gflop += 3.0 * 2.0 * len * ch * ch * 7.0 * 2.0;
    }
    gflop /= 1e9;

    println!(
        "  decode x20   {:>8.1} ms wall  {:>8.1} ms cpu  {:>5.2} cores  ~{:.1} GFLOP -> {:.1} GFLOP/s",
        best * 1000.0,
        best_cpu * 1000.0,
        best_cpu / best,
        gflop,
        gflop / best
    );
    println!("  (per-stage split above, from FFAI_PROFILE=1 inside FlatDecoder)");
    Ok(())
}
