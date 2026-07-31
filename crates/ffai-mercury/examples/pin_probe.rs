//! Depth-6 instrument: is the TTS speed comparison measuring CODE, or is it
//! measuring which class of core the Windows scheduler happened to pick?
//!
//! This box is an i7-14650HX: 8 P-cores (logical 0..15, SMT) + 8 E-cores
//! (logical 16..23, no SMT). Every number this campaign has recorded was taken
//! with the scheduler free to place threads on either class, and rayon's
//! default pool spans all 24 logical processors. Two distinct hazards:
//!
//!   1. PER-CORE IPC — an E-core is materially slower than a P-core. A run that
//!      lands on E-cores is slower for reasons that have nothing to do with the
//!      kernel under test.
//!   2. STRAGGLER — rayon splits a parallel region evenly, but E-cores retire
//!      their share late and the join waits for the slowest worker. Spreading
//!      onto E-cores can be SLOWER than not using them at all.
//!
//! Hazard 1 is an instrument defect. Hazard 2 is a real optimization lever.
//! This probe prices both, and reports CPU time alongside wall time, because
//! wall time accrues while descheduled and CPU time does not.
//!
//! ```text
//! FFAI_PIN=p1 cargo run --release -p ffai-mercury --example pin_probe
//! ```
//!
//! FFAI_PIN: p1 = one P-core, e1 = one E-core, p = all P-cores,
//!           all = every logical processor, none = leave it to the scheduler.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ffai_mercury::tts::phonemize::Phonemizer;
use ffai_mercury::tts::vits::{GaussRng, Vits};

#[cfg(windows)]
mod win {
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn SetProcessAffinityMask(h: isize, mask: usize) -> i32;
        fn SetPriorityClass(h: isize, class: u32) -> i32;
        fn GetProcessTimes(
            h: isize,
            creation: *mut u64,
            exit: *mut u64,
            kernel: *mut u64,
            user: *mut u64,
        ) -> i32;
    }

    /// HIGH_PRIORITY_CLASS — keeps the foreground scheduler from preempting us
    /// with unrelated desktop work mid-measurement.
    const HIGH_PRIORITY_CLASS: u32 = 0x0000_0080;

    pub fn pin(mask: usize) -> bool {
        unsafe {
            let h = GetCurrentProcess();
            SetPriorityClass(h, HIGH_PRIORITY_CLASS);
            SetProcessAffinityMask(h, mask) != 0
        }
    }

    pub fn raise_priority() {
        unsafe {
            SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        }
    }

    /// Total CPU time (kernel + user) charged to this process, in seconds.
    /// Unlike wall time this does not accrue while descheduled, so it is the
    /// honest instrument on a box with foreign load.
    pub fn cpu_secs() -> f64 {
        unsafe {
            let (mut c, mut e, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
            if GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) == 0 {
                return 0.0;
            }
            // FILETIME ticks are 100 ns.
            (k + u) as f64 * 1e-7
        }
    }
}

#[cfg(not(windows))]
mod win {
    pub fn pin(_mask: usize) -> bool {
        false
    }
    pub fn raise_priority() {}
    pub fn cpu_secs() -> f64 {
        0.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let which = std::env::var("FFAI_PIN").unwrap_or_else(|_| "none".into());
    // Logical 0..15 are the 8 SMT P-cores; 16..23 are the 8 E-cores.
    // For P-core sets take one logical per physical core (even bits) so we
    // measure cores, not hyperthread siblings contending for one core.
    let mask: usize = match which.as_str() {
        "p1" => 0x1,          // a single P-core
        "e1" => 0x1_0000,     // a single E-core
        "p" => 0x5555,        // 8 P-cores, one thread each
        "all" => 0xFF_5555,   // 8 P-cores + 8 E-cores
        _ => 0,               // scheduler's choice
    };
    if mask != 0 {
        if !win::pin(mask) {
            eprintln!("warning: could not set affinity mask {mask:#x}");
        }
    } else {
        win::raise_priority();
    }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    println!("pin={which} mask={mask:#x} available_parallelism={threads}");

    let cache = std::env::var("FFAI_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("ffai"));
    let vits = Vits::load(&cache.join("models").join("piper-vits-lessac-medium"))?;
    let phonemizer = Phonemizer::from_manifest_dir(Path::new("models"))?;

    let manifest =
        ffai_bench::corpus::Manifest::load(Path::new("corpora/harvard-sentences-v1.toml"))?;
    let texts: Vec<String> = manifest
        .holdout()
        .take(20)
        .map(|c| std::fs::read_to_string(manifest.clip_path(c)).unwrap().trim().to_string())
        .collect();
    let ids_list: Vec<Vec<i64>> = texts
        .iter()
        .map(|t| vits.id_map.sentence_to_ids(&phonemizer.phonemize(t).unwrap()).0)
        .collect();

    // Hand the reference the EXACT id sequences we run, so the two arms drive
    // the same graph with the same lengths and emit the same audio. Comparing
    // "20 sentences" to "20 different sentences" compares audio durations, not
    // engines.
    if let Ok(path) = std::env::var("FFAI_DUMP_IDS") {
        let mut out = String::new();
        for ids in &ids_list {
            let row: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
            out.push('[');
            out.push_str(&row.join(","));
            out.push_str("]\n");
        }
        std::fs::write(&path, out)?;
        println!("wrote {} id rows to {path}", ids_list.len());
    }

    // Warm: first pass touches weights and lets rayon build its pool, so the
    // measured passes are not paying one-time costs.
    for ids in &ids_list {
        let (m_p, logs_p, hidden) = vits.text_encoder(ids)?;
        let mut rng = GaussRng::new(0);
        let w = vits.durations(&hidden, 0.8, 1.0, &mut rng)?;
        let (m_e, _) = vits.expand_prior(&m_p, &logs_p, &w)?;
        let z = vits.flow_reverse(&m_e)?;
        vits.decode(&z)?;
    }

    let mut best = [f64::MAX; 4];
    // CPU time per stage as well as wall: cpu/wall IS the stage's effective
    // core occupancy, and a stage stuck near 1.0 is running serial while the
    // other 23 logical processors idle. That is invisible to wall time alone.
    let mut best_cpu = [f64::MAX; 4];
    let mut best_total_wall = f64::MAX;
    let mut best_total_cpu = f64::MAX;
    let mut audio_secs = 0f64;
    for _ in 0..5 {
        let (mut t_enc, mut t_dp, mut t_flow, mut t_dec) = (0f64, 0f64, 0f64, 0f64);
        let (mut c_enc, mut c_dp, mut c_flow, mut c_dec) = (0f64, 0f64, 0f64, 0f64);
        audio_secs = 0.0;
        let w0 = Instant::now();
        let c0 = win::cpu_secs();
        for ids in &ids_list {
            let (t0, k0) = (Instant::now(), win::cpu_secs());
            let (m_p, logs_p, hidden) = vits.text_encoder(ids)?;
            t_enc += t0.elapsed().as_secs_f64();
            c_enc += win::cpu_secs() - k0;

            let (t0, k0) = (Instant::now(), win::cpu_secs());
            let mut rng = GaussRng::new(0);
            let w = vits.durations(&hidden, 0.8, 1.0, &mut rng)?;
            t_dp += t0.elapsed().as_secs_f64();
            c_dp += win::cpu_secs() - k0;

            let (t0, k0) = (Instant::now(), win::cpu_secs());
            let (m_e, _logs_e) = vits.expand_prior(&m_p, &logs_p, &w)?;
            let z = vits.flow_reverse(&m_e)?;
            t_flow += t0.elapsed().as_secs_f64();
            c_flow += win::cpu_secs() - k0;

            let (t0, k0) = (Instant::now(), win::cpu_secs());
            let audio = vits.decode(&z)?;
            t_dec += t0.elapsed().as_secs_f64();
            c_dec += win::cpu_secs() - k0;
            audio_secs += audio.len() as f64 / vits.sample_rate as f64;
        }
        best_total_wall = best_total_wall.min(w0.elapsed().as_secs_f64());
        best_total_cpu = best_total_cpu.min(win::cpu_secs() - c0);
        for (b, t) in best.iter_mut().zip([t_enc, t_dp, t_flow, t_dec]) {
            *b = b.min(t);
        }
        for (b, t) in best_cpu.iter_mut().zip([c_enc, c_dp, c_flow, c_dec]) {
            *b = b.min(t);
        }
    }

    println!("  {audio_secs:.1}s audio, best-of-5:");
    println!("    {:<14} {:>9} {:>9} {:>7}", "stage", "wall ms", "cpu ms", "cores");
    for ((name, t), c) in ["text_encoder", "duration_pred", "flow", "decoder"]
        .iter()
        .zip(best)
        .zip(best_cpu)
    {
        println!("    {name:<14} {:>9.1} {:>9.1} {:>7.2}", t * 1000.0, c * 1000.0, c / t);
    }
    println!(
        "    {:<14} {:>8.1} ms wall  {:>8.1} ms cpu  -> {:.2}x realtime  (cpu/wall {:.2})",
        "TOTAL",
        best_total_wall * 1000.0,
        best_total_cpu * 1000.0,
        audio_secs / best_total_wall,
        best_total_cpu / best_total_wall,
    );
    Ok(())
}
