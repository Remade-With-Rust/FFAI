//! Calibrate the footprint instrument against processes of KNOWN size.
//!
//! The first tree-scoped run reported whisper.cpp at 3753 MiB against our
//! 536 MiB — a 7x advantage. That is an extraordinary claim for a model whose
//! own init log accounts for ~224 MB (77.7 MB weights + ~146 MB of compute
//! buffers), and this campaign's rule is that a result which would be
//! extraordinary if true means the probe is broken until shown otherwise.
//! The previous version of this same instrument reported 5 MiB for the same
//! reference.
//!
//! So: measure processes whose size is known independently, and see whether
//! the numbers land where they should.
//!
//! ```sh
//! cargo run --release -p ffai-bench --example footprint_calibrate
//! ```

use std::process::{Command, Stdio};

use ffai_bench::footprint::Job;

fn measure(label: &str, argv: &[&str]) {
    let Some((prog, args)) = argv.split_first() else { return };
    let job = std::sync::Arc::new(Job::create());
    let child = Command::new(prog)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let Ok(mut child) = child else {
        println!("{label:<44} (could not launch)");
        return;
    };
    if let Some(j) = job.as_ref() {
        j.assign(&child);
    }
    // Sample the tree's RESIDENT memory while it runs, exactly as the bench
    // does, and keep the maximum.
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let seen = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let sampler = {
        let (job, done, seen) = (job.clone(), done.clone(), seen.clone());
        std::thread::spawn(move || {
            use std::sync::atomic::Ordering;
            while !done.load(Ordering::Relaxed) {
                if let Some(ws) = job.as_ref().as_ref().and_then(|j| j.working_set_now()) {
                    seen.fetch_max(ws, Ordering::Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        })
    };
    // Drain so a full pipe cannot deadlock the child.
    let out = child.stdout.take();
    let err = child.stderr.take();
    let t1 = std::thread::spawn(move || {
        use std::io::Read;
        let mut b = Vec::new();
        if let Some(mut o) = out {
            o.read_to_end(&mut b).ok();
        }
        b
    });
    let t2 = std::thread::spawn(move || {
        use std::io::Read;
        let mut b = Vec::new();
        if let Some(mut e) = err {
            e.read_to_end(&mut b).ok();
        }
        b
    });
    let _ = child.wait();
    done.store(true, std::sync::atomic::Ordering::Relaxed);
    sampler.join().ok();
    let sampled_ws = seen.load(std::sync::atomic::Ordering::Relaxed);
    let commit = job.as_ref().as_ref().and_then(|j| j.peak_commit());
    let direct = ffai_bench::footprint::peak_child(&child).map(|p| p.0);
    t1.join().ok();
    t2.join().ok();

    let mib = |b: Option<u64>| {
        b.map(|b| format!("{:.0} MiB", b as f64 / (1024.0 * 1024.0)))
            .unwrap_or_else(|| "-".into())
    };
    println!(
        "{label:<44} tree-RSS {:>9}   commit {:>9}   direct {:>9}",
        mib(Some(sampled_ws)),
        mib(commit),
        mib(direct)
    );
}

fn main() {
    println!("footprint calibration — expected sizes on the left\n");

    // A process that does essentially nothing. If this reads in the hundreds
    // of MiB the instrument is measuring something other than memory used.
    measure("trivial process (cmd /c echo)", &["cmd", "/c", "echo", "hi"]);

    // The launcher alone: a Python interpreter with no model.
    measure("python -c pass", &[".venv-bench/Scripts/python.exe", "-c", "pass"]);

    // whisper-cli printing help: binary + DLLs loaded, no model, no inference.
    measure("whisper-cli --help", &[".whispercpp/whisper-cli.exe", "--help"]);

    // whisper-cli doing real work on ONE clip. Its own init log accounts for
    // ~224 MB, so this is the number with a known independent estimate.
    measure(
        "whisper-cli, 1 clip (init log says ~224 MB)",
        &[
            ".whispercpp/whisper-cli.exe",
            "-m",
            ".whispercpp/ggml-tiny.en.bin",
            "-t",
            "24",
            "-bs",
            "1",
            "-bo",
            "1",
            "corpora/clips/librispeech-test-clean/audio/1089-134686-0000.wav",
        ],
    );

    println!(
        "\nRead the trivial row first: it is the instrument's own floor.\n\
         If `job` is far above `direct-child` on a SINGLE-process command,\n\
         the job figure is counting something other than that process."
    );
}
