//! The vision tower's cost, **counted rather than timed**.
//!
//! Deterministic: the same image gives the same numbers on any machine, under
//! any load. That is the point — round 2 opened by chasing a ±12 % wall-clock
//! swing that was larger than most of the wins being tested, and a change that
//! must have been faster measured slower three times running.
//!
//! ```sh
//! cargo run --release -p ffai-argus --example cost_report
//! ```
use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or("workspace root")?;
    let manifests = ffai_models::load_dir(&root.join("models"))?;
    let m = manifests
        .iter()
        .find(|m| m.name == ffai_argus::engine::MODEL)
        .ok_or("no manifest")?;
    let r = m.fetch()?;
    let weights = r.file("model.safetensors")?.to_path_buf();
    let config = std::fs::read_to_string(r.file("config.json")?)?;
    let device = Device::Cpu;
    let vision = ffai_argus::vision::load(&weights, &config, &device)?;

    let (w, h) = (512usize, 512usize);
    let mut px = vec![0u8; w * h * 3];
    for (i, v) in px.iter_mut().enumerate() {
        *v = (i % 251) as u8;
    }
    let pre = ffai_argus::preprocess::preprocess_rgb8(&px, w, h);
    let per = 3 * pre.tile * pre.tile;
    let tile = Tensor::from_vec(
        pre.pixel_values[..per].to_vec(),
        (1, 3, pre.tile, pre.tile),
        &device,
    )?;

    println!("GELU kernel selected at runtime: {}
", ffai_argus::siglip::gelu_kernel_name());

    ffai_argus::cost::start();
    let _ = vision.forward(&tile)?;
    let one = ffai_argus::cost::stop();

    // Reproducibility is the whole claim, so assert it rather than assume it.
    ffai_argus::cost::start();
    let _ = vision.forward(&tile)?;
    let again = ffai_argus::cost::stop();
    assert_eq!(one, again, "counters are not deterministic");

    let t = pre.tiles as f64;
    println!("ONE TILE (identical on a repeat run — deterministic)\n");
    println!("  matmul          {:>9.1} GFLOP in {} calls", one.matmul_flops as f64 / 1e9, one.matmul_calls);
    println!("  elementwise     {:>9.1} M visits in {} calls", one.elem_ops as f64 / 1e6, one.elem_calls);
    println!("  transcendental  {:>9.1} M scalar   (libm, ~75 M/s here)", one.transcendental as f64 / 1e6);
    println!("  transcendental  {:>9.1} M vectorised (candle kernel, ~2.7 G/s)", one.transcendental_vec as f64 / 1e6);
    println!("  memory moved    {:>9.1} MB", one.bytes_moved as f64 / 1e6);
    println!("  layout copies   {:>9} ({:.1} MB)", one.copies, one.copy_bytes as f64 / 1e6);

    println!("\nWHOLE IMAGE ({} tiles)\n", pre.tiles);
    println!("  matmul          {:>9.0} GFLOP", one.matmul_flops as f64 * t / 1e9);
    println!("  elementwise     {:>9.0} M visits", one.elem_ops as f64 * t / 1e6);
    println!("  scalar transc   {:>9.0} M", one.transcendental as f64 * t / 1e6);
    println!("  memory moved    {:>9.1} GB", one.bytes_moved as f64 * t / 1e9);
    println!("  layout copies   {:>9.0} ({:.2} GB)", one.copies as f64 * t, one.copy_bytes as f64 * t / 1e9);

    // Each term converted at this box's measured rate for THAT kind of work.
    // The point is not to predict a wall-clock number — it is to say which
    // term dominates, which is a question the stopwatch could not answer
    // reliably enough to act on.
    let mm = one.matmul_flops as f64 * t / 660e9;
    let el = one.elem_ops as f64 * t / 2.5e9;
    let ts = one.transcendental as f64 * t / 75e6;
    let tv = one.transcendental_vec as f64 * t / 2.7e9;
    let bw = one.bytes_moved as f64 * t / 10e9;
    println!("\nWHERE THE WORK IS (each at its own measured rate)\n");
    println!("  matmul                 ~{mm:>5.1} s   @ 660 GF/s");
    println!("  elementwise            ~{el:>5.1} s   @ 2.5 G visits/s");
    println!("  scalar transcendental  ~{ts:>5.1} s   @ 75 M/s");
    println!("  vectorised transc      ~{tv:>5.1} s   @ 2.7 G/s");
    println!("  ---");
    println!("  memory bandwidth floor ~{bw:>5.1} s   @ 10 GB/s for {:.1} GB", one.bytes_moved as f64 * t / 1e9);
    println!(
        "\n  The largest term is what the next win has to attack. Note that the\n  \
         bandwidth floor is not additive with the others — it is the same work\n  \
         seen from the memory side, and whichever of the two is larger binds."
    );
    Ok(())
}
