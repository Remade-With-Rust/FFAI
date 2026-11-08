//! The Ir instruments for the Argus veins campaign (`docs/plans/argus-ir-veins.md`).
//!
//! ```sh
//! # under WSL — see tools/argus_ir.sh, which sets the environment this needs
//! valgrind --tool=callgrind --collect-atstart=no --toggle-collect='*ir_region_tile*' \
//!     target/release/examples/ir_argus tile
//! ```
//!
//! Four modes, each ONE toggled region after an uncounted warm-up:
//!
//! | mode | region | covers |
//! |---|---|---|
//! | `tile` | `ir_region_tile` | one 512x512 tile through `SmolVlmVision::forward` |
//! | `prefill` | `ir_region_prefill` | `TextDecoder::forward_embeds` on the 1142-token oracle prompt |
//! | `decode` | `ir_region_decode` | 8 greedy steps at positions 1142.. (prefill is the warm-up) |
//! | `caption` | `ir_region_caption` | `describe_image_unsplit` end to end, 16 new tokens |
//!
//! Every region runs inside a ONE-thread rayon pool's `install`, so the toggled
//! function and every `par_chunks` beneath it execute on the same worker thread
//! and callgrind sees one thread doing the work. The engine's `AddInplace` /
//! `SwiGluInplace` see `current_thread_index() == Some(_)` and take their serial
//! arms, which is also what they do under six tile workers in production.
//!
//! Work-parity anchors are printed beside the count: an fnv64 of the region's
//! output bytes, the token ids, `ffai_core::cost` counters, and the counting
//! allocator's calls / bytes / zeroed bytes. A moved anchor means the
//! experiment changed, not the code (`instruction-counting` §3).

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{Device, Tensor};
use ffai_core::engine::VlmOptions;
use ffai_core::types::{ImageBuffer, PixelFormat};

// ---- counting allocator -----------------------------------------------------

struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static ZEROED: AtomicU64 = AtomicU64::new(0);

// SAFETY: a pass-through to `System`; the counters are relaxed atomics and
// never influence the allocation itself.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        ZEROED.fetch_add(l.size() as u64, Ordering::Relaxed);
        System.alloc_zeroed(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(n as u64, Ordering::Relaxed);
        System.realloc(p, l, n)
    }
}

#[global_allocator]
static A: Counting = Counting;

fn alloc_reset() {
    ALLOCS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    ZEROED.store(0, Ordering::Relaxed);
}

fn alloc_report() -> String {
    format!(
        "allocs={} bytes={} zeroed={}",
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
        ZEROED.load(Ordering::Relaxed)
    )
}

// ---- anchors -----------------------------------------------------------------

fn fnv_f32(t: &Tensor) -> u64 {
    let v = t.flatten_all().expect("flatten").to_vec1::<f32>().expect("vec");
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in v {
        for b in x.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn argmax(logits: &Tensor) -> u32 {
    let v = logits.flatten_all().expect("flatten").to_vec1::<f32>().expect("vec");
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as u32
}

// ---- fixtures ------------------------------------------------------------------

const IMG: usize = 512;

/// The oracle's synthetic image, byte for byte (`tests/engine_oracle.rs`).
fn reference_image() -> ImageBuffer {
    let mut data = vec![0u8; IMG * IMG * 3];
    let mut i = 0;
    for y in 0..IMG {
        let fy = y as f64 / IMG as f64;
        for x in 0..IMG {
            let fx = x as f64 / IMG as f64;
            let r = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * fx).sin();
            let g = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * fy + 1.0).sin();
            let b = 0.5 + 0.5 * (6.0 * std::f64::consts::PI * (fx + fy) + 2.0).sin();
            data[i] = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            data[i + 1] = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            data[i + 2] = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            i += 3;
        }
    }
    ImageBuffer { width: IMG as u32, height: IMG as u32, format: PixelFormat::Rgb8, data }
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model_files() -> (PathBuf, String) {
    let m = ffai_models::load_dir(&root().join("models"))
        .expect("models dir")
        .into_iter()
        .find(|m| m.name == ffai_argus::engine::MODEL)
        .expect("smolvlm manifest");
    let weights = m.local_path("model.safetensors").expect("weights in cache");
    let config = std::fs::read_to_string(m.local_path("config.json").expect("config")).expect("read");
    (weights, config)
}

fn oracle_embeds(dev: &Device) -> Tensor {
    let bytes = std::fs::read(root().join(".oracle/smolvlm-prompt/ours_inputs_embeds.f32"))
        .expect("oracle inputs_embeds");
    let v: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let seq = v.len() / 576;
    Tensor::from_vec(v, (1, seq, 576), dev).expect("embeds")
}

// ---- regions -------------------------------------------------------------------

#[inline(never)]
pub fn ir_region_tile(vision: &ffai_argus::vision::SmolVlmVision, px: &Tensor) -> Tensor {
    vision.forward(px).expect("tile forward")
}

#[inline(never)]
pub fn ir_region_prefill(dec: &mut ffai_argus::decode::TextDecoder, embeds: &Tensor) -> Tensor {
    dec.reset();
    dec.forward_embeds(embeds, 0).expect("prefill")
}

#[inline(never)]
pub fn ir_region_decode(
    dec: &mut ffai_argus::decode::TextDecoder,
    mut logits: Tensor,
    start: usize,
    steps: usize,
    dev: &Device,
) -> Vec<u32> {
    let _ = dev;
    let mut out = Vec::with_capacity(steps);
    for pos in start..start + steps {
        let next = ffai_argus::decode::argmax(&logits).expect("argmax");
        out.push(next);
        logits = dec.step(next, pos).expect("step");
    }
    out
}

#[inline(never)]
pub fn ir_region_pre(img: &ImageBuffer) -> ffai_argus::preprocess::Preprocessed {
    ffai_argus::preprocess::preprocess_rgb8(&img.data, img.width as usize, img.height as usize)
}

#[inline(never)]
pub fn ir_region_assemble(engine: &ffai_argus::SmolVlm, blocks: &[Tensor]) -> Tensor {
    engine
        .assemble_for_probe(4, 4, "What is written in this image?", blocks)
        .expect("assemble")
}

#[inline(never)]
pub fn ir_region_caption(engine: &ffai_argus::SmolVlm, img: &ImageBuffer, opts: &VlmOptions) -> String {
    engine.describe_image_unsplit(img, opts).expect("caption")
}

// ---- main ------------------------------------------------------------------------

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "tile".into());
    let dev = Device::Cpu;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("pool");
    // `tools/argus_ir.sh` exports FFAI_ARGUS_TILE_WORKERS=1 and
    // RAYON_NUM_THREADS=1, so the engine's tile loop stays on this thread.

    match mode.as_str() {
        "tile" => {
            let (weights, config) = model_files();
            let vision = ffai_argus::vision::load(&weights, &config, &dev).expect("vision");
            let img = reference_image();
            let pre = ffai_argus::preprocess::preprocess_rgb8(&img.data, IMG, IMG);
            let per = 3 * pre.tile * pre.tile;
            let px = Tensor::from_vec(pre.pixel_values[..per].to_vec(), (1, 3, pre.tile, pre.tile), &dev)
                .expect("px");
            let warm = pool.install(|| ir_region_tile(&vision, &px));
            std::hint::black_box(&warm);
            ffai_core::cost::start();
            alloc_reset();
            let out = pool.install(|| ir_region_tile(&vision, &px));
            let costs = ffai_core::cost::stop();
            println!(
                "tile fnv={:016x} dims={:?} {} | {}",
                fnv_f32(&out),
                out.dims(),
                alloc_report(),
                costs.report("tile")
            );
        }
        "prefill" | "decode" => {
            let (weights, config) = model_files();
            let mut dec = ffai_argus::decode::TextDecoder::load(&weights, &config, &dev).expect("decoder");
            let embeds = oracle_embeds(&dev);
            let seq = embeds.dim(1).expect("seq");
            let warm = pool.install(|| ir_region_prefill(&mut dec, &embeds));
            std::hint::black_box(&warm);
            if mode == "prefill" {
                ffai_core::cost::start();
                alloc_reset();
                let logits = pool.install(|| ir_region_prefill(&mut dec, &embeds));
                let costs = ffai_core::cost::stop();
                println!(
                    "prefill fnv={:016x} seq={seq} argmax={} {} | {}",
                    fnv_f32(&logits),
                    argmax(&logits),
                    alloc_report(),
                    costs.report("prefill")
                );
            } else {
                // Warm-up ALSO primes one decode step so the decode region's
                // first step is steady state (cache capacity already grown).
                let logits = pool.install(|| ir_region_prefill(&mut dec, &embeds));
                let _ = pool.install(|| ir_region_decode(&mut dec, logits.clone(), seq, 1, &dev));
                let logits = pool.install(|| ir_region_prefill(&mut dec, &embeds));
                ffai_core::cost::start();
                alloc_reset();
                let toks = pool.install(|| ir_region_decode(&mut dec, logits, seq, 8, &dev));
                let costs = ffai_core::cost::stop();
                println!("decode tokens={toks:?} {} | {}", alloc_report(), costs.report("decode"));
            }
        }
        "pre" => {
            // The SPLIT path: 512 -> 2048 -> 16 tiles + thumbnail. The unsplit
            // caption never cuts tiles, so it cannot see the tile plumbing.
            let img = reference_image();
            let warm = pool.install(|| ir_region_pre(&img));
            std::hint::black_box(&warm.tiles);
            alloc_reset();
            let pre = pool.install(|| ir_region_pre(&img));
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for x in &pre.pixel_values {
                for b in x.to_le_bytes() {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(0x0100_0000_01b3);
                }
            }
            println!("pre fnv={h:016x} tiles={} {}", pre.tiles, alloc_report());
        }
        "assemble" => {
            // The 17-tile prompt (1142 tokens) with the oracle's own tower output.
            let engine = ffai_argus::SmolVlm::with_manifest_dir(root().join("models"));
            engine.warm().expect("warm");
            let bytes = std::fs::read(root().join(".oracle/smolvlm-prompt/image_hidden.f32"))
                .expect("oracle image_hidden");
            let v: Vec<f32> =
                bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let all = Tensor::from_vec(v, (17, 64, 576), &dev).expect("hidden");
            let blocks: Vec<Tensor> = (0..17).map(|i| all.get(i).expect("block")).collect();
            let warm = pool.install(|| ir_region_assemble(&engine, &blocks));
            std::hint::black_box(&warm);
            alloc_reset();
            let out = pool.install(|| ir_region_assemble(&engine, &blocks));
            println!("assemble fnv={:016x} dims={:?} {}", fnv_f32(&out), out.dims(), alloc_report());
        }
        "caption" => {
            let engine = ffai_argus::SmolVlm::with_manifest_dir(root().join("models"));
            engine.warm().expect("warm");
            let img = reference_image();
            let opts = VlmOptions {
                prompt: Some("What is written in this image?".into()),
                max_new_tokens: Some(16),
                ..VlmOptions::default()
            };
            let warm = pool.install(|| ir_region_caption(&engine, &img, &opts));
            std::hint::black_box(&warm);
            ffai_core::cost::start();
            alloc_reset();
            let text = pool.install(|| ir_region_caption(&engine, &img, &opts));
            let costs = ffai_core::cost::stop();
            println!(
                "caption text={text:?} {} | {}",
                alloc_report(),
                costs.report("caption")
            );
        }
        other => {
            eprintln!("usage: ir_argus tile|prefill|decode|caption|pre|assemble (got {other})");
            std::process::exit(2);
        }
    }
}
