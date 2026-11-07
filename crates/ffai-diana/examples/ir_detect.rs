//! Instrument A of the Ir campaign: ONE serial `detect`, framed for callgrind.
//!
//! ```sh
//! # under WSL — see tools/diana_ir.sh, which sets the environment this needs
//! valgrind --tool=callgrind --collect-atstart=no --toggle-collect='*ir_region_detect*' \
//!     target/release/examples/ir_detect corpora/cache/coco-val2017/000000000785.jpg n
//! ```
//!
//! The warm-up call runs OUTSIDE the toggled region, so the count covers one
//! steady-state image and not the lazy weight load. The region runs inside
//! `serial_scope`, so no pool is installed and every kernel takes its serial
//! arm — callgrind then sees one thread doing all the work, which is what makes
//! the count exact run to run.
//!
//! Work-parity anchors are printed beside the count: a change that moves any of
//! them changed the experiment, not the code (`instruction-counting` §3).

use ffai_core::engine::{DetectEngine, DetectOptions};
use ffai_core::types::{DetectOutput, ImageBuffer};
use ffai_diana::{engine::Yolo26, image::Geometry};

/// The toggled region. `inline(never)` so the symbol exists for callgrind.
#[inline(never)]
pub fn ir_region_detect(engine: &Yolo26, image: &ImageBuffer, opts: &DetectOptions) -> DetectOutput {
    ffai_diana::parallel::serial_scope(|| engine.detect(image, opts)).expect("detect")
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0100_0000_01b3);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: ir_detect <image> [tier] [square]")?;
    let tier = args.next().unwrap_or_else(|| "n".into());
    let geometry = if args.next().as_deref() == Some("square") { Geometry::Square } else { Geometry::Rect };

    let image = ffai_media::load_image(std::path::Path::new(&path))?;
    let engine = Yolo26::build(&tier, geometry, "models");
    let opts = DetectOptions::default();

    // Warm-up: loads the weights and fills every lazy cache. Not counted.
    let warm = ffai_diana::parallel::serial_scope(|| engine.detect(&image, &opts))?;
    std::hint::black_box(&warm);
    // Drain the deterministic counters so the region's reading is its own.
    let _ = ffai_diana::conv3x3::take_macs();
    let _ = ffai_diana::conv3x3::take_im2col_elems();
    let _ = ffai_diana::conv3x3::take_acts();
    let _ = ffai_diana::silu::take_silu_split();

    let out = ir_region_detect(&engine, &image, &opts);
    std::hint::black_box(&out);

    let macs = ffai_diana::conv3x3::take_macs();
    let im2col = ffai_diana::conv3x3::take_im2col_elems();
    let acts = ffai_diana::conv3x3::take_acts();
    let (s_el, s_calls, p_el, p_calls) = ffai_diana::silu::take_silu_split();

    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for d in &out.detections {
        for v in [d.x0, d.y0, d.x1, d.y1, d.confidence] {
            fnv(&mut h, &v.to_bits().to_le_bytes());
        }
        fnv(&mut h, &d.class_id.to_le_bytes());
    }
    println!(
        "anchors tier={tier} geom={geometry:?} img={}x{} dets={} fnv={h:016x} macs={macs} im2col_elems={im2col} acts={acts} silu_serial={s_el}/{s_calls} silu_par={p_el}/{p_calls} warm_eq={}",
        image.width,
        image.height,
        out.detections.len(),
        warm.detections == out.detections
    );
    Ok(())
}
