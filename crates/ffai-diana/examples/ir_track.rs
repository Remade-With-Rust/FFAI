//! Instrument B of the Ir campaign: ByteTrack replayed over cached detections,
//! framed for callgrind. No GEMM anywhere — the opposite corpus shape to
//! `ir_detect`, so a change that trades tracker work for detector work shows
//! its sign here.
//!
//! ```sh
//! valgrind --tool=callgrind --collect-atstart=no --toggle-collect='*ir_region_track*' \
//!     target/release/examples/ir_track corpora/clips/mot17-02/diana_ungated.tsv
//! ```
//!
//! Input is the bench's `diana_ungated.tsv`: `frame<TAB>class<TAB>conf<TAB>x0
//! y0 x1 y1`. Parsing happens outside the region; only `update` is counted.

use ffai_diana::track::{ByteTrack, TrackerConfig};
use std::io::BufRead;

type Frame = (Vec<[f32; 4]>, Vec<f32>, Vec<u32>);

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0100_0000_01b3);
    }
}

/// The toggled region: every frame through the tracker, folded into one hash.
#[inline(never)]
pub fn ir_region_track(frames: &[Frame], cfg: TrackerConfig) -> (u64, u64, u64) {
    let mut tk = ByteTrack::new(cfg);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut reported = 0u64;
    let mut max_id = 0u64;
    for (bx, sc, cl) in frames {
        for t in tk.update(bx, sc, cl) {
            reported += 1;
            max_id = max_id.max(u64::from(t.id));
            fnv(&mut h, &t.id.to_le_bytes());
            for v in t.xyxy() {
                fnv(&mut h, &v.to_bits().to_le_bytes());
            }
        }
    }
    (reported, max_id, h)
}

fn main() -> std::io::Result<()> {
    let path = std::env::args().nth(1).expect("usage: ir_track <diana_ungated.tsv>");
    let file = std::io::BufReader::new(std::fs::File::open(&path)?);
    let mut classes: Vec<String> = Vec::new();
    let mut per_frame: std::collections::BTreeMap<u64, Frame> = Default::default();
    for line in file.lines() {
        let line = line?;
        let p: Vec<&str> = line.trim().split('\t').collect();
        if p.len() < 7 {
            continue;
        }
        let g = |i: usize| p[i].parse::<f32>().unwrap_or(0.0);
        let cls = match classes.iter().position(|c| c == p[1]) {
            Some(i) => i,
            None => {
                classes.push(p[1].to_string());
                classes.len() - 1
            }
        };
        let e = per_frame.entry(p[0].parse().unwrap_or(0)).or_default();
        e.0.push([g(3), g(4), g(5), g(6)]);
        e.1.push(g(2));
        e.2.push(cls as u32);
    }
    let last = per_frame.keys().copied().max().unwrap_or(0);
    let frames: Vec<Frame> = (1..=last).map(|f| per_frame.remove(&f).unwrap_or_default()).collect();
    let dets: usize = frames.iter().map(|f| f.0.len()).sum();

    let (reported, max_id, h) = ir_region_track(&frames, TrackerConfig::default());
    println!(
        "anchors frames={} dets={dets} reported={reported} max_id={max_id} fnv={h:016x}",
        frames.len()
    );
    Ok(())
}
