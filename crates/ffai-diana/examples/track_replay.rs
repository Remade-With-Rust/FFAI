//! Replay cached detections through ByteTrack at given thresholds.
//!
//! ```text
//! track_replay dets.txt 0.5 0.6 > tracks.txt
//! ```
//!
//! A threshold sweep re-runs the TRACKER, not the detector — the detections are
//! identical for every setting. Running the model 48 times to sweep 16 settings
//! across 3 sequences would be ~28 minutes of recomputing the same boxes, and
//! the cache makes the sweep effectively free.
//!
//! It also removes a confound: every arm of the sweep sees byte-identical
//! detections, so a difference between arms cannot be the detector drifting.
//!
//! Input is `frame,x0,y0,x1,y1,conf,class` per line; output is MOT-challenge
//! order, the same as `ffai detect --track`.
use ffai_diana::track::{ByteTrack, TrackerConfig};
use std::io::{BufRead, Write};

fn main() -> std::io::Result<()> {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: track_replay <dets> <track_thresh> <new_thresh>");
    let track_thresh: f32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let new_track_thresh: f32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(0.6);

    let f = std::io::BufReader::new(std::fs::File::open(&path)?);
    let mut per_frame: std::collections::BTreeMap<u64, Vec<([f32; 4], f32, u32)>> =
        Default::default();
    for line in f.lines() {
        let line = line?;
        let p: Vec<&str> = line.trim().split(',').collect();
        if p.len() < 7 {
            continue;
        }
        let g = |i: usize| p[i].parse::<f32>().unwrap_or(0.0);
        per_frame.entry(p[0].parse().unwrap_or(0)).or_default().push((
            [g(1), g(2), g(3), g(4)],
            g(5),
            p[6].parse().unwrap_or(0),
        ));
    }

    // Optional 4th/5th args sweep the crowding ramp; default = adaptive off
    // (lo == hi collapses the ramp to the sparse value).
    let crowd_lo: f32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(1e9);
    let crowd_hi: f32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(1e9);
    let crowded_thresh: f32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(0.6);
    // 6th/7th sweep track SURVIVAL, which is the pair the IDF1 gap points at.
    //
    // Against Ultralytics on the same weights and frames we sit at MOTA 18.58
    // vs 19.24 (level) but IDF1 30.48 vs 34.11 (behind) with 398 ID switches
    // against their 808. Fewer switches AND worse IDF1 is not "more stable" —
    // it is FEWER and SHORTER trajectories, because a track we never report
    // cannot switch. `min_hits` withholds a track for its first N frames and
    // `max_age` decides how long a lost one survives; both cost IDF1 directly
    // as identity false-negatives while leaving MOTA nearly untouched.
    let min_hits: u32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let max_age: u32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(30);
    let mut tk = ByteTrack::new(TrackerConfig {
        track_thresh,
        new_track_thresh,
        crowd_lo,
        crowd_hi,
        new_track_thresh_crowded: crowded_thresh,
        min_hits,
        max_age,
        ..Default::default()
    });
    let out = std::io::stdout();
    let mut w = std::io::BufWriter::new(out.lock());
    let last = per_frame.keys().copied().max().unwrap_or(0);
    for frame in 1..=last {
        let empty = Vec::new();
        let d = per_frame.get(&frame).unwrap_or(&empty);
        let bx: Vec<[f32; 4]> = d.iter().map(|x| x.0).collect();
        let sc: Vec<f32> = d.iter().map(|x| x.1).collect();
        let cl: Vec<u32> = d.iter().map(|x| x.2).collect();
        for t in tk.update(&bx, &sc, &cl) {
            let b = t.xyxy();
            writeln!(
                w,
                "{},{},{:.1},{:.1},{:.1},{:.1},{:.3},-1,-1,-1",
                frame, t.id, b[0], b[1], b[2] - b[0], b[3] - b[1], t.score
            )?;
        }
    }
    Ok(())
}
