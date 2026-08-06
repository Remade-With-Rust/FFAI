//! Replay cached detections through ByteTrack at a given configuration.
//!
//! ```text
//! track_replay dets.txt                              # shipped defaults
//! track_replay dets.txt new_thresh=0.55 max_age=70   # override anything
//! ```
//!
//! A tracker sweep re-runs the TRACKER, not the detector — the detections are
//! identical for every setting. Running the model 48 times to sweep 16 settings
//! across 3 sequences would be ~28 minutes of recomputing the same boxes, and
//! the cache makes the sweep effectively free.
//!
//! It also removes a confound: every arm of the sweep sees byte-identical
//! detections, so a difference between arms cannot be the detector drifting.
//!
//! **Arguments are `key=value`, not positional.** They were positional, and by
//! the twelfth slot a sweep was passing `... "1" "0.8" "4294967295" "1"` — four
//! bare literals whose meaning lived only in the caller. One transposition
//! there yields a plausible wrong number rather than an error, which is the
//! most expensive kind of bug a measurement harness can have.
//!
//! Input is `frame,x0,y0,x1,y1,conf,class` per line; output is MOT-challenge
//! order, the same as `ffai detect --track`.
use ffai_diana::track::{ByteTrack, TrackerConfig};
use std::io::{BufRead, Write};

fn main() -> std::io::Result<()> {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: track_replay <dets> [key=value ...]");

    let mut cfg = TrackerConfig::default();
    for kv in a {
        let (k, v) =
            kv.split_once('=').unwrap_or_else(|| panic!("expected key=value, got {kv:?}"));
        let f = || v.parse::<f32>().unwrap_or_else(|_| panic!("bad f32 for {k}: {v:?}"));
        let u = || v.parse::<u32>().unwrap_or_else(|_| panic!("bad u32 for {k}: {v:?}"));
        match k {
            "track_thresh" => cfg.track_thresh = f(),
            "low_thresh" => cfg.low_thresh = f(),
            "new_thresh" => cfg.new_track_thresh = f(),
            "match_thresh" => cfg.match_thresh = f(),
            "max_age" => cfg.max_age = u(),
            "min_hits" => cfg.min_hits = u(),
            // `crowd_lo=1e9` collapses the crowding ramp, which is how a sweep
            // isolates a single threshold from the adaptive one.
            "crowd_lo" => cfg.crowd_lo = f(),
            "crowd_hi" => cfg.crowd_hi = f(),
            "crowded_thresh" => cfg.new_track_thresh_crowded = f(),
            "fuse_mode" => cfg.fuse_mode = u() as u8,
            "reinit_after" => cfg.reinit_after = u(),
            "defer" => cfg.deferred_unconfirmed = v == "1",
            "unconfirmed_thresh" => cfg.unconfirmed_thresh = f(),
            "pass2_lost" => cfg.pass2_lost = v == "1",
            _ => panic!("unknown key {k:?}"),
        }
    }

    let file = std::io::BufReader::new(std::fs::File::open(&path)?);
    let mut per_frame: std::collections::BTreeMap<u64, Vec<([f32; 4], f32, u32)>> =
        Default::default();
    for line in file.lines() {
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

    let mut tk = ByteTrack::new(cfg);
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
