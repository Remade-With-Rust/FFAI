//! Step 7's gate: `describe_video` — sampling, windowing, and the timed track.
//!
//! # What is actually gated here, and what is not
//!
//! There is no reference to diff against for video. `SmolVLM`-256M-Instruct is
//! an **image** model with no temporal training, so its captions describe
//! frames rather than motion, and no public row exists for it on Video-MME or
//! MVBench. Inventing a quality number for it here would be exactly the
//! self-favouring scorer the whole campaign refuses.
//!
//! So this gates the part that has right answers: **the track**. Segment
//! count, ordering, contiguity, coverage, and the failure modes. Those are
//! structural properties a plumbing mistake breaks and a caption cannot hide —
//! and a wrong timeline is the video defect that ships silently, because every
//! caption in it still reads plausibly.
//!
//! Run `--release`: each frame is a vision-tower pass.

use ffai_core::engine::{VlmEngine, VlmOptions};
use ffai_core::types::{ImageBuffer, PixelFormat, VideoFrame};

use std::path::{Path, PathBuf};

fn manifests() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

/// Small, distinct, non-blank frames. Small because every frame is a tower
/// pass; distinct because identical frames would let a windowing bug that
/// drops or duplicates one go unnoticed.
fn frames(n: usize, step: f64) -> Vec<VideoFrame> {
    (0..n)
        .map(|i| {
            let (w, h) = (96usize, 64usize);
            let mut data = vec![0u8; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let p = (y * w + x) * 3;
                    data[p] = ((x * 4 + i * 37) % 256) as u8;
                    data[p + 1] = ((y * 4 + i * 61) % 256) as u8;
                    data[p + 2] = ((x + y + i * 23) % 256) as u8;
                }
            }
            VideoFrame {
                image: ImageBuffer {
                    width: w as u32,
                    height: h as u32,
                    format: PixelFormat::Rgb8,
                    data,
                },
                timestamp: i as f64 * step,
            }
        })
        .collect()
}

fn engine() -> ffai_argus::SmolVlm {
    ffai_argus::SmolVlm::with_manifest_dir(manifests())
}

fn opts(window: usize) -> VlmOptions {
    VlmOptions {
        prompt: Some("What is happening?".into()),
        // One token: this gate is about the TRACK, and generating sixty tokens
        // per window to assert a timestamp is time spent proving nothing.
        max_new_tokens: Some(1),
        frames_per_window: Some(window),
        ..VlmOptions::default()
    }
}

#[test]
fn windows_tile_the_timeline_without_gaps_or_overlap() {
    let e = engine();
    let f = frames(7, 0.5);
    let track = match e.describe_video(&f, &opts(3)) {
        Ok(t) => t,
        Err(err) => {
            eprintln!("SKIP: engine unavailable: {err}");
            return;
        }
    };

    // 7 frames in windows of 3 = 3 captions (3 + 3 + 1). The remainder window
    // must produce a segment too; dropping it silently loses the tail of every
    // video whose length is not a multiple of the window.
    assert_eq!(track.len(), 3, "7 frames / window 3 should give 3 segments");

    for (i, s) in track.iter().enumerate() {
        eprintln!("  [{:>5.2} - {:>5.2}] {:?}", s.start, s.end, s.value);
        assert!(s.end > s.start, "segment {i} has non-positive duration");
        assert!(!s.value.is_empty(), "segment {i} has no caption");
    }
    // Each window starts where the previous ended: a track with holes in it is
    // a track a player silently shows nothing during.
    for w in track.windows(2) {
        assert!(
            (w[1].start - w[0].end).abs() < 1e-9,
            "gap or overlap between {:?} and {:?}",
            w[0].end,
            w[1].start
        );
    }
    assert!(
        (track[0].start - 0.0).abs() < 1e-9,
        "the track must start at the first frame"
    );
    // Window starts land on the first frame of each window.
    assert!((track[1].start - 1.5).abs() < 1e-9, "window 2 starts at frame 3");
    assert!((track[2].start - 3.0).abs() < 1e-9, "window 3 starts at frame 6");
}

#[test]
fn a_window_of_one_degenerates_to_per_frame_captioning() {
    let e = engine();
    let f = frames(3, 1.0);
    let track = match e.describe_video(&f, &opts(1)) {
        Ok(t) => t,
        Err(err) => {
            eprintln!("SKIP: engine unavailable: {err}");
            return;
        }
    };
    assert_eq!(track.len(), 3, "window 1 is one caption per frame");
    for (i, s) in track.iter().enumerate() {
        assert!(
            (s.start - i as f64).abs() < 1e-9,
            "segment {i} starts at {}, expected {i}",
            s.start
        );
    }
}

#[test]
fn an_empty_clip_is_an_empty_track_not_an_error() {
    // A video that decoded no frames is the CALLER's problem to report — the
    // CLI says so by name. An engine that errored here would make "no frames"
    // and "the model failed" the same event.
    let track = engine()
        .describe_video(&[], &opts(4))
        .expect("empty input must not error");
    assert!(track.is_empty());
}

/// Too many frames in one window must be REFUSED, cheaply, and by name.
///
/// The text tower holds 8192 positions and an unsplit frame is 64 image
/// tokens, so ~128 frames is the ceiling. The check runs on GEOMETRY before
/// any tower pass, which is why this test finishes in milliseconds instead of
/// running 200 vision passes to discover the prompt was never going to fit.
#[test]
fn an_oversized_window_is_refused_before_any_work() {
    let e = engine();
    let f = frames(200, 0.1);
    let t = std::time::Instant::now();
    let err = match e.describe_video(&f, &opts(200)) {
        Err(err) => err,
        Ok(_) => panic!("200 frames in one window must not be accepted"),
    };
    let elapsed = t.elapsed();
    let msg = format!("{err}");
    if msg.contains("manifest") || msg.contains("tokenizer") {
        eprintln!("SKIP: engine unavailable: {msg}");
        return;
    }
    eprintln!("  refused in {elapsed:?}: {msg}");
    assert!(
        msg.contains("--window") || msg.contains("frames_per_window"),
        "the error must name the knob that fixes it: {msg}"
    );
    // The whole point of checking geometry first. 200 tower passes would be
    // minutes; this is the difference between a usable error and a hang.
    assert!(
        elapsed.as_secs() < 20,
        "refusal took {elapsed:?} — the budget check ran AFTER the tower"
    );
}

/// The unsplit video tile is the split path's global thumbnail, exactly.
///
/// Video turns tile splitting off so a window fits: 1 tile at 64 tokens rather
/// than 17 at 1088. The claim that makes that safe is that the single tile is
/// the same tensor the split path already produces as its thumbnail — which is
/// gated bit-exactly against the reference — and not a second, subtly
/// different route to a small image.
#[test]
fn the_unsplit_tile_is_exactly_the_split_paths_thumbnail() {
    use ffai_argus::preprocess::preprocess_rgb8_opts;

    for (w, h) in [(512usize, 512usize), (800, 600), (300, 700)] {
        let px: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
        let split = preprocess_rgb8_opts(&px, w, h, true);
        let flat = preprocess_rgb8_opts(&px, w, h, false);

        assert_eq!(flat.tiles, 1, "{w}x{h}: unsplit must be a single tile");
        assert_eq!((flat.rows, flat.cols), (0, 0));

        let per = 3 * split.tile * split.tile;
        let thumbnail = &split.pixel_values[(split.tiles - 1) * per..];
        assert_eq!(
            flat.pixel_values.as_slice(),
            thumbnail,
            "{w}x{h}: the unsplit tile differs from the split path's thumbnail, so \
             the video path does NOT inherit the thumbnail's oracle gate"
        );
    }
}

/// The geometry helper must agree with what preprocessing actually produces.
///
/// It exists to price a prompt without doing the work, which is only safe
/// while the two agree. A cheap predictor that drifts from the real thing is
/// worse than no predictor: it would refuse valid prompts, or accept ones that
/// then overflow after the tower has run.
#[test]
fn the_cheap_tile_count_matches_the_real_one() {
    use ffai_argus::preprocess::{preprocess_rgb8_opts, tile_geometry};

    for (w, h) in [(512usize, 512usize), (800, 600), (300, 700), (1920, 1080), (64, 64)] {
        for split in [true, false] {
            let px = vec![7u8; w * h * 3];
            let real = preprocess_rgb8_opts(&px, w, h, split);
            let (tiles, rows, cols) = tile_geometry(w, h, split);
            assert_eq!(
                (tiles, rows, cols),
                (real.tiles, real.rows, real.cols),
                "{w}x{h} split={split}: predicted geometry disagrees with the real one"
            );
        }
    }
}
