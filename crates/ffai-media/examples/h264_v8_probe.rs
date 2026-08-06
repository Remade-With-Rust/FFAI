//! Is the CABAC gap fixed in rusty_h264 0.8?
//!
//! We resolve 0.2.1, because `rff-codec-h264` 0.1.0 pins `rusty_h264 ^0.2`.
//! 0.8.0 is six minor versions newer and was published 2026-08-05. The CABAC
//! finding in docs/rff-h264-cabac-report.md was measured against 0.2.1 and is
//! worthless if 0.8 already fixes it — so this checks before anyone files it.
//!
//! Demuxes with rff-format-mp4 (which works) and feeds the elementary stream
//! straight to 0.8's own Decoder, bypassing the pinned adapter entirely.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: h264_v8_probe <clip.mp4>");
    let file = std::fs::File::open(&path)?;

    let mut formats = rff_format::FormatRegistry::new();
    rff_format_mp4::register(&mut formats);
    let mut demux =
        formats.open_demuxer("mp4", Box::new(std::io::BufReader::new(file)))?;
    let streams = demux.read_header()?;
    let (vidx, vstream) = streams
        .iter()
        .enumerate()
        .find(|(_, s)| s.media_type == rff_core::MediaType::Video)
        .expect("no video stream");

    let mut dec = rusty_h264::Decoder::new();
    // MP4 carries SPS/PPS in extradata (avcC), not inline.
    let ed = &vstream.extradata;
    println!("extradata: {} bytes", ed.len());
    if !ed.is_empty() {
        let _ = dec.decode(ed);
    }

    let t0 = std::time::Instant::now();
    let (mut packets, mut frames, mut errs) = (0usize, 0usize, 0usize);
    let (mut dims, mut nonzero, mut ylen) = ((0usize, 0usize), 0usize, 0usize);
    let mut first_err: Option<String> = None;
    loop {
        let pkt = match demux.read_packet() {
            Ok(p) => p,
            Err(rff_core::Error::Eof) => break,
            Err(e) => {
                println!("demux stopped: {e}");
                break;
            }
        };
        if pkt.stream_index != vidx {
            continue;
        }
        packets += 1;
        match dec.decode(&pkt.data) {
            Ok(Some(f)) => {
                // WORK PARITY: 164 EMPTY frames would look identical to 164
                // real ones on a count alone. Record dimensions and how much of
                // the luma plane actually carries signal.
                frames += 1;
                if frames == 1 {
                    dims = (f.width, f.height);
                    ylen = f.y.len();
                    nonzero = f.y.iter().filter(|&&v| v != 0 && v != 128).count();
                }
            }
            Ok(None) => {}
            Err(e) => {
                errs += 1;
                first_err.get_or_insert_with(|| format!("packet {packets}: {e}"));
            }
        }
    }
    let el = t0.elapsed().as_secs_f64();
    println!("packets {packets}  frames {frames}  errors {errs}");
    println!("decode {:.3} s = {:.2} ms/frame", el, el*1000.0/frames.max(1) as f64);
    println!("first error: {}", first_err.unwrap_or_else(|| "-".into()));
    let pct = 100.0 * nonzero as f64 / ylen.max(1) as f64;
    println!("frame 1: {}x{} luma {} bytes, {} carrying signal ({:.0} pct)",
        dims.0, dims.1, ylen, nonzero, pct);
    Ok(())
}
