//! What does the H.264 decoder actually SAY when it stops early?
//!
//! ```text
//! cargo run --release -p ffai-media --example decode_diag -- clip.mp4
//! ```
//!
//! `sample_frames` swallows two error paths:
//!
//! ```ignore
//! if dec.send_packet(&packet).is_err() { continue; }               // discarded
//! while let Ok(rff_core::Frame::Video(v)) = dec.receive_frame() {} // Err ends the loop
//! ```
//!
//! So "decodes 49 of 164 frames" could be the decoder reporting an unsupported
//! feature and us ignoring it, or it could be the decoder failing mutely. Those
//! are different bugs with different owners, and the difference is one probe.
//! This is that probe: identical loop, every error surfaced.

use rff_codec::{CodecParams, CodecRegistry};
use rff_format::FormatRegistry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: decode_diag <clip.mp4>");

    let mut formats = FormatRegistry::new();
    rff_format_mp4::register(&mut formats);
    let mut codecs = CodecRegistry::new();
    rff_codec_h264::register(&mut codecs);

    let file = std::fs::File::open(&path)?;
    let mut demux = formats.open_demuxer("mp4", Box::new(std::io::BufReader::new(file)))?;
    let streams = demux.read_header()?;
    let (vidx, vstream) = streams
        .iter()
        .enumerate()
        .find(|(_, s)| s.media_type == rff_core::MediaType::Video)
        .expect("no video stream");

    let mut dec = codecs.find_decoder(vstream.codec_id)?;
    dec.configure(&CodecParams {
        codec_id: vstream.codec_id,
        width: vstream.width,
        height: vstream.height,
        pixel_format: vstream.pixel_format,
        sample_rate: 0,
        channels: 0,
        sample_format: None,
        extradata: vstream.extradata.clone(),
    })?;

    let (mut packets, mut frames, mut send_err, mut recv_err) = (0, 0, 0, 0);
    let mut first_send_err: Option<String> = None;
    let mut first_recv_err: Option<String> = None;

    loop {
        let packet = match demux.read_packet() {
            Ok(p) => p,
            Err(rff_core::Error::Eof) => break,
            Err(e) => {
                println!("demux error after {packets} packets: {e}");
                break;
            }
        };
        if packet.stream_index != vidx {
            continue;
        }
        packets += 1;
        if let Err(e) = dec.send_packet(&packet) {
            send_err += 1;
            first_send_err.get_or_insert_with(|| format!("packet {packets}: {e}"));
            continue;
        }
        loop {
            match dec.receive_frame() {
                Ok(rff_core::Frame::Video(_)) => frames += 1,
                Ok(other) => {
                    println!("non-video frame variant: {other:?}");
                    break;
                }
                Err(e) => {
                    let s = e.to_string();
                    // A decoder signals "nothing more right now" as an error
                    // too; that one is normal and must not be counted as a
                    // failure or the diagnosis inverts.
                    if s.contains("Again") || s.contains("again") || s.contains("Eof") {
                        break;
                    }
                    recv_err += 1;
                    first_recv_err.get_or_insert_with(|| format!("after {frames} frames: {s}"));
                    break;
                }
            }
        }
    }

    println!("packets fed   : {packets}");
    println!("frames out    : {frames}");
    println!("send_packet errors: {send_err}   first: {}", first_send_err.unwrap_or("-".into()));
    println!("receive errors    : {recv_err}   first: {}", first_recv_err.unwrap_or("-".into()));
    Ok(())
}
