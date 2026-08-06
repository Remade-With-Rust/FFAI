//! Which published rff demuxers actually deliver H.264 our decoder can eat?
//!
//! ```text
//! cargo run --release -p ffai-media --example container_probe -- clip.mkv mkv
//! ```
//!
//! MKV/AVI/TS store H.264 differently from MP4: MP4 carries AVCC (length-
//! prefixed NALs, SPS/PPS in `extradata`), while MKV and TS commonly carry
//! Annex-B (start-code prefixed, parameter sets inline). A demuxer that opens
//! the file is not the same as a pipeline that decodes it, and the difference
//! decoded to ZERO FRAMES SILENTLY the last time it went unchecked.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("clip");
    let kind = std::env::args().nth(2).expect("mp4|mkv|avi|ts");

    let mut formats = rff_format::FormatRegistry::new();
    rff_format_mp4::register(&mut formats);
    rff_format_mkv::register(&mut formats);   // registers as "matroska"
    rff_format_avi::register(&mut formats);
    rff_format_ts::register(&mut formats);    // registers as "mpegts"

    let file = std::fs::File::open(&path)?;
    let mut demux = match formats.open_demuxer(&kind, Box::new(std::io::BufReader::new(file))) {
        Ok(d) => d,
        Err(e) => {
            println!("{kind}: open_demuxer FAILED: {e}");
            return Ok(());
        }
    };
    let streams = match demux.read_header() {
        Ok(s) => s,
        Err(e) => {
            println!("{kind}: read_header FAILED: {e}");
            return Ok(());
        }
    };
    let Some((vidx, vs)) = streams
        .iter()
        .enumerate()
        .find(|(_, s)| s.media_type == rff_core::MediaType::Video)
    else {
        println!("{kind}: no video stream in {} streams", streams.len());
        return Ok(());
    };
    println!(
        "{kind}: codec {:?}  {}x{}  extradata {} bytes",
        vs.codec_id,
        vs.width,
        vs.height,
        vs.extradata.len()
    );

    let mut dec = rusty_h264::Decoder::new();
    if !vs.extradata.is_empty() {
        let _ = dec.decode(&vs.extradata);
    }
    let (mut pk, mut fr, mut er) = (0, 0, 0);
    let mut first = String::new();
    let mut annexb = 0;
    loop {
        let p = match demux.read_packet() {
            Ok(p) => p,
            Err(rff_core::Error::Eof) => break,
            Err(e) => {
                println!("{kind}: read_packet stopped after {pk}: {e}");
                break;
            }
        };
        if p.stream_index != vidx {
            continue;
        }
        pk += 1;
        // Annex-B starts 00 00 01 or 00 00 00 01; AVCC starts with a length.
        if p.data.len() > 4 && (p.data[..3] == [0, 0, 1] || p.data[..4] == [0, 0, 0, 1]) {
            annexb += 1;
        }
        match dec.decode(&p.data) {
            Ok(Some(_)) => fr += 1,
            Ok(None) => {}
            Err(e) => {
                er += 1;
                if first.is_empty() {
                    first = format!("packet {pk}: {e}");
                }
            }
        }
    }
    println!(
        "{kind}: packets {pk}  frames {fr}  errors {er}  annexB-looking {annexb}/{pk}"
    );
    if !first.is_empty() {
        println!("{kind}: first error -> {first}");
    }
    Ok(())
}
