//! AVCC (length-prefixed) H.264 to Annex-B (start-code prefixed).
//!
//! # Why this is here, and when to delete it
//!
//! H.264 travels in two shapes. **Annex-B** prefixes each NAL with `00 00 01`
//! and carries SPS/PPS inline; **AVCC** prefixes each NAL with a big-endian
//! length and keeps SPS/PPS in the container's `extradata` as an `avcC` box.
//!
//! `rff-format-mp4` normalises to Annex-B and leaves `extradata` empty.
//! `rff-format-mkv` does not — it passes AVCC through with a 41-byte `avcC`.
//! `rusty_h264` reads Annex-B. Measured 2026-08-06 on the same source encoded
//! twice:
//!
//! | container | packets | frames | errors |
//! |---|---:|---:|---:|
//! | MP4 | 164 | **164** | 0 |
//! | MKV | 164 | **0** | **0** |
//!
//! Zero frames and *zero errors* — the same silent short read that hid a broken
//! CABAC decoder for six minor versions. This module converts, so MKV decodes.
//!
//! **It is a workaround with an owner.** `docs/rff-gaps-for-ffai.md` item 3 asks
//! `rff-format-mkv` to normalise like `rff-format-mp4` already does; the day it
//! does, delete this file.

/// Parsed `avcC` configuration: the parameter sets plus the NAL length size.
pub struct AvcC {
    /// SPS/PPS already in Annex-B form, ready to feed the decoder first.
    pub parameter_sets: Vec<u8>,
    /// Bytes of length prefix on each NAL — 1, 2 or 4.
    pub nal_length_size: usize,
}

const START: [u8; 4] = [0, 0, 0, 1];

/// Parse an `avcC` box (ISO/IEC 14496-15 §5.2.4.1).
///
/// Returns `None` for anything that is not an `avcC` — an empty `extradata`, or
/// a container that already handed us Annex-B — so the caller can pass packets
/// through untouched rather than mangling them.
pub fn parse_avcc(extradata: &[u8]) -> Option<AvcC> {
    // configurationVersion(1) profile(1) compat(1) level(1) lengthSizeMinusOne(1)
    // numOfSPS(1) then the sets. Anything shorter cannot be an avcC.
    if extradata.len() < 7 || extradata[0] != 1 {
        return None;
    }
    let nal_length_size = (extradata[4] & 0x03) as usize + 1;
    let mut out = Vec::new();
    let mut i = 5usize;

    // SPS count is the low 5 bits; PPS count is a plain byte after them.
    let num_sps = (extradata[i] & 0x1F) as usize;
    i += 1;
    for _ in 0..num_sps {
        if i + 2 > extradata.len() {
            return None;
        }
        let n = u16::from_be_bytes([extradata[i], extradata[i + 1]]) as usize;
        i += 2;
        if i + n > extradata.len() {
            return None;
        }
        out.extend_from_slice(&START);
        out.extend_from_slice(&extradata[i..i + n]);
        i += n;
    }
    if i >= extradata.len() {
        return Some(AvcC {
            parameter_sets: out,
            nal_length_size,
        });
    }
    let num_pps = extradata[i] as usize;
    i += 1;
    for _ in 0..num_pps {
        if i + 2 > extradata.len() {
            return None;
        }
        let n = u16::from_be_bytes([extradata[i], extradata[i + 1]]) as usize;
        i += 2;
        if i + n > extradata.len() {
            return None;
        }
        out.extend_from_slice(&START);
        out.extend_from_slice(&extradata[i..i + n]);
        i += n;
    }
    Some(AvcC {
        parameter_sets: out,
        nal_length_size,
    })
}

/// Rewrite one AVCC packet as Annex-B.
///
/// Only call this when the container declared an `avcC` — the caller knows the
/// format from [`parse_avcc`] and must not re-derive it per packet.
///
/// # The heuristic that was here and was WRONG
///
/// The first version sniffed each packet and refused anything starting
/// `00 00 01`, reasoning that it was already Annex-B. But a **4-byte AVCC
/// length between 256 and 511 is literally `00 00 01 XX`** — so every NAL in
/// that size range was mistaken for a start code, passed through unconverted,
/// and the decoder failed at `packet 14: bitstream truncated`. A per-packet
/// guess cannot beat what the container already stated.
///
/// Returns `None` only when the data does not parse as a chain of
/// length-prefixed NALs, which means the caller's premise was wrong; passing
/// the packet through unchanged is then better than emitting garbage.
pub fn to_annexb(data: &[u8], nal_length_size: usize) -> Option<Vec<u8>> {
    if data.len() < nal_length_size {
        return None;
    }
    let mut out = Vec::with_capacity(data.len() + 16);
    let mut i = 0usize;
    while i + nal_length_size <= data.len() {
        let mut n = 0usize;
        for b in &data[i..i + nal_length_size] {
            n = (n << 8) | *b as usize;
        }
        i += nal_length_size;
        // A length that overruns the packet means this was never AVCC.
        if n == 0 || i + n > data.len() {
            return None;
        }
        out.extend_from_slice(&START);
        out.extend_from_slice(&data[i..i + n]);
        i += n;
    }
    if i != data.len() {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal avcC: one 3-byte SPS, one 2-byte PPS, 4-byte NAL lengths.
    #[test]
    fn parses_an_avcc_into_annexb_parameter_sets() {
        let ed = vec![
            1, 0x42, 0xC0, 0x1E, 0xFF, // version, profile, compat, level, lengthSize=4
            0xE1, 0, 3, 0xAA, 0xBB, 0xCC, // 1 SPS, len 3
            1, 0, 2, 0xDD, 0xEE, // 1 PPS, len 2
        ];
        let c = parse_avcc(&ed).expect("avcC");
        assert_eq!(c.nal_length_size, 4);
        assert_eq!(
            c.parameter_sets,
            vec![0, 0, 0, 1, 0xAA, 0xBB, 0xCC, 0, 0, 0, 1, 0xDD, 0xEE]
        );
    }

    #[test]
    fn converts_length_prefixed_nals() {
        // two NALs: 2 bytes then 3 bytes, 4-byte lengths
        let pkt = vec![0, 0, 0, 2, 0x11, 0x22, 0, 0, 0, 3, 0x33, 0x44, 0x55];
        let got = to_annexb(&pkt, 4).expect("converted");
        assert_eq!(
            got,
            vec![0, 0, 0, 1, 0x11, 0x22, 0, 0, 0, 1, 0x33, 0x44, 0x55]
        );
    }

    /// A 4-byte length of 256-511 bytes IS `00 00 01 XX`. The first version of
    /// this filter read that as a start code, refused to convert, and the
    /// decoder died at "packet 14: bitstream truncated" on a real MKV.
    #[test]
    fn a_length_that_looks_like_a_start_code_still_converts() {
        let n = 300usize;
        let mut pkt = vec![0, 0, 1, (n & 0xFF) as u8]; // == 0x0000012C == 300
        pkt.extend(std::iter::repeat(0xAB).take(n));
        let got = to_annexb(&pkt, 4).expect("must convert, not refuse");
        assert_eq!(&got[..4], &[0, 0, 0, 1]);
        assert_eq!(got.len(), 4 + n);
    }

    /// A length that overruns the packet is not AVCC; refuse rather than guess.
    #[test]
    fn refuses_a_length_that_overruns() {
        let pkt = vec![0, 0, 0, 99, 0x11, 0x22];
        assert!(to_annexb(&pkt, 4).is_none());
    }

    #[test]
    fn empty_extradata_is_not_an_avcc() {
        assert!(parse_avcc(&[]).is_none());
        assert!(parse_avcc(&[0, 0, 0, 1, 0x67]).is_none());
    }
}
