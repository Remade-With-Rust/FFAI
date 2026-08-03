//! Apply the SHIPPED reading order to boxes read from stdin.
//!
//! The ordering question — §8.36 put 13.7 of the 15.5-point gap against
//! PP-StructureV3 on sequence — is cheap to iterate on only if it can be asked
//! without running a detector or a recognizer. The corpus already carries the
//! answer key: every OmniDocBench sidecar has each region's polygon AND its
//! annotated reading order, so a testbed can feed perfect regions in and count
//! inversions out.
//!
//! The obvious shortcut is to reimplement `order_reading` in the probe's own
//! language. That was tried and it reproduced the shipped order on 2 pages out
//! of 12 — the recursion is faithful but line grouping, tie-breaking and
//! erosion are not, and a testbed that silently disagrees with the code it is
//! meant to be measuring produces confident numbers about nothing. So the probe
//! calls the real function instead, and this example is the seam.
//!
//! stdin:  one box per line, `x0 y0 x1 y1` (integers, whitespace separated)
//! stdout: the input indices, one per line, in reading order
//!
//! Each box becomes a one-box "line", which is what the region testbed wants:
//! regions ARE the units being ordered there. Grouping is `group_lines`'s job
//! and is deliberately not exercised here.

use ffai_carmenta::boxes::{order_reading, DetBox};
use std::io::Read;

fn main() {
    // Page width matters: `is_spanning` is `width >= page_w * SPAN_FRAC`, so
    // inferring `page_w` from the boxes' own extent makes every full-width
    // element span by construction. The region testbed has no image and is
    // happy with the inferred value; the LINE testbed must pass the real one.
    let arg_w: Option<usize> = std::env::args().nth(1).and_then(|a| a.parse().ok());
    let mut src = String::new();
    std::io::stdin().read_to_string(&mut src).expect("read stdin");

    let mut lines: Vec<Vec<DetBox>> = Vec::new();
    let mut page_w = 0usize;
    for row in src.lines() {
        // Parse as f64 and round. Region polygons in OmniDocBench are floats
        // ("83.326866"), and parsing straight to usize made every row fail,
        // silently, via filter_map — the probe emitted nothing at all and the
        // caller's permutation check read that as `order_reading` dropping
        // boxes. A parse that discards what it cannot read is the same trap as
        // a scorer treating a dead process as empty output.
        let v: Vec<usize> = row
            .split_whitespace()
            .filter_map(|t| t.parse::<f64>().ok())
            .map(|f| f.round().max(0.0) as usize)
            .collect();
        if v.len() < 4 {
            eprintln!("order_probe: unparsable row `{row}`");
            continue;
        }
        page_w = page_w.max(v[2]);
        // `score` carries the input index. Identifying boxes by geometry
        // instead looked fine and silently collided: two regions sharing a
        // top-left corner both resolved to the first one's index, so the
        // output was not a permutation and the testbed's assertion caught it.
        // f32 is exact on integers this small.
        let tag = lines.len() as f32;
        lines.push(vec![DetBox { x0: v[0], y0: v[1], x1: v[2], y1: v[3], score: tag }]);
    }
    let ordered = order_reading(lines, arg_w.unwrap_or(page_w).max(1));
    for l in &ordered {
        println!("{}", l[0].score as usize);
    }
}
