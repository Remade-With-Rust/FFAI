//! Detection scoring: a COCO-style mAP proxy, beside CER/WER as `der.rs` is
//! beside them for diarization.
//!
//! **Proxy, stated plainly.** This scorer implements the standard matching
//! rule (per-image greedy assignment of confidence-ranked detections to the
//! highest-IoU unmatched ground-truth box at each IoU threshold 0.50:0.95,
//! maxDets=100, 101-point interpolated AP averaged over classes with ground
//! truth) but none of pycocotools' area-range breakdowns or crowd-region
//! ignore logic — the M-D0 corpus deliberately excludes crowd annotations so
//! the ignore rule has nothing to fire on. The scorer is cross-validated
//! against pycocotools on the pinned corpus before any number it produces
//! goes on the board (`tools/diana_validate_scorer.py`); the validation
//! result rides in the ledger notes of the run that used it.
//!
//! Wire formats, shared with the reference adapters and the corpus tool:
//!
//! - ground truth (one JSON file per image):
//!   `{"width": W, "height": H, "objects": [[x0,y0,x1,y1,cls], ...]}`
//! - hypothesis (the adapter's `text` field):
//!   `[[x0,y0,x1,y1,cls,conf], ...]`
//!
//! Boxes are xyxy in original-image pixels. `cls` is the contiguous 0-79
//! COCO index (category ids sorted ascending — the Ultralytics convention),
//! recorded by the corpus tool in `classes.json`. Malformed input is an
//! error, never a silent zero.

use ffai_core::error::{Error, Result};
use std::collections::BTreeMap;

/// IoU thresholds 0.50:0.05:0.95, the COCO ladder.
pub const IOU_THRESHOLDS: [f64; 10] = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95];

/// Detections kept per image, ranked by confidence (COCO maxDets).
///
/// **A recorded difference from pycocotools, not an equivalence.** This
/// scorer truncates per IMAGE across all classes; pycocotools truncates per
/// (image, category) pair inside `evaluateImg`. The two agree on the M-D0
/// board because the reference adapters are themselves pinned to
/// `--max-dets 100`, so an image never carries more than 100 detections
/// total and neither truncation binds. Raise the adapters' cap above this
/// value and the two can legitimately diverge — re-run
/// `tools/diana_validate_scorer.py` if that ever changes.
pub const MAX_DETS: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub bbox: [f64; 4], // x0, y0, x1, y1
    pub class: i64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundTruth {
    pub bbox: [f64; 4],
    pub class: i64,
}

/// Parse a hypothesis payload: `[[x0,y0,x1,y1,cls,conf], ...]`.
pub fn parse_detections(text: &str) -> Result<Vec<Detection>> {
    let value: serde_json::Value = serde_json::from_str(text.trim())
        .map_err(|e| Error::Other(format!("detection payload is not JSON: {e}")))?;
    let rows = value
        .as_array()
        .ok_or_else(|| Error::Other("detection payload is not a JSON array".into()))?;
    rows.iter()
        .map(|row| {
            let nums = numeric_row(row, 6, "detection")?;
            Ok(Detection {
                bbox: [nums[0], nums[1], nums[2], nums[3]],
                class: nums[4] as i64,
                confidence: nums[5],
            })
        })
        .collect()
}

/// Parse a ground-truth file: `{"objects": [[x0,y0,x1,y1,cls], ...], ...}`.
pub fn parse_ground_truth(text: &str) -> Result<Vec<GroundTruth>> {
    let value: serde_json::Value = serde_json::from_str(text.trim())
        .map_err(|e| Error::Other(format!("ground truth is not JSON: {e}")))?;
    let rows = value
        .get("objects")
        .and_then(|o| o.as_array())
        .ok_or_else(|| Error::Other("ground truth has no `objects` array".into()))?;
    rows.iter()
        .map(|row| {
            let nums = numeric_row(row, 5, "ground-truth object")?;
            Ok(GroundTruth {
                bbox: [nums[0], nums[1], nums[2], nums[3]],
                class: nums[4] as i64,
            })
        })
        .collect()
}

fn numeric_row(row: &serde_json::Value, want: usize, what: &str) -> Result<Vec<f64>> {
    let arr = row
        .as_array()
        .ok_or_else(|| Error::Other(format!("{what} row is not an array")))?;
    if arr.len() != want {
        return Err(Error::Other(format!(
            "{what} row has {} fields, expected {want}",
            arr.len()
        )));
    }
    arr.iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| Error::Other(format!("{what} row has a non-numeric field")))
        })
        .collect()
}

fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let ix = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let iy = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = ix * iy;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// One ranked detection's fate: its confidence and, per IoU threshold, whether
/// it matched a ground-truth box (bit i of `tp_mask` = threshold `IOU_THRESHOLDS[i]`).
#[derive(Debug, Clone, Copy)]
struct Ranked {
    confidence: f64,
    tp_mask: u16,
}

/// Accumulates matches across a corpus; ask it for mAP at the end.
#[derive(Debug, Default)]
pub struct MapAccumulator {
    per_class: BTreeMap<i64, Vec<Ranked>>,
    gt_count: BTreeMap<i64, usize>,
}

impl MapAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Score one image's detections against its ground truth.
    pub fn add_image(&mut self, truths: &[GroundTruth], detections: &[Detection]) {
        // maxDets is per image across ALL classes, applied on confidence rank.
        let mut dets: Vec<&Detection> = detections.iter().collect();
        dets.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        dets.truncate(MAX_DETS);

        for truth in truths {
            *self.gt_count.entry(truth.class).or_insert(0) += 1;
        }

        let mut classes: Vec<i64> = dets.iter().map(|d| d.class).collect();
        classes.sort_unstable();
        classes.dedup();
        for class in classes {
            let class_dets: Vec<&&Detection> = dets.iter().filter(|d| d.class == class).collect();
            let class_gts: Vec<&GroundTruth> = truths.iter().filter(|g| g.class == class).collect();
            // Greedy per threshold: each detection, in confidence order, takes
            // the unmatched ground truth with the highest IoU >= threshold.
            let mut masks = vec![0u16; class_dets.len()];
            for (ti, thresh) in IOU_THRESHOLDS.iter().enumerate() {
                let mut taken = vec![false; class_gts.len()];
                for (di, det) in class_dets.iter().enumerate() {
                    let mut best: Option<(usize, f64)> = None;
                    for (gi, gt) in class_gts.iter().enumerate() {
                        if taken[gi] {
                            continue;
                        }
                        let overlap = iou(&det.bbox, &gt.bbox);
                        if overlap >= *thresh && best.is_none_or(|(_, prev)| overlap > prev) {
                            best = Some((gi, overlap));
                        }
                    }
                    if let Some((gi, _)) = best {
                        taken[gi] = true;
                        masks[di] |= 1 << ti;
                    }
                }
            }
            let bucket = self.per_class.entry(class).or_default();
            for (det, mask) in class_dets.iter().zip(masks) {
                bucket.push(Ranked {
                    confidence: det.confidence,
                    tp_mask: mask,
                });
            }
        }
    }

    /// AP at one IoU threshold index, averaged over classes with ground truth.
    fn map_at(&self, ti: usize) -> Option<f64> {
        let mut aps = Vec::new();
        for (class, &n_gt) in &self.gt_count {
            if n_gt == 0 {
                continue;
            }
            let mut ranked: Vec<Ranked> = self.per_class.get(class).cloned().unwrap_or_default();
            ranked.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
            // Precision/recall curve, then 101-point interpolated AP.
            let (mut tp, mut fp) = (0usize, 0usize);
            let mut curve: Vec<(f64, f64)> = Vec::with_capacity(ranked.len()); // (recall, precision)
            for r in &ranked {
                if r.tp_mask & (1 << ti) != 0 {
                    tp += 1;
                } else {
                    fp += 1;
                }
                curve.push((tp as f64 / n_gt as f64, tp as f64 / (tp + fp) as f64));
            }
            // Precision envelope: max precision at recall >= r.
            let mut ap = 0.0;
            for i in 0..=100 {
                let r = i as f64 / 100.0;
                let p = curve
                    .iter()
                    .filter(|(rec, _)| *rec >= r)
                    .map(|(_, prec)| *prec)
                    .fold(0.0f64, f64::max);
                ap += p;
            }
            aps.push(ap / 101.0);
        }
        if aps.is_empty() {
            None
        } else {
            Some(aps.iter().sum::<f64>() / aps.len() as f64)
        }
    }

    /// mAP@0.5 — the headline the quality gate reads (as `1 - mAP50`).
    pub fn map50(&self) -> Option<f64> {
        self.map_at(0)
    }

    /// mAP@0.5:0.95 — the ten-threshold mean, recorded beside `map50`.
    pub fn map5095(&self) -> Option<f64> {
        let aps: Vec<f64> = (0..IOU_THRESHOLDS.len())
            .filter_map(|ti| self.map_at(ti))
            .collect();
        if aps.is_empty() {
            None
        } else {
            Some(aps.iter().sum::<f64>() / aps.len() as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(bbox: [f64; 4], class: i64, confidence: f64) -> Detection {
        Detection {
            bbox,
            class,
            confidence,
        }
    }
    fn gt(bbox: [f64; 4], class: i64) -> GroundTruth {
        GroundTruth { bbox, class }
    }

    #[test]
    fn perfect_detection_scores_one() {
        let mut acc = MapAccumulator::new();
        acc.add_image(
            &[
                gt([0.0, 0.0, 10.0, 10.0], 0),
                gt([20.0, 20.0, 30.0, 30.0], 3),
            ],
            &[
                det([0.0, 0.0, 10.0, 10.0], 0, 0.9),
                det([20.0, 20.0, 30.0, 30.0], 3, 0.8),
            ],
        );
        assert_eq!(acc.map50(), Some(1.0));
        assert_eq!(acc.map5095(), Some(1.0));
    }

    #[test]
    fn higher_confidence_false_positive_halves_ap() {
        // FP ranked above the only TP: precision at full recall is 1/2, and
        // the 101-point envelope is flat at 0.5.
        let mut acc = MapAccumulator::new();
        acc.add_image(
            &[gt([0.0, 0.0, 10.0, 10.0], 0)],
            &[
                det([50.0, 50.0, 60.0, 60.0], 0, 0.95),
                det([0.0, 0.0, 10.0, 10.0], 0, 0.90),
            ],
        );
        let ap = acc.map50().unwrap();
        assert!((ap - 0.5).abs() < 1e-9, "expected 0.5, got {ap}");
    }

    #[test]
    fn wrong_class_is_both_fp_and_fn() {
        let mut acc = MapAccumulator::new();
        acc.add_image(
            &[gt([0.0, 0.0, 10.0, 10.0], 0)],
            &[det([0.0, 0.0, 10.0, 10.0], 1, 0.9)],
        );
        assert_eq!(acc.map50(), Some(0.0));
    }

    #[test]
    fn iou_exactly_at_threshold_counts() {
        // Boxes overlapping at exactly IoU 0.5: [0,0,10,10] vs [0,5,10,15]
        // has inter 50, union 150 -> 1/3; use half-overlap in one axis
        // shifted to give exactly 0.5: [0,0,10,10] vs [0,0,10,5] -> inter 50,
        // union 100 -> 0.5.
        let mut acc = MapAccumulator::new();
        acc.add_image(
            &[gt([0.0, 0.0, 10.0, 5.0], 0)],
            &[det([0.0, 0.0, 10.0, 10.0], 0, 0.9)],
        );
        assert_eq!(acc.map50(), Some(1.0));
        // ...but it fails every higher threshold.
        assert!(acc.map5095().unwrap() < 0.2);
    }

    #[test]
    fn each_ground_truth_matches_at_most_once() {
        // Two detections on one GT: the higher-confidence one is the TP, the
        // duplicate is an FP.
        let mut acc = MapAccumulator::new();
        acc.add_image(
            &[gt([0.0, 0.0, 10.0, 10.0], 0)],
            &[
                det([0.0, 0.0, 10.0, 10.0], 0, 0.9),
                det([0.1, 0.0, 10.0, 10.0], 0, 0.8),
            ],
        );
        // TP first, then FP: envelope is 1.0 up to recall 1.0.
        assert_eq!(acc.map50(), Some(1.0));
    }

    #[test]
    fn parses_wire_formats() {
        let dets = parse_detections("[[0, 1.5, 10, 12, 3, 0.87]]").unwrap();
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class, 3);
        let gts = parse_ground_truth(r#"{"width": 640, "height": 480, "objects": [[0,0,5,5,7]]}"#)
            .unwrap();
        assert_eq!(gts.len(), 1);
        assert_eq!(gts[0].class, 7);
        assert!(parse_detections("[[1,2,3]]").is_err());
        assert!(parse_ground_truth("{}").is_err());
        assert!(parse_detections("not json").is_err());
    }

    #[test]
    fn empty_hypothesis_on_nonempty_truth_scores_zero() {
        let mut acc = MapAccumulator::new();
        acc.add_image(&[gt([0.0, 0.0, 10.0, 10.0], 0)], &[]);
        assert_eq!(acc.map50(), Some(0.0));
    }
}
