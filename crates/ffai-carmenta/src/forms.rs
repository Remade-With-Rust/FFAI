//! Label → value pairs from a read form page (docs/plans/commercial-gaps.md,
//! gap 4).
//!
//! `OcrOutput` is lines with boxes. A form reader wants FIELDS: "Name of
//! Creditor" → "Merrill Lynch". The layout rule printed forms follow, and the
//! one a caller's born-digital reader already relies on, is:
//!
//! * a **label** is a line that ends in `:`;
//! * its **value** is the nearest unlabelled line to its RIGHT on the same
//!   baseline, or, failing that, the nearest one directly BELOW it;
//! * a single line reading `Label: value` (the detector drew one box around
//!   both) is split at its first `:`.
//!
//! Each line is used as a value at most once, closest label first, so two
//! labels never claim the same value. A label with no value is returned with
//! an empty value rather than dropped: "this field is blank" is information.

use ffai_core::types::{BoundingBox, OcrOutput};

/// One field.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    /// The label, without its trailing `:`.
    pub label: String,
    /// The value, or empty when none was found.
    pub value: String,
    pub label_bbox: Option<BoundingBox>,
    pub value_bbox: Option<BoundingBox>,
    /// The lower of the two lines' confidences, when both report one.
    pub confidence: Option<f32>,
}

struct Line {
    text: String,
    bbox: BoundingBox,
    confidence: Option<f32>,
}

fn center_y(b: &BoundingBox) -> f32 {
    b.y + b.height / 2.0
}

/// Every labelled field on the page, in reading order of the labels.
#[must_use]
pub fn pair_fields(out: &OcrOutput) -> Vec<Field> {
    let lines: Vec<Line> = out
        .blocks
        .iter()
        .flat_map(|b| &b.lines)
        .filter_map(|l| {
            let text = l.text.trim().to_string();
            (!text.is_empty()).then_some(Line {
                text,
                bbox: l.bbox?,
                confidence: l.confidence,
            })
        })
        .collect();

    let mut fields = Vec::new();
    // Labels waiting for a value on another line: (field index, label line).
    let mut open: Vec<(usize, usize)> = Vec::new();
    let mut is_label = vec![false; lines.len()];
    for (i, l) in lines.iter().enumerate() {
        if let Some(label) = l.text.strip_suffix(':') {
            is_label[i] = true;
            fields.push(Field {
                label: label.trim().to_string(),
                value: String::new(),
                label_bbox: Some(l.bbox),
                value_bbox: None,
                confidence: l.confidence,
            });
            open.push((fields.len() - 1, i));
        } else if let Some((label, value)) = l.text.split_once(':')
            && !label.trim().is_empty()
            && !value.trim().is_empty()
            // A time ("10:30") or a ratio is not a field.
            && !label.trim().ends_with(|c: char| c.is_ascii_digit())
        {
            is_label[i] = true;
            fields.push(Field {
                label: label.trim().to_string(),
                value: value.trim().to_string(),
                label_bbox: Some(l.bbox),
                value_bbox: Some(l.bbox),
                confidence: l.confidence,
            });
        }
    }

    // Candidate (distance, field, value line), searched right first, then below.
    let mut candidates: Vec<(f32, usize, usize)> = Vec::new();
    for &(f, li) in &open {
        let lb = lines[li].bbox;
        for (vi, v) in lines.iter().enumerate() {
            if is_label[vi] {
                continue;
            }
            let vb = v.bbox;
            let same_row = (center_y(&vb) - center_y(&lb)).abs() <= lb.height * 0.6;
            if same_row && vb.x >= lb.x + lb.width - lb.height * 0.5 {
                candidates.push((vb.x - (lb.x + lb.width), f, vi));
            } else {
                let below = vb.y - (lb.y + lb.height);
                let overlaps = vb.x < lb.x + lb.width + lb.height * 2.0 && vb.x + vb.width > lb.x;
                if below >= -lb.height * 0.3 && below <= lb.height * 1.5 && overlaps {
                    // Below ranks after every same-row candidate.
                    candidates.push((10_000.0 + below, f, vi));
                }
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut taken = vec![false; lines.len()];
    let mut filled = vec![false; fields.len()];
    for (_, f, vi) in candidates {
        if taken[vi] || filled[f] {
            continue;
        }
        taken[vi] = true;
        filled[f] = true;
        let v = &lines[vi];
        fields[f].value.clone_from(&v.text);
        fields[f].value_bbox = Some(v.bbox);
        fields[f].confidence = match (fields[f].confidence, v.confidence) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
    }
    fields
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffai_core::types::{OcrBlock, OcrLine};

    fn line(text: &str, x: f32, y: f32, w: f32) -> OcrLine {
        OcrLine {
            text: text.into(),
            words: Vec::new(),
            bbox: Some(BoundingBox {
                x,
                y,
                width: w,
                height: 30.0,
            }),
            confidence: Some(0.95),
        }
    }

    fn page(lines: Vec<OcrLine>) -> OcrOutput {
        OcrOutput {
            blocks: vec![OcrBlock { lines, bbox: None }],
        }
    }

    #[test]
    fn a_value_to_the_right_on_the_same_row_is_paired() {
        let f = pair_fields(&page(vec![
            line("Name of Creditor:", 40.0, 100.0, 300.0),
            line("Merrill Lynch", 560.0, 102.0, 240.0),
            line("Amount:", 40.0, 160.0, 140.0),
            line("$10,000", 560.0, 158.0, 150.0),
        ]));
        assert_eq!(f.len(), 2);
        assert_eq!(
            (f[0].label.as_str(), f[0].value.as_str()),
            ("Name of Creditor", "Merrill Lynch")
        );
        assert_eq!(
            (f[1].label.as_str(), f[1].value.as_str()),
            ("Amount", "$10,000")
        );
    }

    #[test]
    fn a_value_below_the_label_is_paired_when_nothing_is_to_its_right() {
        let f = pair_fields(&page(vec![
            line("Business Address:", 40.0, 100.0, 300.0),
            line("1117 Oak Street", 40.0, 140.0, 280.0),
        ]));
        assert_eq!(f[0].value, "1117 Oak Street");
    }

    #[test]
    fn a_label_and_value_in_one_line_are_split_and_times_are_not_fields() {
        let f = pair_fields(&page(vec![
            line("Amount: $2,500", 40.0, 100.0, 300.0),
            line("Meeting at 10:30 today", 40.0, 200.0, 400.0),
        ]));
        assert_eq!(f.len(), 1);
        assert_eq!(
            (f[0].label.as_str(), f[0].value.as_str()),
            ("Amount", "$2,500")
        );
    }

    #[test]
    fn a_blank_field_is_reported_blank_and_values_are_not_shared() {
        let f = pair_fields(&page(vec![
            line("Spouse:", 40.0, 100.0, 120.0),
            line("Employer:", 40.0, 300.0, 160.0),
            line("Acme Corp", 560.0, 300.0, 200.0),
        ]));
        assert_eq!(f.len(), 2);
        assert_eq!(
            f[0].value, "",
            "nothing near Spouse: it is blank, not given Employer's value"
        );
        assert_eq!(f[1].value, "Acme Corp");
    }
}
