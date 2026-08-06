//! IoU cost and rectangular Hungarian assignment.
//!
//! # Why Hungarian and not greedy
//!
//! Greedy descending-IoU matching is what a first draft reaches for and it is
//! not what ByteTrack does. The difference does not show up as missed boxes —
//! it shows up as **identity switches**, because greedy will hand a detection
//! to the track that wants it most rather than to the assignment that is best
//! overall. AP50 cannot see that; IDF1 can, and IDF1 is exactly the metric this
//! project has never measured.
//!
//! `greedy_picks_a_worse_global_assignment` below is a case where the two
//! disagree. It exists so that swapping Hungarian back out for greedy fails a
//! test rather than quietly costing IDF1.

/// Intersection over union of two `[x0, y0, x1, y1]` boxes.
pub fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let x0 = a[0].max(b[0]);
    let y0 = a[1].max(b[1]);
    let x1 = a[2].min(b[2]);
    let y1 = a[3].min(b[3]);
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let inter = (x1 - x0) * (y1 - y0);
    let aa = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let bb = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = aa + bb - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// `1 - IoU`, the cost the assignment minimises.
pub fn cost_matrix(tracks: &[[f32; 4]], dets: &[[f32; 4]]) -> Vec<Vec<f32>> {
    tracks
        .iter()
        .map(|t| dets.iter().map(|d| 1.0 - iou(*t, *d)).collect())
        .collect()
}

/// `1 - IoU * score` — the association cost with detection confidence folded
/// in, which is what the reference tracker's `fuse_score` computes.
///
/// The argument for it: when two detections overlap one track similarly, pure
/// IoU picks by geometry alone and is indifferent to the detector saying one of
/// them is far more likely to be a real object. Folding the score in breaks
/// that tie toward the confident box.
///
/// It is NOT free — it makes the `match_thresh` gate stricter, because the
/// product is always ≤ the IoU. A fair test therefore has to re-sweep the
/// threshold rather than compare at the value tuned for pure IoU, or it prices
/// a gate change as an algorithm change.
pub fn cost_matrix_fused(
    tracks: &[[f32; 4]],
    dets: &[[f32; 4]],
    scores: &[f32],
) -> Vec<Vec<f32>> {
    tracks
        .iter()
        .map(|t| {
            dets.iter()
                .zip(scores.iter())
                .map(|(d, s)| 1.0 - iou(*t, *d) * s)
                .collect()
        })
        .collect()
}

/// Raise the cost of every CROSS-CLASS pair beyond any usable gate.
///
/// The cost matrix was pure geometry, and `Track::class_id` was stored, then
/// overwritten by whatever matched — so a car detection sitting where a person
/// was could take that person's track and silently relabel it. On MOT17-13,
/// 36.8 % of our detections are cars, buses and traffic lights, so this is not
/// hypothetical.
///
/// A separate pass rather than a parameter on `cost_matrix` because the second
/// association pass deliberately uses a different gate, and threading a class
/// rule through both is how one of them ends up forgotten.
pub fn forbid_cross_class(cost: &mut [Vec<f32>], track_cls: &[u32], det_cls: &[u32]) {
    for (i, row) in cost.iter_mut().enumerate() {
        for (j, c) in row.iter_mut().enumerate() {
            if track_cls.get(i) != det_cls.get(j) {
                *c = f32::MAX / 4.0;
            }
        }
    }
}

/// Row/column swap, so the transposed solve gates on the transposed matrix.
fn transpose(m: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if m.is_empty() {
        return Vec::new();
    }
    (0..m[0].len()).map(|j| m.iter().map(|r| r[j]).collect()).collect()
}

/// Optimal one-to-one assignment minimising total cost, for a rectangular
/// matrix. Returns `(row, col)` pairs whose cost is at most `max_cost`.
///
/// Jonker-Volgenant-style shortest augmenting path with dual potentials — the
/// same algorithm `scipy.optimize.linear_sum_assignment` runs, which is what
/// the reference tracker calls. O(n^2 m) and n here is the number of live
/// tracks, so tens.
///
/// Pairs are filtered by `max_cost` AFTER solving, not before: removing
/// expensive edges up front changes which assignment is optimal, and a pair the
/// solver only chose because a cheaper one was taken elsewhere still tells you
/// the cheaper one was taken.
pub fn assign(cost: &[Vec<f32>], max_cost: f32) -> Vec<(usize, usize)> {
    assign_gated(cost, cost, max_cost)
}

/// Assign on one cost matrix, ADMIT on another.
///
/// Exists because folding detection confidence into the cost changes two
/// things at once, and they want separating. `1 - IoU * score` is a better
/// RANKING — when two detections overlap a track equally, the one the detector
/// believes in should win — but it is a worse GATE, because the product is
/// always ≤ the IoU, so a fixed `max_cost` silently tightens by a factor of the
/// score. Sweeping the threshold to compensate then prices a gate change as an
/// algorithm change, and the measured response is correspondingly jumpy.
///
/// So: rank by `rank_cost`, admit by `gate_cost`. "Whose box is it" and "is it
/// close enough to be anyone's" are different questions.
pub fn assign_gated(
    rank_cost: &[Vec<f32>],
    gate_cost: &[Vec<f32>],
    max_cost: f32,
) -> Vec<(usize, usize)> {
    let cost = rank_cost;
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }
    let m = cost[0].len();
    if m == 0 {
        return Vec::new();
    }
    // The Jonker-Volgenant formulation below REQUIRES rows <= cols. With more
    // tracks than detections it silently produced a wrong assignment — caught
    // by `rectangular_more_tracks_than_detections`, which is the common case in
    // tracking (objects leave the frame). Transpose, solve, swap back.
    if n > m {
        let t: Vec<Vec<f32>> = (0..m)
            .map(|j| (0..n).map(|i| cost[i][j]).collect())
            .collect();
        let mut out: Vec<(usize, usize)> =
            assign_gated(&t, &transpose(gate_cost), max_cost)
                .into_iter()
                .map(|(a, b)| (b, a))
                .collect();
        out.sort_unstable();
        return out;
    }
    const INF: f32 = f32::INFINITY;

    // u/v are dual potentials, p maps column -> assigned row (1-indexed via the
    // usual sentinel-row trick), way is the augmenting path predecessor.
    let mut u = vec![0.0f32; n + 1];
    let mut v = vec![0.0f32; m + 1];
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![INF; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = INF;
            let mut j1 = 0usize;
            for j in 1..=m {
                if used[j] {
                    continue;
                }
                let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            if !delta.is_finite() {
                break;
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        // Walk the augmenting path back, flipping assignments.
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut out = Vec::new();
    for j in 1..=m {
        let i = p[j];
        if i >= 1 && i <= n && gate_cost[i - 1][j - 1] <= max_cost {
            out.push((i - 1, j - 1));
        }
    }
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ranking and admission are genuinely separate matrices.
    ///
    /// The whole point of `assign_gated`: a pair too expensive to be ADMITTED
    /// under the ranking cost must still be admitted if the GATE cost allows
    /// it. Folding score into the cost made every pair look worse and silently
    /// tightened the threshold — this is the test that stops that returning.
    #[test]
    fn gate_admits_what_the_ranking_cost_would_reject() {
        let rank = vec![vec![0.9f32]];   // 1 - IoU*score, above max_cost
        let gate = vec![vec![0.5f32]];   // 1 - IoU,       below max_cost
        assert!(assign(&rank, 0.8).is_empty(), "ranking cost alone rejects it");
        assert_eq!(assign_gated(&rank, &gate, 0.8), vec![(0, 0)], "gate should admit");
    }

    /// The transposed path (more tracks than detections) must gate on the
    /// transposed GATE matrix, not on the ranking one.
    #[test]
    fn gate_survives_the_transpose_path() {
        // 3 tracks, 1 detection -> rows > cols, so the transpose branch runs.
        let rank = vec![vec![0.95f32], vec![0.90], vec![0.99]];
        let gate = vec![vec![0.70f32], vec![0.10], vec![0.90]];
        let got = assign_gated(&rank, &gate, 0.8);
        assert_eq!(got, vec![(1, 0)], "track 1 is both best-ranked and inside the gate");
    }

    #[test]
    fn iou_basics() {
        assert!((iou([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]) - 1.0).abs() < 1e-6);
        assert_eq!(iou([0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]), 0.0);
        // half-overlap: inter 50, union 150
        let v = iou([0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]);
        assert!((v - 50.0 / 150.0).abs() < 1e-6, "{v}");
    }

    /// THE test that keeps Hungarian in place.
    ///
    /// Greedy takes the globally-cheapest edge first — (0,0) at 0.10 — and is
    /// then forced into (1,1) at 0.90, total 1.00. The optimal assignment is
    /// (0,1) + (1,0) = 0.20 + 0.15 = 0.35. A greedy implementation passes every
    /// other test in this file and fails this one.
    #[test]
    fn greedy_picks_a_worse_global_assignment() {
        let cost = vec![vec![0.10, 0.20], vec![0.15, 0.90]];
        let got = assign(&cost, 1.0);
        let total: f32 = got.iter().map(|(i, j)| cost[*i][*j]).sum();
        assert_eq!(got.len(), 2, "{got:?}");
        assert!(total < 0.40, "total {total} — this is the greedy answer");
        assert!(got.contains(&(0, 1)) && got.contains(&(1, 0)), "{got:?}");
    }

    #[test]
    fn rectangular_more_detections_than_tracks() {
        let cost = vec![vec![0.9, 0.1, 0.8]];
        let got = assign(&cost, 1.0);
        assert_eq!(got, vec![(0, 1)]);
    }

    #[test]
    fn rectangular_more_tracks_than_detections() {
        let cost = vec![vec![0.7], vec![0.2], vec![0.9]];
        let got = assign(&cost, 1.0);
        assert_eq!(got, vec![(1, 0)]);
    }

    /// Pairs above `max_cost` are dropped — an assignment is not a match.
    #[test]
    fn max_cost_filters_after_solving() {
        let cost = vec![vec![0.95, 0.99], vec![0.99, 0.10]];
        let got = assign(&cost, 0.5);
        assert_eq!(got, vec![(1, 1)], "{got:?}");
    }

    #[test]
    fn empty_inputs_are_not_a_panic() {
        assert!(assign(&[], 0.5).is_empty());
        let empty_row: Vec<Vec<f32>> = vec![vec![]];
        assert!(assign(&empty_row, 0.5).is_empty());
    }
}
