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
            assign(&t, max_cost).into_iter().map(|(a, b)| (b, a)).collect();
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
        if i >= 1 && i <= n && cost[i - 1][j - 1] <= max_cost {
            out.push((i - 1, j - 1));
        }
    }
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
