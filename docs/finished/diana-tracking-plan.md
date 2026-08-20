# Diana — multi-object tracking

**Gap:** Diana has been benchmarked on MOT17 — the *Multiple Object Tracking*
benchmark — for this whole campaign, scoring AP50 on detections while
**discarding column 2 of `gt.txt`, which is the track ID**. We do the D and not
the MOT. Ultralytics ships `model.track()` with ByteTrack and BoT-SORT.

**Why now:** the streaming video ingest just landed and a tracker is its natural
consumer; surveillance is Diana's strongest case and surveillance without
identity is half a product; and unusually for this project it needs **no new
weights** — Kalman + assignment is pure algorithm, so no AGPL checkpoint, no
converter, no five-tier oracle.

---

## 1. Which tracker, and why that one

**ByteTrack first.** Its whole idea is that low-confidence boxes are usually
occluded objects rather than noise, so it associates twice: high-score
detections first, then low-score ones against whatever tracks are still
unmatched. That recovers exactly the occlusions MOT17 is full of.

Crucially it is **appearance-free** — no ReID network, no embeddings, no second
model. BoT-SORT adds an appearance model and would drag a whole new weight file
and its licence in with it. ByteTrack keeps Diana weight-free.

## 2. The pieces, each gated on its own

### 2a. Kalman filter — `track/kalman.rs`
Standard SORT 8-dimensional constant-velocity model: state
`[cx, cy, aspect, height, vx, vy, va, vh]`, observation `[cx, cy, aspect,
height]`. Process and measurement noise scaled by height, as in the reference —
a box twice as tall is twice as uncertain in pixels.

**Gate:** predict/update over a synthetic constant-velocity trajectory recovers
the true position to a stated tolerance, and the covariance shrinks with each
update rather than growing.

### 2b. Assignment — `track/assign.rs`
IoU cost matrix plus **Hungarian** (Jonker-Volgenant style) rectangular
assignment.

Greedy descending-IoU matching is simpler and is what a first draft reaches for.
It is not what the reference does, and the difference shows up as ID switches
rather than as missed boxes — so it would be invisible in AP50 and visible in
IDF1, which is precisely the metric this project has not been measuring.
**Implement Hungarian; do not start with greedy and hope.**

**Gate:** unit tests on hand-computed cost matrices, including a case where
greedy picks a worse global assignment than Hungarian, so the test would fail if
someone swapped it back.

### 2c. Track lifecycle — `track/mod.rs`
States: `New` → `Tracked` → `Lost` → `Removed`. A track survives `max_age`
frames of being lost before removal; a new track must be confirmed for
`min_hits` frames before it is reported, so single-frame false positives never
get an ID.

**Gate:** a scripted sequence — an object appears, is occluded for N frames,
reappears — keeps its ID for `N <= max_age` and gets a new one beyond it.

## 3. The real gate: MOT17 with identity metrics

AP50 cannot see a tracker. The metrics that can:

| metric | what it catches |
|---|---|
| **MOTA** | FP + FN + ID-switches against GT count — overall accuracy |
| **IDF1** | identity F1 — whether the SAME object keeps the SAME id |
| **IDSW** | raw count of identity switches |
| **MT / ML** | mostly-tracked / mostly-lost trajectories |

`gt.txt` already has the IDs. Scoring uses the standard MOT convention this
project already applies for detection: `conf=0` and `class != 1` rows dropped.

**Published ByteTrack on MOT17 sits around MOTA 80 / IDF1 77 on the test set
with a strong detector.** Ours will be lower — different detector, no tuning —
and the number to beat is **our own detector's ceiling**, not the paper's.
Report against the paper as standing, against ourselves as progress
(`codec-measurement` §12).

## 4. Wiring

* `ffai detect -i clip.mp4 --track` — IDs in the per-frame line, matching
  Ultralytics' `model.track()` output shape.
* `ffai-py`: `Detector.track(frame)` returning ids alongside boxes.
* Both consume the streaming iterator, so memory stays constant.

## 5. Order of work, and the stop rule

1. Kalman + its gate.
2. Assignment + its gate, including the greedy-loses case.
3. Lifecycle + its scripted gate.
4. MOT17 scorer (MOTA/IDF1/IDSW) — **before** any tuning, so the first number is
   honest.
5. Wire CLI and Python.

**Stop rule:** no threshold tuning until step 4 produces a baseline. This
campaign's repeated failure has been optimising against a number that later
moved; a tracker has four thresholds and it would be easy to tune all of them
into a corpus.
