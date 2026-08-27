# Stereo Visual SLAM — KITTI

A stereo visual SLAM system built from scratch: an ORB feature front-end, keyframe-based pose graph, loop-closure detection, and GTSAM pose-graph optimization. On KITTI sequence 00 (~3.7 km), loop closure reduces trajectory drift from **40.5 m to 3.3 m ATE** — a 92% reduction.


---

## Why this project exists

This is the third stage of a progression, where each stage fixes a specific, nameable limitation of the previous one:

1. **Monocular VO** — ORB + essential-matrix (`findEssentialMat` + `recoverPose`) motion estimation on KITTI 00. `recoverPose` returns a translation *direction* but only a unit-length placeholder for magnitude. Absolute scale is unrecoverable from a single camera. Scale had to be injected per frame from ground-truth displacement (direction from VO, magnitude from ground truth). Result: 82 m ATE, drift-limited, and not truly self-contained.
2. **Stereo VO** fixes scale: the known camera baseline gives metric depth every frame (`Z = fx·b/d`), so the trajectory is correctly scaled with **no ground-truth injection anywhere in the estimate**. Motion comes from `solvePnPRansac` on 3D↔2D correspondences instead of the essential matrix. Result: 40.6 m ATE. But error still accumulates without bound — classic visual-odometry drift.
3. **Stereo SLAM** (this repo) fixes drift: detecting when the camera revisits a place adds a loop-closure constraint, and global pose-graph optimization redistributes the accumulated error across the whole trajectory. Result: 3.3 m ATE, bounded.

The point of the arc: stereo removes the scale problem that plagues monocular; loop-closure SLAM removes the unbounded drift that plagues pure odometry.

---

## Results (KITTI sequence 00)

| Metric | Value |
|---|---|
| Frames processed | 4541 |
| Keyframes | 454 (every 10th frame) |
| Loops detected | 64 (threshold set from match-count distribution) |
| ATE before loop closure (raw stereo VO) | 40.5 m |
| ATE after loop closure (SLAM) | **3.3 m** |
| Drift reduction | 92% |

An intermediate experiment: loop constraints using an **identity-transform approximation** (assuming revisited keyframes share both position and orientation) gave 17.5 m ATE. Replacing that with **PnP-computed relative transforms** between loop keyframes dropped ATE to 3.3 m — showing that accurate loop *geometry* matters far more than mere place recognition.

---

## Pipeline

**Front-end (stereo visual odometry)**
- ORB feature detection (5000 features per image)
- Stereo matching (left ↔ right) with epipolar row check and positive-disparity filtering
- Depth per feature via `Z = fx · b / d`, back-projected to 3D through the inverse intrinsics
- Temporal matching (previous frame → current) to build 3D↔2D correspondences
- Pose estimation with `solvePnPRansac`, refined on inliers
- Frame-to-frame pose accumulation

**Back-end (SLAM)**
- Keyframe selection (fixed interval)
- Pose graph: nodes are keyframe poses, edges are relative transforms between consecutive keyframes
- Loop detection: match each keyframe's ORB descriptors against temporally distant keyframes; a match count above threshold flags a revisit
- Loop-closure edges: relative transform between the two loop keyframes computed via feature matching + PnP
- Global optimization with GTSAM Levenberg-Marquardt

---

## Key design decisions

- **`solvePnP` (3D↔2D) over the essential matrix (2D↔2D):** with stereo depth in hand, PnP lets depth error enter the pose estimate once (in the 3D points) rather than twice, and it recovers real metric scale.
- **Sparse feature-level disparity over dense stereo:** disparity is computed only at ORB keypoints, where matching is reliable, instead of over textureless regions where dense methods guess.
- **`fx` (not `fy`) throughout the depth math:** disparity is a purely horizontal shift, so only the horizontal focal length enters — `fy` never does. (They happen to be equal on KITTI, but the term is chosen for correctness, not the coincidence.)
- **Inlier-only pose refinement was evaluated** and gave no measurable ATE change on KITTI 00; RANSAC's consensus pose was already near-optimal for the inlier set.
- **Loop-detection threshold set empirically:** the match-count distribution showed a clean separation — coincidental keyframe similarity stayed below ~60 matches, while genuine revisits produced 100–1200. The threshold was placed in that gap.
- **PnP-computed loop transforms over identity:** identity assumes same position *and* heading on revisit; PnP measures the true relative pose. This single change moved ATE from 17.5 m to 3.3 m.

---

## Limitations and future work

- **No bundle adjustment:** the back-end optimizes poses only, not 3D landmark points. Full systems (e.g. ORB-SLAM) jointly refine poses and structure.
- **Simple place recognition:** loop detection is brute-force descriptor matching against past keyframes, which does not scale to large maps. DBoW2 / bag-of-visual-words would be the scalable replacement.
- **Fixed-interval keyframes:** selection is every N frames rather than motion-adaptive (keyframe on translation/rotation thresholds), which would place keyframes more efficiently and improve accuracy through turns.
- **Single sequence:** evaluated on KITTI 00 only; validation on additional sequences (05, 07) would demonstrate generality.

---

## Relationship to ORB-SLAM

This system uses ORB features, as ORB-SLAM does, but it is **not** ORB-SLAM. It is a simpler stereo pose-graph SLAM: loop closure plus pose-graph optimization, without ORB-SLAM's bundle adjustment, DBoW2 place recognition, relocalization, or map management.

---

## Running it

Environment (GTSAM requires NumPy < 2.0):

```
python 3.12
numpy 1.26.4
opencv-python 4.10
gtsam 4.2.2
matplotlib
```

```bash
source slam_env/bin/activate
python3 Stereo_slam.py
```

Expects the KITTI odometry grayscale dataset, calibration, and ground-truth poses under `Dataset_kitti/` (git-ignored — download separately from the KITTI odometry benchmark).

---

## The progression at a glance

| Stage | Motion estimation | Scale source | Drift handling | ATE (KITTI 00) |
|---|---|---|---|---|
| Monocular VO | Essential matrix + `recoverPose` | Ground-truth displacement (injected) | None | 82 m |
| Stereo VO | `solvePnPRansac` (3D↔2D) | Stereo baseline (`Z = fx·b/d`), self-contained | None | 40.6 m |
| Stereo SLAM | `solvePnPRansac` + pose-graph optimization | Stereo baseline, self-contained | Loop closure + GTSAM | **3.3 m** |

All three run on KITTI odometry sequence 00 (~3.7 km, 4541 frames) and share the same ORB front-end (5000 features, BFMatcher with Hamming distance, Lowe's ratio 0.75) and the same ATE (Umeyama alignment + RMS) / RPE evaluation.

## Related projects

- Monocular Visual Odometry — [link]
- Stereo Visual Odometry — [link]
