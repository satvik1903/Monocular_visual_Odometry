# Stereo Visual SLAM — KITTI

A stereo visual SLAM system built from scratch: an ORB feature front-end, keyframe-based pose graph, loop-closure detection, and GTSAM pose-graph optimization. On KITTI sequence 00 (~3.7 km), loop closure reduces trajectory drift from **40.5 m to 3.3 m ATE** — a 92% reduction.

![Stereo SLAM loop closure correction on KITTI 00](stereo_slam_kitti00.png)


---

## Why this project exists

This is the third stage of a progression, where each stage fixes a specific, nameable limitation of the previous one:

1. **Monocular VO** — ORB + essential-matrix (`findEssentialMat` + `recoverPose`) motion estimation on KITTI 00. `recoverPose` returns a translation *direction* but only a unit-length placeholder for magnitude. Absolute scale is unrecoverable from a single camera. Scale had to be injected per frame from ground-truth displacement (direction from VO, magnitude from ground truth). 
2. **Stereo VO** fixes scale: the known camera baseline gives metric depth every frame (`Z = fx·b/d`), so the trajectory is correctly scaled with **no ground-truth injection anywhere in the estimate**. Motion comes from `solvePnPRansac` on 3D↔2D correspondences instead of the essential matrix. Result: 40.6 m ATE. But error still accumulates without bounds, classic visual-odometry drift.
3. **Stereo SLAM**  fixes drift: detecting when the camera revisits a place adds a loop-closure constraint, and global pose-graph optimization redistributes the accumulated error across the whole trajectory. 


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

## The progression at a glance

| Stage | Motion estimation | Scale source | Drift handling | ATE (KITTI 00) |
|---|---|---|---|---|
| Monocular VO | Essential matrix + `recoverPose` | Ground-truth displacement (injected) | None | 82 m |
| Stereo VO | `solvePnPRansac` (3D↔2D) | Stereo baseline (`Z = fx·b/d`), self-contained | None | 40.6 m |
| Stereo SLAM | `solvePnPRansac` + pose-graph optimization | Stereo baseline, self-contained | Loop closure + GTSAM | **3.3 m** |

All three run on KITTI odometry sequence 00 (~3.7 km, 4541 frames) and share the same ORB front-end (5000 features, BFMatcher with Hamming distance, Lowe's ratio 0.75) and the same ATE (Umeyama alignment + RMS) / RPE evaluation.
