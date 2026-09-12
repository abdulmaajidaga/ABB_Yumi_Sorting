"""Turns one GSAM2 object mask + a RealSense depth frame into an isolated 3D
point cloud, ready for shape_classifier.classify_shape(). This is the one
piece of the pipeline that genuinely needs the real camera to run for real --
everything here is plain pinhole-camera math, testable synthetically (see
synthetic_depth_camera.py / test_pcd_extractor.py) until the RealSense
arrives.

Pipeline: mask selects which depth pixels belong to this object -> pinhole
deprojection turns (pixel, depth) into a 3D point in the CAMERA frame ->
statistical outlier removal drops "flying pixels" (a real depth-camera/mask
artifact: silhouette-edge pixels that land partway between the object's true
depth and the background's, since the mask boundary rarely aligns exactly
with a depth discontinuity) -> an optional extrinsic transform moves the
cloud into the robot base frame (once camera calibration exists).
"""

import numpy as np
from scipy.spatial import cKDTree


def deproject(depth: np.ndarray, mask: np.ndarray, fx: float, fy: float, cx: float, cy: float,
              depth_scale: float = 1.0, min_depth: float = 0.05, max_depth: float = 2.0) -> np.ndarray:
    """Pinhole deprojection of the masked, valid-depth pixels into the camera
    frame (+X right, +Y down, +Z forward, standard camera-optical convention).
    Returns (N, 3)."""
    v_idx, u_idx = np.where(mask)
    z = depth[v_idx, u_idx].astype(np.float64) * depth_scale

    valid = (z > min_depth) & (z < max_depth) & np.isfinite(z)
    u_idx, v_idx, z = u_idx[valid], v_idx[valid], z[valid]

    x = (u_idx.astype(np.float64) - cx) * z / fx
    y = (v_idx.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def remove_flying_pixels(points: np.ndarray, k: int = 12, std_ratio: float = 2.0) -> np.ndarray:
    """Statistical outlier removal: for each point, the mean distance to its
    k nearest neighbors; points whose mean distance is more than std_ratio
    standard deviations above the cloud's average get dropped. Cheap
    (scipy cKDTree, no Open3D dependency) and this is exactly the artifact
    it's meant for -- mask-edge pixels whose depth doesn't match either the
    object's or the background's cleanly, so they end up as an isolated
    sparse halo around the object's true surface."""
    n = points.shape[0]
    if n < k + 1:
        return points
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)  # includes the point itself at distance 0
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    return points[mean_dists <= threshold]


def transform_to_frame(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous camera-frame -> target-frame transform."""
    rot = extrinsic[:3, :3]
    trans = extrinsic[:3, 3]
    return points @ rot.T + trans


def extract_object_pointcloud(
    depth: np.ndarray, mask: np.ndarray, fx: float, fy: float, cx: float, cy: float,
    depth_scale: float = 1.0, min_depth: float = 0.05, max_depth: float = 2.0,
    extrinsic: np.ndarray = None, outlier_removal: bool = True,
    outlier_k: int = 12, outlier_std_ratio: float = 2.0,
) -> np.ndarray:
    """Full pipeline: mask + depth + intrinsics -> cleaned-up point cloud, in
    the robot frame if `extrinsic` (camera-to-robot, 4x4) is given, otherwise
    in the camera's own optical frame."""
    points = deproject(depth, mask, fx, fy, cx, cy, depth_scale, min_depth, max_depth)
    if points.shape[0] == 0:
        return points
    if outlier_removal:
        points = remove_flying_pixels(points, k=outlier_k, std_ratio=outlier_std_ratio)
    if extrinsic is not None:
        points = transform_to_frame(points, extrinsic)
    return points
