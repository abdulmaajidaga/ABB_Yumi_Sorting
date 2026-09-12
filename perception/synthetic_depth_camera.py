"""Test-only helper: renders a synthetic overhead depth capture from a
full-360-degree synthetic_shapes point cloud, so pcd_extractor.py and
shape_classifier.py can be tested against something that actually resembles
a single-viewpoint RealSense capture -- occluded surfaces (the underside,
the far side) genuinely missing, not just noisy -- rather than the full-
surface clouds used everywhere else so far. Not part of the production
pipeline: a real depth frame comes from the camera itself.

Camera convention (self-consistent with pcd_extractor.deproject, standard
pinhole/OpenCV): +X right, +Y down, +Z forward (into the scene). The camera
sits at world (0, 0, camera_height) looking straight down at the table
(world Z=0), so camera-frame z = camera_height - world_z (depth), and
x_cam = world_x, y_cam = world_y (no additional rotation).
"""

import numpy as np
from scipy.spatial import cKDTree


def estimate_outward_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
    """Per-point outward surface normal, estimated the standard point-cloud
    way (local PCA over each point's k nearest neighbors -- the plane normal
    is the smallest-eigenvalue eigenvector of the local covariance), sign
    resolved by orienting away from the cloud's own centroid. Works on any
    point cloud without needing to know its shape's analytic equation."""
    centroid = points.mean(axis=0)
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    neighbors = points[idx]  # (N, k, 3)
    local = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", local, local)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]  # smallest-eigenvalue eigenvector, per point

    outward = points - centroid
    flip = np.einsum("ni,ni->n", normals, outward) < 0
    normals[flip] *= -1
    return normals


def cull_backfaces(points: np.ndarray, camera_pos: np.ndarray, k: int = 20,
                    min_cos_angle: float = 0.15) -> np.ndarray:
    """Keeps only points whose (estimated) outward normal faces toward the
    camera by a real margin -- the correct visibility test for a convex solid
    (all six of our sorting primitives are convex), and one that doesn't
    depend on point density or projection resolution the way a discrete
    pixel z-buffer does.

    A plain ">0" test isn't enough: for a point far below the camera, the
    direction to the camera is almost purely vertical, so a genuinely
    edge-on face (e.g. a cube's side wall, whose true normal is purely
    horizontal) has a dot product that's mathematically ~0, not cleanly
    negative -- noise in the estimated normal then flips roughly half of
    those truly-invisible edge points to "visible" at random. Requiring the
    (normalized) alignment to clear `min_cos_angle` gives edge-on faces the
    zero-visible-area they actually have, instead of a coin flip."""
    normals = estimate_outward_normals(points, k=k)
    to_camera = camera_pos[None, :] - points
    to_camera = to_camera / np.linalg.norm(to_camera, axis=1, keepdims=True)
    alignment = np.einsum("ni,ni->n", normals, to_camera)
    facing = alignment > min_cos_angle
    return points[facing]


def camera_to_world_extrinsic(camera_height: float) -> np.ndarray:
    """4x4 transform taking points from the camera's optical frame to the
    world/table frame used everywhere else (+Z up from the table)."""
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    T[2, 3] = camera_height
    return T


def project_to_depth_image(points_world: np.ndarray, camera_height: float, fx: float, fy: float,
                            cx: float, cy: float, img_w: int, img_h: int):
    """Projects an ALREADY-VISIBLE point cloud (already back-face-culled --
    see cull_backfaces) through an overhead pinhole camera at local
    (0, 0, camera_height), producing (depth_image, mask_image) in the same
    format a real depth camera would output. No culling here: for a MULTI-
    object scene, cull_backfaces must be called per-object first (its
    "away from the cloud's own centroid" sign heuristic is only valid for
    one convex object at a time -- calling it on several objects combined
    uses one meaningless shared centroid and lets occluded points leak
    through). See render_top_down for the single-object convenience path."""
    x_cam = points_world[:, 0]
    y_cam = points_world[:, 1]
    z_cam = camera_height - points_world[:, 2]

    in_front = z_cam > 1e-6
    x_cam, y_cam, z_cam = x_cam[in_front], y_cam[in_front], z_cam[in_front]

    u = np.round(fx * x_cam / z_cam + cx).astype(np.int64)
    v = np.round(fy * y_cam / z_cam + cy).astype(np.int64)

    in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, z_cam = u[in_bounds], v[in_bounds], z_cam[in_bounds]

    depth_image = np.zeros((img_h, img_w), dtype=np.float64)
    mask_image = np.zeros((img_h, img_w), dtype=bool)
    if u.shape[0] == 0:
        return depth_image, mask_image

    # Nearest point wins per pixel: sort by depth ascending, then for each
    # unique pixel keep the first (nearest) occurrence.
    order = np.argsort(z_cam)
    u_sorted, v_sorted, z_sorted = u[order], v[order], z_cam[order]
    pixel_id_sorted = v_sorted.astype(np.int64) * img_w + u_sorted.astype(np.int64)
    _, first_occurrence = np.unique(pixel_id_sorted, return_index=True)

    win_u = u_sorted[first_occurrence]
    win_v = v_sorted[first_occurrence]
    win_z = z_sorted[first_occurrence]

    depth_image[win_v, win_u] = win_z
    mask_image[win_v, win_u] = True
    return depth_image, mask_image


def render_top_down(points_world: np.ndarray, camera_height: float, fx: float, fy: float,
                     cx: float, cy: float, img_w: int, img_h: int, k_normals: int = 40):
    """Convenience wrapper for the SINGLE-OBJECT case: back-face-culls
    `points_world` (whose centroid must be that one object's own -- see
    cull_backfaces) and projects the result. For a multi-object scene, call
    cull_backfaces separately per object, concatenate the results, and call
    project_to_depth_image directly instead of this function."""
    camera_pos = np.array([0.0, 0.0, camera_height])
    points_world = cull_backfaces(points_world, camera_pos, k=k_normals)
    return project_to_depth_image(points_world, camera_height, fx, fy, cx, cy, img_w, img_h)
