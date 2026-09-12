"""Original geometric classifier (height-slicing + circularity + vertex-count,
no RANSAC), kept alongside the pyransac3d-based shape_classifier.py purely for
a direct accuracy/speed comparison -- this is not the production classifier
(that's shape_classifier.py, per an explicit choice to use pyransac3d for the
shapes it covers). See benchmark_classifiers.py.
"""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.spatial import ConvexHull, QhullError

N_SLICES = 12
MIN_POINTS_PER_SLICE = 6
CIRCULARITY_CIRCLE_MIN = 0.90
CIRCULARITY_QUAD_MIN = 0.77
CUBE_ASPECT_RATIO_MAX = 1.25
CONE_MIN_RELATIVE_IMPROVEMENT = 0.15


@dataclass
class ShapeResult:
    shape: str
    params: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)


def _hull_analysis(xy: np.ndarray) -> Optional[dict]:
    if xy.shape[0] < 3:
        return None
    try:
        hull = ConvexHull(xy)
    except QhullError:
        return None
    area = hull.volume
    pts = xy[hull.vertices]
    perimeter = 0.0
    for i in range(len(pts)):
        perimeter += np.linalg.norm(pts[i] - pts[(i + 1) % len(pts)])
    if perimeter <= 1e-9:
        return None
    circularity = float(4 * np.pi * area / (perimeter ** 2))
    hull_cv = pts.astype(np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(hull_cv, 0.04 * perimeter, True)
    return {"circularity": circularity, "vertex_count": int(len(approx))}


def _slice_stats(points, z_edges):
    slices = []
    for i in range(len(z_edges) - 1):
        lo, hi = z_edges[i], z_edges[i + 1]
        mask = (points[:, 2] >= lo) & (points[:, 2] < hi if i < len(z_edges) - 2 else points[:, 2] <= hi)
        pts = points[mask]
        if pts.shape[0] < MIN_POINTS_PER_SLICE:
            slices.append(None)
            continue
        xy = pts[:, :2]
        centroid = xy.mean(axis=0)
        radii = np.linalg.norm(xy - centroid, axis=1)
        radius = float(np.median(radii))
        hull_info = _hull_analysis(xy)
        slices.append({
            "z_mid": float((lo + hi) / 2), "centroid": centroid, "radius": radius,
            "circularity": hull_info["circularity"] if hull_info else None,
            "vertex_count": hull_info["vertex_count"] if hull_info else None,
            "n": int(pts.shape[0]),
        })
    return slices


def _fit_cuboid_dims(points, height):
    xy = points[:, :2].astype(np.float32)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(xy)
    yaw = float(np.deg2rad(angle_deg))
    dims = sorted([float(w), float(h), float(height)], reverse=True)
    return dims, np.array([cx, cy]), yaw


def classify_shape(points: np.ndarray) -> ShapeResult:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 30:
        return ShapeResult(shape="unknown", debug={"reason": "too few points"})

    z = points[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    height = z_max - z_min
    if height < 1e-6:
        return ShapeResult(shape="unknown", debug={"reason": "degenerate height"})

    z_edges = np.linspace(z_min, z_max, N_SLICES + 1)
    slices = _slice_stats(points, z_edges)
    valid = [s for s in slices if s is not None]
    if len(valid) < 3:
        return ShapeResult(shape="unknown", debug={"reason": "too few valid slices"})

    inner = valid[1:-1] if len(valid) > 4 else valid
    circ_values = [s["circularity"] for s in inner if s["circularity"] is not None]
    mean_circularity = float(np.mean(circ_values)) if circ_values else 0.0
    debug = {"mean_circularity": mean_circularity, "height": height, "n_valid_slices": len(valid)}

    if mean_circularity >= CIRCULARITY_CIRCLE_MIN:
        return _classify_circular(points, valid, height, debug)

    vertex_counts = [s["vertex_count"] for s in inner if s["vertex_count"] in (3, 4)]
    votes_quad = vertex_counts.count(4)
    votes_tri = vertex_counts.count(3)
    is_quad = votes_quad > votes_tri if votes_quad != votes_tri else mean_circularity >= CIRCULARITY_QUAD_MIN

    if is_quad:
        dims, centroid, yaw = _fit_cuboid_dims(points, height)
        ratio = dims[0] / max(dims[2], 1e-9)
        shape = "cube" if ratio <= CUBE_ASPECT_RATIO_MAX else "cuboid"
        debug["dims_sorted_desc"] = dims
        return ShapeResult(shape=shape, params={
            "center_xy": centroid.tolist(), "z_min": z_min, "z_max": z_max, "dims": dims, "yaw": yaw,
        }, debug=debug)
    else:
        centroid = points[:, :2].mean(axis=0)
        return ShapeResult(shape="prism", params={
            "center_xy": centroid.tolist(), "z_min": z_min, "z_max": z_max, "height": height,
        }, debug=debug)


def _classify_circular(points, valid_slices, height, debug):
    h = np.array([s["z_mid"] for s in valid_slices])
    r = np.array([s["radius"] for s in valid_slices])
    h_rel = h - h.min()

    r_mean = r.mean()
    rmse_const = float(np.sqrt(np.mean((r - r_mean) ** 2)))

    A = np.vstack([h_rel, np.ones_like(h_rel)]).T
    (a, b), *_ = np.linalg.lstsq(A, r, rcond=None)
    rmse_linear = float(np.sqrt(np.mean((r - (a * h_rel + b)) ** 2)))

    R_est = height / 2
    inside = np.clip(R_est ** 2 - (h_rel - R_est) ** 2, 0, None)
    rmse_sphere = float(np.sqrt(np.mean((r - np.sqrt(inside)) ** 2)))

    relative_improvement = (rmse_const - rmse_linear) / max(rmse_const, 1e-12)
    rmses = {"cylinder": rmse_const, "sphere": rmse_sphere}
    if relative_improvement >= CONE_MIN_RELATIVE_IMPROVEMENT:
        rmses["cone"] = rmse_linear
    best = min(rmses, key=rmses.get)

    if best == "cylinder":
        return ShapeResult(shape="cylinder", params={"radius": float(r_mean), "height": height}, debug=debug)
    elif best == "sphere":
        return ShapeResult(shape="sphere", params={"radius": float(R_est), "height": height}, debug=debug)
    else:
        apex_up = r[-1] < r[0]
        base_radius = float(r[0] if apex_up else r[-1])
        return ShapeResult(shape="cone", params={
            "base_radius": base_radius, "height": height, "apex_up": bool(apex_up),
        }, debug=debug)
