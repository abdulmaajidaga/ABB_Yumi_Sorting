"""Shape classifier designed for what a single overhead camera actually sees
-- a thin visible cap near the object's TOP, never the full object. Built
after test_pcd_extractor.py showed both shape_classifier.py (pyransac3d) and
shape_classifier_geometric.py collapse to ~50% or worse on realistic single-
viewpoint point clouds, because both rely on signals that need the object's
full height (never visible from one overhead shot: an opaque object's
bottom, and most of its sides, are always occluded by the object itself).

Two signals that ARE always available from one overhead capture, no matter
how little of the object is visible:
  1. The visible cap's SILHOUETTE (XY outline) -- circular, square/
     rectangular, or triangular. Occlusion truncates how much of the object
     you see top-to-bottom, but never distorts the outline of what IS
     visible from directly above.
  2. How Z falls off RADIALLY from the visible cap's own highest point --
     flat (a cylinder/cube/prism's flat top), linearly tapering (a cone's
     point), or falling off like a circular arc (a sphere's dome). This is
     computed relative to the cap's own center and own visible points only,
     never the object's true (unreachable) full height.

Cube vs. cuboid is a real, honest limitation here: an opaque box's true
height can't be measured without seeing a side or the bottom, so this only
distinguishes footprint shape (square vs. rectangular) -- a short box with a
square footprint will read as "cube" even if it isn't one in 3D. That's a
real blind spot of a single top-down view, not a bug to paper over.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import ConvexHull, QhullError

CIRCULARITY_CIRCLE_MIN = 0.90
CIRCULARITY_QUAD_MIN = 0.77
N_RADIAL_BINS = 8
MIN_POINTS_PER_BIN = 5
MODEL_MIN_RELATIVE_IMPROVEMENT = 0.35  # same anti-overfitting guard as
                                        # before: linear/dome models have
                                        # more free parameters than "flat",
                                        # so need a real margin, not just any
                                        # edge, before being preferred.


@dataclass
class ShapeResult:
    shape: str  # "sphere" | "cylinder" | "cone" | "cube" | "cuboid" | "prism" | "unknown"
    params: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)


def _hull_analysis(xy: np.ndarray):
    if xy.shape[0] < 3:
        return None
    try:
        hull = ConvexHull(xy)
    except QhullError:
        return None
    area = hull.volume
    pts = xy[hull.vertices]
    perimeter = sum(np.linalg.norm(pts[i] - pts[(i + 1) % len(pts)]) for i in range(len(pts)))
    if perimeter <= 1e-9:
        return None
    circularity = float(4 * np.pi * area / (perimeter ** 2))
    hull_cv = pts.astype(np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(hull_cv, 0.04 * perimeter, True)
    return {"circularity": circularity, "vertex_count": int(len(approx))}


def _footprint_rect(points: np.ndarray):
    xy = points[:, :2].astype(np.float32)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(xy)
    return float(w), float(h), np.array([cx, cy]), float(np.deg2rad(angle_deg))


def _radial_profile(points: np.ndarray, centroid_xy: np.ndarray, n_bins: int = N_RADIAL_BINS):
    r = np.linalg.norm(points[:, :2] - centroid_xy, axis=1)
    z = points[:, 2]
    r_max = r.max()
    if r_max < 1e-9:
        return None
    edges = np.linspace(0, r_max, n_bins + 1)
    r_mid, z_mean = [], []
    for i in range(n_bins):
        in_bin = (r >= edges[i]) & (r < edges[i + 1] if i < n_bins - 1 else r <= edges[i + 1])
        if in_bin.sum() < MIN_POINTS_PER_BIN:
            continue
        r_mid.append((edges[i] + edges[i + 1]) / 2)
        z_mean.append(float(z[in_bin].mean()))
    if len(r_mid) < 3:
        return None
    return np.array(r_mid), np.array(z_mean)


def _fit_radial_shape(r_mid: np.ndarray, z_mean: np.ndarray, z_top: float):
    """Fit flat / linear-taper (cone) / circular-arc (sphere dome) models to
    how Z falls off from the visible cap's own top point, purely as a
    function of radial distance -- no dependence on the object's true
    (unseen) full height."""
    drop = z_top - z_mean  # how far below the cap's peak, per radial bin

    # "Flat" needs its own free constant here too, not a rigid pin at
    # drop=0 -- z_top is just the single highest SAMPLED point, almost
    # always a slight noise outlier above the surface's true top, so every
    # other point's drop is systematically a little positive even on a
    # genuinely flat cap. Pinning "flat" at exactly 0 while "linear" gets a
    # free intercept let the intercept alone (not real curvature) explain
    # away that bias and made linear look better than it truly was.
    rmse_flat = float(np.sqrt(np.mean((drop - drop.mean()) ** 2)))

    A = np.vstack([r_mid, np.ones_like(r_mid)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, drop, rcond=None)
    rmse_linear = float(np.sqrt(np.mean((drop - (slope * r_mid + intercept)) ** 2)))

    r_max = r_mid.max()
    R_candidates = np.logspace(np.log10(max(r_max * 0.3, 1e-6)), np.log10(r_max * 30), 40)
    best_R, best_rmse_dome = None, np.inf
    for R in R_candidates:
        pred = R - np.sqrt(np.clip(R ** 2 - r_mid ** 2, 0, None))
        rmse = float(np.sqrt(np.mean((drop - pred) ** 2)))
        if rmse < best_rmse_dome:
            best_rmse_dome, best_R = rmse, R

    scale = max(float(np.abs(drop).max()), 1e-9)
    improvement_linear = (rmse_flat - rmse_linear) / max(rmse_flat, 1e-12)
    improvement_dome = (rmse_flat - best_rmse_dome) / max(rmse_flat, 1e-12)

    candidates = {"cylinder": rmse_flat}
    if improvement_linear >= MODEL_MIN_RELATIVE_IMPROVEMENT:
        candidates["cone"] = rmse_linear
    if improvement_dome >= MODEL_MIN_RELATIVE_IMPROVEMENT:
        candidates["sphere"] = best_rmse_dome
    best = min(candidates, key=candidates.get)

    debug = {
        "rmse_flat": rmse_flat / scale, "rmse_linear": rmse_linear / scale,
        "rmse_dome": best_rmse_dome / scale, "linear_slope": float(slope), "dome_R": best_R,
    }
    return best, debug, {"slope": float(slope), "dome_R": best_R}


def classify_shape(points: np.ndarray) -> ShapeResult:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 20:
        return ShapeResult(shape="unknown", debug={"reason": "too few points"})

    xy = points[:, :2]
    hull_info = _hull_analysis(xy)
    if hull_info is None:
        return ShapeResult(shape="unknown", debug={"reason": "degenerate footprint"})

    circularity = hull_info["circularity"]
    debug = {"circularity": circularity}

    if circularity >= CIRCULARITY_CIRCLE_MIN:
        centroid_xy = xy.mean(axis=0)
        z_top = float(points[:, 2].max())
        profile = _radial_profile(points, centroid_xy)
        if profile is None:
            return ShapeResult(shape="unknown", debug={**debug, "reason": "not enough radial spread"})
        r_mid, z_mean = profile
        # Drop the outermost bin before fitting: it's exactly where a small,
        # physically-real edge/rim sliver tends to show up (a hair of the
        # upper side wall, visible only because the camera ray isn't
        # perfectly vertical for off-center points -- see
        # synthetic_depth_camera.py). A single such bin, being at the extreme
        # of the radial range, has outsized leverage on a linear fit's slope
        # and can make "linear" look like a real improvement over "flat" even
        # when the object's actual visible cap is flat everywhere else.
        if len(r_mid) > 4:
            r_mid, z_mean = r_mid[:-1], z_mean[:-1]
        best, fit_debug, fit_params = _fit_radial_shape(r_mid, z_mean, z_top)
        debug.update(fit_debug)

        r_visible = float(np.linalg.norm(xy - centroid_xy, axis=1).max())
        if best == "cylinder":
            return ShapeResult(shape="cylinder", params={
                "center_xy": centroid_xy.tolist(), "z_top": z_top, "visible_radius": r_visible,
            }, debug=debug)
        elif best == "sphere":
            return ShapeResult(shape="sphere", params={
                "center_xy": centroid_xy.tolist(), "z_top": z_top, "radius_estimate": fit_params["dome_R"],
            }, debug=debug)
        else:
            return ShapeResult(shape="cone", params={
                "apex_xy": centroid_xy.tolist(), "z_top": z_top,
                "taper_slope": fit_params["slope"], "visible_radius": r_visible,
            }, debug=debug)

    # Non-circular: quad vs triangular footprint, via convex-hull vertex
    # voting (same technique as before -- circularity alone conflates an
    # elongated rectangle with a triangle).
    z_top = float(points[:, 2].max())
    w, h, footprint_center, yaw = _footprint_rect(points)
    vertex_count = hull_info["vertex_count"]
    debug["vertex_count"] = vertex_count
    debug["footprint_dims"] = [w, h]

    is_quad = vertex_count == 4 or (vertex_count not in (3, 4) and circularity >= CIRCULARITY_QUAD_MIN)

    if is_quad:
        # Cube vs. cuboid from footprint squareness ONLY -- true height isn't
        # knowable from one overhead view of an opaque box (see module
        # docstring). A short square-footprint box will read as "cube" here;
        # that's a real sensor limitation, not something this can fix.
        ratio = max(w, h) / max(min(w, h), 1e-9)
        shape = "cube" if ratio <= 1.25 else "cuboid"
        debug["footprint_aspect_ratio"] = ratio
        return ShapeResult(shape=shape, params={
            "center_xy": footprint_center.tolist(), "z_top": z_top,
            "footprint_dims": [w, h], "yaw": yaw,
        }, debug=debug)
    else:
        return ShapeResult(shape="prism", params={
            "center_xy": footprint_center.tolist(), "z_top": z_top,
            "footprint_dims": [w, h], "yaw": yaw,
        }, debug=debug)
