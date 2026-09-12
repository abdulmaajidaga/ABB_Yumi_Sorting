"""Classifies an isolated object point cloud as one of six known primitives
-- sphere, cylinder, cone, cube, cuboid, prism (triangular) -- using only
geometry, no language/semantics. Meant to run on the point cloud GSAM2's mask
+ depth produces for one object.

Sphere, cylinder, cuboid and cone are classified by running pyRANSAC-3D's
real RANSAC fitters (cloned from GitHub at perception/pyRANSAC-3D -- the
published PyPI release doesn't include Cone) and taking whichever fits the
most points as inliers. Prism (triangular cross-section) has no fitter in
that library at all, so it's the fallback: if none of the four RANSAC models
explain the point cloud well, it's classified as a prism by elimination --
the only shape left in our six-shape universe.

Cube is not a separate RANSAC model -- it's a Cuboid fit whose three fitted
extents happen to be nearly equal.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
import pyransac3d as pyrsc

THRESH_FRAC = 0.03  # RANSAC inlier threshold, as a fraction of the object's
                     # bounding-box diagonal -- pyransac3d's own defaults are
                     # absolute (e.g. thresh=0.2), sized for meter-scale
                     # scenes, so they'd accept almost anything as an inlier
                     # for our centimeter-scale sorting objects.
MIN_INLIER_RATIO = 0.6  # below this for every covered shape -> prism
CUBE_ASPECT_RATIO_MAX = 1.25
CYLINDER_TIE_BREAK_MARGIN = 0.08  # sphere or cone must beat cylinder's ratio
                                   # by this much (relative) to win a close
                                   # call -- both have an extra free parameter
                                   # cylinder doesn't (sphere: no axis/length
                                   # constraint; cone: apex position), letting
                                   # either fit a truly cylindrical point cloud
                                   # almost as well on a short/dumpy cylinder
                                   # -- same overfitting pattern as a raw
                                   # linear-vs-constant fit.


@dataclass
class ShapeResult:
    shape: str  # "sphere" | "cylinder" | "cone" | "cube" | "cuboid" | "prism" | "unknown"
    params: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)


def _cuboid_full_inlier_ratio(points, center, extents, axes, thresh) -> float:
    """pyRANSAC-3D's own Cuboid.fit() only checks 3 of the box's 6 faces (it
    RANSACs 3 orthogonal INFINITE planes through one sampled corner, so the
    3 opposite faces -- parallel planes offset by the extent -- are never
    counted as inliers even on a perfect cuboid). That structurally caps its
    reported inlier ratio near 50%, which isn't comparable to Sphere/
    Cylinder/Cone (all of which check their full continuous surface). Re-score
    against the actual finite 6-face box, using the fit's own center/extents/
    axes, so all four candidates are compared on the same basis."""
    axes = np.asarray(axes, dtype=np.float64)
    extents = np.asarray(extents, dtype=np.float64)
    half = extents / 2
    local = (points - np.asarray(center, dtype=np.float64)) @ axes.T

    is_inlier = np.zeros(points.shape[0], dtype=bool)
    for i in range(3):
        j, k = [a for a in range(3) if a != i]
        near_face = np.abs(np.abs(local[:, i]) - half[i]) <= thresh
        within_j = np.abs(local[:, j]) <= half[j] + thresh
        within_k = np.abs(local[:, k]) <= half[k] + thresh
        is_inlier |= near_face & within_j & within_k
    return float(is_inlier.sum() / points.shape[0])


def classify_shape(points: np.ndarray) -> ShapeResult:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 30:
        return ShapeResult(shape="unknown", debug={"reason": "too few points"})

    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if bbox_diag < 1e-9:
        return ShapeResult(shape="unknown", debug={"reason": "degenerate size"})
    thresh = THRESH_FRAC * bbox_diag
    n = points.shape[0]

    sphere_res = pyrsc.Sphere().fit(points, thresh=thresh)
    cyl_res = pyrsc.Cylinder().fit(points, thresh=thresh)
    cuboid_res = pyrsc.Cuboid().fit(points, thresh=thresh)
    cone_res = pyrsc.Cone().fit(points, thresh=thresh)

    cuboid_ratio = (
        _cuboid_full_inlier_ratio(points, cuboid_res.center, cuboid_res.extents, cuboid_res.axes, thresh)
        if len(cuboid_res.center) else 0.0
    )

    candidates = {
        "sphere": (len(sphere_res.inliers) / n, sphere_res),
        "cylinder": (len(cyl_res.inliers) / n, cyl_res),
        "cuboid": (cuboid_ratio, cuboid_res),
        "cone": (len(cone_res.inliers) / n, cone_res),
    }

    # Sphere/cone vs cylinder tie-break: don't let either win on a wafer-thin
    # margin over cylinder, which has fewer free parameters than both.
    cyl_ratio = candidates["cylinder"][0]
    for name, res in (("sphere", sphere_res), ("cone", cone_res)):
        cand_ratio = candidates[name][0]
        if cand_ratio > cyl_ratio and cand_ratio < cyl_ratio * (1 + CYLINDER_TIE_BREAK_MARGIN):
            candidates[name] = (cyl_ratio - 1e-9, res)  # ensures cylinder wins the argmax

    best_shape = max(candidates, key=lambda k: candidates[k][0])
    best_ratio, best_res = candidates[best_shape]

    debug = {
        "inlier_ratios": {
            "sphere": round(candidates["sphere"][0], 3),
            "cylinder": round(candidates["cylinder"][0], 3),
            "cuboid": round(candidates["cuboid"][0], 3),
            "cone": round(len(cone_res.inliers) / n, 3),
        },
        "thresh": thresh,
    }

    if best_ratio < MIN_INLIER_RATIO:
        # No RANSAC model explains this well -- prism by elimination (the one
        # shape the library can't fit). Still extract real geometry for it
        # (height + an oriented bounding footprint), the same way cuboid does,
        # so grasp planning has something to work with beyond a bare centroid.
        z = points[:, 2]
        height = float(z.max() - z.min())
        xy = points[:, :2].astype(np.float32)
        (cx, cy), (w, h), angle_deg = cv2.minAreaRect(xy)
        yaw = float(np.deg2rad(angle_deg))
        centroid = np.array([cx, cy, float((z.max() + z.min()) / 2)])
        debug["footprint_dims"] = [float(w), float(h)]
        return ShapeResult(shape="prism", params={
            "center": centroid.tolist(), "height": height,
            "footprint_dims": [float(w), float(h)], "yaw": yaw,
        }, debug=debug)

    if best_shape == "sphere":
        return ShapeResult(shape="sphere", params={
            "center": list(sphere_res.center), "radius": float(sphere_res.radius),
        }, debug=debug)

    if best_shape == "cylinder":
        axis = np.asarray(cyl_res.axis, dtype=np.float64)
        axis /= np.linalg.norm(axis)
        # cyl_res.center is just SOME point on the infinite axis line, not
        # necessarily anywhere near the finite cylinder's actual midpoint --
        # recompute the true center as the midpoint of the inliers' extent
        # along the axis, anchored relative to that reference point.
        ref = np.asarray(cyl_res.center, dtype=np.float64)
        proj = (points[cyl_res.inliers] - ref) @ axis
        mid_offset = (proj.max() + proj.min()) / 2
        true_center = ref + axis * mid_offset
        height = float(proj.max() - proj.min())
        return ShapeResult(shape="cylinder", params={
            "center": true_center.tolist(), "axis": axis.tolist(),
            "radius": float(cyl_res.radius), "height": height,
        }, debug=debug)

    if best_shape == "cone":
        axis = np.asarray(cone_res.axis, dtype=np.float64)
        axis /= np.linalg.norm(axis)
        apex = np.asarray(cone_res.apex, dtype=np.float64)
        proj = (points[cone_res.inliers] - apex) @ axis
        height = float(proj.max() - proj.min())
        base_radius = float(np.tan(cone_res.angle) * height)
        return ShapeResult(shape="cone", params={
            "apex": apex.tolist(), "axis": axis.tolist(),
            "angle": float(cone_res.angle), "height": height, "base_radius": base_radius,
        }, debug=debug)

    # cuboid -- keep extents[i] paired with axes[i] (the RANSAC fit's own
    # order) so downstream grasp planning knows which physical face each
    # extent belongs to. The cube/cuboid aspect-ratio check only needs a
    # sorted COPY, not a reordering of what's actually returned.
    extents = [float(e) for e in cuboid_res.extents]
    ratio = max(extents) / max(min(extents), 1e-9)
    shape = "cube" if ratio <= CUBE_ASPECT_RATIO_MAX else "cuboid"
    debug["extents"] = extents
    debug["aspect_ratio"] = ratio
    return ShapeResult(shape=shape, params={
        "center": list(cuboid_res.center), "extents": extents,
        "axes": [list(a) for a in cuboid_res.axes],
    }, debug=debug)
