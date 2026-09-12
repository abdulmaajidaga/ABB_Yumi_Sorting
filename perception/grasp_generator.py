"""Analytic grasp pose generator: takes a ShapeResult (shape type + fitted
geometry, from either shape_classifier.py/shape_classifier_geometric.py --
full 3D geometry -- or shape_classifier_single_view.py -- what a single
overhead capture actually sees) and produces a parallel-jaw grasp pose -- no
learned model, since these are known regular primitives with closed-form
grasp geometry.

Two entry points, because the two kinds of classifier genuinely know
different things:
  generate_grasp_full3d(result)    -- for shape_classifier.py /
                                       shape_classifier_geometric.py results:
                                       true 3D center, axis, height, extents.
  generate_grasp_single_view(result) -- for shape_classifier_single_view.py
                                       results: only what's visible from
                                       directly above -- footprint, the
                                       visible cap's topmost point, and (for
                                       round shapes) a radius/taper estimate.
                                       True object height is NOT knowable
                                       this way (the bottom is never visible
                                       to an overhead camera), so a few grasp
                                       depths here are necessarily a
                                       conservative assumption, not a
                                       measurement -- flagged explicitly
                                       below, not hidden.

Since shape_classifier_single_view.py is the one validated against realistic
single-viewpoint occlusion (see test_pcd_extractor.py / test_single_view_
classifier.py), generate_grasp_single_view is what the real pipeline
actually uses; generate_grasp_full3d stays for the other two classifiers'
own test/comparison use.

Gripper frame convention (verified against yumi_description/urdf/yumi.urdf,
not assumed): gripper_r_base's/gripper_l_base's own local +Z is the approach
direction (both finger links sit forward along +Z from the base), and local
+X is the jaw open/close axis (both prismatic finger joints move symmetrically
along X). A GraspPose's `quaternion_wxyz` rotates that local frame so +Z
points along `approach_axis` and +X points along `jaw_axis`, both given in
whatever frame the input point cloud was in (the robot base frame, once the
PCD extractor's `extrinsic` argument is set from real camera calibration).

Per-shape strategy (full3d):
  sphere:          grab at the equator (widest cross-section), any horizontal
                   jaw direction -- rotationally symmetric.
  cylinder:        grab at mid-length, jaw perpendicular to the axis. Approach
                   top-down when the axis is mostly horizontal (lying down);
                   side-on when the axis is mostly vertical (standing) --
                   top-down on a standing cylinder would align the jaws with
                   the flat end cap, not the round side.
  cone:            same idea as cylinder, but grasped a bit in from the base
                   (not at the very rim, for table clearance) where the
                   locally tapered radius is used for the jaw width.
  cube / cuboid:   pinch across the shortest extent; approach along whichever
                   of the other two box axes is closest to vertical.
  prism:           pinch across the shorter side of the oriented bounding
                   rectangle of its (triangular) footprint, top-down.

Per-shape strategy (single_view) -- see generate_grasp_single_view for the
"why" behind each, but the short version: the classifier only ever reports
"cylinder"/"cone" when the visible silhouette is circular, which itself only
happens for a standing (vertical-axis) object -- so there's no more
standing-vs-lying ambiguity to resolve here, unlike full3d:
  sphere:          top-down, equator height = z_top - radius_estimate (the
                   dome fit's own radius IS a real measurement, not an
                   assumption).
  cylinder:        top-down too -- descend past the flat top and pinch
                   around the round side (like picking up a cup), same idea
                   as the sphere above, at a conservative fixed depth below
                   z_top (true mid-height isn't knowable) -- jaw width uses
                   visible_radius, which IS a real measurement (a standing
                   cylinder's visible top disk radius equals its true one).
                   A side-on (horizontal) approach was tried first and is
                   kinematically valid, but needs real vertical clearance
                   above the table (confirmed via direct IK testing) that a
                   short object's own height may not provide -- top-down has
                   no such requirement.
  cone:            top-down, same idea, conservative fixed depth below the
                   apex, jaw width from the fitted taper_slope at that depth.
  cube / cuboid / prism: top-down, footprint_dims/yaw are real measurements
                   (a silhouette is never occluded), grasped just below
                   z_top since true height isn't needed for a top-down pinch.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

WORLD_Z = np.array([0.0, 0.0, 1.0])
CYLINDER_VERTICAL_THRESH = 0.7  # |axis . world_Z| above this -> treat as standing
CONE_GRASP_HEIGHT_FRAC = 0.2    # fraction of height in from the base, for clearance

# single_view-specific: true object height is never knowable from one
# overhead capture (the bottom is always occluded), so these are deliberate,
# conservative assumptions about how far real material extends below the
# visible top -- tune against real object sizes once the camera is in hand.
SINGLE_VIEW_ROUND_GRASP_DEPTH = 0.015   # cylinder/cone: how far below z_top/apex to
                                         # descend before pinching around the round side

# box/prism: how far below z_top to grasp. A box/prism's side is a flat
# vertical wall all the way down, unlike a sphere/cylinder/cone's curved
# side (which genuinely needs SINGLE_VIEW_ROUND_GRASP_DEPTH of descent to
# reach its true max-width equator) -- so there's no grasp-quality reason
# this needs to be deep. Confirmed by direct physics testing (see
# physics/run_physics_grasp.py and GRIPPER_TIP_OFFSET's comment below) that
# a short object leaves very little table clearance for the FINGERTIPS
# specifically (their own physical reach extends ~2.7cm past the intended
# pinch point) -- shrinking this from the old 0.010 buys that clearance
# back with zero cost to a box/prism grasp's quality. Still not sufficient
# on its own for every object, though -- see PINCH_HEIGHT_FLOOR below, which
# is what actually guarantees no table collision regardless of object
# height, rather than this fixed number alone.
SINGLE_VIEW_TOP_DOWN_CLEARANCE = 0.003

# world Z of the table surface (must match yumi_description/scene/table.yml
# and publish_perception_scene.py's own TABLE_TOP_Z -- there's no single
# shared source of truth for it across the ROS2/system-Python and
# perception-package boundary, so keep these in sync by hand if the table
# ever moves).
TABLE_TOP_Z = 0.10

# Hard floor for a top-down pinch height, regardless of what
# SINGLE_VIEW_TOP_DOWN_CLEARANCE would otherwise compute: guarantees the
# fingertips clear the table even for an object shorter than
# SINGLE_VIEW_TOP_DOWN_CLEARANCE would safely allow, rather than relying on
# a single fixed clearance number to happen to be small enough for every
# object size. Derived from real collision-sphere geometry (yumi.yml), the
# same way GRIPPER_TIP_OFFSET below was: the deepest-reaching fingertip
# sphere extends ~20.9mm past the TCP's nominal aim point along the
# approach axis (its own local reach of ~0.1299m vs GRIPPER_TIP_OFFSET's
# 0.109m), and that sphere's own radius is ~8.8mm -- so the aim point
# itself must stay at least 20.9+8.8=29.7mm above the table for the
# fingertip to clear it, plus a 2mm safety margin against perception noise
# in the measured table height. A box/prism grasp's pinch height is
# max(z_top - SINGLE_VIEW_TOP_DOWN_CLEARANCE, this floor) -- so very short
# objects grasp closer to their own top (still fine, flat sides don't care)
# instead of colliding.
PINCH_HEIGHT_FLOOR = TABLE_TOP_Z + 0.032

# The gripper's TCP frame (gripper_r_base/gripper_l_base -- what curobo's
# GoalToolPose actually targets) sits well BEHIND where the fingers really
# reach. This used to be 0.15, chosen only by checking curobo didn't report
# a table collision (0.10 collided, 0.15 didn't) -- that only tests whether
# the gripper BASE mesh clears the table, and says nothing about whether the
# fingertips end up at the intended pinch height. Running a real grasp
# through PyBullet physics (physics/run_physics_grasp.py) caught the gap:
# fingers closed on thin air just above the object, 0/0 contact. Re-derived
# from the actual geometry instead -- from yumi.urdf, the finger-joint
# origin sits at local z=0.0837 from gripper_r_base, and finger.stl's own
# mesh (measured directly: bounding box z in [-0.002, 0.05216] in the
# finger link's own frame) extends the real contact PAD from z=0.0817 to
# z=0.1359 below that. 0.15 falls entirely past the far end of that pad --
# guaranteed to miss. This is the midpoint of that pad's actual reach
# (0.0817 + 0.1359) / 2, so a pinch-height estimate can be off by ~2.7cm in
# either direction (typical perception noise) and the pad still makes
# contact, instead of needing to land within a hairline of one exact value.
GRIPPER_TIP_OFFSET = 0.109


@dataclass
class GraspPose:
    position: list
    approach_axis: list  # gripper local +Z in the input frame
    jaw_axis: list        # gripper local +X in the input frame
    grasp_width: float
    quaternion_wxyz: list
    debug: dict


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("cannot normalize a near-zero vector")
    return v / n


def _build_grasp(position, approach_axis, jaw_axis, grasp_width, debug=None) -> GraspPose:
    """`position` is the intended PHYSICAL pinch point (where the fingers
    should actually close). The returned GraspPose.position is the TCP
    target curobo should actually plan to -- offset back along -approach_axis
    by GRIPPER_TIP_OFFSET, since the TCP frame sits well behind the fingers."""
    z_local = _normalize(approach_axis)
    x_raw = _normalize(jaw_axis)
    # Gram-Schmidt: make jaw_axis exactly perpendicular to approach_axis (they
    # may not be exactly orthogonal yet, e.g. from RANSAC noise).
    x_local = _normalize(x_raw - np.dot(x_raw, z_local) * z_local)
    y_local = np.cross(z_local, x_local)

    rot_matrix = np.column_stack([x_local, y_local, z_local])
    quat_xyzw = Rotation.from_matrix(rot_matrix).as_quat()
    quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]

    tcp_position = np.asarray(position, dtype=np.float64) - z_local * GRIPPER_TIP_OFFSET

    return GraspPose(
        position=tcp_position.tolist(),
        approach_axis=z_local.tolist(),
        jaw_axis=x_local.tolist(),
        grasp_width=float(grasp_width),
        quaternion_wxyz=quat_wxyz,
        debug=debug or {},
    )


def top_down_tcp_pose(release_point, jaw_axis=(1.0, 0.0, 0.0)):
    """TCP position + quaternion for hovering straight above `release_point`
    (world xyz) with the gripper pointing straight down -- same top-down
    convention and GRIPPER_TIP_OFFSET correction every pick grasp in this
    file uses, but for points that aren't a classified object (a bin's drop
    point, a handoff point): no grasp_width/pinch involved, so `jaw_axis` is
    arbitrary (only the approach direction matters for hovering/releasing).
    Returns (position: list[3], quaternion_wxyz: list[4])."""
    grasp = _build_grasp(release_point, -WORLD_Z, np.asarray(jaw_axis, dtype=np.float64),
                          grasp_width=0.0, debug={"mode": "top_down_release"})
    return grasp.position, grasp.quaternion_wxyz


def _cylindrical_grasp(center, axis, radius, note=""):
    """Shared logic for cylinder and cone-at-a-given-height: grab
    perpendicular to `axis`, top-down if possible, side-on if the axis is
    too close to vertical for that to make sense."""
    axis = _normalize(axis)
    if abs(np.dot(axis, WORLD_Z)) < CYLINDER_VERTICAL_THRESH:
        # Mostly horizontal (lying down) -- top-down approach works, jaws
        # close perpendicular to both the approach direction and the axis.
        approach = -WORLD_Z
        jaw = np.cross(approach, axis)
        mode = "top_down"
    else:
        # Mostly vertical (standing) -- top-down would align jaws with the
        # flat end cap instead of the round side, so approach from the side.
        candidate = np.cross(axis, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(candidate) < 1e-6:
            candidate = np.cross(axis, np.array([0.0, 1.0, 0.0]))
        approach = _normalize(candidate)
        jaw = np.cross(approach, axis)
        mode = "side_on"
    return _build_grasp(center, approach, jaw, 2 * radius, debug={"mode": mode, "note": note})


def generate_grasp_full3d(result) -> GraspPose:
    """For shape_classifier.py / shape_classifier_geometric.py results (true
    3D geometry: center, axis, height, extents)."""
    shape = result.shape
    p = result.params

    if shape == "sphere":
        center = np.asarray(p["center"], dtype=np.float64)
        radius = p["radius"]
        # Grab at the equator (widest cross-section); any horizontal jaw
        # direction works since a sphere is rotationally symmetric.
        return _build_grasp(center, -WORLD_Z, np.array([1.0, 0.0, 0.0]), 2 * radius,
                             debug={"mode": "equator"})

    if shape == "cylinder":
        center = np.asarray(p["center"], dtype=np.float64)
        return _cylindrical_grasp(center, p["axis"], p["radius"])

    if shape == "cone":
        apex = np.asarray(p["apex"], dtype=np.float64)
        axis = _normalize(p["axis"])  # apex -> base
        height = p["height"]
        base_radius = p["base_radius"]
        base_center = apex + axis * height
        grasp_center = base_center - axis * (CONE_GRASP_HEIGHT_FRAC * height)
        radius_here = base_radius * (1 - CONE_GRASP_HEIGHT_FRAC)
        return _cylindrical_grasp(grasp_center, axis, radius_here,
                                   note=f"grasped {CONE_GRASP_HEIGHT_FRAC:.0%} in from base")

    if shape in ("cube", "cuboid"):
        center = np.asarray(p["center"], dtype=np.float64)
        extents = np.asarray(p["extents"], dtype=np.float64)
        axes = np.asarray(p["axes"], dtype=np.float64)

        idx_min = int(np.argmin(extents))
        jaw_axis = axes[idx_min]
        grasp_width = float(extents[idx_min])

        remaining = [i for i in range(3) if i != idx_min]
        alignments = [abs(float(np.dot(axes[i], WORLD_Z))) for i in remaining]
        approach_idx = remaining[int(np.argmax(alignments))]
        approach_raw = axes[approach_idx]
        # Point the approach axis downward (matches the top-down convention
        # -- gripper local +Z should have a non-positive world-Z component).
        approach_axis = approach_raw if np.dot(approach_raw, WORLD_Z) <= 0 else -approach_raw

        return _build_grasp(center, approach_axis, jaw_axis, grasp_width,
                             debug={"pinched_extent_index": idx_min})

    if shape == "prism":
        center = np.asarray(p["center"], dtype=np.float64)
        w, h = p["footprint_dims"]
        yaw = p["yaw"]
        dir_w = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        dir_h = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        jaw_axis, grasp_width = (dir_w, w) if w <= h else (dir_h, h)
        return _build_grasp(center, -WORLD_Z, jaw_axis, grasp_width, debug={"mode": "bounding_rect_approx"})

    raise ValueError(f"no grasp strategy for shape {shape!r}")


def _footprint_dir(yaw):
    dir_w = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    dir_h = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    return dir_w, dir_h


def generate_grasp_single_view(result) -> GraspPose:
    """For shape_classifier_single_view.py results: only what's actually
    visible from one overhead capture (footprint, the visible cap's own top
    point, and -- for round shapes -- a radius/taper estimate). True object
    height is never available (the bottom is always occluded), so cylinder/
    cone grasp depth is a conservative fixed assumption
    (SINGLE_VIEW_ROUND_GRASP_DEPTH / SINGLE_VIEW_TOP_DOWN_CLEARANCE), not a
    measurement -- see the module docstring."""
    shape = result.shape
    p = result.params

    if shape == "sphere":
        cx, cy = p["center_xy"]
        z_top = p["z_top"]
        radius = p["radius_estimate"]  # a real measurement from the dome fit
        center = np.array([cx, cy, z_top - radius])  # equator height
        return _build_grasp(center, -WORLD_Z, np.array([1.0, 0.0, 0.0]), 2 * radius,
                             debug={"mode": "equator"})

    if shape == "cylinder":
        cx, cy = p["center_xy"]
        z_top = p["z_top"]
        radius = p["visible_radius"]  # real: a standing cylinder's visible top
                                       # disk radius equals its true radius
        # Top-down, same trick as the sphere's "equator" grasp: descend PAST
        # the flat top disk by SINGLE_VIEW_ROUND_GRASP_DEPTH and pinch around
        # the round SIDE from above (like picking up a cup or can), rather
        # than a horizontal side-on approach. A side-on approach was tried
        # first and does work kinematically, but needs the wrist swinging in
        # roughly horizontally, which requires real vertical clearance above
        # the table (~7cm, confirmed by direct IK collision testing) --
        # clearance a short object's own body may not provide. Top-down has
        # no such requirement (proven by every other shape in this file
        # grasping only ~1cm below their own top with no issue), so it's the
        # more broadly reliable default regardless of object height.
        grasp_z = z_top - SINGLE_VIEW_ROUND_GRASP_DEPTH
        center = np.array([cx, cy, grasp_z])
        approach = -WORLD_Z
        jaw = np.array([1.0, 0.0, 0.0])  # arbitrary -- radially symmetric
        return _build_grasp(center, approach, jaw, 2 * radius,
                             debug={"mode": "top_down", "assumed_depth_below_top": SINGLE_VIEW_ROUND_GRASP_DEPTH})

    if shape == "cone":
        ax_, ay = p["apex_xy"]
        z_top = p["z_top"]  # the apex itself
        slope = p["taper_slope"]  # drop per unit radius, from the radial fit
        depth = SINGLE_VIEW_ROUND_GRASP_DEPTH
        radius_here = depth / max(slope, 1e-6)
        center = np.array([ax_, ay, z_top - depth])
        approach = -WORLD_Z  # top-down, same reasoning as cylinder above
        jaw = np.array([1.0, 0.0, 0.0])
        return _build_grasp(center, approach, jaw, 2 * radius_here,
                             debug={"mode": "top_down", "assumed_depth_below_apex": depth})

    if shape in ("cube", "cuboid", "prism"):
        cx, cy = p["center_xy"]
        z_top = p["z_top"]
        w, h = p["footprint_dims"]
        yaw = p["yaw"]
        dir_w, dir_h = _footprint_dir(yaw)
        jaw_axis, grasp_width = (dir_w, w) if w <= h else (dir_h, h)
        grasp_z = max(z_top - SINGLE_VIEW_TOP_DOWN_CLEARANCE, PINCH_HEIGHT_FLOOR)
        center = np.array([cx, cy, grasp_z])
        return _build_grasp(center, -WORLD_Z, jaw_axis, grasp_width,
                             debug={"mode": "top_down",
                                    "assumed_depth_below_top": z_top - grasp_z,
                                    "clamped_to_floor": grasp_z == PINCH_HEIGHT_FLOOR})

    raise ValueError(f"no grasp strategy for shape {shape!r}")
