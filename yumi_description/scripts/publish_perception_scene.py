"""Publishes a synthetic table-top scene (one of each of the six sorting
primitives) through the REAL perception pipeline -- synthetic_shapes ->
synthetic_depth_camera (ONE simulated single-viewpoint capture of the WHOLE
table) -> per-object segmentation (connected components -- a stand-in for
GSAM2, which isn't wired into this ROS2 node) -> pcd_extractor ->
shape_classifier_single_view -> grasp_generator -- so it can be checked
visually in the same Foxglove session as the robot/table, without needing
the real RealSense or real objects yet.

ONE shared overhead camera, not one per object: positioned at the midpoint
of the two arms' own base attachment points (yumi_joint_1_r/l, computed from
the URDF chain: world -> yumi_base_link -> yumi_body -> joint origins =
(0.0536, 0, 0.5149)), raised a little and moved forward so it looks down at
the table instead of the robot's own torso -- (0.30, 0.0, 0.60), i.e. only
8.5cm above the shoulders and centered over the table's front region. Uses
realistic RealSense-D435-like depth FOV (87 x 58 degrees) rather than an
arbitrary placeholder, both because it's needed to actually cover the full
object row from this height and because the whole point here is simulating
the real setup, not just something convenient.

Since one shared shot means multiple objects in the same depth image, a
mask that isolates a single object (which GSAM2 would normally provide) has
to come from somewhere -- connected-component labeling on the combined
mask stands in for that here, since GSAM2 itself isn't wired into this
node (it needs its own heavy venv/torch). This is a stand-in for real
segmentation, not a replacement for it.

Published (all frame_id="world", matching publish_table_marker.py):
    /perception_scene/ground_truth   -- each object's full (360-degree)
                                         surface, faint -- what it actually
                                         is.
    /perception_scene/captured       -- the extracted single-viewpoint
                                         cloud per detected blob, bright
                                         orange -- what the pipeline
                                         actually sees and works from.
    /perception_scene/labels         -- text showing true shape ->
                                         classified shape per object.
    /perception_scene/camera_mounts  -- single marker at the one shared
                                         camera position.
    /perception_scene/grasps         -- magenta arrow (approach direction)
                                         + cyan line (jaw span) per object.
    /perception_scene/bins           -- the six sorting bins (orange=right
                                         arm's side, blue=left arm's side),
                                         labeled by which shape goes there.

No torch/curobo needed here -- plain numpy/scipy/cv2, all available in
ROS2's system Python -- so this runs directly, no cross-Python TCP bridge.

Usage:
    source /opt/ros/humble/setup.bash
    python3 yumi_description/scripts/publish_perception_scene.py
"""

import json
import math
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.ndimage import label as cc_label, binary_closing
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "perception"))

import synthetic_shapes as syn
from synthetic_depth_camera import cull_backfaces, project_to_depth_image
from pcd_extractor import extract_object_pointcloud
from shape_classifier_single_view import classify_shape
from grasp_generator import generate_grasp_single_view, top_down_tcp_pose

# Table surface is at world z=0.10 (yumi_description/scene/table.yml: world
# z=0 is where the robot touches the table, slab is 0.013 to 0.10 thick).
TABLE_TOP_Z = 0.10

# Every run's scene is randomized (see _build_scene) -- exported here so
# curobo_live.py (which can't import this pipeline directly:
# curobo/.venv has no cv2/pip, and CLAUDE.md says not to add packages there)
# can plan grasps for the SAME objects this node is showing in Foxglove,
# not an independently-randomized scene that wouldn't match what's on screen.
GRASP_EXPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "scene", "last_perception_scene.json")

# Real Intel RealSense D457 specs (Intel's own datasheet): depth FOV
# 87 x 58 degrees, native depth resolution 1280x720, min range 0.6m.
IMG_W, IMG_H = 1280, 720
FOV_H_DEG, FOV_V_DEG = 87.0, 58.0
FX = (IMG_W / 2) / math.tan(math.radians(FOV_H_DEG / 2))
FY = (IMG_H / 2) / math.tan(math.radians(FOV_V_DEG / 2))
CX, CY = IMG_W / 2, IMG_H / 2
D457_MIN_RANGE = 0.6

# One shared camera: midpoint of the two arm-base attachment points
# (yumi_joint_1_r/l in yumi.urdf, chained through to world), raised and
# moved forward from x=0.0536 to x=0.30 so it looks at the table. Height
# must clear the real D457's 0.6m minimum range from the CLOSEST thing it
# can see -- that's an object's TOP surface, not the table -- so it's not
# just "table + 0.6m": it's table + 0.6m + the tallest object's own height
# (~6cm here) + a margin, or every object would sit inside the sensor's
# dead zone and read as no depth at all (caught by testing: blobs had
# hundreds of mask pixels but almost no valid extracted points).
MAX_OBJECT_HEIGHT = 0.06
DEPTH_MARGIN = 0.04
CAMERA_WORLD = np.array([0.30, 0.0, TABLE_TOP_Z + D457_MIN_RANGE + MAX_OBJECT_HEIGHT + DEPTH_MARGIN])

MIN_BLOB_PIXELS = 200  # discard specks too small to be a real detection --
                        # real objects here run 900+ px; small stray
                        # fragments (a residual seam morphological closing
                        # doesn't fully bridge) run under 100px

SHAPES = ["sphere", "cylinder", "cone", "cube", "cuboid", "prism"]
OBJECTS_PER_SHAPE = 3  # multiple of each, so classification is actually
                        # exercised more than once per shape, not just
                        # confirmed on a single lucky instance. Was 4 (24
                        # total) when objects were scattered across the
                        # whole table; now that they're clustered centrally
                        # (see PLACEMENT_*_RANGE below) to leave the two
                        # side wings free for sorting bins, the available
                        # area is roughly half what it was, so object count
                        # is reduced to keep packing density safely under
                        # the ~55% random-circle-packing jamming limit
                        # (18 objects in the new area is ~35% -- comfortable
                        # margin for rejection sampling to actually succeed).

# Random scatter region: shrunk from the old table-spanning layout and
# pulled toward the center, specifically so it does NOT overlap the sorting
# bins (see BIN_POSITIONS below), which sit on the table's side wings at
# x < 0.144 -- since this region starts at x=0.18, there's no geometric
# overlap regardless of Y, but keeping Y narrower too (vs. the old +-0.35)
# is what actually makes the layout read as "objects clustered in the
# middle, bins flanking left and right" rather than objects sprawling all
# the way out toward the bins.
PLACEMENT_X_RANGE = (0.18, 0.50)
PLACEMENT_Y_RANGE = (-0.25, 0.25)
MIN_OBJECT_SEPARATION = 0.075  # still > any object's diameter (max ~6cm) with margin

# Six sorting bins on the table's two side wings (table_side_neg_y/pos_y in
# table.yml -- x in [-0.361, 0.1436], otherwise-unused table area). Right
# wing (y<0, the right arm's side) holds the three ROUND shapes; left wing
# holds the three BOX shapes -- same round-vs-box grouping grasp_generator.py
# already uses for its two different grasp strategies. An arm can only
# reach bins on its own side (see yumi_description/scene/bins.yml for the
# exact collision geometry curobo plans around -- keep these positions in
# sync with that file). "arm" here means which arm can physically reach
# that bin, used by curobo_live.py to decide whether an
# object needs a same-side delivery or a cross-arm handoff.
BIN_POSITIONS = {
    "sphere":   {"xy": (-0.05, -0.30), "arm": "right"},
    "cylinder": {"xy": (-0.05, -0.42), "arm": "right"},
    "cone":     {"xy": (-0.05, -0.54), "arm": "right"},
    "cube":     {"xy": (-0.05, 0.30), "arm": "left"},
    "cuboid":   {"xy": (-0.05, 0.42), "arm": "left"},
    "prism":    {"xy": (-0.05, 0.54), "arm": "left"},
}
BIN_TOP_Z = 0.15  # bin bottom at TABLE_TOP_Z, 5cm tall (bins.yml)
BIN_RELEASE_HOVER = 0.05  # release point this far above the bin's rim, so
                           # the object actually falls IN rather than being
                           # released right at/inside the solid collision box

# A single shared point near the table's front-center, reachable by both
# arms without either needing an extreme reach, used to hand an object from
# the arm that picked it up to the arm whose side actually has its bin.
HANDOFF_POINT_XY = (0.15, 0.0)
# Must clear grasp_generator.PINCH_HEIGHT_FLOOR (table + 3.2cm) -- learned
# the hard way earlier this session that anything less lets the fingertip
# sphere's own real reach dip into the table (confirmed there via direct
# per-sphere collision checking). This is a generic placement, not a
# precise per-object pinch height, so there's no reason to cut it close;
# well above the floor with real margin instead of hugging it.
HANDOFF_RELEASE_Z = TABLE_TOP_Z + 0.05


def scatter_positions(n, rng, max_attempts_per_point=1000):
    """Rejection-sampled (x, y) positions, each at least MIN_OBJECT_SEPARATION
    from every already-placed one -- objects placed like they'd actually
    land on a table (scattered, not lined up), never overlapping."""
    positions = []
    for _ in range(n):
        for _ in range(max_attempts_per_point):
            x = rng.uniform(*PLACEMENT_X_RANGE)
            y = rng.uniform(*PLACEMENT_Y_RANGE)
            if all(math.hypot(x - px, y - py) >= MIN_OBJECT_SEPARATION for px, py in positions):
                positions.append((x, y))
                break
        else:
            positions.append((x, y))  # last attempt, accepted anyway -- rare
    return positions


# Higher than synthetic_shapes' own default (2000) -- with the camera now
# farther away (0.6m table clearance, matching the real D457's minimum
# range) and at native 1280x720 resolution, each object's silhouette covers
# far more pixels than in the earlier close-up per-object-camera version, so
# the old point count left gaps between individual "hit" pixels and
# fragmented each object's mask into hundreds of one-pixel specks instead of
# one solid blob. A real depth camera has no such gap (every pixel gets a
# real depth reading); this compensates for that synthetic-only artifact.
N_POINTS = 15000


def make_shape(shape_name, rng, center_xy, yaw):
    if shape_name == "sphere":
        return syn.sphere(radius=0.025, center_xy=center_xy, n=N_POINTS, noise_frac=0.01, rng=rng)
    if shape_name == "cylinder":
        return syn.cylinder(radius=0.018, height=0.06, center_xy=center_xy, n=N_POINTS, noise_frac=0.01, rng=rng)
    if shape_name == "cone":
        return syn.cone(base_radius=0.025, height=0.05, apex_up=True, center_xy=center_xy,
                         n=N_POINTS, noise_frac=0.01, rng=rng)
    if shape_name == "cube":
        return syn.cube(side=0.035, center_xy=center_xy, yaw=yaw, n=N_POINTS, noise_frac=0.01, rng=rng)
    if shape_name == "cuboid":
        return syn.cuboid(l=0.045, w=0.025, h=0.03, center_xy=center_xy, yaw=yaw, n=N_POINTS, noise_frac=0.01, rng=rng)
    if shape_name == "prism":
        return syn.prism(side=0.04, height=0.05, center_xy=center_xy, yaw=yaw, n=N_POINTS, noise_frac=0.01, rng=rng)
    raise ValueError(shape_name)


def shared_camera_extrinsic():
    """4x4 camera-optical-frame -> world transform for CAMERA_WORLD (full
    xyz, not just height -- the camera isn't centered over world (0,0))."""
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    T[:3, 3] = CAMERA_WORLD
    return T


def points_to_msg_list(points_xyz):
    return [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points_xyz]


class PerceptionSceneNode(Node):
    def __init__(self):
        super().__init__("perception_scene_demo")
        self.gt_pub = self.create_publisher(MarkerArray, "/perception_scene/ground_truth", 10)
        self.cap_pub = self.create_publisher(MarkerArray, "/perception_scene/captured", 10)
        self.label_pub = self.create_publisher(MarkerArray, "/perception_scene/labels", 10)
        self.cam_pub = self.create_publisher(MarkerArray, "/perception_scene/camera_mounts", 10)
        self.grasp_pub = self.create_publisher(MarkerArray, "/perception_scene/grasps", 10)
        self.bin_pub = self.create_publisher(MarkerArray, "/perception_scene/bins", 10)

        # Clear any stale markers left behind by a previous run of this script
        # with a different object count (MarkerArray doesn't auto-remove IDs
        # that simply stop being published -- they just freeze in place;
        # DELETEALL guarantees a clean slate regardless of run history).
        delete_all = Marker(action=Marker.DELETEALL)
        for pub in (self.gt_pub, self.cap_pub, self.label_pub, self.cam_pub, self.grasp_pub, self.bin_pub):
            pub.publish(MarkerArray(markers=[delete_all]))

        # Bin drop poses and the shared handoff pose are fixed (not
        # per-object), computed once here the same way every pick grasp
        # computes its TCP target -- top-down, GRIPPER_TIP_OFFSET applied --
        # so curobo plans to them the exact same way.
        self.bin_poses = {}
        for shape, spec in BIN_POSITIONS.items():
            bx, by = spec["xy"]
            pos, quat = top_down_tcp_pose((bx, by, BIN_TOP_Z + BIN_RELEASE_HOVER))
            self.bin_poses[shape] = {"position": pos, "quaternion_wxyz": quat, "arm": spec["arm"]}
        hx, hy = HANDOFF_POINT_XY
        self.handoff_position, self.handoff_quaternion_wxyz = top_down_tcp_pose(
            (hx, hy, HANDOFF_RELEASE_Z))

        self.get_logger().info(
            f"Shared camera at world {CAMERA_WORLD.tolist()} "
            f"({CAMERA_WORLD[2] - TABLE_TOP_Z:.2f}m above the table), "
            f"FOV {FOV_H_DEG:.0f}x{FOV_V_DEG:.0f} deg -- building the scene and "
            f"running one combined capture through the perception pipeline ..."
        )
        n_expected = len(SHAPES) * OBJECTS_PER_SHAPE
        self.scene = self._build_scene()
        correct = 0
        for obj in self.scene:
            ok = obj["classified"] == obj["true_shape"]
            correct += int(ok)
            self.get_logger().info(f"  {obj['true_shape']:10s} -> {obj['classified']:10s} "
                                    f"[{'OK' if ok else 'MISCLASSIFIED'}]")
        n_detected = len(self.scene)
        self.get_logger().info(f"{correct}/{n_detected} correctly classified "
                                f"({n_detected}/{n_expected} objects detected as separate blobs)")
        if n_detected < n_expected:
            self.get_logger().warn(f"only {n_detected}/{n_expected} objects were detected as "
                                    f"separate blobs in the combined capture")

        self._export_for_curobo()

        self.create_timer(1.0, self.tick)

    def _export_for_curobo(self):
        """Writes this run's graspable objects (shape, arm, pose) to
        GRASP_EXPORT_PATH, so curobo_live.py can plan for
        the exact same randomly-placed scene shown here, not a separately
        (and differently) randomized one."""
        objects = []
        for obj in self.scene:
            if obj["grasp_world"] is None:
                continue
            g = obj["grasp_world"]
            classified = obj["classified"]
            # Sort by what the pipeline BELIEVES the shape is, same as the
            # grasp itself -- a real system has no other information to sort
            # by. bin_pose may be None if classify_shape ever returns
            # "unknown" reaching here (shouldn't -- grasp generation already
            # requires a known shape -- kept defensive rather than assumed).
            bin_pose = self.bin_poses.get(classified)
            objects.append({
                "shape": classified,
                "true_shape": obj["true_shape"],
                "arm": "right" if obj["world_y"] < 0 else "left",
                "position": g["position"],
                "quaternion_wxyz": g["quaternion_wxyz"],
                "grasp_width": g["grasp_width"],
                # True object pose (ground truth, not the pipeline's belief) --
                # only needed by the physics sim, to spawn the actual rigid
                # body it's trying to grasp.
                "true_world_x": obj["world_x"],
                "true_world_y": obj["world_y"],
                "true_yaw": obj["yaw"],
                # Where this object needs to end up: its classified shape's
                # bin (which arm can reach it may differ from the arm that
                # picks it up -- curobo_live.py handles the
                # handoff when it does).
                "bin_position": bin_pose["position"] if bin_pose else None,
                "bin_quaternion_wxyz": bin_pose["quaternion_wxyz"] if bin_pose else None,
                "bin_arm": bin_pose["arm"] if bin_pose else None,
            })
        export = {
            "objects": objects,
            "handoff_position": self.handoff_position,
            "handoff_quaternion_wxyz": self.handoff_quaternion_wxyz,
        }
        with open(GRASP_EXPORT_PATH, "w") as f:
            json.dump(export, f, indent=2)
        self.get_logger().info(f"exported {len(objects)} graspable object(s) to {GRASP_EXPORT_PATH}")

    def _build_scene(self):
        rng = np.random.default_rng()  # no fixed seed -- a different scatter every run
        true_objects = []  # (shape_name, world_x, world_y, full_world_cloud)
        visible_points = []

        shape_list = SHAPES * OBJECTS_PER_SHAPE
        rng.shuffle(shape_list)  # scattered, not grouped by type
        positions = scatter_positions(len(shape_list), rng)

        camera_pos_local = np.array([0.0, 0.0, CAMERA_WORLD[2]])
        yaw_by_pos = {}
        for shape_name, (world_x, world_y) in zip(shape_list, positions):
            yaw = rng.uniform(0, 2 * math.pi)
            yaw_by_pos[(world_x, world_y)] = yaw
            full_world = make_shape(shape_name, rng, (world_x, world_y), yaw)
            full_world = full_world + np.array([0.0, 0.0, TABLE_TOP_Z])
            true_objects.append((shape_name, world_x, world_y, full_world))

            # Back-face-cull THIS OBJECT ALONE (its "away from centroid" sign
            # heuristic is only valid per-object -- see cull_backfaces): shift
            # into the camera-centered local frame, cull, then shift back.
            local = full_world - np.array([CAMERA_WORLD[0], CAMERA_WORLD[1], 0.0])
            visible_local = cull_backfaces(local, camera_pos_local)
            visible_points.append(visible_local)

        visible_points = np.concatenate(visible_points, axis=0)
        depth_img, mask_img = project_to_depth_image(
            visible_points, CAMERA_WORLD[2], FX, FY, CX, CY, IMG_W, IMG_H,
        )

        extrinsic = shared_camera_extrinsic()

        # Morphological closing: bridges thin gaps in a single object's mask
        # (seen on curved shapes -- sphere/cylinder's smooth back-face-culling
        # boundary can leave a narrow seam that splits one true object into
        # two blobs) before connected-component labeling. Compensates for a
        # sparse-point-rendering artifact; a real continuous depth image has
        # no such gaps to begin with.
        closed_mask = binary_closing(mask_img, structure=np.ones((7, 7)))
        labeled, n_blobs = cc_label(closed_mask, structure=np.ones((3, 3)))  # 8-connectivity
        self.get_logger().info(f"combined capture: {mask_img.sum()} depth pixels, "
                                f"{n_blobs} connected-component blob(s) found")

        scene = []
        for blob_id in range(1, n_blobs + 1):
            blob_mask = labeled == blob_id
            if blob_mask.sum() < MIN_BLOB_PIXELS:
                continue

            extracted_world = extract_object_pointcloud(
                depth_img, blob_mask, FX, FY, CX, CY, extrinsic=extrinsic, outlier_removal=True,
                min_depth=D457_MIN_RANGE,
            )
            if extracted_world.shape[0] == 0:
                continue

            result = classify_shape(extracted_world)
            grasp = None
            if result.shape != "unknown":
                try:
                    grasp = generate_grasp_single_view(result)
                except Exception as e:
                    self.get_logger().warn(f"grasp generation failed for blob {blob_id}: {e}")

            # Match this blob to the nearest true object by XY centroid --
            # only for ground-truth comparison/labeling in this synthetic
            # demo (a real system wouldn't have this available).
            blob_centroid_xy = extracted_world[:, :2].mean(axis=0)
            dists = [np.hypot(blob_centroid_xy[0] - ox, blob_centroid_xy[1] - oy)
                     for _, ox, oy, _ in true_objects]
            match_idx = int(np.argmin(dists))
            true_shape, world_x, world_y, full_world = true_objects[match_idx]

            grasp_world = None
            if grasp is not None:
                grasp_world = {
                    "position": grasp.position,  # already in world frame (extrinsic applied upstream)
                    "quaternion_wxyz": grasp.quaternion_wxyz,
                    "approach_axis": grasp.approach_axis,
                    "jaw_axis": grasp.jaw_axis,
                    "grasp_width": grasp.grasp_width,
                }

            scene.append({
                "true_shape": true_shape,
                "classified": result.shape,
                "full_world": full_world,
                "extracted_world": extracted_world,
                "grasp_world": grasp_world,
                "label_pos": (world_x, world_y, TABLE_TOP_Z + 0.10),
                "world_x": world_x,
                "world_y": world_y,
                "yaw": yaw_by_pos.get((world_x, world_y), 0.0),
            })
        return scene

    def tick(self):
        stamp = self.get_clock().now().to_msg()
        gt_array, cap_array = MarkerArray(), MarkerArray()
        label_array, cam_array, grasp_array = MarkerArray(), MarkerArray(), MarkerArray()

        cam = Marker()
        cam.header.frame_id = "world"; cam.header.stamp = stamp
        cam.ns = "camera_mounts"; cam.id = 0
        cam.type = Marker.CUBE; cam.action = Marker.ADD
        cam.pose.position.x, cam.pose.position.y, cam.pose.position.z = CAMERA_WORLD.tolist()
        cam.pose.orientation.w = 1.0
        cam.scale.x = cam.scale.y = cam.scale.z = 0.03
        cam.color.r, cam.color.g, cam.color.b, cam.color.a = 0.2, 0.2, 0.2, 0.9
        cam_array.markers.append(cam)

        for i, obj in enumerate(self.scene):
            gt = Marker()
            gt.header.frame_id = "world"; gt.header.stamp = stamp
            gt.ns = "ground_truth"; gt.id = i
            gt.type = Marker.POINTS; gt.action = Marker.ADD
            gt.points = points_to_msg_list(obj["full_world"])
            gt.scale.x = gt.scale.y = 0.0015
            gt.color.r, gt.color.g, gt.color.b, gt.color.a = 0.6, 0.6, 0.9, 0.25
            gt_array.markers.append(gt)

            cap = Marker()
            cap.header.frame_id = "world"; cap.header.stamp = stamp
            cap.ns = "captured"; cap.id = i
            cap.type = Marker.POINTS; cap.action = Marker.ADD
            cap.points = points_to_msg_list(obj["extracted_world"])
            cap.scale.x = cap.scale.y = 0.003
            cap.color.r, cap.color.g, cap.color.b, cap.color.a = 1.0, 0.55, 0.0, 0.9
            cap_array.markers.append(cap)

            correct = obj["classified"] == obj["true_shape"]
            label = Marker()
            label.header.frame_id = "world"; label.header.stamp = stamp
            label.ns = "labels"; label.id = i
            label.type = Marker.TEXT_VIEW_FACING; label.action = Marker.ADD
            lx, ly, lz = obj["label_pos"]
            label.pose.position.x, label.pose.position.y, label.pose.position.z = lx, ly, lz
            label.pose.orientation.w = 1.0
            label.scale.z = 0.02
            label.text = f"{obj['true_shape']} -> {obj['classified']} [{'OK' if correct else 'MISS'}]"
            label.color.r, label.color.g, label.color.b, label.color.a = \
                (0.2, 1.0, 0.2, 1.0) if correct else (1.0, 0.2, 0.2, 1.0)
            label_array.markers.append(label)

            if obj["grasp_world"] is not None:
                g = obj["grasp_world"]
                pos = np.array(g["position"])
                approach = np.array(g["approach_axis"])
                arrow_start, arrow_end = pos - approach * 0.04, pos

                arrow = Marker()
                arrow.header.frame_id = "world"; arrow.header.stamp = stamp
                arrow.ns = "grasp_approach"; arrow.id = i
                arrow.type = Marker.ARROW; arrow.action = Marker.ADD
                arrow.points = [Point(x=float(arrow_start[0]), y=float(arrow_start[1]), z=float(arrow_start[2])),
                                Point(x=float(arrow_end[0]), y=float(arrow_end[1]), z=float(arrow_end[2]))]
                arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.003, 0.006, 0.0
                arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 1.0, 0.0, 1.0, 1.0
                grasp_array.markers.append(arrow)

                jaw = np.array(g["jaw_axis"])
                half_w = g["grasp_width"] / 2
                p1, p2 = pos - jaw * half_w, pos + jaw * half_w
                jaw_line = Marker()
                jaw_line.header.frame_id = "world"; jaw_line.header.stamp = stamp
                jaw_line.ns = "grasp_jaw"; jaw_line.id = i
                jaw_line.type = Marker.LINE_LIST; jaw_line.action = Marker.ADD
                jaw_line.points = [Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
                                    Point(x=float(p2[0]), y=float(p2[1]), z=float(p2[2]))]
                jaw_line.scale.x = 0.003
                jaw_line.color.r, jaw_line.color.g, jaw_line.color.b, jaw_line.color.a = 0.0, 1.0, 1.0, 1.0
                grasp_array.markers.append(jaw_line)

        bin_array = MarkerArray()
        bin_colors = {"right": (0.9, 0.5, 0.1, 0.5), "left": (0.1, 0.5, 0.9, 0.5)}
        for i, (shape, spec) in enumerate(BIN_POSITIONS.items()):
            bx, by = spec["xy"]
            r, g, b, a = bin_colors[spec["arm"]]

            box = Marker()
            box.header.frame_id = "world"; box.header.stamp = stamp
            box.ns = "bins"; box.id = i
            box.type = Marker.CUBE; box.action = Marker.ADD
            box.pose.position.x, box.pose.position.y = bx, by
            box.pose.position.z = TABLE_TOP_Z + (BIN_TOP_Z - TABLE_TOP_Z) / 2
            box.pose.orientation.w = 1.0
            box.scale.x = box.scale.y = 0.08
            box.scale.z = BIN_TOP_Z - TABLE_TOP_Z
            box.color.r, box.color.g, box.color.b, box.color.a = r, g, b, a
            bin_array.markers.append(box)

            label = Marker()
            label.header.frame_id = "world"; label.header.stamp = stamp
            label.ns = "bin_labels"; label.id = i
            label.type = Marker.TEXT_VIEW_FACING; label.action = Marker.ADD
            label.pose.position.x, label.pose.position.y = bx, by
            label.pose.position.z = BIN_TOP_Z + 0.03
            label.pose.orientation.w = 1.0
            label.scale.z = 0.02
            label.text = shape
            label.color.r, label.color.g, label.color.b, label.color.a = r, g, b, 1.0
            bin_array.markers.append(label)

        self.gt_pub.publish(gt_array)
        self.cap_pub.publish(cap_array)
        self.label_pub.publish(label_array)
        self.cam_pub.publish(cam_array)
        self.grasp_pub.publish(grasp_array)
        self.bin_pub.publish(bin_array)


def main():
    rclpy.init()
    node = PerceptionSceneNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass  # already shut down (e.g. external SIGTERM raced our own call)


if __name__ == "__main__":
    main()
