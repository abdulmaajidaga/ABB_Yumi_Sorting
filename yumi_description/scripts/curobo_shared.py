"""Shared, LIGHTWEIGHT pieces of the sorting pipeline -- paths, home poses,
task-list construction, goal-building, GPU warmup -- used by both
curobo_live.py (the live Foxglove demo) and
curobo_all.py (the headless full-sequence export for
physics/physics_all.py). Kept as its own module (rather than importing
curobo_live.py's copy) just for tidiness -- no need for
the export script to pull in that script's relay/socket/self-collision-
display-only code to reuse this part.

See warm_gpu() below for a real, confirmed-necessary fix: on this machine
the GPU idles at its lowest power state (210MHz, vs a 3105MHz max) between
bursts of work, and the driver is slow to ramp it back up for curobo's
own short, latency-sensitive CUDA graph calls -- caught directly by
comparing `nvidia-smi` clocks during a run that was stuck for 10+ minutes
on a single pose (210MHz, P8, ~15% CPU -- mostly idle, not computing) against
the same exact pose immediately after a few seconds of unrelated heavy
GPU work (2355MHz, P3 -- and the pose then planned in 0.16s). A plain
matmul loop forces the ramp-up well before curobo's own first real call
needs it.
"""

import json
from pathlib import Path

REPO_ROOT = Path("/home/abdul/Documents/yumi")
DESC_ROOT = REPO_ROOT / "yumi_description"
YML_PATH = DESC_ROOT / "yumi.yml"
URDF_PATH = DESC_ROOT / "urdf" / "yumi.urdf"
SCENE_PATH = DESC_ROOT / "scene" / "table.yml"
BINS_SCENE_PATH = DESC_ROOT / "scene" / "bins.yml"
PERCEPTION_EXPORT_PATH = DESC_ROOT / "scene" / "last_perception_scene.json"

DEVICE = "cuda:0"


def warm_gpu(seconds=4.0):
    """Forces the GPU out of its idle low-power state (confirmed on this
    machine: 210MHz idle vs 3105MHz max) BEFORE building a MotionPlanner --
    see this module's own docstring for how big a difference this makes
    (10+ minutes stuck vs 0.16s, for the identical pose). Call this first,
    before any curobo imports/objects are created."""
    import time
    import torch
    x = torch.randn(4096, 4096, device=DEVICE)
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        x = x @ x.T
        x = x / x.norm()
    torch.cuda.synchronize()


# Right/left arm home-pose tool positions -- computed once via FK, used as
# a "stay put" goal on tasks/legs where that arm has nothing to do.
RIGHT_HOME_POS = [-0.0098, -0.1824, 0.2970]
RIGHT_HOME_QUAT = [0.0665, -0.8423, -0.1112, -0.5233]  # wxyz
LEFT_HOME_POS = [-0.0103, 0.1830, 0.2975]
LEFT_HOME_QUAT = [0.0648, 0.8428, -0.1112, 0.5226]  # wxyz


def load_tasks():
    """Reads publish_perception_scene.py's export (each object already
    tagged with its pick pose, its classified shape's bin pose, and which
    arm can reach that bin) and turns it into a flat, ORDERED, single-arm-
    at-a-time task list:
      - same side (the arm that can reach the object is the same arm whose
        side has that shape's bin): one "deliver" task -- pick, carry
        straight to the bin, retreat.
      - cross side (e.g. a cube -- whose bin is on the left -- gets
        scattered on the right half of the table, so the right arm is the
        one that can actually reach it): two SEPARATE, sequential tasks --
        "handoff_out" (origin arm carries it to the shared handoff point
        and releases) then "handoff_in" (the bin's own arm picks it up
        from that same point and delivers it). These are two entries in
        the flat list, not one combined step, so the executing loop runs
        them one after another rather than trying to choreograph both arms
        at once near the same point.
    Every task moves exactly one arm; the other arm's target is simply its
    own home pose (unchanged) for that task's whole approach/carry/retreat
    sequence, so tasks can be executed strictly in order with no risk of
    the "idle" arm's resting pose being disturbed mid-task.
    """
    if not PERCEPTION_EXPORT_PATH.exists():
        raise SystemExit(
            f"{PERCEPTION_EXPORT_PATH} doesn't exist yet -- run "
            f"publish_perception_scene.py first (it writes this file as well "
            f"as showing the scene in Foxglove)."
        )
    with open(PERCEPTION_EXPORT_PATH) as f:
        data = json.load(f)

    handoff_pos = data["handoff_position"]
    handoff_quat = data["handoff_quaternion_wxyz"]
    objects = [o for o in data["objects"] if o.get("bin_position") is not None]
    print(f"Loaded {len(objects)} object(s) with a known destination bin "
          f"from {PERCEPTION_EXPORT_PATH}")

    tasks = []
    for obj_index, obj in enumerate(objects):
        origin_arm, bin_arm = obj["arm"], obj["bin_arm"]
        pick_pos, pick_quat = obj["position"], obj["quaternion_wxyz"]
        bin_pos, bin_quat = obj["bin_position"], obj["bin_quaternion_wxyz"]
        # obj_index identifies which physical object this task is about --
        # its index into this same filtered `objects` list -- so a
        # ground-truth-spawning consumer (physics/physics_all.py) can
        # match a task back to the exact rigid body it should grab.
        common = {"shape": obj["shape"], "obj_index": obj_index,
                  "true_shape": obj["true_shape"], "world_x": obj["true_world_x"],
                  "world_y": obj["true_world_y"], "yaw": obj["true_yaw"]}
        if origin_arm == bin_arm:
            tasks.append({
                **common, "kind": "deliver", "arm": origin_arm,
                "pick_pos": pick_pos, "pick_quat": pick_quat,
                "place_pos": bin_pos, "place_quat": bin_quat,
                "right_obj": obj if origin_arm == "right" else None,
            })
        else:
            tasks.append({
                **common, "kind": "handoff_out", "arm": origin_arm,
                "pick_pos": pick_pos, "pick_quat": pick_quat,
                "place_pos": handoff_pos, "place_quat": handoff_quat,
                "right_obj": obj if origin_arm == "right" else None,
            })
            tasks.append({
                **common, "kind": "handoff_in", "arm": bin_arm,
                "pick_pos": handoff_pos, "pick_quat": handoff_quat,
                "place_pos": bin_pos, "place_quat": bin_quat,
                "right_obj": obj if bin_arm == "right" else None,
            })
    return tasks


def build_robot_cfg_dict():
    import yaml
    with open(YML_PATH) as f:
        data = yaml.safe_load(f)
    kin = data["robot_cfg"]["kinematics"]
    kin["urdf_path"] = str(URDF_PATH)
    kin["asset_root_path"] = str(REPO_ROOT) + "/"
    return data["robot_cfg"]


def pose_goal(tool_frames, pos_r, quat_r, pos_l, quat_l):
    import torch
    from curobo.types import GoalToolPose
    position = torch.tensor(
        [[[[pos_r], [pos_l]]]], device=DEVICE, dtype=torch.float32,
    )  # [1,1,2,1,3] = [B,H,L,G,3]
    quaternion = torch.tensor(
        [[[[quat_r], [quat_l]]]], device=DEVICE, dtype=torch.float32
    )  # [1,1,2,1,4]
    return GoalToolPose(tool_frames=tool_frames, position=position, quaternion=quaternion)


def dual_pose(arm, pos, quat):
    """Builds the (pos_r, quat_r, pos_l, quat_l) goal for a single-arm
    task: `arm` gets the real target, the other arm's target is simply its
    own home pose (it isn't part of this task)."""
    if arm == "right":
        return pos, quat, LEFT_HOME_POS, LEFT_HOME_QUAT
    return RIGHT_HOME_POS, RIGHT_HOME_QUAT, pos, quat


# Table surface height (world z) -- must match publish_perception_scene.py's
# own TABLE_TOP_Z and table.yml.
TABLE_TOP_Z = 0.10

# Same primitive dimensions publish_perception_scene.py's make_shape() (and
# physics/physics_one.py's SHAPE_DIMS) use -- kept as its own copy
# here rather than importing across venvs, same as those two already do.
SHAPE_DIMS = {
    "sphere": {"radius": 0.025},
    "cylinder": {"radius": 0.018, "height": 0.06},
    "cone": {"base_radius": 0.025, "height": 0.05},
    "cube": {"side": 0.035},
    "cuboid": {"l": 0.045, "w": 0.025, "h": 0.03},
    "prism": {"side": 0.04, "height": 0.05},
}

OBSTACLE_XY_MARGIN = 0.015  # extra clearance beyond the object's own footprint


def obstacle_cuboid(shape, x, y, yaw=0.0):
    """A conservative, ROTATION-INVARIANT cuboid obstacle for one object:
    square in xy sized to the object's own diagonal footprint (so any yaw
    is safely covered without needing to encode it in the pose) plus a
    clearance margin, at the object's real height, sitting on the table.
    Used to make curobo actually plan AROUND every object it isn't
    currently handling, instead of only knowing about the table/bins and
    happily sweeping straight through wherever other objects are sitting
    (confirmed as a real problem: a full-scene PyBullet replay without this
    knocked every single object out of place, 0/18 correctly sorted)."""
    d = SHAPE_DIMS[shape]
    if shape == "sphere":
        diag, height = 2 * d["radius"], 2 * d["radius"]
    elif shape == "cylinder":
        diag, height = 2 * d["radius"], d["height"]
    elif shape == "cone":
        diag, height = 2 * d["base_radius"], d["height"]
    elif shape == "cube":
        diag, height = d["side"] * 1.42, d["side"]
    elif shape == "cuboid":
        diag, height = (d["l"] ** 2 + d["w"] ** 2) ** 0.5, d["h"]
    elif shape == "prism":
        diag, height = 2 * d["side"] / (3 ** 0.5), d["height"]
    else:
        raise ValueError(shape)
    side = diag + 2 * OBSTACLE_XY_MARGIN
    return {
        "dims": [side, side, height],
        "pose": [x, y, TABLE_TOP_Z + height / 2, 1, 0, 0, 0.0],
    }
