"""Plans ONE real curobo grasp (home -> grasp pose, table obstacle included)
for the current perception scene and exports the actual planned joint
trajectory -- not a hand-picked pose -- so physics/physics_one.py (a
separate PyBullet venv, since curobo/.venv has no pip to add pybullet to)
can replay the exact same motion under real gravity/contact/friction physics
and check whether the grasp would actually hold, rather than just checking
that it's kinematically reachable and collision-free (which is all curobo's
own planner checks).

Picks the first right-arm object from last_perception_scene.json (written by
publish_perception_scene.py -- run that first). The left arm is commanded to
its own next object if one exists, purely so both arms move together the
same way the interactive demo does; only the right arm's object is spawned
and grasp-checked in the physics sim (checking both would just double the
work for no extra signal about the technique itself).

Exports yumi_description/scene/last_grasp_trajectory.json:
    joint_names          -- all 16 cspace joint names, in order
    interp_dt             -- seconds between waypoints
    forward_traj           -- home -> grasp, list of 16-value joint lists
    (reverse is just forward_traj reversed -- physics script does that itself)
    right_object            -- {true_shape, world_x, world_y, yaw, grasp_width}
                               ground-truth pose of the object the right arm
                               is actually reaching for (for spawning the real
                               rigid body in PyBullet)

Run in the curobo venv:
    source curobo/.venv/bin/activate
    python yumi_description/scripts/curobo_one.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curobo_shared import (
    DESC_ROOT, DEVICE, LEFT_HOME_POS, LEFT_HOME_QUAT, PERCEPTION_EXPORT_PATH,
    RIGHT_HOME_POS, RIGHT_HOME_QUAT, SCENE_PATH, YML_PATH,
    build_robot_cfg_dict, pose_goal, warm_gpu,
)

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState

TRAJ_EXPORT_PATH = DESC_ROOT / "scene" / "last_grasp_trajectory.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0,
                     help="which right-arm object to target (0 = first)")
    ap.add_argument("--only-correct", action="store_true",
                     help="pick the first right-arm object the pipeline classified "
                          "correctly (shape == true_shape), ignoring --index")
    args = ap.parse_args()

    if not PERCEPTION_EXPORT_PATH.exists():
        raise SystemExit(f"{PERCEPTION_EXPORT_PATH} doesn't exist -- run "
                          f"publish_perception_scene.py first.")
    with open(PERCEPTION_EXPORT_PATH) as f:
        data = json.load(f)

    right_objs = [o for o in data["objects"] if o["arm"] == "right"]
    if not right_objs:
        raise SystemExit("No right-arm objects in the current scene -- re-run "
                          "publish_perception_scene.py to get a new random layout.")
    if args.only_correct:
        candidates = [o for o in right_objs if o["shape"] == o["true_shape"]]
        if not candidates:
            raise SystemExit("No correctly-classified right-arm objects in this scene.")
        right = candidates[0]
    else:
        if args.index >= len(right_objs):
            raise SystemExit(f"--index {args.index} out of range (only {len(right_objs)} "
                              f"right-arm object(s) in this scene).")
        right = right_objs[args.index]
    print(f"Classification: {'CORRECT' if right['shape'] == right['true_shape'] else 'WRONG'} "
          f"(true={right['true_shape']}, pipeline believes={right['shape']})")
    left_objs = [o for o in data["objects"] if o["arm"] == "left"]
    left = left_objs[0] if left_objs else None

    pos_r, quat_r = right["position"], right["quaternion_wxyz"]
    pos_l, quat_l = (left["position"], left["quaternion_wxyz"]) if left else (LEFT_HOME_POS, LEFT_HOME_QUAT)

    print(f"Right arm target: {right['true_shape']} (classified as {right['shape']}) "
          f"at world ({right['true_world_x']:.3f}, {right['true_world_y']:.3f})")

    print("Warming up the GPU clock (idle-state fix, see curobo_shared.warm_gpu)...")
    warm_gpu()

    robot_cfg = build_robot_cfg_dict()
    scene = yaml.safe_load(open(SCENE_PATH))

    print("Building MotionPlanner with table obstacle (~10-20s warmup)...")
    config = MotionPlannerCfg.create(robot=robot_cfg, scene_model=scene)
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    print("MotionPlanner ready.")

    cfg = yaml.safe_load(open(YML_PATH))["robot_cfg"]["kinematics"]
    home = torch.tensor(cfg["cspace"]["default_joint_position"], device=DEVICE, dtype=torch.float32)
    joint_names = cfg["cspace"]["joint_names"]
    interp_dt = planner.trajopt_solver.config.interpolation_dt

    q_start = JointState.from_position(home.unsqueeze(0), joint_names=joint_names)
    goal = pose_goal(planner.tool_frames, pos_r, quat_r, pos_l, quat_l)

    result = None
    for attempt in range(1, 16):
        planner.reset_seed()
        result = planner.plan_pose(goal, q_start, max_attempts=20)
        if result is not None and result.success.any():
            break
    else:
        raise SystemExit(f"Planning FAILED after 15 retries for {right['true_shape']} "
                          f"at ({right['true_world_x']:.3f}, {right['true_world_y']:.3f}).")

    interp = result.get_interpolated_plan()
    traj = interp.position.squeeze(0).squeeze(0)  # (N, dof)
    n_wp = traj.shape[0]
    print(f"Planned {n_wp} waypoints ({n_wp * interp_dt:.2f}s).")

    export = {
        "joint_names": joint_names,
        "interp_dt": float(interp_dt),
        "forward_traj": [[round(float(v), 6) for v in row] for row in traj],
        "right_object": {
            "true_shape": right["true_shape"],
            "classified_shape": right["shape"],
            "world_x": right["true_world_x"],
            "world_y": right["true_world_y"],
            "yaw": right["true_yaw"],
            "grasp_width": right["grasp_width"],
        },
    }
    with open(TRAJ_EXPORT_PATH, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Exported to {TRAJ_EXPORT_PATH}")


if __name__ == "__main__":
    main()
