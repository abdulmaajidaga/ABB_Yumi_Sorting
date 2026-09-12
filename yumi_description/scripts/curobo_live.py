"""SORTS every object in whatever scene publish_perception_scene.py most
recently built and exported: HOME -> pick it up -> carry it to its
classified shape's bin -> HOME, for each one in turn. Each pick pose is a
real output of the perception pipeline (shape_classifier_single_view.py +
grasp_generator.py); each bin pose is one of the six fixed sorting bins
(yumi_description/scene/bins.yml) publish_perception_scene.py also
computes and exports. Read live from
yumi_description/scene/last_perception_scene.json, not hardcoded -- run
publish_perception_scene.py first each time you want a fresh scene.

An arm can only physically reach the three bins on its OWN side (right:
sphere/cylinder/cone: left: cube/cuboid/prism -- see BIN_POSITIONS in
publish_perception_scene.py), but objects are scattered randomly and don't
know that: a cube can land on the right half of the table just as easily
as the left. So an object whose pick side matches its bin's side gets one
"deliver" task (pick -> straight to its bin -> home); a cross-side object
gets split into two SEPARATE, sequential tasks instead -- "handoff_out"
(the arm that can reach it carries it to one shared handoff point on the
table and releases it) then "handoff_in" (the bin's own arm picks it up
from that same point and finishes the delivery). These run one full
task at a time, never two arms converging on the same point at once --
simpler and safer than trying to choreograph a true simultaneous handoff,
at the cost of speed. Every task moves exactly one arm; the other stays
parked at its own home pose for that task's whole
approach/carry/retreat sequence. If any leg fails to plan, that task is
skipped/aborted (logged clearly) rather than stopping the whole run.

This is kinematic-only: the object's own marker doesn't follow the
gripper in Foxglove (no physical grasp is modeled here, same as every
earlier demo in this project) -- what you see is the ARM actually
traveling from each object's location to the correct bin's location
and back, not the object visibly riding along. physics/physics_one.py
is where an actual pick is checked with real contact/gravity.

All shapes use a top-down grasp (matches every previous demo in this
project). cylinder/cone originally used a SIDE-ON (horizontal) approach
instead, reasoning that a parallel gripper can't pinch a flat top disk from
directly above -- but that reasoning was wrong: the same trick already used
for sphere (descend PAST the flat top and pinch around the round SIDE from
above, the way you'd pick up a cup) works just as well for a standing
cylinder/cone, and avoids an entire class of problems the side-on approach
had (confirmed by direct IK testing: side-on needs ~7cm of vertical
clearance above the table that a short object's own height may not
provide -- a genuine table collision, not a positioning bug). See
grasp_generator.py's module docstring for the full story.

The perception pipeline itself can't run inside this script: curobo/.venv
has no cv2/pip, and CLAUDE.md says not to add packages there. Run
publish_perception_scene.py (system Python, ROS2 sourced) first -- it both
shows the scene in Foxglove AND writes the JSON this script reads.

After each round's grasp succeeds, its right-arm trajectory is exported to
yumi_description/scene/last_grasp_trajectory.json (same schema
curobo_one.py writes) -- run
physics/physics_one.py (physics venv) right after watching a round to
validate that EXACT grasp under real gravity/contact/friction.

Run in the curobo venv:
    source curobo/.venv/bin/activate
    python yumi_description/scripts/curobo_live.py [--loop]
"""

import argparse
import json
import socket
import time
from pathlib import Path

import torch
import yaml

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState
from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.types.content_path import ContentPath
from curobo._src.types.device_cfg import DeviceCfg

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from curobo_shared import (  # noqa: E402
    BINS_SCENE_PATH, DESC_ROOT, DEVICE, LEFT_HOME_POS, LEFT_HOME_QUAT,
    REPO_ROOT, RIGHT_HOME_POS, RIGHT_HOME_QUAT, SCENE_PATH, URDF_PATH,
    YML_PATH, build_robot_cfg_dict, dual_pose, load_tasks, pose_goal, warm_gpu,
)

TRAJ_EXPORT_PATH = DESC_ROOT / "scene" / "last_grasp_trajectory.json"
HOST, PORT = "127.0.0.1", 5555


def export_trajectory_for_physics(joint_names, interp_dt, traj, right_obj):
    """Writes the just-completed round's right-arm trajectory in exactly the
    format physics/physics_one.py expects (same schema
    curobo_one.py writes) -- so after watching a grasp
    happen live in Foxglove, running the physics venv's script immediately
    after validates that EXACT grasp under real gravity/contact/friction,
    not a separately re-planned one."""
    if right_obj is None:
        return
    export = {
        "joint_names": joint_names,
        "interp_dt": float(interp_dt),
        "forward_traj": [[round(float(v), 6) for v in row] for row in traj],
        "right_object": {
            "true_shape": right_obj["true_shape"],
            "classified_shape": right_obj["shape"],
            "world_x": right_obj["true_world_x"],
            "world_y": right_obj["true_world_y"],
            "yaw": right_obj["true_yaw"],
            "grasp_width": right_obj["grasp_width"],
        },
    }
    with open(TRAJ_EXPORT_PATH, "w") as f:
        json.dump(export, f, indent=2)
    print(f"[physics] exported this round's right-arm grasp to {TRAJ_EXPORT_PATH} -- "
          f"run physics/physics_one.py (physics venv) to validate it under real physics.")


def build_standalone_kinematics():
    content_path = ContentPath(
        robot_config_absolute_path=str(YML_PATH),
        robot_urdf_absolute_path=str(URDF_PATH),
        robot_asset_absolute_path=str(REPO_ROOT) + "/",
    )
    device_cfg = DeviceCfg(device=DEVICE)
    kcfg = KinematicsCfg.from_content_path(content_path, device_cfg=device_cfg)
    return Kinematics(kcfg)


def build_self_collision_cost(kin):
    sc_kin_cfg = kin.get_self_collision_config()
    cost_cfg = SelfCollisionCostCfg(weight=1.0, self_collision_kin_config=sc_kin_cfg)
    cost = SelfCollisionCost(cost_cfg)
    cost.setup_batch_tensors(1, 1)
    return cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="repeat the sequence forever")
    args = ap.parse_args()

    # Fail fast, before the ~15s MotionPlanner warmup, if there's no scene
    # to grasp from yet.
    tasks = load_tasks()

    print("Warming up the GPU clock (idle-state fix, see curobo_shared.warm_gpu)...")
    warm_gpu()

    robot_cfg = build_robot_cfg_dict()
    table_scene = yaml.safe_load(open(SCENE_PATH))
    bins_scene = yaml.safe_load(open(BINS_SCENE_PATH))
    scene = {"cuboid": {**table_scene["cuboid"], **bins_scene["cuboid"]}}

    print("Building MotionPlanner with table+bins obstacles (this warms up CUDA graphs, ~10-20s)...")
    config = MotionPlannerCfg.create(robot=robot_cfg, scene_model=scene)
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    print("MotionPlanner ready.")

    kin = build_standalone_kinematics()
    cost = build_self_collision_cost(kin)

    cfg = yaml.safe_load(open(YML_PATH))["robot_cfg"]["kinematics"]
    home = torch.tensor(cfg["cspace"]["default_joint_position"], device=DEVICE, dtype=torch.float32)
    joint_names = cfg["cspace"]["joint_names"]

    interp_dt = planner.trajopt_solver.config.interpolation_dt

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"Listening on {HOST}:{PORT} for the ROS relay...")

    conn = None

    def wait_for_relay():
        nonlocal conn
        srv.settimeout(0.5)
        while conn is None:
            try:
                conn, addr = srv.accept()
                print(f"Relay connected from {addr}")
            except socket.timeout:
                continue

    def send(q, state, extra=None):
        nonlocal conn
        spheres = state.robot_spheres.squeeze(0).squeeze(0).detach().cpu().numpy()
        msg = {
            "stamp": time.monotonic(),
            "positions": [round(float(v), 6) for v in q.detach().cpu().numpy()],
            "self_collision_cost": 0.0,
            "spheres": [[round(float(c), 4) for c in sph] for sph in spheres],
        }
        if extra:
            msg.update(extra)
        payload = (json.dumps(msg) + "\n").encode("utf-8")
        try:
            conn.sendall(payload)
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("Relay disconnected")
            conn.close()
            conn = None

    wait_for_relay()

    js_home = JointState.from_position(home.unsqueeze(0), joint_names=kin.joint_names)
    state_home = kin.compute_kinematics(js_home)
    send(home, state_home, extra={"leg": "idle", "planning": False})
    input("\nReady -- robot is at home. Press Enter to plan+run the "
          "perception-driven grasp (Ctrl-C to abort) ...")

    def stream_traj(label, traj):
        n_wp = traj.shape[0]
        print(f"[{label}] retracing ({n_wp} waypoints, {n_wp * interp_dt:.2f}s)")
        for i in range(n_wp):
            if conn is None:
                wait_for_relay()
            q_i = traj[i]
            js_i = JointState.from_position(q_i.unsqueeze(0), joint_names=kin.joint_names)
            state_i = kin.compute_kinematics(js_i)
            d = cost.forward(state_i.robot_spheres).item()
            send(q_i, state_i, extra={"self_collision_cost": round(float(d), 8),
                                       "leg": label, "planning": True})
            time.sleep(interp_dt)
        return traj[-1].clone()

    def run_leg(label, q_current, goal_kind, goal_data, retries=5):
        q_start = JointState.from_position(q_current.unsqueeze(0), joint_names=kin.joint_names)

        result = None
        plan_time = 0.0
        for attempt in range(1, retries + 1):
            t0 = time.monotonic()
            planner.reset_seed()
            if goal_kind == "pose":
                pos_r, quat_r, pos_l, quat_l = goal_data
                goal = pose_goal(planner.tool_frames, pos_r, quat_r, pos_l, quat_l)
                result = planner.plan_pose(goal, q_start, max_attempts=20)
            else:
                q_goal = JointState.from_position(goal_data.unsqueeze(0), joint_names=kin.joint_names)
                result = planner.plan_cspace(q_goal, q_start, max_attempts=20)
            plan_time += time.monotonic() - t0
            if result is not None and result.success.any():
                if attempt > 1:
                    print(f"[{label}] succeeded on retry {attempt}/{retries}")
                break
        else:
            print(f"[{label}] FAILED to plan after {retries} retries ({plan_time:.2f}s total).")
            return q_current, False, None

        interp = result.get_interpolated_plan()
        traj = interp.position.squeeze(0).squeeze(0)  # (N, dof)
        n_wp = traj.shape[0]
        print(f"[{label}] success ({plan_time:.2f}s), {n_wp} waypoints, {n_wp * interp_dt:.2f}s")

        for i in range(n_wp):
            if conn is None:
                wait_for_relay()
            q_i = traj[i]
            js_i = JointState.from_position(q_i.unsqueeze(0), joint_names=kin.joint_names)
            state_i = kin.compute_kinematics(js_i)
            d = cost.forward(state_i.robot_spheres).item()
            send(q_i, state_i, extra={"self_collision_cost": round(float(d), 8),
                                       "leg": label, "planning": True})
            time.sleep(interp_dt)

        return traj[-1].clone(), True, traj

    try:
        while True:
            q_current = home.clone()
            print(f"\n=== sequence start (from home) ===")

            n_tasks = len(tasks)
            n_completed = 0
            for task_idx, task in enumerate(tasks, 1):
                arm, shape, kind = task["arm"], task["shape"], task["kind"]
                tag = f"{kind}[{shape}]({arm})"
                home_pos, home_quat = (RIGHT_HOME_POS, RIGHT_HOME_QUAT) if arm == "right" \
                    else (LEFT_HOME_POS, LEFT_HOME_QUAT)

                q_after_pick, ok, traj_pick = run_leg(
                    f"{tag} approach", q_current, "pose",
                    dual_pose(arm, task["pick_pos"], task["pick_quat"]))
                if not ok:
                    print(f"*** Task {task_idx}/{n_tasks} ({tag}) SKIPPED -- "
                          f"approach failed to plan. ***")
                    continue
                if task["right_obj"] is not None:
                    export_trajectory_for_physics(kin.joint_names, interp_dt, traj_pick, task["right_obj"])

                q_after_place, ok2, _ = run_leg(
                    f"{tag} carry", q_after_pick, "pose",
                    dual_pose(arm, task["place_pos"], task["place_quat"]))
                if not ok2:
                    print(f"*** Task {task_idx}/{n_tasks} ({tag}) PARTIAL -- picked up but "
                          f"carry failed to plan; retreating from the pick pose. ***")
                    q_current = stream_traj(f"{tag} abort-retreat", traj_pick.flip(0))
                    continue

                q_current, ok3, _ = run_leg(
                    f"{tag} retreat", q_after_place, "pose",
                    dual_pose(arm, home_pos, home_quat))
                if not ok3:
                    # Delivered, but couldn't plan the way back to home --
                    # rare (the whole point of home being the shared rest
                    # pose is that it's reachable from anywhere). Don't lose
                    # track of where the arm actually is: keep going from
                    # the place pose rather than assuming it reached home.
                    print(f"*** Task {task_idx}/{n_tasks} ({tag}) delivered, but retreat-to-home "
                          f"failed to plan; continuing from the place pose. ***")
                    q_current = q_after_place

                print(f"*** Task {task_idx}/{n_tasks}: {tag} complete. ***")
                n_completed += 1
            print(f"=== {n_completed}/{n_tasks} tasks completed, holding at home ===")

            if not args.loop:
                js_final = JointState.from_position(q_current.unsqueeze(0), joint_names=kin.joint_names)
                state_final = kin.compute_kinematics(js_final)
                while True:
                    if conn is None:
                        wait_for_relay()
                    send(q_current, state_final, extra={"leg": "idle", "planning": False})
                    time.sleep(0.2)

    except KeyboardInterrupt:
        pass
    finally:
        if conn is not None:
            conn.close()
        srv.close()


if __name__ == "__main__":
    main()
