"""Deterministic zero-pose + pick-and-place demo for YuMi (sim only, for
Foxglove verification before this runs on the real robot).

Sequence, both arms together:
    START (arbitrary, stands in for wherever the real robot happens to be)
      -> HOME  (ABB's calibration/zero position -- NOT all-zero joint angles,
                this is yumi.yml's cspace.default_joint_position)
      -> PICK  (fixed hover point per arm)
      -> PLACE (fixed hover point per arm)
      -> HOME

Each leg is planned with curobo's real MotionPlanner, table wired in as an
obstacle, and streamed over the same local TCP protocol used by
curobo_random_reach_demo.py, so the existing ros_relay_node.py works
unchanged.

Run in the curobo venv:
    source curobo/.venv/bin/activate
    python yumi_description/scripts/curobo_pick_place_demo.py [--loop]
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

REPO_ROOT = Path("/home/abdul/Documents/yumi")
DESC_ROOT = REPO_ROOT / "yumi_description"
YML_PATH = DESC_ROOT / "yumi.yml"
URDF_PATH = DESC_ROOT / "urdf" / "yumi.urdf"
SCENE_PATH = DESC_ROOT / "scene" / "table.yml"

DEVICE = "cuda:0"
HOST, PORT = "127.0.0.1", 5555

# Same table-surface / hover geometry as curobo_random_reach_demo.py.
TABLE_TOP_Z = 0.10
HOVER_HEIGHT = 0.02
GRIPPER_TIP_OFFSET = 0.15
Z = TABLE_TOP_Z + HOVER_HEIGHT + GRIPPER_TIP_OFFSET

QUAT_DOWN_R = [1.7e-05, 0.991397, 0.130891, -4.4e-05]   # wxyz
QUAT_DOWN_L = [2.2e-06, 0.991408, -0.130807, 5.7e-06]   # wxyz

# Fixed pick/place points -- within the same safe sampling zones already
# exercised (and verified collision-free against the table) by
# curobo_random_reach_demo.py. Adjust these after watching it in Foxglove.
PICK_R = (0.45, -0.35)
PLACE_R = (0.25, -0.10)
PICK_L = (0.45, 0.35)
PLACE_L = (0.25, 0.10)

# Arbitrary distinct starting pose for the sim demo only, so the initial
# "go to zero pose" leg is actually visible. On the real robot this gets
# replaced by the arm's live joint feedback read over TCP.
START_OFFSET_RAD = {"shoulder": 0.35, "elbow": -0.2}


def build_robot_cfg_dict():
    with open(YML_PATH) as f:
        data = yaml.safe_load(f)
    kin = data["robot_cfg"]["kinematics"]
    kin["urdf_path"] = str(URDF_PATH)
    kin["asset_root_path"] = str(REPO_ROOT) + "/"
    return data["robot_cfg"]


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


def pose_goal(tool_frames, xy_r, xy_l):
    position = torch.tensor(
        [[[[[xy_r[0], xy_r[1], Z]], [[xy_l[0], xy_l[1], Z]]]]],
        device=DEVICE, dtype=torch.float32,
    )  # [1,1,2,1,3] = [B,H,L,G,3]
    quaternion = torch.tensor(
        [[[[QUAT_DOWN_R], [QUAT_DOWN_L]]]], device=DEVICE, dtype=torch.float32
    )  # [1,1,2,1,4]
    return GoalToolPose(tool_frames=tool_frames, position=position, quaternion=quaternion)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="repeat the sequence forever")
    args = ap.parse_args()

    robot_cfg = build_robot_cfg_dict()
    scene = yaml.safe_load(open(SCENE_PATH))

    print("Building MotionPlanner with table obstacle (this warms up CUDA graphs, ~10-20s)...")
    config = MotionPlannerCfg.create(robot=robot_cfg, scene_model=scene)
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    print("MotionPlanner ready.")

    kin = build_standalone_kinematics()
    cost = build_self_collision_cost(kin)

    cfg = yaml.safe_load(open(YML_PATH))["robot_cfg"]["kinematics"]
    home = torch.tensor(cfg["cspace"]["default_joint_position"], device=DEVICE, dtype=torch.float32)
    joint_names = cfg["cspace"]["joint_names"]

    start = home.clone()
    for jn_idx, jn in enumerate(joint_names):
        if jn in ("yumi_joint_2_r", "yumi_joint_2_l"):
            start[jn_idx] += START_OFFSET_RAD["shoulder"]
        elif jn in ("yumi_joint_3_r", "yumi_joint_3_l"):
            start[jn_idx] += START_OFFSET_RAD["elbow"]

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

    # Publish the (perturbed) start pose right away so Foxglove has something
    # to show while you get the view set up, then wait for you to be ready.
    js_start = JointState.from_position(start.unsqueeze(0), joint_names=kin.joint_names)
    state_start = kin.compute_kinematics(js_start)
    send(start, state_start, extra={"leg": "idle", "planning": False})
    input("\nReady -- robot is at the start pose. Press Enter to run the "
          "pick-and-place sequence (Ctrl-C to abort) ...")

    def stream_traj(label, traj):
        """Stream an already-known-good trajectory (no planning) -- used to
        retrace a forward leg backward, which is always collision-free and
        exact since it's just the same path in reverse."""
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
                xy_r, xy_l = goal_data
                goal = pose_goal(planner.tool_frames, xy_r, xy_l)
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
            q_current = start.clone()
            print(f"\n=== sequence start (from perturbed start pose) ===")

            q_current, ok, _ = run_leg("start->home", q_current, "joint", home)
            if not ok:
                print("Stopping sequence: 'start->home' failed to plan.")
            else:
                q_current, ok, traj_pick = run_leg("home->pick", q_current, "pose", (PICK_R, PICK_L))
                if not ok:
                    print("Stopping sequence: 'home->pick' failed to plan.")
                else:
                    q_current, ok, traj_place = run_leg(
                        "pick->place", q_current, "pose", (PLACE_R, PLACE_L))
                    if not ok:
                        print("Stopping sequence: 'pick->place' failed to plan.")
                    else:
                        # Return trip: retrace the forward path exactly backward
                        # instead of re-planning place->home in joint space --
                        # a fresh cspace plan can land in a different (but
                        # equally valid) IK branch that's hard to unwind back
                        # to the exact home joint values; reversing the path we
                        # already know is collision-free sidesteps that entirely.
                        q_current = stream_traj("place->pick (retrace)", traj_place.flip(0))
                        q_current = stream_traj("pick->home (retrace)", traj_pick.flip(0))
                        print("=== sequence complete, holding at home ===")

            if not args.loop:
                # Idle, keep streaming the final pose so Foxglove/relay stays alive.
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
