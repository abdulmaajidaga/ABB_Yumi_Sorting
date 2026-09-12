"""Real-time random pick-and-place demo for YuMi.

Continuously: samples a random point on the table surface for each gripper
(right arm on its side of the table, left arm on its side), with a top-down
approach orientation, asks curobo's real MotionPlanner -- with the table
wired in as a real world obstacle -- to plan a collision-free trajectory from
the current state to both target poses simultaneously, executes it by
streaming joint states + world-frame collision-sphere positions over the same
local TCP protocol used before, then repeats with a fresh pair of targets.

Run this in the curobo venv; ros_relay_node.py (unchanged) republishes
whatever arrives on the socket to /joint_states and /collision_spheres.

    source curobo/.venv/bin/activate
    python yumi_description/scripts/curobo_random_reach_demo.py
"""

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

# Table top surface is at world z=0.10 (see scene/table.yml: bottom 0.013 +
# 8.7cm thickness). Pick/place points hover this far above the surface,
# measured at the FINGERTIPS -- gripper_r_base/gripper_l_base's own origin
# (what GoalToolPose targets) sits well back from the fingertips along the
# gripper's approach axis. The URDF's gripper_*_joint origin alone (z=0.0837)
# undercounts this -- the fingers extend further still -- so this was
# calibrated empirically against the table collision checker (0.10 still
# collided, 0.15 is clean).
TABLE_TOP_Z = 0.10
HOVER_HEIGHT = 0.02
GRIPPER_TIP_OFFSET = 0.15

# Sampling region: front slab only (x in [0.20, 0.50], within the table_front
# cuboid's [0.144, 0.539] span, inset for margin), split across y so each arm
# stays on its own side (right = -y, left = +y, matching the mount offsets).
X_RANGE = (0.20, 0.50)
Y_RANGE_RIGHT = (-0.45, -0.05)
Y_RANGE_LEFT = (0.05, 0.45)

# Top-down approach orientations (gripper's local +Z axis pointing straight
# down, i.e. -world Z), one minimal rotation away from each gripper's home
# orientation -- computed once from yumi.yml's verified home pose.
QUAT_DOWN_R = [1.7e-05, 0.991397, 0.130891, -4.4e-05]   # wxyz
QUAT_DOWN_L = [2.2e-06, 0.991408, -0.130807, 5.7e-06]   # wxyz


def build_robot_cfg_dict():
    with open(YML_PATH) as f:
        data = yaml.safe_load(f)
    kin = data["robot_cfg"]["kinematics"]
    kin["urdf_path"] = str(URDF_PATH)
    kin["asset_root_path"] = str(REPO_ROOT) + "/"
    return data["robot_cfg"]


def build_standalone_kinematics():
    """A Kinematics instance independent of the planner's, for our own FK /
    self-collision-cost / sphere-broadcast calls. Reusing planner.kinematics
    directly is unsafe: its internal CUDA-graph batch/horizon shapes change
    after plan_pose() runs, which breaks later ad-hoc compute_kinematics calls.
    """
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
    cost_cfg = SelfCollisionCostCfg(
        weight=1.0, self_collision_kin_config=sc_kin_cfg,
    )
    cost = SelfCollisionCost(cost_cfg)
    cost.setup_batch_tensors(1, 1)
    return cost


def sample_table_targets():
    """Sample one pick/place point per arm, hovering above the table."""
    x_r = torch.empty(1, device=DEVICE).uniform_(*X_RANGE).item()
    y_r = torch.empty(1, device=DEVICE).uniform_(*Y_RANGE_RIGHT).item()
    x_l = torch.empty(1, device=DEVICE).uniform_(*X_RANGE).item()
    y_l = torch.empty(1, device=DEVICE).uniform_(*Y_RANGE_LEFT).item()
    z = TABLE_TOP_Z + HOVER_HEIGHT + GRIPPER_TIP_OFFSET

    position = torch.tensor(
        [[[[[x_r, y_r, z]], [[x_l, y_l, z]]]]], device=DEVICE, dtype=torch.float32
    )  # [1,1,2,1,3] = [B,H,L,G,3]
    quaternion = torch.tensor(
        [[[[QUAT_DOWN_R], [QUAT_DOWN_L]]]], device=DEVICE, dtype=torch.float32
    )  # [1,1,2,1,4]
    return position, quaternion


def main():
    robot_cfg = build_robot_cfg_dict()
    scene = yaml.safe_load(open(SCENE_PATH))

    print("Building MotionPlanner with table obstacle (this warms up CUDA graphs, ~10-20s)...")
    config = MotionPlannerCfg.create(robot=robot_cfg, scene_model=scene)
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    print("MotionPlanner ready.")
    print("tool_frames:", planner.tool_frames)
    print("joint_names:", planner.joint_names)

    kin = build_standalone_kinematics()
    cost = build_self_collision_cost(kin)

    cfg = yaml.safe_load(open(YML_PATH))["robot_cfg"]["kinematics"]
    home = torch.tensor(cfg["cspace"]["default_joint_position"], device=DEVICE, dtype=torch.float32)

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

    q_current = home.clone()
    n_plans = 0
    n_success = 0

    try:
        while True:
            if conn is None:
                wait_for_relay()

            position, quaternion = sample_table_targets()
            goal = GoalToolPose(
                tool_frames=planner.tool_frames,
                position=position,
                quaternion=quaternion,
            )

            q_start = JointState.from_position(q_current.unsqueeze(0), joint_names=kin.joint_names)

            n_plans += 1
            t0 = time.monotonic()
            result = planner.plan_pose(goal, q_start, max_attempts=5)
            plan_time = time.monotonic() - t0

            if result is None or not result.success.any():
                print(f"[plan {n_plans}] FAILED ({plan_time:.2f}s) -- resampling target")
                continue

            n_success += 1
            interp = result.get_interpolated_plan()
            traj = interp.position.squeeze(0).squeeze(0)  # [B,H,N,dof] -> (N, dof)
            n_wp = traj.shape[0]
            print(f"[plan {n_plans}] success ({plan_time:.2f}s), {n_wp} waypoints, "
                  f"{n_wp * interp_dt:.2f}s  [{n_success}/{n_plans} succeeded overall]  "
                  f"targets: R=({position[0,0,0,0,0]:.2f},{position[0,0,0,0,1]:.2f}) "
                  f"L=({position[0,0,1,0,0]:.2f},{position[0,0,1,0,1]:.2f})")

            for i in range(n_wp):
                if conn is None:
                    break
                q_i = traj[i]
                js_i = JointState.from_position(q_i.unsqueeze(0), joint_names=kin.joint_names)
                state_i = kin.compute_kinematics(js_i)
                d = cost.forward(state_i.robot_spheres).item()
                send(q_i, state_i, extra={"self_collision_cost": round(float(d), 8),
                                           "planning": True})
                time.sleep(interp_dt)

            q_current = traj[-1].clone()

    except KeyboardInterrupt:
        pass
    finally:
        if conn is not None:
            conn.close()
        srv.close()


if __name__ == "__main__":
    main()
