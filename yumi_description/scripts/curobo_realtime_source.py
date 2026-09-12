"""Real-time YuMi motion + collision-sphere source, computed live with curobo
every tick (not precomputed), streamed over a local TCP socket as newline-
delimited JSON so a plain-rclpy relay (running under ROS2's system Python,
which can't load curobo/torch) can republish it to ROS2 topics immediately.

Run this in the curobo venv:
    source curobo/.venv/bin/activate
    python yumi_description/scripts/curobo_realtime_source.py

Then run ros_relay_node.py under the system Python (ROS2 sourced) to actually
publish /joint_states and /collision_spheres.
"""

import json
import math
import socket
import time
from pathlib import Path

import torch
import yaml

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState as CuroboJointState
from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.types.content_path import ContentPath
from curobo._src.types.device_cfg import DeviceCfg

REPO_ROOT = Path("/home/abdul/Documents/yumi")
DESC_ROOT = REPO_ROOT / "yumi_description"
YML_PATH = DESC_ROOT / "yumi.yml"
URDF_PATH = DESC_ROOT / "urdf" / "yumi.urdf"

DEVICE = "cuda:0"
HZ = 30.0
HOST, PORT = "127.0.0.1", 5555


def build_kinematics_and_cost():
    content_path = ContentPath(
        robot_config_absolute_path=str(YML_PATH),
        robot_urdf_absolute_path=str(URDF_PATH),
        robot_asset_absolute_path=str(REPO_ROOT) + "/",
    )
    device_cfg = DeviceCfg(device=DEVICE)
    kcfg = KinematicsCfg.from_content_path(content_path, device_cfg=device_cfg)
    kin = Kinematics(kcfg)
    sc_kin_cfg = kin.get_self_collision_config()
    cost_cfg = SelfCollisionCostCfg(
        weight=1.0, self_collision_kin_config=sc_kin_cfg, device_cfg=device_cfg
    )
    cost = SelfCollisionCost(cost_cfg)
    cost.setup_batch_tensors(1, 1)
    return kin, cost


def main():
    kin, cost = build_kinematics_and_cost()
    joint_names = kin.joint_names
    print("Joint order:", joint_names)

    cfg = yaml.safe_load(open(YML_PATH))["robot_cfg"]["kinematics"]
    home = torch.tensor(cfg["cspace"]["default_joint_position"], device=DEVICE, dtype=torch.float32)

    d2r = math.pi / 180.0
    delta_deg = {
        "yumi_joint_2_r": 25.0, "yumi_joint_2_l": 25.0,
        "yumi_joint_4_r": -20.0, "yumi_joint_4_l": -20.0,
        "yumi_joint_7_r": -15.0, "yumi_joint_7_l": -15.0,
    }
    reach = home.clone()
    for jn, deg in delta_deg.items():
        idx = joint_names.index(jn)
        reach[idx] = reach[idx] + deg * d2r
    reach[joint_names.index("gripper_r_joint")] = 0.025
    reach[joint_names.index("gripper_l_joint")] = 0.025

    period_s = 5.0  # full home->reach->home cycle

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"Listening on {HOST}:{PORT} for the ROS relay...")

    conn = None
    t0 = time.monotonic()
    next_tick = t0
    dt = 1.0 / HZ

    try:
        while True:
            if conn is None:
                srv.settimeout(0.5)
                try:
                    conn, addr = srv.accept()
                    conn.setblocking(True)
                    print(f"Relay connected from {addr}")
                except socket.timeout:
                    continue

            now = time.monotonic()
            t = (now - t0) % period_s
            # triangle wave 0->1->0 over the period, i.e. home->reach->home
            alpha = (t / (period_s / 2.0)) if t < period_s / 2.0 else (2.0 - t / (period_s / 2.0))
            q = home + alpha * (reach - home)

            js = CuroboJointState.from_position(q.unsqueeze(0), joint_names=joint_names)
            state = kin.compute_kinematics(js)
            dist = cost.forward(state.robot_spheres).item()
            spheres = state.robot_spheres.squeeze(0).squeeze(0).detach().cpu().numpy()
            q_np = q.detach().cpu().numpy()

            msg = {
                "stamp": now,
                "positions": [round(float(v), 6) for v in q_np],
                "self_collision_cost": round(float(dist), 8),
                "spheres": [[round(float(c), 4) for c in sph] for sph in spheres],
            }
            payload = (json.dumps(msg) + "\n").encode("utf-8")
            try:
                conn.sendall(payload)
            except (BrokenPipeError, ConnectionResetError):
                print("Relay disconnected")
                conn.close()
                conn = None

            next_tick += dt
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if conn is not None:
            conn.close()
        srv.close()


if __name__ == "__main__":
    main()
