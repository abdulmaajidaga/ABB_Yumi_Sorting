"""curobo-side worker for the yumi unified_planner FK/IK services.

Builds one Kinematics model and one always-both-arms-active MotionPlanner
(table wired in as a real obstacle) ONCE at startup, then answers FK/IK
requests over a local TCP socket for unified_planner_node.py (which runs
under ROS2's system Python -- incompatible with this venv's torch/CUDA build,
hence the socket bridge instead of a direct import).

lock_joints_override (freezing one arm) is handled per-request, not by
rebuilding the model: the "locked" arm's target pose in an IK request is
replaced with its own current pose before solving, so the optimizer sees a
trivially-satisfied goal there. There is exactly one compiled/warmed-up
planner for the whole process lifetime -- locking never triggers a rebuild.

Run in the curobo venv:
    source curobo/.venv/bin/activate
    python ros2_ws/src/yumi_curobo_ros/curobo_worker.py
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
from curobo._src.types.content_path import ContentPath
from curobo._src.types.device_cfg import DeviceCfg

REPO_ROOT = Path("/home/abdul/Documents/yumi")
DESC_ROOT = REPO_ROOT / "yumi_description"
YML_PATH = DESC_ROOT / "yumi.yml"
URDF_PATH = DESC_ROOT / "urdf" / "yumi.urdf"
SCENE_PATH = DESC_ROOT / "scene" / "table.yml"

DEVICE = "cuda:0"
HOST, PORT = "127.0.0.1", 5556
TOOL_FRAMES = ["gripper_r_base", "gripper_l_base"]
RIGHT_ARM_JOINTS = {"yumi_joint_1_r", "yumi_joint_2_r", "yumi_joint_7_r", "yumi_joint_3_r",
                    "yumi_joint_4_r", "yumi_joint_5_r", "yumi_joint_6_r", "gripper_r_joint"}
LEFT_ARM_JOINTS = {"yumi_joint_1_l", "yumi_joint_2_l", "yumi_joint_7_l", "yumi_joint_3_l",
                   "yumi_joint_4_l", "yumi_joint_5_l", "yumi_joint_6_l", "gripper_l_joint"}


def build_robot_cfg_dict():
    with open(YML_PATH) as f:
        data = yaml.safe_load(f)
    kin = data["robot_cfg"]["kinematics"]
    kin["urdf_path"] = str(URDF_PATH)
    kin["asset_root_path"] = str(REPO_ROOT) + "/"
    return data["robot_cfg"]


def build_standalone_kinematics():
    """Independent of the planner's own kinematics -- see curobo_random_reach_demo.py
    for why sharing planner.kinematics for ad-hoc FK calls is unsafe."""
    content_path = ContentPath(
        robot_config_absolute_path=str(YML_PATH),
        robot_urdf_absolute_path=str(URDF_PATH),
        robot_asset_absolute_path=str(REPO_ROOT) + "/",
    )
    device_cfg = DeviceCfg(device=DEVICE)
    kcfg = KinematicsCfg.from_content_path(content_path, device_cfg=device_cfg)
    return Kinematics(kcfg)


class Worker:
    def __init__(self):
        robot_cfg = build_robot_cfg_dict()
        scene = yaml.safe_load(open(SCENE_PATH))

        print("Building MotionPlanner (all 16 joints active, table obstacle wired in)...")
        config = MotionPlannerCfg.create(robot=robot_cfg, scene_model=scene)
        self.planner = MotionPlanner(config)
        self.planner.warmup(enable_graph=True, num_warmup_iterations=5)
        print("MotionPlanner ready. joint_names:", self.planner.joint_names)

        self.kin = build_standalone_kinematics()
        self.joint_names = self.kin.joint_names

        cfg = yaml.safe_load(open(YML_PATH))["robot_cfg"]["kinematics"]
        self.last_q = torch.tensor(
            cfg["cspace"]["default_joint_position"], device=DEVICE, dtype=torch.float32
        )

    def _fk_all(self, q: torch.Tensor):
        js = JointState.from_position(q.unsqueeze(0), joint_names=self.joint_names)
        return self.kin.compute_kinematics(js)

    def handle_fk(self, req):
        names = req["joint_state"]["name"]
        positions = req["joint_state"]["position"]
        q = self.last_q.clone()
        if names:
            idx = {n: i for i, n in enumerate(self.joint_names)}
            for n, p in zip(names, positions):
                if n in idx:
                    q[idx[n]] = p
        state = self._fk_all(q)
        pos = state.tool_poses.position.squeeze(0).squeeze(0).detach().cpu().numpy()
        quat = state.tool_poses.quaternion.squeeze(0).squeeze(0).detach().cpu().numpy()
        poses = []
        for p, q_ in zip(pos, quat):
            # curobo quaternion is wxyz; ROS geometry_msgs/Pose wants xyzw
            poses.append({
                "position": [float(p[0]), float(p[1]), float(p[2])],
                "orientation": [float(q_[1]), float(q_[2]), float(q_[3]), float(q_[0])],
            })
        return {"success": True, "message": "ok",
                "tool_frames": state.tool_poses.tool_frames, "poses": poses}

    def handle_ik(self, req):
        tool_frames = req["tool_frames"]
        if tool_frames != TOOL_FRAMES:
            return {"success": False, "message": f"tool_frames must be {TOOL_FRAMES}",
                    "joint_names": [], "joint_positions": []}

        seed_names = req["seed_state"]["name"]
        seed_positions = req["seed_state"]["position"]
        q_seed = self.last_q.clone()
        if seed_names:
            idx = {n: i for i, n in enumerate(self.joint_names)}
            for n, p in zip(seed_names, seed_positions):
                if n in idx:
                    q_seed[idx[n]] = p

        current_state = self._fk_all(q_seed)
        cur_pos = current_state.tool_poses.position.clone()
        cur_quat = current_state.tool_poses.quaternion.clone()

        position = cur_pos.clone()
        quaternion = cur_quat.clone()
        lock_flags = [req["lock_right"], req["lock_left"]]
        for i, (pose, locked) in enumerate(zip(req["poses"], lock_flags)):
            if locked:
                continue  # keep current pose (already in position/quaternion)
            px, py, pz = pose["position"]
            qx, qy, qz, qw = pose["orientation"]  # ROS xyzw -> curobo wxyz
            position[0, 0, i, 0] = px
            position[0, 0, i, 1] = py
            position[0, 0, i, 2] = pz
            quaternion[0, 0, i, 0] = qw
            quaternion[0, 0, i, 1] = qx
            quaternion[0, 0, i, 2] = qy
            quaternion[0, 0, i, 3] = qz

        goal = GoalToolPose(
            tool_frames=self.planner.tool_frames,
            position=position.unsqueeze(-2).to(torch.float32),
            quaternion=quaternion.unsqueeze(-2).to(torch.float32),
        )
        q_start = JointState.from_position(q_seed.unsqueeze(0), joint_names=self.joint_names)

        result = self.planner.plan_pose(goal, q_start, max_attempts=5)
        if result is None or not result.success.any():
            return {"success": False, "message": "planning failed (unreachable or in collision)",
                    "joint_names": self.joint_names, "joint_positions": []}

        interp = result.get_interpolated_plan()
        traj = interp.position.squeeze(0).squeeze(0)
        q_final = traj[-1]
        self.last_q = q_final.clone()
        return {"success": True, "message": "ok",
                "joint_names": self.joint_names,
                "joint_positions": [float(v) for v in q_final.detach().cpu().numpy()]}

    def handle(self, req):
        op = req.get("op")
        if op == "fk":
            return self.handle_fk(req)
        elif op == "ik":
            return self.handle_ik(req)
        return {"success": False, "message": f"unknown op {op}"}


def main():
    worker = Worker()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"curobo_worker listening on {HOST}:{PORT}")

    while True:
        conn, addr = srv.accept()
        print(f"unified_planner_node connected from {addr}")
        buf = b""
        try:
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line:
                        continue
                    req = json.loads(line)
                    t0 = time.monotonic()
                    try:
                        resp = worker.handle(req)
                    except Exception as e:
                        resp = {"success": False, "message": f"exception: {e}"}
                    resp["_time_s"] = time.monotonic() - t0
                    conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()
            print("unified_planner_node disconnected")


if __name__ == "__main__":
    main()
