"""Drive the real YuMi through the same HOME -> PICK -> PLACE -> HOME sequence
already verified in Foxglove (yumi_description/scripts/curobo_pick_place_demo.py),
via TrajExecR/TrajExecL over the same TCP mechanism proven in robot_test.py.

Prerequisites (do these first, same as any other real-hardware run):
  1. Upload + load TrajExecR.modx into T_ROB_R and TrajExecL.modx into T_ROB_L
     via rws_upload.py.
  2. On the FlexPendant: both tasks enabled, PP at main, key in Auto, Motors On,
     Play. Both should log "waiting for PC to connect".

Run in the curobo venv (talks to the robot over plain sockets, no ROS needed):
    source curobo/.venv/bin/activate
    python test/run_real_pickplace.py
"""

import math
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
ROBOT_IP = "192.168.125.1"
PORT_R = 6512
PORT_L = 6513

# Must match yumi_description/scripts/curobo_pick_place_demo.py -- this is the
# same sequence, already verified in Foxglove.
TABLE_TOP_Z = 0.10
HOVER_HEIGHT = 0.02
GRIPPER_TIP_OFFSET = 0.15
Z = TABLE_TOP_Z + HOVER_HEIGHT + GRIPPER_TIP_OFFSET
QUAT_DOWN_R = [1.7e-05, 0.991397, 0.130891, -4.4e-05]
QUAT_DOWN_L = [2.2e-06, 0.991408, -0.130807, 5.7e-06]
PICK_R = (0.45, -0.35)
PLACE_R = (0.25, -0.10)
PICK_L = (0.45, 0.35)
PLACE_L = (0.25, 0.10)

# kinematics joint order: [1r,2r,7r,3r,4r,5r,6r, 1l,2l,7l,3l,4l,5l,6l, grip_r, grip_l]
# RAPID wire order per arm: [rax_1,rax_2,rax_3,rax_4,rax_5,rax_6,eax_a] (physical order)
R_KIN_IDX = [0, 1, 3, 4, 5, 6, 2]
L_KIN_IDX = [7, 8, 10, 11, 12, 13, 9]


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


def pose_goal(tool_frames, xy_r, xy_l):
    position = torch.tensor(
        [[[[[xy_r[0], xy_r[1], Z]], [[xy_l[0], xy_l[1], Z]]]]], device=DEVICE, dtype=torch.float32
    )
    quaternion = torch.tensor(
        [[[[QUAT_DOWN_R], [QUAT_DOWN_L]]]], device=DEVICE, dtype=torch.float32
    )
    return GoalToolPose(tool_frames=tool_frames, position=position, quaternion=quaternion)


def connect(port, label):
    print(f"[{label}] Connecting to {ROBOT_IP}:{port} -- make sure TrajExec{label[-1]} is "
          f"Played and waiting ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ROBOT_IP, port))
    print(f"[{label}] Connected.")
    return sock


def recv_line(sock):
    data = sock.recv(4096).decode().strip()
    return data


def read_current_joints_deg(sock):
    line = recv_line(sock)
    return [float(v) for v in line.split(",")]


def wire_to_kin_rad(vals_deg, kin_idx, q_full):
    """Physical-order 7 degrees -> fill the matching 7 kinematic slots of q_full (radians).
    kin_idx (R_KIN_IDX/L_KIN_IDX) is already listed in wire order, i.e. kin_idx[i] is the
    kinematic slot for wire position i -- no further reordering needed here."""
    for slot, v_deg in zip(kin_idx, vals_deg):
        q_full[slot] = math.radians(v_deg)


def kin_to_wire_deg(q_full, kin_idx):
    return [math.degrees(float(q_full[i])) for i in kin_idx]


def main():
    print("=== REAL ROBOT run: connecting to both arms ===")
    sock_r = connect(PORT_R, "TrajExec-R")
    sock_l = connect(PORT_L, "TrajExec-L")

    cfg = yaml.safe_load(open(YML_PATH))["robot_cfg"]["kinematics"]
    home = torch.tensor(cfg["cspace"]["default_joint_position"], device=DEVICE, dtype=torch.float32)
    joint_names = cfg["cspace"]["joint_names"]

    real_start = home.clone()
    r_deg = read_current_joints_deg(sock_r)
    l_deg = read_current_joints_deg(sock_l)
    print(f"[TrajExec-R] Reported current joints (deg): {r_deg}")
    print(f"[TrajExec-L] Reported current joints (deg): {l_deg}")
    wire_to_kin_rad(r_deg, R_KIN_IDX, real_start)
    wire_to_kin_rad(l_deg, L_KIN_IDX, real_start)
    # Grippers aren't reported over this protocol -- leave them at home's value.

    print("\nBuilding MotionPlanner with table obstacle (this warms up CUDA graphs, ~10-20s)...")
    robot_cfg = build_robot_cfg_dict()
    scene = yaml.safe_load(open(SCENE_PATH))
    config = MotionPlannerCfg.create(robot=robot_cfg, scene_model=scene)
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    kin = build_standalone_kinematics()
    interp_dt = planner.trajopt_solver.config.interpolation_dt
    print(f"MotionPlanner ready. interpolation_dt={interp_dt * 1000:.1f}ms")

    def plan_leg(label, q_current, goal_kind, goal_data, retries=5):
        q_start = JointState.from_position(q_current.unsqueeze(0), joint_names=kin.joint_names)
        result = None
        for attempt in range(1, retries + 1):
            planner.reset_seed()
            if goal_kind == "pose":
                xy_r, xy_l = goal_data
                result = planner.plan_pose(pose_goal(planner.tool_frames, xy_r, xy_l),
                                            q_start, max_attempts=20)
            else:
                q_goal = JointState.from_position(goal_data.unsqueeze(0), joint_names=kin.joint_names)
                result = planner.plan_cspace(q_goal, q_start, max_attempts=20)
            if result is not None and result.success.any():
                break
        else:
            raise SystemExit(f"[{label}] FAILED to plan after {retries} retries. Aborting -- "
                              f"nothing unsafe was sent to the robot.")
        traj = result.get_interpolated_plan().position.squeeze(0).squeeze(0)
        print(f"[{label}] planned: {traj.shape[0]} waypoints")
        return traj

    print("\nPlanning full sequence from the robot's REAL current position ...")
    traj_home = plan_leg("real->home", real_start, "joint", home)
    traj_pick = plan_leg("home->pick", traj_home[-1], "pose", (PICK_R, PICK_L))
    traj_place = plan_leg("pick->place", traj_pick[-1], "pose", (PLACE_R, PLACE_L))

    # fine_at_end: stop precisely (RAPID `fine`) on the LAST waypoint of that
    # leg -- i.e. the actual pick/place arrival, where a real gripper action
    # would happen. Every other waypoint (including transit and the retrace
    # legs) blends through with `z10` for smoother, better-synced motion.
    legs = [
        ("real->home", traj_home, False),
        ("home->pick", traj_pick, True),
        ("pick->place", traj_place, True),
        ("place->pick (retrace)", traj_place.flip(0), False),
        ("pick->home (retrace)", traj_pick.flip(0), False),
    ]
    total_wp = sum(t.shape[0] for _, t, _ in legs)

    # Pre-flight sanity check: the very first waypoint of the very first leg
    # should be ~identical to what the robot just reported as its current
    # position. Verify this in Python before touching the sockets at all.
    wp0_r = kin_to_wire_deg(traj_home[0], R_KIN_IDX)
    wp0_l = kin_to_wire_deg(traj_home[0], L_KIN_IDX)
    print("\n[preflight] wp0 (R) vs reported current (R):")
    for name, a, b in zip(("rax1", "rax2", "rax3", "rax4", "rax5", "rax6", "eax_a"), wp0_r, r_deg):
        print(f"    {name}: wp0={a:.4f}  reported={b:.4f}  delta={a - b:+.4f}")
    print("[preflight] wp0 (L) vs reported current (L):")
    for name, a, b in zip(("rax1", "rax2", "rax3", "rax4", "rax5", "rax6", "eax_a"), wp0_l, l_deg):
        print(f"    {name}: wp0={a:.4f}  reported={b:.4f}  delta={a - b:+.4f}")

    print(f"\n{total_wp} total waypoints across {len(legs)} legs, v50, z10 transit / "
          f"fine at pick+place arrival, both arms.")
    input("Ready to execute on the REAL robot. Press Enter to start (Ctrl-C to abort) ...")

    # A stuck/crashed RAPID task should surface as a clear timeout, not an
    # indefinite hang requiring Ctrl-C. v50/fine moves finish well under this.
    sock_r.settimeout(15.0)
    sock_l.settimeout(15.0)

    FIELD_NAMES = ["rax1", "rax2", "rax3", "rax4", "rax5", "rax6", "eax_a"]
    last_r, last_l = list(r_deg), list(l_deg)

    def describe_abort(label, resp, sent_vals, last_vals):
        parts = resp.split(",")
        try:
            axis = int(parts[1])
        except (IndexError, ValueError):
            print(f"  [{label}] ABORT response didn't include a parseable axis: {resp!r}")
            return
        robot_parsed = parts[2] if len(parts) > 2 else "?"
        robot_current = parts[3] if len(parts) > 3 else "?"
        idx = axis - 1
        if 0 <= idx < 7:
            name = FIELD_NAMES[idx]
            sent = sent_vals[idx]
            prev = last_vals[idx]
            print(f"  [{label}] robot rejected axis {axis} ({name}):")
            print(f"      python:  sent={sent:.4f}  last_confirmed={prev:.4f}  delta={sent - prev:+.4f}")
            print(f"      robot:   parsed={robot_parsed}  current={robot_current}")
        else:
            print(f"  [{label}] robot reported an out-of-range axis index: {axis}")

    try:
        for leg_label, traj, fine_at_end in legs:
            n_wp = traj.shape[0]
            print(f"\n[{leg_label}] executing {n_wp} waypoints on real hardware ...")
            for i in range(n_wp):
                t0 = time.monotonic()
                q = traj[i]
                is_fine = fine_at_end and (i == n_wp - 1)
                fine_field = "1" if is_fine else "0"
                vals_r = kin_to_wire_deg(q, R_KIN_IDX)
                vals_l = kin_to_wire_deg(q, L_KIN_IDX)
                line_r = ",".join(f"{v:.4f}" for v in vals_r) + "," + fine_field
                line_l = ",".join(f"{v:.4f}" for v in vals_l) + "," + fine_field
                if is_fine:
                    print(f"  [{leg_label}] wp {i} (arrival) using fine stop")
                if i < 2:
                    print(f"  [{leg_label}] wp {i} sending R={line_r!r}")
                    print(f"  [{leg_label}] wp {i} sending L={line_l!r}")
                sock_r.sendall(line_r.encode())
                sock_l.sendall(line_l.encode())
                try:
                    resp_r = recv_line(sock_r)
                    resp_l = recv_line(sock_l)
                except socket.timeout:
                    raise SystemExit(f"[{leg_label}] wp {i}: no response within 15s -- the robot "
                                      f"task likely faulted/stopped. Check the FlexPendant event "
                                      f"log. Nothing further was sent.")
                if i < 2:
                    print(f"  [{leg_label}] wp {i} resp    R={resp_r!r}")
                    print(f"  [{leg_label}] wp {i} resp    L={resp_l!r}")
                if resp_r.startswith("ABORT") or resp_l.startswith("ABORT"):
                    print(f"[{leg_label}] wp {i}: an arm rejected the waypoint and stopped itself.")
                    if resp_r.startswith("ABORT"):
                        describe_abort("R", resp_r, vals_r, last_r)
                    if resp_l.startswith("ABORT"):
                        describe_abort("L", resp_l, vals_l, last_l)
                    raise SystemExit("Stopping.")
                last_r, last_l = vals_r, vals_l
                if (i + 1) % 10 == 0 or i == n_wp - 1:
                    print(f"  [{leg_label}] {i + 1}/{n_wp} done")
                # Pace sends to curobo's own planned cadence (interp_dt) instead
                # of firing as fast as the network round-trip allows -- without
                # this, a zone-blended (z10) MoveAbsJ queue can build up faster
                # than an arm can physically execute it, especially on legs
                # where that arm's per-step joint travel is larger.
                elapsed = time.monotonic() - t0
                remaining = interp_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            print(f"[{leg_label}] complete.")

        print("\n=== sequence complete on real hardware, robot is at home ===")
    finally:
        for sock, label in ((sock_r, "TrajExec-R"), (sock_l, "TrajExec-L")):
            try:
                sock.sendall(b"END")
                recv_line(sock)
            except OSError:
                pass
            sock.close()


if __name__ == "__main__":
    main()
