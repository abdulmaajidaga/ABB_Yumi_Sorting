"""Replays the ENTIRE sorting sequence (exported by
curobo_all.py) inside PyBullet: every object AND every
bin you'd see in Foxglove is spawned here too, as real rigid bodies, and
the robot actually sorts them under real gravity/contact/friction -- the
physics equivalent of curobo_live.py's live Foxglove
demo, not just the single-grasp check physics_one.py does.

No object-identity bookkeeping is needed to decide what a gripper grabs:
every "approach" leg ends by closing the gripper wherever curobo's plan
put it, and whatever's physically there -- a freshly spawned object, or
one a previous task already carried to the handoff point and released --
gets gripped for real, exactly the same contact-based mechanism
physics_one.py already uses for a single object. There's no fake
"attach" logic; if a handoff release left the object somewhere the next
task's approach doesn't actually reach, the close phase will simply find
nothing there, same as any other real miss.

Gripper mechanics (open/close timing, contact-stopped closing, mimic
joint) are identical to physics_one.py -- see that file's docstring
for the full reasoning. This script processes MANY such grab/carry/release
cycles back to back, one per exported step, so it takes a while: this is
the full sort, not a single pick.

Run in the physics venv:
    source physics/.venv/bin/activate
    python3 physics/physics_all.py [--headless]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p

import physics_one as rpg

REPO_ROOT = Path(__file__).resolve().parent.parent
DESC_ROOT = REPO_ROOT / "yumi_description"
SEQUENCE_PATH = DESC_ROOT / "scene" / "last_full_sequence.json"
TABLE_YML = DESC_ROOT / "scene" / "table.yml"
FLAT_URDF = Path(__file__).resolve().parent / "yumi_flat.urdf"

GRIPPER_LOWER, GRIPPER_UPPER = rpg.GRIPPER_LOWER, rpg.GRIPPER_UPPER
GRIPPER_MAX_FORCE = rpg.GRIPPER_MAX_FORCE
TABLE_TOP_Z = 0.10

# Must match yumi_description/scene/bins.yml and BIN_POSITIONS in
# publish_perception_scene.py -- kept in sync by hand, same as SHAPE_DIMS
# above being a hand-kept copy of synthetic_shapes' own dimensions.
BIN_XY = {
    "sphere": (-0.05, -0.30), "cylinder": (-0.05, -0.42), "cone": (-0.05, -0.54),
    "cube": (-0.05, 0.30), "cuboid": (-0.05, 0.42), "prism": (-0.05, 0.54),
}
BIN_HALF_XY, BIN_HEIGHT = 0.04, 0.05  # 8x8cm footprint, 5cm tall


def load_bins():
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[BIN_HALF_XY, BIN_HALF_XY, BIN_HEIGHT / 2])
    for shape, (x, y) in BIN_XY.items():
        color = [0.9, 0.5, 0.1, 0.8] if shape in ("sphere", "cylinder", "cone") else [0.1, 0.5, 0.9, 0.8]
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[BIN_HALF_XY, BIN_HALF_XY, BIN_HEIGHT / 2],
                                   rgbaColor=color)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                           basePosition=[x, y, TABLE_TOP_Z + BIN_HEIGHT / 2])


def which_bin(x, y):
    for shape, (bx, by) in BIN_XY.items():
        if abs(x - bx) <= BIN_HALF_XY and abs(y - by) <= BIN_HALF_XY:
            return shape
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true",
                     help="run without the PyBullet window (default: show it)")
    args = ap.parse_args()

    if not SEQUENCE_PATH.exists():
        raise SystemExit(f"{SEQUENCE_PATH} doesn't exist -- run "
                          f"curobo_all.py first (in the curobo venv).")
    data = json.loads(SEQUENCE_PATH.read_text())
    joint_names = data["joint_names"]
    interp_dt = data["interp_dt"]
    objects = data["objects"]  # {"<index>": {...}}
    steps = data["steps"]
    if not steps:
        raise SystemExit("No steps in the exported sequence -- nothing to replay.")

    print(f"{len(objects)} object(s), {len(steps)} step(s) across the whole sort -- "
          f"this will take a while, it's the full sequence, not one pick.")

    p.connect(p.DIRECT if args.headless else p.GUI)
    p.setGravity(0, 0, -9.81)
    PHYSICS_DT = 1.0 / 240.0
    p.setTimeStep(PHYSICS_DT)
    p.setPhysicsEngineParameter(numSolverIterations=150)
    substeps = max(1, round(interp_dt / PHYSICS_DT))

    robot = p.loadURDF(str(FLAT_URDF), basePosition=[0, 0, 0], useFixedBase=True)
    jmap = rpg.joint_name_map(robot)
    rpg.load_table(TABLE_YML)
    load_bins()

    bodies = {}
    starts = {}
    for idx_str, o in objects.items():
        idx = int(idx_str)
        half_h = rpg.object_half_height(o["true_shape"])
        col, vis, _ = rpg.make_object_shapes(o["true_shape"])
        pos = [o["world_x"], o["world_y"], TABLE_TOP_Z + half_h + 0.001]
        quat = p.getQuaternionFromEuler([0, 0, o["yaw"]])
        body = p.createMultiBody(baseMass=0.03, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                  basePosition=pos, baseOrientation=quat)
        p.changeDynamics(body, -1, lateralFriction=0.9, spinningFriction=0.3, rollingFriction=0.1)
        bodies[idx] = body
        starts[idx] = (pos[0], pos[1])

    for link in range(-1, p.getNumJoints(robot)):
        p.changeDynamics(robot, link, lateralFriction=1.2)

    arm_joint_names = [jn for jn in joint_names if jn not in ("gripper_r_joint", "gripper_l_joint")]
    arm_col_idx = {jn: joint_names.index(jn) for jn in arm_joint_names}

    init_row = steps[0]["traj"][0]
    for jn in arm_joint_names:
        p.resetJointState(robot, jmap[jn], init_row[arm_col_idx[jn]])
    for gj in ("gripper_r_joint", "gripper_r_joint_m", "gripper_l_joint", "gripper_l_joint_m"):
        p.resetJointState(robot, jmap[gj], GRIPPER_UPPER)
    for _ in range(20):
        p.stepSimulation()

    if not args.headless:
        input(f"\nPyBullet window ready -- {len(objects)} object(s) and 6 bins loaded, same as "
              f"Foxglove. Check Foxglove too if you have it open, then press Enter to start "
              f"the full sort (Ctrl-C to abort) ...")

    def hold_arm_at(row):
        for jn in arm_joint_names:
            rpg.drive(robot, jmap[jn], row[arm_col_idx[jn]])

    grip_target = {"right": GRIPPER_UPPER, "left": GRIPPER_UPPER}

    def drive_grippers():
        rpg.drive(robot, jmap["gripper_r_joint"], grip_target["right"], max_force=GRIPPER_MAX_FORCE)
        rpg.drive(robot, jmap["gripper_r_joint_m"], grip_target["right"], max_force=GRIPPER_MAX_FORCE)
        rpg.drive(robot, jmap["gripper_l_joint"], grip_target["left"], max_force=GRIPPER_MAX_FORCE)
        rpg.drive(robot, jmap["gripper_l_joint_m"], grip_target["left"], max_force=GRIPPER_MAX_FORCE)

    def stream(traj):
        for row in traj:
            hold_arm_at(row)
            drive_grippers()
            for _ in range(substeps):
                p.stepSimulation()
                if not args.headless:
                    time.sleep(PHYSICS_DT)

    t0 = time.monotonic()
    for step_i, step in enumerate(steps, 1):
        arm, leg, kind, shape = step["arm"], step["leg"], step["kind"], step["shape"]
        traj = np.array(step["traj"])
        r_main, r_mim = ("gripper_r_joint", "gripper_r_joint_m") if arm == "right" \
            else ("gripper_l_joint", "gripper_l_joint_m")
        tag = f"[{step_i}/{len(steps)}] task#{step['task_idx']} {kind}[{shape}]({arm}) {leg}"

        if leg == "approach":
            grip_target[arm] = GRIPPER_UPPER
            stream(traj)
            print(f"{tag}: closing gripper...")
            n_close = 200
            for i in range(n_close):
                t = (i + 1) / n_close
                target = GRIPPER_UPPER + t * (GRIPPER_LOWER - GRIPPER_UPPER)
                hold_arm_at(traj[-1])
                rpg.drive(robot, jmap[r_main], target, max_force=GRIPPER_MAX_FORCE, max_velocity=0.02)
                rpg.drive(robot, jmap[r_mim], target, max_force=GRIPPER_MAX_FORCE, max_velocity=0.02)
                p.stepSimulation()
                if not args.headless:
                    time.sleep(1.0 / 240.0)
            for _ in range(60):
                hold_arm_at(traj[-1])
                rpg.drive(robot, jmap[r_main], GRIPPER_LOWER, max_force=GRIPPER_MAX_FORCE)
                rpg.drive(robot, jmap[r_mim], GRIPPER_LOWER, max_force=GRIPPER_MAX_FORCE)
                p.stepSimulation()
            r_final = p.getJointState(robot, jmap[r_main])[0]
            grip_target[arm] = r_final
            print(f"{tag}: fingers stopped at {r_final * 1000:.1f}mm")

        elif leg == "carry":
            stream(traj)
            if step.get("release"):
                start = grip_target[arm]
                n_open = 150
                for i in range(n_open):
                    t = (i + 1) / n_open
                    grip_target[arm] = start + t * (GRIPPER_UPPER - start)
                    hold_arm_at(traj[-1])
                    drive_grippers()
                    p.stepSimulation()
                    if not args.headless:
                        time.sleep(1.0 / 240.0)
                grip_target[arm] = GRIPPER_UPPER
                print(f"{tag}: released ({step['release']})")

        elif leg in ("retreat", "abort_retreat"):
            stream(traj)

    for _ in range(120):
        p.stepSimulation()
        if not args.headless:
            time.sleep(PHYSICS_DT)

    dt_all = time.monotonic() - t0
    print(f"\nSort replay finished in {dt_all:.1f}s (sim time). Final resting places:")
    n_in_right_bin = 0
    for idx_str, o in sorted(objects.items(), key=lambda kv: int(kv[0])):
        idx = int(idx_str)
        pos, _ = p.getBasePositionAndOrientation(bodies[idx])
        landed = which_bin(pos[0], pos[1])
        intended = o["classified_shape"]
        ok = landed == intended
        n_in_right_bin += int(ok)
        print(f"  obj#{idx:2d} {o['true_shape']:8s} (classified {intended:8s}) -> "
              f"{'bin:' + landed if landed else 'NOT in a bin':16s} "
              f"[{'OK' if ok else 'MISPLACED' if landed else 'MISSED'}] "
              f"final=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
    print(f"\n{n_in_right_bin}/{len(objects)} object(s) ended up in the bin the robot intended.")

    p.disconnect()


if __name__ == "__main__":
    main()
