"""Replays a REAL curobo-planned grasp trajectory (exported by
curobo_one.py) inside PyBullet, with the target object
spawned as an actual dynamic rigid body (mass, gravity, contact, friction) --
so we can check whether the grasp would actually HOLD, not just whether
curobo's planner considers it reachable and collision-free (that's all
curobo itself checks; it has no dynamics simulation at all).

curobo's plan only covers the ARM reaching the tool pose -- it does not
choreograph the gripper (the exported trajectory's gripper column sits near
0 throughout, since no cost pushes an otherwise-unconstrained joint away
from its start value). So the gripper open/close timing is added here:
    1. approach   -- replay forward_traj's arm joints; gripper held OPEN
    2. close       -- arm holds at the final (grasp) pose; gripper closes
                      to (grasp_width/2 - squeeze margin), i.e. squeezes
                      slightly past the object's own extent, like a real
                      parallel gripper would
    3. retreat     -- replay forward_traj REVERSED; gripper stays closed
    4. settle      -- hold for ~1s, then check the object's final height

Gripper convention (confirmed from yumi.urdf): gripper_r_joint's axis is
-X with no origin rotation; its mimic gripper_r_joint_m has the same -X
axis but a 180-degree Z origin rotation, flipping it to effectively +X in
the shared parent frame -- so as the joint value rises from its lower limit
0 to upper limit 0.025, the two fingers move APART. 0 = fully closed,
0.025 = fully open (per finger; matches the already-established 0.05m
GRIPPER_MAX_OPENING for the pair). PyBullet does NOT auto-implement URDF
<mimic> tags, so gripper_r_joint_m is driven manually here, in lockstep
with gripper_r_joint (multiplier=1, offset=0 per the URDF's <mimic> tag).

Run in the physics venv:
    source physics/.venv/bin/activate
    python3 physics/physics_one.py [--headless]
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pybullet as p

REPO_ROOT = Path(__file__).resolve().parent.parent
DESC_ROOT = REPO_ROOT / "yumi_description"
TRAJ_PATH = DESC_ROOT / "scene" / "last_grasp_trajectory.json"
TABLE_YML = DESC_ROOT / "scene" / "table.yml"
FLAT_URDF = Path(__file__).resolve().parent / "yumi_flat.urdf"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

GRIPPER_LOWER, GRIPPER_UPPER = 0.0, 0.025  # per-finger, confirmed from URDF
GRIPPER_MAX_FORCE = 20.0  # matches yumi.urdf's <limit effort="20">

# Closing target is GRIPPER_LOWER (fully closed), not grasp_width-derived --
# grasp_width comes from a noisy, partially-occluded single-view point
# cloud (see grasp_generator.py) and is only accurate enough for PLANNING
# (how far apart the fingers need to start), not precise enough to use as
# a literal stopping position: measured on a real cylinder here, it came out
# 50.8mm for a true 36mm diameter -- closing to "grasp_width/2 minus a
# margin" would then stop the fingers wider than the object's real edge and
# never make contact at all. A real gripper doesn't work that way either:
# it closes at constant force (POSITION_CONTROL with maxForce=20N, matching
# yumi.urdf's own <limit effort="20"> for this joint) and stops wherever it
# meets resistance -- so do that here too, instead of trusting the measured
# width as ground truth.

# Same primitive dimensions publish_perception_scene.py's make_shape() uses
# -- so the physics object matches the one the perception pipeline actually
# saw (its ground-truth geometry, not its possibly-wrong classification).
SHAPE_DIMS = {
    "sphere": {"radius": 0.025},
    "cylinder": {"radius": 0.018, "height": 0.06},
    "cone": {"base_radius": 0.025, "height": 0.05},
    "cube": {"side": 0.035},
    "cuboid": {"l": 0.045, "w": 0.025, "h": 0.03},
    "prism": {"side": 0.04, "height": 0.05},
}


def object_half_height(shape):
    d = SHAPE_DIMS[shape]
    if shape == "sphere":
        return d["radius"]
    if shape in ("cylinder", "cone", "prism"):
        return d["height"] / 2
    if shape == "cube":
        return d["side"] / 2
    if shape == "cuboid":
        return d["h"] / 2
    raise ValueError(shape)


def write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write("f " + " ".join(str(i + 1) for i in face) + "\n")


def generate_cone_mesh(base_radius, height, n=24):
    """Apex-up cone (matches synthetic_shapes.cone(apex_up=True)): base
    circle at z=0, apex at z=height. Bullet computes the convex hull of
    whatever mesh it's given for a dynamic body, so a simple triangle fan
    is enough -- no need for a "correct" solid mesh."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    base = np.stack([base_radius * np.cos(angles), base_radius * np.sin(angles),
                      np.zeros(n)], axis=1)
    apex = np.array([[0.0, 0.0, height]])
    verts = np.concatenate([base, apex], axis=0)
    apex_idx = n
    faces = [[apex_idx, i, (i + 1) % n] for i in range(n)]  # side fan
    faces += [[0, i, i + 1] for i in range(1, n - 1)]  # base cap fan
    return verts, faces


def generate_prism_mesh(side, height):
    """Triangular prism, equilateral cross-section, vertical axis -- matches
    synthetic_shapes.prism()'s own vertex geometry exactly (circumradius
    side/sqrt(3), corners at 90/210/330 degrees)."""
    r_circ = side / np.sqrt(3)
    angles = np.array([90, 210, 330]) * np.pi / 180
    tri = np.stack([r_circ * np.cos(angles), r_circ * np.sin(angles)], axis=1)
    bottom = np.concatenate([tri, np.zeros((3, 1))], axis=1)
    top = np.concatenate([tri, np.full((3, 1), height)], axis=1)
    verts = np.concatenate([bottom, top], axis=0)  # 0,1,2 = bottom; 3,4,5 = top
    faces = [
        [0, 1, 2],  # bottom cap
        [3, 5, 4],  # top cap
        [0, 3, 1], [1, 3, 4],  # side 0-1
        [1, 4, 2], [2, 4, 5],  # side 1-2
        [2, 5, 0], [0, 5, 3],  # side 2-0
    ]
    return verts, faces


def make_object_shapes(true_shape):
    """Returns (collisionShapeId, visualShapeId, half_height) for the given
    ground-truth shape, centered on its own local origin sitting ON the
    table (local z=0 = bottom face, consistent with the mesh generators and
    with GEOM_SPHERE/CYLINDER/BOX's own centered convention handled via a
    z-offset in the multibody call)."""
    d = SHAPE_DIMS[true_shape]
    if true_shape == "sphere":
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=d["radius"])
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=d["radius"], rgbaColor=[0.8, 0.2, 0.2, 1])
    elif true_shape == "cylinder":
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=d["radius"], height=d["height"])
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=d["radius"], length=d["height"],
                                   rgbaColor=[0.2, 0.6, 0.8, 1])
    elif true_shape == "cube":
        he = [d["side"] / 2] * 3
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=[0.2, 0.8, 0.3, 1])
    elif true_shape == "cuboid":
        he = [d["l"] / 2, d["w"] / 2, d["h"] / 2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=[0.8, 0.7, 0.2, 1])
    elif true_shape in ("cone", "prism"):
        ASSETS_DIR.mkdir(exist_ok=True)
        obj_path = ASSETS_DIR / f"{true_shape}.obj"
        if not obj_path.exists():
            verts, faces = (generate_cone_mesh(d["base_radius"], d["height"]) if true_shape == "cone"
                             else generate_prism_mesh(d["side"], d["height"]))
            write_obj(obj_path, verts, faces)
        col = p.createCollisionShape(p.GEOM_MESH, fileName=str(obj_path))
        vis = p.createVisualShape(p.GEOM_MESH, fileName=str(obj_path),
                                   rgbaColor=[0.7, 0.3, 0.7, 1])
    else:
        raise ValueError(true_shape)
    return col, vis, object_half_height(true_shape)


def load_table(table_yml_path):
    import yaml
    data = yaml.safe_load(open(table_yml_path))
    ids = []
    for _name, spec in data["cuboid"].items():
        dims = spec["dims"]  # full extents
        pose = spec["pose"]  # [x, y, z, qw, qx, qy, qz]
        half = [d / 2 for d in dims]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.55, 0.4, 0.25, 1])
        x, y, z, qw, qx, qy, qz = pose
        body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                  baseVisualShapeIndex=vis, basePosition=[x, y, z],
                                  baseOrientation=[qx, qy, qz, qw])
        ids.append(body)
    return ids


def joint_name_map(body_id):
    m = {}
    for i in range(p.getNumJoints(body_id)):
        m[p.getJointInfo(body_id, i)[1].decode()] = i
    return m


def drive(body_id, joint_idx, target, max_force=200.0, max_velocity=2.0):
    p.setJointMotorControl2(body_id, joint_idx, p.POSITION_CONTROL, targetPosition=target,
                             force=max_force, maxVelocity=max_velocity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true",
                     help="run without the PyBullet window (default: show it -- the whole "
                          "point of this script is watching the grasp happen under real physics)")
    ap.add_argument("--settle-steps", type=int, default=240)
    args = ap.parse_args()

    if not TRAJ_PATH.exists():
        raise SystemExit(f"{TRAJ_PATH} doesn't exist -- run "
                          f"curobo_one.py first (in the curobo venv).")
    traj_data = json.loads(TRAJ_PATH.read_text())
    joint_names = traj_data["joint_names"]
    interp_dt = traj_data["interp_dt"]
    forward = np.array(traj_data["forward_traj"])  # (N, 16)
    obj = traj_data["right_object"]
    true_shape = obj["true_shape"]
    grasp_width = obj["grasp_width"]
    classified = obj.get("classified_shape", true_shape)

    print(f"Grasp target: {true_shape} (pipeline classified as: {classified}) at world "
          f"({obj['world_x']:.3f}, {obj['world_y']:.3f}), grasp_width={grasp_width * 1000:.1f}mm")

    p.connect(p.DIRECT if args.headless else p.GUI)
    p.setGravity(0, 0, -9.81)
    PHYSICS_DT = 1.0 / 240.0
    p.setTimeStep(PHYSICS_DT)
    p.setPhysicsEngineParameter(numSolverIterations=150)
    # curobo's trajectory is spaced at interp_dt (here 0.025s/waypoint); a
    # single p.stepSimulation() call is only PHYSICS_DT (~0.0042s). Stepping
    # once per waypoint was compressing the whole real-time motion into 1/6
    # of the simulated time -- the arm moving ~6x too fast, which turned an
    # OPEN gripper brushing near the object during descent into a violent,
    # high-velocity impact that flung it away before the real pinch could
    # happen (confirmed directly: contacts already present at the end of a
    # "static hold" approach, then the object jumping several cm on the
    # very first close step). Substepping to really match interp_dt fixes
    # the pacing to what curobo actually intended.
    substeps = max(1, round(interp_dt / PHYSICS_DT))

    robot = p.loadURDF(str(FLAT_URDF), basePosition=[0, 0, 0], useFixedBase=True)
    jmap = joint_name_map(robot)

    load_table(TABLE_YML)

    half_h = object_half_height(true_shape)
    col, vis, _ = make_object_shapes(true_shape)
    obj_start_pos = [obj["world_x"], obj["world_y"], 0.10 + half_h + 0.001]  # 0.10 = TABLE_TOP_Z
    obj_start_quat = p.getQuaternionFromEuler([0, 0, obj["yaw"]])
    obj_body = p.createMultiBody(baseMass=0.03, baseCollisionShapeIndex=col,
                                  baseVisualShapeIndex=vis, basePosition=obj_start_pos,
                                  baseOrientation=obj_start_quat)
    p.changeDynamics(obj_body, -1, lateralFriction=0.9, spinningFriction=0.3, rollingFriction=0.1)
    for link in range(-1, p.getNumJoints(robot)):
        p.changeDynamics(robot, link, lateralFriction=1.2)

    arm_joint_names = [jn for jn in joint_names if jn not in ("gripper_r_joint", "gripper_l_joint")]
    arm_col_idx = {jn: joint_names.index(jn) for jn in arm_joint_names}
    r_grip_idx = joint_names.index("gripper_r_joint")

    # Initial pose: teleport straight to it (no need to physically settle
    # into a pose we know is collision-free -- curobo already verified
    # that against the table).
    for jn in arm_joint_names:
        p.resetJointState(robot, jmap[jn], forward[0, arm_col_idx[jn]])
    p.resetJointState(robot, jmap["gripper_r_joint"], GRIPPER_UPPER)
    p.resetJointState(robot, jmap["gripper_r_joint_m"], GRIPPER_UPPER)
    p.resetJointState(robot, jmap["gripper_l_joint"], GRIPPER_UPPER)
    p.resetJointState(robot, jmap["gripper_l_joint_m"], GRIPPER_UPPER)
    for _ in range(20):
        p.stepSimulation()

    obj_z_start = p.getBasePositionAndOrientation(obj_body)[0][2]

    if not args.headless:
        input("\nPyBullet window ready -- robot, table, and object are loaded. Check Foxglove "
              "too if you have it open, then press Enter to start the grasp simulation "
              "(Ctrl-C to abort) ...")

    def hold_arm_at(row):
        for jn in arm_joint_names:
            drive(robot, jmap[jn], row[arm_col_idx[jn]])

    print(f"Approach: {forward.shape[0]} waypoints x {substeps} substeps, gripper held OPEN ({GRIPPER_UPPER}m/finger)")
    for row in forward:
        hold_arm_at(row)
        drive(robot, jmap["gripper_r_joint"], GRIPPER_UPPER, max_force=GRIPPER_MAX_FORCE)
        drive(robot, jmap["gripper_r_joint_m"], GRIPPER_UPPER, max_force=GRIPPER_MAX_FORCE)
        drive(robot, jmap["gripper_l_joint"], GRIPPER_UPPER, max_force=GRIPPER_MAX_FORCE)
        drive(robot, jmap["gripper_l_joint_m"], GRIPPER_UPPER, max_force=GRIPPER_MAX_FORCE)
        for _ in range(substeps):
            p.stepSimulation()
            if not args.headless:
                time.sleep(PHYSICS_DT)

    close_target = GRIPPER_LOWER  # fully closed -- constant force stops it on contact
    print(f"Close: driving to fully-closed ({close_target * 1000:.1f}mm/finger) at "
          f"{GRIPPER_MAX_FORCE:.0f}N max force -- expect it to stop early on real contact "
          f"(true object radius ~{grasp_width / 2 * 1000:.1f}mm-ish per perception, unreliable)")
    n_close = 200
    for i in range(n_close):
        t = (i + 1) / n_close
        target = GRIPPER_UPPER + t * (close_target - GRIPPER_UPPER)
        hold_arm_at(forward[-1])
        drive(robot, jmap["gripper_r_joint"], target, max_force=GRIPPER_MAX_FORCE, max_velocity=0.02)
        drive(robot, jmap["gripper_r_joint_m"], target, max_force=GRIPPER_MAX_FORCE, max_velocity=0.02)
        p.stepSimulation()
        if not args.headless:
            time.sleep(1.0 / 240.0)
    for _ in range(60):
        hold_arm_at(forward[-1])
        drive(robot, jmap["gripper_r_joint"], close_target, max_force=GRIPPER_MAX_FORCE)
        drive(robot, jmap["gripper_r_joint_m"], close_target, max_force=GRIPPER_MAX_FORCE)
        p.stepSimulation()

    r_final, r_m_final = (p.getJointState(robot, jmap["gripper_r_joint"])[0],
                           p.getJointState(robot, jmap["gripper_r_joint_m"])[0])
    print(f"Fingers stopped at {r_final * 1000:.1f}mm / {r_m_final * 1000:.1f}mm "
          f"(0mm = fully closed, no object in the way)")

    # Keep commanding fully-closed (not just holding at the contact-stopped
    # position) -- that's what a real force-limited gripper does too: it
    # keeps trying to close, and contact with the object (not the target
    # value) is what actually stops it, so the squeeze force stays live
    # throughout the retreat instead of going slack the moment the object
    # settles a hair away from its exact first-contact position.
    print(f"Retreat: {forward.shape[0]} waypoints, gripper commanded fully-closed "
          f"(contact keeps it open near {r_final * 1000:.1f}mm, squeeze force stays live)")
    for row in forward[::-1]:
        hold_arm_at(row)
        drive(robot, jmap["gripper_r_joint"], close_target, max_force=GRIPPER_MAX_FORCE)
        drive(robot, jmap["gripper_r_joint_m"], close_target, max_force=GRIPPER_MAX_FORCE)
        drive(robot, jmap["gripper_l_joint"], GRIPPER_UPPER, max_force=GRIPPER_MAX_FORCE)
        drive(robot, jmap["gripper_l_joint_m"], GRIPPER_UPPER, max_force=GRIPPER_MAX_FORCE)
        for _ in range(substeps):
            p.stepSimulation()
            if not args.headless:
                time.sleep(PHYSICS_DT)

    for _ in range(args.settle_steps):
        p.stepSimulation()
        if not args.headless:
            time.sleep(1.0 / 240.0)

    obj_pos_final, _ = p.getBasePositionAndOrientation(obj_body)
    r_finger_link = jmap["gripper_r_joint"]
    gripper_pos_final = p.getLinkState(robot, r_finger_link)[0]

    lift = obj_pos_final[2] - obj_z_start
    xy_drift = math.hypot(obj_pos_final[0] - obj_start_pos[0], obj_pos_final[1] - obj_start_pos[1])
    gripper_to_obj = math.dist(obj_pos_final, gripper_pos_final)

    print(f"\nObject start z: {obj_z_start:.4f}, final z: {obj_pos_final[2]:.4f} (lift: {lift * 100:+.1f}cm)")
    print(f"XY drift from start: {xy_drift * 100:.1f}cm")
    print(f"Final distance object<->gripper: {gripper_to_obj * 100:.1f}cm")

    success = lift > 0.05 and gripper_to_obj < 0.08
    print(f"\n{'GRASP SUCCEEDED' if success else 'GRASP FAILED'} -- object "
          f"{'followed the gripper up' if success else 'was not carried by the gripper'}.")

    p.disconnect()
    return success


if __name__ == "__main__":
    main()
