"""Plans the ENTIRE sorting sequence (every task curobo_perception_grasp_
demo.py would run, both arms, including handoffs) for the current scene,
headless -- no relay, no Foxglove, no interactive Enter prompt -- and
exports every planned leg's trajectory plus enough object identity to
replay it faithfully. This is what lets physics/physics_all.py (a
separate PyBullet venv) simulate the WHOLE sort -- every object and every
bin actually spawned as real rigid bodies -- instead of the single-object
check curobo_one.py enables.

Reuses curobo_shared.py's task-building/constants (shared with
curobo_live.py, the live Foxglove version) rather than
duplicating that logic -- see curobo_shared.py's own docstring for why
this deliberately does NOT import curobo_live.py itself,
and that file's module docstring for the full same-side/handoff routing
explanation.

Exports yumi_description/scene/last_full_sequence.json:
    joint_names, interp_dt        -- same as every other export in this project
    objects                        -- ground-truth (true_shape, world_x,
                                       world_y, yaw) for every task-bearing
                                       object, indexed the same way tasks
                                       reference them via obj_index -- so
                                       the physics side can spawn the exact
                                       same set of rigid bodies.
    steps                          -- ordered list of executed legs. Each
                                       step is one arm moving along one
                                       trajectory; "grab" (which body to
                                       close the gripper on, if any) and
                                       "release" (whether to open the
                                       gripper at the end of this leg) tell
                                       the physics side when to actually
                                       attach/detach, since only curobo's
                                       PLANNED motion is known here --
                                       gripper choreography, same as every
                                       other script in this project, isn't
                                       modeled by curobo itself.

Run in the curobo venv:
    source curobo/.venv/bin/activate
    python yumi_description/scripts/curobo_all.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Deliberately NOT importing curobo_live here -- see
# curobo_shared.py's own docstring for why: importing that module (even
# without calling anything self-collision-related in it) made every
# planner.plan_pose() call afterward pathologically slow in testing
# (10+ minutes for a single pose that takes 0.14s standalone). This module
# only pulls in what's actually needed here.
from curobo_shared import (
    BINS_SCENE_PATH, DESC_ROOT, LEFT_HOME_POS, LEFT_HOME_QUAT,
    PERCEPTION_EXPORT_PATH, RIGHT_HOME_POS, RIGHT_HOME_QUAT, SCENE_PATH,
    YML_PATH, DEVICE, build_robot_cfg_dict, dual_pose, load_tasks,
    obstacle_cuboid, pose_goal, warm_gpu,
)

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState

SEQUENCE_EXPORT_PATH = DESC_ROOT / "scene" / "last_full_sequence.json"


def plan_leg(planner, joint_names, q_current, arm, pos, quat, retries=5):
    """Returns (q_end, success, traj (N,dof) or None). Mirrors run_leg() in
    curobo_live.py but without any relay/streaming side
    effects -- this script only ever plans, never executes anything live."""
    q_start = JointState.from_position(q_current.unsqueeze(0), joint_names=joint_names)
    goal = pose_goal(planner.tool_frames, *dual_pose(arm, pos, quat))
    result = None
    for attempt in range(1, retries + 1):
        planner.reset_seed()
        result = planner.plan_pose(goal, q_start, max_attempts=20)
        if result is not None and result.success.any():
            break
    else:
        return q_current, False, None
    interp = result.get_interpolated_plan()
    traj = interp.position.squeeze(0).squeeze(0)  # (N, dof)
    return traj[-1].clone(), True, traj


def build_scene_dict(table_scene, bins_scene, objects_by_index, object_state, handoff_xy, exclude_idx):
    """Table + bins + a conservative cuboid obstacle for every OTHER object
    still out on the table (or sitting at the handoff point), so curobo
    actually plans around them instead of through them -- see
    curobo_shared.obstacle_cuboid's docstring for why this matters (a full
    PyBullet replay without it: 0/18 objects correctly sorted, arm knocking
    everything else out of place every time it moved between objects)."""
    cuboids = dict(table_scene["cuboid"])
    cuboids.update(bins_scene["cuboid"])
    for idx, o in objects_by_index.items():
        if idx == exclude_idx:
            continue
        state = object_state[idx]
        if state == "delivered":
            continue  # already inside a bin -- the bin's own box covers it
        x, y = handoff_xy if state == "handoff" else (o["world_x"], o["world_y"])
        cuboids[f"obj_obstacle_{idx}"] = obstacle_cuboid(o["true_shape"], x, y)
    return {"cuboid": cuboids}


def build_planner(robot_cfg, scene):
    config = MotionPlannerCfg.create(robot=robot_cfg, scene_model=scene)
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    return planner


def main():
    tasks = load_tasks()
    if not tasks:
        raise SystemExit("No sortable objects in the current scene.")

    print("Warming up the GPU clock (idle-state fix, see curobo_shared.warm_gpu)...")
    warm_gpu()

    robot_cfg = build_robot_cfg_dict()
    table_scene = yaml.safe_load(open(SCENE_PATH))
    bins_scene = yaml.safe_load(open(BINS_SCENE_PATH))
    handoff_xy = tuple(json.loads(PERCEPTION_EXPORT_PATH.read_text())["handoff_position"][:2])

    cfg = yaml.safe_load(open(YML_PATH))["robot_cfg"]["kinematics"]
    home = torch.tensor(cfg["cspace"]["default_joint_position"], device=DEVICE, dtype=torch.float32)
    joint_names = cfg["cspace"]["joint_names"]

    # Ground truth for every object a task references, indexed by obj_index
    # (see load_tasks()) -- exported once, separately from the per-task
    # references, so the physics side spawns each rigid body exactly once.
    objects_by_index = {}
    for t in tasks:
        objects_by_index[t["obj_index"]] = {
            "true_shape": t["true_shape"], "classified_shape": t["shape"],
            "world_x": t["world_x"], "world_y": t["world_y"], "yaw": t["yaw"],
        }
    # Where each object physically is right now, updated as tasks complete
    # (see the end of the loop below) -- "table" (original spot), "handoff"
    # (released at the shared handoff point, awaiting pickup), "delivered"
    # (in its bin, done).
    object_state = {idx: "table" for idx in objects_by_index}

    print("Building the first MotionPlanner (table+bins obstacles, ~10-20s warmup)...")
    scene = build_scene_dict(table_scene, bins_scene, objects_by_index, object_state, handoff_xy, None)
    planner = build_planner(robot_cfg, scene)
    interp_dt = planner.trajopt_solver.config.interpolation_dt
    print("MotionPlanner ready.")

    steps = []
    q_current = home.clone()
    n_tasks = len(tasks)
    n_completed = 0
    t0_all = time.monotonic()
    for task_idx, task in enumerate(tasks, 1):
        arm, kind, shape, obj_index = task["arm"], task["kind"], task["shape"], task["obj_index"]
        tag = f"{kind}[{shape}]({arm}) obj#{obj_index}"
        home_pos, home_quat = (RIGHT_HOME_POS, RIGHT_HOME_QUAT) if arm == "right" \
            else (LEFT_HOME_POS, LEFT_HOME_QUAT)

        # Rebuild the planner's world for THIS task: every other object as
        # an obstacle at its current known location, this task's own
        # object excluded throughout (it's either the approach target or,
        # once picked, moving with the gripper -- never a static obstacle
        # for its own task).
        scene = build_scene_dict(table_scene, bins_scene, objects_by_index, object_state,
                                  handoff_xy, obj_index)
        planner = build_planner(robot_cfg, scene)

        q_after_pick, ok, traj_pick = plan_leg(
            planner, joint_names, q_current, arm, task["pick_pos"], task["pick_quat"])
        if not ok:
            print(f"[{task_idx}/{n_tasks}] {tag}: SKIPPED -- approach failed to plan.")
            continue
        # "grab" only on handoff_in's approach targets whatever was last
        # released at the handoff point (grab_spawn_index is None then);
        # every other approach targets a specific spawned object.
        grab_spawn_index = None if kind == "handoff_in" else obj_index
        steps.append({
            "task_idx": task_idx, "kind": kind, "arm": arm, "shape": shape, "leg": "approach",
            "grab_spawn_index": grab_spawn_index, "grab_handoff": kind == "handoff_in",
            "traj": [[round(float(v), 6) for v in row] for row in traj_pick],
        })

        q_after_place, ok2, traj_carry = plan_leg(
            planner, joint_names, q_after_pick, arm, task["place_pos"], task["place_quat"])
        if not ok2:
            print(f"[{task_idx}/{n_tasks}] {tag}: PARTIAL -- picked up but carry failed to plan; "
                  f"aborting back to the pick pose.")
            steps.append({
                "task_idx": task_idx, "kind": kind, "arm": arm, "shape": shape, "leg": "abort_retreat",
                "grab_spawn_index": None, "grab_handoff": False, "release": False,
                "traj": [[round(float(v), 6) for v in row] for row in reversed(traj_pick.tolist())],
            })
            q_current = q_after_pick  # curobo itself doesn't re-plan a real path
            # here (see module docstring); reusing the reversed approach
            # trajectory below via traj.flip in the physics/relay side is
            # the same shortcut curobo_live.py takes.
            continue
        # "release" (open the gripper) is where the object is actually left
        # -- at the end of every successful carry leg, whether that's a bin
        # (done) or the handoff point (handoff_out; a later handoff_in task
        # picks it back up with grab_handoff=True).
        release_kind = "bin" if kind in ("deliver", "handoff_in") else "handoff"
        object_state[obj_index] = "delivered" if release_kind == "bin" else "handoff"
        steps.append({
            "task_idx": task_idx, "kind": kind, "arm": arm, "shape": shape, "leg": "carry",
            "grab_spawn_index": None, "grab_handoff": False, "release": release_kind,
            "traj": [[round(float(v), 6) for v in row] for row in traj_carry],
        })

        q_current, ok3, traj_retreat = plan_leg(
            planner, joint_names, q_after_place, arm, home_pos, home_quat)
        if ok3:
            steps.append({
                "task_idx": task_idx, "kind": kind, "arm": arm, "shape": shape, "leg": "retreat",
                "grab_spawn_index": None, "grab_handoff": False, "release": False,
                "traj": [[round(float(v), 6) for v in row] for row in traj_retreat],
            })
        else:
            print(f"[{task_idx}/{n_tasks}] {tag}: delivered, but retreat-to-home failed to "
                  f"plan; continuing from the place pose.")
            q_current = q_after_place

        print(f"[{task_idx}/{n_tasks}] {tag}: complete.")
        n_completed += 1

    dt_all = time.monotonic() - t0_all
    print(f"\n{n_completed}/{n_tasks} tasks planned successfully ({dt_all:.1f}s total).")

    export = {
        "joint_names": joint_names,
        "interp_dt": float(interp_dt),
        "objects": objects_by_index,
        "steps": steps,
    }
    with open(SEQUENCE_EXPORT_PATH, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Exported {len(steps)} step(s) covering {n_completed} task(s) to {SEQUENCE_EXPORT_PATH}")


if __name__ == "__main__":
    main()
