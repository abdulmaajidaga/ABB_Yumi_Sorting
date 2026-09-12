# ABB YuMi Sorting

Perception-driven object sorting on an ABB YuMi dual-arm robot: a synthetic depth
camera + shape classifier identifies six primitive shapes on a table, curobo plans
collision-aware pick-and-place motions (including cross-arm handoffs when an
object's bin is on the opposite side from where it landed), and a PyBullet replay
validates the plan under real gravity/contact/friction physics.

## Layout

- `yumi_description/` — robot URDF/meshes, ROS2 scripts, curobo scene config
  - `scripts/publish_home_pose.py`, `publish_table_marker.py`, `publish_perception_scene.py` — ROS2/Foxglove side (system Python)
  - `scripts/curobo_live.py`, `curobo_one.py`, `curobo_all.py`, `curobo_shared.py` — motion planning (curobo venv)
  - `scene/table.yml`, `scene/bins.yml` — static obstacle geometry
- `perception/` — synthetic camera, point-cloud extraction, shape classification, grasp generation
- `physics/` — PyBullet physics validation (`physics_one.py`, `physics_all.py`, its own venv)
- `demo_one.sh`, `demo_live.sh` — one-command end-to-end demos

## Running it

```bash
./demo_one.sh --only-correct   # fast: plan one grasp, validate it in PyBullet (GUI by default)
./demo_live.sh                 # watch the whole sort happen live in Foxglove
```

Both scripts bring up ROS2 (`robot_state_publisher`, `foxglove_bridge`), the table,
and a fresh randomized scene on their own.

## Dependencies

- ROS2 Humble
- [curobo](https://github.com/NVlabs/curobo) — vendored separately at `curobo/` (not tracked in this repo; clone it there and set up its own venv per its docs)
- PyBullet — see `physics/` for its own venv setup
