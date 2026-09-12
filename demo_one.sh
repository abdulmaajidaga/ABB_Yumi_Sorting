#!/usr/bin/env bash
# One command instead of five terminals: stops any stale instances of these
# specific nodes/scripts (so re-running this doesn't pile up duplicate
# publishers -- the actual source of "weird" behavior like stale markers or
# double-published scenes when a script gets re-run in a fresh terminal
# without killing the old one), opens Foxglove Studio itself, brings up the
# ROS2 visualization stack (robot_state_publisher, foxglove_bridge, idle
# home pose, table, a fresh randomized perception scene), waits for that
# scene to be exported, then runs the curobo grasp plan and the PyBullet
# physics validation in their own venvs, in order.
#
# PyBullet and Foxglove end up running side by side: once the grasp is
# planned, the PyBullet window opens with the robot/table/object already
# loaded and PAUSES there ("Press Enter to start the grasp simulation") --
# so you get a real chance to look at both windows before anything moves,
# same as the live demo's own gate.
#
# The ROS2 visualization stack (and Foxglove Studio) is left running in the
# background afterward -- Ctrl-C this script's own process group or use the
# printed PID to stop it, or just re-run this script (it cleans up after
# itself first).
#
# Usage:
#   ./demo_one.sh [--index N | --only-correct] [--headless]
#     --index N        curobo_one.py: which right-arm
#                       object to target (default: first)
#     --only-correct   curobo_one.py: force a
#                       correctly-classified object
#     --headless       run the physics validation without a visible window
#                       (default: shows it -- watching the grasp happen
#                       under real physics is the actual point of this stage)

set -eo pipefail  # not -u: ROS2's setup.bash and venv activate scripts both
                   # reference variables that are unset before first sourcing

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$REPO_ROOT/yumi_description/scripts"
SCENE_JSON="$REPO_ROOT/yumi_description/scene/last_perception_scene.json"
LOG_DIR="/tmp/yumi_pipeline_logs"
mkdir -p "$LOG_DIR"

EXPORT_ARGS=()
PHYSICS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --index) EXPORT_ARGS+=(--index "$2"); shift 2 ;;
    --only-correct) EXPORT_ARGS+=(--only-correct); shift ;;
    --headless) PHYSICS_ARGS+=(--headless); shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

echo "[pipeline] stopping any stale nodes from earlier runs..."
pkill -f "robot_state_publisher robot_state_publisher" 2>/dev/null || true
pkill -f "publish_home_pose.py" 2>/dev/null || true
pkill -f "publish_table_marker.py" 2>/dev/null || true
pkill -f "publish_perception_scene.py" 2>/dev/null || true
sleep 1

echo "[pipeline] opening Foxglove Studio..."
nohup foxglove-studio > "$LOG_DIR/foxglove_studio.log" 2>&1 &
disown

echo "[pipeline] starting visualization stack via yumi_bringup.launch.py (logs in $LOG_DIR)..."
source /opt/ros/humble/setup.bash

rm -f "$SCENE_JSON"
ros2 launch "$REPO_ROOT/yumi_description/launch/yumi_bringup.launch.py" \
  > "$LOG_DIR/yumi_bringup.log" 2>&1 &
BRINGUP_PID=$!
echo "[pipeline] yumi_bringup launched (pid $BRINGUP_PID)"

echo "[pipeline] waiting for a fresh scene export..."
for i in $(seq 1 30); do
  [[ -f "$SCENE_JSON" ]] && break
  if ! kill -0 "$BRINGUP_PID" 2>/dev/null; then
    echo "[pipeline] yumi_bringup exited early -- check $LOG_DIR/yumi_bringup.log"
    exit 1
  fi
  sleep 1
done
if [[ ! -f "$SCENE_JSON" ]]; then
  echo "[pipeline] timed out waiting for $SCENE_JSON -- check $LOG_DIR/yumi_bringup.log"
  exit 1
fi
echo "[pipeline] scene ready: $SCENE_JSON"
grep -a "correctly classified" "$LOG_DIR/yumi_bringup.log" | tail -1

echo
echo "[pipeline] planning a real grasp with curobo..."
# curobo's own process segfaults on CUDA shutdown EVERY time, even after a
# successful plan (a known, harmless cleanup crash in this build -- not a
# planning failure) -- so exit code alone is unreliable and would abort
# this script under `set -e` right after a real success. Check the log for
# "Exported to" instead of trusting $?.
CUROBO_LOG="$LOG_DIR/curobo_export.log"
set +e
(
  cd "$REPO_ROOT"
  source curobo/.venv/bin/activate
  python3 "$SCRIPTS/curobo_one.py" "${EXPORT_ARGS[@]}"
) 2>&1 | tee "$CUROBO_LOG"
set -e
if ! grep -q "^Exported to" "$CUROBO_LOG"; then
  echo "[pipeline] curobo planning failed -- stopping before the physics stage."
  exit 1
fi

echo
echo "[pipeline] validating the grasp under real physics..."
(
  cd "$REPO_ROOT/physics"
  source .venv/bin/activate
  python3 physics_one.py "${PHYSICS_ARGS[@]}"
)

echo
echo "[pipeline] done. Visualization stack (pid $BRINGUP_PID) is still running in the background for Foxglove."
echo "[pipeline] stop it with: kill $BRINGUP_PID   (or just re-run this script -- it cleans up first)"
echo "[pipeline] re-run this script any time for a fresh random scene + grasp + physics check."
