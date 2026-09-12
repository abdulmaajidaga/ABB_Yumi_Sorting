#!/usr/bin/env bash
# One command for the LIVE, watch-it-happen demo (as opposed to
# demo_one.sh's fast headless plan+physics check): opens Foxglove
# Studio itself, brings up the bridge/robot/table/objects, starts the ROS
# relay that overlays curobo's live joint motion + collision spheres, then
# runs the actual curobo grasp sequence IN THE FOREGROUND of this terminal
# -- it already pauses on "Press Enter to plan+run the perception-driven
# grasp" once the robot's at its start pose and the relay is connected, so
# you get a real chance to look at the robot/table/objects in Foxglove
# before anything moves.
#
# After each round's grasp completes, its exact trajectory is written to
# yumi_description/scene/last_grasp_trajectory.json -- run
# physics/physics_one.py (physics venv) in another terminal any time
# after a round to validate that SAME grasp under real gravity/contact.
#
# Usage:
#   ./demo_live.sh [--loop]

set -eo pipefail  # not -u: ROS2's setup.bash and venv activate scripts both
                   # reference variables that are unset before first sourcing

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$REPO_ROOT/yumi_description/scripts"
SCENE_JSON="$REPO_ROOT/yumi_description/scene/last_perception_scene.json"
LOG_DIR="/tmp/yumi_pipeline_logs"
mkdir -p "$LOG_DIR"

echo "[demo] stopping any stale nodes from earlier runs..."
pkill -f "robot_state_publisher robot_state_publisher" 2>/dev/null || true
pkill -f "publish_home_pose.py" 2>/dev/null || true
pkill -f "publish_table_marker.py" 2>/dev/null || true
pkill -f "publish_perception_scene.py" 2>/dev/null || true
pkill -f "ros_relay_node.py" 2>/dev/null || true
pkill -f "curobo_live.py" 2>/dev/null || true
sleep 1

echo "[demo] opening Foxglove Studio..."
nohup foxglove-studio > "$LOG_DIR/foxglove_studio.log" 2>&1 &
disown

echo "[demo] starting visualization stack (robot, table, objects)..."
source /opt/ros/humble/setup.bash
rm -f "$SCENE_JSON"
ros2 launch "$REPO_ROOT/yumi_description/launch/yumi_bringup.launch.py" \
  > "$LOG_DIR/yumi_bringup.log" 2>&1 &
BRINGUP_PID=$!
echo "[demo] yumi_bringup launched (pid $BRINGUP_PID)"

echo "[demo] waiting for a fresh scene export..."
for i in $(seq 1 30); do
  [[ -f "$SCENE_JSON" ]] && break
  if ! kill -0 "$BRINGUP_PID" 2>/dev/null; then
    echo "[demo] yumi_bringup exited early -- check $LOG_DIR/yumi_bringup.log"
    exit 1
  fi
  sleep 1
done
if [[ ! -f "$SCENE_JSON" ]]; then
  echo "[demo] timed out waiting for $SCENE_JSON -- check $LOG_DIR/yumi_bringup.log"
  exit 1
fi
echo "[demo] scene ready: $SCENE_JSON"
grep -a "correctly classified" "$LOG_DIR/yumi_bringup.log" | tail -1

# publish_home_pose.py (started by yumi_bringup.launch.py, for a sane idle
# pose while nothing else is active) and ros_relay_node.py both publish to
# /joint_states -- leaving both running makes robot_state_publisher flicker
# between the idle home pose and whatever curobo is actually commanding,
# which looks exactly like the robot randomly glitching between configs.
# The relay is about to become the sole source of truth, so home_pose's
# job here is done.
pkill -f "publish_home_pose.py" 2>/dev/null || true

echo "[demo] starting the ROS relay (curobo's live joint motion + collision-sphere overlay)..."
python3 "$SCRIPTS/ros_relay_node.py" > "$LOG_DIR/ros_relay_node.log" 2>&1 &
RELAY_PID=$!
echo "[demo] ros_relay_node started (pid $RELAY_PID) -- it'll keep retrying until curobo connects"

echo
echo "[demo] starting the curobo grasp demo (curobo venv) -- this runs in THIS terminal."
echo "[demo] it will pause with \"Press Enter\" once the robot is at its start pose and"
echo "[demo] the relay has connected -- look at Foxglove, then press Enter to run it."
echo
cd "$REPO_ROOT"
source curobo/.venv/bin/activate
python3 "$SCRIPTS/curobo_live.py" "$@" || true   # curobo segfaults on
                                                                    # shutdown even after a
                                                                    # clean run -- not a real error

echo
echo "[demo] curobo demo exited. Visualization stack (pid $BRINGUP_PID) and relay (pid $RELAY_PID)"
echo "[demo] are still running in the background for Foxglove."
echo "[demo] to physics-validate the last completed round's grasp:"
echo "[demo]   cd physics && source .venv/bin/activate && python3 physics_one.py"
