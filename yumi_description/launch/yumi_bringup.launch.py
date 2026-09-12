"""Brings up the whole visualization stack in one command instead of one
per terminal: robot_state_publisher (with package:// resolved to real file
paths in yumi.urdf, the same substitution the working manual command used),
foxglove_bridge, an idle /joint_states publisher (so the robot shows its
real calibrated home pose rather than robot_state_publisher's all-zero
default when nothing else is driving it -- see publish_home_pose.py),
the table marker, and the perception scene (which scatters a fresh random
24-object layout, runs it through the real perception pipeline, and writes
yumi_description/scene/last_perception_scene.json for
curobo_one.py to pick up next).

This only covers the ROS2/system-Python half of the pipeline -- the curobo
and physics stages need their own venvs (curobo/.venv, physics/.venv) that
ros2 launch has no way to activate, so they're NOT part of this file. Use
demo_one.sh (or demo_live.sh) at the repo root to run this plus those two stages
end to end with one command.

Usage:
    source /opt/ros/humble/setup.bash
    ros2 launch yumi_description/launch/yumi_bringup.launch.py
"""

import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess

REPO_ROOT = "/home/abdul/Documents/yumi"
URDF_PATH = f"{REPO_ROOT}/yumi_description/urdf/yumi.urdf"
SCRIPTS = f"{REPO_ROOT}/yumi_description/scripts"
FOXGLOVE_BRIDGE_LAUNCH = "/opt/ros/humble/share/foxglove_bridge/launch/foxglove_bridge_launch.xml"

# Exact substitution the working manual command used (package:// isn't a
# real URI robot_state_publisher can resolve on its own).
ROBOT_DESCRIPTION_CMD = (
    f"sed 's#package://yumi_description/#file://{REPO_ROOT}/yumi_description/#g' {URDF_PATH}"
)


def generate_launch_description():
    robot_state_publisher = ExecuteProcess(
        cmd=[
            "bash", "-c",
            'ros2 run robot_state_publisher robot_state_publisher --ros-args '
            f'-p robot_description:="$({ROBOT_DESCRIPTION_CMD})"',
        ],
        name="robot_state_publisher",
        output="screen",
    )

    # Skip instead of crashing the whole launch if one's already bound to
    # the port (e.g. left running from an earlier terminal) -- ros2 launch
    # has no built-in "only if not already running" check.
    foxglove_bridge = ExecuteProcess(
        cmd=["bash", "-c",
             f'(echo > /dev/tcp/127.0.0.1/8765) 2>/dev/null && '
             f'echo "[yumi_bringup] foxglove_bridge already running on 8765, skipping" || '
             f'ros2 launch {FOXGLOVE_BRIDGE_LAUNCH}'],
        name="foxglove_bridge",
        output="screen",
    )

    home_pose = ExecuteProcess(
        cmd=["python3", f"{SCRIPTS}/publish_home_pose.py"],
        name="publish_home_pose",
        output="screen",
    )

    table_marker = ExecuteProcess(
        cmd=["python3", f"{SCRIPTS}/publish_table_marker.py"],
        name="publish_table_marker",
        output="screen",
    )

    perception_scene = ExecuteProcess(
        cmd=["python3", f"{SCRIPTS}/publish_perception_scene.py"],
        name="publish_perception_scene",
        output="screen",
    )

    return LaunchDescription([
        robot_state_publisher,
        foxglove_bridge,
        home_pose,
        table_marker,
        perception_scene,
    ])
