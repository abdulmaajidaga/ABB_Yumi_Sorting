"""Publishes YuMi's real calibrated home pose (from yumi.yml's
default_joint_position -- NOT all-zero joint angles, ABB's own zero isn't
the natural resting pose) to /joint_states as a static, repeating message.

Without anything publishing /joint_states, robot_state_publisher defaults
every joint to 0 radians, which looks stretched out and unnatural for this
7-DOF redundant arm -- this script exists purely to give it a sane resting
pose to show while nothing else (no curobo motion pipeline) is running.

No torch/curobo dependency -- plain rclpy, runs directly under ROS2's
system Python.

Usage:
    source /opt/ros/humble/setup.bash
    python3 yumi_description/scripts/publish_home_pose.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINT_NAMES = [
    "yumi_joint_1_r", "yumi_joint_2_r", "yumi_joint_7_r", "yumi_joint_3_r",
    "yumi_joint_4_r", "yumi_joint_5_r", "yumi_joint_6_r",
    "yumi_joint_1_l", "yumi_joint_2_l", "yumi_joint_7_l", "yumi_joint_3_l",
    "yumi_joint_4_l", "yumi_joint_5_l", "yumi_joint_6_l",
    "gripper_r_joint", "gripper_l_joint",
]

# From yumi_description/yumi.yml -> robot_cfg.kinematics.cspace.default_joint_position
HOME_POSE = [
    0.0, -2.268928, -2.356194, 0.523599, 0.0, 0.698132, 0.0,
    0.0, -2.268928, 2.356194, 0.523599, 0.0, 0.698132, 0.0,
    0.0, 0.0,
]


class HomePosePublisher(Node):
    def __init__(self):
        super().__init__("yumi_home_pose_publisher")
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.create_timer(0.5, self.tick)

    def tick(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = HOME_POSE
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = HomePosePublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
