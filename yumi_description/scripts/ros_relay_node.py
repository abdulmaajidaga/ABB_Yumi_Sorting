"""Connects to curobo_realtime_source.py over a local TCP socket and
republishes each real-time computed tick to ROS2 as /joint_states and
/collision_spheres (visualization_msgs/MarkerArray), immediately, with no
buffering. Run under ROS2's system Python (plain rclpy, no curobo/torch).

Usage:
    source /opt/ros/humble/setup.bash
    python3 yumi_description/scripts/ros_relay_node.py
"""

import json
import socket
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

HOST, PORT = "127.0.0.1", 5555

JOINT_NAMES = [
    "yumi_joint_1_r", "yumi_joint_2_r", "yumi_joint_7_r", "yumi_joint_3_r",
    "yumi_joint_4_r", "yumi_joint_5_r", "yumi_joint_6_r",
    "yumi_joint_1_l", "yumi_joint_2_l", "yumi_joint_7_l", "yumi_joint_3_l",
    "yumi_joint_4_l", "yumi_joint_5_l", "yumi_joint_6_l",
    "gripper_r_joint", "gripper_l_joint",
]


class RelayNode(Node):
    def __init__(self):
        super().__init__("yumi_curobo_relay")
        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/collision_spheres", 10)
        self._last_cost_log = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._socket_loop, daemon=True)
        self._thread.start()

    def _socket_loop(self):
        while not self._stop:
            try:
                self.get_logger().info(f"Connecting to curobo source at {HOST}:{PORT} ...")
                sock = socket.create_connection((HOST, PORT), timeout=5.0)
                self.get_logger().info("Connected to curobo real-time source")
                buf = b""
                sock.settimeout(2.0)
                while not self._stop:
                    try:
                        chunk = sock.recv(65536)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if line:
                            self._handle_message(json.loads(line))
            except (ConnectionRefusedError, OSError) as e:
                self.get_logger().warn(f"curobo source not available ({e}), retrying in 1s")
                import time
                time.sleep(1.0)

    def _handle_message(self, msg):
        now = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = now
        js.name = JOINT_NAMES
        js.position = msg["positions"]
        self.js_pub.publish(js)

        marray = MarkerArray()
        for i, (x, y, z, r) in enumerate(msg["spheres"]):
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = now
            m.ns = "collision_spheres"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 2.0 * r
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.9, 0.35
            marray.markers.append(m)
        self.marker_pub.publish(marray)

        cost = msg.get("self_collision_cost", 0.0)
        t = msg.get("stamp", 0.0)
        if cost > 0 and t - self._last_cost_log > 0.5:
            self.get_logger().warn(f"self-collision cost={cost:.6f} (should not happen along this path)")
            self._last_cost_log = t

    def destroy_node(self):
        self._stop = True
        super().destroy_node()


def main():
    rclpy.init()
    node = RelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
