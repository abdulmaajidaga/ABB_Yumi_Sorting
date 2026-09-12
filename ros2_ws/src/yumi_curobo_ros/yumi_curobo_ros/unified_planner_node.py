"""ROS2 node exposing /unified_planner/fk_compute and /unified_planner/ik_batch_poses,
backed by curobo_worker.py over a local TCP socket (plain rclpy here -- no
curobo/torch import, so this runs fine under ROS2's system Python).
"""

import json
import socket
import time

import rclpy
from rclpy.node import Node

from yumi_curobo_msgs.srv import ComputeFk, ComputeIk

HOST, PORT = "127.0.0.1", 5556


class UnifiedPlannerNode(Node):
    def __init__(self):
        super().__init__("unified_planner")
        self._sock = None
        self._buf = b""
        self._connect()

        self.create_service(ComputeFk, "/unified_planner/fk_compute", self.on_fk)
        self.create_service(ComputeIk, "/unified_planner/ik_batch_poses", self.on_ik)
        self.get_logger().info("unified_planner ready: /unified_planner/fk_compute, "
                                "/unified_planner/ik_batch_poses")

    def _connect(self):
        while rclpy.ok():
            try:
                self._sock = socket.create_connection((HOST, PORT), timeout=5.0)
                self._sock.settimeout(30.0)
                self.get_logger().info(f"connected to curobo_worker at {HOST}:{PORT}")
                return
            except (ConnectionRefusedError, OSError) as e:
                self.get_logger().warn(f"curobo_worker not available ({e}), retrying in 1s")
                time.sleep(1.0)

    def _call(self, req: dict) -> dict:
        payload = (json.dumps(req) + "\n").encode("utf-8")
        try:
            self._sock.sendall(payload)
        except (BrokenPipeError, OSError):
            self._connect()
            self._sock.sendall(payload)

        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                self._connect()
                continue
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line)

    def on_fk(self, request, response):
        req = {
            "op": "fk",
            "joint_state": {
                "name": list(request.joint_state.name),
                "position": list(request.joint_state.position),
            },
        }
        resp = self._call(req)
        response.success = resp["success"]
        response.message = resp["message"]
        response.tool_frames = resp.get("tool_frames", [])
        for p in resp.get("poses", []):
            from geometry_msgs.msg import Pose
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = p["position"]
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = p["orientation"]
            response.poses.append(pose)
        return response

    def on_ik(self, request, response):
        poses = []
        for p in request.poses:
            poses.append({
                "position": [p.position.x, p.position.y, p.position.z],
                "orientation": [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w],
            })
        req = {
            "op": "ik",
            "tool_frames": list(request.tool_frames),
            "poses": poses,
            "lock_right": bool(request.lock_right),
            "lock_left": bool(request.lock_left),
            "seed_state": {
                "name": list(request.seed_state.name),
                "position": list(request.seed_state.position),
            },
        }
        resp = self._call(req)
        response.success = resp["success"]
        response.message = resp["message"]
        response.joint_names = resp.get("joint_names", [])
        response.joint_positions = resp.get("joint_positions", [])
        return response


def main():
    rclpy.init()
    node = UnifiedPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
