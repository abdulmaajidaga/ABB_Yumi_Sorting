"""Publish the table (yumi_description/scene/table.yml, one or more named
cuboids) as visible visualization_msgs/Marker(s) so it can be checked against
the real robot in RViz2/Foxglove before it's used for collision checking.
Plain rclpy, no curobo/torch dependency.

Usage:
    source /opt/ros/humble/setup.bash
    python3 yumi_description/scripts/publish_table_marker.py
"""

import yaml
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker

SCENE_PATH = "/home/abdul/Documents/yumi/yumi_description/scene/table.yml"


class TableMarkerPublisher(Node):
    def __init__(self):
        super().__init__("yumi_table_marker_publisher")
        self.pub = self.create_publisher(Marker, "/table_marker", 10)
        self.cuboids = yaml.safe_load(open(SCENE_PATH))["cuboid"]
        self.get_logger().info(f"Publishing {len(self.cuboids)} table cuboid(s): "
                                f"{list(self.cuboids.keys())}")
        self.create_timer(1.0, self.tick)

    def tick(self):
        stamp = self.get_clock().now().to_msg()
        for i, (name, cfg) in enumerate(self.cuboids.items()):
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = stamp
            m.ns = "table"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            x, y, z, qw, qx, qy, qz = (float(v) for v in cfg["pose"])
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z
            m.pose.orientation.w = qw
            m.pose.orientation.x = qx
            m.pose.orientation.y = qy
            m.pose.orientation.z = qz
            m.scale.x, m.scale.y, m.scale.z = (float(v) for v in cfg["dims"])
            m.color.r, m.color.g, m.color.b, m.color.a = 0.55, 0.35, 0.15, 0.6
            m.text = name
            self.pub.publish(m)


def main():
    rclpy.init()
    node = TableMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
