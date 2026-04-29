from __future__ import annotations

import numpy as np
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path

from .config import TopicConfig


class SwarmPublishers:
    def __init__(self, node, frame_id: str = "world", topics: TopicConfig | None = None):
        self.node = node
        self.frame_id = frame_id
        self.topics = topics or TopicConfig()
        self.pub_path = node.create_publisher(Path, self.topics.global_path_topic, 1)
        self.pub_target = node.create_publisher(PointStamped, self.topics.path_target_topic, 1)
        self.pub_cf1_pos = node.create_publisher(PointStamped, self.topics.cf1_position_topic, 1)
        self.pub_cf2_target = node.create_publisher(PointStamped, self.topics.cf2_target_topic, 1)
        self.pub_cf3_target = node.create_publisher(PointStamped, self.topics.cf3_target_topic, 1)

    def publish_cf1_pos(self, xyz):
        self.pub_cf1_pos.publish(self._point_msg(xyz))

    def publish_target(self, xyz):
        self.pub_target.publish(self._point_msg(xyz))

    def publish_cf2_target(self, xyz):
        self.pub_cf2_target.publish(self._point_msg(xyz))

    def publish_cf3_target(self, xyz):
        self.pub_cf3_target.publish(self._point_msg(xyz))

    def publish_path(self, points: np.ndarray):
        msg = Path()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        for point in points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = float(point[2])
            msg.poses.append(pose)

        self.pub_path.publish(msg)

    def _point_msg(self, xyz):
        xyz = np.asarray(xyz, dtype=float).reshape(3,)
        msg = PointStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.point.x = float(xyz[0])
        msg.point.y = float(xyz[1])
        msg.point.z = float(xyz[2])
        return msg
