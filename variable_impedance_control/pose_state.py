from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NamedObstacle:
    name: str
    pos: np.ndarray

    @property
    def type(self) -> str:
        return self.name.split("_")[0].lower() if self.name else ""


@dataclass
class SwarmPoseState:
    leader: np.ndarray | None = None
    cf2: np.ndarray | None = None
    cf3: np.ndarray | None = None
    obstacles: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float))
    obstacles_named: list[NamedObstacle] = field(default_factory=list)

    def update_from_named_pose_array(self, msg, obstacle_names: list[str]):
        cf1_pose = self._find_pose(msg, "cf1")
        cf2_pose = self._find_pose(msg, "cf2")
        cf3_pose = self._find_pose(msg, "cf3")

        if cf1_pose is not None:
            self.leader = self._pose_xyz(cf1_pose)
        if cf2_pose is not None:
            self.cf2 = self._pose_xyz(cf2_pose)
        if cf3_pose is not None:
            self.cf3 = self._pose_xyz(cf3_pose)

        obstacles_named = []
        for obstacle_name in obstacle_names:
            obstacle_pose = self._find_pose(msg, obstacle_name)
            if obstacle_pose is None:
                continue
            pos = self._pose_xyz(obstacle_pose)
            if np.isfinite(pos).all():
                obstacles_named.append(NamedObstacle(obstacle_name, pos))

        self.obstacles_named = obstacles_named
        if obstacles_named:
            self.obstacles = np.array([o.pos for o in obstacles_named], dtype=float)
        else:
            self.obstacles = np.empty((0, 3), dtype=float)

    def follower_position(self, index: int) -> np.ndarray:
        if index == 1 and self.cf2 is not None:
            return np.array(self.cf2, dtype=float)
        if index == 2 and self.cf3 is not None:
            return np.array(self.cf3, dtype=float)
        if self.leader is None:
            return np.zeros(3, dtype=float)
        return np.array(self.leader, dtype=float)

    @staticmethod
    def _find_pose(msg, name: str):
        return next((pose for pose in msg.poses if pose.name == name), None)

    @staticmethod
    def _pose_xyz(named_pose) -> np.ndarray:
        pose = named_pose.pose.position
        return np.array([pose.x, pose.y, pose.z], dtype=float)
