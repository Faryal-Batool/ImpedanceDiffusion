from __future__ import annotations

import numpy as np


class ArtificialPotentialField:
    """Artificial potential field local planner."""

    def __init__(
        self,
        attraction_gain: float = 1.5,
        repulsion_gain: float = 1.5,
        influence_distance: float = 0.45,
        max_velocity: float = 1.5,
    ):
        self.zeta = attraction_gain
        self.eta = repulsion_gain
        self.d_0 = influence_distance
        self.max_v = max_velocity

    def getRepulsiveForce(self, cur_pos: np.ndarray, obstacles: np.ndarray):
        if obstacles is None or len(obstacles) == 0:
            return np.zeros(2, dtype=float)

        cur_pos = np.asarray(cur_pos, dtype=float).reshape(2,)
        obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 2)

        distances = np.linalg.norm(obstacles - cur_pos[np.newaxis, :], axis=1, keepdims=True)
        distances = np.clip(distances, 1e-6, None)

        rep_force = (
            (1 / distances - 1 / self.d_0)
            * (1 / distances) ** 2
            * (cur_pos - obstacles)
        )
        valid_mask = np.argwhere((1 / distances - 1 / self.d_0) > 0)[:, 0]
        if valid_mask.size == 0:
            return np.zeros(2, dtype=float)

        rep_force = np.sum(rep_force[valid_mask, :], axis=0)
        norm = np.linalg.norm(rep_force)
        if norm > 0.0:
            rep_force = rep_force / norm
        return rep_force

    def getAttractiveForce(self, cur_pos: np.ndarray, tgt_pos: np.ndarray):
        cur_pos = np.asarray(cur_pos, dtype=float).reshape(-1,)
        tgt_pos = np.asarray(tgt_pos, dtype=float).reshape(-1,)
        attr_force = tgt_pos - cur_pos
        norm = np.linalg.norm(attr_force)
        if norm > 0.0:
            attr_force = attr_force / norm
        return attr_force

    def compute_local_velocity(
        self,
        cur_xyz: np.ndarray,
        waypoint_xyz: np.ndarray,
        obstacles_xyz: np.ndarray,
        gain: float = 0.01,
    ) -> np.ndarray:
        cur_xyz = np.asarray(cur_xyz, dtype=float).reshape(3,)
        waypoint_xyz = np.asarray(waypoint_xyz, dtype=float).reshape(3,)

        attr = self.getAttractiveForce(cur_xyz, waypoint_xyz)
        if obstacles_xyz is not None and len(obstacles_xyz):
            obstacle_xy = np.asarray(obstacles_xyz, dtype=float)[:, :2]
        else:
            obstacle_xy = np.empty((0, 2), dtype=float)
        rep_xy = self.getRepulsiveForce(cur_xyz[:2], obstacle_xy)
        rep = np.array([rep_xy[0], rep_xy[1], 0.0], dtype=float)

        velocity = (self.zeta * attr + self.eta * rep) * gain
        norm = np.linalg.norm(velocity)
        if norm > self.max_v:
            velocity = (velocity / norm) * self.max_v
        return velocity


APF = ArtificialPotentialField
