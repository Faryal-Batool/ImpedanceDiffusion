from __future__ import annotations

import numpy as np

from .config import ImpedanceConfig
from .pose_state import NamedObstacle


class ObstaclePolicy:
    def __init__(self, config: ImpedanceConfig, num_drones: int):
        self.config = config
        self.active_obs_by_drone: dict[int, str] = {}
        self.passed_obs: set[str] = set()
        self.near_human_by_drone = {j: False for j in range(1, num_drones)}

    def params_for(self, obs_name: str) -> tuple[float, float, float, float]:
        obs_type = self.type_from_name(obs_name)
        return self.config.obs_param_by_type.get(obs_type, self.config.default_obstacle_params)

    def separation_for(self, obs_name: str) -> float:
        obs_type = self.type_from_name(obs_name)
        if obs_type == "human":
            return 0.55
        if obs_type == "gate":
            return 0.40
        if obs_type == "chair":
            return 0.55
        return 0.50

    def gate_group(self, obstacles: list[NamedObstacle]):
        gates = [o for o in obstacles if o.name.startswith("gate_")]
        if len(gates) < 2:
            return None
        midpoint = 0.5 * (gates[0].pos[:2] + gates[1].pos[:2])
        return {
            "kind": "gate_group",
            "name": "gate_group",
            "pos": midpoint,
            "members": [gate.name for gate in gates],
        }

    def update_human_flag(self, drone_index: int, drone_xyz: np.ndarray, obstacles):
        dmin = float("inf")
        for obstacle in obstacles:
            if obstacle.name.startswith("human_"):
                dmin = min(dmin, float(np.linalg.norm(drone_xyz[:2] - obstacle.pos[:2])))

        previous = bool(self.near_human_by_drone.get(drone_index, False))
        if not np.isfinite(dmin):
            current = False
        elif previous:
            current = dmin <= float(self.config.human_exit_dist)
        else:
            current = dmin <= float(self.config.human_enter_dist)

        if current != previous:
            mode = "HUMAN" if current else "HARD"
            print(
                f"[IMPEDANCE SWITCH] Drone {drone_index} -> {mode} "
                f"impedance (d_min={dmin:.2f})"
            )

        self.near_human_by_drone[drone_index] = current
        return current

    def nearest_obstacle(self, drone_xy: np.ndarray, obstacles: list[NamedObstacle]):
        nearest = None
        nearest_dist = float("inf")

        for obstacle in obstacles:
            if obstacle.name.startswith("gate_"):
                continue
            distance = float(np.linalg.norm(drone_xy - obstacle.pos[:2]))
            if distance < nearest_dist:
                nearest_dist = distance
                nearest = {
                    "kind": "single",
                    "name": obstacle.name,
                    "pos": obstacle.pos,
                }

        gate = self.gate_group(obstacles)
        if gate is not None:
            distance = float(np.linalg.norm(drone_xy - gate["pos"]))
            if distance < nearest_dist:
                nearest_dist = distance
                nearest = gate

        return nearest, nearest_dist

    def prune_passed_obstacles(
        self,
        drone_index: int,
        drone_xy: np.ndarray,
        obstacles: list[NamedObstacle],
    ) -> list[NamedObstacle]:
        active = self.active_obs_by_drone.get(drone_index)
        if active is None:
            return obstacles

        if active == "gate_group":
            gate = self.gate_group(obstacles)
            if gate is None:
                self.active_obs_by_drone.pop(drone_index, None)
                return obstacles

            distance = float(np.linalg.norm(drone_xy - gate["pos"]))
            exit_dist = float(self.config.obs_param_by_type["gate"][3]) + self.config.obs_exit_margin
            if distance > exit_dist:
                members = set(gate["members"])
                self.passed_obs.update(members)
                self.active_obs_by_drone.pop(drone_index, None)
                return [o for o in obstacles if o.name not in members]
            return obstacles

        active_obj = next((o for o in obstacles if o.name == active), None)
        if active_obj is None:
            self.active_obs_by_drone.pop(drone_index, None)
            return obstacles

        distance = float(np.linalg.norm(drone_xy - active_obj.pos[:2]))
        exit_dist = float(self.params_for(active)[3]) + self.config.obs_exit_margin
        if distance > exit_dist:
            self.passed_obs.add(active)
            self.active_obs_by_drone.pop(drone_index, None)
            return [o for o in obstacles if o.name != active]

        return obstacles

    @staticmethod
    def type_from_name(name: str) -> str:
        return name.split("_")[0].lower() if name else ""
