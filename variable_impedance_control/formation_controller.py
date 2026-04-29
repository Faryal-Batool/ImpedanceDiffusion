from __future__ import annotations

import numpy as np

from .config import FormationConfig, ImpedanceConfig
from .impedance import ImpedanceController
from .obstacle_policy import ObstaclePolicy
from .pose_state import SwarmPoseState


class FormationController:
    def __init__(
        self,
        num_drones: int,
        formation_config: FormationConfig,
        impedance_config: ImpedanceConfig,
    ):
        self.num_drones = num_drones
        self.formation_config = formation_config
        self.impedance_config = impedance_config
        self.impedance = ImpedanceController()
        self.obstacle_policy = ObstaclePolicy(impedance_config, num_drones)

        self.separation_distance = formation_config.separation_distance
        self.imp_pose_prev_by_drone = {
            j: np.zeros(3, dtype=float) for j in range(1, num_drones)
        }
        self.imp_vel_prev_by_drone = {
            j: np.zeros(3, dtype=float) for j in range(1, num_drones)
        }

    def initial_positions(self, center_position: np.ndarray) -> np.ndarray:
        init_xyzs = np.zeros((self.num_drones, 3), dtype=float)
        init_xyzs[0] = center_position
        for drone_index in range(1, self.num_drones):
            offset = self._formation_offset(drone_index, self.formation_config.separation_distance)
            init_xyzs[drone_index] = center_position + offset
        return init_xyzs

    def compute_follower_targets(
        self,
        pose_state: SwarmPoseState,
        leader_velocity: np.ndarray,
        time_step: float,
    ) -> dict[int, np.ndarray]:
        if pose_state.leader is None:
            return {}

        targets: dict[int, np.ndarray] = {}
        obstacles_named = list(pose_state.obstacles_named)

        for drone_index in range(1, self.num_drones):
            follower_xyz = pose_state.follower_position(drone_index)
            near_human = self.obstacle_policy.update_human_flag(
                drone_index,
                follower_xyz,
                obstacles_named,
            )
            m_use, k_use, d_use = self._formation_impedance_params(near_human)

            imp_pose, imp_vel = self.impedance.impedance(
                leader_velocity,
                self.imp_pose_prev_by_drone[drone_index],
                self.imp_vel_prev_by_drone[drone_index],
                time_step,
                m=m_use,
                k=k_use,
                d=d_use,
            )
            self.imp_pose_prev_by_drone[drone_index] = imp_pose
            self.imp_vel_prev_by_drone[drone_index] = imp_vel

            offset = self._formation_offset(drone_index, self.separation_distance)
            drone_pose = np.array(pose_state.leader, dtype=float) + (0.2 * imp_pose) + offset

            drone_pose, obstacles_named = self._apply_obstacle_response(
                drone_index,
                drone_pose,
                obstacles_named,
                time_step,
            )
            targets[drone_index] = drone_pose

        pose_state.obstacles_named = obstacles_named
        if obstacles_named:
            pose_state.obstacles = np.array([o.pos for o in obstacles_named], dtype=float)
        else:
            pose_state.obstacles = np.empty((0, 3), dtype=float)

        return targets

    def _formation_offset(self, drone_index: int, separation_distance: float) -> np.ndarray:
        angle = (
            (2 * np.pi * (drone_index - 1)) / (self.num_drones - 1)
            + self.formation_config.rotation_deg / 57.3
        )
        return np.array(
            [
                separation_distance * np.cos(angle),
                separation_distance * np.sin(angle),
                0.0,
            ],
            dtype=float,
        )

    def _formation_impedance_params(self, near_human: bool):
        cfg = self.impedance_config
        if near_human:
            return cfg.drone_m_human, cfg.drone_k_human, cfg.drone_d_human
        return cfg.drone_m, cfg.drone_k, cfg.drone_d

    def _apply_obstacle_response(self, drone_index, drone_pose, obstacles_named, time_step: float):
        nearest, nearest_dist = self.obstacle_policy.nearest_obstacle(
            drone_pose[:2],
            obstacles_named,
        )
        if nearest is None:
            return drone_pose, obstacles_named

        obs_name = nearest["name"]
        obs_xy = nearest["pos"][:2]

        if nearest["kind"] == "gate_group":
            self.separation_distance = 0.40
            k, d, m, deflection_obs = self.impedance_config.obs_param_by_type["gate"]
        else:
            self.separation_distance = self.obstacle_policy.separation_for(obs_name)
            k, d, m, deflection_obs = self.obstacle_policy.params_for(obs_name)

        obs_radius = 0.08 + deflection_obs
        def_dist = deflection_obs * self.impedance_config.force_coeff
        if nearest_dist < float(obs_radius):
            self.obstacle_policy.active_obs_by_drone[drone_index] = obs_name
            print(f"Current obstacle is {obs_name}. Changing impedance parameters")
            drone_pose[:2] = self.impedance.impedance_obs_dynamic(
                drone_pose[:2],
                obs_xy,
                float(def_dist),
                float(k),
                float(d),
                float(m),
                float(time_step),
            )

        obstacles_named = self.obstacle_policy.prune_passed_obstacles(
            drone_index,
            drone_pose[:2],
            obstacles_named,
        )
        return drone_pose, obstacles_named
