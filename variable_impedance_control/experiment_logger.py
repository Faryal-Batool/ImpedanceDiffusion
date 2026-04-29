from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python 3.8 compatibility on older ROS setups.
    ZoneInfo = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import OutputConfig


class ExperimentOutput:
    def __init__(self, config: OutputConfig):
        self.config = config
        self.experiment_dir = self._create_experiment_dir()
        self.logs_dir = self.experiment_dir / "logs"
        self.graphs_dir = self.experiment_dir / "graphs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

        self.drone_poses_txt = self.logs_dir / f"drone_poses_{config.experiment_name}.txt"
        self.obstacles_txt = self.logs_dir / f"obstacles_{config.experiment_name}.txt"
        self.drone_poses_csv = self.logs_dir / f"drone_poses_{config.experiment_name}.csv"
        self.obstacles_csv = self.logs_dir / f"obstacles_{config.experiment_name}.csv"
        self.plot_file = self.graphs_dir / "trajectories_exp.png"

    def _create_experiment_dir(self) -> Path:
        base_dir = self.config.resolve_base_dir()
        if self.config.timestamped:
            timezone = ZoneInfo("Europe/Moscow") if ZoneInfo is not None else None
            timestamp = datetime.now(timezone).strftime("%Y-%m-%d_%H-%M-%S")
            dirname = f"{self.config.experiment_name}_{timestamp}"
        else:
            dirname = self.config.experiment_name
        path = base_dir / dirname
        path.mkdir(parents=True, exist_ok=True)
        return path


class ExperimentLogger:
    def __init__(self, output: ExperimentOutput):
        self.output = output
        self.t_log = []
        self.leader_log = []
        self.cf2_log = []
        self.cf3_log = []
        self.obs_log = []

    def record(self, current_time, leader, cf2, cf3, obstacles):
        self.t_log.append(float(current_time))
        self.leader_log.append(self._xyz_or_nan(leader))
        self.cf2_log.append(self._xyz_or_nan(cf2))
        self.cf3_log.append(self._xyz_or_nan(cf3))
        if obstacles is None:
            self.obs_log.append(np.empty((0, 3), dtype=float))
        else:
            self.obs_log.append(np.array(obstacles, dtype=float).reshape(-1, 3))

    def dump(self):
        self._write_drone_poses_txt()
        self._write_obstacles_txt()
        self._write_drone_poses_csv()
        self._write_obstacles_csv()

    def _write_drone_poses_txt(self):
        with self.output.drone_poses_txt.open("w", encoding="utf-8") as file:
            for t, leader, cf2, cf3 in zip(
                self.t_log,
                self.leader_log,
                self.cf2_log,
                self.cf3_log,
            ):
                file.write(
                    f"Time: {t:.4f}, "
                    f"Leader: [{leader[0]:.4f}, {leader[1]:.4f}, {leader[2]:.4f}], "
                    f"CF2: [{cf2[0]:.4f}, {cf2[1]:.4f}, {cf2[2]:.4f}], "
                    f"CF3: [{cf3[0]:.4f}, {cf3[1]:.4f}, {cf3[2]:.4f}]\n"
                )

    def _write_obstacles_txt(self):
        with self.output.obstacles_txt.open("w", encoding="utf-8") as file:
            for t, obs in zip(self.t_log, self.obs_log):
                file.write(f"Time: {t:.4f}\n")
                file.write("Obstacles:\n")
                if obs is None or len(obs) == 0:
                    file.write("[]\n\n")
                else:
                    for row in obs:
                        file.write(f"  [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}]\n")
                    file.write("\n")

    def _write_drone_poses_csv(self):
        with self.output.drone_poses_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "time",
                    "leader_x",
                    "leader_y",
                    "leader_z",
                    "cf2_x",
                    "cf2_y",
                    "cf2_z",
                    "cf3_x",
                    "cf3_y",
                    "cf3_z",
                ]
            )
            for t, leader, cf2, cf3 in zip(
                self.t_log,
                self.leader_log,
                self.cf2_log,
                self.cf3_log,
            ):
                writer.writerow([t, *leader, *cf2, *cf3])

    def _write_obstacles_csv(self):
        with self.output.obstacles_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "obstacle_index", "x", "y", "z"])
            for t, obs in zip(self.t_log, self.obs_log):
                if obs is None or len(obs) == 0:
                    continue
                for index, row in enumerate(obs):
                    writer.writerow([t, index, row[0], row[1], row[2]])

    @staticmethod
    def _xyz_or_nan(position):
        if position is None:
            return [np.nan, np.nan, np.nan]
        return [float(position[0]), float(position[1]), float(position[2])]


class TrajectoryPlotter:
    def __init__(self, output: ExperimentOutput):
        self.output = output

    def plot(self, logger: ExperimentLogger, path_points: np.ndarray):
        leader = np.array(logger.leader_log, dtype=float)
        cf2 = np.array(logger.cf2_log, dtype=float)
        cf3 = np.array(logger.cf3_log, dtype=float)

        plt.figure(figsize=(10, 8))
        if path_points is not None and len(path_points) > 0:
            plt.plot(path_points[:, 0], path_points[:, 1], linewidth=2, label="Global path")

        if len(leader):
            plt.plot(leader[:, 0], leader[:, 1], label="Leader (cf1)")
        if len(cf2):
            plt.plot(cf2[:, 0], cf2[:, 1], label="CF2")
        if len(cf3):
            plt.plot(cf3[:, 0], cf3[:, 1], label="CF3")

        obs_xy_all = []
        for obs in logger.obs_log:
            if obs is not None and len(obs) > 0:
                obs_xy_all.append(obs[:, :2])

        if obs_xy_all:
            obs_xy_all = np.vstack(obs_xy_all)
            plt.scatter(obs_xy_all[:, 0], obs_xy_all[:, 1], marker="x", s=30, label="Obstacle(s)")

        plt.title("Global path + swarm trajectories + obstacles (XY)")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()
        plt.savefig(self.output.plot_file, dpi=200, bbox_inches="tight")
        plt.close()
