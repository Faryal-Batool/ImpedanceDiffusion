from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def refactored_code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return refactored_code_root().parent


@dataclass
class PathConfig:
    csv_path: Path | None = None
    z_takeoff: float = 1.0
    frame_id: str = "world"

    def resolve_csv_path(self) -> Path:
        candidates = []
        if self.csv_path is not None:
            candidates.append(Path(self.csv_path))

        env_csv = os.environ.get("SWARM_GLOBAL_PATH_CSV")
        if env_csv:
            candidates.append(Path(env_csv))

        root = refactored_code_root()
        candidates.extend(
            [
                Path.cwd() / "global_path_60.csv",
                Path.cwd() / "Global_paths" / "exp_9_global_path_60.csv",
                root / "global_path_60.csv",
                root / "Global_paths" / "exp_9_global_path_60.csv",
                root.parent / "Experiments_for_paper" / "global_path_60.csv",
                root.parent / "Experiments_for_paper" / "Global_paths" / "exp_9_global_path_60.csv",
            ]
        )

        for candidate in candidates:
            candidate = candidate.expanduser()
            if candidate.exists():
                return candidate.resolve()

        searched = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            "No global path CSV found. Pass --csv-path or set SWARM_GLOBAL_PATH_CSV.\n"
            f"Searched:\n{searched}"
        )


@dataclass
class OutputConfig:
    experiment_name: str = "Exp_9_dynamic"
    base_dir: Path | None = None
    timestamped: bool = True

    def resolve_base_dir(self) -> Path:
        if self.base_dir is not None:
            return Path(self.base_dir).expanduser().resolve()

        env_output_dir = os.environ.get("SWARM_OUTPUT_DIR")
        if env_output_dir:
            return Path(env_output_dir).expanduser().resolve()

        return (refactored_code_root() / "output_results").resolve()


@dataclass
class FlightConfig:
    control_rate_hz: float = 24.0
    reach_radius: float = 0.3
    goal_tolerance: float = 0.10
    alt_tolerance: float = 0.08
    center_position: np.ndarray = field(
        default_factory=lambda: np.array([1.8, -1.5, 1.0], dtype=float)
    )


@dataclass
class FormationConfig:
    separation_distance: float = 0.55
    rotation_deg: float = 90.0
    follower_command_duration: float = 0.05
    leader_command_duration: float = 0.3


@dataclass
class ImpedanceConfig:
    drone_k: float = 7.0
    drone_d: float = 3.0
    drone_m: float = 1.0
    drone_k_human: float = 0.1
    drone_d_human: float = 2.0
    drone_m_human: float = 5.0
    obs_k: float = 9.0
    obs_d: float = 5.0
    obs_m: float = 1.0
    obs_def: float = 0.65
    force_coeff: float = 0.45
    human_enter_dist: float = 0.90
    human_exit_dist: float = 1.10
    obs_exit_margin: float = 0.08
    obs_param_by_type: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "cylinder": (9.0, 5.0, 1.0, 0.65),
            "gate": (8.0, 3.0, 1.2, 0.45),
            "chair": (10.0, 5.5, 0.8, 0.8),
            "trolley": (5.0, 3.0, 0.8, 1.20),
            "human": (16.0, 4.0, 1.0, 1.0),
        }
    )

    @property
    def default_obstacle_params(self) -> tuple[float, float, float, float]:
        return (
            self.obs_def * self.force_coeff,
            self.obs_k,
            self.obs_d,
            self.obs_m,
        )


@dataclass
class ObstacleConfig:
    names: list[str] = field(default_factory=lambda: ["human_1", "trolley_1"])


@dataclass
class TopicConfig:
    pose_topic: str = "/poses"
    global_path_topic: str = "/global_path"
    path_target_topic: str = "/path_target"
    cf1_position_topic: str = "/cf1_position"
    cf2_target_topic: str = "/cf2_target"
    cf3_target_topic: str = "/cf3_target"


@dataclass
class SwarmExperimentConfig:
    path: PathConfig = field(default_factory=PathConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    flight: FlightConfig = field(default_factory=FlightConfig)
    formation: FormationConfig = field(default_factory=FormationConfig)
    impedance: ImpedanceConfig = field(default_factory=ImpedanceConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    topics: TopicConfig = field(default_factory=TopicConfig)
