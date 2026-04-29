#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from variable_impedance_control.config import (
    ObstacleConfig,
    OutputConfig,
    PathConfig,
    SwarmExperimentConfig,
    TopicConfig,
)


def build_config(args) -> SwarmExperimentConfig:
    config = SwarmExperimentConfig()
    if args.csv_path:
        config.path = PathConfig(
            csv_path=Path(args.csv_path),
            z_takeoff=args.z_takeoff,
            frame_id=args.frame_id,
        )
    else:
        config.path.z_takeoff = args.z_takeoff
        config.path.frame_id = args.frame_id

    config.output = OutputConfig(
        experiment_name=args.experiment,
        base_dir=Path(args.output_dir) if args.output_dir else None,
        timestamped=not args.no_timestamp,
    )

    if args.obstacles:
        config.obstacles = ObstacleConfig(
            names=[name.strip() for name in args.obstacles.split(",") if name.strip()]
        )

    if args.center:
        config.flight.center_position = np.array(
            [float(value) for value in args.center.split(",")],
            dtype=float,
        )

    config.flight.control_rate_hz = args.control_rate
    config.flight.reach_radius = args.reach_radius
    config.flight.goal_tolerance = args.goal_tolerance

    config.topics = TopicConfig(
        pose_topic=args.pose_topic,
        global_path_topic=args.global_path_topic,
        path_target_topic=args.path_target_topic,
        cf1_position_topic=args.cf1_position_topic,
        cf2_target_topic=args.cf2_target_topic,
        cf3_target_topic=args.cf3_target_topic,
    )

    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Run the refactored APF + impedance swarm experiment.")
    parser.add_argument("--csv-path", default=None, help="Global path CSV with x_m, y_m, z_m columns.")
    parser.add_argument("--experiment", default="Exp_9_dynamic", help="Experiment/output name.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base directory for logs and graphs. Defaults to SWARM_OUTPUT_DIR or ./output_results beside this runner.",
    )
    parser.add_argument("--z-takeoff", type=float, default=1.0, help="Takeoff and path-following altitude.")
    parser.add_argument("--frame-id", default="world", help="ROS frame id for published path/points.")
    parser.add_argument(
        "--obstacles",
        default=None,
        help="Comma-separated mocap obstacle names, for example human_1,trolley_1.",
    )
    parser.add_argument("--no-timestamp", action="store_true", help="Write directly into output-dir/experiment.")
    parser.add_argument(
        "--center",
        default=None,
        help="Takeoff center as x,y,z. If omitted, uses the config default.",
    )
    parser.add_argument("--control-rate", type=float, default=24.0, help="Control timer rate in Hz.")
    parser.add_argument("--reach-radius", type=float, default=0.3, help="Waypoint reach radius in meters.")
    parser.add_argument("--goal-tolerance", type=float, default=0.10, help="Final path tolerance in meters.")
    parser.add_argument("--pose-topic", default="/poses", help="NamedPoseArray mocap topic.")
    parser.add_argument("--global-path-topic", default="/global_path", help="Published nav_msgs/Path topic.")
    parser.add_argument("--path-target-topic", default="/path_target", help="Published leader target topic.")
    parser.add_argument("--cf1-position-topic", default="/cf1_position", help="Published cf1 position topic.")
    parser.add_argument("--cf2-target-topic", default="/cf2_target", help="Published cf2 target topic.")
    parser.add_argument("--cf3-target-topic", default="/cf3_target", help="Published cf3 target topic.")
    return parser.parse_args()


def main():
    args = parse_args()
    from variable_impedance_control.swarm_node import run_experiment

    run_experiment(build_config(args))


if __name__ == "__main__":
    main()
