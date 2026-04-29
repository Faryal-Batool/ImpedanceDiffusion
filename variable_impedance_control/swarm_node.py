from __future__ import annotations

import time

import numpy as np
import rclpy
from crazyflie_py.crazyflie import CrazyflieServer, TimeHelper
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.task import Future

from .apf import ArtificialPotentialField
from .config import SwarmExperimentConfig
from .experiment_logger import ExperimentLogger, ExperimentOutput, TrajectoryPlotter
from .formation_controller import FormationController
from .path_manager import GlobalPath
from .pose_state import SwarmPoseState
from .ros_publishers import SwarmPublishers


class APFImpedanceSwarmNode(Node):
    def __init__(self, config: SwarmExperimentConfig | None = None):
        super().__init__("APF_IMP")
        self.config = config or SwarmExperimentConfig()

        self.z_takeoff = float(self.config.path.z_takeoff)
        self.config.flight.center_position[2] = self.z_takeoff

        self.pose_state = SwarmPoseState()
        self.apf = ArtificialPotentialField()
        self.global_path = GlobalPath(
            self.config.path.resolve_csv_path(),
            z_takeoff=self.z_takeoff,
        )
        if len(self.global_path) < 2:
            self.get_logger().error(f"CSV path too short/invalid: '{self.global_path.csv_path}'")

        self.output = ExperimentOutput(self.config.output)
        self.logger = ExperimentLogger(self.output)
        self.plotter = TrajectoryPlotter(self.output)

        self._create_future_signals()
        self._create_crazyflie_handles()
        self._create_pose_subscription()
        self.publishers = SwarmPublishers(
            self,
            frame_id=self.config.path.frame_id,
            topics=self.config.topics,
        )

        self.num_drones = self._get_num_drones()
        self.formation = FormationController(
            self.num_drones,
            self.config.formation,
            self.config.impedance,
        )
        self.initial_positions = self.formation.initial_positions(
            self.config.flight.center_position
        )

        self.current_pos = None
        self.path_follower = None
        self.wp_idx = 0
        self.previous_leader = np.zeros(3, dtype=float)
        self.start_time = time.time()
        self.previous_time = 0.0
        self.timer = None

        self.publishers.publish_path(self.global_path.points)

    def _create_future_signals(self):
        self.check_takeoff = Future()
        self.check_target = Future()
        self.check_land = Future()

    def _create_crazyflie_handles(self):
        self.allcfs = CrazyflieServer()
        self.time_helper = TimeHelper(self.allcfs)
        if len(self.allcfs.crazyflies) < 1:
            raise RuntimeError("No Crazyflies found in CrazyflieServer().")

    def _create_pose_subscription(self):
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=0, nanoseconds=int(1e9 / 100.0)),
        )
        self.sub_pose = self.create_subscription(
            NamedPoseArray,
            self.config.topics.pose_topic,
            self.pose_callback,
            qos,
        )

    def _get_num_drones(self) -> int:
        try:
            return len(self.allcfs.crazyflies)
        except Exception:
            return 3

    def pose_callback(self, msg: NamedPoseArray):
        self.pose_state.update_from_named_pose_array(msg, self.config.obstacles.names)

        if self.pose_state.leader is not None:
            self.publishers.publish_cf1_pos(self.pose_state.leader)

        current_time = time.time() - self.start_time
        self.logger.record(
            current_time,
            self.pose_state.leader,
            self.pose_state.cf2,
            self.pose_state.cf3,
            self.pose_state.obstacles,
        )

    def takeoff(self):
        target_height = self.z_takeoff
        for drone_index in range(self.num_drones):
            self.allcfs.crazyflies[drone_index].takeoff(
                targetHeight=target_height,
                duration=1.0 + target_height,
            )

        leader = self.pose_state.leader
        if leader is not None:
            error = np.linalg.norm(np.array(leader) - self.config.flight.center_position)
            if error < 0.25:
                self.check_takeoff.set_result(True)

    def init_apf_imp(self):
        self.start_time = time.time()
        self.previous_time = 0.0
        self.previous_leader = (
            np.array(self.pose_state.leader, dtype=float)
            if self.pose_state.leader is not None
            else np.zeros(3, dtype=float)
        )
        self.path_follower = self.global_path.points
        self.get_logger().info("Reading global path")

    def run(self):
        if self.check_target.done():
            print("Reached Target")
            return

        if self.pose_state.leader is None or self.path_follower is None:
            print("Invalid Path")
            return

        self.current_pos = np.array(self.pose_state.leader, dtype=float)
        self.publishers.publish_path(self.global_path.points)

        current_time = time.time() - self.start_time
        goal = self._current_waypoint_goal()

        v_local = self.apf.compute_local_velocity(
            cur_xyz=self.current_pos,
            waypoint_xyz=goal,
            obstacles_xyz=self.pose_state.obstacles,
            gain=0.1,
        )
        target_leader = np.array(
            [
                self.current_pos[0] + v_local[0],
                self.current_pos[1] + v_local[1],
                self.z_takeoff,
            ],
            dtype=float,
        )

        self.allcfs.crazyflies[0].goTo(
            target_leader,
            0,
            self.config.formation.leader_command_duration,
        )
        self.publishers.publish_target(target_leader)

        dt = max(float(current_time - self.previous_time), 1e-3)
        leader_velocity = (self.current_pos - self.previous_leader) / dt
        follower_targets = self.formation.compute_follower_targets(
            self.pose_state,
            leader_velocity,
            dt,
        )
        self._send_follower_targets(follower_targets)

        self.previous_leader = self.current_pos
        self.previous_time = current_time

        if self._at_end_of_path():
            self.get_logger().info("[DONE] REACHED END OF PATH")
            self.check_target.set_result(True)

    def _current_waypoint_goal(self):
        goal = self.global_path.waypoint(self.wp_idx)
        distance_to_goal = np.linalg.norm(goal[:2] - self.current_pos[:2])
        if distance_to_goal <= float(self.config.flight.reach_radius):
            self.wp_idx = min(self.wp_idx + 3, len(self.global_path) - 1)
            goal = self.global_path.waypoint(self.wp_idx)
        return goal

    def _send_follower_targets(self, follower_targets: dict[int, np.ndarray]):
        for drone_index, target in follower_targets.items():
            target = np.array([target[0], target[1], self.z_takeoff], dtype=float)
            if drone_index == 1:
                self.publishers.publish_cf2_target(target)
            elif drone_index == 2:
                self.publishers.publish_cf3_target(target)

            self.allcfs.crazyflies[drone_index].goTo(
                target,
                0,
                self.config.formation.follower_command_duration,
            )

    def _at_end_of_path(self) -> bool:
        end_xy = self.global_path.final_xy
        return (
            float(np.linalg.norm(end_xy - self.current_pos[:2]))
            <= float(self.config.flight.goal_tolerance)
        )

    def land(self):
        target_height = self.z_takeoff
        self.allcfs.land(targetHeight=0.02, duration=1.0 + target_height)
        self.check_land.set_result(True)

    def finish_outputs(self):
        self.logger.dump()
        self.plotter.plot(self.logger, self.global_path.points)
        self.get_logger().info(f"[OUTPUT] Saved logs and graphs in: {self.output.experiment_dir}")


def run_experiment(config: SwarmExperimentConfig | None = None):
    rclpy.init(args=None)
    node = None

    try:
        node = APFImpedanceSwarmNode(config)
        rate = node.config.flight.control_rate_hz

        node.timer = node.create_timer(1.0 / rate, node.takeoff)
        rclpy.spin_until_future_complete(node, node.check_takeoff)
        node.timer.cancel()
        print("Takeoff Successful")

        print("Initiating Diffusion APF Loop")
        node.init_apf_imp()

        print("Initiating Path follower")
        node.timer = node.create_timer(1.0 / rate, node.run)
        rclpy.spin_until_future_complete(node, node.check_target)
        node.timer.cancel()
        print("Reached Target")

        node.timer = node.create_timer(1.0 / rate, node.land)
        rclpy.spin_until_future_complete(node, node.check_land)
        node.timer.cancel()

        node.finish_outputs()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
