from __future__ import annotations

import numpy as np


class ImpedanceController:
    def mass_spring_damper(self, state, force, m=0.1, d=0.2, k=0.2):
        position = state[0]
        velocity = state[1]
        m = float(m)
        d = float(d)
        k = float(k)
        acceleration = -d / m * velocity + (-k / m) * position + force / m
        return [velocity, acceleration]

    def MassSpringDamper(self, state, t, F, m=0.1, d=0.2, k=0.2):
        return self.mass_spring_damper(state, F, m=m, d=d, k=k)

    def impedance_obs(self, curr_drone_pos, obstacle_center, rr_imp):
        force_coeff = 0.42
        curr_drone_pos = np.asarray(curr_drone_pos, dtype=float).copy()
        obstacle_center = np.asarray(obstacle_center, dtype=float).copy()

        direction_to_center = obstacle_center - curr_drone_pos
        norm = np.linalg.norm(direction_to_center)
        if norm < 1e-6:
            return curr_drone_pos

        direction_to_center /= norm
        deflection_distance = force_coeff * rr_imp
        curr_drone_pos -= deflection_distance * direction_to_center + 0.01
        return curr_drone_pos

    def impedance_obs_dynamic(
        self,
        curr_xy: np.ndarray,
        obstacle_xy: np.ndarray,
        deflection_distance: float,
        k: float = 16.0,
        d: float = 4.0,
        m: float = 1.0,
        time_step: float = 1.0,
    ) -> np.ndarray:
        curr_xy = np.asarray(curr_xy, dtype=float).reshape(2,)
        obstacle_xy = np.asarray(obstacle_xy, dtype=float).reshape(2,)
        deflection_distance = float(deflection_distance)
        time_step = max(float(time_step), 1e-3)
        m = max(float(m), 1e-6)
        d = float(d)
        k = float(k)

        offset = curr_xy - obstacle_xy
        distance = float(np.linalg.norm(offset))
        if distance < 1e-6:
            offset = np.array([1.0, 0.0], dtype=float)
            distance = 1e-6

        if distance >= deflection_distance:
            return curr_xy


        # 1) compute the obstacle-normal deflection delta inside the active region
        # 2) evaluate the mass-spring-damper virtual force along the normal
        # 3) convert that force into a discrete-time position correction
        normal_hat = offset / distance
        delta = deflection_distance - distance

        # The paper uses a mass-spring-damper force F_n = k_o*delta + d_o*delta_dot + m_o*delta_ddot.
        # This implementation keeps the response discrete-time and stateless, so the derivative terms
        # are approximated as zero and the displacement is obtained from the virtual force magnitude.
        delta_dot = delta / time_step
        delta_ddot = delta_dot / time_step
        force_normal = (k * delta) + (d * delta_dot) + (m * delta_ddot)

        displacement = 0.5 * (force_normal / m) * (time_step ** 2) * normal_hat
        return curr_xy + displacement

    def impedance(
        self,
        leader_vel,
        imp_pose_prev,
        imp_vel_prev,
        time_step,
        m=5.0,
        k=0.1,
        d=2.0,
    ):
        force_coeff = 0.45
        leader_vel = np.asarray(leader_vel, dtype=float).reshape(-1,)
        force = 0.01 * force_coeff * leader_vel
        time_step = float(time_step)

        imp_pose = np.zeros_like(imp_pose_prev, dtype=float)
        imp_vel = np.zeros_like(imp_vel_prev, dtype=float)

        for axis in range(len(leader_vel)):
            state0 = [imp_pose_prev[axis], imp_vel_prev[axis]]
            state = self.mass_spring_damper(state0, force[axis], m=m, d=d, k=k)
            imp_pose[axis] = state[0] * time_step
            imp_vel[axis] = state[1] * time_step

        return imp_pose, imp_vel


IMPEDANCE = ImpedanceController
