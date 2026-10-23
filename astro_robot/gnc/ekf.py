from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .math3d import (
    euler_from_quat,
    quat_from_euler,
    quat_from_rotvec,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
    roll_pitch_from_accel,
    skew,
)


@dataclass
class EKFEstimate:
    quaternion: np.ndarray
    gyro_bias: np.ndarray
    omega_rps: np.ndarray
    roll: float
    pitch: float
    yaw: float
    accel_used: bool
    accel_norm_g: float
    covariance_trace: float


class AttitudeMEKF:
    """Small multiplicative/error-state EKF for body attitude.

    Nominal state: quaternion q_BW (body -> world) and gyro bias b_g.
    Error state: [delta_theta(3), delta_bias(3)].

    Gyro propagation runs every cycle. A normalized accelerometer gravity vector
    corrects roll/pitch only when its norm is sufficiently close to 1 g. Yaw is
    intentionally not magnetometer-corrected because motors/servo wiring can
    strongly disturb a magnetometer on a small robot.
    """

    def __init__(
        self,
        gyro_noise: float = 0.035,
        bias_rw: float = 0.002,
        accel_noise: float = 0.055,
        accel_gate_min: float = 0.75,
        accel_gate_max: float = 1.25,
    ):
        self.gyro_noise = float(gyro_noise)
        self.bias_rw = float(bias_rw)
        self.accel_noise = float(accel_noise)
        self.accel_gate_min = float(accel_gate_min)
        self.accel_gate_max = float(accel_gate_max)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.bias = np.zeros(3, dtype=float)
        self.P = np.diag([math.radians(8)**2]*3 + [math.radians(2)**2]*3)
        self.initialized = False
        self.last_omega = np.zeros(3, dtype=float)
        self.last_accel_used = False
        self.last_accel_norm = 1.0

    def reset(self) -> None:
        self.__init__(self.gyro_noise, self.bias_rw, self.accel_noise, self.accel_gate_min, self.accel_gate_max)

    def initialize_from_accel(self, accel_g: np.ndarray) -> None:
        roll, pitch = roll_pitch_from_accel(accel_g)
        self.q = quat_from_euler(roll, pitch, 0.0)
        self.initialized = True

    def set_gyro_bias(self, bias_rps: np.ndarray) -> None:
        self.bias = np.asarray(bias_rps, dtype=float).reshape(3).copy()

    def predict(self, gyro_rps: np.ndarray, dt: float) -> None:
        gyro = np.asarray(gyro_rps, dtype=float).reshape(3)
        dt = float(np.clip(dt, 1e-4, 0.2))
        omega = gyro - self.bias
        self.last_omega = omega.copy()
        self.q = quat_normalize(quat_multiply(self.q, quat_from_rotvec(omega * dt)))

        F = np.zeros((6, 6), dtype=float)
        F[:3, :3] = -skew(omega)
        F[:3, 3:] = -np.eye(3)
        Phi = np.eye(6) + F * dt
        q_theta = (self.gyro_noise ** 2) * dt
        q_bias = (self.bias_rw ** 2) * dt
        Qd = np.diag([q_theta]*3 + [q_bias]*3)
        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = 0.5 * (self.P + self.P.T)

    def _gravity_body(self, q: np.ndarray | None = None) -> np.ndarray:
        R_bw = quat_to_rotmat(self.q if q is None else q)
        return R_bw.T @ np.array([0.0, 0.0, 1.0], dtype=float)

    def update_accel(self, accel_g: np.ndarray) -> bool:
        a = np.asarray(accel_g, dtype=float).reshape(3)
        norm = float(np.linalg.norm(a))
        self.last_accel_norm = norm
        if not np.isfinite(norm) or norm < self.accel_gate_min or norm > self.accel_gate_max:
            self.last_accel_used = False
            return False
        z = a / norm
        h = self._gravity_body()

        # Numerical Jacobian of gravity direction with respect to the local
        # right-multiplicative attitude error. It avoids convention/sign bugs
        # while remaining trivial computationally for a 3-axis measurement.
        eps = 1e-5
        Htheta = np.empty((3, 3), dtype=float)
        for i in range(3):
            d = np.zeros(3); d[i] = eps
            q2 = quat_normalize(quat_multiply(self.q, quat_from_rotvec(d)))
            Htheta[:, i] = (self._gravity_body(q2) - h) / eps
        H = np.zeros((3, 6), dtype=float)
        H[:, :3] = Htheta

        dynamic_factor = 1.0 + 12.0 * abs(norm - 1.0)
        Rm = np.eye(3) * (self.accel_noise * dynamic_factor) ** 2
        innovation = z - h
        S = H @ self.P @ H.T + Rm
        try:
            K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            self.last_accel_used = False
            return False
        dx = K @ innovation
        dtheta, dbias = dx[:3], dx[3:]
        max_corr = math.radians(12)
        n = float(np.linalg.norm(dtheta))
        if n > max_corr:
            dtheta *= max_corr / n
        self.q = quat_normalize(quat_multiply(self.q, quat_from_rotvec(dtheta)))
        self.bias += dbias
        I = np.eye(6)
        # Joseph covariance update preserves PSD better than (I-KH)P.
        A = I - K @ H
        self.P = A @ self.P @ A.T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.last_accel_used = True
        return True

    def step(self, gyro_rps: np.ndarray, accel_g: np.ndarray, dt: float) -> EKFEstimate:
        if not self.initialized:
            self.initialize_from_accel(accel_g)
        self.predict(gyro_rps, dt)
        self.update_accel(accel_g)
        roll, pitch, yaw = euler_from_quat(self.q)
        return EKFEstimate(
            quaternion=self.q.copy(),
            gyro_bias=self.bias.copy(),
            omega_rps=self.last_omega.copy(),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            accel_used=self.last_accel_used,
            accel_norm_g=self.last_accel_norm,
            covariance_trace=float(np.trace(self.P)),
        )
