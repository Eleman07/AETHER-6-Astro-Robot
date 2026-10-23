from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class LQROutput:
    command_roll_rad: float
    command_pitch_rad: float
    state_error: np.ndarray
    saturated: bool


def solve_discrete_are_iterative(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, *, iterations: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """Solve the DARE by Riccati fixed-point iteration using NumPy only."""
    P = Q.copy().astype(float)
    for _ in range(iterations):
        S = R + B.T @ P @ B
        try:
            K = np.linalg.solve(S, B.T @ P @ A)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("LQR Riccati solve failed") from exc
        Pn = A.T @ P @ A - A.T @ P @ B @ K + Q
        Pn = 0.5 * (Pn + Pn.T)
        if np.linalg.norm(Pn - P, ord="fro") < tol:
            return Pn
        P = Pn
    return P


class BodyAttitudeLQR:
    """2-axis LQR for body roll/pitch stabilization.

    State x = [roll_error, pitch_error, roll_rate, pitch_rate].
    Input u = vendor body-attitude correction [roll_cmd, pitch_cmd] in radians.

    The plant model is deliberately explicit/configurable because the direct Astro
    TCP protocol does not expose joint torques or contact forces. `actuator_gain`
    therefore represents the experimentally identified angular-acceleration per
    radian of body-attitude correction around the level operating point.
    """

    def __init__(
        self,
        dt: float = 0.04,
        actuator_gain: float = 10.0,
        damping: float = 2.4,
        max_command_deg: float = 4.0,
        q_attitude: float = 28.0,
        q_rate: float = 2.5,
        r_command: float = 8.0,
    ):
        self.dt = float(dt)
        self.actuator_gain = float(actuator_gain)
        self.damping = float(damping)
        self.max_command = math.radians(float(max_command_deg))
        Ac = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, -self.damping, 0],
            [0, 0, 0, -self.damping],
        ], dtype=float)
        Bc = np.array([
            [0, 0],
            [0, 0],
            [self.actuator_gain, 0],
            [0, self.actuator_gain],
        ], dtype=float)
        # ZOH approximation adequate for a 25 Hz low-bandwidth body loop.
        self.A = np.eye(4) + Ac * self.dt
        self.B = Bc * self.dt
        Q = np.diag([q_attitude, q_attitude, q_rate, q_rate]).astype(float)
        R = np.diag([r_command, r_command]).astype(float)
        P = solve_discrete_are_iterative(self.A, self.B, Q, R)
        self.K = np.linalg.solve(R + self.B.T @ P @ self.B, self.B.T @ P @ self.A)

    def control(self, roll: float, pitch: float, p: float, q: float, ref_roll: float, ref_pitch: float) -> LQROutput:
        x = np.array([roll - ref_roll, pitch - ref_pitch, p, q], dtype=float)
        u = -self.K @ x
        clipped = np.clip(u, -self.max_command, self.max_command)
        return LQROutput(
            command_roll_rad=float(clipped[0]),
            command_pitch_rad=float(clipped[1]),
            state_error=x,
            saturated=bool(np.any(np.abs(u - clipped) > 1e-12)),
        )

    def metadata(self) -> dict:
        return {
            "K": self.K.round(6).tolist(),
            "dt_s": self.dt,
            "actuator_gain": self.actuator_gain,
            "damping": self.damping,
            "max_command_deg": math.degrees(self.max_command),
        }
