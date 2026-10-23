from __future__ import annotations

from dataclasses import dataclass
import math
import time


@dataclass
class SupervisorDecision:
    allow_actuation: bool
    reason: str
    fault: bool = False


class StabilitySupervisor:
    """Independent guards around the linearized LQR loop."""

    def __init__(self, max_tilt_deg: float = 20.0, max_rate_dps: float = 120.0, sensor_timeout_s: float = 0.25):
        self.max_tilt = math.radians(float(max_tilt_deg))
        self.max_rate = math.radians(float(max_rate_dps))
        self.sensor_timeout_s = float(sensor_timeout_s)

    def evaluate(self, *, mode: str, configured_actuation: bool, calibrated: bool, moving: bool,
                 allow_moving: bool = False, roll: float, pitch: float, p: float, q: float,
                 last_sensor_monotonic: float) -> SupervisorDecision:
        if mode != "stabilize":
            return SupervisorDecision(False, f"mode={mode}")
        if not configured_actuation:
            return SupervisorDecision(False, "actuation locked by ASTRO_GNC_ALLOW_ACTUATION=false")
        if not calibrated:
            return SupervisorDecision(False, "gyro calibration required")
        if moving and not allow_moving:
            return SupervisorDecision(False, "suspended while walking/turning")
        if time.monotonic() - last_sensor_monotonic > self.sensor_timeout_s:
            return SupervisorDecision(False, "IMU timeout", True)
        if not all(math.isfinite(v) for v in (roll, pitch, p, q)):
            return SupervisorDecision(False, "non-finite estimator state", True)
        if abs(roll) > self.max_tilt or abs(pitch) > self.max_tilt:
            return SupervisorDecision(False, "outside LQR linear operating envelope", True)
        if abs(p) > self.max_rate or abs(q) > self.max_rate:
            return SupervisorDecision(False, "angular rate exceeds safety limit", True)
        return SupervisorDecision(True, "nominal")
