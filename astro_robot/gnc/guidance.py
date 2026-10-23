from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class AttitudeReference:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


class AttitudeGuidance:
    """Rate-limited body attitude guidance for self-leveling.

    References are stored in radians. The limiter prevents a UI/agent request
    from asking the body IK to jump instantaneously to a new pose.
    """

    def __init__(self, max_tilt_deg: float = 8.0, slew_deg_s: float = 4.0):
        self.max_tilt = math.radians(float(max_tilt_deg))
        self.slew = math.radians(float(slew_deg_s))
        self.target = AttitudeReference()
        self.current = AttitudeReference()

    def set_target_deg(self, roll_deg: float = 0.0, pitch_deg: float = 0.0, yaw_deg: float = 0.0) -> None:
        self.target = AttitudeReference(
            roll=max(-self.max_tilt, min(self.max_tilt, math.radians(float(roll_deg)))),
            pitch=max(-self.max_tilt, min(self.max_tilt, math.radians(float(pitch_deg)))),
            yaw=math.radians(float(yaw_deg)),
        )

    def hold_current_deg(self, roll_deg: float, pitch_deg: float) -> tuple[float, float]:
        """Capture a small, already-established body attitude without a slew transient.

        This is a reference hold, not terrain estimation: the IMU alone cannot
        identify the ground plane or individual foot contacts.
        """
        roll_deg, pitch_deg = float(roll_deg), float(pitch_deg)
        if not math.isfinite(roll_deg) or not math.isfinite(pitch_deg):
            raise ValueError("Current attitude must be finite")
        limit_deg = math.degrees(self.max_tilt)
        if abs(roll_deg) > limit_deg or abs(pitch_deg) > limit_deg:
            raise ValueError(f"Current attitude exceeds the reference limit of ±{limit_deg:g}°")
        reference = AttitudeReference(math.radians(roll_deg), math.radians(pitch_deg), 0.0)
        self.target = AttitudeReference(reference.roll, reference.pitch, 0.0)
        self.current = AttitudeReference(reference.roll, reference.pitch, 0.0)
        return roll_deg, pitch_deg

    @staticmethod
    def _slew(current: float, target: float, step: float) -> float:
        d = target - current
        if abs(d) <= step:
            return target
        return current + (step if d > 0 else -step)

    def step(self, dt: float) -> AttitudeReference:
        step = self.slew * max(0.0, float(dt))
        self.current.roll = self._slew(self.current.roll, self.target.roll, step)
        self.current.pitch = self._slew(self.current.pitch, self.target.pitch, step)
        self.current.yaw = self._slew(self.current.yaw, self.target.yaw, step)
        return AttitudeReference(self.current.roll, self.current.pitch, self.current.yaw)

    def reset_level(self) -> None:
        self.target = AttitudeReference()
        self.current = AttitudeReference()
