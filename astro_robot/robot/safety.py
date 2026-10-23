from __future__ import annotations

import threading
import time
import math


class MotionBlocked(RuntimeError):
    pass


class MotionSafety:
    """Independent dead-man safety layer for movement commands."""

    def __init__(self, robot, deadman_seconds: float = 0.8, *,
                 obstacle_stop_cm: float = 24.0, range_timeout_s: float = 1.5,
                 sonar_required: bool = False):
        self.robot = robot
        self.deadman_seconds = deadman_seconds
        self._deadline = 0.0
        self._moving = False
        self.obstacle_stop_cm = float(obstacle_stop_cm)
        self.range_timeout_s = float(range_timeout_s)
        self.sonar_required = bool(sonar_required)
        self._last_command = {"x": 0.0, "y": 0.0, "speed": 0, "yaw": 0.0, "gait": 1}
        self._blocked_reason: str | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def command(self, x: float, y: float, speed: int, yaw: float, gait: int) -> None:
        with self._lock:
            if y > 0.01:
                age = time.monotonic() - self.robot.state.sonic_updated_monotonic
                distance = self.robot.state.sonic_cm
                if distance is None or not math.isfinite(distance) or age > self.range_timeout_s:
                    if self.sonar_required:
                        self._stop_locked("Ultrasonic range is stale; forward movement blocked")
                        raise MotionBlocked(self._blocked_reason or "Ultrasonic range is stale")
                elif distance <= self.obstacle_stop_cm:
                    self._stop_locked(f"Obstacle at {distance:.0f} cm; forward movement blocked")
                    raise MotionBlocked(self._blocked_reason or "Obstacle too close")
            self.robot.move(x=x, y=y, speed=speed, yaw=yaw, gait=gait)
            self._moving = (abs(x) > 0.01 or abs(y) > 0.01 or abs(yaw) > 0.01)
            self._deadline = time.monotonic() + self.deadman_seconds
            self._last_command = {"x": x, "y": y, "speed": speed, "yaw": yaw, "gait": gait}
            self._blocked_reason = None

    @property
    def is_moving(self) -> bool:
        with self._lock:
            return bool(self._moving)

    def stop(self) -> None:
        with self._lock:
            self._stop_locked(None)

    def _stop_locked(self, reason: str | None) -> None:
        try:
            self.robot.stop()
        except Exception:
            reason = reason or "Astro direct stop command failed"
        finally:
            self._moving = False
            self._deadline = 0.0
            self._blocked_reason = reason
            self._last_command = {"x": 0.0, "y": 0.0, "speed": 0, "yaw": 0.0, "gait": 1}

    def snapshot(self) -> dict:
        with self._lock:
            age = time.monotonic() - self.robot.state.sonic_updated_monotonic if self.robot.state.sonic_updated_monotonic else None
            return {"moving": self._moving, "command": dict(self._last_command),
                    "blocked_reason": self._blocked_reason,
                    "obstacle_stop_cm": self.obstacle_stop_cm,
                    "sonar_required": self.sonar_required,
                    "range_age_s": age}

    def _loop(self) -> None:
        while True:
            time.sleep(0.1)
            with self._lock:
                try:
                    if self._moving and time.monotonic() > self._deadline:
                        self._stop_locked("Dead-man timeout")
                    elif self._moving and self._last_command["y"] > 0.01:
                        age = time.monotonic() - self.robot.state.sonic_updated_monotonic
                        distance = self.robot.state.sonic_cm
                        if (distance is None or not math.isfinite(distance) or age > self.range_timeout_s) and self.sonar_required:
                            self._stop_locked("Ultrasonic range stale during forward movement")
                        elif distance is not None and math.isfinite(distance) and age <= self.range_timeout_s and distance <= self.obstacle_stop_cm:
                            self._stop_locked(f"Obstacle at {distance:.0f} cm")
                except Exception:
                    self._moving = False
                    self._deadline = 0.0
                    self._blocked_reason = "Astro direct stop command failed"
