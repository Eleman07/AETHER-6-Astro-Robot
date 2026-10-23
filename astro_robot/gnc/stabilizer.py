from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from collections import deque
import threading
import time
from typing import Callable

import numpy as np

from .ekf import AttitudeMEKF, EKFEstimate
from .guidance import AttitudeGuidance
from .lqr import BodyAttitudeLQR
from .math3d import level_reference_rotation
from .supervisor import StabilitySupervisor


@dataclass
class GNCStatus:
    mode: str = "disabled"
    calibrated: bool = False
    actuation_configured: bool = False
    controller_active: bool = False
    supervisor_reason: str = "not started"
    fault: str | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    p_dps: float | None = None
    q_dps: float | None = None
    r_dps: float | None = None
    ref_roll_deg: float = 0.0
    ref_pitch_deg: float = 0.0
    cmd_roll_deg: float = 0.0
    cmd_pitch_deg: float = 0.0
    accel_norm_g: float | None = None
    accel_used: bool = False
    covariance_trace: float | None = None
    lqr_saturated: bool = False
    loop_hz: float = 0.0
    samples: int = 0


class GNCStabilizer:
    """Background Navigation -> Guidance -> LQR stabilization loop.

    Actuation is deliberately locked unless ASTRO_GNC_ALLOW_ACTUATION=true.
    Even then it requires an explicit gyro calibration and mode=stabilize.
    """

    VALID_MODES = {"disabled", "monitor", "stabilize"}

    def __init__(self, settings, sense, robot, motion_safety):
        self.settings = settings
        self.sense = sense
        self.robot = robot
        self.motion_safety = motion_safety
        self.rate_hz = max(5.0, min(50.0, float(settings.astro_gnc_rate_hz)))
        self.dt_nominal = 1.0 / self.rate_hz
        self.ekf = AttitudeMEKF(
            gyro_noise=settings.astro_gnc_gyro_noise,
            bias_rw=settings.astro_gnc_bias_rw,
            accel_noise=settings.astro_gnc_accel_noise,
            accel_gate_min=settings.astro_gnc_accel_gate_min,
            accel_gate_max=settings.astro_gnc_accel_gate_max,
        )
        self.guidance = AttitudeGuidance(settings.astro_gnc_guidance_max_tilt_deg, settings.astro_gnc_guidance_slew_deg_s)
        self.lqr = BodyAttitudeLQR(
            dt=self.dt_nominal,
            actuator_gain=settings.astro_gnc_lqr_actuator_gain,
            damping=settings.astro_gnc_lqr_damping,
            max_command_deg=settings.astro_gnc_max_command_deg,
            q_attitude=settings.astro_gnc_lqr_q_attitude,
            q_rate=settings.astro_gnc_lqr_q_rate,
            r_command=settings.astro_gnc_lqr_r_command,
        )
        self.supervisor = StabilitySupervisor(settings.astro_gnc_max_tilt_deg, settings.astro_gnc_max_rate_dps, settings.astro_gnc_sensor_timeout_s)
        self.imu_level_rotation = level_reference_rotation(
            math.radians(settings.astro_gnc_imu_level_roll_deg),
            math.radians(settings.astro_gnc_imu_level_pitch_deg),
        )
        default_mode = str(settings.astro_gnc_default_mode).lower().strip()
        # Closed-loop mode always requires a fresh stationary calibration after boot.
        if default_mode not in {"disabled", "monitor"}:
            default_mode = "monitor"
        self.status = GNCStatus(mode=default_mode, actuation_configured=settings.astro_gnc_allow_actuation)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sensor = 0.0
        self._last_cmd = (0.0, 0.0)
        self._fault_fallback_used = False
        # In-memory black-box recorder: 5,000 samples (~200 s at 25 Hz), no SD writes in the control loop.
        self._blackbox = deque(maxlen=5000)
        self._next_event: str | None = None

    def _map_xyz(self, d: dict[str, float]) -> np.ndarray:
        names = [x.strip().lower() for x in self.settings.astro_gnc_imu_axes.split(",")]
        signs = [float(x.strip()) for x in self.settings.astro_gnc_imu_signs.split(",")]
        if len(names) != 3 or len(signs) != 3 or any(n not in {"x", "y", "z"} for n in names):
            raise ValueError("ASTRO_GNC_IMU_AXES/SIGNS must contain exactly 3 axes/signs")
        return np.array([float(d[n]) * signs[i] for i, n in enumerate(names)], dtype=float)

    def _map_imu_vector(self, d: dict[str, float]) -> np.ndarray:
        """Map sensor axes to body axes and compensate the fixed mounting tilt."""
        return self.imu_level_rotation @ self._map_xyz(d)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="astro-gnc", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._zero_command()

    def set_mode(self, mode: str) -> dict:
        mode = str(mode).lower().strip()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}")
        with self._lock:
            if mode == "stabilize" and not self.settings.astro_gnc_allow_actuation:
                raise RuntimeError("GNC actuation is locked. Set ASTRO_GNC_ALLOW_ACTUATION=true only after bench calibration/testing.")
            if mode == "stabilize" and not self.status.calibrated:
                raise RuntimeError("Calibrate the stationary IMU before enabling stabilization.")
            self.status.mode = mode
            self.status.fault = None
            self._fault_fallback_used = False
        if mode == "stabilize":
            # Never run vendor balancing and Astro LQR simultaneously.
            self.robot.balance(False)
        else:
            self._zero_command()
        return self.snapshot()

    def set_reference(self, roll_deg: float, pitch_deg: float) -> dict:
        self.guidance.set_target_deg(roll_deg, pitch_deg, 0.0)
        return self.snapshot()

    def capture_current_reference(self) -> dict:
        """Hold the present small body attitude; never infer terrain/contact."""
        with self._lock:
            if self.status.mode != "monitor" or self.status.controller_active:
                raise RuntimeError("Switch to Monitor before capturing the current attitude")
            if not self.status.calibrated:
                raise RuntimeError("Calibrate the stationary IMU before capturing a reference")
            if self._last_sensor <= 0 or time.monotonic() - self._last_sensor > self.settings.astro_gnc_sensor_timeout_s:
                raise RuntimeError("Current IMU attitude is stale")
            roll_deg, pitch_deg = self.status.roll_deg, self.status.pitch_deg
            p_dps, q_dps = self.status.p_dps, self.status.q_dps
        if self.motion_safety.is_moving:
            raise RuntimeError("Stop the robot before capturing a reference")
        if any(value is None or not math.isfinite(value) for value in (roll_deg, pitch_deg, p_dps, q_dps)):
            raise RuntimeError("Current IMU attitude is unavailable")
        if abs(p_dps) > 5.0 or abs(q_dps) > 5.0:
            raise RuntimeError("Keep Astro still; measured roll/pitch rate must be below 5°/s")
        reference = self.guidance.hold_current_deg(roll_deg, pitch_deg)
        return {"ok": True, "reference_roll_deg": reference[0], "reference_pitch_deg": reference[1], "status": self.snapshot()}

    def reset(self) -> dict:
        with self._lock:
            old_bias = self.ekf.bias.copy()
            calibrated = self.status.calibrated
            self.ekf = AttitudeMEKF(
                self.settings.astro_gnc_gyro_noise,
                self.settings.astro_gnc_bias_rw,
                self.settings.astro_gnc_accel_noise,
                self.settings.astro_gnc_accel_gate_min,
                self.settings.astro_gnc_accel_gate_max,
            )
            if calibrated:
                self.ekf.set_gyro_bias(old_bias)
            self.guidance.reset_level()
            self.status.fault = None
            self.status.controller_active = False
            self.status.supervisor_reason = "reset"
        self._zero_command()
        return self.snapshot()

    def calibrate_gyro(self, seconds: float = 2.5) -> dict:
        seconds = max(1.0, min(8.0, float(seconds)))
        if self.status.mode == "stabilize":
            raise RuntimeError("Disable stabilization or switch to Monitor before IMU calibration")
        if self.motion_safety.is_moving:
            raise RuntimeError("Stop the robot before IMU calibration")
        samples = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            raw = self.sense.imu_sample()
            if not raw.get("available"):
                raise RuntimeError(raw.get("error") or "MPU6050 IMU unavailable")
            samples.append(self._map_imu_vector(raw["gyro_rps"]) * self.settings.astro_gnc_gyro_scale)
            time.sleep(0.02)
        arr = np.asarray(samples)
        if len(arr) < 20:
            raise RuntimeError("Too few IMU samples")
        std = np.std(arr, axis=0)
        if float(np.max(std)) > math.radians(self.settings.astro_gnc_calibration_max_std_dps):
            raise RuntimeError(f"Robot/IMU moved during calibration (gyro std={np.degrees(std).round(3).tolist()} deg/s)")
        bias = np.mean(arr, axis=0)
        self.ekf.set_gyro_bias(bias)
        with self._lock:
            self.status.calibrated = True
            self.status.supervisor_reason = "gyro calibrated"
            self.status.fault = None
        return {"ok": True, "samples": len(arr), "gyro_bias_rps": bias.tolist(), "gyro_std_dps": np.degrees(std).tolist(), "status": self.snapshot()}

    def _zero_command(self) -> None:
        if self._last_cmd != (0.0, 0.0):
            try:
                self.robot.attitude(0, 0, 0)
            except Exception:
                pass
        self._last_cmd = (0.0, 0.0)
        with self._lock:
            self.status.cmd_roll_deg = 0.0
            self.status.cmd_pitch_deg = 0.0
            self.status.controller_active = False

    def _fault(self, reason: str) -> None:
        with self._lock:
            self.status.fault = reason
            self.status.mode = "monitor"  # latch out of active control until explicitly re-enabled
            self.status.controller_active = False
            self.status.supervisor_reason = reason
        # A failed estimator/controller must also end any active gait command.
        try:
            self.motion_safety.stop()
        except Exception:
            pass
        self._zero_command()
        # Do not silently hand control to the vendor balancer after a GNC fault.
        # A fault stops gait and requires an operator to inspect/re-arm the GNC.

    def _update_status(self, est: EKFEstimate, ref, out, active: bool, reason: str, loop_hz: float) -> None:
        with self._lock:
            self.status.roll_deg = math.degrees(est.roll)
            self.status.pitch_deg = math.degrees(est.pitch)
            self.status.yaw_deg = math.degrees(est.yaw)
            self.status.p_dps = math.degrees(est.omega_rps[0])
            self.status.q_dps = math.degrees(est.omega_rps[1])
            self.status.r_dps = math.degrees(est.omega_rps[2])
            self.status.ref_roll_deg = math.degrees(ref.roll)
            self.status.ref_pitch_deg = math.degrees(ref.pitch)
            self.status.accel_norm_g = est.accel_norm_g
            self.status.accel_used = est.accel_used
            self.status.covariance_trace = est.covariance_trace
            self.status.lqr_saturated = bool(out.saturated)
            self.status.controller_active = bool(active)
            self.status.supervisor_reason = reason
            self.status.loop_hz = loop_hz
            self.status.samples += 1
            self._blackbox.append({
                "t_monotonic": time.monotonic(),
                "mode": self.status.mode,
                "roll_deg": self.status.roll_deg,
                "pitch_deg": self.status.pitch_deg,
                "yaw_deg": self.status.yaw_deg,
                "p_dps": self.status.p_dps,
                "q_dps": self.status.q_dps,
                "r_dps": self.status.r_dps,
                "ref_roll_deg": self.status.ref_roll_deg,
                "ref_pitch_deg": self.status.ref_pitch_deg,
                "cmd_roll_deg": self.status.cmd_roll_deg,
                "cmd_pitch_deg": self.status.cmd_pitch_deg,
                "accel_norm_g": self.status.accel_norm_g,
                "accel_used": self.status.accel_used,
                "lqr_saturated": self.status.lqr_saturated,
                "controller_active": self.status.controller_active,
                "reason": self.status.supervisor_reason,
                "sonic_cm": self.robot.state.sonic_cm,
                "battery_1_v": self.robot.state.battery_1_v,
                "battery_2_v": self.robot.state.battery_2_v,
                "motion_x": self.motion_safety.snapshot()["command"]["x"],
                "motion_y": self.motion_safety.snapshot()["command"]["y"],
                "motion_yaw": self.motion_safety.snapshot()["command"]["yaw"],
                "event": self._next_event or "",
            })
            self._next_event = None

    def mark_event(self, label: str) -> dict:
        label = str(label).strip().replace("\n", " ")[:80]
        if not label:
            raise ValueError("Event label is required")
        with self._lock:
            self._next_event = label
        return {"ok": True, "event": label}

    def _loop(self) -> None:
        last = time.monotonic()
        next_t = last
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(next_t - now, 0.01)); continue
            dt = max(0.005, min(0.12, now - last))
            last = now
            next_t = now + self.dt_nominal
            try:
                raw = self.sense.imu_sample()
                if not raw.get("available"):
                    reason = raw.get("error") or "MPU6050 unavailable"
                    if self.status.mode == "stabilize":
                        self._fault(reason)
                    else:
                        with self._lock:
                            self.status.supervisor_reason = reason
                    continue
                gyro = self._map_imu_vector(raw["gyro_rps"]) * self.settings.astro_gnc_gyro_scale
                accel = self._map_imu_vector(raw["accel_g"])
                self._last_sensor = time.monotonic()
                est = self.ekf.step(gyro, accel, dt)
                ref = self.guidance.step(dt)
                out = self.lqr.control(est.roll, est.pitch, est.omega_rps[0], est.omega_rps[1], ref.roll, ref.pitch)
                with self._lock:
                    mode = self.status.mode
                decision = self.supervisor.evaluate(
                    mode=mode,
                    configured_actuation=self.settings.astro_gnc_allow_actuation,
                    calibrated=self.status.calibrated,
                    moving=self.motion_safety.is_moving,
                    allow_moving=self.settings.astro_gnc_walk_control_enabled,
                    roll=est.roll, pitch=est.pitch,
                    p=est.omega_rps[0], q=est.omega_rps[1],
                    last_sensor_monotonic=self._last_sensor,
                )
                active = decision.allow_actuation
                if decision.fault:
                    self._fault(decision.reason)
                    active = False
                elif active:
                    roll_deg = math.degrees(out.command_roll_rad) * self.settings.astro_gnc_actuator_sign_roll
                    pitch_deg = math.degrees(out.command_pitch_rad) * self.settings.astro_gnc_actuator_sign_pitch
                    self.robot.attitude(roll_deg, pitch_deg, 0.0)
                    self._last_cmd = (roll_deg, pitch_deg)
                    with self._lock:
                        self.status.cmd_roll_deg = roll_deg
                        self.status.cmd_pitch_deg = pitch_deg
                else:
                    self._zero_command()
                self._update_status(est, ref, out, active, decision.reason, 1.0 / dt if dt > 0 else 0.0)
            except Exception as exc:
                self._fault(f"GNC loop error: {exc}")
                time.sleep(0.1)

    def blackbox(self, limit: int = 1000) -> list[dict]:
        with self._lock:
            n = max(1, min(int(limit), 5000))
            return list(self._blackbox)[-n:]

    def snapshot(self) -> dict:
        with self._lock:
            d = asdict(self.status)
        d["lqr"] = self.lqr.metadata()
        d["imu_axes"] = self.settings.astro_gnc_imu_axes
        d["imu_signs"] = self.settings.astro_gnc_imu_signs
        d["imu_level_roll_deg"] = self.settings.astro_gnc_imu_level_roll_deg
        d["imu_level_pitch_deg"] = self.settings.astro_gnc_imu_level_pitch_deg
        d["actuation_locked"] = not self.settings.astro_gnc_allow_actuation
        return d
