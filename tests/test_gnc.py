#!/usr/bin/env python3
"""Offline deterministic tests for Astro v0.8 GNC math.

Does not touch the real robot or physical IMU. Run with:
  PYTHONPATH=pi python scripts/test_gnc.py
"""
from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace
import numpy as np

from astro_robot.gnc.ekf import AttitudeMEKF
from astro_robot.gnc.guidance import AttitudeGuidance
from astro_robot.gnc.lqr import BodyAttitudeLQR
from astro_robot.gnc.math3d import (
    level_reference_rotation,
    quat_from_euler,
    quat_to_rotmat,
    roll_pitch_from_accel,
)
from astro_robot.gnc.supervisor import StabilitySupervisor
from astro_robot.gnc.stabilizer import GNCStabilizer


def test_ekf_static_attitude() -> None:
    true_roll = math.radians(8.0)
    true_pitch = math.radians(-5.0)
    q = quat_from_euler(true_roll, true_pitch, 0.0)
    accel = quat_to_rotmat(q).T @ np.array([0.0, 0.0, 1.0])
    bias = np.array([0.010, -0.008, 0.004])
    ekf = AttitudeMEKF()
    ekf.set_gyro_bias(bias)
    for _ in range(300):
        est = ekf.step(bias, accel, 0.04)
    er = abs(math.degrees(est.roll - true_roll))
    ep = abs(math.degrees(est.pitch - true_pitch))
    assert er < 0.15 and ep < 0.15, (er, ep)
    assert est.accel_used


def test_accel_gate() -> None:
    ekf = AttitudeMEKF()
    ekf.step(np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.04)
    est = ekf.step(np.zeros(3), np.array([0.0, 0.0, 1.8]), 0.04)
    assert not est.accel_used


def test_imu_mount_level_compensation() -> None:
    roll, pitch = math.radians(-0.57), math.radians(-9.70)
    gravity_in_sensor_frame = np.array([
        -math.sin(pitch),
        math.sin(roll) * math.cos(pitch),
        math.cos(roll) * math.cos(pitch),
    ])
    corrected = level_reference_rotation(roll, pitch) @ gravity_in_sensor_frame
    corrected_roll, corrected_pitch = roll_pitch_from_accel(corrected)
    assert abs(math.degrees(corrected_roll)) < 1e-8
    assert abs(math.degrees(corrected_pitch)) < 1e-8


def test_guidance_slew() -> None:
    g = AttitudeGuidance(max_tilt_deg=8, slew_deg_s=4)
    g.set_target_deg(8, -8)
    r = g.step(0.5)
    assert abs(math.degrees(r.roll) - 2.0) < 1e-6
    assert abs(math.degrees(r.pitch) + 2.0) < 1e-6


def test_guidance_captures_slope_reference_without_transient() -> None:
    g = AttitudeGuidance(max_tilt_deg=8, slew_deg_s=4)
    captured = g.hold_current_deg(5.0, -3.0)
    assert captured == (5.0, -3.0)
    r = g.step(0.04)
    assert abs(math.degrees(r.roll) - 5.0) < 1e-8
    assert abs(math.degrees(r.pitch) + 3.0) < 1e-8
    for values in ((9, 0), (0, -9), (float("nan"), 0)):
        try:
            g.hold_current_deg(*values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe attitude reference accepted: {values}")


def test_capture_current_reference_guards() -> None:
    def make_gnc():
        gnc = GNCStabilizer.__new__(GNCStabilizer)
        gnc._lock = threading.RLock()
        gnc.status = SimpleNamespace(
            mode="monitor", controller_active=False, calibrated=True,
            roll_deg=4.0, pitch_deg=-2.0, p_dps=0.1, q_dps=-0.1,
        )
        gnc._last_sensor = time.monotonic()
        gnc.settings = SimpleNamespace(astro_gnc_sensor_timeout_s=0.25)
        gnc.motion_safety = SimpleNamespace(is_moving=False)
        gnc.guidance = AttitudeGuidance(max_tilt_deg=8, slew_deg_s=4)
        gnc.snapshot = lambda: {"mode": gnc.status.mode}
        return gnc

    gnc = make_gnc()
    result = gnc.capture_current_reference()
    assert result["ok"] and result["reference_roll_deg"] == 4.0
    assert result["reference_pitch_deg"] == -2.0
    for mutate in (
        lambda g: setattr(g.status, "calibrated", False),
        lambda g: setattr(g.motion_safety, "is_moving", True),
        lambda g: setattr(g.status, "mode", "stabilize"),
        lambda g: setattr(g.status, "p_dps", 6.0),
        lambda g: setattr(g, "_last_sensor", 0.0),
    ):
        candidate = make_gnc()
        mutate(candidate)
        try:
            candidate.capture_current_reference()
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsafe current-attitude capture was accepted")


def test_supervisor_guards() -> None:
    import time
    sup = StabilitySupervisor(max_tilt_deg=20, max_rate_dps=120, sensor_timeout_s=0.25)
    base = dict(mode="stabilize", configured_actuation=True, calibrated=True, moving=False,
                roll=0.0, pitch=0.0, p=0.0, q=0.0, last_sensor_monotonic=time.monotonic())
    assert sup.evaluate(**base).allow_actuation
    b = dict(base); b["moving"] = True
    assert not sup.evaluate(**b).allow_actuation
    b = dict(base); b["calibrated"] = False
    assert not sup.evaluate(**b).allow_actuation
    b = dict(base); b["roll"] = math.radians(25)
    d = sup.evaluate(**b)
    assert not d.allow_actuation and d.fault


def test_lqr_closed_loop() -> None:
    c = BodyAttitudeLQR(dt=0.04)
    x = np.array([math.radians(10), math.radians(-7), 0.0, 0.0], dtype=float)
    initial = float(np.linalg.norm(x[:2]))
    saturated_seen = False
    for _ in range(250):
        out = c.control(x[0], x[1], x[2], x[3], 0.0, 0.0)
        u = np.array([out.command_roll_rad, out.command_pitch_rad])
        saturated_seen |= out.saturated
        x = c.A @ x + c.B @ u
    final = float(np.linalg.norm(x[:2]))
    assert final < math.radians(0.05), math.degrees(final)
    assert final < initial / 100
    assert saturated_seen


def main() -> None:
    tests = [test_ekf_static_attitude, test_accel_gate, test_imu_mount_level_compensation,
             test_guidance_slew, test_guidance_captures_slope_reference_without_transient,
             test_capture_current_reference_guards, test_supervisor_guards, test_lqr_closed_loop]
    for t in tests:
        t(); print(f"PASS {t.__name__}")
    print("Astro GNC offline tests PASSED")


if __name__ == "__main__":
    main()
