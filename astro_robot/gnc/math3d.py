from __future__ import annotations

import math
import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q = q / n
    # q and -q represent the same attitude. Keeping w >= 0 limits sign jumps.
    return -q if q[0] < 0 else q


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=float).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=float).reshape(4)
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dtype=float)


def quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    rv = np.asarray(rv, dtype=float).reshape(3)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-10:
        return quat_normalize(np.array([1.0, *(0.5 * rv)], dtype=float))
    axis = rv / angle
    h = 0.5 * angle
    return np.array([math.cos(h), *(axis * math.sin(h))], dtype=float)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def quat_from_euler(roll: float, pitch: float, yaw: float = 0.0) -> np.ndarray:
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return quat_normalize(np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ], dtype=float))


def euler_from_quat(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = quat_normalize(q)
    sinr = 2 * (w*x + y*z)
    cosr = 1 - 2 * (x*x + y*y)
    roll = math.atan2(sinr, cosr)
    sinp = 2 * (w*y - z*x)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny = 2 * (w*z + x*y)
    cosy = 1 - 2 * (y*y + z*z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def roll_pitch_from_accel(accel_g: np.ndarray) -> tuple[float, float]:
    ax, ay, az = np.asarray(accel_g, dtype=float).reshape(3)
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))
    return roll, pitch


def level_reference_rotation(roll: float, pitch: float) -> np.ndarray:
    """Rotate mapped IMU vectors so the measured level pose maps to +Z gravity."""
    gravity = np.array([
        -math.sin(pitch),
        math.sin(roll) * math.cos(pitch),
        math.cos(roll) * math.cos(pitch),
    ], dtype=float)
    gravity /= max(float(np.linalg.norm(gravity)), 1e-12)
    target = np.array([0.0, 0.0, 1.0], dtype=float)
    cross = np.cross(gravity, target)
    cosine = float(np.clip(np.dot(gravity, target), -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.eye(3, dtype=float)
    if cosine < -1.0 + 1e-12:
        # A 180-degree mounting error is outside the supported trim range.
        return np.diag([1.0, -1.0, -1.0])
    cross_matrix = skew(cross)
    return np.eye(3) + cross_matrix + (cross_matrix @ cross_matrix) / (1.0 + cosine)
