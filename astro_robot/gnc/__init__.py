"""Astro self-stabilization GNC stack.

Navigation: multiplicative/error-state EKF (quaternion + gyro bias)
Guidance: rate-limited body-attitude reference
Control: discrete LQR on roll/pitch attitude + rates
Actuation: bounded Astro body-attitude commands through the direct PCA9685 IK layer
"""

from .stabilizer import GNCStabilizer

__all__ = ["GNCStabilizer"]
