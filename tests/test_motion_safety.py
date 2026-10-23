#!/usr/bin/env python3
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi"))
from astro_robot.robot.safety import MotionBlocked, MotionSafety


class MotionSafetySonarTests(unittest.TestCase):
    def make_safety(self, distance=None, updated=None, *, required=False):
        robot = Mock()
        robot.state = SimpleNamespace(
            sonic_cm=distance,
            sonic_updated_monotonic=time.monotonic() if updated is None and distance is not None else (updated or 0.0),
        )
        return MotionSafety(robot, sonar_required=required), robot

    def test_missing_sonar_does_not_block_when_optional(self):
        safety, robot = self.make_safety()
        safety.command(0, 1, 2, 0, 1)
        robot.move.assert_called_once_with(x=0, y=1, speed=2, yaw=0, gait=1)
        safety.stop()

    def test_stale_sonar_does_not_block_when_optional(self):
        safety, robot = self.make_safety(100, time.monotonic() - 10)
        safety.command(0, 1, 2, 0, 1)
        robot.move.assert_called_once()
        safety.stop()

    def test_fresh_obstacle_still_blocks_forward_motion(self):
        safety, robot = self.make_safety(10)
        with self.assertRaisesRegex(MotionBlocked, "Obstacle"):
            safety.command(0, 1, 2, 0, 1)
        robot.move.assert_not_called()
        robot.stop.assert_called_once()

    def test_required_sonar_keeps_fail_closed_behavior(self):
        safety, robot = self.make_safety(required=True)
        with self.assertRaisesRegex(MotionBlocked, "stale"):
            safety.command(0, 1, 2, 0, 1)
        robot.move.assert_not_called()
        robot.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
