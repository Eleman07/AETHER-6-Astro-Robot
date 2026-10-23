from __future__ import annotations

import math
import threading
import time
from typing import Any


class MPU6050Adapter:
    """Direct MPU6050/MPU6500-compatible reader over I2C.

    The sensor must be exposed on the Pi I2C bus (normally 0x68; AD0 high
    commonly selects 0x69). WHO_AM_I is checked before telemetry is trusted;
    MPU-6050 reports 0x68 and MPU-6500 reports 0x70.
    A failed boot-time probe is retried, so service start order is not critical.
    """
    ACCEL_LSB_G = 16384.0  # +/-2 g
    GYRO_LSB_DPS = 131.0   # +/-250 deg/s
    RETRY_INTERVAL_S = 2.0

    def __init__(self, enabled: bool = True, bus_num: int = 1, address: int = 0x68):
        self.driver = "mpu6050_i2c"
        self.enabled = enabled
        self.bus_num = int(bus_num)
        self.address = int(address)
        self.available = False
        self.error: str | None = None
        self._bus = None
        self._lock = threading.RLock()
        self._last_init_attempt = 0.0
        self._last_sample: dict[str, Any] | None = None
        if not enabled:
            self.error = "IMU disabled by configuration"
            return
        self._initialize()

    def _close_bus(self) -> None:
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None

    def _initialize(self) -> bool:
        self._last_init_attempt = time.monotonic()
        bus = None
        try:
            from smbus2 import SMBus
            bus = SMBus(self.bus_num)
            who = bus.read_byte_data(self.address, 0x75)
            if who not in (0x68, 0x69, 0x70):
                raise RuntimeError(f"Unsupported MPU6050/MPU6500 WHO_AM_I=0x{who:02x}")
            # Wake up; DLPF ~44/42 Hz; +/-2g and +/-250 deg/s.
            bus.write_byte_data(self.address, 0x6B, 0x00)
            bus.write_byte_data(self.address, 0x1A, 0x03)
            bus.write_byte_data(self.address, 0x1C, 0x00)
            bus.write_byte_data(self.address, 0x1B, 0x00)
            time.sleep(0.05)
            self._bus = bus
            self.available = True
            self.error = None
            return True
        except Exception as exc:
            self.error = str(exc)
            if bus is not None:
                try:
                    bus.close()
                except Exception:
                    pass
            self._close_bus()
            self.available = False
            return False

    @staticmethod
    def _s16(hi: int, lo: int) -> int:
        value = (hi << 8) | lo
        return value - 65536 if value & 0x8000 else value

    def _read_raw(self) -> tuple[list[int], list[int], int]:
        # One contiguous burst: accel XYZ, temperature, gyro XYZ.
        b = self._bus.read_i2c_block_data(self.address, 0x3B, 14)
        a = [self._s16(b[i], b[i+1]) for i in (0,2,4)]
        temp = self._s16(b[6], b[7])
        g = [self._s16(b[i], b[i+1]) for i in (8,10,12)]
        return a, g, temp

    def imu_sample(self) -> dict[str, Any]:
        if not self.enabled:
            return {"available": False, "error": self.error or "IMU disabled by configuration"}
        with self._lock:
            if (not self.available or self._bus is None) and time.monotonic() - self._last_init_attempt >= self.RETRY_INTERVAL_S:
                self._initialize()
        if not self.available or self._bus is None:
            return {"available": False, "error": self.error or "MPU6050 unavailable"}
        try:
            with self._lock:
                a, g, _ = self._read_raw()
                sample = {
                    "available": True,
                    "timestamp_monotonic": time.monotonic(),
                    "accel_g": dict(zip(("x", "y", "z"), (v/self.ACCEL_LSB_G for v in a))),
                    "gyro_rps": dict(zip(("x", "y", "z"), (math.radians(v/self.GYRO_LSB_DPS) for v in g))),
                }
                self._last_sample = sample
            return sample
        except Exception as exc:
            self.error = str(exc)
            self.available = False
            with self._lock:
                self._last_sample = None
                self._close_bus()
            return {"available": False, "error": self.error}

    def telemetry(self) -> dict[str, Any]:
        # Telemetry is served by HTTP/WS request handlers. Return the GNC
        # loop's latest sample instead of doing a synchronous I2C transaction
        # inside a network request.
        with self._lock:
            sample = self._last_sample
            available = self.available
            error = self.error
        if not sample or time.monotonic() - sample["timestamp_monotonic"] > 1.0:
            return {
                "available": False,
                "driver": "mpu6050",
                "bus": self.bus_num,
                "address": hex(self.address),
                "error": (error or "Waiting for a fresh IMU sample") if available else error,
            }
        return {
            "available": True, "driver": "mpu6050", "bus": self.bus_num, "address": hex(self.address),
            "accel_g": {k: round(v, 4) for k,v in sample["accel_g"].items()},
            "gyro_rps": {k: round(v, 5) for k,v in sample["gyro_rps"].items()},
        }

    # Compatibility methods: the old UI called these Sense HAT display methods.
    # Without the LED matrix they intentionally do nothing.
    def show_status(self, status: str) -> None:
        return

    def message(self, text: str) -> None:
        return

    def close(self) -> None:
        with self._lock:
            self._close_bus()
            self.available = False


