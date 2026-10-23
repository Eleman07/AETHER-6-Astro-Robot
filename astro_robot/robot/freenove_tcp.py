"""Fail-closed client for the Freenove Big Hexapod local TCP command server.

This adapter never probes or writes I2C itself. The vendor server must be
running on the Pi, preferably bound to loopback. Motion is disarmed unless
ASTRO_FREENOVE_ACTUATION_ARMED is explicitly enabled.
"""
from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass


@dataclass
class FreenoveRobotState:
    connected: bool = False
    sonic_cm: float | None = None
    battery_1_v: float | None = None
    battery_2_v: float | None = None
    last_error: str | None = "Freenove TCP server has not been contacted"
    sonic_updated_monotonic: float = 0.0
    power_updated_monotonic: float = 0.0
    imu_accel_mps2: tuple[float, float, float] | None = None
    imu_gyro_dps: tuple[float, float, float] | None = None
    imu_updated_monotonic: float = 0.0
    imu_error: str | None = "Freenove TCP protocol does not expose raw IMU samples"


class FreenoveTCPRobot:
    """Narrow, bounded subset of Freenove's newline-delimited TCP protocol."""

    def __init__(self, host: str, port: int, armed: bool, timeout_s: float = 1.0,
                 sonar_enabled: bool = True):
        self.host = host
        self.port = int(port)
        self.armed = bool(armed)
        self.timeout_s = max(0.2, min(5.0, float(timeout_s)))
        self.sonar_enabled = bool(sonar_enabled)
        self.state = FreenoveRobotState()
        self._sock: socket.socket | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._rx_buffer = b""

    @property
    def latest_jpeg(self) -> bytes | None:
        return self._latest_jpeg

    def connect(self) -> None:
        with self._lock:
            self._disconnect_locked()
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
                sock.settimeout(self.timeout_s)
                self._sock = sock
                self._rx_buffer = b""
                self.state.connected = True
                self.state.last_error = None
            except OSError as exc:
                self.state.connected = False
                self.state.last_error = f"Freenove TCP {self.host}:{self.port} unavailable: {exc}"

    def disconnect(self) -> None:
        self.stop()
        self._stop.set()
        with self._lock:
            self._disconnect_locked()

    def _disconnect_locked(self) -> None:
        sock, self._sock = self._sock, None
        self.state.connected = False
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def ensure_connected(self) -> None:
        if not self.state.connected:
            self.connect()

    def start_video(self) -> None:
        # Freenove video is a separate port/protocol and is intentionally not
        # brought up by the motor controller adapter.
        return

    def start_telemetry(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._connection_loop, name="astro-freenove-tcp", daemon=True)
        self._thread.start()

    def _connection_loop(self) -> None:
        while not self._stop.is_set():
            if not self.state.connected:
                self.connect()
            self._poll_sonar()
            self._stop.wait(0.35)

    def _poll_sonar(self) -> None:
        # The Freenove server handles CMD_SONIC synchronously in the same
        # receive loop as movement. If the optional sensor hangs, polling it
        # can starve CMD_MOVE processing; skip the request entirely when off.
        if not self.sonar_enabled or not self.state.connected:
            return
        values = self._query("CMD_SONIC", "CMD_SONIC")
        if values and 0 < values[0] <= 400:
            self.state.sonic_cm = values[0]
            self.state.sonic_updated_monotonic = time.monotonic()

    def _send(self, command: str, *, motion: bool = False) -> None:
        if motion and not self.armed:
            raise RuntimeError("Freenove movement is disarmed (ASTRO_FREENOVE_ACTUATION_ARMED=false)")
        self.ensure_connected()
        payload = (command.strip() + "\n").encode("ascii")
        with self._lock:
            if self._sock is None:
                raise RuntimeError(self.state.last_error or "Freenove TCP server unavailable")
            try:
                self._sock.sendall(payload)
            except OSError as exc:
                self.state.last_error = f"Freenove TCP send failed: {exc}"
                self._disconnect_locked()
                raise RuntimeError(self.state.last_error) from exc

    def _query(self, command: str, prefix: str) -> list[float] | None:
        """Read a vendor sensor reply on the telemetry thread only."""
        self.ensure_connected()
        with self._lock:
            if self._sock is None:
                return None
            try:
                self._sock.sendall((command + "\n").encode("ascii"))
                deadline = time.monotonic() + self.timeout_s
                while time.monotonic() < deadline:
                    if b"\n" in self._rx_buffer:
                        line, self._rx_buffer = self._rx_buffer.split(b"\n", 1)
                        parts = line.decode("utf-8", errors="replace").strip().split("#")
                        if parts[0] == prefix:
                            if prefix == "CMD_SONIC":
                                # The Freenove server may reply with an empty or
                                # non-numeric range when the optional ultrasonic
                                # sensor is absent. Treat that as missing sonar
                                # data; never tear down the motor-control socket.
                                try:
                                    values = [float(part) for part in parts[1:] if part.strip()]
                                    return values or None
                                except ValueError:
                                    return None
                            return [float(part) for part in parts[1:]]
                        continue
                    self._sock.settimeout(max(0.05, deadline - time.monotonic()))
                    chunk = self._sock.recv(256)
                    if not chunk:
                        raise OSError("Freenove server closed the command socket")
                    self._rx_buffer += chunk
                return None
            except socket.timeout as exc:
                # Sonar is optional. The Freenove firmware can receive
                # CMD_SONIC without replying when the sensor is absent or
                # disconnected. A read timeout must not tear down the shared
                # single-client socket used for movement commands.
                if prefix == "CMD_SONIC":
                    self.state.last_error = f"Optional Freenove sonar did not reply: {exc}"
                    return None
                self.state.last_error = f"Freenove {prefix} query failed: {exc}"
                self._disconnect_locked()
                return None
            except (OSError, ValueError) as exc:
                self.state.last_error = f"Freenove {prefix} query failed: {exc}"
                self._disconnect_locked()
                return None

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    def move(self, x: float, y: float, speed: int = 2, yaw: float = 0, gait: int = 1) -> None:
        # Match the vendor FNK0052 walking example (translation 35, speed 10)
        # while keeping every input bounded and the Pi dead-man stop active.
        x = round(self._clamp(x, -35, 35))
        y = round(self._clamp(y, -35, 35))
        speed = max(2, min(10, int(speed)))
        yaw = round(self._clamp(yaw, -10, 10))
        gait = max(1, min(4, int(gait)))
        self._send(f"CMD_MOVE#{gait}#{x:g}#{y:g}#{speed}#{yaw:g}", motion=True)

    def stop(self) -> None:
        # Emergency stop is allowed while disarmed.
        if self.state.connected:
            try:
                self._send("CMD_MOVE#1#0#0#2#0")
            except RuntimeError:
                pass

    def servo_power(self, enabled: bool) -> None:
        raise RuntimeError("Remote servo power toggling is disabled; use the physical LOAD switch")

    def balance(self, enabled: bool) -> None:
        if enabled and not self.armed:
            raise RuntimeError("Freenove balance is disarmed until ASTRO_FREENOVE_ACTUATION_ARMED=true")
        if enabled:
            self._send("CMD_BALANCE#1", motion=True)
        else:
            # Freenove's balance loop exits when the next motion command is queued.
            self._send("CMD_MOVE#1#0#0#2#0")

    def buzzer(self, enabled: bool) -> None:
        self._send(f"CMD_BUZZER#{1 if enabled else 0}")

    def led_rgb(self, red: int, green: int, blue: int) -> None:
        """Set the Freenove board's RGB LEDs using its documented TCP command."""
        channels = (int(red), int(green), int(blue))
        if any(channel < 0 or channel > 255 for channel in channels):
            raise ValueError("RGB channels must be integers from 0 to 255")
        self._send("CMD_LED#{}#{}#{}".format(*channels))

    def head(self, axis: int, angle: int) -> None:
        raise RuntimeError("Camera pan/tilt is disabled in the initial Freenove integration")

    def attitude(self, roll: float, pitch: float, yaw: float) -> None:
        roll = self._clamp(roll, -5, 5)
        pitch = self._clamp(pitch, -5, 5)
        yaw = self._clamp(yaw, -5, 5)
        self._send(f"CMD_ATTITUDE#{round(roll)}#{round(pitch)}#{round(yaw)}", motion=True)

    def position(self, x: float, y: float, z: float) -> None:
        x = self._clamp(x, -5, 5)
        y = self._clamp(y, -5, 5)
        z = self._clamp(z, -5, 5)
        self._send(f"CMD_POSITION#{round(x)}#{round(y)}#{round(z)}", motion=True)

    def request_sonic(self) -> None:
        # The background thread owns the vendor's single-client socket.
        return

    def request_power(self) -> None:
        # Do not query: Freenove's server may sound its low-voltage alarm as a
        # side effect of CMD_POWER. Battery telemetry needs a read-only vendor API.
        return
