# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PX4 Offboard MAVLink adapter for one drone."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
import time
from typing import Any

import numpy as np
from pymavlink import mavutil  # type: ignore[import-not-found, import-untyped]

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

POSITION_VELOCITY_TYPE_MASK = 0b0000111111000000
VELOCITY_ONLY_TYPE_MASK = 0b0000111111000111
HORIZONTAL_VELOCITY_ALTITUDE_TYPE_MASK = 0b0000110111100011
PX4_FORCE_ARM_MAGIC = 21196.0
PX4_SITL_DEMO_PARAMS = (
    ("NAV_DLL_ACT", 0.0, mavutil.mavlink.MAV_PARAM_TYPE_INT32),
    ("COM_DLL_EXCEPT", 4.0, mavutil.mavlink.MAV_PARAM_TYPE_INT32),
    ("COM_RCL_EXCEPT", 4.0, mavutil.mavlink.MAV_PARAM_TYPE_INT32),
    ("COM_RC_IN_MODE", 4.0, mavutil.mavlink.MAV_PARAM_TYPE_INT32),
    ("COM_OF_LOSS_T", 10.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32),
    ("COM_FAIL_ACT_T", 0.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32),
    ("COM_LOW_BAT_ACT", 0.0, mavutil.mavlink.MAV_PARAM_TYPE_INT32),
    ("COM_FLTT_LOW_ACT", 0.0, mavutil.mavlink.MAV_PARAM_TYPE_INT32),
    ("COM_DISARM_PRFLT", 60.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32),
    ("COM_ARM_BAT_MIN", -1.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32),
    ("CBRK_SUPPLY_CHK", 894281.0, mavutil.mavlink.MAV_PARAM_TYPE_INT32),
    ("BAT_LOW_THR", 0.12, mavutil.mavlink.MAV_PARAM_TYPE_REAL32),
    ("BAT_CRIT_THR", 0.05, mavutil.mavlink.MAV_PARAM_TYPE_REAL32),
    ("BAT_EMERGEN_THR", 0.03, mavutil.mavlink.MAV_PARAM_TYPE_REAL32),
    # Disable EKF2 GPS quality checks (PDOP, hAcc, drift, ...) so indoor bench arms despite a
    # weak GPS fix. Persisted on the FC; revert before outdoor flight (PX4 default ~245).
    ("EKF2_GPS_CHECK", 0.0, mavutil.mavlink.MAV_PARAM_TYPE_INT32),
    ("MPC_TKO_SPEED", 2.5, mavutil.mavlink.MAV_PARAM_TYPE_REAL32),
    ("MPC_Z_VEL_MAX_UP", 3.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32),
    ("MPC_Z_V_AUTO_UP", 3.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32),
)
PX4_REQUIRED_ARM_SENSOR_MASK = (
    mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_GYRO
    | mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_ACCEL
    | mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_MAG
    | mavutil.mavlink.MAV_SYS_STATUS_SENSOR_ABSOLUTE_PRESSURE
)
PX4_REQUIRED_ARM_SENSOR_NAMES = (
    (mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_GYRO, "gyro"),
    (mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_ACCEL, "accel"),
    (mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_MAG, "compass"),
    (mavutil.mavlink.MAV_SYS_STATUS_SENSOR_ABSOLUTE_PRESSURE, "barometer"),
)
PX4_STATUS_HISTORY_LIMIT = 12
PX4_COMMAND_ACK_HISTORY_LIMIT = 12


@dataclass(frozen=True)
class PX4DroneConfig:
    """MAVLink addressing and frame configuration for one PX4 drone."""

    key: str
    connection_string: str
    origin_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    source_system: int = 250
    local_position_hz: float = 30.0
    heartbeat_hz: float = 2.0
    sys_status_hz: float = 2.0
    global_position_hz: float = 5.0
    extended_sys_state_hz: float = 2.0


@dataclass
class PX4DroneSnapshot:
    """Current state snapshot for a PX4 drone."""

    key: str
    connected: bool = False
    armed: bool = False
    mode: str = "UNKNOWN"
    autopilot: int = 0
    base_mode: int = 0
    custom_mode: int = 0
    position_enu: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity_enu: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    battery_remaining: int = -1
    battery_voltage: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    relative_alt: float = 0.0
    amsl_alt: float = 0.0
    last_update_s: float = 0.0
    has_local_position: bool = False
    sensors_present: int = 0
    sensors_enabled: int = 0
    sensors_health: int = 0
    landed_state: int = 0
    landed_state_name: str = "MAV_LANDED_STATE_UNDEFINED"
    landed_state_s: float = 0.0
    takeoff_detected: bool = False
    takeoff_detected_s: float = 0.0
    last_command_ack: str = ""
    command_ack_history: list[dict[str, Any]] = field(default_factory=list)
    last_status_text: str = ""
    last_status_severity: int = 255
    last_status_text_s: float = 0.0
    status_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot."""
        return {
            "key": self.key,
            "connected": self.connected,
            "armed": self.armed,
            "mode": self.mode,
            "autopilot": self.autopilot,
            "base_mode": self.base_mode,
            "custom_mode": self.custom_mode,
            "position_enu": list(self.position_enu),
            "velocity_enu": list(self.velocity_enu),
            "battery_remaining": self.battery_remaining,
            "battery_voltage": self.battery_voltage,
            "lat": self.lat,
            "lon": self.lon,
            "relative_alt": self.relative_alt,
            "amsl_alt": self.amsl_alt,
            "last_update_s": self.last_update_s,
            "has_local_position": self.has_local_position,
            "sensors_present": self.sensors_present,
            "sensors_enabled": self.sensors_enabled,
            "sensors_health": self.sensors_health,
            "landed_state": self.landed_state,
            "landed_state_name": self.landed_state_name,
            "landed_state_s": self.landed_state_s,
            "takeoff_detected": self.takeoff_detected,
            "takeoff_detected_s": self.takeoff_detected_s,
            "last_command_ack": self.last_command_ack,
            "command_ack_history": list(self.command_ack_history),
            "last_status_text": self.last_status_text,
            "last_status_severity": self.last_status_severity,
            "last_status_text_s": self.last_status_text_s,
            "status_history": list(self.status_history),
        }


class PX4OffboardDrone:
    """One PX4 vehicle controlled through Offboard velocity setpoints."""

    def __init__(self, config: PX4DroneConfig) -> None:
        self.config = config
        self.master: Any | None = None
        self.target_system = 1
        self.target_component = 1
        self.origin_enu = np.asarray(config.origin_enu, dtype=float)
        self.snapshot = PX4DroneSnapshot(
            key=config.key,
            position_enu=list(self.origin_enu),
            last_update_s=time.time(),
        )
        self._last_gcs_heartbeat = 0.0
        self._io_lock = RLock()

    @property
    def connected(self) -> bool:
        """Return whether this MAVLink client has an active connection object."""
        return self.master is not None and self.snapshot.connected

    def connect(self, timeout: float = 30.0) -> bool:
        """Open the MAVLink connection and wait for a PX4 heartbeat."""
        try:
            logger.info(
                "Connecting PX4 drone",
                key=self.config.key,
                connection=self.config.connection_string,
            )
            self.master = mavutil.mavlink_connection(
                self.config.connection_string,
                source_system=self.config.source_system,
                autoreconnect=True,
            )
            heartbeat = self._wait_vehicle_heartbeat(timeout=timeout)
            if heartbeat is None:
                logger.error("No PX4 heartbeat", key=self.config.key)
                return False
            self.target_system = int(heartbeat.get_srcSystem())
            self.target_component = int(heartbeat.get_srcComponent())
            self.snapshot.connected = True
            self._handle_heartbeat(heartbeat)
            self.request_message_interval(
                mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, self.config.local_position_hz
            )
            self.request_message_interval(
                mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT, self.config.heartbeat_hz
            )
            self.request_message_interval(
                mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, self.config.sys_status_hz
            )
            self.request_message_interval(
                mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, self.config.global_position_hz
            )
            self.request_message_interval(
                mavutil.mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE,
                self.config.extended_sys_state_hz,
            )
            for _ in range(5):
                self.send_gcs_heartbeat(force=True)
                time.sleep(0.05)
            return True
        except Exception as exc:
            logger.error("PX4 connection failed", key=self.config.key, error=str(exc))
            self.snapshot.connected = False
            return False

    def _wait_vehicle_heartbeat(self, timeout: float) -> Any | None:
        """Wait for a vehicle heartbeat, ignoring QGC and other forwarded MAVLink clients."""
        if self.master is None:
            return None
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.master.recv_match(type="HEARTBEAT", blocking=True, timeout=0.2)
            if msg is None:
                continue
            if self._is_vehicle_heartbeat(msg):
                return msg
        return None

    def request_message_interval(self, message_id: int, hz: float) -> None:
        """Request a telemetry message stream rate from PX4."""
        if self.master is None:
            return
        interval_us = int(1_000_000 / hz)
        with self._io_lock:
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                message_id,
                interval_us,
                0,
                0,
                0,
                0,
                0,
            )

    def configure_sitl_failsafes(self) -> dict[str, bool]:
        """Apply local-SITL-only failsafe settings for repeatable Offboard demos."""
        results: dict[str, bool] = {}
        for name, value, param_type in PX4_SITL_DEMO_PARAMS:
            results[name] = self.param_set(name, value, param_type=param_type)
        return results

    def update(self, timeout: float = 0.0) -> PX4DroneSnapshot:
        """Drain telemetry messages and return the latest snapshot."""
        if self.master is None:
            return self.snapshot
        deadline = time.time() + timeout
        with self._io_lock:
            if self.master is None:
                return self.snapshot
            while True:
                msg = self.master.recv_match(blocking=False)
                if msg is not None:
                    self._handle_message(msg)
                    continue
                if timeout <= 0.0 or time.time() >= deadline:
                    return self.snapshot
                time.sleep(0.002)

    def send_velocity_enu(self, velocity_enu: np.ndarray[Any, Any] | list[float]) -> bool:
        """Send a world-frame ENU velocity setpoint to PX4 Offboard."""
        return self.send_velocity_only_enu(velocity_enu)

    def send_velocity_only_enu(self, velocity_enu: np.ndarray[Any, Any] | list[float]) -> bool:
        """Send a world-frame ENU velocity-only setpoint to PX4 Offboard."""
        if self.master is None:
            return False
        with self._io_lock:
            self.send_gcs_heartbeat()
            velocity = np.asarray(velocity_enu, dtype=float)
            self.master.mav.set_position_target_local_ned_send(
                int(time.time() * 1000) & 0xFFFFFFFF,
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                VELOCITY_ONLY_TYPE_MASK,
                0,
                0,
                0,
                float(velocity[0]),
                float(velocity[1]),
                -float(velocity[2]),
                0,
                0,
                0,
                0,
                0,
            )
        return True

    def send_position_velocity_enu(
        self,
        position_enu: np.ndarray[Any, Any] | list[float],
        velocity_enu: np.ndarray[Any, Any] | list[float],
    ) -> bool:
        """Send a world-frame ENU position setpoint with velocity feed-forward."""
        if self.master is None:
            return False
        with self._io_lock:
            self.send_gcs_heartbeat()
            position = np.asarray(position_enu, dtype=float) - self.origin_enu
            velocity = np.asarray(velocity_enu, dtype=float)
            self.master.mav.set_position_target_local_ned_send(
                int(time.time() * 1000) & 0xFFFFFFFF,
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                POSITION_VELOCITY_TYPE_MASK,
                float(position[0]),
                float(position[1]),
                -float(position[2]),
                float(velocity[0]),
                float(velocity[1]),
                -float(velocity[2]),
                0,
                0,
                0,
                0,
                0,
            )
        return True

    def send_horizontal_velocity_altitude_enu(
        self,
        altitude_enu_m: float,
        velocity_enu: np.ndarray[Any, Any] | list[float],
    ) -> bool:
        """Send ENU x/y velocity while PX4 holds an ENU altitude setpoint."""
        if self.master is None:
            return False
        with self._io_lock:
            self.send_gcs_heartbeat()
            velocity = np.asarray(velocity_enu, dtype=float)
            local_altitude = float(altitude_enu_m) - float(self.origin_enu[2])
            self.master.mav.set_position_target_local_ned_send(
                int(time.time() * 1000) & 0xFFFFFFFF,
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                HORIZONTAL_VELOCITY_ALTITUDE_TYPE_MASK,
                float("nan"),
                float("nan"),
                -local_altitude,
                float(velocity[0]),
                float(velocity[1]),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            )
        return True

    def arm(self, force: bool = False) -> bool:
        """Arm the PX4 vehicle."""
        if self.master is None:
            return False
        force_value = PX4_FORCE_ARM_MAGIC if force else 0.0
        with self._io_lock:
            self.send_gcs_heartbeat(force=True)
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,
                force_value,
                0,
                0,
                0,
                0,
                0,
            )
            return self._wait_command_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, timeout=5.0)

    def disarm(self, force: bool = False) -> bool:
        """Disarm the PX4 vehicle."""
        if self.master is None:
            return False
        force_value = PX4_FORCE_ARM_MAGIC if force else 0.0
        with self._io_lock:
            self.send_gcs_heartbeat(force=True)
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0,
                force_value,
                0,
                0,
                0,
                0,
                0,
            )
            return self._wait_command_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, timeout=2.0)

    def request_arm(self, force: bool = False) -> bool:
        """Send an arm request without blocking for ACK."""
        if self.master is None:
            return False
        force_value = PX4_FORCE_ARM_MAGIC if force else 0.0
        with self._io_lock:
            self.send_gcs_heartbeat(force=True)
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,
                force_value,
                0,
                0,
                0,
                0,
                0,
            )
        return True

    def set_mode(self, mode: str) -> bool:
        """Set a PX4 flight mode by name."""
        if self.master is None:
            return False
        mode_upper = mode.upper()
        px4_mode = mavutil.px4_map.get(mode_upper)
        if px4_mode is None:
            logger.error("Unknown PX4 mode", key=self.config.key, mode=mode)
            return False
        _, custom_main_mode, custom_sub_mode = px4_mode
        with self._io_lock:
            self.send_gcs_heartbeat(force=True)
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                custom_main_mode,
                custom_sub_mode,
                0,
                0,
                0,
                0,
            )
            accepted = self._wait_command_ack(mavutil.mavlink.MAV_CMD_DO_SET_MODE, timeout=2.0)
            if not accepted:
                self._record_status_text(
                    f"Mode change to {mode_upper} was not accepted",
                    mavutil.mavlink.MAV_SEVERITY_WARNING,
                )
                return False
        deadline = time.time() + 2.5
        while time.time() < deadline:
            self.send_gcs_heartbeat(force=True)
            snapshot = self.update(timeout=0.05)
            if self._is_px4_mode(snapshot, mode_upper):
                return True
            time.sleep(0.05)
        self._record_status_text(
            f"Mode change to {mode_upper} was accepted but not confirmed",
            mavutil.mavlink.MAV_SEVERITY_WARNING,
        )
        return False

    def request_mode(self, mode: str) -> bool:
        """Send a PX4 mode request without blocking for ACK."""
        if self.master is None:
            return False
        mode_upper = mode.upper()
        px4_mode = mavutil.px4_map.get(mode_upper)
        if px4_mode is None:
            logger.error("Unknown PX4 mode", key=self.config.key, mode=mode)
            return False
        _, custom_main_mode, custom_sub_mode = px4_mode
        with self._io_lock:
            self.send_gcs_heartbeat(force=True)
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                custom_main_mode,
                custom_sub_mode,
                0,
                0,
                0,
                0,
            )
        return True

    def is_mode(self, mode: str) -> bool:
        """Return whether the last heartbeat matches a PX4 mode name."""
        self.update(timeout=0.0)
        return self._is_px4_mode(self.snapshot, mode.upper())

    def land(self) -> bool:
        """Command PX4 to land at its current position."""
        return self._send_command(mavutil.mavlink.MAV_CMD_NAV_LAND, timeout=2.0)

    def takeoff(self, altitude_m: float) -> bool:
        """Command PX4 native takeoff to an altitude above launch."""
        if self.master is None:
            return False
        with self._io_lock:
            self.send_gcs_heartbeat(force=True)
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0,
                0,
                0,
                float("nan"),
                float("nan"),
                float("nan"),
                # MAV_CMD_NAV_TAKEOFF param7 is AMSL: command current ground AMSL + relative climb.
                self.snapshot.amsl_alt + max(altitude_m, 1.0),
            )
            return self._wait_command_ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, timeout=5.0)

    def request_takeoff(self, altitude_m: float) -> bool:
        """Send a PX4 native takeoff request without blocking for ACK."""
        if self.master is None:
            return False
        with self._io_lock:
            self.send_gcs_heartbeat(force=True)
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0,
                0,
                0,
                float("nan"),
                float("nan"),
                float("nan"),
                # MAV_CMD_NAV_TAKEOFF param7 is AMSL: command current ground AMSL + relative climb.
                self.snapshot.amsl_alt + max(altitude_m, 1.0),
            )
        return True

    def rtl(self) -> bool:
        """Command PX4 native return-to-launch."""
        if self.set_mode("RTL"):
            return True
        return self._send_command(mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, timeout=2.0)

    def param_set(
        self,
        name: str,
        value: float,
        *,
        param_type: int = mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        timeout: float = 3.0,
    ) -> bool:
        """Set one PX4 parameter and wait for the echoed PARAM_VALUE."""
        if self.master is None:
            return False
        with self._io_lock:
            self._drain_messages("PARAM_VALUE")
            encoded_name = name.encode("ascii")
            deadline = time.time() + timeout
            while time.time() < deadline:
                self.send_gcs_heartbeat(force=True)
                self.master.mav.param_set_send(
                    self.target_system,
                    self.target_component,
                    encoded_name,
                    float(value),
                    param_type,
                )
                attempt_deadline = min(deadline, time.time() + 0.5)
                while time.time() < attempt_deadline:
                    msg = self.master.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.1)
                    if msg is None:
                        continue
                    if self._param_name(msg.param_id) != name:
                        continue
                    if abs(float(msg.param_value) - float(value)) <= 1.0e-3:
                        return True
        logger.warning(
            "PX4 parameter was not confirmed", key=self.config.key, name=name, value=value
        )
        return False

    def core_sensor_failures(self) -> list[str]:
        """Return required PX4 core sensors that are missing or unhealthy."""
        self.update(timeout=0.0)
        failures: list[str] = []
        for flag, name in PX4_REQUIRED_ARM_SENSOR_NAMES:
            if not bool(self.snapshot.sensors_present & flag):
                failures.append(f"{name}:missing")
            elif not bool(self.snapshot.sensors_enabled & flag):
                failures.append(f"{name}:disabled")
            elif not bool(self.snapshot.sensors_health & flag):
                failures.append(f"{name}:unhealthy")
        return failures

    def core_sensors_ready(self) -> bool:
        """Return whether required PX4 arming sensors are present, enabled, and healthy."""
        self.update(timeout=0.0)
        required = PX4_REQUIRED_ARM_SENSOR_MASK
        return bool(
            (self.snapshot.sensors_present & required) == required
            and (self.snapshot.sensors_enabled & required) == required
            and (self.snapshot.sensors_health & required) == required
        )

    @staticmethod
    def _append_bounded(
        history: list[dict[str, Any]], entry: dict[str, Any], limit: int
    ) -> None:
        history.append(entry)
        if len(history) > limit:
            del history[: len(history) - limit]

    @staticmethod
    def _status_has_failure_text(text: str) -> bool:
        lower_text = text.lower()
        return (
            "preflight fail" in lower_text
            or " fail:" in lower_text
            or " failed" in lower_text
            or "failure" in lower_text
            or "timeout" in lower_text
            or "flight termination" in lower_text
        )

    def _record_status_text(self, text: str, severity_value: int = 255) -> None:
        now = time.time()
        severity = self._severity_name(severity_value)
        formatted = f"{severity}: {text}"
        self.snapshot.last_status_text = formatted
        self.snapshot.last_status_severity = severity_value
        self.snapshot.last_status_text_s = now
        self.snapshot.last_update_s = now
        self._append_bounded(
            self.snapshot.status_history,
            {
                "time_s": now,
                "severity": severity,
                "severity_value": severity_value,
                "text": text,
                "message": formatted,
            },
            PX4_STATUS_HISTORY_LIMIT,
        )
        if "takeoff detected" in text.lower():
            self.snapshot.takeoff_detected = True
            self.snapshot.takeoff_detected_s = now

    def _record_command_ack_text(self, ack: str) -> None:
        now = time.time()
        self.snapshot.last_command_ack = ack
        self.snapshot.last_update_s = now
        self._append_bounded(
            self.snapshot.command_ack_history,
            {"time_s": now, "ack": ack},
            PX4_COMMAND_ACK_HISTORY_LIMIT,
        )

    def _record_command_ack(self, command: int, result: int) -> None:
        self._record_command_ack_text(
            f"{self._command_name(command)}={self._mav_result_name(result)}"
        )

    def recent_critical_status(self, max_age_s: float = 5.0) -> str:
        """Return a recent PX4 failure status text that should block arming."""
        self.update(timeout=0.0)
        now = time.time()
        for entry in reversed(self.snapshot.status_history):
            if now - float(entry.get("time_s", 0.0)) > max_age_s:
                continue
            message = str(entry.get("message", ""))
            severity_value = int(entry.get("severity_value", 255))
            if (
                severity_value <= mavutil.mavlink.MAV_SEVERITY_ERROR
                or self._status_has_failure_text(message)
            ):
                return message
        return ""

    def send_gcs_heartbeat(self, force: bool = False) -> None:
        """Keep PX4's GCS-link health check satisfied for standalone SITL."""
        if self.master is None:
            return
        now = time.time()
        if not force and now - self._last_gcs_heartbeat < 0.2:
            return
        with self._io_lock:
            self.master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                mavutil.mavlink.MAV_STATE_ACTIVE,
            )
            self._last_gcs_heartbeat = now

    def close(self) -> None:
        """Close the MAVLink connection."""
        with self._io_lock:
            if self.master is not None:
                self.master.close()
                self.master = None
        self.snapshot.connected = False

    def _send_command(self, command: int, timeout: float = 2.0) -> bool:
        if self.master is None:
            return False
        with self._io_lock:
            self.send_gcs_heartbeat(force=True)
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                command,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )
            return self._wait_command_ack(command, timeout=timeout)

    def _wait_command_ack(self, command: int, timeout: float) -> bool:
        if self.master is None:
            return False
        with self._io_lock:
            deadline = time.time() + timeout
            while time.time() < deadline:
                msg = self.master.recv_match(blocking=True, timeout=0.2)
                if msg is None:
                    continue
                if not self._is_from_vehicle(msg):
                    continue
                self._handle_message(msg)
                if msg.get_type() != "COMMAND_ACK" or int(msg.command) != int(command):
                    continue
                result = int(msg.result)
                return result in (
                    mavutil.mavlink.MAV_RESULT_ACCEPTED,
                    mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
                )
            self._record_command_ack_text(f"{self._command_name(command)}=ACK_TIMEOUT")
            return False

    def _handle_message(self, msg: Any) -> None:
        if not self._is_from_vehicle(msg):
            return
        msg_type = msg.get_type()
        if msg_type == "HEARTBEAT":
            self._handle_heartbeat(msg)
        elif msg_type == "LOCAL_POSITION_NED":
            self.snapshot.position_enu = [
                float(self.origin_enu[0] + msg.x),
                float(self.origin_enu[1] + msg.y),
                float(self.origin_enu[2] - msg.z),
            ]
            self.snapshot.velocity_enu = [float(msg.vx), float(msg.vy), -float(msg.vz)]
            self.snapshot.relative_alt = self.snapshot.position_enu[2]
            self.snapshot.has_local_position = True
            self.snapshot.last_update_s = time.time()
        elif msg_type == "SYS_STATUS":
            self.snapshot.sensors_present = int(getattr(msg, "onboard_control_sensors_present", 0))
            self.snapshot.sensors_enabled = int(getattr(msg, "onboard_control_sensors_enabled", 0))
            self.snapshot.sensors_health = int(getattr(msg, "onboard_control_sensors_health", 0))
            self.snapshot.battery_remaining = int(getattr(msg, "battery_remaining", -1))
            self.snapshot.battery_voltage = float(getattr(msg, "voltage_battery", 0)) / 1000.0
            self.snapshot.last_update_s = time.time()
        elif msg_type == "GLOBAL_POSITION_INT":
            self.snapshot.lat = float(getattr(msg, "lat", 0)) / 1.0e7
            self.snapshot.lon = float(getattr(msg, "lon", 0)) / 1.0e7
            self.snapshot.relative_alt = float(getattr(msg, "relative_alt", 0)) / 1000.0
            self.snapshot.amsl_alt = float(getattr(msg, "alt", 0)) / 1000.0
            self.snapshot.last_update_s = time.time()
        elif msg_type == "EXTENDED_SYS_STATE":
            landed_state = int(getattr(msg, "landed_state", 0))
            self.snapshot.landed_state = landed_state
            self.snapshot.landed_state_name = self._enum_name(
                "MAV_LANDED_STATE", landed_state, "MAV_LANDED_STATE"
            )
            now = time.time()
            self.snapshot.landed_state_s = now
            self.snapshot.last_update_s = now
        elif msg_type == "STATUSTEXT":
            text = self._message_text(getattr(msg, "text", ""))
            if text:
                severity_value = int(getattr(msg, "severity", 255))
                self._record_status_text(text, severity_value)
        elif msg_type == "COMMAND_ACK":
            command = int(getattr(msg, "command", 0))
            result = int(getattr(msg, "result", -1))
            self._record_command_ack(command, result)

    def _handle_heartbeat(self, msg: Any) -> None:
        was_armed = self.snapshot.armed
        self.snapshot.autopilot = int(getattr(msg, "autopilot", 0))
        self.snapshot.base_mode = int(getattr(msg, "base_mode", 0))
        self.snapshot.custom_mode = int(getattr(msg, "custom_mode", 0))
        self.snapshot.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        if was_armed and not self.snapshot.armed:
            self.snapshot.takeoff_detected = False
            self.snapshot.takeoff_detected_s = 0.0
        try:
            self.snapshot.mode = self._px4_mode_name(
                self.snapshot.base_mode, self.snapshot.custom_mode
            )
        except Exception:
            self.snapshot.mode = str(getattr(msg, "custom_mode", "UNKNOWN"))
        self.snapshot.last_update_s = time.time()

    def _is_from_vehicle(self, msg: Any) -> bool:
        """Return whether a MAVLink message came from this PX4 vehicle."""
        try:
            source_system = int(msg.get_srcSystem())
        except Exception:
            return True
        return source_system == int(self.target_system)

    @staticmethod
    def _is_vehicle_heartbeat(msg: Any) -> bool:
        """Return whether a heartbeat belongs to a vehicle autopilot, not QGC or DimOS."""
        try:
            vehicle_type = int(getattr(msg, "type", 0))
            autopilot = int(getattr(msg, "autopilot", 0))
        except Exception:
            return False
        return (
            vehicle_type != mavutil.mavlink.MAV_TYPE_GCS
            and autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID
        )

    def _drain_messages(self, message_type: str) -> None:
        if self.master is None:
            return
        while self.master.recv_match(type=message_type, blocking=False) is not None:
            pass

    @staticmethod
    def _param_name(param_id: Any) -> str:
        if isinstance(param_id, bytes):
            return param_id.decode("ascii", errors="ignore").rstrip("\x00")
        return str(param_id).rstrip("\x00")

    @staticmethod
    def _message_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore").rstrip("\x00")
        return str(value).rstrip("\x00")

    @staticmethod
    def _enum_name(enum_name: str, value: int, fallback: str) -> str:
        try:
            return str(mavutil.mavlink.enums[enum_name][value].name)
        except Exception:
            return f"{fallback}_{value}"

    @classmethod
    def _command_name(cls, command: int) -> str:
        return cls._enum_name("MAV_CMD", command, "MAV_CMD")

    @classmethod
    def _mav_result_name(cls, result: int) -> str:
        return cls._enum_name("MAV_RESULT", result, "MAV_RESULT")

    @classmethod
    def _severity_name(cls, severity: int) -> str:
        return cls._enum_name("MAV_SEVERITY", severity, "MAV_SEVERITY")

    @staticmethod
    def _px4_main_mode(custom_mode: int) -> int:
        return (custom_mode & 0xFF0000) >> 16

    @staticmethod
    def _px4_sub_mode(custom_mode: int) -> int:
        return (custom_mode & 0xFF000000) >> 24

    @classmethod
    def _px4_mode_name(cls, base_mode: int, custom_mode: int) -> str:
        main_mode = cls._px4_main_mode(custom_mode)
        sub_mode = cls._px4_sub_mode(custom_mode)
        for name, (_, expected_main, expected_sub) in mavutil.px4_map.items():
            if main_mode != expected_main:
                continue
            if expected_sub == 0 or sub_mode == expected_sub:
                return str(name)
        try:
            return str(mavutil.interpret_px4_mode(base_mode, custom_mode))
        except Exception:
            return f"Mode(base=0x{base_mode:02x}, custom=0x{custom_mode:08x})"

    @classmethod
    def _is_px4_mode(cls, snapshot: PX4DroneSnapshot, mode: str) -> bool:
        px4_mode = mavutil.px4_map.get(mode)
        if px4_mode is None:
            return False
        _, expected_main_raw, expected_sub_raw = px4_mode
        expected_main = int(expected_main_raw)
        expected_sub = int(expected_sub_raw)
        main_mode = cls._px4_main_mode(snapshot.custom_mode)
        sub_mode = cls._px4_sub_mode(snapshot.custom_mode)
        return main_mode == expected_main and (expected_sub == 0 or sub_mode == expected_sub)
