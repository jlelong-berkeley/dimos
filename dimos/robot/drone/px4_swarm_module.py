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

"""DimOS module for a PX4 MAVLink swarm controlled through velocity setpoints."""

from __future__ import annotations

from itertools import permutations
import json
import math
from threading import Event, RLock, Thread, current_thread
import time
from typing import TYPE_CHECKING, Any

from dimos_lcm.std_msgs import String
import numpy as np

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import Out
from dimos.mapping.types import LatLon
from dimos.robot.drone.px4_offboard_connection import PX4DroneConfig, PX4OffboardDrone
from dimos.robot.drone.px4_swarm_behavior import PROJECT3_BEST_DESIGN, SwarmBehaviorLaw
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_SWARM_SIZE = 3
DEFAULT_SITL_PORT_START = 14540
DEFAULT_MIN_SEPARATION_M = 3.0
DEFAULT_TAKEOFF_ALTITUDE_M = 5.0
DEFAULT_TASK_ALTITUDE_M = 8.0
DEFAULT_LINE_SPACING_M = 6.0
AIRBORNE_ALTITUDE_M = 0.5
MOTION_STATIONARY_SPEED_MPS = 0.25
MOTION_STATIONARY_DWELL_S = 1.5
MOTION_PRE_COMMAND_HOLD_S = 0.5
MOTION_TIMEOUT_HOLD_S = 1.0
MOTION_PROGRESS_LOG_INTERVAL_S = 5.0
SWARM_MAX_VERTICAL_SPEED_MPS = 2.0
NATIVE_TAKEOFF_ALTITUDE_M = 1.5


def _default_connection_strings(n_drones: int) -> list[str]:
    return [f"udpin:127.0.0.1:{DEFAULT_SITL_PORT_START + index}" for index in range(n_drones)]


def _default_origins(n_drones: int, spacing_m: float = DEFAULT_LINE_SPACING_M) -> list[list[float]]:
    y_values = (np.arange(n_drones) - ((n_drones - 1) / 2.0)) * spacing_m
    return [[0.0, float(y), 0.0] for y in y_values]


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class PX4SwarmModule(Module):
    """Manage multiple PX4 drones behind one swarm-aware DimOS skill surface."""

    fleet_status: Out[Any]
    fleet_telemetry: Out[Any]
    gps_location: Out[LatLon]

    def __init__(
        self,
        connection_strings: list[str] | None = None,
        origin_positions_enu: list[list[float]] | None = None,
        drone_keys: list[str] | None = None,
        command_hz: float = 20.0,
        min_separation_m: float = DEFAULT_MIN_SEPARATION_M,
        takeoff_altitude_m: float = DEFAULT_TAKEOFF_ALTITUDE_M,
        default_task_altitude_m: float = DEFAULT_TASK_ALTITUDE_M,
        configure_sitl_failsafes: bool = False,
        design_vector: list[float] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the PX4 swarm module.

        Args:
            connection_strings: MAVLink connection string per drone.
            origin_positions_enu: Per-drone local-frame offsets in shared ENU meters.
            drone_keys: Optional names such as x500_0, x500_1, x500_2.
            command_hz: Velocity command loop rate.
            min_separation_m: Guardrail spacing used by the swarm law.
            takeoff_altitude_m: Default takeoff altitude in meters.
            default_task_altitude_m: Default mission altitude in meters.
            configure_sitl_failsafes: True only for local PX4 SITL demos.
            design_vector: Optional replacement for the Project3 optimized design.
        """
        super().__init__(*args, **kwargs)
        n_drones = len(connection_strings) if connection_strings else DEFAULT_SWARM_SIZE
        self.connection_strings = connection_strings or _default_connection_strings(n_drones)
        self.origin_positions_enu = origin_positions_enu or _default_origins(
            len(self.connection_strings)
        )
        self.drone_keys = drone_keys or [
            f"x500_{index}" for index in range(len(self.connection_strings))
        ]
        self.command_hz = max(command_hz, 2.0)
        self.min_separation_m = min_separation_m
        self.takeoff_altitude_m = takeoff_altitude_m
        self.default_task_altitude_m = default_task_altitude_m
        self.configure_sitl_failsafes = configure_sitl_failsafes
        self.design_vector = design_vector or list(PROJECT3_BEST_DESIGN)
        self._behavior = SwarmBehaviorLaw.from_design_vector(
            self.design_vector,
            min_separation=self.min_separation_m,
            max_vertical_speed=SWARM_MAX_VERTICAL_SPEED_MPS,
        )
        self._drones: list[PX4OffboardDrone] = []
        self._running = False
        self._telemetry_thread: Thread | None = None
        self._state_lock = RLock()
        self._command_lock = RLock()
        self._velocity_control_ready = False
        self._hold_stop_event: Event | None = None
        self._hold_thread: Thread | None = None
        self._hold_targets: np.ndarray[Any, Any] | None = None
        self._startup_blocker: str | None = None
        self._last_status_publish_s = 0.0
        self._home_positions = np.asarray(self.origin_positions_enu, dtype=float)

    @rpc
    def start(self) -> None:
        """Connect to all configured PX4 MAVLink endpoints."""
        super().start()
        if len(self.connection_strings) != len(self.origin_positions_enu):
            raise ValueError(
                "connection_strings and origin_positions_enu must have the same length"
            )
        if len(self.connection_strings) != len(self.drone_keys):
            raise ValueError("connection_strings and drone_keys must have the same length")

        self._drones = []
        for key, connection_string, origin in zip(
            self.drone_keys,
            self.connection_strings,
            self.origin_positions_enu,
            strict=True,
        ):
            origin_tuple = (float(origin[0]), float(origin[1]), float(origin[2]))
            drone = PX4OffboardDrone(
                PX4DroneConfig(
                    key=key,
                    connection_string=connection_string,
                    origin_enu=origin_tuple,
                )
            )
            drone.connect(timeout=30.0)
            self._drones.append(drone)

        self._prime_telemetry(seconds=2.0)
        self._startup_blocker = self._wait_for_position_stream(timeout_s=30.0)
        if self.configure_sitl_failsafes and self._startup_blocker is None:
            self._startup_blocker = self._configure_sitl_failsafes()

        self._running = True
        self._telemetry_thread = Thread(target=self._telemetry_loop, daemon=True)
        self._telemetry_thread.start()
        self._home_positions = self._positions_enu()
        self._publish_fleet_status(force=True)
        logger.info("PX4SwarmModule started", drones=[drone.config.key for drone in self._drones])

    @rpc
    def stop(self) -> None:
        """Stop the module and close MAVLink clients."""
        self._running = False
        self._stop_background_hold()
        if self._telemetry_thread and self._telemetry_thread.is_alive():
            self._telemetry_thread.join(timeout=2.0)
        for drone in self._drones:
            drone.close()
        self._drones.clear()
        logger.info("PX4SwarmModule stopped")
        super().stop()

    def _telemetry_loop(self) -> None:
        while self._running:
            try:
                for drone in self._drones:
                    drone.send_gcs_heartbeat()
                    drone.update(timeout=0.002)
                self._publish_fleet_status()
            except Exception as exc:
                logger.debug("PX4 swarm telemetry loop error", error=str(exc))
            time.sleep(0.03)

    def _publish_fleet_status(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_status_publish_s < 0.5:
            return
        self._last_status_publish_s = now
        payload = self._fleet_state_dict()
        swarm_gps = self._swarm_centroid_gps(payload["drones"])
        if swarm_gps is not None:
            payload["gps_location"] = {
                "lat": swarm_gps.lat,
                "lon": swarm_gps.lon,
                "alt": swarm_gps.alt,
            }
            self.gps_location.publish(swarm_gps)
        encoded = String(json.dumps(payload))
        self.fleet_status.publish(encoded)
        self.fleet_telemetry.publish(encoded)

    @staticmethod
    def _swarm_centroid_gps(drones: list[dict[str, Any]]) -> LatLon | None:
        valid = [
            drone
            for drone in drones
            if abs(_to_float(drone.get("lat"))) > 1.0e-7
            and abs(_to_float(drone.get("lon"))) > 1.0e-7
        ]
        if not valid:
            return None

        lat = sum(_to_float(drone.get("lat")) for drone in valid) / len(valid)
        lon = sum(_to_float(drone.get("lon")) for drone in valid) / len(valid)
        alt = sum(_to_float(drone.get("relative_alt")) for drone in valid) / len(valid)
        return LatLon(lat=lat, lon=lon, alt=alt)

    def _fleet_state_dict(self) -> dict[str, Any]:
        with self._state_lock:
            drones = [drone.snapshot.to_dict() for drone in self._drones]
        return {
            "drones": drones,
            "pairwise_distances_m": self._pairwise_distances_dict(),
            "min_separation_m": self.min_separation_m,
            "all_connected": all(bool(item["connected"]) for item in drones) if drones else False,
            "velocity_control_ready": self._velocity_control_ready,
            "hold_active": self._hold_active(),
            "timestamp_s": time.time(),
        }

    def _positions_enu(self) -> np.ndarray[Any, Any]:
        if not self._drones:
            return np.zeros((0, 3), dtype=float)
        positions = []
        for index, drone in enumerate(self._drones):
            snapshot = drone.update(timeout=0.0)
            if snapshot.has_local_position:
                positions.append(snapshot.position_enu)
            else:
                positions.append(self.origin_positions_enu[index])
        return np.asarray(positions, dtype=float)

    def _velocities_enu(self) -> np.ndarray[Any, Any]:
        if not self._drones:
            return np.zeros((0, 3), dtype=float)
        velocities = []
        for drone in self._drones:
            snapshot = drone.update(timeout=0.0)
            velocities.append(snapshot.velocity_enu)
        return np.asarray(velocities, dtype=float)

    def _pairwise_distances_dict(self) -> dict[str, float]:
        positions = self._positions_enu()
        distances: dict[str, float] = {}
        for i in range(len(self._drones)):
            for j in range(i + 1, len(self._drones)):
                key = f"{self._drones[i].config.key}:{self._drones[j].config.key}"
                distances[key] = float(np.linalg.norm(positions[i] - positions[j]))
        return distances

    def _resolve_drone_index(self, drone: str) -> int:
        target = drone.strip().lower().replace(" ", "").replace("-", "_")
        if not target:
            raise ValueError("Specify a drone key such as x500_0")
        for index, px4_drone in enumerate(self._drones):
            key = px4_drone.config.key.lower()
            aliases = {
                key,
                key.replace("_", ""),
                str(index),
                f"drone_{index}",
                f"drone{index}",
                f"x500_{index}",
                f"x500{index}",
            }
            if target in aliases:
                return index
        raise ValueError(f"Unknown drone target: {drone}")

    def _ensure_connected(self) -> str | None:
        if not self._drones:
            return "Failed: no PX4 drones are configured"
        disconnected = [drone.config.key for drone in self._drones if not drone.connected]
        if disconnected:
            return f"Failed: disconnected drones: {', '.join(disconnected)}"
        return None

    def _ensure_velocity_control(self, require_airborne: bool = True) -> str | None:
        failure = self._ensure_connected()
        if failure is not None:
            return failure
        if self._velocity_control_ready:
            state_failure = self._velocity_control_state_failure(require_airborne=require_airborne)
            if state_failure is None:
                return None
            logger.warning("PX4 velocity control readiness expired", reason=state_failure)
            self._velocity_control_ready = False
        if self._startup_blocker is not None:
            return self._startup_blocker

        position_failure = self._wait_for_position_stream(timeout_s=20.0)
        if position_failure is not None:
            self._cleanup_control_setup_failure()
            return position_failure
        health_failure = self._wait_for_armable_health(timeout_s=12.0)
        if health_failure is not None:
            self._cleanup_control_setup_failure()
            return health_failure

        setup_failure = self._run_with_zero_velocity_stream(
            lambda: self._enter_offboard_and_arm_all(use_native_takeoff=require_airborne)
        )
        if setup_failure is not None:
            self._cleanup_control_setup_failure()
            return setup_failure

        self._velocity_control_ready = True
        state_failure = self._velocity_control_state_failure(require_airborne=require_airborne)
        if state_failure is not None:
            self._cleanup_control_setup_failure()
            return state_failure
        return None

    def _velocity_control_state_failure(self, require_airborne: bool) -> str | None:
        failure = self._ensure_connected()
        if failure is not None:
            return failure
        failures: list[str] = []
        for drone in self._drones:
            snapshot = drone.update(timeout=0.02)
            if not snapshot.has_local_position:
                failures.append(self._drone_failure_detail(drone, "has no local position"))
                continue
            if not snapshot.armed:
                failures.append(self._drone_failure_detail(drone, "is disarmed"))
                continue
            if require_airborne and not self._snapshot_airborne(snapshot):
                failures.append(
                    self._drone_failure_detail(
                        drone,
                        f"is not airborne altitude={snapshot.position_enu[2]:.1f}m",
                    )
                )
                continue
            if not drone.is_mode("OFFBOARD"):
                failures.append(
                    self._drone_failure_detail(drone, f"is in {snapshot.mode}, not OFFBOARD")
                )
        if failures:
            return "Failed: velocity control unavailable: " + "; ".join(failures)
        return None

    def _airborne_state_failure(self) -> str | None:
        failure = self._ensure_connected()
        if failure is not None:
            return failure
        failures: list[str] = []
        for drone in self._drones:
            snapshot = drone.update(timeout=0.02)
            if not snapshot.has_local_position:
                failures.append(self._drone_failure_detail(drone, "has no local position"))
            elif not snapshot.armed:
                failures.append(self._drone_failure_detail(drone, "is disarmed"))
            elif not self._snapshot_airborne(snapshot):
                failures.append(
                    self._drone_failure_detail(
                        drone,
                        f"is not airborne altitude={snapshot.position_enu[2]:.1f}m",
                    )
                )
        if failures:
            return "Failed: hover requires airborne drones; use takeoff_swarm first: " + "; ".join(
                failures
            )
        return None

    @staticmethod
    def _snapshot_airborne(snapshot: Any) -> bool:
        if snapshot.position_enu[2] >= AIRBORNE_ALTITUDE_M:
            return True
        status_recent = time.time() - snapshot.last_status_text_s <= 10.0
        return bool(
            snapshot.armed
            and status_recent
            and "takeoff detected" in snapshot.last_status_text.lower()
        )

    def _enter_offboard_and_arm_all(self, use_native_takeoff: bool = True) -> str | None:
        time.sleep(2.0)
        reset_failure = self._reset_landing_drones_before_arm()
        if reset_failure is not None:
            return reset_failure
        mode_reset_failure = self._reset_takeoff_modes_before_arm()
        if mode_reset_failure is not None:
            return mode_reset_failure
        force_arm = self.configure_sitl_failsafes
        armed_failure = self._arm_all(timeout_s=45.0, force=force_arm)
        if armed_failure is not None:
            return armed_failure

        if use_native_takeoff:
            takeoff_failure = self._native_takeoff_until_airborne()
            if takeoff_failure is not None:
                return takeoff_failure

        time.sleep(0.5)
        offboard_failures = []
        for drone in self._drones:
            if not drone.set_mode("OFFBOARD"):
                offboard_failures.append(
                    self._drone_failure_detail(drone, "rejected OFFBOARD mode")
                )
        if offboard_failures and not use_native_takeoff:
            takeoff_failure = self._native_takeoff_until_airborne()
            if takeoff_failure is not None:
                return takeoff_failure
            offboard_failures = []
            for drone in self._drones:
                if not drone.set_mode("OFFBOARD"):
                    offboard_failures.append(
                        self._drone_failure_detail(drone, "rejected OFFBOARD mode")
                    )
        if offboard_failures:
            return "Failed: " + "; ".join(offboard_failures)
        offboard_failure = self._wait_until_mode("OFFBOARD", timeout_s=5.0)
        if offboard_failure is not None:
            return offboard_failure
        return None

    def _reset_landing_drones_before_arm(self) -> str | None:
        """Disarm drones that are still in PX4 LAND at ground level before rearming."""
        reset_drones: list[PX4OffboardDrone] = []
        for drone in self._drones:
            snapshot = drone.update(timeout=0.02)
            status_text = snapshot.last_status_text.lower()
            landing_state = (
                "land" in snapshot.mode.lower()
                or "landing" in status_text
                or "disarmed by landing" in status_text
            )
            near_ground = snapshot.position_enu[2] <= AIRBORNE_ALTITUDE_M + 0.3
            if snapshot.armed and near_ground and landing_state:
                reset_drones.append(drone)
                drone.disarm()
            if near_ground and landing_state:
                drone.request_mode("LOITER")
        if not reset_drones:
            return None

        deadline = time.time() + 15.0
        still_armed = reset_drones
        while time.time() < deadline:
            still_armed = [drone for drone in reset_drones if drone.update(timeout=0.02).armed]
            if not still_armed:
                time.sleep(1.0)
                return None
            time.sleep(0.1)

        failures = [
            self._drone_failure_detail(drone, "did not leave LAND/disarm before rearm")
            for drone in still_armed
        ]
        return "Failed: " + "; ".join(failures)

    def _reset_takeoff_modes_before_arm(self) -> str | None:
        """Move landed or low-altitude drones out of terminal modes before arming."""
        failures: list[str] = []
        ground_ok_modes = {"LOITER", "MANUAL", "POSCTL", "ALTCTL"}
        for drone in self._drones:
            snapshot = drone.update(timeout=0.02)
            if not snapshot.has_local_position:
                continue
            near_ground = snapshot.position_enu[2] <= AIRBORNE_ALTITUDE_M + 0.3
            status_text = snapshot.last_status_text.lower()
            reset_needed = (
                snapshot.mode.upper() not in ground_ok_modes
                or "land" in snapshot.mode.lower()
                or "landing" in status_text
                or "disarmed by landing" in status_text
            )
            if near_ground and reset_needed and not drone.set_mode("LOITER") and snapshot.armed:
                failures.append(
                    self._drone_failure_detail(
                        drone, "could not reset mode to LOITER before takeoff"
                    )
                )
        if failures:
            return "Failed: " + "; ".join(failures)
        return None

    def _native_takeoff_until_airborne(self) -> str | None:
        target_altitude = max(min(self.takeoff_altitude_m, NATIVE_TAKEOFF_ALTITUDE_M), 1.2)
        min_airborne_altitude = min(0.25, target_altitude * 0.25)
        positions = self._positions_enu()
        low_drones = [
            drone
            for drone, position in zip(self._drones, positions, strict=True)
            if position[2] < min_airborne_altitude
        ]
        if not low_drones:
            return None

        for drone in low_drones:
            requested = drone.request_takeoff(target_altitude)
            logger.info("PX4 takeoff command requested", key=drone.config.key, requested=requested)

        deadline = time.time() + 15.0
        next_takeoff_request_s = time.time() + 2.0
        while time.time() < deadline:
            snapshots = [drone.update(timeout=0.03) for drone in self._drones]
            positions = np.asarray([snapshot.position_enu for snapshot in snapshots], dtype=float)
            takeoff_detected = [
                "takeoff detected" in snapshot.last_status_text.lower() for snapshot in snapshots
            ]
            if all(
                position[2] >= min_airborne_altitude or detected
                for position, detected in zip(positions, takeoff_detected, strict=True)
            ):
                return None
            now = time.time()
            if now >= next_takeoff_request_s:
                for drone, position in zip(self._drones, positions, strict=True):
                    if position[2] < min_airborne_altitude:
                        drone.request_takeoff(target_altitude)
                next_takeoff_request_s = now + 2.0
            for drone in self._drones:
                drone.send_gcs_heartbeat(force=True)
            time.sleep(0.1)

        grace_deadline = time.time() + 3.0
        while time.time() < grace_deadline:
            snapshots = [drone.update(timeout=0.05) for drone in self._drones]
            positions = np.asarray([snapshot.position_enu for snapshot in snapshots], dtype=float)
            if all(
                position[2] >= min_airborne_altitude or self._snapshot_airborne(snapshot)
                for position, snapshot in zip(positions, snapshots, strict=True)
            ):
                return None
            time.sleep(0.1)

        details = []
        positions = np.asarray(
            [drone.update(timeout=0.1).position_enu for drone in self._drones],
            dtype=float,
        )
        for drone, position in zip(self._drones, positions, strict=True):
            if position[2] < min_airborne_altitude:
                details.append(
                    self._drone_failure_detail(
                        drone,
                        f"did not reach airborne altitude {min_airborne_altitude:.1f}m",
                    )
                )
        if not details:
            return None
        return "Failed: " + "; ".join(details)

    def _wait_until_mode(self, mode: str, timeout_s: float) -> str | None:
        deadline = time.time() + max(0.0, timeout_s)
        next_request_s = 0.0
        not_in_mode: list[PX4OffboardDrone] = []
        while time.time() < deadline:
            not_in_mode = [drone for drone in self._drones if not drone.is_mode(mode)]
            if not not_in_mode:
                return None
            now = time.time()
            if now >= next_request_s:
                for drone in not_in_mode:
                    drone.request_mode(mode)
                next_request_s = now + 1.0
            time.sleep(0.05)
        grace_deadline = time.time() + 2.0
        while time.time() < grace_deadline:
            not_in_mode = [drone for drone in self._drones if not drone.is_mode(mode)]
            if not not_in_mode:
                return None
            time.sleep(0.05)
        failures = [
            self._drone_failure_detail(drone, f"did not enter {mode}") for drone in not_in_mode
        ]
        return "Failed: " + "; ".join(failures)

    def _arm_all(self, timeout_s: float, force: bool = False) -> str | None:
        deadline = time.time() + max(0.0, timeout_s)
        next_arm_request_s = {drone.config.key: 0.0 for drone in self._drones}
        not_armed: list[PX4OffboardDrone] = []
        for drone in self._drones:
            requested = drone.request_arm(force=force)
            logger.info(
                "PX4 arm command requested",
                key=drone.config.key,
                requested=requested,
                force=force,
            )
            next_arm_request_s[drone.config.key] = time.time() + 1.5
        while time.time() < deadline:
            not_armed = [drone for drone in self._drones if not drone.update(timeout=0.05).armed]
            if not not_armed:
                return None
            now = time.time()
            for drone in not_armed:
                if now < next_arm_request_s[drone.config.key]:
                    continue
                accepted = drone.arm(force=force)
                logger.info(
                    "PX4 arm command sent",
                    key=drone.config.key,
                    accepted=accepted,
                    force=force,
                )
                drone.update(timeout=0.5)
                next_arm_request_s[drone.config.key] = time.time() + 2.0
            time.sleep(0.1)
        time.sleep(1.0)
        not_armed = [drone for drone in self._drones if not drone.update(timeout=0.05).armed]
        if not not_armed:
            return None
        if not_armed:
            failures = [
                self._drone_failure_detail(drone, "did not report armed") for drone in not_armed
            ]
            return "Failed: " + "; ".join(failures)
        return None

    def _wait_until_landed(self, timeout_s: float) -> bool:
        deadline = time.time() + max(0.0, timeout_s)
        grounded_since_s: float | None = None
        while time.time() < deadline:
            for drone in self._drones:
                drone.send_gcs_heartbeat(force=True)
            snapshots = [drone.update(timeout=0.02) for drone in self._drones]
            all_grounded = bool(snapshots) and all(
                snapshot.position_enu[2] <= AIRBORNE_ALTITUDE_M + 0.3 for snapshot in snapshots
            )
            if all_grounded:
                if grounded_since_s is None:
                    grounded_since_s = time.time()
                elif time.time() - grounded_since_s >= 3.0:
                    for drone, snapshot in zip(self._drones, snapshots, strict=True):
                        if snapshot.armed:
                            drone.disarm()
            else:
                grounded_since_s = None

            if snapshots and all(
                not snapshot.armed and snapshot.position_enu[2] <= AIRBORNE_ALTITUDE_M + 0.3
                for snapshot in snapshots
            ):
                return True
            time.sleep(0.2)
        return False

    def _reset_ground_modes_after_landing(self) -> None:
        for drone in self._drones:
            snapshot = drone.update(timeout=0.05)
            if snapshot.position_enu[2] > AIRBORNE_ALTITUDE_M + 0.3:
                continue
            if snapshot.armed:
                drone.disarm()
            drone.request_mode("LOITER")

    def _run_with_zero_velocity_stream(self, callback: Callable[[], str | None]) -> str | None:
        stop_event = Event()
        zeros = np.zeros((len(self._drones), 3), dtype=float)

        def stream() -> None:
            period = 1.0 / self.command_hz
            while not stop_event.is_set():
                try:
                    self._send_all_velocities(zeros)
                except Exception as exc:
                    logger.debug("PX4 setup velocity stream error", error=str(exc))
                time.sleep(period)

        thread = Thread(target=stream, daemon=True)
        thread.start()
        try:
            return callback()
        finally:
            stop_event.set()
            thread.join(timeout=1.0)

    def _prime_telemetry(self, seconds: float) -> None:
        if not self._drones:
            return
        zeros = np.zeros((len(self._drones), 3), dtype=float)
        period = 1.0 / self.command_hz
        end_time = time.time() + max(0.0, seconds)
        while time.time() < end_time:
            for drone in self._drones:
                drone.send_gcs_heartbeat(force=True)
                drone.update(timeout=0.002)
            self._send_all_velocities(zeros)
            time.sleep(period)

    def _configure_sitl_failsafes(self) -> str | None:
        failures: list[str] = []
        for drone in self._drones:
            if not drone.connected:
                failures.append(f"{drone.config.key}: disconnected")
                continue
            results = drone.configure_sitl_failsafes()
            failed = [name for name, ok in results.items() if not ok]
            if failed:
                failures.append(f"{drone.config.key}: {', '.join(failed)}")
        if failures:
            return (
                "Failed: SITL failsafe parameters were not confirmed; refusing swarm motion. "
                + "; ".join(failures)
            )
        return None

    def _wait_for_position_stream(self, timeout_s: float) -> str | None:
        zeros = np.zeros((len(self._drones), 3), dtype=float)
        deadline = time.time() + max(0.0, timeout_s)
        missing: list[PX4OffboardDrone] = []
        while time.time() < deadline:
            self._send_all_velocities(zeros)
            missing = []
            for drone in self._drones:
                snapshot = drone.update(timeout=0.005)
                if not snapshot.has_local_position:
                    missing.append(drone)
            if not missing:
                return None
            time.sleep(1.0 / self.command_hz)

        details = [
            self._drone_failure_detail(drone, "no LOCAL_POSITION_NED stream") for drone in missing
        ]
        return "Failed: " + "; ".join(details)

    def _wait_for_armable_health(self, timeout_s: float) -> str | None:
        deadline = time.time() + max(0.0, timeout_s)
        unhealthy: list[PX4OffboardDrone] = []
        while time.time() < deadline:
            unhealthy = []
            for drone in self._drones:
                drone.send_gcs_heartbeat(force=True)
                drone.update(timeout=0.02)
                if not drone.core_sensors_ready() or drone.recent_critical_status(max_age_s=5.0):
                    unhealthy.append(drone)
            if not unhealthy:
                return None
            time.sleep(0.2)

        details = []
        for drone in unhealthy:
            sensor_failures = drone.core_sensor_failures()
            reason = "core sensors not armable"
            if sensor_failures:
                reason += f": {', '.join(sensor_failures)}"
            critical_status = drone.recent_critical_status(max_age_s=10.0)
            if critical_status:
                reason += f"; recent PX4 status={critical_status}"
            details.append(self._drone_failure_detail(drone, reason))
        return "Failed: " + "; ".join(details)

    def _cleanup_control_setup_failure(self) -> None:
        self._stop_background_hold()
        self._velocity_control_ready = False
        if not self._drones:
            return
        zeros = np.zeros((len(self._drones), 3), dtype=float)
        self._command_all_for(zeros, seconds=0.2)
        snapshots = [drone.update(timeout=0.02) for drone in self._drones]
        any_airborne = any(
            snapshot.armed and snapshot.position_enu[2] > AIRBORNE_ALTITUDE_M + 0.3
            for snapshot in snapshots
        )
        if any_airborne:
            logger.warning(
                "PX4 control setup failed while airborne; leaving vehicles in current mode"
            )
            return
        for drone in self._drones:
            snapshot = drone.update(timeout=0.02)
            if snapshot.armed and snapshot.position_enu[2] <= 0.5:
                drone.disarm()
        time.sleep(0.5)
        for drone in self._drones:
            snapshot = drone.update(timeout=0.02)
            if snapshot.armed and snapshot.position_enu[2] <= 0.5:
                drone.disarm()
        self._command_all_for(zeros, seconds=0.2)

    def _recover_after_command_exception(self) -> None:
        """Prefer holding present positions after an in-flight command error."""
        self._stop_background_hold()
        if not self._drones:
            return
        snapshots = [drone.update(timeout=0.02) for drone in self._drones]
        any_airborne = any(
            snapshot.armed and snapshot.position_enu[2] > AIRBORNE_ALTITUDE_M + 0.3
            for snapshot in snapshots
        )
        if not any_airborne:
            self._cleanup_control_setup_failure()
            return

        hold_points = self._hold_current_positions_for(seconds=0.5)
        state_failure = self._velocity_control_state_failure(require_airborne=True)
        if state_failure is None:
            self._start_background_hold(hold_points)
        else:
            self._velocity_control_ready = False
            logger.warning(
                "PX4 swarm could not resume hold after command error", reason=state_failure
            )

    def _drone_failure_detail(self, drone: PX4OffboardDrone, reason: str) -> str:
        snapshot = drone.update(timeout=0.0)
        details = [f"{drone.config.key} {reason}"]
        if snapshot.last_command_ack:
            details.append(f"ack={snapshot.last_command_ack}")
        if snapshot.last_status_text:
            details.append(f"status={snapshot.last_status_text}")
        if not snapshot.has_local_position:
            details.append("local_position=False")
        return (
            " (".join([details[0], "; ".join(details[1:]) + ")"])
            if len(details) > 1
            else details[0]
        )

    def _command_all_for(self, velocities_enu: np.ndarray[Any, Any], seconds: float) -> None:
        period = 1.0 / self.command_hz
        end_time = time.time() + max(0.0, seconds)
        while time.time() < end_time:
            self._send_all_velocities(velocities_enu)
            time.sleep(period)

    def _hold_current_positions_for(self, seconds: float = 0.3) -> np.ndarray[Any, Any]:
        """Latch the current positions with zero feed-forward instead of flying to an old target."""
        hold_points = self._positions_enu()
        zeros = np.zeros((len(self._drones), 3), dtype=float)
        period = 1.0 / self.command_hz
        end_time = time.time() + max(0.0, seconds)
        while time.time() < end_time:
            hold_points = self._positions_enu()
            self._send_all_position_velocity_setpoints(hold_points, zeros)
            time.sleep(period)
        return hold_points

    def _send_all_velocities(self, velocities_enu: np.ndarray[Any, Any]) -> None:
        for drone, velocity in zip(self._drones, velocities_enu, strict=True):
            drone.send_velocity_enu(velocity)

    def _send_all_position_velocity_setpoints(
        self,
        positions_enu: np.ndarray[Any, Any],
        velocities_enu: np.ndarray[Any, Any],
    ) -> None:
        for drone, position, velocity in zip(
            self._drones, positions_enu, velocities_enu, strict=True
        ):
            drone.send_position_velocity_enu(position, velocity)

    @staticmethod
    def _selected_errors(
        positions_enu: np.ndarray[Any, Any],
        target_points_enu: np.ndarray[Any, Any],
        selected_indices: set[int],
    ) -> list[float]:
        if not selected_indices:
            return []
        return [
            float(np.linalg.norm(positions_enu[index] - target_points_enu[index]))
            for index in selected_indices
        ]

    @classmethod
    def _selected_max_error(
        cls,
        positions_enu: np.ndarray[Any, Any],
        target_points_enu: np.ndarray[Any, Any],
        selected_indices: set[int],
    ) -> float:
        errors = cls._selected_errors(positions_enu, target_points_enu, selected_indices)
        return max(errors) if errors else 0.0

    @classmethod
    def _selected_mean_error(
        cls,
        positions_enu: np.ndarray[Any, Any],
        target_points_enu: np.ndarray[Any, Any],
        selected_indices: set[int],
    ) -> float:
        errors = cls._selected_errors(positions_enu, target_points_enu, selected_indices)
        return float(np.mean(errors)) if errors else 0.0

    def _selected_max_speed(self, selected_indices: set[int]) -> float:
        if not selected_indices:
            return 0.0
        velocities = self._velocities_enu()
        speeds = [float(np.linalg.norm(velocities[index])) for index in selected_indices]
        return max(speeds) if speeds else 0.0

    def _motion_velocity_commands(
        self,
        positions_enu: np.ndarray[Any, Any],
        _desired_points_enu: np.ndarray[Any, Any],
        active_mask: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        commands = self._behavior.separation_velocity_commands(
            positions_enu,
            active_mask,
        )
        commands[:, 2] = 0.0
        return commands

    def _send_tracking_setpoints(
        self,
        _positions_enu: np.ndarray[Any, Any],
        desired_points_enu: np.ndarray[Any, Any],
        velocities_enu: np.ndarray[Any, Any],
    ) -> None:
        self._send_all_position_velocity_setpoints(desired_points_enu, velocities_enu)

    def _hold_active(self) -> bool:
        return self._hold_thread is not None and self._hold_thread.is_alive()

    def _stop_background_hold(self) -> None:
        stop_event = self._hold_stop_event
        thread = self._hold_thread
        self._hold_stop_event = None
        self._hold_thread = None
        self._hold_targets = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive() and thread is not current_thread():
            thread.join(timeout=2.0)

    def _start_background_hold(self, target_points_enu: np.ndarray[Any, Any]) -> None:
        self._stop_background_hold()
        targets = np.asarray(target_points_enu, dtype=float).copy()
        stop_event = Event()
        self._hold_stop_event = stop_event
        self._hold_targets = targets
        self._hold_thread = Thread(
            target=self._background_hold_loop,
            args=(stop_event, targets),
            daemon=True,
        )
        self._hold_thread.start()
        logger.info("PX4 swarm background hold started")

    def _background_hold_loop(
        self,
        stop_event: Event,
        target_points_enu: np.ndarray[Any, Any],
    ) -> None:
        period = 1.0 / self.command_hz
        active = np.ones(len(self._drones), dtype=bool)
        while self._running and not stop_event.is_set():
            loop_start = time.time()
            state_failure = self._velocity_control_state_failure(require_airborne=True)
            if state_failure is not None:
                self._velocity_control_ready = False
                logger.warning("PX4 swarm background hold stopped", reason=state_failure)
                break
            positions = self._positions_enu()
            commands = self._motion_velocity_commands(positions, target_points_enu, active)
            self._send_tracking_setpoints(positions, target_points_enu, commands)
            self._publish_fleet_status()
            time.sleep(max(0.0, period - (time.time() - loop_start)))

    def _start_hold_after_motion(
        self,
        target_points_enu: np.ndarray[Any, Any] | None = None,
        *,
        reached_target: bool = False,
    ) -> None:
        state_failure = self._velocity_control_state_failure(require_airborne=True)
        if state_failure is not None:
            self._velocity_control_ready = False
            logger.warning("PX4 swarm hold not started", reason=state_failure)
            return
        hold_points = (
            np.asarray(target_points_enu, dtype=float).copy()
            if reached_target and target_points_enu is not None
            else self._positions_enu()
        )
        if not reached_target:
            logger.info("PX4 swarm holding current positions after incomplete command")
        self._start_background_hold(hold_points)

    def _run_to_points(
        self,
        target_points_enu: np.ndarray[Any, Any],
        selected_indices: set[int] | None = None,
        radius_m: float = 2.0,
        dwell_s: float = 1.0,
        timeout_s: float = 60.0,
        require_airborne: bool = True,
    ) -> tuple[bool, float, float]:
        command_start_time = time.time()
        failure = self._ensure_velocity_control(require_airborne=require_airborne)
        if failure is not None:
            raise RuntimeError(failure)
        self._hold_current_positions_for(seconds=MOTION_PRE_COMMAND_HOLD_S)

        selected = (
            selected_indices if selected_indices is not None else set(range(len(self._drones)))
        )
        hold_points = self._positions_enu()
        inside_since_s: float | None = None
        stationary_near_since_s: float | None = None
        period = 1.0 / self.command_hz
        active = np.ones(len(self._drones), dtype=bool)
        final_error = float("inf")
        next_progress_log_s = time.time() + MOTION_PROGRESS_LOG_INTERVAL_S
        motion_start_time = time.time()

        while time.time() - motion_start_time < timeout_s:
            loop_start = time.time()
            now = time.time()
            state_failure = self._velocity_control_state_failure(require_airborne=require_airborne)
            if state_failure is not None:
                self._velocity_control_ready = False
                raise RuntimeError(state_failure)
            positions = self._positions_enu()
            desired = hold_points.copy()
            for index in selected:
                desired[index] = target_points_enu[index]

            commands = self._motion_velocity_commands(positions, desired, active)
            self._send_tracking_setpoints(positions, desired, commands)
            self._publish_fleet_status()

            max_error = self._selected_max_error(positions, target_points_enu, selected)
            final_error = self._selected_mean_error(positions, target_points_enu, selected)
            max_speed = self._selected_max_speed(selected)
            if now >= next_progress_log_s:
                logger.info(
                    "PX4 swarm motion progress",
                    selected=[self._drones[index].config.key for index in sorted(selected)],
                    mean_error=final_error,
                    max_error=max_error,
                    max_speed=max_speed,
                    positions=np.round(positions, 2).tolist(),
                    targets=np.round(target_points_enu, 2).tolist(),
                    separation_ff=np.round(commands, 2).tolist(),
                )
                next_progress_log_s = now + MOTION_PROGRESS_LOG_INTERVAL_S

            if max_error <= radius_m:
                if inside_since_s is None:
                    inside_since_s = now
                if now - inside_since_s >= dwell_s:
                    self._hold_current_positions_for(seconds=0.3)
                    return True, time.time() - command_start_time, final_error
            else:
                inside_since_s = None

            settled_radius = radius_m + min(0.75, radius_m * 0.25)
            if max_error <= settled_radius and max_speed <= MOTION_STATIONARY_SPEED_MPS:
                if stationary_near_since_s is None:
                    stationary_near_since_s = now
                if now - stationary_near_since_s >= MOTION_STATIONARY_DWELL_S:
                    self._hold_current_positions_for(seconds=0.3)
                    return True, time.time() - command_start_time, final_error
            else:
                stationary_near_since_s = None

            time.sleep(max(0.0, period - (time.time() - loop_start)))

        self._hold_current_positions_for(seconds=MOTION_TIMEOUT_HOLD_S)
        return False, time.time() - command_start_time, final_error

    def _run_waypoint_sequences(
        self,
        waypoint_sequences: list[list[list[float]]],
        radius_m: float,
        timeout_s: float,
    ) -> tuple[bool, float, dict[str, int]]:
        command_start_time = time.time()
        failure = self._ensure_velocity_control()
        if failure is not None:
            raise RuntimeError(failure)
        self._hold_current_positions_for(seconds=MOTION_PRE_COMMAND_HOLD_S)

        sequence_indices = [0 for _ in self._drones]
        hold_points = self._positions_enu()
        active = np.ones(len(self._drones), dtype=bool)
        period = 1.0 / self.command_hz
        motion_start_time = time.time()

        while time.time() - motion_start_time < timeout_s:
            loop_start = time.time()
            state_failure = self._velocity_control_state_failure(require_airborne=True)
            if state_failure is not None:
                self._velocity_control_ready = False
                raise RuntimeError(state_failure)
            positions = self._positions_enu()
            desired = hold_points.copy()
            complete_count = 0

            for drone_index, sequence in enumerate(waypoint_sequences):
                if not sequence:
                    complete_count += 1
                    continue
                target_index = min(sequence_indices[drone_index], len(sequence) - 1)
                desired[drone_index] = np.asarray(sequence[target_index], dtype=float)
                error = float(np.linalg.norm(positions[drone_index] - desired[drone_index]))
                if error <= radius_m:
                    if sequence_indices[drone_index] < len(sequence) - 1:
                        sequence_indices[drone_index] += 1
                    else:
                        complete_count += 1

            commands = self._motion_velocity_commands(positions, desired, active)
            self._send_tracking_setpoints(positions, desired, commands)
            self._publish_fleet_status()

            if complete_count == len(self._drones):
                self._hold_current_positions_for(seconds=0.3)
                return (
                    True,
                    time.time() - command_start_time,
                    self._sequence_progress(sequence_indices),
                )

            time.sleep(max(0.0, period - (time.time() - loop_start)))

        self._hold_current_positions_for(seconds=MOTION_TIMEOUT_HOLD_S)
        return False, time.time() - command_start_time, self._sequence_progress(sequence_indices)

    def _run_to_shared_center(
        self,
        center_enu: np.ndarray[Any, Any],
        formation_radius_m: float,
        radius_m: float = 1.5,
        dwell_s: float = 1.0,
        timeout_s: float = 90.0,
    ) -> tuple[bool, float, float, np.ndarray[Any, Any]]:
        command_start_time = time.time()
        failure = self._ensure_velocity_control()
        if failure is not None:
            raise RuntimeError(failure)
        self._hold_current_positions_for(seconds=MOTION_PRE_COMMAND_HOLD_S)

        selected = set(range(len(self._drones)))
        desired = self._selected_target_points(selected, center_enu, formation_radius_m)
        active = np.ones(len(self._drones), dtype=bool)
        period = 1.0 / self.command_hz
        inside_since_s: float | None = None
        stationary_near_since_s: float | None = None
        final_error = float("inf")
        complete_radius = formation_radius_m + radius_m
        next_progress_log_s = time.time() + MOTION_PROGRESS_LOG_INTERVAL_S
        motion_start_time = time.time()

        while time.time() - motion_start_time < timeout_s:
            loop_start = time.time()
            now = time.time()
            state_failure = self._velocity_control_state_failure(require_airborne=True)
            if state_failure is not None:
                self._velocity_control_ready = False
                raise RuntimeError(state_failure)

            positions = self._positions_enu()
            commands = self._motion_velocity_commands(positions, desired, active)
            self._send_tracking_setpoints(positions, desired, commands)
            self._publish_fleet_status()

            center_errors = np.linalg.norm(positions - center_enu[np.newaxis, :], axis=1)
            max_center_error = float(np.max(center_errors)) if len(center_errors) else float("inf")
            max_target_error = self._selected_max_error(positions, desired, selected)
            final_error = self._selected_mean_error(positions, desired, selected)
            max_speed = self._selected_max_speed(selected)
            if now >= next_progress_log_s:
                logger.info(
                    "PX4 swarm shared-point motion progress",
                    mean_error=final_error,
                    max_error=max_target_error,
                    center_error=max_center_error,
                    max_speed=max_speed,
                    positions=np.round(positions, 2).tolist(),
                    desired=np.round(desired, 2).tolist(),
                    separation_ff=np.round(commands, 2).tolist(),
                    center=np.round(center_enu, 2).tolist(),
                )
                next_progress_log_s = now + MOTION_PROGRESS_LOG_INTERVAL_S

            if max_target_error <= radius_m and max_center_error <= complete_radius:
                if inside_since_s is None:
                    inside_since_s = now
                if now - inside_since_s >= dwell_s:
                    self._hold_current_positions_for(seconds=0.3)
                    return True, time.time() - command_start_time, final_error, desired
            else:
                inside_since_s = None

            settled_radius = radius_m + min(0.75, radius_m * 0.25)
            if (
                max_target_error <= settled_radius
                and max_center_error <= complete_radius
                and max_speed <= MOTION_STATIONARY_SPEED_MPS
            ):
                if stationary_near_since_s is None:
                    stationary_near_since_s = now
                if now - stationary_near_since_s >= MOTION_STATIONARY_DWELL_S:
                    self._hold_current_positions_for(seconds=0.3)
                    return True, time.time() - command_start_time, final_error, desired
            else:
                stationary_near_since_s = None

            time.sleep(max(0.0, period - (time.time() - loop_start)))

        self._hold_current_positions_for(seconds=MOTION_TIMEOUT_HOLD_S)
        return False, time.time() - command_start_time, final_error, desired

    def _sequence_progress(self, sequence_indices: list[int]) -> dict[str, int]:
        return {
            drone.config.key: int(index)
            for drone, index in zip(self._drones, sequence_indices, strict=True)
        }

    def _selected_target_points(
        self,
        selected_indices: set[int],
        center: np.ndarray[Any, Any],
        formation_radius_m: float,
    ) -> np.ndarray[Any, Any]:
        points = self._positions_enu()
        selected_sorted = sorted(selected_indices)
        if len(selected_sorted) == 1:
            points[selected_sorted[0]] = center
            return points

        candidate_points = []
        for offset_index in range(len(selected_sorted)):
            angle = (2.0 * math.pi * offset_index) / len(selected_sorted)
            candidate_points.append(
                center
                + np.array(
                    [
                        formation_radius_m * math.cos(angle),
                        formation_radius_m * math.sin(angle),
                        0.0,
                    ],
                    dtype=float,
                )
            )
        candidates = np.asarray(candidate_points, dtype=float)
        current = points[selected_sorted]

        if len(selected_sorted) <= 8:
            best_assignment = min(
                permutations(range(len(selected_sorted))),
                key=lambda assignment: sum(
                    float(np.linalg.norm(current[row] - candidates[column]))
                    for row, column in enumerate(assignment)
                ),
            )
        else:
            remaining = set(range(len(selected_sorted)))
            assignment = []
            for row in range(len(selected_sorted)):
                column = min(
                    remaining,
                    key=lambda item: float(np.linalg.norm(current[row] - candidates[item])),
                )
                assignment.append(column)
                remaining.remove(column)
            best_assignment = tuple(assignment)

        for row, drone_index in enumerate(selected_sorted):
            points[drone_index] = candidates[best_assignment[row]]
        return points

    def _closest_drone_indices(self, center: np.ndarray[Any, Any], count: int) -> set[int]:
        if count <= 0 or count >= len(self._drones):
            return set(range(len(self._drones)))
        positions = self._positions_enu()
        distances = np.linalg.norm(positions - center[np.newaxis, :], axis=1)
        selected = np.argsort(distances)[: max(1, min(count, len(self._drones)))]
        return {int(index) for index in selected}

    def _line_points(
        self, center_x: float, center_y: float, altitude: float, spacing_m: float
    ) -> np.ndarray[Any, Any]:
        points = np.zeros((len(self._drones), 3), dtype=float)
        y_values = (np.arange(len(self._drones)) - ((len(self._drones) - 1) / 2.0)) * spacing_m
        points[:, 0] = center_x
        points[:, 1] = center_y + y_values
        points[:, 2] = altitude
        return points

    def _point_from_json_item(self, item: Any, altitude: float) -> np.ndarray[Any, Any]:
        if isinstance(item, dict):
            x = _to_float(item.get("x"))
            y = _to_float(item.get("y"))
            z = _to_float(item.get("z"), altitude)
        elif isinstance(item, list | tuple) and len(item) >= 2:
            x = _to_float(item[0])
            y = _to_float(item[1])
            z = _to_float(item[2], altitude) if len(item) >= 3 else altitude
        else:
            raise ValueError("Each point must be [x, y, z] or an object with x/y/z")
        return np.array([x, y, z], dtype=float)

    def _explicit_points_from_json(self, points_json: str, altitude: float) -> np.ndarray[Any, Any]:
        try:
            parsed = json.loads(points_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid points_json: {exc}") from exc
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("points_json must be a non-empty JSON list")
        return np.asarray(
            [self._point_from_json_item(item, altitude) for item in parsed],
            dtype=float,
        )

    def _targets_for_explicit_points(
        self,
        points_json: str,
        altitude: float,
    ) -> tuple[np.ndarray[Any, Any], set[int], int]:
        explicit_points = self._explicit_points_from_json(points_json, altitude)
        points = self._positions_enu()
        if not self._drones:
            return points, set(), len(explicit_points)

        n_targets = min(len(explicit_points), len(self._drones))
        if len(explicit_points) >= len(self._drones):
            selected = set(range(len(self._drones)))
            points[: len(self._drones)] = explicit_points[: len(self._drones)]
            return points, selected, len(explicit_points)

        current = points.copy()
        if len(self._drones) <= 8:
            best_assignment = min(
                permutations(range(len(self._drones)), n_targets),
                key=lambda assignment: sum(
                    float(np.linalg.norm(current[drone_index] - explicit_points[row]))
                    for row, drone_index in enumerate(assignment)
                ),
            )
        else:
            remaining = set(range(len(self._drones)))
            assignment = []
            for row in range(n_targets):
                drone_index = min(
                    remaining,
                    key=lambda index: float(np.linalg.norm(current[index] - explicit_points[row])),
                )
                assignment.append(drone_index)
                remaining.remove(drone_index)
            best_assignment = tuple(assignment)

        selected = {int(index) for index in best_assignment}
        for row, drone_index in enumerate(best_assignment):
            points[drone_index] = explicit_points[row]
        return points, selected, len(explicit_points)

    def _parse_points_json(self, points_json: str, altitude: float) -> np.ndarray[Any, Any]:
        targets, _, _ = self._targets_for_explicit_points(points_json, altitude)
        return targets

    def _run_skill_task(self, callback: Callable[[], str]) -> str:
        with self._command_lock:
            self._stop_background_hold()
            try:
                return callback()
            except Exception as exc:
                logger.exception("PX4 swarm command failed")
                self._recover_after_command_exception()
                message = str(exc)
                return message if message.startswith("Failed:") else f"Failed: {message}"

    @skill
    def list_drones(self) -> str:
        """List configured PX4 drones, positions, modes, batteries, and spacing."""
        return self.get_fleet_state()

    @skill
    def get_fleet_state(self) -> str:
        """Return current fleet state and pairwise distances."""
        state = self._fleet_state_dict()
        lines = []
        for drone in state["drones"]:
            position = drone["position_enu"]
            lines.append(
                f"{drone['key']}: connected={drone['connected']}, armed={drone['armed']}, "
                f"mode={drone['mode']}, pos=({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}), "
                f"battery={drone['battery_remaining']}, local_pos={drone['has_local_position']}"
            )
            notes = []
            if drone.get("last_command_ack"):
                notes.append(f"ack={drone['last_command_ack']}")
            if drone.get("last_status_text"):
                notes.append(f"status={drone['last_status_text']}")
            if notes:
                lines.append("  " + "; ".join(notes))
        distances = state["pairwise_distances_m"]
        if distances:
            lines.append(
                "pairwise: "
                + ", ".join(f"{pair}={distance:.1f}m" for pair, distance in distances.items())
            )
        lines.append(
            f"control: velocity_ready={state['velocity_control_ready']}, "
            f"hold_active={state['hold_active']}"
        )
        return "\n".join(lines) if lines else "No PX4 drones configured"

    @skill
    def takeoff_swarm(self, altitude: float = DEFAULT_TAKEOFF_ALTITUDE_M) -> str:
        """Arm all drones and take off using swarm velocity commands.

        Args:
            altitude: Target altitude in meters above launch.
        """

        def run() -> str:
            targets = self._positions_enu()
            target_altitude = max(altitude, 1.0)
            targets[:, 2] = target_altitude
            airborne_completion_radius = max(0.8, target_altitude - 1.2)
            complete, duration_s, final_error = self._run_to_points(
                targets,
                radius_m=airborne_completion_radius,
                dwell_s=0.5,
                timeout_s=max(15.0, altitude * 3.0),
                require_airborne=True,
            )
            self._home_positions = self._positions_enu()
            self._start_hold_after_motion(targets, reached_target=complete)
            status = "airborne" if complete else "timed out"
            hold = ", hold_active=True" if self._hold_active() else ", hold_active=False"
            return (
                f"takeoff_swarm {status}: duration={duration_s:.1f}s, "
                f"target_altitude={target_altitude:.1f}m, avg_error={final_error:.2f}m{hold}"
            )

        return self._run_skill_task(run)

    @skill
    def hover_swarm(self, duration: float = 2.0) -> str:
        """Hold all drones at their current positions.

        Args:
            duration: Seconds to confirm stable hover. Values <= 0 keep holding until another swarm command.
        """

        def run() -> str:
            if duration <= 0.0:
                airborne_failure = self._airborne_state_failure()
                if airborne_failure is not None:
                    raise RuntimeError(airborne_failure)
                failure = self._ensure_velocity_control()
                if failure is not None:
                    raise RuntimeError(failure)
                targets = self._positions_enu()
                self._start_background_hold(targets)
                return "hover_swarm persistent: hold_active=True"

            targets = self._positions_enu()
            complete, duration_s, final_error = self._run_to_points(
                targets,
                radius_m=0.8,
                dwell_s=max(duration, 0.5),
                timeout_s=max(duration + 5.0, 5.0),
            )
            self._start_hold_after_motion(targets, reached_target=complete)
            status = "complete" if complete else "timed out"
            hold = ", hold_active=True" if self._hold_active() else ", hold_active=False"
            return f"hover_swarm {status}: duration={duration_s:.1f}s, final_error={final_error:.2f}m{hold}"

        return self._run_skill_task(run)

    @skill
    def land_swarm(self) -> str:
        """Land all drones at their current positions."""

        def run() -> str:
            self._command_all_for(np.zeros((len(self._drones), 3)), seconds=0.5)
            max_altitude = float(np.max(self._positions_enu()[:, 2])) if self._drones else 0.0
            results = [f"{drone.config.key}={drone.land()}" for drone in self._drones]
            self._velocity_control_ready = False
            landed = self._wait_until_landed(timeout_s=max(75.0, max_altitude * 4.0 + 20.0))
            if landed:
                self._reset_ground_modes_after_landing()
            status = "complete" if landed else "landing"
            return f"land_swarm {status}: " + ", ".join(results)

        return self._run_skill_task(run)

    @skill
    def return_to_launch(self, altitude: float = DEFAULT_TASK_ALTITUDE_M, land: bool = True) -> str:
        """Return above each launch position with swarm spacing, then optionally land.

        Args:
            altitude: Return altitude in meters.
            land: Land after reaching the launch line.
        """

        def run() -> str:
            targets = np.asarray(self._home_positions, dtype=float).copy()
            targets[:, 2] = altitude
            complete, duration_s, final_error = self._run_to_points(
                targets,
                radius_m=2.0,
                dwell_s=1.0,
                timeout_s=90.0,
            )
            if not complete:
                self._start_hold_after_motion(targets, reached_target=False)
            elif land:
                max_altitude = float(np.max(self._positions_enu()[:, 2])) if self._drones else 0.0
                for drone in self._drones:
                    drone.land()
                self._velocity_control_ready = False
                if self._wait_until_landed(timeout_s=max(75.0, max_altitude * 4.0 + 20.0)):
                    self._reset_ground_modes_after_landing()
            elif not land:
                self._start_hold_after_motion(targets, reached_target=complete)
            status = "complete" if complete else "timed out"
            landing = " and land sent" if land and complete else ""
            if land and not complete:
                landing = "; land not sent because return point was not reached"
            return (
                f"return_to_launch {status}{landing}: duration={duration_s:.1f}s, "
                f"final_error={final_error:.2f}m"
            )

        return self._run_skill_task(run)

    @skill
    def px4_native_rtl(self) -> str:
        """Send PX4 native RTL mode to all drones without swarm velocity allocation."""

        def run() -> str:
            results = [f"{drone.config.key}={drone.rtl()}" for drone in self._drones]
            self._velocity_control_ready = False
            return "px4_native_rtl sent: " + ", ".join(results)

        return self._run_skill_task(run)

    @skill
    def return_to_line_formation(
        self,
        center_x: float = 0.0,
        center_y: float = 0.0,
        altitude: float = DEFAULT_TASK_ALTITUDE_M,
        spacing: float = DEFAULT_LINE_SPACING_M,
    ) -> str:
        """Move all drones to a line formation using swarm velocity commands.

        Args:
            center_x: ENU x/north coordinate of the line center in meters.
            center_y: ENU y/east coordinate of the line center in meters.
            altitude: Formation altitude in meters.
            spacing: Adjacent-drone spacing in meters.
        """

        def run() -> str:
            targets = self._line_points(center_x, center_y, altitude, spacing)
            complete, duration_s, final_error = self._run_to_points(
                targets,
                radius_m=1.5,
                dwell_s=1.0,
                timeout_s=75.0,
            )
            self._start_hold_after_motion(targets, reached_target=complete)
            status = "complete" if complete else "timed out"
            return (
                f"return_to_line_formation {status}: duration={duration_s:.1f}s, "
                f"final_error={final_error:.2f}m"
            )

        return self._run_skill_task(run)

    @skill
    def go_to_points(
        self,
        points_json: str,
        altitude: float = DEFAULT_TASK_ALTITUDE_M,
        formation_radius: float = 0.0,
    ) -> str:
        """Send drones to explicit ENU points through the swarm velocity layer.

        Args:
            points_json: JSON list of explicit ENU points. Fewer points than drones tasks the closest drones.
            altitude: Default z value when a point omits altitude.
            formation_radius: Deprecated. Use go_to_shared_point for shared-point formations.
        """

        def run() -> str:
            targets, selected, n_requested = self._targets_for_explicit_points(
                points_json,
                altitude,
            )
            if not selected:
                return "Failed: no PX4 drones are configured"
            complete, duration_s, final_error = self._run_to_points(
                targets,
                selected_indices=selected,
                radius_m=1.5,
                dwell_s=1.0,
                timeout_s=90.0,
            )
            self._start_hold_after_motion(targets, reached_target=complete)
            selected_names = [self._drones[index].config.key for index in sorted(selected)]
            status = "complete" if complete else "timed out"
            return (
                f"go_to_points {status}: drones={selected_names}, requested_points={n_requested}, "
                f"duration={duration_s:.1f}s, final_error={final_error:.2f}m"
            )

        return self._run_skill_task(run)

    @skill
    def go_to_shared_point(
        self,
        x: float,
        y: float,
        z: float = DEFAULT_TASK_ALTITUDE_M,
        formation_radius: float = 4.5,
    ) -> str:
        """Send all drones to a formation around one ENU point through the swarm velocity layer.

        Args:
            x: ENU x/north coordinate in meters.
            y: ENU y/east coordinate in meters.
            z: ENU altitude in meters.
            formation_radius: Radius around the shared point.
        """

        def run() -> str:
            center = np.array([x, y, z], dtype=float)
            complete, duration_s, final_error, targets = self._run_to_shared_center(
                center,
                formation_radius_m=formation_radius,
                radius_m=1.5,
                dwell_s=1.0,
                timeout_s=90.0,
            )
            self._start_hold_after_motion(targets, reached_target=complete)
            status = "complete" if complete else "timed out"
            return (
                f"go_to_shared_point {status}: duration={duration_s:.1f}s, "
                f"final_error={final_error:.2f}m"
            )

        return self._run_skill_task(run)

    @skill
    def move_swarm_relative(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        spacing: float = 0.0,
        timeout: float = 0.0,
    ) -> str:
        """Move the whole swarm by a relative ENU offset while preserving the current formation.

        Args:
            dx: Relative x/north movement in meters.
            dy: Relative y/east movement in meters.
            dz: Relative z/up movement in meters.
            spacing: Optional adjacent-drone line spacing in meters. Values <= 0 preserve current formation.
            timeout: Optional timeout in seconds. Values <= 0 choose one from distance.
        """

        def run() -> str:
            offset = np.array([dx, dy, dz], dtype=float)
            targets = self._positions_enu() + offset[np.newaxis, :]
            if spacing > 0.0 and len(targets) > 1:
                center = np.mean(targets, axis=0)
                y_values = (np.arange(len(targets)) - ((len(targets) - 1) / 2.0)) * max(
                    spacing,
                    self.min_separation_m,
                )
                targets[:, 0] = center[0]
                targets[:, 1] = center[1] + y_values
                targets[:, 2] = center[2]
            distance = float(np.linalg.norm(offset))
            speed = max(float(self._behavior.weights.v_max), 0.5)
            timeout_s = timeout if timeout > 0.0 else max(45.0, (distance / speed) * 2.5 + 20.0)
            complete, duration_s, final_error = self._run_to_points(
                targets,
                radius_m=1.5,
                dwell_s=1.0,
                timeout_s=timeout_s,
            )
            self._start_hold_after_motion(targets, reached_target=complete)
            status = "complete" if complete else "timed out"
            return (
                f"move_swarm_relative {status}: duration={duration_s:.1f}s, "
                f"final_error={final_error:.2f}m"
            )

        return self._run_skill_task(run)

    @skill
    def investigate_relative(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        units: int = 0,
        formation_radius: float = 2.0,
    ) -> str:
        """Task the closest units to investigate a point relative to the current swarm center.

        Args:
            dx: Relative x/north offset in meters from the current swarm center.
            dy: Relative y/east offset in meters from the current swarm center.
            dz: Relative z/up offset in meters from the current swarm center.
            units: Number of drones to task. Values <= 0 task every drone.
            formation_radius: Radius around the relative investigation point for multiple units.
        """

        def run() -> str:
            positions = self._positions_enu()
            if len(positions) == 0:
                return "Failed: no PX4 drones are configured"
            center = np.mean(positions, axis=0) + np.array([dx, dy, dz], dtype=float)
            selected = self._closest_drone_indices(center, units)
            targets = self._selected_target_points(selected, center, formation_radius)
            distance = float(np.linalg.norm(np.array([dx, dy, dz], dtype=float)))
            speed = max(float(self._behavior.weights.v_max), 0.5)
            timeout_s = max(45.0, (distance / speed) * 2.5 + 20.0)
            complete, duration_s, final_error = self._run_to_points(
                targets,
                selected_indices=selected,
                radius_m=2.0,
                dwell_s=1.0,
                timeout_s=timeout_s,
            )
            self._start_hold_after_motion(targets, reached_target=complete)
            selected_names = [self._drones[index].config.key for index in sorted(selected)]
            status = "complete" if complete else "timed out"
            return (
                f"investigate_relative {status}: drones={selected_names}, "
                f"center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), "
                f"duration={duration_s:.1f}s, avg_error={final_error:.2f}m"
            )

        return self._run_skill_task(run)

    @skill
    def task_two_units_to_investigate_relative(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
    ) -> str:
        """Task the two closest drones to investigate a relative ENU offset from the swarm center.

        Args:
            dx: Relative x/north offset in meters.
            dy: Relative y/east offset in meters.
            dz: Relative z/up offset in meters.
        """
        return self.investigate_relative(dx=dx, dy=dy, dz=dz, units=2)

    @skill
    def investigate_coordinate(
        self,
        x: float,
        y: float,
        z: float = DEFAULT_TASK_ALTITUDE_M,
        units: int = 0,
        formation_radius: float = 2.0,
    ) -> str:
        """Task the closest units to investigate an ENU coordinate.

        Args:
            x: ENU x/north coordinate in meters.
            y: ENU y/east coordinate in meters.
            z: Target altitude in meters.
            units: Number of drones to task. Values <= 0 task every drone.
            formation_radius: Radius around the coordinate for multiple units.
        """

        def run() -> str:
            center = np.array([x, y, z], dtype=float)
            selected = self._closest_drone_indices(center, units)
            targets = self._selected_target_points(selected, center, formation_radius)
            complete, duration_s, final_error = self._run_to_points(
                targets,
                selected_indices=selected,
                radius_m=2.0,
                dwell_s=1.0,
                timeout_s=90.0,
            )
            self._start_hold_after_motion(targets, reached_target=complete)
            selected_names = [self._drones[index].config.key for index in sorted(selected)]
            status = "complete" if complete else "timed out"
            return (
                f"investigate_coordinate {status}: drones={selected_names}, "
                f"duration={duration_s:.1f}s, avg_error={final_error:.2f}m"
            )

        return self._run_skill_task(run)

    @skill
    def task_two_units_to_investigate_coordinate(
        self,
        x: float,
        y: float,
        z: float = DEFAULT_TASK_ALTITUDE_M,
    ) -> str:
        """Task the two closest drones to investigate an ENU coordinate.

        Args:
            x: ENU x/north coordinate in meters.
            y: ENU y/east coordinate in meters.
            z: Target altitude in meters.
        """
        return self.investigate_coordinate(x=x, y=y, z=z, units=2)

    @skill
    def sweep_grid(
        self,
        origin_x: float = 10.0,
        origin_y: float = -12.0,
        width: float = 30.0,
        height: float = 24.0,
        altitude: float = DEFAULT_TASK_ALTITUDE_M,
        lane_spacing: float = 6.0,
        timeout: float = 180.0,
    ) -> str:
        """Sweep a rectangular grid with non-overlapping lanes assigned across the swarm.

        Args:
            origin_x: ENU x/north coordinate for the southwest sweep corner.
            origin_y: ENU y/east coordinate for the southwest sweep corner.
            width: Sweep width in x/north meters.
            height: Sweep height in y/east meters.
            altitude: Sweep altitude in meters.
            lane_spacing: Distance between adjacent lanes in meters.
            timeout: Maximum sweep time in seconds.
        """

        def run() -> str:
            if width <= 0.0 or height <= 0.0:
                return "Failed: width and height must be positive"
            lane_count = max(1, math.floor(height / max(lane_spacing, 0.5)) + 1)
            lanes = [origin_y + min(index * lane_spacing, height) for index in range(lane_count)]
            if lanes[-1] < origin_y + height:
                lanes.append(origin_y + height)

            sequences: list[list[list[float]]] = [[] for _ in self._drones]
            for lane_index, lane_y in enumerate(lanes):
                drone_index = lane_index % max(len(self._drones), 1)
                if (lane_index // max(len(self._drones), 1)) % 2 == 0:
                    start_x = origin_x
                    end_x = origin_x + width
                else:
                    start_x = origin_x + width
                    end_x = origin_x
                sequences[drone_index].append([start_x, lane_y, altitude])
                sequences[drone_index].append([end_x, lane_y, altitude])

            complete, duration_s, progress = self._run_waypoint_sequences(
                sequences,
                radius_m=2.0,
                timeout_s=timeout,
            )
            self._start_hold_after_motion()
            status = "complete" if complete else "timed out"
            return f"sweep_grid {status}: duration={duration_s:.1f}s, progress={progress}"

        return self._run_skill_task(run)

    @skill
    def count_units_within_radius_of_drone(self, drone: str, radius: float = 100.0) -> str:
        """Count other drones within a radius of one drone.

        Args:
            drone: Reference drone such as x500_0.
            radius: Radius in meters.
        """
        try:
            index = self._resolve_drone_index(drone)
        except ValueError as exc:
            return f"Failed: {exc}"
        positions = self._positions_enu()
        if index >= len(positions):
            return f"Failed: {drone} has no position"
        distances = np.linalg.norm(positions - positions[index], axis=1)
        nearby = [
            self._drones[i].config.key
            for i, distance in enumerate(distances)
            if i != index and float(distance) <= radius
        ]
        return f"{len(nearby)} unit(s) within {radius:.1f}m of {self._drones[index].config.key}: {nearby}"


px4_swarm_module = PX4SwarmModule.blueprint

__all__ = ["PX4SwarmModule", "px4_swarm_module"]
