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

"""PX4 swarm GPS map-navigation skills."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Protocol

from dimos_lcm.std_msgs import String

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In
from dimos.mapping.types import LatLon
from dimos.spec.utils import Spec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

DEFAULT_MAP_TARGET_ALTITUDE_M = 8.0
DEFAULT_MAX_MAP_TARGET_DISTANCE_M = 2000.0


class PX4SwarmMapNavigationTarget(Spec, Protocol):
    def go_to_shared_point(
        self,
        x: float,
        y: float,
        z: float = DEFAULT_MAP_TARGET_ALTITUDE_M,
        formation_radius: float = 4.5,
    ) -> str: ...

    def investigate_coordinate(
        self,
        x: float,
        y: float,
        z: float = DEFAULT_MAP_TARGET_ALTITUDE_M,
        units: int = 0,
        formation_radius: float = 2.0,
    ) -> str: ...


@dataclass(frozen=True)
class SwarmGeoReference:
    gps: LatLon
    enu: tuple[float, float, float]


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _meters_per_degree(latitude_deg: float) -> tuple[float, float]:
    lat_rad = math.radians(latitude_deg)
    meters_per_degree_lat = (
        111132.92 - 559.82 * math.cos(2.0 * lat_rad) + 1.175 * math.cos(4.0 * lat_rad)
    )
    meters_per_degree_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3.0 * lat_rad)
    return meters_per_degree_lat, meters_per_degree_lon


class PX4SwarmMapNavigationSkill(Module):
    """Bridge global GPS coordinates into the PX4 swarm local ENU command surface."""

    _px4_swarm: PX4SwarmMapNavigationTarget

    fleet_telemetry: In[Any]

    def __init__(self) -> None:
        super().__init__()
        self._latest_reference: SwarmGeoReference | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(self.fleet_telemetry.subscribe(self._on_fleet_telemetry))  # type: ignore[arg-type]

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_fleet_telemetry(self, msg: Any) -> None:
        payload = self._decode_fleet_telemetry(msg)
        if payload is None:
            return

        reference = self._reference_from_payload(payload)
        if reference is None:
            return

        self._latest_reference = reference

    def _decode_fleet_telemetry(self, msg: Any) -> dict[str, Any] | None:
        if isinstance(msg, dict):
            return msg
        if isinstance(msg, String):
            raw = msg.data
        elif isinstance(msg, str):
            raw = msg
        else:
            raw = getattr(msg, "data", None)

        if not isinstance(raw, str):
            return None

        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Could not decode PX4 fleet telemetry JSON")
            return None

        return loaded if isinstance(loaded, dict) else None

    def _reference_from_payload(self, payload: dict[str, Any]) -> SwarmGeoReference | None:
        drones = payload.get("drones", [])
        if not isinstance(drones, list):
            return None

        valid: list[tuple[float, float, float, list[float]]] = []
        for drone in drones:
            if not isinstance(drone, dict):
                continue
            position = drone.get("position_enu")
            if (
                not isinstance(position, list)
                or len(position) < 3
                or not drone.get("has_local_position", True)
            ):
                continue
            lat = _to_float(drone.get("lat"))
            lon = _to_float(drone.get("lon"))
            if abs(lat) <= 1.0e-7 or abs(lon) <= 1.0e-7:
                continue
            valid.append(
                (
                    lat,
                    lon,
                    _to_float(drone.get("relative_alt")),
                    [_to_float(position[0]), _to_float(position[1]), _to_float(position[2])],
                )
            )

        if not valid:
            return None

        n = len(valid)
        gps = LatLon(
            lat=sum(item[0] for item in valid) / n,
            lon=sum(item[1] for item in valid) / n,
            alt=sum(item[2] for item in valid) / n,
        )
        enu = (
            sum(item[3][0] for item in valid) / n,
            sum(item[3][1] for item in valid) / n,
            sum(item[3][2] for item in valid) / n,
        )
        return SwarmGeoReference(gps=gps, enu=enu)

    def _get_reference(self) -> SwarmGeoReference | None:
        return self._latest_reference

    def _latlon_to_enu(self, target: LatLon, reference: SwarmGeoReference) -> tuple[float, float]:
        meters_per_degree_lat, meters_per_degree_lon = _meters_per_degree(reference.gps.lat)
        north_m = (target.lat - reference.gps.lat) * meters_per_degree_lat
        east_m = (target.lon - reference.gps.lon) * meters_per_degree_lon
        return reference.enu[0] + north_m, reference.enu[1] + east_m

    def _target_from_gps(
        self,
        lat: float,
        lon: float,
        altitude: float,
        max_distance_m: float,
    ) -> tuple[float, float, float] | str:
        reference = self._get_reference()
        if reference is None:
            return "Failed: no PX4 fleet GPS/ENU reference has been received yet."

        x, y = self._latlon_to_enu(LatLon(lat=lat, lon=lon), reference)
        distance_m = math.hypot(x - reference.enu[0], y - reference.enu[1])
        if max_distance_m > 0.0 and distance_m > max_distance_m:
            return (
                f"Failed: GPS target is {distance_m:.1f}m from the swarm centroid, "
                f"which exceeds max_distance_m={max_distance_m:.1f}."
            )

        return x, y, altitude

    @skill
    def go_to_map_location(
        self,
        lat: float,
        lon: float,
        altitude: float = DEFAULT_MAP_TARGET_ALTITUDE_M,
        formation_radius: float = 4.5,
        max_distance_m: float = DEFAULT_MAX_MAP_TARGET_DISTANCE_M,
    ) -> str:
        """Send the full PX4 swarm to a GPS latitude/longitude using local ENU control.

        Use this after `get_gps_position_for_queries` returns coordinates for a map place.

        Args:
            lat: Target latitude in degrees.
            lon: Target longitude in degrees.
            altitude: Target ENU altitude in meters.
            formation_radius: Radius around the target point for the swarm formation.
            max_distance_m: Reject targets farther than this from the current swarm centroid.
                Values <= 0 disable this guardrail.
        """

        target = self._target_from_gps(lat, lon, altitude, max_distance_m)
        if isinstance(target, str):
            return target

        x, y, z = target
        return self._px4_swarm.go_to_shared_point(
            x=x,
            y=y,
            z=z,
            formation_radius=formation_radius,
        )

    @skill
    def investigate_map_location(
        self,
        lat: float,
        lon: float,
        altitude: float = DEFAULT_MAP_TARGET_ALTITUDE_M,
        units: int = 0,
        formation_radius: float = 2.0,
        max_distance_m: float = DEFAULT_MAX_MAP_TARGET_DISTANCE_M,
    ) -> str:
        """Task PX4 swarm units to investigate a GPS latitude/longitude using local ENU control.

        Use this after `get_gps_position_for_queries` returns coordinates for a map place.

        Args:
            lat: Target latitude in degrees.
            lon: Target longitude in degrees.
            altitude: Target ENU altitude in meters.
            units: Number of drones to task. Values <= 0 task every drone.
            formation_radius: Radius around the target point for multiple units.
            max_distance_m: Reject targets farther than this from the current swarm centroid.
                Values <= 0 disable this guardrail.
        """

        target = self._target_from_gps(lat, lon, altitude, max_distance_m)
        if isinstance(target, str):
            return target

        x, y, z = target
        return self._px4_swarm.investigate_coordinate(
            x=x,
            y=y,
            z=z,
            units=units,
            formation_radius=formation_radius,
        )


__all__ = ["PX4SwarmMapNavigationSkill"]
