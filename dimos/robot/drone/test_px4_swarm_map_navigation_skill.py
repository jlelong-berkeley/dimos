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

from collections.abc import Iterator
import json
from typing import Any

from dimos_lcm.std_msgs import String
import pytest

from dimos.robot.drone.px4_swarm_map_navigation_skill import (
    PX4SwarmMapNavigationSkill,
    _meters_per_degree,
)


class FakePX4Swarm:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def go_to_shared_point(
        self,
        x: float,
        y: float,
        z: float = 8.0,
        formation_radius: float = 4.5,
    ) -> str:
        self.calls.append(
            (
                "go_to_shared_point",
                {"x": x, "y": y, "z": z, "formation_radius": formation_radius},
            )
        )
        return "go_to_shared_point called"

    def investigate_coordinate(
        self,
        x: float,
        y: float,
        z: float = 8.0,
        units: int = 0,
        formation_radius: float = 2.0,
    ) -> str:
        self.calls.append(
            (
                "investigate_coordinate",
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "units": units,
                    "formation_radius": formation_radius,
                },
            )
        )
        return "investigate_coordinate called"


def _fleet_payload(lat: float = 47.397742, lon: float = 8.545594) -> dict[str, Any]:
    return {
        "drones": [
            {
                "key": "x500_0",
                "lat": lat,
                "lon": lon,
                "relative_alt": 5.0,
                "position_enu": [10.0, -3.0, 5.0],
                "has_local_position": True,
            },
            {
                "key": "x500_1",
                "lat": lat,
                "lon": lon,
                "relative_alt": 5.0,
                "position_enu": [10.0, 3.0, 5.0],
                "has_local_position": True,
            },
        ]
    }


@pytest.fixture
def map_skill() -> Iterator[PX4SwarmMapNavigationSkill]:
    skill = PX4SwarmMapNavigationSkill()
    try:
        yield skill
    finally:
        skill.stop()


def test_go_to_map_location_uses_current_swarm_centroid_as_reference(
    map_skill: PX4SwarmMapNavigationSkill,
) -> None:
    fake_px4 = FakePX4Swarm()
    map_skill._px4_swarm = fake_px4  # type: ignore[assignment]
    map_skill._on_fleet_telemetry(String(json.dumps(_fleet_payload())))

    result = map_skill.go_to_map_location(
        lat=47.397742,
        lon=8.545594,
        altitude=12.0,
        formation_radius=7.0,
    )

    assert result == "go_to_shared_point called"
    assert fake_px4.calls == [
        (
            "go_to_shared_point",
            {"x": 10.0, "y": 0.0, "z": 12.0, "formation_radius": 7.0},
        )
    ]


def test_investigate_map_location_converts_gps_offset_to_enu(
    map_skill: PX4SwarmMapNavigationSkill,
) -> None:
    lat = 47.397742
    lon = 8.545594
    meters_per_degree_lat, meters_per_degree_lon = _meters_per_degree(lat)
    target_lat = lat + (25.0 / meters_per_degree_lat)
    target_lon = lon + (-12.0 / meters_per_degree_lon)

    fake_px4 = FakePX4Swarm()
    map_skill._px4_swarm = fake_px4  # type: ignore[assignment]
    map_skill._on_fleet_telemetry(String(json.dumps(_fleet_payload(lat=lat, lon=lon))))

    result = map_skill.investigate_map_location(
        lat=target_lat,
        lon=target_lon,
        altitude=9.0,
        units=2,
    )

    assert result == "investigate_coordinate called"
    name, kwargs = fake_px4.calls[0]
    assert name == "investigate_coordinate"
    assert kwargs["x"] == pytest.approx(35.0)
    assert kwargs["y"] == pytest.approx(-12.0)
    assert kwargs["z"] == 9.0
    assert kwargs["units"] == 2
