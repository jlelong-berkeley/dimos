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

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from dimos.core.global_config import GlobalConfig
from dimos.msgs.sensor_msgs import Image, ImageFormat
from dimos.robot.drone.tello_fleet_config import (
    get_tello_fleet_configs,
    get_tello_fleet_validation_error,
)
from dimos.robot.drone.tello_fleet_module import TelloFleetModule


def test_tello_fleet_config_supports_single_robot_ip() -> None:
    cfg = GlobalConfig(robot_ip="192.168.10.1")

    drones = get_tello_fleet_configs(cfg)

    assert [drone.key for drone in drones] == ["drone-1"]
    assert drones[0].tello_ip == "192.168.10.1"
    assert drones[0].local_ip == ""
    assert drones[0].state_port == 8890
    assert drones[0].video_port == 11111


def test_tello_fleet_config_supports_multi_with_one_laptop_ip() -> None:
    cfg = GlobalConfig(robot_ips="192.168.10.1,192.168.20.1")

    assert get_tello_fleet_validation_error(cfg) is None

    drones = get_tello_fleet_configs(cfg)

    assert [drone.key for drone in drones] == ["drone-1", "drone-2"]
    assert drones[0].local_ip == ""
    assert drones[1].local_ip == ""
    assert drones[0].state_port == 8890
    assert drones[1].state_port == 8891
    assert drones[0].video_port == 11111
    assert drones[1].video_port == 11112


def test_tello_fleet_config_accepts_shared_local_ip_override() -> None:
    cfg = GlobalConfig(
        robot_ips="192.168.10.1,192.168.20.1",
        robot_local_ips="192.168.1.7",
    )

    assert get_tello_fleet_validation_error(cfg) is None

    drones = get_tello_fleet_configs(cfg)
    assert drones[0].local_ip == "192.168.1.7"
    assert drones[1].local_ip == "192.168.1.7"


def test_tello_fleet_resolves_targets_by_index_or_ip() -> None:
    cfg = GlobalConfig(
        robot_ips="192.168.10.1,192.168.20.1",
        robot_local_ips="10.0.0.2,10.0.1.2",
    )
    module = TelloFleetModule(cfg=cfg)
    try:
        assert module._resolve_drone_key("drone 1") == "drone-1"
        assert module._resolve_drone_key("2") == "drone-2"
        assert module._resolve_drone_key("10.0.1.2") == "drone-2"
        assert module._resolve_drone_key("192.168.10.1") == "drone-1"
    finally:
        module.stop()


def test_tello_fleet_negotiates_default_slot_for_default_only_drone() -> None:
    cfg = GlobalConfig(robot_ips="192.168.10.1,192.168.20.1")
    module = TelloFleetModule(cfg=cfg)
    try:
        def fake_probe(drone: Any) -> bool:
            if drone.tello_ip == "192.168.20.1" and drone.state_port == 8891:
                return False
            return True

        module._probe_config = MagicMock(side_effect=fake_probe)

        negotiated = module._negotiate_drone_configs()

        assert negotiated[0].key == "drone-1"
        assert negotiated[0].state_port == 8891
        assert negotiated[0].video_port == 11112
        assert negotiated[1].key == "drone-2"
        assert negotiated[1].state_port == 8890
        assert negotiated[1].video_port == 11111
    finally:
        module.stop()


def test_dispatch_segmented_plan_runs_per_drone_sequences() -> None:
    cfg = GlobalConfig(
        robot_ips="192.168.10.1,192.168.20.1",
        robot_local_ips="10.0.0.2,10.0.1.2",
    )
    module = TelloFleetModule(cfg=cfg)
    try:
        drone1 = MagicMock()
        drone1.connected = True
        drone1.takeoff.return_value = True
        drone1.land.return_value = True

        drone2 = MagicMock()
        drone2.connected = True
        drone2.yaw.return_value = True

        module._clients = {
            "drone-1": drone1,
            "drone-2": drone2,
        }

        result = module.dispatch_segmented_plan(
            '[{"drone":"drone-1","action":"takeoff"},'
            '{"drone":"drone-1","action":"land"},'
            '{"drone":"drone-2","action":"yaw","degrees":90}]'
        )

        drone1.takeoff.assert_called_once_with()
        drone1.land.assert_called_once_with()
        drone2.yaw.assert_called_once_with(90.0)
        assert "drone-1:" in result
        assert "drone-2:" in result
    finally:
        module.stop()


def test_tracker_status_controls_fleet_ext_tof_polling() -> None:
    cfg = GlobalConfig(robot_ips="192.168.10.1")
    module = TelloFleetModule(cfg=cfg)
    try:
        client = MagicMock()
        client.connected = True
        module._clients = {"drone-1": client}

        module._on_tracker_status("drone-1", {"status": "tracking"})
        module._on_tracker_status("drone-1", {"status": "stopped"})

        client.start_ext_tof_polling.assert_called_once_with()
        client.stop_ext_tof_polling.assert_called_once_with()
    finally:
        module.stop()


def test_follow_object_impl_fails_when_target_is_not_found() -> None:
    cfg = GlobalConfig(robot_ips="192.168.10.1")
    module = TelloFleetModule(cfg=cfg)
    try:
        client = MagicMock()
        client.connected = True
        module._clients = {"drone-1": client}
        module._wait_until_airborne = MagicMock(return_value=True)

        tracker = MagicMock()
        tracker.get_status.return_value = {"active": False}
        tracker.track_object.return_value = "No object detected for: person"
        module._ensure_tracker = MagicMock(return_value=tracker)

        result = module._follow_object_impl(
            drone="drone-1",
            object_description="person",
            max_scan_steps=1,
            scan_step_deg=0.0,
        )

        assert result.startswith("Failed:")
    finally:
        module.stop()


def test_dispatch_segmented_plan_waits_for_tracking_before_next_step() -> None:
    cfg = GlobalConfig(robot_ips="192.168.10.1")
    module = TelloFleetModule(cfg=cfg)
    try:
        events: list[str] = []

        def fake_run_plan_step(drone: str, action: str, command: dict[str, Any]) -> str:
            del drone, command
            events.append(action)
            if action == "follow_object":
                return "drone-1: following person for up to 120s. Mode=full."
            return "drone-1: land command sent"

        module._run_plan_step = MagicMock(side_effect=fake_run_plan_step)
        module._wait_for_tracking_completion = MagicMock(
            side_effect=lambda drone, timeout: events.append("wait_tracking") or True
        )

        result = module.dispatch_segmented_plan(
            '[{"drone":"drone-1","action":"follow_object"},'
            '{"drone":"drone-1","action":"land"}]'
        )

        assert events == ["follow_object", "wait_tracking", "land"]
        assert "drone-1:" in result
    finally:
        module.stop()


def test_fleet_tracking_overlay_grid_renders_latest_overlay() -> None:
    cfg = GlobalConfig(robot_ips="192.168.10.1")
    module = TelloFleetModule(cfg=cfg)
    try:
        module._latest_status["drone-1"] = {"battery": 87}
        module._latest_tracking_overlays["drone-1"] = Image.from_numpy(
            np.zeros((12, 16, 3), dtype=np.uint8),
            format=ImageFormat.BGR,
        )

        grid = module._latest_tracking_overlay_grid()

        assert grid is not None
        assert grid.data.shape == (12, 16, 3)
    finally:
        module.stop()


def test_fleet_tracking_overlay_grid_falls_back_to_latest_video_frame() -> None:
    cfg = GlobalConfig(robot_ips="192.168.10.1")
    module = TelloFleetModule(cfg=cfg)
    try:
        module._latest_status["drone-1"] = {"battery": 87}
        module._latest_frames["drone-1"] = Image.from_numpy(
            np.full((12, 16, 3), 50, dtype=np.uint8),
            format=ImageFormat.BGR,
        )

        grid = module._latest_tracking_overlay_grid()

        assert grid is not None
        assert grid.data.shape == (12, 16, 3)
        assert int(grid.data.mean()) > 0
    finally:
        module.stop()
