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

import json
from unittest.mock import MagicMock, patch

from dimos_lcm.std_msgs import String
import numpy as np

from dimos.robot.drone.blueprints.basic.drone_tello_tt_basic import _rerun_config
from dimos.robot.drone.drone_tracking_module import DroneTrackingModule
from dimos.robot.drone.tello_connection_module import (
    HOVER_OVERRIDE_SEC,
    TelloConnectionModule,
)
from dimos.robot.drone.tello_sdk import (
    KEEPALIVE_RESPONSE_TIMEOUT_SEC,
    TAKEOFF_RESPONSE_TIMEOUT_SEC,
    TelloSdkClient,
)


def test_tracking_module_stop_command_cancels_active_tracking() -> None:
    module = DroneTrackingModule()
    try:
        module._stop_tracking = MagicMock()
        module.track_object = MagicMock()

        module._on_follow_object_cmd(String(json.dumps({"command": "stop"})))

        module._stop_tracking.assert_called_once_with()
        module.track_object.assert_not_called()
    finally:
        module.stop()


def test_tello_sdk_flip_normalizes_direction_aliases() -> None:
    client = TelloSdkClient()
    client.send_command = MagicMock(return_value="ok")

    assert client.flip("left")
    client.send_command.assert_called_once_with("flip l")


def test_tello_sdk_flip_rejects_invalid_direction() -> None:
    client = TelloSdkClient()
    client.send_command = MagicMock(return_value="ok")

    assert not client.flip("sideways")
    client.send_command.assert_not_called()


def test_tello_sdk_resets_default_stream_ports_explicitly() -> None:
    client = TelloSdkClient()
    client.send_command = MagicMock(return_value="ok")

    assert client._configure_stream_ports() is True
    client.send_command.assert_called_once_with("port 8890 11111", retries=0)


def test_tello_sdk_configures_custom_stream_ports() -> None:
    client = TelloSdkClient(state_port=8891, video_port=11112)
    client.send_command = MagicMock(return_value="ok")

    assert client._configure_stream_ports() is True
    client.send_command.assert_called_once_with("port 8891 11112", retries=0)


def test_tello_sdk_query_ext_tof_parses_measurement() -> None:
    client = TelloSdkClient()
    client._send_command_with_timeout = MagicMock(return_value="tof 876")

    measurement = client.query_ext_tof()

    assert measurement is not None
    assert measurement["ext_tof_mm"] == 876
    assert measurement["ext_tof_valid"] is True
    assert measurement["ext_tof_m"] == 0.876


def test_tello_sdk_filters_default_video_stream_by_source_ip() -> None:
    client = TelloSdkClient(tello_ip="192.168.0.132", local_ip="0.0.0.0")
    client._running = False

    capture = MagicMock()
    capture.isOpened.return_value = True

    with patch("dimos.robot.drone.tello_sdk.cv2.VideoCapture", return_value=capture) as mock_capture:
        client._video_loop()

    video_uri = mock_capture.call_args.args[0]
    assert "udp://@0.0.0.0:11111" in video_uri
    assert "sources=192.168.0.132" in video_uri


def test_tello_sdk_takeoff_recovers_from_timeout_via_height_telemetry() -> None:
    client = TelloSdkClient()
    client._send_command_with_timeout = MagicMock(return_value="error:timeout")
    heights = [{"h": 0}, {"h": 0}, {"h": 22}]
    client.get_state = MagicMock(side_effect=lambda: heights.pop(0) if heights else {"h": 22})

    with patch("dimos.robot.drone.tello_sdk.time.sleep", return_value=None):
        assert client.takeoff() is True

    client._send_command_with_timeout.assert_called_once_with(
        "takeoff",
        retries=0,
        response_timeout=max(client.command_timeout, TAKEOFF_RESPONSE_TIMEOUT_SEC),
    )


def test_tello_sdk_idle_hover_sends_keepalive_when_airborne() -> None:
    client = TelloSdkClient()
    client.connected = True
    client._running = True
    client._latest_state = {"h": 35}
    client._last_command_ts = 0.0
    client._send_command_with_timeout = MagicMock(return_value="ok")

    sent = client._send_keepalive_if_needed(now=client._keepalive_interval_sec + 0.1)

    assert sent is True
    client._send_command_with_timeout.assert_called_once_with(
        "command",
        retries=0,
        response_timeout=KEEPALIVE_RESPONSE_TIMEOUT_SEC,
    )


def test_hover_skill_stops_tracking_and_hovers() -> None:
    module = TelloConnectionModule()
    try:
        module.connection = MagicMock()
        module.connection.stop.return_value = True
        module.follow_object_cmd.transport = MagicMock()
        module.follow_object_cmd.publish = MagicMock()
        module.send_manual_rc = MagicMock(return_value=True)
        module._wait_for_tracking_status = MagicMock(
            return_value=("stopped", {"status": "stopped"})
        )

        result = module.hover()

        module.send_manual_rc.assert_called_once_with(
            0, 0, 0, 0, manual_override_sec=HOVER_OVERRIDE_SEC
        )
        module.connection.stop.assert_called_once_with()
        payload = json.loads(module.follow_object_cmd.publish.call_args.args[0].data)
        assert payload == {"command": "stop"}
        assert "tracking was cancelled" in result.lower()
    finally:
        module.stop()


def test_flip_skill_cancels_tracking_before_flip() -> None:
    module = TelloConnectionModule()
    try:
        module.connection = MagicMock()
        module.connection.flip.return_value = True
        module.follow_object_cmd.transport = MagicMock()
        module.follow_object_cmd.publish = MagicMock()
        module.send_manual_rc = MagicMock(return_value=True)
        module._wait_for_tracking_status = MagicMock(
            return_value=("stopped", {"status": "stopped"})
        )

        result = module.flip("right")

        payload = json.loads(module.follow_object_cmd.publish.call_args.args[0].data)
        assert payload == {"command": "stop"}
        module.send_manual_rc.assert_called_once_with(0, 0, 0, 0, manual_override_sec=0.75)
        module.connection.flip.assert_called_once_with("right")
        assert "direction: right" in result.lower()
    finally:
        module.stop()


def test_tracking_status_controls_ext_tof_polling() -> None:
    module = TelloConnectionModule()
    try:
        module.connection = MagicMock()

        module._on_tracking_status({"status": "tracking"})
        module._on_tracking_status({"status": "stopped"})

        module.connection.start_ext_tof_polling.assert_called_once_with()
        module.connection.stop_ext_tof_polling.assert_called_once_with()
    finally:
        module.stop()


def test_tracking_module_uses_measured_distance_for_follow_speed() -> None:
    module = DroneTrackingModule()
    try:
        module._target_distance_m = 1.0

        hold_vx, _vy, _vz, _yaw = module._compute_person_follow_command(
            bbox=(220, 120, 420, 320),
            frame_width=640,
            _frame_height=360,
            current_x=320.0,
            measured_distance_m=1.0,
        )
        backoff_vx, _vy, _vz, _yaw = module._compute_person_follow_command(
            bbox=(220, 120, 420, 320),
            frame_width=640,
            _frame_height=360,
            current_x=320.0,
            measured_distance_m=0.75,
        )

        assert hold_vx == 0.0
        assert backoff_vx < 0.0
    finally:
        module.stop()


def test_tracking_module_local_subscribers_receive_outputs_without_transport() -> None:
    module = DroneTrackingModule()
    try:
        status_cb = MagicMock()
        cmd_vel_cb = MagicMock()
        overlay_cb = MagicMock()
        module.tracking_status.subscribe(status_cb)
        module.cmd_vel.subscribe(cmd_vel_cb)
        module.tracking_overlay.subscribe(overlay_cb)

        module._publish_status({"status": "tracking", "object": "person"})
        module._publish_cmd_vel(0.2, -0.1, 0.3)
        module._publish_overlay_image(np.zeros((8, 8, 3), dtype=np.uint8))

        status_payload = json.loads(status_cb.call_args.args[0].data)
        twist = cmd_vel_cb.call_args.args[0]
        overlay = overlay_cb.call_args.args[0]

        assert status_payload["status"] == "tracking"
        assert twist.linear.x == 0.2
        assert twist.linear.y == -0.1
        assert twist.angular.z == 0.3
        assert overlay.data.shape == (8, 8, 3)
    finally:
        module.stop()


def test_tello_rerun_config_suppresses_duplicate_image_topics() -> None:
    assert _rerun_config["min_interval_sec"] == 0.25
    visual_override = _rerun_config["visual_override"]
    assert visual_override["world/color_image"] is None
    assert visual_override["world/gesture_overlay"] is None
    assert callable(visual_override["world/video"])
    assert callable(visual_override["world/tracking_overlay"])
