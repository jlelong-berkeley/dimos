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

import importlib.util

import numpy as np
import pytest

from dimos.msgs.sensor_msgs import Image, ImageFormat
from dimos.robot.drone.tello_gesture_control_module import (
    StableGestureBuffer,
    TelloGestureRecognizer,
    resolve_gesture_intent,
)


def test_stable_gesture_buffer_requires_consensus() -> None:
    buffer = StableGestureBuffer(buffer_len=5)
    for action in ["forward", "forward", "stop", "forward", "forward"]:
        buffer.add(action)

    assert buffer.stable_action() == "forward"


def test_stable_gesture_buffer_returns_none_without_threshold() -> None:
    buffer = StableGestureBuffer(buffer_len=5)
    for action in ["forward", "stop", "forward", "stop", None]:
        buffer.add(action)

    assert buffer.stable_action() is None


def test_resolve_gesture_intent_uses_takeoff_when_grounded() -> None:
    intent = resolve_gesture_intent(
        "up",
        airborne=False,
        allow_takeoff_gesture=True,
        forward_speed=35,
        lateral_speed=25,
        vertical_speed=25,
        yaw_speed=35,
    )

    assert intent is not None
    assert intent.takeoff is True
    assert intent.rc is None


def test_resolve_gesture_intent_uses_vertical_motion_when_airborne() -> None:
    intent = resolve_gesture_intent(
        "up",
        airborne=True,
        allow_takeoff_gesture=True,
        forward_speed=35,
        lateral_speed=25,
        vertical_speed=25,
        yaw_speed=35,
    )

    assert intent is not None
    assert intent.takeoff is False
    assert intent.rc == (0, 0, 25, 0)


def test_resolve_gesture_intent_maps_left_and_yaw() -> None:
    left = resolve_gesture_intent(
        "left",
        airborne=True,
        allow_takeoff_gesture=True,
        forward_speed=35,
        lateral_speed=25,
        vertical_speed=25,
        yaw_speed=35,
    )
    yaw = resolve_gesture_intent(
        "yaw_ccw",
        airborne=True,
        allow_takeoff_gesture=True,
        forward_speed=35,
        lateral_speed=25,
        vertical_speed=25,
        yaw_speed=35,
    )

    assert left is not None
    assert left.rc == (-25, 0, 0, 0)
    assert yaw is not None
    assert yaw.rc == (0, 0, 0, -35)


def test_resolve_gesture_intent_maps_land_to_stop_plus_land() -> None:
    intent = resolve_gesture_intent(
        "land",
        airborne=True,
        allow_takeoff_gesture=True,
        forward_speed=35,
        lateral_speed=25,
        vertical_speed=25,
        yaw_speed=35,
    )

    assert intent is not None
    assert intent.land is True
    assert intent.rc == (0, 0, 0, 0)


def test_tello_gesture_recognizer_initializes_with_installed_mediapipe() -> None:
    pytest.importorskip("mediapipe")
    if (
        importlib.util.find_spec("tflite_runtime") is None
        and importlib.util.find_spec("tensorflow") is None
    ):
        pytest.skip("Tello gesture recognizer needs tensorflow or tflite-runtime")

    recognizer = TelloGestureRecognizer(num_threads=1)
    try:
        frame = Image.from_numpy(
            np.zeros((64, 64, 3), dtype=np.uint8),
            format=ImageFormat.BGR,
            frame_id="test",
            ts=0.0,
        )
        result = recognizer.recognize(frame)
    finally:
        recognizer.close()

    assert result.overlay.shape == (64, 64, 3)
    assert result.candidate_action is None
