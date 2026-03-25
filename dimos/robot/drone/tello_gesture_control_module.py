#!/usr/bin/env python3
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

"""DimOS-native hand-gesture control for RoboMaster TT / Tello.

Adapted from the Apache-2.0 `droneWork/tello-gesture-control` reference and
rewired as a DimOS module that consumes the existing Tello video stream and
issues manual override commands through the Tello connection module.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from importlib import resources
import json
import threading
import time
from typing import TYPE_CHECKING, Any, TypeAlias

import cv2
from dimos_lcm.std_msgs import String
import numpy as np
from reactivex.disposable import CompositeDisposable, Disposable

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs import Image, ImageFormat
from dimos.robot.drone.tello_gesture_control_spec import TelloGestureControlSpec
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    ImageArray: TypeAlias = NDArray[Any]
else:
    ImageArray = np.ndarray

logger = setup_logger()

RcCommand: TypeAlias = tuple[int, int, int, int]

_ASSET_PACKAGE = "dimos.robot.drone.tello_gesture_assets"
_HAND_ACTIONS: dict[int, str] = {
    0: "forward",
    1: "stop",
    2: "up",
    3: "land",
    4: "down",
    5: "back",
    6: "left",
    7: "right",
}
_FINGER_ACTIONS: dict[int, str] = {
    1: "yaw_cw",
    2: "yaw_ccw",
}


def _add_disposable(composite: CompositeDisposable, item: Disposable | Any) -> None:
    if isinstance(item, Disposable):
        composite.add(item)
    elif callable(item):
        composite.add(Disposable(item))


def _asset_path(filename: str) -> str:
    return str(resources.files(_ASSET_PACKAGE).joinpath(filename))


def _required_asset_path(filename: str) -> str:
    asset = resources.files(_ASSET_PACKAGE).joinpath(filename)
    if not asset.is_file():
        raise ImportError(
            f"Gesture control requires the bundled asset `{filename}`, but it was not found in "
            "`dimos.robot.drone.tello_gesture_assets`."
        )
    return str(asset)


def _load_labels(filename: str) -> list[str]:
    text = resources.files(_ASSET_PACKAGE).joinpath(filename).read_text(encoding="utf-8-sig")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class GestureRecognitionResult:
    """Recognition output for a single frame."""

    overlay: ImageArray
    hand_sign_id: int | None
    hand_label: str | None
    finger_gesture_id: int | None
    finger_gesture_label: str | None
    candidate_action: str | None


@dataclass(frozen=True)
class GestureControlIntent:
    """Resolved control intent for a stable gesture."""

    action: str
    rc: RcCommand | None = None
    takeoff: bool = False
    land: bool = False


@dataclass(frozen=True)
class _DetectedHand:
    """Single-hand detection normalized across MediaPipe API variants."""

    landmarks: Any
    handedness_label: str


def resolve_gesture_intent(
    action: str | None,
    *,
    airborne: bool,
    allow_takeoff_gesture: bool,
    forward_speed: int,
    lateral_speed: int,
    vertical_speed: int,
    yaw_speed: int,
) -> GestureControlIntent | None:
    """Resolve a gesture action into a concrete manual-control intent."""
    if action is None:
        return None
    if action == "land":
        return GestureControlIntent(action=action, rc=(0, 0, 0, 0), land=True)
    if action == "up" and not airborne and allow_takeoff_gesture:
        return GestureControlIntent(action=action, takeoff=True)
    if action == "stop":
        return GestureControlIntent(action=action, rc=(0, 0, 0, 0))
    if action == "forward":
        return GestureControlIntent(action=action, rc=(0, forward_speed, 0, 0))
    if action == "back":
        return GestureControlIntent(action=action, rc=(0, -forward_speed, 0, 0))
    if action == "left":
        return GestureControlIntent(action=action, rc=(-lateral_speed, 0, 0, 0))
    if action == "right":
        return GestureControlIntent(action=action, rc=(lateral_speed, 0, 0, 0))
    if action == "up":
        return GestureControlIntent(action=action, rc=(0, 0, vertical_speed, 0))
    if action == "down":
        return GestureControlIntent(action=action, rc=(0, 0, -vertical_speed, 0))
    if action == "yaw_cw":
        return GestureControlIntent(action=action, rc=(0, 0, 0, yaw_speed))
    if action == "yaw_ccw":
        return GestureControlIntent(action=action, rc=(0, 0, 0, -yaw_speed))
    return None


class StableGestureBuffer:
    """Debounce gesture predictions into a stable action."""

    def __init__(self, buffer_len: int = 5, stable_threshold: int | None = None) -> None:
        self._buffer: deque[str | None] = deque(maxlen=max(1, buffer_len))
        self._stable_threshold = stable_threshold or max(2, buffer_len - 1)

    def add(self, action: str | None) -> None:
        self._buffer.append(action)

    def clear(self) -> None:
        self._buffer.clear()

    def stable_action(self) -> str | None:
        if not self._buffer:
            return None
        action, count = Counter(self._buffer).most_common(1)[0]
        if count < self._stable_threshold:
            return None
        return action


class _LiteInterpreter:
    """Lazy TFLite interpreter wrapper.

    Supports either `tflite-runtime` or TensorFlow's bundled Lite runtime.
    """

    def __init__(self, model_path: str, num_threads: int = 1) -> None:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore[import-not-found]
        except ImportError:
            try:
                import tensorflow as tf  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "Gesture control requires `mediapipe` and either `tensorflow` or "
                    "`tflite-runtime`."
                ) from exc
            Interpreter = tf.lite.Interpreter  # type: ignore[assignment]

        self._interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def classify(self, values: list[float]) -> NDArray[np.float32]:
        input_index = self._input_details[0]["index"]
        output_index = self._output_details[0]["index"]
        self._interpreter.set_tensor(input_index, np.array([values], dtype=np.float32))
        self._interpreter.invoke()
        return np.asarray(self._interpreter.get_tensor(output_index), dtype=np.float32)


class _KeyPointClassifier:
    def __init__(self, model_path: str, num_threads: int = 1) -> None:
        self._interpreter = _LiteInterpreter(model_path=model_path, num_threads=num_threads)

    def __call__(self, landmark_list: list[float]) -> int:
        result = self._interpreter.classify(landmark_list)
        return int(np.argmax(np.squeeze(result)))


class _PointHistoryClassifier:
    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.5,
        invalid_value: int = 0,
        num_threads: int = 1,
    ) -> None:
        self._interpreter = _LiteInterpreter(model_path=model_path, num_threads=num_threads)
        self._score_threshold = score_threshold
        self._invalid_value = invalid_value

    def __call__(self, point_history: list[float]) -> int:
        result = self._interpreter.classify(point_history)
        squeezed = np.squeeze(result)
        result_index = int(np.argmax(squeezed))
        if float(squeezed[result_index]) < self._score_threshold:
            return self._invalid_value
        return result_index


class TelloGestureRecognizer:
    """Hand-gesture recognizer built on Mediapipe Hands + TFLite classifiers."""

    def __init__(
        self,
        *,
        use_static_image_mode: bool = False,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        history_length: int = 16,
        num_threads: int = 1,
    ) -> None:
        try:
            import mediapipe as mp  # type: ignore[import-not-found, import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Gesture control requires `mediapipe`. Install `mediapipe` and either "
                "`tensorflow` or `tflite-runtime`."
            ) from exc

        self._mp = mp
        self._history_length = history_length
        self._hands: Any = None
        self._drawing_utils: Any = None
        self._drawing_styles: Any = None
        self._hand_connections: Any = None
        self._tasks_video_mode = False
        self._last_video_timestamp_ms = 0
        self._configure_mediapipe_backend(
            mp=mp,
            use_static_image_mode=use_static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._hand_labels = _load_labels("keypoint_classifier_label.csv")
        self._point_history_labels = _load_labels("point_history_classifier_label.csv")
        self._keypoint_classifier = _KeyPointClassifier(
            model_path=_asset_path("keypoint_classifier.tflite"),
            num_threads=num_threads,
        )
        self._point_history_classifier = _PointHistoryClassifier(
            model_path=_asset_path("point_history_classifier.tflite"),
            num_threads=num_threads,
        )
        self._point_history: deque[list[int]] = deque(maxlen=history_length)
        self._finger_gesture_history: deque[int] = deque(maxlen=history_length)

    def close(self) -> None:
        self._hands.close()

    def _configure_mediapipe_backend(
        self,
        *,
        mp: Any,
        use_static_image_mode: bool,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        solutions = getattr(mp, "solutions", None)
        if solutions is not None:
            self._hands = solutions.hands.Hands(
                static_image_mode=use_static_image_mode,
                max_num_hands=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._drawing_utils = solutions.drawing_utils
            self._drawing_styles = solutions.drawing_styles
            self._hand_connections = solutions.hands.HAND_CONNECTIONS
            return

        try:
            from mediapipe.tasks.python import vision  # type: ignore[import-untyped]
            from mediapipe.tasks.python.core.base_options import (  # type: ignore[import-untyped]
                BaseOptions,
            )
        except ImportError as exc:
            raise ImportError(
                "Installed `mediapipe` does not expose `mp.solutions`, and the Tasks "
                "HandLandmarker API is unavailable."
            ) from exc

        try:
            running_mode = vision.RunningMode.IMAGE
            if not use_static_image_mode:
                running_mode = vision.RunningMode.VIDEO
                self._tasks_video_mode = True
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=_required_asset_path("hand_landmarker.task")
                ),
                running_mode=running_mode,
                num_hands=1,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._hands = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise ImportError(
                f"Failed to initialize MediaPipe Tasks hand landmarker: {exc}"
            ) from exc

        self._drawing_utils = vision.drawing_utils
        self._drawing_styles = vision.drawing_styles
        self._hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    def _detect_hand(self, rgb: ImageArray, *, frame_ts: float | None) -> _DetectedHand | None:
        if self._tasks_video_mode:
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            results = self._hands.detect_for_video(
                mp_image,
                self._next_video_timestamp_ms(frame_ts),
            )
            if not results.hand_landmarks or not results.handedness:
                return None
            return _DetectedHand(
                landmarks=results.hand_landmarks[0],
                handedness_label=self._tasks_handedness_label(results.handedness[0]),
            )

        if getattr(self._mp, "solutions", None) is None:
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            results = self._hands.detect(mp_image)
            if not results.hand_landmarks or not results.handedness:
                return None
            return _DetectedHand(
                landmarks=results.hand_landmarks[0],
                handedness_label=self._tasks_handedness_label(results.handedness[0]),
            )

        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks or not results.multi_handedness:
            return None
        handedness = results.multi_handedness[0]
        handedness_label = str(getattr(handedness.classification[0], "label", "Unknown"))
        return _DetectedHand(
            landmarks=results.multi_hand_landmarks[0],
            handedness_label=handedness_label,
        )

    def _tasks_handedness_label(self, handedness: list[Any]) -> str:
        if not handedness:
            return "Unknown"
        category = handedness[0]
        label = getattr(category, "display_name", None) or getattr(category, "category_name", None)
        return str(label or "Unknown")

    def _next_video_timestamp_ms(self, frame_ts: float | None) -> int:
        candidate = int(time.time() * 1000)
        if frame_ts is not None:
            candidate = max(candidate, int(frame_ts * 1000))
        timestamp_ms = max(candidate, self._last_video_timestamp_ms + 1)
        self._last_video_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _landmark_points(self, landmarks: Any) -> list[Any]:
        raw_landmarks = getattr(landmarks, "landmark", landmarks)
        return list(raw_landmarks)

    def _draw_landmarks(self, image: ImageArray, hand_landmarks: Any) -> None:
        self._drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            self._hand_connections,
            self._drawing_styles.get_default_hand_landmarks_style(),
            self._drawing_styles.get_default_hand_connections_style(),
        )

    def recognize(self, frame: Image) -> GestureRecognitionResult:
        image = frame.to_opencv()
        debug_image = cv2.flip(image, 1)

        hand_sign_id: int | None = None
        hand_label: str | None = None
        finger_gesture_id: int | None = None
        finger_gesture_label: str | None = None
        candidate_action: str | None = None

        rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        detected_hand = self._detect_hand(rgb, frame_ts=frame.ts)
        rgb.flags.writeable = True

        if detected_hand is not None:
            hand_landmarks = detected_hand.landmarks

            brect = self._calc_bounding_rect(debug_image, hand_landmarks)
            landmark_list = self._calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmarks = self._pre_process_landmark(landmark_list)
            hand_sign_id = self._keypoint_classifier(pre_processed_landmarks)
            hand_label = self._safe_label(self._hand_labels, hand_sign_id)

            if hand_sign_id == 2:
                self._point_history.append(landmark_list[8])
            else:
                self._point_history.append([0, 0])

            pre_processed_point_history = self._pre_process_point_history(
                debug_image,
                self._point_history,
            )

            latest_finger_gesture_id = 0
            if len(pre_processed_point_history) == (self._history_length * 2):
                latest_finger_gesture_id = self._point_history_classifier(
                    pre_processed_point_history
                )
            self._finger_gesture_history.append(latest_finger_gesture_id)
            finger_gesture_id = Counter(self._finger_gesture_history).most_common(1)[0][0]
            finger_gesture_label = self._safe_label(self._point_history_labels, finger_gesture_id)

            candidate_action = self._candidate_action(hand_sign_id, finger_gesture_id)

            self._draw_landmarks(debug_image, hand_landmarks)
            self._draw_bounding_rect(debug_image, brect)
            self._draw_info_text(
                debug_image,
                brect=brect,
                handedness_label=detected_hand.handedness_label,
                hand_sign_text=hand_label,
                finger_gesture_text=finger_gesture_label,
            )
        else:
            self._point_history.append([0, 0])

        self._draw_point_history(debug_image, self._point_history)
        self._draw_footer(
            debug_image,
            hand_label=hand_label,
            finger_gesture_label=finger_gesture_label,
            action=candidate_action,
        )

        return GestureRecognitionResult(
            overlay=debug_image,
            hand_sign_id=hand_sign_id,
            hand_label=hand_label,
            finger_gesture_id=finger_gesture_id,
            finger_gesture_label=finger_gesture_label,
            candidate_action=candidate_action,
        )

    def _candidate_action(
        self, hand_sign_id: int | None, finger_gesture_id: int | None
    ) -> str | None:
        if hand_sign_id == 2 and finger_gesture_id in _FINGER_ACTIONS:
            return _FINGER_ACTIONS[finger_gesture_id]
        if hand_sign_id in _HAND_ACTIONS:
            return _HAND_ACTIONS[hand_sign_id]
        return None

    def _safe_label(self, labels: list[str], index: int | None) -> str | None:
        if index is None or index < 0 or index >= len(labels):
            return None
        return labels[index]

    def _calc_bounding_rect(self, image: ImageArray, landmarks: Any) -> list[int]:
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), dtype=int)

        for landmark in self._landmark_points(landmarks):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_array = np.append(landmark_array, [[landmark_x, landmark_y]], axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def _calc_landmark_list(self, image: ImageArray, landmarks: Any) -> list[list[int]]:
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_points: list[list[int]] = []

        for landmark in self._landmark_points(landmarks):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_points.append([landmark_x, landmark_y])

        return landmark_points

    def _pre_process_landmark(self, landmark_list: list[list[int]]) -> list[float]:
        if not landmark_list:
            return [0.0]

        temp_landmarks = [point.copy() for point in landmark_list]
        base_x, base_y = temp_landmarks[0]
        for point in temp_landmarks:
            point[0] -= base_x
            point[1] -= base_y

        flattened = [coord for point in temp_landmarks for coord in point]
        max_value = max(max(map(abs, flattened)), 1.0)
        return [value / max_value for value in flattened]

    def _pre_process_point_history(
        self,
        image: ImageArray,
        point_history: deque[list[int]],
    ) -> list[float]:
        image_width, image_height = image.shape[1], image.shape[0]
        temp_points = [point.copy() for point in point_history]
        if not temp_points:
            return []

        base_x, base_y = temp_points[0]
        normalized: list[float] = []
        for point in temp_points:
            normalized.extend(
                [
                    (point[0] - base_x) / max(image_width, 1),
                    (point[1] - base_y) / max(image_height, 1),
                ]
            )
        return normalized

    def _draw_bounding_rect(self, image: ImageArray, brect: list[int]) -> None:
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    def _draw_info_text(
        self,
        image: ImageArray,
        *,
        brect: list[int],
        handedness_label: str,
        hand_sign_text: str | None,
        finger_gesture_text: str | None,
    ) -> None:
        cv2.rectangle(image, (brect[0], brect[1] - 44), (brect[2], brect[1]), (0, 0, 0), -1)
        header = handedness_label
        if hand_sign_text:
            header = f"{header}: {hand_sign_text}"
        cv2.putText(
            image,
            header,
            (brect[0] + 5, brect[1] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if finger_gesture_text:
            cv2.putText(
                image,
                f"Motion: {finger_gesture_text}",
                (brect[0] + 5, brect[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 255, 200),
                1,
                cv2.LINE_AA,
            )

    def _draw_point_history(
        self,
        image: ImageArray,
        point_history: deque[list[int]],
    ) -> None:
        for index, point in enumerate(point_history):
            if point[0] == 0 and point[1] == 0:
                continue
            cv2.circle(
                image,
                (point[0], point[1]),
                1 + int(index / 2),
                (152, 251, 152),
                2,
            )

    def _draw_footer(
        self,
        image: ImageArray,
        *,
        hand_label: str | None,
        finger_gesture_label: str | None,
        action: str | None,
    ) -> None:
        footer = (
            f"gesture={hand_label or '-'} | motion={finger_gesture_label or '-'} "
            f"| action={action or '-'}"
        )
        cv2.rectangle(
            image,
            (0, image.shape[0] - 28),
            (image.shape[1], image.shape[0]),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            footer,
            (8, image.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


class TelloGestureControlModule(Module):
    """Gesture-control module for the RoboMaster TT / Tello platform."""

    video: In[Image]

    tracking_overlay: Out[Image]
    gesture_status: Out[Any]

    _tello: TelloGestureControlSpec

    def __init__(
        self,
        *,
        enabled_on_start: bool = False,
        allow_takeoff_gesture: bool = True,
        takeoff_altitude_m: float = 1.0,
        airborne_height_threshold_m: float = 0.18,
        buffer_len: int = 5,
        control_hz: float = 12.0,
        command_refresh_sec: float = 0.25,
        action_lost_timeout_sec: float = 0.45,
        manual_override_sec: float = 0.35,
        forward_speed: int = 35,
        lateral_speed: int = 25,
        vertical_speed: int = 25,
        yaw_speed: int = 35,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        num_threads: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.enabled_on_start = enabled_on_start
        self.allow_takeoff_gesture = allow_takeoff_gesture
        self.takeoff_altitude_m = takeoff_altitude_m
        self.airborne_height_threshold_m = airborne_height_threshold_m
        self.control_hz = max(1.0, control_hz)
        self.command_refresh_sec = max(0.05, command_refresh_sec)
        self.action_lost_timeout_sec = max(0.1, action_lost_timeout_sec)
        self.manual_override_sec = max(0.0, manual_override_sec)
        self.forward_speed = forward_speed
        self.lateral_speed = lateral_speed
        self.vertical_speed = vertical_speed
        self.yaw_speed = yaw_speed
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.num_threads = num_threads

        self._latest_frame: Image | None = None
        self._frame_lock = threading.Lock()
        self._buffer = StableGestureBuffer(buffer_len=buffer_len)
        self._recognizer: TelloGestureRecognizer | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._enabled = enabled_on_start
        self._active_action: str | None = None
        self._last_action_sent_at = 0.0
        self._last_seen_action_at = 0.0
        self._reset_action: str | None = None
        self._latest_status: dict[str, Any] = {
            "enabled": self._enabled,
            "active_action": None,
            "error": None,
            "ts": time.time(),
        }

    @rpc
    def start(self) -> None:
        """Start gesture recognition and control."""
        self._enabled = self.enabled_on_start
        self._stop_event.clear()
        self._buffer.clear()
        self._active_action = None
        self._last_action_sent_at = 0.0
        self._last_seen_action_at = 0.0
        self._reset_action = None

        try:
            self._recognizer = TelloGestureRecognizer(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                num_threads=self.num_threads,
            )
        except ImportError as exc:
            logger.error(f"Tello gesture control disabled: {exc}")
            self._publish_status(
                {
                    "enabled": False,
                    "active_action": None,
                    "error": str(exc),
                    "ts": time.time(),
                }
            )
            return

        if not self.video.transport:
            logger.warning("TelloGestureControlModule video input is not connected")
        else:
            _add_disposable(self._disposables, self.video.subscribe(self._on_video_frame))

        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()
        self._publish_status(
            {
                "enabled": self._enabled,
                "active_action": None,
                "error": None,
                "ts": time.time(),
            }
        )
        logger.info("TelloGestureControlModule started")

    def _on_video_frame(self, frame: Image) -> None:
        with self._frame_lock:
            self._latest_frame = frame

    def _processing_loop(self) -> None:
        period = 1.0 / self.control_hz
        while not self._stop_event.is_set():
            started_at = time.time()
            frame = self._pop_latest_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            recognizer = self._recognizer
            if recognizer is None:
                time.sleep(0.05)
                continue

            result = recognizer.recognize(frame)
            now = time.time()
            self._buffer.add(result.candidate_action)
            stable_action = self._buffer.stable_action()

            if result.candidate_action is not None:
                self._last_seen_action_at = now

            effective_action = self._apply_reset_latch(stable_action)
            if self._enabled:
                self._drive_from_action(effective_action, now)
            elif self._active_action is not None:
                self._send_stop_command()

            overlay = self._annotate_overlay(result.overlay.copy(), effective_action, now)
            if self.tracking_overlay.transport:
                self.tracking_overlay.publish(
                    Image.from_numpy(
                        overlay, format=ImageFormat.BGR, frame_id=frame.frame_id, ts=now
                    )
                )

            self._publish_status(
                {
                    "enabled": self._enabled,
                    "active_action": self._active_action,
                    "candidate_action": result.candidate_action,
                    "stable_action": effective_action,
                    "hand_gesture": result.hand_label,
                    "motion_gesture": result.finger_gesture_label,
                    "reset_action": self._reset_action,
                    "error": None,
                    "ts": now,
                }
            )

            sleep_for = period - (time.time() - started_at)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _pop_latest_frame(self) -> Image | None:
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
        return frame

    def _apply_reset_latch(self, stable_action: str | None) -> str | None:
        if self._reset_action is None:
            return stable_action
        if stable_action != self._reset_action:
            self._reset_action = None
            return stable_action
        return None

    def _drive_from_action(self, action: str | None, now: float) -> None:
        if action is None:
            if (
                self._active_action is not None
                and (now - self._last_seen_action_at) >= self.action_lost_timeout_sec
            ):
                self._send_stop_command()
            return

        intent = resolve_gesture_intent(
            action,
            airborne=self._is_airborne(),
            allow_takeoff_gesture=self.allow_takeoff_gesture,
            forward_speed=self.forward_speed,
            lateral_speed=self.lateral_speed,
            vertical_speed=self.vertical_speed,
            yaw_speed=self.yaw_speed,
        )
        if intent is None:
            return

        should_refresh = (now - self._last_action_sent_at) >= self.command_refresh_sec
        if intent.action == self._active_action and not should_refresh:
            return

        if intent.takeoff:
            self._tello.takeoff(altitude=self.takeoff_altitude_m)
            self._active_action = None
            self._reset_action = intent.action
            self._last_action_sent_at = now
            return

        if intent.rc is not None:
            self._tello.send_manual_rc(
                left_right=intent.rc[0],
                forward_back=intent.rc[1],
                up_down=intent.rc[2],
                yaw=intent.rc[3],
                manual_override_sec=self.manual_override_sec,
            )
            self._last_action_sent_at = now
            self._active_action = intent.action if any(intent.rc) else None

        if intent.land:
            self._tello.land()
            self._active_action = None
            self._reset_action = intent.action

    def _is_airborne(self) -> bool:
        status = self._tello.get_status()
        return _to_float(status.get("height_m")) >= self.airborne_height_threshold_m

    def _send_stop_command(self) -> None:
        self._tello.send_manual_rc(
            left_right=0,
            forward_back=0,
            up_down=0,
            yaw=0,
            manual_override_sec=0.0,
        )
        self._active_action = None
        self._last_action_sent_at = time.time()

    def _annotate_overlay(
        self,
        overlay: ImageArray,
        stable_action: str | None,
        ts: float,
    ) -> ImageArray:
        status_text = (
            f"gesture_control={'ON' if self._enabled else 'OFF'} | "
            f"stable={stable_action or '-'} | active={self._active_action or '-'}"
        )
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            status_text,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if not self._enabled:
            cv2.putText(
                overlay,
                "Enable gesture control to send manual override commands",
                (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 210, 120),
                1,
                cv2.LINE_AA,
            )
        elif self.allow_takeoff_gesture:
            cv2.putText(
                overlay,
                "Hold UP to take off when grounded. LAND gesture lands immediately.",
                (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 255, 180),
                1,
                cv2.LINE_AA,
            )
        return overlay

    def _publish_status(self, status: dict[str, Any]) -> None:
        self._latest_status = dict(status)
        if self.gesture_status.transport:
            self.gesture_status.publish(String(json.dumps(status)))

    @skill
    def enable_gesture_control(self) -> str:
        """Enable live hand-gesture control for the running Tello video stream."""
        if self._recognizer is None:
            error = self._latest_status.get("error")
            if error:
                return f"Failed: {error}"
        self._enabled = True
        self._publish_status(
            {
                **self._latest_status,
                "enabled": True,
                "ts": time.time(),
            }
        )
        return "Gesture control enabled"

    @skill
    def disable_gesture_control(self) -> str:
        """Disable hand-gesture control and send a zero-velocity manual override."""
        self._enabled = False
        self._send_stop_command()
        self._buffer.clear()
        self._publish_status(
            {
                **self._latest_status,
                "enabled": False,
                "active_action": None,
                "ts": time.time(),
            }
        )
        return "Gesture control disabled"

    @skill
    def gesture_control_status(self) -> str:
        """Summarize the current gesture-control state."""
        state = self._latest_status
        enabled = bool(state.get("enabled"))
        active = state.get("active_action") or "idle"
        stable = state.get("stable_action") or "none"
        error = state.get("error")
        if error:
            return f"Gesture control unavailable: {error}"
        return (
            f"Gesture control is {'enabled' if enabled else 'disabled'}; "
            f"stable gesture={stable}; active action={active}"
        )

    @rpc
    def get_gesture_status(self) -> dict[str, Any]:
        """Return the latest gesture-control status dictionary."""
        return dict(self._latest_status)

    @rpc
    def stop(self) -> None:
        """Stop gesture recognition and clear manual override."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._active_action is not None:
            self._send_stop_command()
        if self._recognizer is not None:
            self._recognizer.close()
            self._recognizer = None
        logger.info("TelloGestureControlModule stopped")
        super().stop()


tello_gesture_control_module = TelloGestureControlModule.blueprint

__all__ = [
    "GestureControlIntent",
    "GestureRecognitionResult",
    "StableGestureBuffer",
    "TelloGestureControlModule",
    "resolve_gesture_intent",
    "tello_gesture_control_module",
]
