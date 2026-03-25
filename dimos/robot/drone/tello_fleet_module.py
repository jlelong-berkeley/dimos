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

"""Multi-drone Tello control with segmented fleet commands and web UI."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import re
from threading import RLock, Thread
import time
from typing import Any

import cv2
from dimos_lcm.std_msgs import String
import numpy as np
import reactivex as rx
import reactivex.operators as ops

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.global_config import GlobalConfig, global_config
from dimos.core.module import Module
from dimos.core.stream import Out
from dimos.core.transport import pLCMTransport
from dimos.msgs.geometry_msgs import PoseStamped, Twist, Vector3
from dimos.msgs.sensor_msgs import Image, ImageFormat
from dimos.robot.drone.drone_tracking_module import DroneTrackingModule
from dimos.robot.drone.tello_fleet_config import TelloDroneConfig, get_tello_fleet_configs
from dimos.robot.drone.tello_sdk import (
    DEFAULT_STATE_PORT,
    DEFAULT_VIDEO_PORT,
    TelloSdkClient,
)
from dimos.utils.logging_config import setup_logger
from dimos.web.robot_web_interface import RobotWebInterface

logger = setup_logger()

_LABEL_COLOR = (40, 180, 90)
_EMPTY_FEED_SIZE = (360, 640)
FLEET_VIDEO_INTERVAL_SEC = 0.25
TRACKING_START_STATUSES = {"tracking", "not_found", "failed", "lost"}
TRACKING_STOP_STATUSES = {"stopped", "not_found", "failed", "lost"}
LONG_RUNNING_TRACKING_ACTIONS = {"follow_object", "center_person_by_yaw"}


def _normalize_target(value: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "", value.lower())


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (float, int)):
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


def _as_bgr_frame(image: Image) -> np.ndarray[Any, Any]:
    return image.to_bgr().to_opencv()


class TelloFleetModule(Module):
    """Manage one or more Tello drones behind a single agent-facing skill surface."""

    video: Out[Image]
    tracking_overlay: Out[Image]

    _global_config: GlobalConfig
    _drone_configs: list[TelloDroneConfig]
    _clients: dict[str, TelloSdkClient]
    _latest_frames: dict[str, Image | None]
    _latest_status: dict[str, dict[str, Any]]
    _latest_telemetry: dict[str, dict[str, Any]]
    _latest_odom: dict[str, PoseStamped | None]
    _latest_ext_tof: dict[str, dict[str, Any]]
    _latest_tracking_overlays: dict[str, Image | None]
    _latest_tracking_status: dict[str, dict[str, Any] | None]
    _tracking_modules: dict[str, DroneTrackingModule]
    _state_lock: RLock
    _web_interface: RobotWebInterface | None
    _web_thread: Thread | None
    _human_transport: pLCMTransport[str] | None
    _agent_transport: pLCMTransport[Any] | None
    _agent_unsub: Any
    _agent_response_subject: rx.subject.Subject[str] | None

    def __init__(
        self,
        port: int = 5555,
        cfg: GlobalConfig = global_config,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._global_config = cfg
        self.port = port
        self._drone_configs = get_tello_fleet_configs(cfg)
        self._clients = {}
        self._latest_frames = {}
        self._latest_status = {}
        self._latest_telemetry = {}
        self._latest_odom = {}
        self._latest_ext_tof = {}
        self._latest_tracking_overlays = {}
        self._latest_tracking_status = {}
        self._tracking_modules = {}
        self._last_fleet_video_publish_ts = 0.0
        self._state_lock = RLock()
        self._web_interface = None
        self._web_thread = None
        self._human_transport = None
        self._agent_transport = None
        self._agent_unsub = None
        self._agent_response_subject = None

    @rpc
    def start(self) -> None:
        super().start()

        self._drone_configs = self._negotiate_drone_configs()
        video_streams: dict[str, rx.Observable[np.ndarray[Any, Any]]] = {}

        connect_order = sorted(self._drone_configs, key=self._uses_default_stream_ports)
        for drone in connect_order:
            client = self._build_client(drone)
            self._clients[drone.key] = client
            self._latest_frames[drone.key] = None
            self._latest_status[drone.key] = {
                "drone": drone.key,
                "tello_ip": drone.tello_ip,
                "local_ip": drone.local_ip or "",
                "state_port": drone.state_port,
                "video_port": drone.video_port,
                "connected": False,
                "stream_port_acknowledged": False,
                "video_stream_active": False,
                "video_frame_received": False,
            }
            self._latest_telemetry[drone.key] = {}
            self._latest_odom[drone.key] = None
            self._latest_ext_tof[drone.key] = {}
            self._latest_tracking_overlays[drone.key] = None
            self._latest_tracking_status[drone.key] = None

            self._disposables.add(
                client.status_stream().subscribe(
                    lambda status, key=drone.key: self._on_status(key, status)
                )
            )
            self._disposables.add(
                client.telemetry_stream().subscribe(
                    lambda telemetry, key=drone.key: self._on_telemetry(key, telemetry)
                )
            )
            self._disposables.add(
                client.odom_stream().subscribe(lambda odom, key=drone.key: self._on_odom(key, odom))
            )
            self._disposables.add(
                client.video_stream().subscribe(
                    lambda frame, key=drone.key: self._on_video_frame(key, frame)
                )
            )
            self._disposables.add(
                client.ext_tof_stream().subscribe(
                    lambda measurement, key=drone.key: self._on_ext_tof(key, measurement)
                )
            )

            video_streams[drone.key] = client.video_stream().pipe(
                ops.map(_as_bgr_frame),
                ops.share(),
            )

            connected = client.connect()
            if connected and not self._client_ready_for_config(drone, client):
                logger.warning(
                    "Tello fleet member connected without the required stream routing; "
                    "keeping command-only mode disabled for safety",
                    drone=drone.key,
                    tello_ip=drone.tello_ip,
                    state_port=drone.state_port,
                    video_port=drone.video_port,
                    stream_port_response=client.stream_port_response,
                )
                client.disconnect()
                connected = False

            if not connected:
                logger.warning(
                    "Tello fleet member failed to connect",
                    drone=drone.key,
                    tello_ip=drone.tello_ip,
                    local_ip=drone.local_ip,
                )

        self._start_web_interface(video_streams)
        for drone in self._drone_configs:
            self._ensure_tracker(drone.key)
        self._publish_fleet_visuals_if_due(force=True)
        logger.info("TelloFleetModule started", drones=[drone.key for drone in self._drone_configs])

    def _start_web_interface(
        self, video_streams: dict[str, rx.Observable[np.ndarray[Any, Any]]]
    ) -> None:
        self._human_transport = pLCMTransport("/human_input")
        self._agent_transport = pLCMTransport("/agent")
        self._agent_response_subject = rx.subject.Subject()
        self._web_interface = RobotWebInterface(
            port=self.port,
            text_streams={"agent_responses": self._agent_response_subject},
            **video_streams,
        )

        query_subscription = self._web_interface.query_stream.subscribe(self._publish_human_text)
        self._disposables.add(query_subscription)
        self._agent_unsub = self._agent_transport.subscribe(self._on_agent_message)

        self._web_thread = Thread(target=self._web_interface.run, daemon=True)
        self._web_thread.start()
        logger.info(f"Tello fleet web interface started at http://localhost:{self.port}")

    def _publish_human_text(self, text: Any) -> None:
        if self._human_transport is None:
            return
        cleaned = str(text).strip()
        if cleaned:
            self._human_transport.publish(cleaned)

    def _on_agent_message(self, msg: Any) -> None:
        if self._agent_response_subject is None:
            return
        text = self._format_agent_message(msg)
        if text:
            self._agent_response_subject.on_next(text)

    @staticmethod
    def _format_agent_message(msg: Any) -> str:
        msg_type = str(getattr(msg, "type", msg.__class__.__name__)).lower()
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            additional = getattr(msg, "additional_kwargs", None)
            if isinstance(additional, dict):
                tool_calls = additional.get("tool_calls", [])

        lines: list[str] = []
        if content:
            lines.append(str(content))
        if tool_calls:
            lines.append("tool_calls:")
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    name = tool_call.get("name", "unknown")
                    args = tool_call.get("args", {})
                    lines.append(f"- {name}({args})")
                else:
                    lines.append(f"- {tool_call}")
        if not lines:
            lines.append("<no response>")
        return f"[{msg_type}] " + "\n".join(lines)

    def _on_status(self, drone: str, status: dict[str, Any]) -> None:
        with self._state_lock:
            self._latest_status[drone] = {
                **status,
                "drone": drone,
                "tello_ip": self._config_for(drone).tello_ip,
                "local_ip": self._config_for(drone).local_ip or "",
                "state_port": self._config_for(drone).state_port,
                "video_port": self._config_for(drone).video_port,
            }

    def _on_telemetry(self, drone: str, telemetry: dict[str, Any]) -> None:
        with self._state_lock:
            self._latest_telemetry[drone] = dict(telemetry)

    def _on_odom(self, drone: str, odom: PoseStamped) -> None:
        with self._state_lock:
            self._latest_odom[drone] = odom

    def _on_video_frame(self, drone: str, frame: Image) -> None:
        tracker: DroneTrackingModule | None = None
        with self._state_lock:
            self._latest_frames[drone] = frame
            tracker = self._tracking_modules.get(drone)
        if tracker is not None:
            tracker._on_new_frame(frame)
        self._publish_fleet_visuals_if_due()

    def _on_ext_tof(self, drone: str, measurement: dict[str, Any]) -> None:
        tracker: DroneTrackingModule | None = None
        with self._state_lock:
            self._latest_ext_tof[drone] = dict(measurement)
            tracker = self._tracking_modules.get(drone)
        if tracker is not None:
            tracker.update_range_measurement(measurement)

    def _on_tracker_overlay(self, drone: str, overlay: Image) -> None:
        with self._state_lock:
            self._latest_tracking_overlays[drone] = overlay
        self._publish_fleet_visuals_if_due()

    def _publish_fleet_visuals_if_due(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_fleet_video_publish_ts) < FLEET_VIDEO_INTERVAL_SEC:
            return
        video_frame = self._latest_frame_grid()
        overlay_frame = self._latest_tracking_overlay_grid()
        if video_frame is None and overlay_frame is None:
            return
        self._last_fleet_video_publish_ts = now
        if video_frame is not None:
            self.video.publish(video_frame)
        if overlay_frame is not None:
            self.tracking_overlay.publish(overlay_frame)

    def _on_tracker_cmd_vel(self, drone: str, twist: Twist) -> None:
        client = self._clients.get(drone)
        if client is None or not client.connected:
            return
        client.move_twist(twist, duration=0.0, lock_altitude=True)

    def _on_tracker_status(self, drone: str, status: Any) -> None:
        payload: dict[str, Any] | None = None
        if isinstance(status, String):
            try:
                payload = json.loads(status.data)
            except json.JSONDecodeError:
                payload = None
        elif isinstance(status, dict):
            payload = status

        if payload is None:
            return

        with self._state_lock:
            self._latest_tracking_status[drone] = dict(payload)

        client = self._clients.get(drone)
        if client is None:
            return
        value = payload.get("status")
        if isinstance(value, str):
            if value == "tracking":
                client.start_ext_tof_polling()
            elif value in TRACKING_STOP_STATUSES:
                client.stop_ext_tof_polling()

    def _ensure_tracker(self, drone: str) -> DroneTrackingModule:
        tracker = self._tracking_modules.get(drone)
        if tracker is not None:
            return tracker

        tracker = DroneTrackingModule(
            outdoor=False,
            enable_passive_overlay=True,
            use_local_person_detector=True,
            force_detection_servoing_for_person=True,
            person_follow_policy="yaw_forward_constant",
        )
        tracker._close_rpc()
        tracker.cmd_vel.subscribe(lambda twist, key=drone: self._on_tracker_cmd_vel(key, twist))
        tracker.tracking_status.subscribe(
            lambda status, key=drone: self._on_tracker_status(key, status)
        )
        tracker.tracking_overlay.subscribe(
            lambda overlay, key=drone: self._on_tracker_overlay(key, overlay)
        )

        with self._state_lock:
            self._tracking_modules[drone] = tracker
            latest_frame = self._latest_frames.get(drone)
            latest_range = self._latest_ext_tof.get(drone)

        if latest_frame is not None:
            tracker._on_new_frame(latest_frame)
        if latest_range:
            tracker.update_range_measurement(latest_range)
        tracker._start_passive_overlay_loop()
        return tracker

    def _config_for(self, drone: str) -> TelloDroneConfig:
        for config in self._drone_configs:
            if config.key == drone:
                return config
        raise KeyError(drone)

    @staticmethod
    def _uses_default_stream_ports(config: TelloDroneConfig) -> bool:
        return config.state_port == DEFAULT_STATE_PORT and config.video_port == DEFAULT_VIDEO_PORT

    def _build_client(self, drone: TelloDroneConfig) -> TelloSdkClient:
        return TelloSdkClient(
            tello_ip=drone.tello_ip,
            command_port=drone.command_port,
            state_port=drone.state_port,
            video_port=drone.video_port,
            local_ip=drone.local_ip,
            local_command_port=drone.local_command_port,
        )

    def _client_ready_for_config(self, drone: TelloDroneConfig, client: TelloSdkClient) -> bool:
        if not client.connected:
            return False
        if self._uses_default_stream_ports(drone):
            return True
        return client.stream_port_acknowledged

    def _probe_config(self, drone: TelloDroneConfig) -> bool:
        client = self._build_client(drone)
        try:
            if not client.connect(start_video=False):
                return False
            return self._client_ready_for_config(drone, client)
        finally:
            client.disconnect()

    def _negotiate_drone_configs(self) -> list[TelloDroneConfig]:
        if len(self._drone_configs) <= 1:
            return list(self._drone_configs)

        planned = list(self._drone_configs)
        default_index = next(
            (
                index
                for index, drone in enumerate(planned)
                if self._uses_default_stream_ports(drone)
            ),
            None,
        )
        if default_index is None:
            return planned

        for custom_index, drone in enumerate(planned):
            if custom_index == default_index or self._uses_default_stream_ports(drone):
                continue
            if self._probe_config(drone):
                continue

            default_drone = planned[default_index]
            proposed_default = replace(
                drone,
                state_port=DEFAULT_STATE_PORT,
                video_port=DEFAULT_VIDEO_PORT,
            )
            proposed_custom = replace(
                default_drone,
                state_port=drone.state_port,
                video_port=drone.video_port,
            )

            logger.warning(
                "Custom Tello stream ports were rejected; attempting to swap the default "
                "state/video ports to another drone",
                rejected_drone=drone.key,
                rejected_tello_ip=drone.tello_ip,
                proposed_default_drone=drone.key,
                proposed_custom_drone=default_drone.key,
                state_port=drone.state_port,
                video_port=drone.video_port,
            )

            if self._probe_config(proposed_default) and self._probe_config(proposed_custom):
                planned[custom_index] = proposed_default
                planned[default_index] = proposed_custom
                default_index = custom_index
                logger.info(
                    "Tello fleet stream-port assignment updated",
                    default_drone=proposed_default.key,
                    custom_drone=proposed_custom.key,
                    custom_state_port=proposed_custom.state_port,
                    custom_video_port=proposed_custom.video_port,
                )
            else:
                logger.warning(
                    "Tello fleet could not find a compatible default/custom port assignment",
                    rejected_drone=drone.key,
                    rejected_tello_ip=drone.tello_ip,
                )

        return planned

    def _aliases_for(self, drone: TelloDroneConfig) -> set[str]:
        index = int(drone.key.removeprefix("drone-"))
        aliases = {
            drone.key,
            f"drone{index}",
            f"drone {index}",
            str(index),
            drone.tello_ip,
        }
        if drone.local_ip:
            aliases.add(drone.local_ip)
        return aliases

    def _resolve_drone_key(self, drone: str = "") -> str:
        if not drone.strip():
            if len(self._drone_configs) == 1:
                return self._drone_configs[0].key
            raise ValueError(
                "Multiple drones are configured. Specify drone as drone-1, drone-2, or an IP."
            )

        wanted = _normalize_target(drone)
        matches = [
            config.key
            for config in self._drone_configs
            if wanted in {_normalize_target(alias) for alias in self._aliases_for(config)}
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"Unknown drone target: {drone}")
        raise ValueError(f"Ambiguous drone target: {drone}. Use drone-1, drone-2, and so on.")

    def _client_for(self, drone: str = "") -> tuple[str, TelloSdkClient]:
        key = self._resolve_drone_key(drone)
        return key, self._clients[key]

    def _connected(self, client: TelloSdkClient) -> bool:
        return client.connected

    def _wait_for_tracking_status(
        self,
        drone: str,
        timeout: float,
        accepted_statuses: set[str] | None = None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        statuses = accepted_statuses or TRACKING_START_STATUSES
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._state_lock:
                status = self._latest_tracking_status.get(drone)
                status_copy = dict(status) if status is not None else None
            if status_copy is not None:
                value = status_copy.get("status")
                if isinstance(value, str) and value in statuses:
                    return value, status_copy
            time.sleep(0.05)
        with self._state_lock:
            status = self._latest_tracking_status.get(drone)
            status_copy = dict(status) if status is not None else None
        return None, status_copy

    def _wait_until_airborne(
        self,
        client: TelloSdkClient,
        min_height_cm: float = 20.0,
        timeout: float = 12.0,
    ) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = client.get_state()
            if _to_float(state.get("h")) >= min_height_cm:
                return True
            time.sleep(0.1)
        state = client.get_state()
        return _to_float(state.get("h")) >= min_height_cm

    def _wait_for_tracking_completion(self, drone: str, timeout: float) -> bool:
        deadline = time.time() + max(0.5, timeout)
        while time.time() < deadline:
            tracker = self._tracking_modules.get(drone)
            if tracker is None:
                return True
            status = tracker.get_status()
            if not bool(status.get("active", False)):
                return True
            time.sleep(0.1)
        tracker = self._tracking_modules.get(drone)
        if tracker is None:
            return True
        return not bool(tracker.get_status().get("active", False))

    def _cancel_tracking_for_key(self, drone: str) -> bool:
        tracker = self._tracking_modules.get(drone)
        client = self._clients.get(drone)
        if client is not None:
            client.stop_ext_tof_polling()
        if tracker is None:
            return False
        status = tracker.get_status()
        if not bool(status.get("active", False)):
            return False
        tracker.stop_tracking()
        self._wait_for_tracking_status(
            drone,
            timeout=1.0,
            accepted_statuses=TRACKING_STOP_STATUSES,
        )
        return True

    def _stop_tracking_impl(self, drone: str) -> str:
        key, client = self._client_for(drone)
        if not self._connected(client):
            return f"Failed: {key} is not connected"
        cancelled = self._cancel_tracking_for_key(key)
        if cancelled:
            return f"{key}: tracking stopped"
        return f"{key}: no active tracking"

    def _move_impl(
        self, drone: str, x: float = 0.0, y: float = 0.0, z: float = 0.0, duration: float = 0.0
    ) -> str:
        key, client = self._client_for(drone)
        if not self._connected(client):
            return f"Failed: {key} is not connected"
        ok = client.move(Vector3(x, y, z), duration=duration)
        return f"{key}: move command sent" if ok else f"Failed: {key} move command rejected"

    def _move_relative_impl(
        self, drone: str, x: float = 0.0, y: float = 0.0, z: float = 0.0, speed: float = 0.4
    ) -> str:
        key, client = self._client_for(drone)
        if not self._connected(client):
            return f"Failed: {key} is not connected"
        ok = client.move_relative(x=x, y=y, z=z, speed=speed)
        return (
            f"{key}: relative move command sent"
            if ok
            else f"Failed: {key} move_relative rejected"
        )

    def _yaw_impl(self, drone: str, degrees: float) -> str:
        key, client = self._client_for(drone)
        if not self._connected(client):
            return f"Failed: {key} is not connected"
        ok = client.yaw(degrees)
        return f"{key}: yaw command sent" if ok else f"Failed: {key} yaw command rejected"

    def _takeoff_impl(self, drone: str, altitude: float = 1.0) -> str:
        key, client = self._client_for(drone)
        if not self._connected(client):
            return f"Failed: {key} is not connected"
        ok = client.takeoff()
        if not ok:
            return f"Failed: {key} takeoff rejected"
        if altitude > 1.0:
            return (
                f"{key}: takeoff command sent. Use move_relative_drone(z=...) "
                "to climb above the default hover height."
            )
        return f"{key}: takeoff command sent"

    def _land_impl(self, drone: str) -> str:
        key, client = self._client_for(drone)
        if not self._connected(client):
            return f"Failed: {key} is not connected"
        self._cancel_tracking_for_key(key)
        ok = client.land()
        return f"{key}: land command sent" if ok else f"Failed: {key} land rejected"

    def _hover_impl(self, drone: str) -> str:
        key, client = self._client_for(drone)
        if not self._connected(client):
            return f"Failed: {key} is not connected"
        cancelled = self._cancel_tracking_for_key(key)
        rc_ok = client.rc(0, 0, 0, 0)
        stop_ok = client.stop()
        if cancelled and (rc_ok or stop_ok):
            return f"{key}: hover override sent and tracking cancelled"
        if rc_ok or stop_ok:
            return f"{key}: hover override sent"
        return f"Failed: {key} hover override rejected"

    def _rc_impl(
        self,
        drone: str,
        left_right: int = 0,
        forward_back: int = 0,
        up_down: int = 0,
        yaw: int = 0,
        duration: float = 0.0,
    ) -> str:
        key, client = self._client_for(drone)
        if not self._connected(client):
            return f"Failed: {key} is not connected"
        ok = client.rc(left_right, forward_back, up_down, yaw)
        if duration > 0:
            import time

            time.sleep(duration)
            client.rc(0, 0, 0, 0)
        return f"{key}: rc command sent" if ok else f"Failed: {key} rc command rejected"

    def _flip_impl(self, drone: str, direction: str = "forward") -> str:
        key, client = self._client_for(drone)
        if not self._connected(client):
            return f"Failed: {key} is not connected"
        self._cancel_tracking_for_key(key)
        ok = client.flip(direction)
        if ok:
            return f"{key}: flip command sent in direction {direction}"
        return f"Failed: {key} flip rejected"

    def _send_ext_impl(self, drone: str, ext_command: str) -> str:
        key, client = self._client_for(drone)
        if not self._connected(client):
            return f"Failed: {key} is not connected"
        ok = client.send_ext(ext_command)
        return f"{key}: EXT command sent" if ok else f"Failed: {key} EXT command rejected"

    def _emergency_stop_impl(self, drone: str) -> str:
        key, client = self._client_for(drone)
        if not self._connected(client):
            return f"Failed: {key} is not connected"
        self._cancel_tracking_for_key(key)
        ok = client.emergency()
        return f"{key}: emergency command sent" if ok else f"Failed: {key} emergency rejected"

    def _follow_object_impl(
        self,
        drone: str,
        object_description: str = "person",
        distance_m: float = 1.0,
        duration: float = 120.0,
        scan_step_deg: float = 30.0,
        max_scan_steps: int = 12,
        control_mode: str = "full",
    ) -> str:
        key, client = self._client_for(drone)
        if not self._connected(client):
            return f"Failed: {key} is not connected"
        if not self._wait_until_airborne(client, min_height_cm=20.0, timeout=12.0):
            return f"Failed: {key} is not airborne yet"

        tracker = self._ensure_tracker(key)
        if tracker.get_status().get("active", False):
            return f"Failed: {key} is already tracking"

        steps = max(1, max_scan_steps)
        scan_budget_s = max(2.0, min(duration, 30.0))
        per_step_budget_s = max(0.8, scan_budget_s / steps)
        follow_duration_s = max(duration, 60.0)

        for step in range(steps):
            with self._state_lock:
                self._latest_tracking_status[key] = None

            result = tracker.track_object(
                object_name=object_description,
                duration=follow_duration_s,
                distance_m=distance_m,
                control_mode=control_mode,
            )
            if result.startswith("Tracking started") or result.startswith("Yaw-only tracking started"):
                status, detail = self._wait_for_tracking_status(
                    key,
                    timeout=min(1.5, per_step_budget_s),
                    accepted_statuses=TRACKING_START_STATUSES,
                )
                if status == "tracking":
                    return (
                        f"{key}: following {object_description} for up to {follow_duration_s:.0f}s. "
                        f"Mode={control_mode}."
                    )
                self._cancel_tracking_for_key(key)
                if status == "lost":
                    if step < steps - 1 and abs(scan_step_deg) > 0.0:
                        client.yaw(scan_step_deg)
                        time.sleep(0.5)
                        continue
                    return f"Failed: {key} lost {object_description} before follow could stabilize"
                if status == "failed":
                    reason = detail.get("error", "tracker failed") if detail else "tracker failed"
                    return f"Failed: {key} could not start tracking {object_description}: {reason}"
                if step < steps - 1 and abs(scan_step_deg) > 0.0:
                    client.yaw(scan_step_deg)
                    time.sleep(0.5)
                    continue
                return f"Failed: {key} could not acquire {object_description} for follow"

            result_lower = result.lower()
            if "already tracking" in result_lower:
                return f"Failed: {key} is already tracking"
            if result_lower.startswith("failed"):
                return f"{key}: {result}"

            if step < steps - 1 and abs(scan_step_deg) > 0.0:
                client.yaw(scan_step_deg)
                time.sleep(0.5)

        return (
            f"Failed: {key} could not find {object_description} after scanning "
            f"for about {scan_budget_s:.0f}s"
        )

    def _center_person_by_yaw_impl(
        self,
        drone: str,
        duration: float = 120.0,
        scan_step_deg: float = 0.0,
        max_scan_steps: int = 1,
    ) -> str:
        return self._follow_object_impl(
            drone=drone,
            object_description="person",
            distance_m=0.0,
            duration=duration,
            scan_step_deg=scan_step_deg,
            max_scan_steps=max_scan_steps,
            control_mode="yaw_only",
        )

    def _fleet_summary(self) -> str:
        with self._state_lock:
            lines = []
            for drone in self._drone_configs:
                status = dict(self._latest_status.get(drone.key, {}))
                connected = bool(status.get("connected", False))
                battery = status.get("battery", "unknown")
                local_ip = drone.local_ip or "auto"
                state_port = status.get("state_port", drone.state_port)
                video_port = status.get("video_port", drone.video_port)
                stream_ports_ok = bool(status.get("stream_port_acknowledged", False))
                video_ready = bool(status.get("video_frame_received", False))
                lines.append(
                    f"{drone.key}: tello_ip={drone.tello_ip}, local_ip={local_ip}, "
                    f"state_port={state_port}, video_port={video_port}, "
                    f"connected={connected}, stream_ports_ok={stream_ports_ok}, "
                    f"video_ready={video_ready}, battery={battery}"
                )
        return "\n".join(lines)

    def _render_label(self, frame: np.ndarray[Any, Any], label: str) -> np.ndarray[Any, Any]:
        canvas = frame.copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (10, 10, 10), -1)
        cv2.putText(
            canvas,
            label,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            _LABEL_COLOR,
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _placeholder_frame(self, label: str, status_text: str = "No video") -> np.ndarray[Any, Any]:
        height, width = _EMPTY_FEED_SIZE
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            _LABEL_COLOR,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            status_text,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _latest_labeled_grid(
        self,
        images_by_drone: dict[str, Image | None],
        status_text: str,
    ) -> Image | None:
        rendered: list[np.ndarray[Any, Any]] = []
        with self._state_lock:
            for drone in self._drone_configs:
                image = images_by_drone.get(drone.key)
                status = self._latest_status.get(drone.key, {})
                battery = status.get("battery", "unknown")
                label = f"{drone.key} | {drone.tello_ip} | bat={battery}"
                if image is None:
                    rendered.append(self._placeholder_frame(label, status_text=status_text))
                else:
                    rendered.append(self._render_label(_as_bgr_frame(image), label))

        if not rendered:
            return None

        cols = math.ceil(math.sqrt(len(rendered)))
        rows = math.ceil(len(rendered) / cols)
        height = max(frame.shape[0] for frame in rendered)
        width = max(frame.shape[1] for frame in rendered)
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        padded = []
        for frame in rendered:
            if frame.shape[:2] != (height, width):
                padded_frame = blank.copy()
                padded_frame[: frame.shape[0], : frame.shape[1]] = frame
                padded.append(padded_frame)
            else:
                padded.append(frame)

        while len(padded) < rows * cols:
            padded.append(blank.copy())

        row_images = []
        for row_index in range(rows):
            start = row_index * cols
            row_images.append(np.hstack(padded[start : start + cols]))
        grid = np.vstack(row_images)
        return Image.from_opencv(grid, format=ImageFormat.BGR)

    def _latest_frame_grid(self) -> Image | None:
        return self._latest_labeled_grid(self._latest_frames, status_text="No video")

    def _latest_tracking_overlay_grid(self) -> Image | None:
        with self._state_lock:
            overlays = {
                drone.key: (
                    self._latest_tracking_overlays.get(drone.key)
                    if self._latest_tracking_overlays.get(drone.key) is not None
                    else self._latest_frames.get(drone.key)
                )
                for drone in self._drone_configs
            }
        return self._latest_labeled_grid(overlays, status_text="Overlay pending")

    @skill
    def list_drones(self) -> str:
        """List the configured drones and their addressing keys."""
        return self._fleet_summary()

    @skill
    def observe_drone(self, drone: str = "") -> Image | None:
        """Return the latest camera frame for one drone.

        Args:
            drone: Target drone like "drone-1" or an IP address.
        """
        key = self._resolve_drone_key(drone)
        with self._state_lock:
            return self._latest_frames.get(key)

    @skill
    def observe_fleet(self) -> Image | None:
        """Return a labeled grid of the latest frames from all drones."""
        return self._latest_frame_grid()

    @skill
    def move_drone(
        self,
        drone: str = "",
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        duration: float = 0.0,
    ) -> str:
        """Move one drone with body-frame velocity.

        Args:
            drone: Target drone like "drone-1" or an IP address.
            x: Forward velocity in m/s.
            y: Left velocity in m/s.
            z: Up velocity in m/s.
            duration: Optional duration in seconds.
        """
        return self._move_impl(drone=drone, x=x, y=y, z=z, duration=duration)

    @skill
    def move_relative_drone(
        self,
        drone: str = "",
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        speed: float = 0.4,
    ) -> str:
        """Move one drone by a relative offset.

        Args:
            drone: Target drone like "drone-1" or an IP address.
            x: Forward distance in meters.
            y: Left distance in meters.
            z: Up distance in meters.
            speed: Translation speed in m/s.
        """
        return self._move_relative_impl(drone=drone, x=x, y=y, z=z, speed=speed)

    @skill
    def yaw_drone(self, drone: str = "", degrees: float = 0.0) -> str:
        """Rotate one drone in place.

        Args:
            drone: Target drone like "drone-1" or an IP address.
            degrees: Positive rotates counter-clockwise, negative clockwise.
        """
        return self._yaw_impl(drone=drone, degrees=degrees)

    @skill
    def takeoff_drone(self, drone: str = "", altitude: float = 1.0) -> str:
        """Take off one drone and hover.

        Args:
            drone: Target drone like "drone-1" or an IP address.
            altitude: Desired altitude hint in meters.
        """
        return self._takeoff_impl(drone=drone, altitude=altitude)

    @skill
    def land_drone(self, drone: str = "") -> str:
        """Land one drone immediately.

        Args:
            drone: Target drone like "drone-1" or an IP address.
        """
        return self._land_impl(drone=drone)

    @skill
    def hover_drone(self, drone: str = "") -> str:
        """Stop one drone and hold position.

        Args:
            drone: Target drone like "drone-1" or an IP address.
        """
        return self._hover_impl(drone=drone)

    @skill
    def rc_drone(
        self,
        drone: str = "",
        left_right: int = 0,
        forward_back: int = 0,
        up_down: int = 0,
        yaw: int = 0,
        duration: float = 0.0,
    ) -> str:
        """Send a low-level rc command to one drone.

        Args:
            drone: Target drone like "drone-1" or an IP address.
            left_right: Left/right channel in [-100, 100].
            forward_back: Forward/back channel in [-100, 100].
            up_down: Vertical channel in [-100, 100].
            yaw: Yaw channel in [-100, 100].
            duration: Optional duration before auto-stop.
        """
        return self._rc_impl(
            drone=drone,
            left_right=left_right,
            forward_back=forward_back,
            up_down=up_down,
            yaw=yaw,
            duration=duration,
        )

    @skill
    def flip_drone(self, drone: str = "", direction: str = "forward") -> str:
        """Execute a flip with one drone.

        Args:
            drone: Target drone like "drone-1" or an IP address.
            direction: forward, back, left, or right.
        """
        return self._flip_impl(drone=drone, direction=direction)

    @skill
    def send_ext_drone(self, drone: str = "", ext_command: str = "") -> str:
        """Send a TT EXT command to one drone.

        Args:
            drone: Target drone like "drone-1" or an IP address.
            ext_command: Raw EXT payload, with or without the EXT prefix.
        """
        return self._send_ext_impl(drone=drone, ext_command=ext_command)

    @skill
    def emergency_stop_drone(self, drone: str = "") -> str:
        """Emergency-stop one drone immediately.

        Args:
            drone: Target drone like "drone-1" or an IP address.
        """
        return self._emergency_stop_impl(drone=drone)

    @skill
    def follow_object_drone(
        self,
        drone: str = "",
        object_description: str = "person",
        distance_m: float = 1.0,
        duration: float = 120.0,
        scan_step_deg: float = 30.0,
        max_scan_steps: int = 12,
        control_mode: str = "full",
    ) -> str:
        """Scan for and follow an object with one drone.

        Args:
            drone: Target drone like "drone-1" or an IP address.
            object_description: Object to search for, usually "person".
            distance_m: Desired following distance in meters. Uses TT extension TOF
                when a valid reading is available.
            duration: Maximum follow duration in seconds once acquired.
            scan_step_deg: Yaw step used between search attempts.
            max_scan_steps: Number of scan attempts before giving up.
            control_mode: "full" for translation+tracking, or "yaw_only" to rotate only.
        """
        return self._follow_object_impl(
            drone=drone,
            object_description=object_description,
            distance_m=distance_m,
            duration=duration,
            scan_step_deg=scan_step_deg,
            max_scan_steps=max_scan_steps,
            control_mode=control_mode,
        )

    @skill
    def center_person_by_yaw_drone(
        self,
        drone: str = "",
        duration: float = 120.0,
        scan_step_deg: float = 0.0,
        max_scan_steps: int = 1,
    ) -> str:
        """Hover and rotate one drone to keep a detected person centered in view.

        Args:
            drone: Target drone like "drone-1" or an IP address.
            duration: Maximum tracking duration in seconds once acquired.
            scan_step_deg: Optional yaw scan step between search attempts.
            max_scan_steps: Number of search orientations before giving up.
        """
        return self._center_person_by_yaw_impl(
            drone=drone,
            duration=duration,
            scan_step_deg=scan_step_deg,
            max_scan_steps=max_scan_steps,
        )

    @skill
    def stop_tracking_drone(self, drone: str = "") -> str:
        """Stop current tracking for one drone.

        Args:
            drone: Target drone like "drone-1" or an IP address.
        """
        return self._stop_tracking_impl(drone=drone)

    def _run_plan_step(self, drone: str, action: str, command: dict[str, Any]) -> str:
        if action == "takeoff":
            return self._takeoff_impl(drone=drone, altitude=_to_float(command.get("altitude"), 1.0))
        if action == "land":
            return self._land_impl(drone=drone)
        if action == "hover":
            return self._hover_impl(drone=drone)
        if action == "move":
            return self._move_impl(
                drone=drone,
                x=_to_float(command.get("x")),
                y=_to_float(command.get("y")),
                z=_to_float(command.get("z")),
                duration=_to_float(command.get("duration")),
            )
        if action == "move_relative":
            return self._move_relative_impl(
                drone=drone,
                x=_to_float(command.get("x")),
                y=_to_float(command.get("y")),
                z=_to_float(command.get("z")),
                speed=_to_float(command.get("speed"), 0.4),
            )
        if action == "yaw":
            return self._yaw_impl(drone=drone, degrees=_to_float(command.get("degrees")))
        if action == "rc":
            return self._rc_impl(
                drone=drone,
                left_right=_to_int(command.get("left_right")),
                forward_back=_to_int(command.get("forward_back")),
                up_down=_to_int(command.get("up_down")),
                yaw=_to_int(command.get("yaw")),
                duration=_to_float(command.get("duration")),
            )
        if action == "flip":
            return self._flip_impl(drone=drone, direction=str(command.get("direction", "forward")))
        if action == "send_ext":
            return self._send_ext_impl(drone=drone, ext_command=str(command.get("ext_command", "")))
        if action == "follow_object":
            return self._follow_object_impl(
                drone=drone,
                object_description=str(command.get("object_description", "person")),
                distance_m=_to_float(command.get("distance_m"), 1.0),
                duration=_to_float(command.get("duration"), 120.0),
                scan_step_deg=_to_float(command.get("scan_step_deg"), 30.0),
                max_scan_steps=_to_int(command.get("max_scan_steps"), 12),
                control_mode=str(command.get("control_mode", "full")),
            )
        if action == "center_person_by_yaw":
            return self._center_person_by_yaw_impl(
                drone=drone,
                duration=_to_float(command.get("duration"), 120.0),
                scan_step_deg=_to_float(command.get("scan_step_deg"), 0.0),
                max_scan_steps=_to_int(command.get("max_scan_steps"), 1),
            )
        if action == "stop_tracking":
            return self._stop_tracking_impl(drone=drone)
        if action == "emergency_stop":
            return self._emergency_stop_impl(drone=drone)
        raise ValueError(
            "Unsupported action in segmented plan. Use takeoff, land, hover, move, "
            "move_relative, yaw, rc, flip, send_ext, follow_object, center_person_by_yaw, "
            "stop_tracking, or emergency_stop."
        )

    @skill
    def dispatch_segmented_plan(self, plan_json: str) -> str:
        """Run separate per-drone command sequences in parallel.

        Args:
            plan_json: JSON list of commands. Each item needs "drone" and "action".
        """
        try:
            parsed = json.loads(plan_json)
        except json.JSONDecodeError as exc:
            return f"Failed: invalid plan_json: {exc}"

        commands_raw = parsed.get("commands") if isinstance(parsed, dict) else parsed
        if not isinstance(commands_raw, list) or not commands_raw:
            return "Failed: plan_json must be a non-empty JSON list or a dict with commands"

        grouped: dict[str, list[dict[str, Any]]] = {}
        try:
            for command in commands_raw:
                if not isinstance(command, dict):
                    raise ValueError("Each segmented command must be a JSON object")
                drone_key = self._resolve_drone_key(str(command.get("drone", "")))
                grouped.setdefault(drone_key, []).append(command)
        except ValueError as exc:
            return f"Failed: {exc}"

        results: dict[str, list[str]] = {key: [] for key in grouped}
        result_lock = RLock()

        def run_drone_sequence(drone_key: str, commands: list[dict[str, Any]]) -> None:
            for command in commands:
                action = str(command.get("action", "")).strip().lower()
                if not action:
                    outcome = "Failed: missing action"
                else:
                    try:
                        outcome = self._run_plan_step(drone=drone_key, action=action, command=command)
                    except Exception as exc:
                        outcome = f"Failed: {exc}"
                with result_lock:
                    results[drone_key].append(outcome)
                if outcome.lower().startswith("failed"):
                    break
                if action in LONG_RUNNING_TRACKING_ACTIONS:
                    tracking_timeout = max(_to_float(command.get("duration"), 120.0) + 2.0, 2.0)
                    completed = self._wait_for_tracking_completion(
                        drone=drone_key,
                        timeout=tracking_timeout,
                    )
                    if not completed:
                        with result_lock:
                            results[drone_key][-1] = (
                                f"Failed: {drone_key} tracking did not finish within "
                                f"{tracking_timeout:.0f}s"
                            )
                        break

        threads = [
            Thread(target=run_drone_sequence, args=(drone_key, commands), daemon=True)
            for drone_key, commands in grouped.items()
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        summary_lines = []
        for drone_key in grouped:
            joined = " | ".join(results.get(drone_key, [])) or "No commands executed"
            summary_lines.append(f"{drone_key}: {joined}")
        return "\n".join(summary_lines)

    @rpc
    def stop(self) -> None:
        for tracker in self._tracking_modules.values():
            tracker.stop()
        self._tracking_modules.clear()
        if self._agent_unsub:
            self._agent_unsub()
            self._agent_unsub = None
        if self._agent_transport:
            self._agent_transport.stop()
            self._agent_transport = None
        if self._human_transport:
            self._human_transport.stop()
            self._human_transport = None
        if self._agent_response_subject:
            self._agent_response_subject.on_completed()
            self._agent_response_subject = None
        if self._web_interface:
            self._web_interface.shutdown()
            self._web_interface = None
        if self._web_thread:
            self._web_thread.join(timeout=1.0)
            self._web_thread = None
        for client in self._clients.values():
            client.stop_ext_tof_polling()
            client.disconnect()
        self._clients.clear()
        logger.info("TelloFleetModule stopped")
        super().stop()


tello_fleet_module = TelloFleetModule.blueprint

__all__ = ["TelloFleetModule", "tello_fleet_module"]
