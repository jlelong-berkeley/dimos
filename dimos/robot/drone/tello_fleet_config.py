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

"""Helpers for configuring one or more Tello drones."""

from __future__ import annotations

from dataclasses import dataclass

from dimos.core.global_config import GlobalConfig, global_config

DEFAULT_TELLO_IP = "192.168.10.1"
DEFAULT_COMMAND_PORT = 8889
DEFAULT_LOCAL_COMMAND_PORT_BASE = 9000
DEFAULT_LOCAL_STATE_PORT_BASE = 8890
DEFAULT_LOCAL_VIDEO_PORT_BASE = 11111


@dataclass(frozen=True)
class TelloDroneConfig:
    """Configuration for a single Tello in a fleet."""

    key: str
    tello_ip: str
    local_ip: str
    local_command_port: int
    state_port: int
    video_port: int
    command_port: int = DEFAULT_COMMAND_PORT


def _split_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def get_tello_fleet_validation_error(cfg: GlobalConfig = global_config) -> str | None:
    tello_ips = _split_csv(cfg.robot_ips)
    if not tello_ips:
        tello_ips = [cfg.robot_ip or DEFAULT_TELLO_IP]

    local_ips = _split_csv(cfg.robot_local_ips)

    if local_ips and len(local_ips) not in (1, len(tello_ips)):
        return (
            "robot_local_ips must contain either one local bind IP for all drones "
            "or one entry per drone. Example: "
            "--robot-ips 192.168.1.9,192.168.1.10 --robot-local-ips 192.168.1.7"
        )

    return None


def get_tello_fleet_configs(cfg: GlobalConfig = global_config) -> list[TelloDroneConfig]:
    error = get_tello_fleet_validation_error(cfg)
    if error is not None:
        raise ValueError(error)

    tello_ips = _split_csv(cfg.robot_ips)
    if not tello_ips:
        tello_ips = [cfg.robot_ip or DEFAULT_TELLO_IP]

    local_ips = _split_csv(cfg.robot_local_ips)
    shared_local_ip = local_ips[0] if len(local_ips) == 1 else ""

    return [
        TelloDroneConfig(
            key=f"drone-{index + 1}",
            tello_ip=tello_ip,
            local_ip=local_ips[index] if len(local_ips) > 1 else shared_local_ip,
            local_command_port=DEFAULT_LOCAL_COMMAND_PORT_BASE + index,
            state_port=DEFAULT_LOCAL_STATE_PORT_BASE + index,
            video_port=DEFAULT_LOCAL_VIDEO_PORT_BASE + index,
        )
        for index, tello_ip in enumerate(tello_ips)
    ]


def format_tello_fleet_prompt_block(cfg: GlobalConfig = global_config) -> str:
    """Render a concise prompt section describing the configured fleet."""
    drones = get_tello_fleet_configs(cfg)
    lines = []
    for drone in drones:
        local_ip = drone.local_ip or "auto"
        lines.append(
            f"- {drone.key}: tello_ip={drone.tello_ip}, local_ip={local_ip}, "
            f"state_port={drone.state_port}, video_port={drone.video_port}"
        )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_COMMAND_PORT",
    "DEFAULT_LOCAL_COMMAND_PORT_BASE",
    "DEFAULT_LOCAL_STATE_PORT_BASE",
    "DEFAULT_LOCAL_VIDEO_PORT_BASE",
    "DEFAULT_TELLO_IP",
    "TelloDroneConfig",
    "format_tello_fleet_prompt_block",
    "get_tello_fleet_configs",
    "get_tello_fleet_validation_error",
]
