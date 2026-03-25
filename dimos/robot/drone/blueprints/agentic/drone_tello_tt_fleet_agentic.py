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

"""Agentic multi-drone Tello blueprint."""

from dimos.agents.agent import agent
from dimos.core.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.msgs.sensor_msgs import Image
from dimos.protocol.pubsub.impl.lcmpubsub import LCM
from dimos.robot.drone.tello_fleet_config import (
    format_tello_fleet_prompt_block,
    get_tello_fleet_configs,
    get_tello_fleet_validation_error,
)
from dimos.robot.drone.tello_fleet_module import TelloFleetModule

RERUN_IMAGE_MAX_WIDTH = 960
RERUN_IMAGE_MAX_HEIGHT = 720
RERUN_IMAGE_INTERVAL_SEC = 0.25


def _validate_tello_fleet_config() -> str | None:
    return get_tello_fleet_validation_error(global_config)


def _fleet_downsample_rerun_image(msg: Image) -> Image:
    resized, _scale = msg.resize_to_fit(
        max_width=RERUN_IMAGE_MAX_WIDTH,
        max_height=RERUN_IMAGE_MAX_HEIGHT,
    )
    return resized


def _fleet_rerun_blueprint() -> object:
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="world/video", name="Fleet Feeds"),
            rrb.Spatial2DView(origin="world/tracking_overlay", name="Tracking Overlay"),
            column_shares=[1, 1],
        ),
    )


_rerun_config = {
    "blueprint": _fleet_rerun_blueprint,
    "pubsubs": [LCM()],
    "min_interval_sec": RERUN_IMAGE_INTERVAL_SEC,
    "visual_override": {
        "world/video": _fleet_downsample_rerun_image,
        "world/tracking_overlay": _fleet_downsample_rerun_image,
    },
}

if global_config.viewer == "foxglove":
    from dimos.robot.foxglove_bridge import foxglove_bridge

    _vis = foxglove_bridge()
elif global_config.viewer.startswith("rerun"):
    from dimos.visualization.rerun.bridge import _resolve_viewer_mode, rerun_bridge

    _vis = rerun_bridge(viewer_mode=_resolve_viewer_mode(), **_rerun_config)
else:
    _vis = autoconnect()


_FLEET_CONFIG_ERROR = get_tello_fleet_validation_error(global_config)
if _FLEET_CONFIG_ERROR is None:
    _FLEET_PROMPT_BLOCK = format_tello_fleet_prompt_block(global_config)
    _FLEET_SIZE = len(get_tello_fleet_configs(global_config))
else:
    _FLEET_PROMPT_BLOCK = (
        "- drone-1: tello_ip=192.168.10.1, local_ip=auto, state_port=8890, video_port=11111"
    )
    _FLEET_SIZE = 1

TELLO_TT_FLEET_SYSTEM_PROMPT = f"""\
You are controlling {_FLEET_SIZE} RoboMaster TT (Tello Talent) drone(s).
Configured drones:
{_FLEET_PROMPT_BLOCK}
Always reason about which drone should receive each command.
Use list_drones() if the user is unclear about the available drones or connectivity.
Use observe_drone(drone=...) or observe_fleet() before aggressive motion.
Use takeoff_drone, land_drone, hover_drone, move_drone, move_relative_drone, yaw_drone,
rc_drone, flip_drone, send_ext_drone, follow_object_drone, center_person_by_yaw_drone,
stop_tracking_drone, and emergency_stop_drone for targeted control.
takeoff_drone already leaves the drone hovering, so do not call hover_drone immediately after
takeoff_drone unless you are cancelling a later motion command.
For person follow, `distance_m` uses the TT extension TOF sensor when it has a valid reading.
For hover-and-center requests, use center_person_by_yaw_drone(drone=..., ...).
When the user assigns different actions to different drones, use dispatch_segmented_plan(plan_json=...)
so each drone executes its own sequence in parallel.
Do not append land_drone after follow_object_drone or center_person_by_yaw_drone unless the user
explicitly asks to land.
If only one drone is configured, you may omit the drone argument and default to that drone.
Keep altitude conservative indoors and report which drones you commanded.
"""


drone_tello_tt_fleet_agentic = autoconnect(
    _vis,
    TelloFleetModule.blueprint(),
    agent(system_prompt=TELLO_TT_FLEET_SYSTEM_PROMPT, model="gpt-4o"),
).requirements(_validate_tello_fleet_config)


__all__ = ["TELLO_TT_FLEET_SYSTEM_PROMPT", "drone_tello_tt_fleet_agentic"]
