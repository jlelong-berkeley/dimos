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

"""Basic RoboMaster TT/Tello blueprint with connection, camera, and visualization."""

from typing import Any

from dimos.core.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.msgs.sensor_msgs import Image
from dimos.protocol.pubsub.impl.lcmpubsub import LCM
from dimos.robot.drone.camera_module import DroneCameraModule
from dimos.robot.drone.tello_connection_module import TelloConnectionModule
from dimos.web.websocket_vis.websocket_vis_module import websocket_vis

RERUN_IMAGE_MAX_WIDTH = 640
RERUN_IMAGE_MAX_HEIGHT = 480
RERUN_IMAGE_INTERVAL_SEC = 0.25


def _static_drone_body(rr: Any) -> list[Any]:
    """Static visualization of TT body."""
    return [
        rr.Boxes3D(
            half_sizes=[0.09, 0.09, 0.03],
            colors=[(40, 180, 90)],
        ),
        rr.Transform3D(parent_frame="tf#/base_link"),
    ]


def _drone_rerun_blueprint() -> Any:
    """Split layout: camera feed + 3D world view side by side."""
    import rerun as rr
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="world/video", name="Camera"),
            rrb.Spatial2DView(origin="world/tracking_overlay", name="Tracking Overlay"),
            rrb.Spatial3DView(
                origin="world",
                name="3D",
                background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
                line_grid=rrb.LineGrid3D(
                    plane=rr.components.Plane3D.XY.with_distance(0.2),
                ),
            ),
            column_shares=[1, 1, 2],
        ),
    )


def _downsample_rerun_image(msg: Image) -> Image:
    resized, _scale = msg.resize_to_fit(
        max_width=RERUN_IMAGE_MAX_WIDTH,
        max_height=RERUN_IMAGE_MAX_HEIGHT,
    )
    return resized


_rerun_config = {
    "blueprint": _drone_rerun_blueprint,
    "pubsubs": [LCM()],
    "min_interval_sec": RERUN_IMAGE_INTERVAL_SEC,
    "visual_override": {
        # /color_image republishes the same camera frame as /video and doubles
        # the viewer bandwidth for no extra value in this layout.
        "world/color_image": None,
        # Gesture overlay is not displayed in the TT rerun layout; suppress it
        # to avoid another high-rate full-frame image stream.
        "world/gesture_overlay": None,
        # Keep the displayed feeds interactive by reducing per-frame size before
        # they hit the viewer transport and quotas.
        "world/video": _downsample_rerun_image,
        "world/tracking_overlay": _downsample_rerun_image,
    },
    "static": {
        "world/tf/base_link": _static_drone_body,
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

tello_ip = global_config.robot_ip or "192.168.10.1"

drone_tello_tt_basic = autoconnect(
    _vis,
    TelloConnectionModule.blueprint(
        tello_ip=tello_ip,
        local_ip="",
        local_command_port=9000,
        state_port=8890,
        video_port=11111,
    ),
    DroneCameraModule.blueprint(camera_intrinsics=[920.0, 920.0, 480.0, 360.0]),
    websocket_vis(),
)

__all__ = ["drone_tello_tt_basic"]
