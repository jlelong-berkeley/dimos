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

"""Generic drone module for MAVLink-based drones."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "camera_module": ["DroneCameraModule"],
        "connection_module": ["DroneConnectionModule"],
        "tello_fleet_config": ["TelloDroneConfig"],
        "tello_fleet_module": ["TelloFleetModule"],
        "tello_gesture_control_module": ["TelloGestureControlModule"],
        "mavlink_connection": ["MavlinkConnection"],
        "px4_offboard_connection": ["PX4OffboardDrone"],
        "px4_swarm_behavior": ["SwarmBehaviorLaw"],
        "px4_swarm_module": ["PX4SwarmModule"],
        "tello_connection_module": ["TelloConnectionModule"],
        "tello_sdk": ["TelloSdkClient"],
    },
)
