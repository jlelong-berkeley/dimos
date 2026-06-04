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

import time

from dimos.robot.drone.px4_offboard_connection import PX4DroneConfig, PX4OffboardDrone
from dimos.robot.drone.px4_swarm_module import (
    AIRBORNE_ALTITUDE_M,
    PX4_LANDED_STATE_IN_AIR,
    PX4_LANDED_STATE_ON_GROUND,
    PX4_LANDED_STATE_TAKEOFF,
    PX4SwarmModule,
)


def test_status_history_preserves_takeoff_detected_after_later_status() -> None:
    drone = PX4OffboardDrone(PX4DroneConfig(key="x500_0", connection_string="udpin:0"))

    drone._record_status_text("Takeoff detected\t")
    drone._record_status_text("[logger] /fs/microsd/log/test.ulg")

    assert drone.snapshot.takeoff_detected
    assert "logger" in drone.snapshot.last_status_text
    messages = [entry["message"] for entry in drone.snapshot.status_history]
    assert any("Takeoff detected" in message for message in messages)


def test_airborne_gate_prefers_fresh_px4_landed_state_over_local_altitude() -> None:
    drone = PX4OffboardDrone(PX4DroneConfig(key="x500_0", connection_string="udpin:0"))
    snapshot = drone.snapshot
    snapshot.armed = True
    snapshot.position_enu = [0.0, 0.0, AIRBORNE_ALTITUDE_M + 1.0]
    snapshot.landed_state = PX4_LANDED_STATE_ON_GROUND
    snapshot.landed_state_s = time.time()

    assert not PX4SwarmModule._snapshot_airborne(snapshot)


def test_airborne_gate_accepts_px4_in_air_or_takeoff_state() -> None:
    drone = PX4OffboardDrone(PX4DroneConfig(key="x500_0", connection_string="udpin:0"))
    snapshot = drone.snapshot
    snapshot.armed = True
    snapshot.position_enu = [0.0, 0.0, 0.0]
    snapshot.landed_state = PX4_LANDED_STATE_IN_AIR
    snapshot.landed_state_s = time.time()

    assert PX4SwarmModule._snapshot_airborne(snapshot)

    snapshot.landed_state = PX4_LANDED_STATE_TAKEOFF
    snapshot.landed_state_s = time.time()

    assert PX4SwarmModule._snapshot_airborne(snapshot)


def test_newer_takeoff_detected_overrides_older_landed_state() -> None:
    drone = PX4OffboardDrone(PX4DroneConfig(key="x500_0", connection_string="udpin:0"))
    snapshot = drone.snapshot
    snapshot.armed = True
    snapshot.position_enu = [0.0, 0.0, 0.0]
    snapshot.landed_state = PX4_LANDED_STATE_ON_GROUND
    snapshot.landed_state_s = time.time() - 0.2
    snapshot.takeoff_detected = True
    snapshot.takeoff_detected_s = time.time()

    assert PX4SwarmModule._snapshot_airborne(snapshot)


def test_newer_landed_state_overrides_older_takeoff_detected() -> None:
    drone = PX4OffboardDrone(PX4DroneConfig(key="x500_0", connection_string="udpin:0"))
    snapshot = drone.snapshot
    snapshot.armed = True
    snapshot.position_enu = [0.0, 0.0, AIRBORNE_ALTITUDE_M + 1.0]
    snapshot.takeoff_detected = True
    snapshot.takeoff_detected_s = time.time() - 1.0
    snapshot.landed_state = PX4_LANDED_STATE_ON_GROUND
    snapshot.landed_state_s = time.time()

    assert not PX4SwarmModule._snapshot_airborne(snapshot)
