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

from threading import RLock
import time

from dimos.robot.drone.px4_offboard_connection import (
    PX4_FORCE_ARM_MAGIC,
    PX4DroneConfig,
    PX4OffboardDrone,
)
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


def test_airborne_gate_accepts_px4_in_air_state() -> None:
    drone = PX4OffboardDrone(PX4DroneConfig(key="x500_0", connection_string="udpin:0"))
    snapshot = drone.snapshot
    snapshot.armed = True
    snapshot.position_enu = [0.0, 0.0, 0.0]
    snapshot.landed_state = PX4_LANDED_STATE_IN_AIR
    snapshot.landed_state_s = time.time()

    assert PX4SwarmModule._snapshot_airborne(snapshot)


def test_takeoff_state_still_needs_altitude_to_complete_native_gate() -> None:
    drone = PX4OffboardDrone(PX4DroneConfig(key="x500_0", connection_string="udpin:0"))
    snapshot = drone.snapshot
    snapshot.armed = True
    snapshot.position_enu = [0.0, 0.0, 0.0]
    snapshot.landed_state = PX4_LANDED_STATE_TAKEOFF
    snapshot.landed_state_s = time.time()

    assert not PX4SwarmModule._snapshot_native_takeoff_complete(
        snapshot, min_airborne_altitude=0.25
    )

    snapshot.position_enu = [0.0, 0.0, 0.3]

    assert PX4SwarmModule._snapshot_native_takeoff_complete(
        snapshot, min_airborne_altitude=0.25
    )


def test_fresh_on_ground_state_overrides_takeoff_detected_latch() -> None:
    drone = PX4OffboardDrone(PX4DroneConfig(key="x500_0", connection_string="udpin:0"))
    snapshot = drone.snapshot
    snapshot.armed = True
    snapshot.position_enu = [0.0, 0.0, 0.0]
    snapshot.landed_state = PX4_LANDED_STATE_ON_GROUND
    snapshot.landed_state_s = time.time() - 0.2
    snapshot.takeoff_detected = True
    snapshot.takeoff_detected_s = time.time()

    assert not PX4SwarmModule._snapshot_airborne(snapshot)


def test_takeoff_detected_latch_is_fallback_without_fresh_landed_state() -> None:
    drone = PX4OffboardDrone(PX4DroneConfig(key="x500_0", connection_string="udpin:0"))
    snapshot = drone.snapshot
    snapshot.armed = True
    snapshot.position_enu = [0.0, 0.0, 0.0]
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


class _FakeMav:
    def __init__(self) -> None:
        self.command_args: tuple[float, ...] | None = None

    def command_long_send(self, *args: float) -> None:
        self.command_args = args


class _FakeMaster:
    def __init__(self) -> None:
        self.mav = _FakeMav()


def test_px4_force_disarm_sets_force_magic_param() -> None:
    drone = PX4OffboardDrone(PX4DroneConfig(key="x500_0", connection_string="udpin:0"))
    fake_master = _FakeMaster()
    drone.master = fake_master
    drone.send_gcs_heartbeat = lambda force=False: None  # type: ignore[method-assign]
    drone._wait_command_ack = lambda command, timeout: True  # type: ignore[method-assign]

    assert drone.disarm(force=True)
    assert fake_master.mav.command_args is not None
    assert fake_master.mav.command_args[4] == 0
    assert fake_master.mav.command_args[5] == PX4_FORCE_ARM_MAGIC


class _FakeDroneConfig:
    def __init__(self, key: str) -> None:
        self.key = key


class _FakeEmergencyDrone:
    def __init__(self, key: str) -> None:
        self.config = _FakeDroneConfig(key)
        self.disarm_forces: list[bool] = []

    def disarm(self, force: bool = False) -> bool:
        self.disarm_forces.append(force)
        return True


def _emergency_module() -> tuple[PX4SwarmModule, list[_FakeEmergencyDrone]]:
    module = PX4SwarmModule.__new__(PX4SwarmModule)
    drones = [_FakeEmergencyDrone("x500_0"), _FakeEmergencyDrone("x500_1")]
    module._command_lock = RLock()
    module._drones = drones  # type: ignore[assignment]
    module._hold_stop_event = None
    module._hold_thread = None
    module._hold_targets = None
    module._velocity_control_ready = True
    module._publish_fleet_status = lambda force=False: None  # type: ignore[method-assign]
    return module, drones


def test_emergency_force_disarm_requires_confirmation() -> None:
    module, drones = _emergency_module()

    result = module.emergency_force_disarm_swarm()

    assert result == "Failed: emergency force disarm requires confirm='FORCE_DISARM'"
    assert all(not drone.disarm_forces for drone in drones)


def test_emergency_force_disarm_passes_force_to_all_drones() -> None:
    module, drones = _emergency_module()

    result = module.emergency_force_disarm_swarm(confirm="force_disarm")

    assert result == "emergency_force_disarm_swarm sent: x500_0=True, x500_1=True"
    assert [drone.disarm_forces for drone in drones] == [[True], [True]]
    assert not module._velocity_control_ready
