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

"""Real-hardware PX4 swarm blueprint, NON-RTK (SiK radios bridged to local UDP).

Bring-up order:

1. Pair each SiK radio (a distinct NetID per pair) and give each flight controller a distinct
   MAV_SYS_ID.
2. Start the bridges (one --device per radio, order = x500_0, x500_1, ...)::

       python -m dimos.robot.drone.px4_mavlink_router --device <by-id> [--device <by-id> ...]

3. (optional) Start QGroundControl on the router's GCS endpoint (UDP 14550).
4. ``PX4_HW_N_DRONES=2 dimos run drone-px4-swarm-hardware`` (set the count in .env or the env).

Hardware-safe settings: real PX4 failsafes intact (``configure_sitl_failsafes=False``), reduced
telemetry/command rates for the bandwidth-limited radio link, and GPS-based origin alignment so
the drones share one ENU frame (drone 0 = reference) without manual surveying.

This is the NON-RTK variant: GPS relative error (~1-3 m/drone) makes tight spacing unsafe, so it
defaults to a generous separation guardrail. Once you have RTK, use ``drone-px4-swarm-hardware-rtk``
for tight formations. Tune without editing code via ``.env``:

    px4_hw_n_drones=2            # how many drones (also pass that many --device to the router)
    px4_hw_min_separation_m=8    # override the separation guardrail
    px4_hw_auto_origin=false     # disable GPS origin alignment (props-OFF indoor bench tests)
    px4_hw_bench_force_arm=true  # PROPS-OFF INDOOR BENCH ONLY: force-arm + SITL safety bundle
                                 # (persists params on the FC; revert in QGC before real flight)

SITL is unaffected; use ``drone-px4-swarm-sitl`` for simulation.
"""

from dimos.core.blueprints import autoconnect
from dimos.robot.drone.blueprints.basic._px4_hardware_common import hardware_swarm_kwargs
from dimos.robot.drone.px4_swarm_module import PX4SwarmModule

drone_px4_swarm_hardware = autoconnect(
    PX4SwarmModule.blueprint(**hardware_swarm_kwargs(default_min_separation_m=8.0)),
)

__all__ = ["drone_px4_swarm_hardware"]
