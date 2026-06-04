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

"""Real-hardware PX4 swarm blueprint, RTK variant (tight separation).

Identical wiring to ``drone-px4-swarm-hardware`` but assumes RTK-grade relative positioning, so
it defaults to a tight separation guardrail. Origins are still aligned from GPS at startup
(drone 0 = reference) -- RTK simply makes that alignment centimeter-accurate, which is what makes
tight formations safe. Same bring-up steps as the non-RTK blueprint; same ``.env`` knobs
(``px4_hw_n_drones``, ``px4_hw_min_separation_m`` to override, ``px4_hw_auto_origin``).

Use this once your radios/FCs are running RTK; otherwise use ``drone-px4-swarm-hardware``.
"""

from dimos.core.blueprints import autoconnect
from dimos.robot.drone.blueprints.basic._px4_hardware_common import hardware_swarm_kwargs
from dimos.robot.drone.px4_swarm_module import PX4SwarmModule

drone_px4_swarm_hardware_rtk = autoconnect(
    PX4SwarmModule.blueprint(**hardware_swarm_kwargs(default_min_separation_m=3.0)),
)

__all__ = ["drone_px4_swarm_hardware_rtk"]
