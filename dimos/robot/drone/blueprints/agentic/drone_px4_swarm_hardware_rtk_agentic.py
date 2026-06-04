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

"""Agentic real-hardware PX4 swarm blueprint, RTK variant (tight separation).

Mirrors ``drone-px4-swarm-hardware-agentic`` but commands the RTK hardware stack
(``drone_px4_swarm_hardware_rtk``). Start the mavlink-router bridges first, then
``dimos run drone-px4-swarm-hardware-rtk-agentic``.
"""

from dimos.agents.agent import agent
from dimos.agents.skills.google_maps_skill_container import GoogleMapsSkillContainer
from dimos.core.blueprints import autoconnect
from dimos.robot.drone.blueprints.agentic.drone_px4_swarm_agentic import PX4_SWARM_SYSTEM_PROMPT
from dimos.robot.drone.blueprints.basic.drone_px4_swarm_hardware_rtk import (
    drone_px4_swarm_hardware_rtk,
)
from dimos.robot.drone.px4_swarm_map_navigation_skill import PX4SwarmMapNavigationSkill

drone_px4_swarm_hardware_rtk_agentic = autoconnect(
    drone_px4_swarm_hardware_rtk,
    GoogleMapsSkillContainer.blueprint(),
    PX4SwarmMapNavigationSkill.blueprint(),
    agent(system_prompt=PX4_SWARM_SYSTEM_PROMPT, model="gpt-4o"),
)

__all__ = ["drone_px4_swarm_hardware_rtk_agentic"]
