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

"""Agentic real-hardware PX4 swarm blueprint (SiK radios bridged via mavlink-router).

Mirrors ``drone-px4-swarm-agentic`` but commands real hardware through the
``drone_px4_swarm_hardware`` stack. Start the mavlink-router bridges first (see
``dimos.robot.drone.px4_mavlink_router``), then ``dimos run drone-px4-swarm-hardware-agentic``.
"""

from dimos.agents.agent import agent
from dimos.agents.skills.google_maps_skill_container import GoogleMapsSkillContainer
from dimos.core.blueprints import autoconnect
from dimos.robot.drone.blueprints.agentic.drone_px4_swarm_agentic import PX4_SWARM_SYSTEM_PROMPT
from dimos.robot.drone.blueprints.basic.drone_px4_swarm_hardware import drone_px4_swarm_hardware
from dimos.robot.drone.px4_swarm_map_navigation_skill import PX4SwarmMapNavigationSkill

drone_px4_swarm_hardware_agentic = autoconnect(
    drone_px4_swarm_hardware,
    GoogleMapsSkillContainer.blueprint(),
    PX4SwarmMapNavigationSkill.blueprint(),
    agent(system_prompt=PX4_SWARM_SYSTEM_PROMPT, model="gpt-4o"),
)

__all__ = ["drone_px4_swarm_hardware_agentic"]
