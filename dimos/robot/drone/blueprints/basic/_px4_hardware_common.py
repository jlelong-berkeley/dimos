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

"""Shared wiring for the real-hardware PX4 swarm blueprints (non-RTK and RTK variants).

The non-RTK and RTK runnables differ only in their default separation guardrail; everything
else (SiK-friendly rates, real failsafes, GPS-based origin alignment) is identical and lives
here. All knobs are read from GlobalConfig so they can be set via ``.env`` / environment
variables (read at import time -- the ``--`` CLI flags apply too late).
"""

from __future__ import annotations

from typing import Any

from dimos.core.global_config import global_config

# Conservative stream/command rates for a shared SiK link (vs SITL's 30/20 Hz on zero-latency
# localhost) so the Offboard setpoint stream and telemetry stay within the radio's bandwidth.
HARDWARE_TELEMETRY_RATES_HZ = {
    "local_position": 5.0,
    "heartbeat": 1.0,
    "sys_status": 1.0,
    "global_position": 2.0,
    "extended_sys_state": 1.0,
}
HARDWARE_COMMAND_HZ = 10.0


def hardware_swarm_kwargs(default_min_separation_m: float) -> dict[str, Any]:
    """Build PX4SwarmModule kwargs for a real-hardware swarm.

    Reads ``px4_hw_n_drones``, ``px4_hw_min_separation_m``, ``px4_hw_auto_origin`` and
    ``px4_hw_bench_force_arm`` from GlobalConfig. ``default_min_separation_m`` is the
    per-blueprint default (generous for non-RTK, tight for RTK) and is overridden by
    ``px4_hw_min_separation_m`` when that is set.

    Origins start at zero and are aligned from GPS at startup (drone 0 = reference) whenever
    auto-origin is enabled and more than one drone is connected, so the swarm shares one ENU
    frame without manual surveying.

    ``px4_hw_bench_force_arm=true`` enables the SITL safety-relaxation bundle
    (force-arm magic on every arm call, ``COM_RC_IN_MODE=4``, ``COM_DISARM_PRFLT=60``, low
    battery thresholds, supply-check circuit breaker, ...). INDOOR BENCH ONLY, PROPS OFF. The
    relaxed parameters are *persisted* on the flight controller; revert them in QGC before any
    real flight.
    """
    n_drones = max(1, int(global_config.px4_hw_n_drones))
    min_separation_m = global_config.px4_hw_min_separation_m
    if min_separation_m is None:
        min_separation_m = default_min_separation_m
    return {
        "connection_strings": [f"udpin:127.0.0.1:{14540 + index}" for index in range(n_drones)],
        # Placeholder; overwritten by GPS alignment at startup when auto-origin + >1 drone.
        "origin_positions_enu": [[0.0, 0.0, 0.0] for _ in range(n_drones)],
        "configure_sitl_failsafes": bool(global_config.px4_hw_bench_force_arm),
        "command_hz": HARDWARE_COMMAND_HZ,
        "telemetry_rates_hz": HARDWARE_TELEMETRY_RATES_HZ,
        "min_separation_m": float(min_separation_m),
        "auto_origin_from_gps": bool(global_config.px4_hw_auto_origin),
    }
