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

"""Portable PX4 swarm velocity allocation law.

The law is intentionally independent of DimOS modules, MAVLink, and Gazebo. It
is the behavior layer optimized in Project3: attraction to desired points plus
repulsion from other drones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

DESIGN_VARIABLES = (
    "W_goal",
    "W_sep",
    "w_goal",
    "a_goal",
    "w_sep_near",
    "w_sep_far",
    "c_sep_near",
    "c_sep_far",
    "v_max",
    "z_gain",
)

PROJECT3_BEST_DESIGN = (
    1.7948754374476419,
    1.9579069872542347,
    2.0,
    0.6291365632040238,
    0.6350236258734429,
    1.7880159181510507,
    1.7889509174766158,
    0.19954835261128723,
    2.0,
    1.336972818184744,
)


def _safe_norm(values: np.ndarray[Any, Any], axis: int = -1) -> np.ndarray[Any, Any]:
    return cast("np.ndarray[Any, Any]", np.linalg.norm(values, axis=axis))


@dataclass(frozen=True)
class SwarmBehaviorWeights:
    """Weights for the Project3 swarm behavior equation."""

    W_goal: float
    W_sep: float
    w_goal: float
    a_goal: float
    w_sep_near: float
    w_sep_far: float
    c_sep_near: float
    c_sep_far: float
    v_max: float
    z_gain: float

    @classmethod
    def from_design_vector(cls, values: list[float] | tuple[float, ...] | np.ndarray[Any, Any]) -> SwarmBehaviorWeights:
        """Create behavior weights from a Project3-style design vector."""
        vector = np.asarray(values, dtype=float)
        if len(vector) != len(DESIGN_VARIABLES):
            raise ValueError(f"Expected {len(DESIGN_VARIABLES)} design values, got {len(vector)}")
        return cls(
            W_goal=float(vector[0]),
            W_sep=float(vector[1]),
            w_goal=float(vector[2]),
            a_goal=max(float(vector[3]), 1.0e-3),
            w_sep_near=float(vector[4]),
            w_sep_far=float(vector[5]),
            c_sep_near=max(float(vector[6]), 1.0e-3),
            c_sep_far=max(float(vector[7]), 1.0e-3),
            v_max=max(float(vector[8]), 0.05),
            z_gain=max(float(vector[9]), 0.05),
        )

    def to_design_vector(self) -> list[float]:
        """Return the design vector in Project3 variable order."""
        return [float(getattr(self, name)) for name in DESIGN_VARIABLES]


class SwarmBehaviorLaw:
    """Goal-attraction plus drone-drone-repulsion velocity law."""

    def __init__(
        self,
        weights: SwarmBehaviorWeights | None = None,
        min_separation: float = 3.0,
        max_vertical_speed: float = 1.0,
    ) -> None:
        self.weights = weights or SwarmBehaviorWeights.from_design_vector(PROJECT3_BEST_DESIGN)
        self.min_separation = min_separation
        self.max_vertical_speed = max_vertical_speed

    @classmethod
    def from_design_vector(
        cls,
        values: list[float] | tuple[float, ...] | np.ndarray[Any, Any],
        min_separation: float = 3.0,
        max_vertical_speed: float = 1.0,
    ) -> SwarmBehaviorLaw:
        """Build a law from the optimized Project3 design vector."""
        return cls(
            weights=SwarmBehaviorWeights.from_design_vector(values),
            min_separation=min_separation,
            max_vertical_speed=max_vertical_speed,
        )

    def velocity_commands(
        self,
        positions_enu: np.ndarray[Any, Any],
        desired_points_enu: np.ndarray[Any, Any],
        active_mask: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        """Compute ENU velocity commands for every drone.

        Args:
            positions_enu: ``(N, 3)`` positions in meters, z positive up.
            desired_points_enu: ``(N, 3)`` desired point for each drone.
            active_mask: Boolean mask for drones still receiving commands.
        """
        positions = np.asarray(positions_enu, dtype=float)
        desired = np.asarray(desired_points_enu, dtype=float)
        if positions.shape != desired.shape or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions_enu and desired_points_enu must both have shape (N, 3)")

        n_drones = positions.shape[0]
        if active_mask is None:
            active = np.ones(n_drones, dtype=bool)
        else:
            active = np.asarray(active_mask, dtype=bool)

        goal_vectors = self._goal_attraction(positions, desired)
        separation_vectors = self._separation_repulsion(positions, active)

        command = (self.weights.W_goal * goal_vectors) + (self.weights.W_sep * separation_vectors)
        command[:, 2] *= self.weights.z_gain
        command[~active] = 0.0

        command = self._clip_speed(command, self.weights.v_max)
        command[:, 2] = np.clip(command[:, 2], -self.max_vertical_speed, self.max_vertical_speed)
        return command

    def _goal_attraction(
        self,
        positions: np.ndarray[Any, Any],
        desired: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        diff = desired - positions
        distance = _safe_norm(diff, axis=1)
        direction = np.divide(
            diff,
            distance[:, np.newaxis],
            out=np.zeros_like(diff),
            where=distance[:, None] > 1.0e-9,
        )
        magnitude = self.weights.w_goal * (1.0 - np.exp(-self.weights.a_goal * distance))
        return cast("np.ndarray[Any, Any]", magnitude[:, np.newaxis] * direction)

    def _separation_repulsion(
        self,
        positions: np.ndarray[Any, Any],
        active: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        repulsion = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            if not bool(active[i]):
                continue
            for j in range(positions.shape[0]):
                if i == j or not bool(active[j]):
                    continue
                away = positions[i] - positions[j]
                distance = float(np.linalg.norm(away))
                if distance <= 1.0e-9:
                    continue
                direction = away / distance
                exponential = (
                    self.weights.w_sep_near * np.exp(-self.weights.c_sep_near * distance)
                    + self.weights.w_sep_far * np.exp(-self.weights.c_sep_far * distance)
                )
                safety_scale = (self.min_separation / max(distance, 1.0e-3)) ** 2
                repulsion[i] += exponential * safety_scale * direction
        return repulsion

    @staticmethod
    def _clip_speed(command: np.ndarray[Any, Any], max_speed: float) -> np.ndarray[Any, Any]:
        speed = _safe_norm(command, axis=1)
        scale = np.ones_like(speed)
        over = speed > max_speed
        scale[over] = max_speed / speed[over]
        return cast("np.ndarray[Any, Any]", command * scale[:, np.newaxis])
