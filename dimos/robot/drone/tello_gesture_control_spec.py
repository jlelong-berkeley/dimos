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

"""Spec for the RPC surface used by Tello gesture control."""

from __future__ import annotations

from typing import Any, Protocol

from dimos.spec.utils import Spec


class TelloGestureControlSpec(Spec, Protocol):
    """RPC surface required by the gesture-control module."""

    def send_manual_rc(
        self,
        left_right: int = 0,
        forward_back: int = 0,
        up_down: int = 0,
        yaw: int = 0,
        manual_override_sec: float = 0.35,
    ) -> bool: ...

    def takeoff(self, altitude: float = 1.0) -> str: ...

    def land(self) -> str: ...

    def get_status(self) -> dict[str, Any]: ...
