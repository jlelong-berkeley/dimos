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

"""Launch a multi-X500 PX4/Gazebo SITL swarm for DimOS to command over MAVLink."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np


class PX4SITLSwarmLauncher:
    """Launch PX4 SITL instances without commanding them."""

    def __init__(
        self,
        px4_dir: Path,
        n_drones: int = 3,
        instance_start: int = 0,
        model: str = "gz_x500",
        world: str = "dimos_grid",
        headless: bool = False,
        follow_camera: bool = False,
        spacing_m: float = 6.0,
        speed_factor: float = 1.0,
        clean: bool = True,
        first_spawn_delay_s: float = 4.0,
        spawn_delay_s: float = 2.0,
    ) -> None:
        self.px4_dir = px4_dir
        self.n_drones = n_drones
        self.instance_start = instance_start
        self.model = model
        self.world = world
        self.headless = headless
        self.follow_camera = follow_camera
        self.spacing_m = spacing_m
        self.speed_factor = speed_factor
        self.clean = clean
        self.first_spawn_delay_s = first_spawn_delay_s
        self.spawn_delay_s = spawn_delay_s
        self.processes: list[subprocess.Popen[bytes]] = []
        self._log_handles: list[Any] = []

    @property
    def build_dir(self) -> Path:
        """Return the PX4 SITL build directory."""
        return self.px4_dir / "build" / "px4_sitl_default"

    @property
    def px4_binary(self) -> Path:
        """Return the PX4 SITL binary path."""
        return self.build_dir / "bin" / "px4"

    def launch(self) -> None:
        """Launch Gazebo plus N PX4 instances."""
        if not self.px4_binary.exists():
            raise FileNotFoundError(f"PX4 SITL binary not found: {self.px4_binary}")

        log_dir = self.build_dir / "dimos_swarm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        base_env = self._px4_gazebo_env()
        positions = self._spawn_positions()

        for offset, pose in enumerate(positions):
            instance = self.instance_start + offset
            work_dir = self.build_dir / f"dimos_swarm_instance_{instance}"
            if self.clean and work_dir.exists():
                shutil.rmtree(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

            env = base_env.copy()
            env.update(
                {
                    "PX4_SYS_AUTOSTART": "4001",
                    "PX4_SIM_MODEL": self.model,
                    "PX4_GZ_WORLD": self.world,
                    "PX4_GZ_MODEL_POSE": f"{pose[0]},{pose[1]},0.0,0,0,0",
                    "PX4_GZ_NO_FOLLOW": "1",
                    "PX4_SIM_SPEED_FACTOR": str(self.speed_factor),
                    "GZ_IP": "127.0.0.1",
                }
            )
            if self.headless:
                env["HEADLESS"] = "1"
            if offset > 0:
                env["PX4_GZ_STANDALONE"] = "1"

            stdout = open(log_dir / f"px4_{instance}.out.log", "wb")
            stderr = open(log_dir / f"px4_{instance}.err.log", "wb")
            self._log_handles.extend([stdout, stderr])
            process = subprocess.Popen(
                [
                    str(self.px4_binary),
                    "-i",
                    str(instance),
                    "-d",
                    str(self.build_dir / "etc"),
                ],
                cwd=work_dir,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            self.processes.append(process)
            time.sleep(self.first_spawn_delay_s if offset == 0 else self.spawn_delay_s)

        self._configure_gui_camera()
        print("Launched PX4/Gazebo swarm:")
        for offset, pose in enumerate(positions):
            instance = self.instance_start + offset
            port = 14540 + instance if instance <= 9 else 14549
            print(f"  x500_{instance}: pose=({pose[0]:.1f}, {pose[1]:.1f}, 0.0), mavlink=udpin:127.0.0.1:{port}")
        print(f"Logs: {log_dir}")

    def stop(self) -> None:
        """Terminate launched PX4/Gazebo processes."""
        for process in self.processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=8.0)
                except Exception:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except Exception:
                        pass
        self.processes.clear()
        for handle in self._log_handles:
            handle.close()
        self._log_handles.clear()

    def _px4_gazebo_env(self) -> dict[str, str]:
        env_script = self.build_dir / "rootfs" / "gz_env.sh"
        if not env_script.exists():
            raise FileNotFoundError(f"Gazebo environment script not found: {env_script}")

        completed = subprocess.run(
            ["bash", "-lc", f"set -a && source {env_script} >/dev/null && env -0"],
            check=True,
            capture_output=True,
            env=os.environ.copy(),
        )
        env = os.environ.copy()
        for item in completed.stdout.split(b"\0"):
            if not item or b"=" not in item:
                continue
            key, value = item.split(b"=", 1)
            env[key.decode()] = value.decode()
        return env

    def _spawn_positions(self) -> np.ndarray[Any, Any]:
        y_values = (np.arange(self.n_drones) - ((self.n_drones - 1) / 2.0)) * self.spacing_m
        return np.column_stack((np.zeros(self.n_drones), y_values))

    def _configure_gui_camera(self) -> None:
        if self.headless or self.n_drones <= 0:
            return
        if not self.follow_camera:
            subprocess.run(
                ["gz", "topic", "-t", "/gui/track", "-m", "gz.msgs.CameraTrack", "-p", "track_mode: NONE"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        target_instance = self.instance_start + (self.n_drones // 2)
        model_name = self.model.removeprefix("gz_")
        payload = (
            "track_mode: FOLLOW_FREE_LOOK, "
            f"follow_target: {{name: '{model_name}_{target_instance}'}}, "
            "follow_offset: {x: -28.0, y: 0.0, z: 18.0}, "
            "follow_pgain: 0.7, track_pgain: 0.7"
        )
        subprocess.run(
            ["gz", "topic", "-t", "/gui/track", "-m", "gz.msgs.CameraTrack", "-p", payload],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px4-dir", type=Path, default=Path("droneWork/PX4-Autopilot"))
    parser.add_argument("--n-drones", type=int, default=3)
    parser.add_argument("--instance-start", type=int, default=0)
    parser.add_argument("--model", type=str, default="gz_x500")
    parser.add_argument("--world", type=str, default="dimos_grid")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--follow-camera", action="store_true", help="Track the middle drone in Gazebo")
    parser.add_argument("--spacing", type=float, default=6.0)
    parser.add_argument("--speed-factor", type=float, default=1.0)
    parser.add_argument("--no-clean", action="store_true", help="Reuse prior PX4 instance work directories")
    parser.add_argument("--first-spawn-delay", type=float, default=4.0)
    parser.add_argument("--spawn-delay", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    """Run the standalone launcher until interrupted."""
    args = parse_args()
    launcher = PX4SITLSwarmLauncher(
        px4_dir=args.px4_dir.resolve(),
        n_drones=args.n_drones,
        instance_start=args.instance_start,
        model=args.model,
        world=args.world,
        headless=args.headless,
        follow_camera=args.follow_camera,
        spacing_m=args.spacing,
        speed_factor=args.speed_factor,
        clean=not args.no_clean,
        first_spawn_delay_s=args.first_spawn_delay,
        spawn_delay_s=args.spawn_delay,
    )
    launcher.launch()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping PX4/Gazebo swarm...")
    finally:
        launcher.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
