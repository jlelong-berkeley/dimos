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

"""Bridge SiK telemetry radios to local MAVLink UDP endpoints for the PX4 hardware swarm.

The DimOS PX4 swarm module connects to one ``udpin:127.0.0.1:14540+i`` endpoint per drone,
exactly mirroring the SITL port layout. On real hardware each X500 is reached through its own
USB SiK telemetry radio, so this launcher runs one ``mavlink-routerd`` bridge per radio that:

* reads one SiK serial device (pinned by a stable ``/dev/serial/by-id/...`` path),
* forwards that vehicle to ``udpin:127.0.0.1:14540+i`` for DimOS, and
* forwards the same vehicle to a shared GCS endpoint (default ``127.0.0.1:14550``) so
  QGroundControl can attach in parallel.

Running one bridge per radio keeps each vehicle isolated on its own DimOS UDP port without
relying on per-endpoint sysid filters.

Hardware notes:

* Differentiate radios by stable ``/dev/serial/by-id/...`` paths, never ``/dev/ttyUSB*``
  (which renumber across reboots) and never by baud.
* Give each SiK *pair* a distinct ``NETID`` and each flight controller a distinct
  ``MAV_SYS_ID``.
* Requires ``mavlink-routerd`` on PATH: https://github.com/mavlink-router/mavlink-router

Example::

    python -m dimos.robot.drone.px4_mavlink_router \\
        --device /dev/serial/by-id/usb-FTDI_FT231X_USB_UART_D0001-if00-port0 \\
        --device /dev/serial/by-id/usb-FTDI_FT231X_USB_UART_D0002-if00-port0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

DEFAULT_BAUD = 57600
DEFAULT_UDP_BASE_PORT = 14540
DEFAULT_GCS_HOST = "127.0.0.1"
DEFAULT_GCS_PORT = 14550
ROUTER_BINARY = "mavlink-routerd"


class PX4MavlinkRouterLauncher:
    """Run one mavlink-router bridge per SiK radio (serial -> DimOS UDP + QGC UDP)."""

    def __init__(
        self,
        devices: list[str],
        baud: int = DEFAULT_BAUD,
        udp_base_port: int = DEFAULT_UDP_BASE_PORT,
        gcs_host: str = DEFAULT_GCS_HOST,
        gcs_port: int = DEFAULT_GCS_PORT,
        config_dir: Path | None = None,
    ) -> None:
        if not devices:
            raise ValueError("At least one serial device path is required")
        self.devices = devices
        self.baud = baud
        self.udp_base_port = udp_base_port
        self.gcs_host = gcs_host
        self.gcs_port = gcs_port
        self.config_dir = config_dir or Path(tempfile.mkdtemp(prefix="px4_mavlink_router_"))
        self.processes: list[subprocess.Popen[bytes]] = []
        self._log_handles: list[Any] = []

    def dimos_port(self, index: int) -> int:
        """Return the DimOS UDP port for the radio at the given index."""
        return self.udp_base_port + index

    def render_config(self, index: int, device: str) -> str:
        """Render one mavlink-router config bridging a single radio.

        Both UDP endpoints use ``Mode=Normal`` (client): mavlink-router sends the vehicle to
        whoever is listening on those ports. DimOS binds them with ``udpin:`` and QGroundControl
        binds the shared GCS port, so replies route back to each peer automatically.
        """
        dimos_port = self.dimos_port(index)
        return (
            "[General]\n"
            "ReportStats=false\n"
            "\n"
            f"[UartEndpoint vehicle_{index}]\n"
            f"Device={device}\n"
            f"Baud={self.baud}\n"
            "\n"
            f"[UdpEndpoint dimos_{index}]\n"
            "Mode=Normal\n"
            "Address=127.0.0.1\n"
            f"Port={dimos_port}\n"
            "\n"
            f"[UdpEndpoint gcs_{index}]\n"
            "Mode=Normal\n"
            f"Address={self.gcs_host}\n"
            f"Port={self.gcs_port}\n"
        )

    def launch(self) -> None:
        """Write per-radio configs and spawn one mavlink-routerd bridge each."""
        if shutil.which(ROUTER_BINARY) is None:
            raise FileNotFoundError(
                f"{ROUTER_BINARY!r} not found on PATH. Install mavlink-router: "
                "https://github.com/mavlink-router/mavlink-router"
            )
        self.config_dir.mkdir(parents=True, exist_ok=True)
        print("Launching PX4 SiK MAVLink router bridges:")
        for index, device in enumerate(self.devices):
            config_path = self.config_dir / f"router_{index}.conf"
            config_path.write_text(self.render_config(index, device))
            stdout = open(self.config_dir / f"router_{index}.out.log", "wb")
            stderr = open(self.config_dir / f"router_{index}.err.log", "wb")
            self._log_handles.extend([stdout, stderr])
            process = subprocess.Popen(
                [ROUTER_BINARY, "-c", str(config_path)],
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            self.processes.append(process)
            time.sleep(0.5)
            if process.poll() is not None:
                detail = ""
                try:
                    detail = (
                        (self.config_dir / f"router_{index}.err.log")
                        .read_text(errors="ignore")
                        .strip()
                    )
                except Exception:
                    pass
                self.stop()
                raise RuntimeError(
                    f"mavlink-routerd for {device} exited immediately (rc={process.returncode}). "
                    f"The serial port is most likely already open by another bridge instance, "
                    f"QGroundControl, or a daemon (ModemManager/brltty) -> 'Device or resource "
                    f"busy'. Router log:\n{detail or '(empty)'}"
                )
            print(
                f"  radio[{index}] {device} (baud {self.baud}) -> "
                f"udpin:127.0.0.1:{self.dimos_port(index)} (DimOS), "
                f"{self.gcs_host}:{self.gcs_port} (QGC)"
            )
        print(f"Configs and logs: {self.config_dir}")
        print(
            "DimOS connection strings: "
            + ", ".join(
                f"udpin:127.0.0.1:{self.dimos_port(i)}" for i in range(len(self.devices))
            )
        )

    def stop(self) -> None:
        """Terminate all mavlink-routerd bridges."""
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--device",
        dest="devices",
        action="append",
        default=[],
        help="Serial path of a SiK radio (repeatable, one per drone). Prefer /dev/serial/by-id/...",
    )
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--udp-base-port", type=int, default=DEFAULT_UDP_BASE_PORT)
    parser.add_argument("--gcs-host", type=str, default=DEFAULT_GCS_HOST)
    parser.add_argument("--gcs-port", type=int, default=DEFAULT_GCS_PORT)
    return parser.parse_args()


def main() -> int:
    """Run the router bridges until interrupted."""
    args = parse_args()
    if not args.devices:
        print("error: at least one --device is required", file=sys.stderr)
        return 2
    launcher = PX4MavlinkRouterLauncher(
        devices=args.devices,
        baud=args.baud,
        udp_base_port=args.udp_base_port,
        gcs_host=args.gcs_host,
        gcs_port=args.gcs_port,
    )
    launcher.launch()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping MAVLink router bridges...")
    finally:
        launcher.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
