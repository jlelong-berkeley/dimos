# RoboMaster TT / Tello Fleet Mode

This guide covers the `drone-tello-tt-fleet-agentic` blueprint: how to find drone IPs, start the fleet stack, view the camera outputs, and use the available fleet commands.

## When To Use This Mode

Use fleet mode when one DimOS agent should control one or more RoboMaster TT / Tello drones by target name:

- `drone-1`, `drone-2`, and so on, assigned in the same order as `--robot-ips`.
- The drone IP address itself, such as `192.168.0.164`.

Fleet mode is the right entry point for prompts like:

```text
take off with drone 1 and hover
follow the person with both drones
drone 1 move left while drone 2 rotates right
```

## Network Setup

For multiple Tellos, put the drones and the DimOS laptop on the same Wi-Fi network or LAN. The drone IPs should be router-assigned addresses such as `192.168.0.164`, not the default single-drone AP address `192.168.10.1`.

To find the drone IPs:

1. Power on each Tello / RoboMaster TT and connect it to the shared Wi-Fi network.
2. Open the Wi-Fi router or access point admin page.
3. Go to the connected-clients, DHCP leases, or attached-devices page.
4. Look for the Tello / RoboMaster TT devices and note their exposed IP addresses.
5. Keep the order stable when launching DimOS. The first IP becomes `drone-1`, the second becomes `drone-2`.

Example:

```text
192.168.0.164 -> drone-1
192.168.0.133 -> drone-2
```

Before launch, verify basic reachability:

```bash
ping -c 1 192.168.0.164
ping -c 1 192.168.0.133
```

If you only connect directly to one drone's `TELLO-XXXXXX` access point, that drone normally appears as `192.168.10.1`. That setup is fine for solo mode, but it is not the normal fleet setup because multiple Tellos would all use the same default IP.

## Start The Fleet

Run the fleet blueprint with a comma-separated `--robot-ips` list:

```bash
dimos --robot-ips 192.168.0.164,192.168.0.133 run drone-tello-tt-fleet-agentic
```

Optional: force a specific local laptop interface/IP if the machine has multiple network paths and auto-binding is not choosing the right one:

```bash
dimos \
  --robot-ips 192.168.0.164,192.168.0.133 \
  --robot-local-ips 192.168.0.42 \
  run \
  drone-tello-tt-fleet-agentic
```

The blueprint opens the Rerun viewer by default when `GlobalConfig.viewer` is `rerun` or `rerun-web`.

Useful runtime commands:

```bash
dimos status
dimos log -f
dimos humancli
dimos agent-send "list the drones"
dimos stop
```

## Output And Interfaces

Fleet mode exposes three operator-facing surfaces:

- Rerun viewer: opens at startup and shows `Fleet Feeds` plus `Tracking Overlay`.
- Fleet web UI: served at `http://localhost:5555` with per-drone camera feeds and agent interaction.
- CLI / logs: `dimos humancli`, `dimos agent-send`, and `dimos log -f`.

Rerun views:

- `Fleet Feeds`: labeled grid of raw drone camera feeds.
- `Tracking Overlay`: labeled grid with the person/object detection boxes, target centers, and search overlays.

The tracking overlay may initially show `Overlay pending` until the first camera frames and detector pass arrive. During `follow_object_drone` or `center_person_by_yaw_drone`, it should show bounding boxes when the local detector sees a person.

## Available Fleet Skills

The agent sees these skills from `TelloFleetModule`.

| Skill | Purpose |
|-------|---------|
| `list_drones()` | Print configured drones, IPs, ports, video readiness, and battery. |
| `observe_drone(drone=...)` | Return the latest frame from one drone. |
| `observe_fleet()` | Return a labeled grid of all latest frames. |
| `takeoff_drone(drone=..., altitude=1.0)` | Take off one drone. Tello hovers automatically after takeoff. |
| `land_drone(drone=...)` | Land one drone immediately. |
| `hover_drone(drone=...)` | Stop motion/tracking and hold position without landing. |
| `move_drone(drone=..., x=..., y=..., z=..., duration=...)` | Body-frame velocity control in meters per second. |
| `move_relative_drone(drone=..., x=..., y=..., z=..., speed=...)` | Relative position move in meters. |
| `yaw_drone(drone=..., degrees=...)` | Rotate one drone. Positive is counter-clockwise. |
| `rc_drone(drone=..., left_right=..., forward_back=..., up_down=..., yaw=..., duration=...)` | Low-level Tello RC channel command in `[-100, 100]`. Use cautiously. |
| `flip_drone(drone=..., direction=...)` | Flip one drone: `forward`, `back`, `left`, or `right`. |
| `send_ext_drone(drone=..., ext_command=...)` | Send a RoboMaster TT expansion-board `EXT` command. |
| `follow_object_drone(drone=..., object_description="person", distance_m=1.0, duration=120.0, scan_step_deg=30.0, max_scan_steps=12, control_mode="full")` | Scan for and follow a person/object. Person follow uses the TT TOF sensor when valid. |
| `center_person_by_yaw_drone(drone=..., duration=120.0, scan_step_deg=0.0, max_scan_steps=1)` | Hover and yaw to keep a detected person centered. |
| `stop_tracking_drone(drone=...)` | Stop active follow/yaw-centering for one drone. |
| `emergency_stop_drone(drone=...)` | Send Tello emergency stop. Motors stop immediately. |
| `dispatch_segmented_plan(plan_json=...)` | Run separate per-drone sequences in parallel. |

## Example Agent Prompts

Basic checks:

```text
list the drones
show me the fleet
what is the battery for drone 1?
```

Takeoff and hover:

```text
take off with drone 1
take off with both drones and hover
```

Movement:

```text
move drone 1 up half a meter
rotate drone 2 clockwise 45 degrees
move drone 1 forward slowly for 2 seconds
```

Tracking:

```text
follow the person with both drones
make drone 1 center the person by yaw only
stop tracking with drone 2
```

Coordinated fleet command:

```text
drone 1 move left while drone 2 rotates right
```

For explicit structured dispatch, the agent can call:

```json
{
  "commands": [
    {"drone": "drone-1", "action": "takeoff"},
    {"drone": "drone-1", "action": "move_relative", "y": 0.5, "speed": 0.3},
    {"drone": "drone-2", "action": "takeoff"},
    {"drone": "drone-2", "action": "yaw", "degrees": -45}
  ]
}
```

Supported `dispatch_segmented_plan` actions are:

```text
takeoff, land, hover, move, move_relative, yaw, rc, flip, send_ext,
follow_object, center_person_by_yaw, stop_tracking, emergency_stop
```

## Tello-Specific Notes

- Tello auto-lands if it receives no command for roughly 15 seconds after takeoff. The DimOS SDK client sends keepalive commands while airborne so idle hover should not trigger this.
- `takeoff_drone` already leaves the drone hovering. Calling `hover_drone` immediately after takeoff is usually unnecessary.
- Multi-drone video separation depends on each drone's support for the Tello SDK `port` command. The fleet module negotiates port assignments and filters video by source IP, but some firmware rejects custom ports.
- If one drone rejects custom stream ports, DimOS may assign that drone to the default state/video ports and move a compatible drone to custom ports.
- Use `land_drone` for normal shutdown and `emergency_stop_drone` only when immediate motor stop is needed.

## Troubleshooting

If one drone connects but video is poor or corrupted:

- Confirm no phone app or old DimOS process is also streaming from the same Tello.
- Check `dimos log -f` for `unexpected sender` or H.264 decode errors.
- Make sure the `--robot-ips` list matches the actual router-assigned drone IPs.

If the Rerun `Tracking Overlay` view is black:

- Wait until both video streams show first frames.
- Prompt: `follow the person with drone 1`.
- Check `dimos log -f` for `YOLO person detection bbox=...` or `YOLO detector found no person`.
- If the raw camera feed is live but the overlay remains empty, restart the blueprint and verify the viewer version warning is not masking display issues.

If the agent reports `takeoff rejected` but the drone lifted off:

- Run `list_drones` or check the Rerun feed. The SDK uses telemetry height to recover from missed takeoff acknowledgements, but poor Wi-Fi can still delay status.
- Avoid sending `hover_drone` immediately after `takeoff_drone`; takeoff already enters hover.

If the agent cannot see or follow a person:

- Check the `Tracking Overlay` view first. If no box appears, improve lighting, distance, or camera angle.
- For close-range follow, keep the person fully inside the frame. Partial bodies at frame edges are harder to confirm.
- Use `center_person_by_yaw_drone` to debug yaw-only tracking before enabling full follow.
