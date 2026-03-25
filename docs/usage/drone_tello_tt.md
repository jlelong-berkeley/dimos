# RoboMaster TT / Tello Integration

This page documents the DimOS Tello integration and the compatibility constraints used to keep existing drone functionality intact.

## Overview

DimOS now includes a RoboMaster TT / Tello implementation that runs the main runtime on an edge computer and communicates with the drone over the DJI Tello UDP SDK.

High-level stack:

```text
DimOS (laptop / edge)
  -> TelloConnectionModule
  -> Tello UDP SDK (cmd/state/video)
  -> Optional EXT commands (TT expansion board)

Video/state
  -> DroneTrackingModule + CameraModule
  -> agent / teleop / visualization modules
```

## New Blueprints

- `drone-tello-tt-basic`
- `drone-tello-tt-agentic`
- `drone-tello-tt-fleet-agentic`
- `drone-tello-tt-gesture`

List and run:

```bash
dimos list
dimos --robot-ip 192.168.10.1 run drone-tello-tt-agentic --disable web-input
dimos --robot-ips 192.168.1.9,192.168.1.10 run drone-tello-tt-fleet-agentic
dimos --robot-ip 192.168.10.1 run drone-tello-tt-gesture
```

## Hardware + Network Connection Steps

For reliable Tello control plus cloud API access (OpenAI, optional Qwen, etc.), use two network paths:

1. Power on the RoboMaster TT / Tello and wait for SSID `TELLO-XXXXXX`.
2. Connect your primary Wi-Fi interface to `TELLO-XXXXXX` (this provides the `192.168.10.x` control link).
3. Keep internet on a second interface:
   - preferred: wired Ethernet (`eth0`), or
   - alternate: a second USB Wi-Fi adapter connected to home/office internet.
4. Verify routes:
   - Tello subnet exists on the Tello interface: `192.168.10.0/24`
   - default route points to your internet interface (not the Tello AP).
5. Verify connectivity:
   - `ping -c 1 192.168.10.1` (drone reachable)
   - `curl -I https://api.openai.com` (internet reachable)
6. Launch DimOS:
   - `dimos --robot-ip 192.168.10.1 run drone-tello-tt-agentic --disable web-input`

If you only have a single Wi-Fi radio, connecting to Tello usually removes internet access and cloud tool-calls will fail.

## Multi-Drone Fleet Mode

`drone-tello-tt-fleet-agentic` is the segmented multi-drone mode.

See [RoboMaster TT / Tello Fleet](/docs/usage/drone_tello_tt_fleet.md) for the full operator guide covering startup, IP discovery, viewer output, web UI, and available fleet commands.

- It supports one or many Tellos behind a single agent.
- Each drone is exposed as `drone-1`, `drone-2`, and so on, and can also be addressed by IP.
- The fleet web UI serves separate video feeds for every configured drone on `http://localhost:5555`.
- The agent can issue targeted actions such as `takeoff_drone(drone="drone-1")`.
- For "drone 1 do X while drone 2 does Y" requests, use `dispatch_segmented_plan(plan_json=...)` so each drone sequence runs in parallel.

Network requirement:

- Multi-drone mode works with one laptop IP.
- DimOS assigns a unique local command/state/video port set to each drone and uses the Tello SDK `port` command so telemetry and video remain separated per drone.
- `--robot-local-ips` is now optional. Use it only if you want to force a specific local bind IP or interface.

Example:

```bash
dimos \
  --robot-ips 192.168.1.9,192.168.1.10 \
  run \
  drone-tello-tt-fleet-agentic
```

Optional interface pinning:

```bash
dimos \
  --robot-ips 192.168.1.9,192.168.1.10 \
  --robot-local-ips 192.168.1.7 \
  run \
  drone-tello-tt-fleet-agentic
```

## New Modules

- `dimos.robot.drone.tello_sdk.TelloSdkClient`
  - Raw UDP command channel (`8889`)
  - State stream (`8890`)
  - Video stream (`11111`) decode + reconnect handling
- `dimos.robot.drone.tello_connection_module.TelloConnectionModule`
  - Skills: `takeoff`, `land`, `hover`, `move`, `move_relative`, `yaw`, `flip`, `rc`, `send_ext`, `follow_object`, `center_person_by_yaw`, `orbit_object`, `observe`
  - Publishes telemetry/status/odom/video and follow-command stream
- `dimos.robot.drone.tello_fleet_module.TelloFleetModule`
  - Multi-drone Tello SDK manager with per-drone skills, segmented-plan dispatch, and integrated multi-stream web UI
- `dimos.robot.drone.tello_gesture_control_module.TelloGestureControlModule`
  - Adapts `droneWork/tello-gesture-control` into a DimOS module
  - Reads the live Tello camera stream, publishes overlay/status, and issues manual override RC commands
  - Adds gesture-debounced `Forward`, `Back`, `Left`, `Right`, `Up`, `Down`, `Stop`, `Land`, plus dynamic clockwise / counter-clockwise yaw gestures

## Gesture Control Blueprint

`drone-tello-tt-gesture` composes on top of `drone-tello-tt-basic` and enables hand-gesture control by default.

Install the runtime dependencies first:

```bash
uv pip install mediapipe tensorflow
# or: uv pip install mediapipe tflite-runtime
```

Run it:

```bash
dimos --robot-ip 192.168.10.1 run drone-tello-tt-gesture
```

Gesture behavior:

- Hold a gesture steadily for roughly half a second to trigger it.
- `Up` takes off when grounded, and becomes an upward motion command once airborne.
- `Land` sends a zero-RC command and then lands immediately.
- Dynamic clockwise / counter-clockwise finger motions map to yaw commands.
- The recognized hand skeleton and resolved action are rendered to the existing `tracking_overlay` view.

## Gesture Control In `drone-tello-tt-agentic`

`drone-tello-tt-agentic` now also includes the gesture-control module, but it starts disabled.

Use natural language with the agent:

```text
enable gesture control
disable gesture control
```

Behavior in the agentic blueprint:

- The agent calls `enable_gesture_control()` only when you explicitly ask for it.
- Once enabled, the live Tello camera feed is used for gesture recognition and manual override RC commands.
- When disabled, the gesture module keeps its status path alive but does not command the drone.
- This keeps normal agent / tracking behavior intact until you opt into gestures.

## Tracking and Follow Behavior

`DroneTrackingModule` now supports both legacy and Tello-optimized behavior.

Compatibility defaults:

- `enable_passive_overlay=False`
- `use_local_person_detector=False`
- `force_detection_servoing_for_person=False`
- `person_follow_policy="legacy_pid"`

These defaults preserve existing `drone-agentic` behavior (Qwen + tracker path) unless a blueprint explicitly opts in.

Tello blueprint opt-in:

- `enable_passive_overlay=True`
- `use_local_person_detector=True`
- `force_detection_servoing_for_person=True`
- `person_follow_policy="yaw_forward_constant"`

This keeps Tello-specific behavior scoped to Tello blueprints.

### Hover Override

`hover()` is the Tello-side stop override for visual tracking.

- It publishes a stop command to `DroneTrackingModule`, so `follow_object()` and `center_person_by_yaw()` stop issuing new `cmd_vel` updates.
- It then sends zero-RC plus the native SDK `stop` command, which maps to hover in place.
- Use this when you want the drone to hold position and be ready for the next skill without landing.

### Flip Skill

`flip(direction)` exposes the native Tello SDK flip command.

- Supported directions: `forward`, `back`, `left`, `right`
- Single-letter aliases also work: `f`, `b`, `l`, `r`
- The drone must already be airborne and have enough free space around it

## Optional Local Detector/GPU Settings

Local YOLO person detection is optional and only used when enabled by module options.

Supported env overrides:

```bash
export DIMOS_DRONE_YOLO_DEVICE=cuda:0   # or cpu
export DIMOS_DRONE_YOLO_MODEL=yolo11s-pose.pt
export DIMOS_DRONE_YOLO_IMGSZ=416
export DIMOS_DRONE_YOLO_MAX_DET=5
```

Notes:

- CUDA is not required.
- If CUDA init fails, tracking falls back to CPU safely.
- No laptop-specific paths or machine-specific constants are required.

## Files Added

- `dimos/robot/drone/tello_sdk.py`
- `dimos/robot/drone/tello_connection_module.py`
- `dimos/robot/drone/tello_fleet_config.py`
- `dimos/robot/drone/tello_fleet_module.py`
- `dimos/robot/drone/tello_gesture_control_module.py`
- `dimos/robot/drone/blueprints/basic/drone_tello_tt_gesture.py`
- `dimos/robot/drone/blueprints/basic/drone_tello_tt_basic.py`
- `dimos/robot/drone/blueprints/agentic/drone_tello_tt_agentic.py`
- `dimos/robot/drone/blueprints/agentic/drone_tello_tt_fleet_agentic.py`

## Files Updated

- `dimos/robot/drone/drone_tracking_module.py`
  - Added configurable tracking modes for compatibility and Tello-specific behavior
  - Added optional local detector path and passive overlay
  - Added yaw-only centering and follow policies
- `dimos/robot/drone/__init__.py`
- `dimos/robot/drone/blueprints/__init__.py`
- `dimos/robot/drone/blueprints/basic/__init__.py`
- `dimos/robot/drone/blueprints/agentic/__init__.py`
- `dimos/robot/all_blueprints.py`
- `dimos/robot/drone/README.md`
- `pyproject.toml`

## Non-Regression Intent

The integration is designed so legacy drone stacks are unaffected unless they opt into Tello-oriented tracking options.

Validation commands:

```bash
uv run ruff check dimos/robot/drone/
uv run mypy dimos/robot/drone/
python3 -m py_compile dimos/robot/drone/drone_tracking_module.py
python3 -m py_compile dimos/robot/drone/tello_connection_module.py
python3 -m py_compile dimos/robot/drone/tello_fleet_module.py
python3 -m py_compile dimos/robot/drone/tello_gesture_control_module.py
uv run pytest dimos/robot/drone/test_tello_connection_module.py -v
uv run pytest dimos/robot/drone/test_tello_fleet_module.py -v
uv run pytest dimos/robot/drone/test_tello_gesture_control_module.py -v
```
