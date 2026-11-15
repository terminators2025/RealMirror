# Teleoperation Architecture

This page describes how a teleoperation session is assembled: the two input paths, which module owns each layer, and the file contract between XR-Linker and the simulator. Step-by-step setup lives in the per-mode guides:

- PICO VR controller — [xr_linker_quik_start.md](xr_linker_quik_start.md)
- Bare-hand gestures — [hand_tracking_quick_start.md](hand_tracking_quick_start.md)

## Two input paths, one execution stack

```
Path A — VR controller
  PICO headset browser (WebXR)  ──HTTPS / Socket.IO──►  XR-Linker (Node.js, PC)
                                                              │ atomic write
                                                              ▼
                                                   data/live_pico_data.json
                                                              │ file polling
                                                              ▼
                                  PicoController  →  A2PicoMapping  ─┐
                                                                     │
Path B — bare-hand gesture                                           │
  RealMirror PICO APK  ──UDP :8090──►  HandTrackUDPServer            │
                                          │                          │
                       HandTrackController  →  A2HandTrackMapping ───┤
                                                                     ▼
                       TeleopManager  →  IKSolver  →  AgiBot A2 robot
                       SlowHomingManager / DataRecorder (HDF5)
```

## Path A — PICO VR controller

1. The headset browser runs the WebXR page served by XR-Linker and emits `controller-data` frames over HTTPS/Socket.IO.
2. XR-Linker validates each frame and replaces `data/live_pico_data.json` atomically (write `.tmp`, then rename), so a reader never observes a partial frame.
3. `PicoController` polls that file and exposes pose, buttons and axes; `A2PicoMapping` turns them into A2 joint-space targets.

The hand-off is a file rather than a socket, so both processes must agree on the same path: `--pico-json-path` on the simulator side defaults to `tools/xr_linker/data/live_pico_data.json`, while XR-Linker resolves `DATA_STORAGE_DIR` relative to its working directory. Start XR-Linker from `tools/xr_linker/`, or override both sides explicitly.

## Path B — bare-hand gestures

1. The RealMirror PICO APK streams 26-DOF hand joints to `HandTrackUDPServer` over UDP (default port `8090`).
2. The server parses each packet, drops malformed or non-finite frames, and keeps the latest snapshot per hand in memory.
3. `HandTrackController` reads that snapshot directly — no file I/O on the hot path — and derives pinch state, wrist orientation and position-offset calibration; `A2HandTrackMapping` supplies the gripper action.

`HandTrackUDPServer` can also mirror frames to a JSON file when it is constructed with an output path, but the controller still prefers the in-memory snapshot.

## Frame validation

Neither path lets a bad frame reach the robot:

- the UDP parser drops any packet whose joints contain `NaN` or `Inf`;
- `PicoController` and `HandTrackController` return no pose when a position or quaternion is non-finite or has the wrong arity;
- `TeleopManager` re-checks the controller pose and the offset-corrected target before calling `set_world_pose`.

A rejected frame never updates the jump filter's reference point, so one bad sample cannot poison the next comparison.

## The shared file contract

Only Path A uses a file. `data/live_pico_data.json` always holds a single frame with the required `timestamp` and `data` fields, plus an optional server-generated `_latency` block used for latency statistics. The authoritative field-by-field schema, including which fields are persisted and which stay in the in-memory cache, is in [the XR-Linker README](../tools/xr_linker/README.md#data-contract).

## Source layout

| Layer | Module | Responsibility |
|---|---|---|
| Path A input | `tools/xr_linker/` | WebXR capture (client) + HTTPS/Socket.IO server + atomic persistence |
| Path A adapter | `controllers/pico_controller.py` | Polls the shared JSON, exposes pose / buttons / axes |
| Path B input | `hand_tracking/udp_server.py`, `hand_tracking/hand_data.py` | UDP receiver, thread-safe snapshot, 26-joint data model |
| Path B adapter | `controllers/hand_track_controller.py` | Pinch detection, wrist orientation, offset calibration |
| Mapping | `controllers/control_mapping.py` | `A2PicoMapping` / `A2HandTrackMapping` to A2 joint space |
| Execution | `teleoperation/teleop_manager.py`, `ik_solver.py`, `slow_homing_manager.py` | Target updates, Lula IK with 30° jump filtering, smooth homing |
| Recording | `data_recording/recorder.py` | Command-as-Action 26-DOF HDF5 capture |
| Entry point | `script/teleop.py` | Argument parsing and assembly of either path |

## Selecting a path

`script/teleop.py` assembles exactly one path per session, chosen by `--controller-type`: `vr_controller` (default) wires up the XR-Linker path, `hand_track` starts the in-process UDP server. Both then share the identical mapping → manager → IK → recording pipeline, so running both inputs at once would be a change to the entry point rather than to the execution layer.

Recorded episodes are always HDF5 in LeRobot-convertible form, so a machine without a headset can still train and evaluate.

## When the robot does not move

Check these in order — the first three account for most reports:

1. `--controller-type` matches the hardware you are actually using.
2. For Path A, `data/live_pico_data.json` is being rewritten while the headset is connected, and both processes point at the same path.
3. For Path B, the APK's `server_config.txt` holds the workstation IP and the UDP port matches `--udp-port` (default `8090`).
4. Teleoperation is enabled for the arm — press `T` (right) or `G` (left).
5. The simulation log shows no repeated frame rejections caused by non-finite or malformed input.
