# Hand Tracking Visualizer

A real-time 3D visualization tool for PICO VR hand tracking data. Renders live skeletal overlays directly from the UDP stream — no Isaac Sim required. Use it to verify the full data pipeline (APK → UDP → RealMirror) is healthy before starting a teleoperation session.

![skeleton preview](../../docs/image/HandTrackShow.gif)

---

## Features

- **Live 3D skeleton** — joint scatter + bone connections for both hands simultaneously (left: blue, right: red)
- **Wrist orientation axes** — magenta X, lime Y, cyan Z arrows to validate quaternion correctness
- **Zero Isaac Sim dependency** — runs in any standard Python environment
- **Configurable** — port, bind address, and refresh rate via CLI flags

---

## Requirements

Python 3.11+ with:

```bash
pip install matplotlib numpy
```

---

## Quick Start

```bash
# Default settings — bind 0.0.0.0:8090, ~30 fps
python tools/hand_tracking_visualizer/visualize_hand_tracking.py

# Custom port
python tools/hand_tracking_visualizer/visualize_hand_tracking.py --port 8090

# Lower refresh rate (e.g. on slower machines)
python tools/hand_tracking_visualizer/visualize_hand_tracking.py --interval 50
```

---

## Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | UDP bind address |
| `--port` | `8090` | UDP port — must match the target port set in the PICO APK |
| `--interval` | `33` | Animation refresh interval in milliseconds (33 ms ≈ 30 fps) |

---

## Prerequisites

Before running, ensure:

1. The RealMirror APK is installed and running on the PICO headset
2. The headset and this machine are on the same local network
3. The APK's target IP is set to this machine's IP, and the port matches `--port`

---

## Verifying the Pipeline

On startup you should see:

```
[HandTrackingVisualizer] Listening on 0.0.0.0:8090 ...
```

Put on the headset and raise both hands. The Matplotlib window should show blue (left) and red (right) skeletal overlays, with three colored axes at each wrist that rotate as you move your hands.

**Nothing appearing?** Check:
- Network connectivity between headset and machine (`ping <machine-ip>` from the headset side)
- APK target IP matches this machine's IP on the shared network interface
- No firewall blocking UDP on the configured port

---

## Packet Format

The visualizer decodes the same binary UDP packets used by `hand_tracking/udp_server.py`:

```
1 byte   hand_id        (0 = left, 1 = right)
26 × 28  joint data     (per joint: pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w — 7 × float32 LE)
─────────────────────────────────────────────────────
Total minimum: 729 bytes per packet
```

Joint indices follow the PICO SDK convention (26 joints/hand). Wrist is index 1 and is used for orientation visualization.
