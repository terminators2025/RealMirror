# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
"""PICO hand tracking real-time visualizer.

Receives 26-DOF hand joint data from the PICO VR headset via UDP and renders
a live 3D skeleton visualization using Matplotlib. Use this tool to verify that
the hand tracking data pipeline (APK -> UDP -> RealMirror) is functioning
correctly before starting a teleoperation session.

Usage:
    python tools/hand_tracking_visualizer/visualize_hand_tracking.py
    python tools/hand_tracking_visualizer/visualize_hand_tracking.py --port 8090
    python tools/hand_tracking_visualizer/visualize_hand_tracking.py --port 8090 --interval 33

Requirements:
    pip install matplotlib numpy

The PICO VR headset must be running the RealMirror APK and connected to the
same network as this machine. Default UDP port is 8090.
"""

import argparse
import socket
import struct
import threading
from typing import Dict, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection

# ---------------------------------------------------------------------------
# Joint and skeleton definitions (PICO SDK, 26 joints per hand)
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    "Palm",
    "Wrist",
    "ThumbMetacarpal",
    "ThumbProximal",
    "ThumbDistal",
    "ThumbTip",
    "IndexMetacarpal",
    "IndexProximal",
    "IndexIntermediate",
    "IndexDistal",
    "IndexTip",
    "MiddleMetacarpal",
    "MiddleProximal",
    "MiddleIntermediate",
    "MiddleDistal",
    "MiddleTip",
    "RingMetacarpal",
    "RingProximal",
    "RingIntermediate",
    "RingDistal",
    "RingTip",
    "LittleMetacarpal",
    "LittleProximal",
    "LittleIntermediate",
    "LittleDistal",
    "LittleTip",
]

# Bone connections as (parent_index, child_index) pairs.
# Index 1 (Wrist) is the root of all fingers.
HAND_BONES = [
    (0, 1),   # Palm -> Wrist
    (1, 2),   (2, 3),   (3, 4),   (4, 5),    # Thumb chain
    (1, 6),   (6, 7),   (7, 8),   (8, 9),    (9, 10),   # Index chain
    (1, 11),  (11, 12), (12, 13), (13, 14),  (14, 15),  # Middle chain
    (1, 16),  (16, 17), (17, 18), (18, 19),  (19, 20),  # Ring chain
    (1, 21),  (21, 22), (22, 23), (23, 24),  (24, 25),  # Little chain
]

# Wrist joint index used for orientation visualization
_WRIST_INDEX = 1

# Hand ID -> display color / label
_HAND_COLORS = {0: "blue", 1: "red"}
_HAND_LABELS = {0: "Left", 1: "Right"}

# Axis colors for wrist orientation visualization
_AXIS_COLORS = {"x": "magenta", "y": "lime", "z": "cyan"}
_AXIS_LENGTH = 0.05


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------


def _quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix.

    Args:
        q: Quaternion as [x, y, z, w].

    Returns:
        3x3 rotation matrix.
    """
    x, y, z, w = q
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)],
        ]
    )


# ---------------------------------------------------------------------------
# UDP listener
# ---------------------------------------------------------------------------

# Packet layout: 1 byte hand_id + 26 joints x 7 floats (pos xyz + quat xyzw)
_PACKET_MIN_SIZE = 1 + 26 * 7 * 4  # 729 bytes


class HandTrackingVisualizer:
    """Receives PICO hand tracking data via UDP and renders a live 3D view.

    Args:
        host: UDP bind address.
        port: UDP port to listen on (default 8090).
        interval_ms: Matplotlib animation refresh interval in milliseconds.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8090, interval_ms: int = 33) -> None:
        self._host = host
        self._port = port
        self._interval_ms = interval_ms

        # Shared state: hand_id -> (positions [26,3], rotations [26,4])
        self._hand_poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
            0: (np.zeros((26, 3)), np.zeros((26, 4))),
            1: (np.zeros((26, 3)), np.zeros((26, 4))),
        }
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # UDP receive loop
    # ------------------------------------------------------------------

    def _parse_packet(self, raw: bytes) -> None:
        """Parse one UDP packet and update shared hand pose state.

        Packet format (little-endian):
            1 byte  : hand_id (0 = left, 1 = right)
            26 x 7 floats: per-joint [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]

        Args:
            raw: Raw bytes from the UDP socket.
        """
        if len(raw) < _PACKET_MIN_SIZE:
            return

        hand_id = struct.unpack_from("<B", raw, 0)[0]
        if hand_id not in (0, 1):
            return

        positions = np.zeros((26, 3), dtype=np.float64)
        rotations = np.zeros((26, 4), dtype=np.float64)

        offset = 1
        for i in range(26):
            values = struct.unpack_from("<7f", raw, offset)
            offset += 28  # 7 floats x 4 bytes
            positions[i] = values[0:3]
            rotations[i] = values[3:7]

        with self._lock:
            self._hand_poses[hand_id] = (positions, rotations)

    def _udp_listener(self) -> None:
        """Background thread: receive UDP packets and update shared state."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self._host, self._port))
            sock.settimeout(1.0)
            print(f"[HandTrackingVisualizer] Listening on {self._host}:{self._port} ...")

            while True:
                try:
                    data, _ = sock.recvfrom(2048)
                    self._parse_packet(data)
                except socket.timeout:
                    continue
                except struct.error as exc:
                    print(f"[HandTrackingVisualizer] Packet parse error: {exc}")
                except OSError:
                    break

    # ------------------------------------------------------------------
    # Matplotlib animation
    # ------------------------------------------------------------------

    def _update_plot(self, _frame: int, ax: "Axes3D") -> None:  # type: ignore[name-defined]
        """Matplotlib FuncAnimation callback: redraw the skeleton each frame.

        Args:
            _frame: Frame index (unused).
            ax: The 3D axes to draw on.
        """
        ax.cla()

        # Fixed viewport sized to typical hand workspace (~20 cm)
        ax.set_xlim(-0.10, 0.10)
        ax.set_ylim(0.00, 0.15)
        ax.set_zlim(-0.10, 0.10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("RealMirror Hand Tracking Visualizer")
        ax.view_init(elev=15.0, azim=-90.0)

        with self._lock:
            local_poses = {k: (v[0].copy(), v[1].copy()) for k, v in self._hand_poses.items()}

        for hand_id, (positions, rotations) in local_poses.items():
            if not np.any(positions):
                continue  # No data received yet for this hand

            color = _HAND_COLORS[hand_id]
            label = _HAND_LABELS[hand_id]

            # Coordinate transform: flip Z axis for correct viewing direction,
            # then fix the mirror artifact on quaternion components.
            positions[:, 2] *= -1
            rotations[:, 0] *= -1  # qx
            rotations[:, 1] *= -1  # qy

            # Joint scatter
            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color=color,
                label=label,
                s=30,
            )

            # Bone lines
            for parent_idx, child_idx in HAND_BONES:
                p0 = positions[parent_idx]
                p1 = positions[child_idx]
                ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    [p0[2], p1[2]],
                    color=color,
                    linewidth=2,
                )

            # Wrist orientation axes (validate quaternion data is non-zero)
            wrist_pos = positions[_WRIST_INDEX]
            wrist_quat = rotations[_WRIST_INDEX]
            if np.any(wrist_quat):
                rot_matrix = _quat_to_rotation_matrix(wrist_quat)
                for col, axis_color in zip(range(3), _AXIS_COLORS.values()):
                    axis_vec = rot_matrix[:, col]
                    ax.quiver(
                        wrist_pos[0],
                        wrist_pos[1],
                        wrist_pos[2],
                        axis_vec[0],
                        axis_vec[1],
                        axis_vec[2],
                        length=_AXIS_LENGTH,
                        color=axis_color,
                        normalize=True,
                    )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the UDP listener thread and launch the Matplotlib window."""
        listener = threading.Thread(target=self._udp_listener, daemon=True)
        listener.start()

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        ani = animation.FuncAnimation(  # noqa: F841 - must be kept alive
            fig,
            self._update_plot,
            fargs=(ax,),
            interval=self._interval_ms,
            cache_frame_data=False,
        )

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RealMirror PICO hand tracking real-time visualizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="UDP bind address.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="UDP port to listen on (must match the PICO APK setting).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=33,
        dest="interval_ms",
        help="Matplotlib animation refresh interval in milliseconds (~30 fps).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()
    visualizer = HandTrackingVisualizer(
        host=args.host,
        port=args.port,
        interval_ms=args.interval_ms,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
