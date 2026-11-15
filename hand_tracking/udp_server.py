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
"""UDP server for receiving PICO hand tracking data.

Receives 26-DOF hand joint data via UDP, applies coordinate transformations,
and writes JSON output for the HandTrackController to read.

Aligned with hand_track_opensource_project/realmirror_sever_data.py.
"""

import json
import socket
import struct
import threading
import time
from typing import Optional

import numpy as np

from utility.logger import Logger

_JOINTS_PER_HAND = 26
_FLOATS_PER_JOINT = 7
_PACKET_HEADER_SIZE = 1
_PACKET_MIN_SIZE = _PACKET_HEADER_SIZE + _JOINTS_PER_HAND * _FLOATS_PER_JOINT * 4


class HandTrackUDPServer:
    """UDP server that receives PICO hand tracking data.

    Parsed hand data is available via thread-safe in-memory access
    (``get_latest_snapshot()``) and optionally written to a JSON file
    for external consumers.

    Args:
        output_path: File path for JSON output (live_hand_data.json).
            Pass ``None`` to skip file I/O entirely.
        port: UDP port to listen on (default 8090).
        host: Bind address (default '0.0.0.0').
    """

    def __init__(self, output_path: Optional[str], port: int = 8090, host: str = "0.0.0.0"):
        self.output_path = output_path
        self.port = port
        self.host = host
        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._data_lock = threading.Lock()
        self._latest_data: list = []

        # Thread-safe snapshot for in-memory consumers (no file I/O needed).
        # Dict format matches the JSON structure: {"timestamp": int, "data": [...]}.
        self._snapshot: Optional[dict] = None

    def start(self) -> None:
        """Start the UDP server in a background thread."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        Logger.info("HandTrackUDPServer started on %s:%d", self.host, self.port)

    def stop(self) -> None:
        """Stop the UDP server."""
        self._running = False
        if self._socket:
            self._socket.close()
        if self._thread:
            self._thread.join(timeout=2.0)
        Logger.info("HandTrackUDPServer stopped.")

    def get_latest_snapshot(self) -> Optional[dict]:
        """Return the latest parsed hand data dict (thread-safe, zero-copy read).

        Returns:
            Dict with 'timestamp' and 'data' keys matching JSON structure,
            or ``None`` if no data has been received yet.
        """
        with self._data_lock:
            return self._snapshot

    def _run(self) -> None:
        """Main receive loop running in background thread."""
        while self._running:
            if self._socket is None:
                break
            try:
                data, _ = self._socket.recvfrom(2048)
                self._process_packet(data)
            except socket.timeout:
                continue
            except OSError:
                break

    def _process_packet(self, raw_data: bytes) -> None:
        """Process a single UDP packet and update in-memory snapshot.

        Optionally writes JSON to disk if ``output_path`` is set.

        Args:
            raw_data: Raw bytes from UDP packet.
        """
        parsed = self._parse_packet(raw_data)
        if parsed is None:
            return

        with self._data_lock:
            updated = False
            for i, existing in enumerate(self._latest_data):
                if existing["handedness"] == parsed["handedness"]:
                    self._latest_data[i] = parsed
                    updated = True
                    break
            if not updated:
                self._latest_data.append(parsed)

            output = {"timestamp": parsed["timestamp"], "data": list(self._latest_data)}
            self._snapshot = output

        # File I/O outside the lock to minimise hold time
        if self.output_path:
            try:
                with open(self.output_path, "w") as f:
                    json.dump(output, f)
            except OSError:
                pass

    def _parse_packet(self, raw_data: bytes) -> Optional[dict]:
        """Parse binary UDP packet into hand data dict.

        Args:
            raw_data: Raw bytes from UDP packet.

        Returns:
            Parsed hand data dict, or None if packet is invalid.
        """
        if len(raw_data) < _PACKET_MIN_SIZE:
            Logger.warning("Packet too short: %d bytes", len(raw_data))
            return None

        hand_id = struct.unpack_from("<B", raw_data, 0)[0]
        handedness = "left" if hand_id == 0 else "right"

        joints = []
        offset = _PACKET_HEADER_SIZE
        for i in range(_JOINTS_PER_HAND):
            values = struct.unpack_from("<7f", raw_data, offset)
            if not np.isfinite(values).all():
                Logger.warning("Dropping packet: joint %d carries NaN/Inf (hand=%s)", i, handedness)
                return None
            old_pos = np.array(values[0:3])
            old_quat = np.array(values[3:7])

            new_pos = self._transform_position(old_pos)
            new_quat = self._transform_quaternion(old_quat)

            joints.append(
                {
                    "name": f"Joint{i}",
                    "position": new_pos.tolist(),
                    "orientation": new_quat.tolist(),
                }
            )
            offset += _FLOATS_PER_JOINT * 4

        # Wrist joint (index 1) for hand pose, aligned with source
        # WRIST_JOINT_INDEX = 1 in realmirror_sever_data.py
        wrist_pose = {
            "position": joints[1]["position"],
            "orientation": joints[1]["orientation"],
        }

        return {
            "id": handedness,
            "handedness": handedness,
            "timestamp": int(time.time() * 1000),
            "pose": wrist_pose,
            "joints": joints,
        }

    @staticmethod
    def _transform_position(old_pos: np.ndarray) -> np.ndarray:
        """Transform PICO position to robot coordinate space.

        Transform: new_x = old_z, new_y = -old_x, new_z = old_y.

        Args:
            old_pos: Position in PICO coordinate space [x, y, z].

        Returns:
            Position in robot coordinate space [x, y, z].
        """
        return np.array([old_pos[2], -old_pos[0], old_pos[1]])

    @staticmethod
    def _transform_quaternion(old_quat: np.ndarray) -> np.ndarray:
        """Transform PICO quaternion to robot coordinate space.

        Transform: (x, y, z, w) -> (x, z, y, w).

        Args:
            old_quat: Quaternion in PICO coordinate space [x, y, z, w].

        Returns:
            Quaternion in robot coordinate space [x, y, z, w].
        """
        return np.array([old_quat[0], old_quat[2], old_quat[1], old_quat[3]])
