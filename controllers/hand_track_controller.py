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
"""Hand tracking controller for PICO VR hand tracking teleoperation.

Implements BaseController interface for hand tracking data, including:
- Position offset calibration (aligned with hand_track lines 2283-2299)
- Pinch detection with edge-triggered gripper control (lines 2347-2360)
- Wrist orientation mapping with left/right arm sign differences (lines 2312-2424)
"""

import json
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from controllers.base_controller import BaseController
from hand_tracking.hand_data import HandData, parse_hand_json
from utility.logger import Logger

# Gripper states (aligned with hand_track lines 386-410)
GRIPPER_STATE_OPEN = 0
GRIPPER_STATE_CLOSING_S1 = 1
GRIPPER_STATE_CLOSING_S2 = 2
GRIPPER_STATE_CLOSED = 3
GRIPPER_STATE_OPENING = 4

# Constants (aligned with hand_track source)
GRIPPER_PINCH_THRESHOLD: float = 0.035  # 35mm (line 217)
TARGET_CUBE_MOVE_THRESHOLD_MAX: float = 0.10  # 10cm (line 2219)
TARGET_CUBE_MOVE_THRESHOLD_MIN: float = 0.001  # 1mm (line 2218)
GRIPPER_ANGULAR_SPEED: float = np.deg2rad(980.0)  # 980 deg/s (line 407)
GRIPPER_CONVERGENCE_THRESHOLD: float = np.deg2rad(2.0)  # 2 deg (line 409)


class HandTrackController(BaseController):
    """Controller for PICO VR hand tracking teleoperation.

    Reads hand tracking data from a HandTrackUDPServer's in-memory snapshot
    (preferred, zero file I/O) or falls back to polling a JSON file on disk.
    Provides position (with offset calibration), orientation, and pinch-based
    gripper control through the BaseController interface.
    """

    def __init__(self, json_path: str):
        self._json_path = json_path

        # Direct reference to the UDP server for in-memory reads (set via set_udp_server).
        self._udp_server: Optional[Any] = None

        # Position offset calibration (hand_track g_hand_offset_*)
        self._hand_offset: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._hand_offset_is_set: dict[str, bool] = {"left": False, "right": False}

        # Orientation base rotation calibration (hand_track g_base_rotation_*)
        self._base_rotation: dict[str, Optional[Rotation]] = {"left": None, "right": None}

        # Pinch detection state (hand_track g_*_gripper_is_pinching)
        self._gripper_is_pinching: dict[str, bool] = {"left": False, "right": False}
        self._gripper_state: dict[str, int] = {
            "left": GRIPPER_STATE_OPEN,
            "right": GRIPPER_STATE_OPEN,
        }

        # Raw hand data cache
        self._hand_data: dict[str, HandData] = {}

        # Wrist mapping method: "rotvec" (default, no gimbal lock) or "euler" (original)
        self._wrist_method: str = "rotvec"

        # EMA smoothing state for wrist joint targets
        self._smoothed_targets: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}

        # EMA smoothing factor (0.3 at ~83 Hz control rate ≈ 6 Hz cutoff, ~40ms latency)
        self._wrist_ema_alpha: float = 0.3

        # Amplitude clamp for rotvec components (radians)
        self._wrist_clamp_rad: float = 1.2

    def read_data(self) -> Optional[Dict[str, Any]]:
        """Poll hand data and return keyed by 'left'/'right'.

        Reads from the UDP server's in-memory snapshot when available
        (zero file I/O, no read-write race). Falls back to JSON file
        polling if no server reference is set.
        """
        raw: Optional[dict] = None

        # Prefer in-memory snapshot (thread-safe, no disk I/O)
        if self._udp_server is not None:
            raw = self._udp_server.get_latest_snapshot()
        else:
            try:
                with open(self._json_path) as f:
                    raw = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return None

        if raw is None:
            return None

        self._hand_data = parse_hand_json(raw)

        if not self._hand_data:
            return None

        result = {}
        for arm, hand in self._hand_data.items():
            result[arm] = {
                "handedness": hand.handedness,
                "wrist_position": hand.wrist_position,
                "wrist_orientation": hand.wrist_orientation,
                "joints": hand.joints,
            }
        return result

    def set_udp_server(self, server: object) -> None:
        """Attach a HandTrackUDPServer for zero-copy in-memory reads.

        When set, ``read_data()`` will call ``server.get_latest_snapshot()``
        instead of reading the JSON file from disk.

        Args:
            server: A ``HandTrackUDPServer`` instance (or any object with
                a ``get_latest_snapshot()`` method).
        """
        self._udp_server = server
        Logger.info("HandTrackController: using in-memory data path (no file I/O).")

    def get_position(self, controller_data: Dict) -> Optional[list]:
        """Get raw wrist position (no offset applied).

        Position offset calibration is handled by TeleopManager.update_target_position(),
        aligned with source project lines 2283-2299 where offset is applied in the main loop.
        """
        pos = controller_data.get("wrist_position")
        if pos is None:
            return None
        try:
            arr = np.asarray(pos, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if arr.shape != (3,) or not np.isfinite(arr).all():
            Logger.warning("Rejecting hand wrist position with invalid geometry: %s", pos)
            return None
        return list(arr)

    def get_orientation(self, controller_data: Dict) -> Optional[list]:
        """Get wrist quaternion [x, y, z, w]."""
        orient = controller_data.get("wrist_orientation")
        if orient is None:
            return None
        try:
            arr = np.asarray(orient, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if arr.shape != (4,) or not np.isfinite(arr).all():
            Logger.warning("Rejecting hand wrist orientation with invalid geometry: %s", orient)
            return None
        return list(arr)

    def get_buttons(self, controller_data: Dict) -> Dict[str, bool]:
        """Detect pinch gesture from thumb-index tip distance.

        Aligned with hand_track lines 2347-2360.
        Uses rising/falling edge detection for gripper state transitions.
        """
        handedness = controller_data.get("handedness", "right")
        joints = controller_data.get("joints", [])

        is_pinching = False
        if len(joints) == 26:
            thumb_tip = np.asarray(joints[5].position, dtype=np.float64)
            index_tip = np.asarray(joints[10].position, dtype=np.float64)
            pinch_distance = np.linalg.norm(thumb_tip - index_tip)
            is_pinching = bool(pinch_distance < GRIPPER_PINCH_THRESHOLD)

        # Edge detection (aligned with hand_track lines 2354-2359)
        was_pinching = self._gripper_is_pinching[handedness]
        if is_pinching != was_pinching:
            self._handle_pinch_edge(handedness, is_pinching)

        self._gripper_is_pinching[handedness] = is_pinching
        return {"gripper": bool(is_pinching)}

    def get_axes(self, controller_data: Dict) -> list:
        """Hand tracking has no joystick axes."""
        return []

    def set_position_offset(self, arm: str, offset: np.ndarray) -> None:
        """Set position offset for calibration (called by TeleopManager)."""
        self._hand_offset[arm] = offset
        self._hand_offset_is_set[arm] = True

    def get_position_offset(self, arm: str) -> Optional[np.ndarray]:
        """Get current position offset, or None if not calibrated."""
        return self._hand_offset[arm]

    def set_wrist_method(self, method: str) -> None:
        """Set wrist orientation mapping method.

        Args:
            method: 'rotvec' (rotation vector, no gimbal lock) or
                    'euler' (original Euler angle method).

        Raises:
            ValueError: If method is not 'rotvec' or 'euler'.
        """
        if method not in ("rotvec", "euler"):
            raise ValueError(f"Unknown wrist method: {method!r}, expected 'rotvec' or 'euler'")
        self._wrist_method = method
        Logger.info("Wrist mapping method set to: %s", method)

    def get_wrist_joint_targets(
        self,
        arm: str,
        initial_joint_positions: np.ndarray,
        wrist_dof_indices: dict,
    ) -> Optional[dict]:
        """Map wrist orientation to 3 wrist joint angles.

        Supports two decomposition methods selectable via set_wrist_method():
        - 'rotvec': Rotation vector decomposition (no gimbal lock, recommended).
        - 'euler': Original Euler angle decomposition (source project lines 2312-2424).

        Both methods include EMA temporal smoothing to reduce sensor noise jitter.

        Args:
            arm: 'left' or 'right'.
            initial_joint_positions: Full robot joint position array.
            wrist_dof_indices: Dict with 'roll', 'yaw', 'pitch' keys mapping to
                joint indices in initial_joint_positions.

        Returns:
            Dict with 'roll', 'yaw', 'pitch' target joint angles, or None.
        """
        if arm not in self._hand_data:
            return None

        hand = self._hand_data[arm]
        R_current = Rotation.from_quat(hand.wrist_orientation)

        # Calibration: first call sets base rotation
        if self._base_rotation[arm] is None:
            self._base_rotation[arm] = R_current.inv()
            self._smoothed_targets[arm] = None
            Logger.info("%s arm hand base rotation INITIALIZED.", arm.upper())
            return {
                "roll": initial_joint_positions[wrist_dof_indices["roll"]],
                "yaw": initial_joint_positions[wrist_dof_indices["yaw"]],
                "pitch": initial_joint_positions[wrist_dof_indices["pitch"]],
            }

        # Per-frame: compute relative rotation and map to joint targets
        R_relative = R_current * self._base_rotation[arm]

        if self._wrist_method == "rotvec":
            raw_targets = self._map_wrist_rotvec(arm, R_relative, initial_joint_positions, wrist_dof_indices)
        else:
            raw_targets = self._map_wrist_euler(arm, R_relative, initial_joint_positions, wrist_dof_indices)

        # Apply EMA smoothing on output joint targets
        targets_arr = np.array([raw_targets["roll"], raw_targets["yaw"], raw_targets["pitch"]])
        smoothed = self._smoothed_targets[arm]
        if smoothed is None:
            smoothed = targets_arr
        else:
            alpha = self._wrist_ema_alpha
            smoothed = alpha * targets_arr + (1.0 - alpha) * smoothed
        self._smoothed_targets[arm] = smoothed

        return {
            "roll": float(smoothed[0]),
            "yaw": float(smoothed[1]),
            "pitch": float(smoothed[2]),
        }

    def _map_wrist_euler(
        self,
        arm: str,
        R_relative: Rotation,
        initial_joint_positions: np.ndarray,
        wrist_dof_indices: dict,
    ) -> dict:
        """Original Euler angle wrist mapping (source project lines 2312-2424).

        Uses intrinsic XYZ Euler decomposition with cross-mapping to robot joints.
        Known limitation: gimbal lock when Y-axis component approaches +/-90 deg.

        Args:
            arm: 'left' or 'right'.
            R_relative: Relative rotation from base calibration pose.
            initial_joint_positions: Full robot joint position array.
            wrist_dof_indices: Dict with 'roll', 'yaw', 'pitch' DOF index mapping.

        Returns:
            Dict with 'roll', 'yaw', 'pitch' raw target joint angles.
        """
        relative_euler = R_relative.as_euler("xyz", degrees=False)
        pitch, yaw, roll = relative_euler[0], relative_euler[1], relative_euler[2]

        initial_roll = initial_joint_positions[wrist_dof_indices["roll"]]
        initial_yaw = initial_joint_positions[wrist_dof_indices["yaw"]]
        initial_pitch = initial_joint_positions[wrist_dof_indices["pitch"]]

        if arm == "right":
            return {
                "roll": initial_roll + yaw,
                "yaw": initial_yaw + 0.8 * pitch,
                "pitch": initial_pitch - 0.8 * roll,
            }
        else:
            return {
                "roll": initial_roll - yaw,
                "yaw": initial_yaw - 0.8 * pitch,
                "pitch": initial_pitch - 0.8 * roll,
            }

    def _map_wrist_rotvec(
        self,
        arm: str,
        R_relative: Rotation,
        initial_joint_positions: np.ndarray,
        wrist_dof_indices: dict,
    ) -> dict:
        """Rotation vector wrist mapping -- no gimbal lock.

        Uses as_rotvec() which returns [rx, ry, rz] where the direction is the
        rotation axis and the magnitude is the rotation angle (radians). Components
        are continuous and numerically stable at all orientations (no singularity
        for |theta| < pi, covering full human wrist ROM).

        Same cross-mapping as Euler method for consistent feel:
        rv[1] (Y-axis) -> robot roll, rv[0] (X-axis) -> robot yaw,
        rv[2] (Z-axis) -> robot pitch.

        Args:
            arm: 'left' or 'right'.
            R_relative: Relative rotation from base calibration pose.
            initial_joint_positions: Full robot joint position array.
            wrist_dof_indices: Dict with 'roll', 'yaw', 'pitch' DOF index mapping.

        Returns:
            Dict with 'roll', 'yaw', 'pitch' raw target joint angles.
        """
        rv = R_relative.as_rotvec()
        rv = np.clip(rv, -self._wrist_clamp_rad, self._wrist_clamp_rad)

        initial_roll = initial_joint_positions[wrist_dof_indices["roll"]]
        initial_yaw = initial_joint_positions[wrist_dof_indices["yaw"]]
        initial_pitch = initial_joint_positions[wrist_dof_indices["pitch"]]

        if arm == "right":
            return {
                "roll": initial_roll + rv[1],
                "yaw": initial_yaw + 0.8 * rv[0],
                "pitch": initial_pitch - 0.8 * rv[2],
            }
        else:
            return {
                "roll": initial_roll - rv[1],
                "yaw": initial_yaw - 0.8 * rv[0],
                "pitch": initial_pitch - 0.8 * rv[2],
            }

    def reset_base_rotation(self) -> None:
        """Reset orientation base rotation for recalibration.

        Aligned with hand_track reset_hand_teleop_base_rotations() (lines 698-703).
        Next get_wrist_joint_targets() call will auto-recalibrate.
        Also clears EMA smoothing state for a fresh start.
        """
        self._base_rotation["left"] = None
        self._base_rotation["right"] = None
        self._smoothed_targets["left"] = None
        self._smoothed_targets["right"] = None
        Logger.info("Hand teleop base rotations reset. Will recalibrate on next frame.")

    def reset_hand_offset(self, arm: str) -> None:
        """Reset position offset and smoothing state (for teleop toggle off/on)."""
        self._hand_offset[arm] = None
        self._hand_offset_is_set[arm] = False
        self._smoothed_targets[arm] = None

    def get_gripper_state(self, arm: str) -> int:
        """Get current gripper state machine state."""
        return self._gripper_state[arm]

    def set_gripper_state(self, arm: str, state: int) -> None:
        """Set gripper state (used by A2HandTrackMapping after animation)."""
        self._gripper_state[arm] = state

    def _handle_pinch_edge(self, arm: str, is_pinching: bool) -> None:
        """Handle pinch rising/falling edge for gripper state transitions.

        Aligned with hand_track lines 2354-2359:
        - Rising edge (not pinching -> pinching): OPEN -> CLOSING_S1
        - Falling edge (pinching -> not pinching): CLOSED -> OPENING
        """
        if is_pinching:
            # Rising edge
            if self._gripper_state[arm] == GRIPPER_STATE_OPEN:
                self._gripper_state[arm] = GRIPPER_STATE_CLOSING_S1
        else:
            # Falling edge
            if self._gripper_state[arm] == GRIPPER_STATE_CLOSED:
                self._gripper_state[arm] = GRIPPER_STATE_OPENING
