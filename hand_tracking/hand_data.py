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
"""Data models and parsing for PICO hand tracking JSON data."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class HandJointData:
    """Single hand joint with position and orientation.

    Attributes:
        name: Joint name (e.g., 'ThumbTip', 'IndexTip').
        position: 3D position [x, y, z] in robot coordinate space.
        orientation: Quaternion [x, y, z, w].
    """

    name: str
    position: np.ndarray
    orientation: np.ndarray


@dataclass
class HandData:
    """Hand tracking data for one hand with 26 joints.

    Attributes:
        handedness: 'left' or 'right'.
        wrist_position: Wrist 3D position [x, y, z].
        wrist_orientation: Wrist quaternion [x, y, z, w].
        joints: List of 26 HandJointData entries.
    """

    handedness: str
    wrist_position: np.ndarray
    wrist_orientation: np.ndarray
    joints: list["HandJointData"] = field(default_factory=list)

    @property
    def thumb_tip(self) -> Optional[np.ndarray]:
        """ThumbTip position (joint index 5)."""
        if len(self.joints) > 5:
            return self.joints[5].position
        return None

    @property
    def index_tip(self) -> Optional[np.ndarray]:
        """IndexTip position (joint index 10)."""
        if len(self.joints) > 10:
            return self.joints[10].position
        return None


def parse_hand_json(json_data: dict) -> dict[str, HandData]:
    """Parse hand tracking JSON into HandData objects keyed by handedness.

    Args:
        json_data: Raw dict from live_hand_data.json with 'data' list.

    Returns:
        Dict with 'left' and/or 'right' keys mapping to HandData.
    """
    result: dict[str, HandData] = {}
    for hand_entry in json_data.get("data", []):
        handedness = hand_entry.get("handedness", "")
        if handedness not in ("left", "right"):
            continue

        pose = hand_entry.get("pose", {})
        position = np.array(pose.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
        orientation = np.array(pose.get("orientation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)

        joints: list[HandJointData] = []
        for joint in hand_entry.get("joints", []):
            joints.append(
                HandJointData(
                    name=joint.get("name", ""),
                    position=np.array(joint.get("position", [0.0, 0.0, 0.0]), dtype=np.float64),
                    orientation=np.array(joint.get("orientation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64),
                )
            )

        result[handedness] = HandData(
            handedness=handedness,
            wrist_position=position,
            wrist_orientation=orientation,
            joints=joints,
        )

    return result
