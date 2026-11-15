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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from utility.logger import Logger


class ControlMapping(ABC):

    #: Per-arm offset between the device pose and the robot target
    position_offset: Dict[str, Optional[np.ndarray]]

    @abstractmethod
    def reset_offset(self, arm: str) -> None:
        pass

    @abstractmethod
    def set_offset(self, arm: str, offset: np.ndarray) -> None:
        pass

    @abstractmethod
    def map_position_to_target(self, controller_pos: np.ndarray, arm: str) -> np.ndarray:

        pass

    @abstractmethod
    def map_buttons_to_actions(self, buttons: Dict[str, bool], arm: str) -> Dict[str, Any]:

        pass

    @abstractmethod
    def map_axes_to_joints(self, axes: list, arm: str) -> Dict[str, float]:

        pass


class A2PicoMapping(ControlMapping):

    def __init__(self, config=None):
        self.position_scale = 1.0
        self.position_offset = {"left": None, "right": None}
        self.joystick_deadzone = 0.03 if not config else config.joystick_deadzone

        self.wrist_limits = {"pitch": np.deg2rad(20.0), "yaw": np.deg2rad(30.0)}

    def reset_offset(self, arm: str):

        self.position_offset[arm] = None

    def set_offset(self, arm: str, offset: np.ndarray):

        self.position_offset[arm] = offset

    def map_position_to_target(self, controller_pos: np.ndarray, arm: str) -> np.ndarray:

        if self.position_offset[arm] is None:
            return controller_pos
        return np.asarray(controller_pos + self.position_offset[arm])

    def map_buttons_to_actions(self, buttons: Dict[str, bool], arm: str) -> Dict[str, Any]:

        actions: Dict[str, Any] = {}
        if buttons.get("button_x/a", False):
            actions["gripper"] = "toggle"

        if arm == "left":
            if buttons.get("Trigger", False):
                actions["wrist_roll"] = "positive"
            elif buttons.get("Squeeze", False):
                actions["wrist_roll"] = "negative"
        else:
            if buttons.get("Trigger", False):
                actions["wrist_roll"] = "negative"
            elif buttons.get("Squeeze", False):
                actions["wrist_roll"] = "positive"

        if buttons.get("button_y/b", False):
            actions["teleop_toggle"] = True

        if arm == "left" and buttons.get("Thumbstick", False):
            actions["reset_pose"] = True
            Logger.debug("Reset pose triggered by left thumbstick button.")

        if arm == "right" and buttons.get("Thumbstick", False):
            actions["slow_homing"] = True
            Logger.debug("Slow homing triggered by right thumbstick button.")

        return actions

    def map_axes_to_joints(self, axes: list, arm: str) -> Dict[str, float]:

        joint_values = {}

        if len(axes) >= 4:
            x_axis = axes[2] if abs(axes[2]) > self.joystick_deadzone else 0
            y_axis = axes[3] if abs(axes[3]) > self.joystick_deadzone else 0

            if arm == "left":
                joint_values["pitch"] = -y_axis * self.wrist_limits["pitch"]
            else:
                joint_values["pitch"] = y_axis * self.wrist_limits["pitch"]

            joint_values["yaw"] = -x_axis * self.wrist_limits["yaw"]

        return joint_values


class A2HandTrackMapping(ControlMapping):
    """Mapping for hand tracking gestures to A2 robot actions.

    Simpler than A2PicoMapping because wrist orientation and pinch detection
    are handled by HandTrackController. This mapping only handles:
    - Position offset (calibrated by TeleopManager)
    - Gripper action from pinch button
    """

    def __init__(self):
        self.position_offset: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}

    def reset_offset(self, arm: str) -> None:
        """Reset the position offset for the given arm."""
        self.position_offset[arm] = None

    def set_offset(self, arm: str, offset: np.ndarray) -> None:
        """Set the position offset for the given arm."""
        self.position_offset[arm] = offset

    def map_position_to_target(self, controller_pos: np.ndarray, arm: str) -> np.ndarray:
        """Map controller position to target position with optional offset."""
        if self.position_offset[arm] is None:
            return controller_pos
        return np.asarray(controller_pos + self.position_offset[arm])

    def map_buttons_to_actions(self, buttons: Dict[str, bool], arm: str) -> Dict[str, Any]:
        """Map pinch state to gripper action: close when pinching, open when not."""
        actions: Dict[str, Any] = {}
        if buttons.get("gripper", False):
            actions["gripper"] = "close"
        else:
            actions["gripper"] = "open"
        return actions

    def map_axes_to_joints(self, axes: list, arm: str) -> Dict[str, float]:
        """Hand tracking has no joystick axes."""
        return {}
