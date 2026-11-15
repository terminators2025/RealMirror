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

from typing import Dict, List, Optional

import numpy as np

from utility.logger import Logger


class SlowHomingManager:
    """
    Slow homing manager

    Manages smooth homing animations for robot joints, including:
    - triggering and stopping homing
    - per-frame interpolation and joint control
    - cleanup after homing completes

    Attributes:
        robot: reference to the robot object
        teleop_manager: reference to the teleoperation manager (used to realign IK targets after homing)
        duration: total duration of the homing animation (seconds)
        is_active: whether homing is currently active
        progress: current animation progress (0.0-1.0)
        start_positions: start joint positions for homing
        target_positions: target joint positions for homing
        dof_indices: list of DOF indices for controlled joints
        controlled_joint_names: list of controlled joint names
    """

    def __init__(self, robot, teleop_manager=None, duration: float = 0.2, config=None):
        """
        Initialize the slow homing manager

        Args:
            robot: robot object, must provide methods like get_joint_positions
            teleop_manager: teleoperation manager used to realign IK targets after homing
            duration: homing animation duration in seconds (default 0.2)
            config: optional config object which may contain duration settings
        """
        self.robot = robot
        self.teleop_manager = teleop_manager

        self.duration = duration if not config else getattr(config, "slow_homing_duration", duration)

        self.is_active = False
        self.progress = 0.0

        self.start_positions: Optional[np.ndarray] = None
        self.target_positions: Optional[np.ndarray] = None
        self.dof_indices: Optional[List[int]] = None
        self.controlled_joint_names: List[str] = []

        Logger.info(f"SlowHomingManager initialized with duration={self.duration}s")

    def setup_controlled_joints(
        self,
        left_arm_joints: List[str],
        right_arm_joints: List[str],
        left_hand_joints: List[str],
        right_hand_joints: List[str],
    ):
        """
        Configure the list of controlled joints

        Args:
            left_arm_joints: list of left arm joint names
            right_arm_joints: list of right arm joint names
            left_hand_joints: list of left hand joint names
            right_hand_joints: list of right hand joint names
        """
        self.controlled_joint_names = left_arm_joints + right_arm_joints + left_hand_joints + right_hand_joints

        if self.robot and self.robot.is_initialized:
            self.dof_indices = []
            for joint_name in self.controlled_joint_names:
                idx = self.robot.get_dof_index(joint_name)
                if idx != -1:
                    self.dof_indices.append(idx)
                else:
                    Logger.warning(f"Joint '{joint_name}' not found in robot")

            Logger.info(
                f"Slow homing configured for {len(self.dof_indices)} joints "
                f"(expected {len(self.controlled_joint_names)})"
            )
        else:
            Logger.warning("Robot not initialized, cannot setup DOF indices")

    def trigger(self, target_arm_angles: Optional[Dict[str, float]] = None):
        """
        Trigger the slow homing animation

        Args:
            target_arm_angles: optional dict mapping joint name -> angle (radians).
                              If None, use initial positions from the robot configuration if available.

        Returns:
            bool: True if homing was successfully triggered
        """
        # If already homing, do not trigger again
        if self.is_active:
            Logger.warning("Slow homing already active, ignoring trigger")
            return False

        # Check preconditions
        if not self.robot or not self.robot.is_initialized:
            Logger.error("Cannot trigger slow homing: robot not initialized")
            return False

        if not self.dof_indices or not self.controlled_joint_names:
            Logger.error("Cannot trigger slow homing: controlled joints not configured")
            return False

        Logger.info("Triggering slow homing animation...")

        # 1. Read current joint positions as the start
        all_joint_positions = self.robot.get_joint_positions()
        if all_joint_positions is None:
            Logger.error("Failed to get current joint positions")
            return False

        self.start_positions = all_joint_positions[self.dof_indices]

        # 2. Build the target joint positions
        target_positions_list = []

        # If no explicit target angles provided, use initial positions from robot config
        if target_arm_angles is None and hasattr(self.robot, "config"):
            target_arm_angles = self.robot.config.initial_joint_positions

        for joint_name in self.controlled_joint_names:
            if target_arm_angles and joint_name in target_arm_angles:
                target_positions_list.append(target_arm_angles[joint_name])
            elif "thumb_swing_joint" in joint_name:
                target_positions_list.append(np.deg2rad(90.0))
            else:
                target_positions_list.append(0.0)

        self.target_positions = np.array(target_positions_list, dtype=np.float32)

        # 3. Reset progress and activate
        self.progress = 0.0
        self.is_active = True

        Logger.info(f"Slow homing started: {len(self.dof_indices)} joints, " f"duration={self.duration}s")
        return True

    def update(self, dt: float) -> bool:
        """
        Update the homing animation (called each frame)

        Args:
            dt: physics timestep in seconds

        Returns:
            bool: True if homing is complete
        """
        if not self.is_active:
            return False

        # advance progress
        self.progress += dt / self.duration

        # check completion
        is_final_frame = False
        if self.progress >= 1.0:
            self.progress = 1.0
            is_final_frame = True

        # linear interpolation for current joint angles
        if self.start_positions is None or self.target_positions is None:
            Logger.error("Start or target positions not set")
            self.is_active = False
            return True

        interpolated_positions = self.start_positions + (self.target_positions - self.start_positions) * self.progress

        # apply joint positions
        try:
            self.robot.apply_joint_positions(interpolated_positions, self.dof_indices)
        except Exception as e:
            Logger.error(f"Failed to apply joint positions during slow homing: {e}")
            self.is_active = False
            return True

        if is_final_frame:
            self.is_active = False
            if self.teleop_manager:
                self.teleop_manager.align_all_targets_to_effectors()
            Logger.info("Slow homing completed and IK targets realigned")
            return True

        return False

    def cancel(self):
        """Cancel the current homing animation"""
        if self.is_active:
            self.is_active = False
            self.progress = 0.0
            Logger.info("Slow homing cancelled")

    def is_homing(self) -> bool:
        """
        Check whether homing is in progress

        Returns:
            bool: True if homing is active
        """
        return self.is_active

    def get_progress(self) -> float:
        """
        Get current homing progress

        Returns:
            float: progress value (0.0-1.0). Returns 0.0 if not active.
        """
        return self.progress if self.is_active else 0.0
