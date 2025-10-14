# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.types import ArticulationAction


class BaseRobot(ABC):

    def __init__(self, config):
        self.config = config
        self.robot_ref: Optional[Robot] = None
        self.robot_view: Optional[ArticulationView] = None
        self.gripper_states = {"left": None, "right": None}
        self.is_initialized = False

    @abstractmethod
    def initialize(self, world):

        pass

    @abstractmethod
    def setup_gripper(self, arm: str):

        pass

    @abstractmethod
    def update_gripper(self, arm: str, action: str, dt: float):

        pass

    @abstractmethod
    def update_wrist_control(self, arm: str, control_type: str, value: Any, dt: float):

        pass

    @abstractmethod
    def get_end_effector_pose(self, arm: str) -> Tuple[np.ndarray, np.ndarray]:

        pass

    def apply_joint_positions(
        self, joint_positions: np.ndarray, joint_indices: List[int]
    ):

        if self.robot_ref and self.robot_ref.is_valid():
            action = ArticulationAction(
                joint_positions=joint_positions.astype(np.float32),
                joint_indices=np.array(joint_indices, dtype=np.int32),
            )
            self.robot_ref.get_articulation_controller().apply_action(action)

    def get_joint_positions(self) -> Optional[np.ndarray]:

        if self.robot_ref and self.robot_ref.is_valid():
            return self.robot_ref.get_joint_positions()
        return None

    def get_joint_velocities(self) -> Optional[np.ndarray]:

        if self.robot_ref and self.robot_ref.is_valid():
            return self.robot_ref.get_joint_velocities()
        return None

    def get_world_pose(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

        if self.robot_ref and self.robot_ref.is_valid():
            return self.robot_ref.get_world_pose()
        return None, None

    def get_linear_velocity(self) -> Optional[np.ndarray]:

        if self.robot_ref and self.robot_ref.is_valid():
            return self.robot_ref.get_linear_velocity()
        return None

    def get_angular_velocity(self) -> Optional[np.ndarray]:

        if self.robot_ref and self.robot_ref.is_valid():
            return self.robot_ref.get_angular_velocity()
        return None

    def set_joint_positions(self, positions: np.ndarray):

        if self.robot_ref and self.robot_ref.is_valid():
            self.robot_ref.set_joint_positions(positions)

    def get_dof_index(self, joint_name: str) -> int:

        if self.robot_ref and self.robot_ref.is_valid():
            return self.robot_ref.get_dof_index(joint_name)
        return -1

    def get_dof_names(self) -> Optional[List[str]]:

        if self.robot_ref and self.robot_ref.is_valid():
            return self.robot_ref.dof_names
        return None
