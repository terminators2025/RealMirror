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

from typing import Any, Optional, Tuple

import numpy as np
import yaml

from utility.logger import Logger


class IKSolver:

    def __init__(self, robot, robot_config):
        self.robot = robot
        self.robot_config = robot_config
        self.solvers = {"left": None, "right": None}
        self.articulation_solvers = {"left": None, "right": None}

    def initialize(self):

        try:
            from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import (
                ArticulationKinematicsSolver,
            )
            from isaacsim.robot_motion.motion_generation.lula.kinematics import (
                LulaKinematicsSolver,
            )

            for arm in ["left", "right"]:
                descriptor_path = self.robot_config.robot_descriptor_paths.get(arm)
                if not descriptor_path:
                    Logger.error(f"No descriptor path for {arm} arm")
                    continue

                try:
                    with open(descriptor_path, "r") as f:
                        descriptor = yaml.safe_load(f)
                except Exception as e:
                    Logger.error(f"Failed to load {arm} descriptor: {e}")
                    continue

                ee_frame = descriptor.get("end_effectors", {}).get(f"{arm}_arm_ee", {}).get("frame")
                if not ee_frame:
                    Logger.error(f"No end effector frame found for {arm} arm")
                    continue

                try:
                    lula_solver = LulaKinematicsSolver(
                        urdf_path=self.robot_config.urdf_path,
                        robot_description_path=descriptor_path,
                    )

                    if ee_frame not in lula_solver.get_all_frame_names():
                        Logger.error(f"End effector frame '{ee_frame}' not found in Lula solver")
                        continue

                    articulation_solver = ArticulationKinematicsSolver(self.robot.robot_ref, lula_solver, ee_frame)

                    self.solvers[arm] = lula_solver
                    self.articulation_solvers[arm] = articulation_solver

                    Logger.info(f"IK solver initialized for {arm} arm")

                except Exception as e:
                    Logger.error(f"Failed to create IK solver for {arm} arm: {e}")

        except ImportError as e:
            Logger.error(f"Failed to import IK solver modules: {e}")

    def compute_inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray],
        arm: str,
    ) -> Tuple[Optional[Any], bool]:

        if arm not in self.articulation_solvers or not self.articulation_solvers[arm]:
            return None, False

        try:
            ik_action, success = self.articulation_solvers[arm].compute_inverse_kinematics(
                target_position, target_orientation
            )

            return ik_action, success

        except Exception as e:
            Logger.error(f"IK computation failed for {arm} arm: {e}")
            return None, False

    def is_initialized(self, arm: Optional[str] = None) -> bool:

        if arm:
            return self.articulation_solvers.get(arm) is not None
        return any(solver is not None for solver in self.articulation_solvers.values())
