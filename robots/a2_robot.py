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

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import yaml
import carb
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.xforms import get_world_pose
from pxr import UsdPhysics, Sdf
import omni.usd
from .base_robot import BaseRobot
from utility.logger import Logger


class A2Robot(BaseRobot):

    # Gripper state enumeration
    GRIPPER_STATE_OPEN = 0
    GRIPPER_STATE_CLOSING_S1 = 1
    GRIPPER_STATE_CLOSING_S2 = 2
    GRIPPER_STATE_CLOSED = 3
    GRIPPER_STATE_OPENING = 4

    def __init__(self, config):
        super().__init__(config)

        self.gripper_dof_indices = {"left": {}, "right": {}}
        self.gripper_current_angles = {"left": None, "right": None}
        self.gripper_all_joint_names = {"left": [], "right": []}

        self.wrist_dof_indices = {"left": {}, "right": {}}
        self.wrist_roll_rotating = {
            "left": {"neg": False, "pos": False},
            "right": {"neg": False, "pos": False},
        }

        self.robot_descriptors = {}

    def initialize(self, world, simulation_app):

        Logger.info("Initializing A2 Robot...")

        add_reference_to_stage(
            usd_path=self.config.usd_path, prim_path=self.config.prim_path
        )
        self.robot_ref = world.scene.add(
            Robot(prim_path=self.config.prim_path, name=self.config.name)
        )
        self.robot_view = ArticulationView(
            prim_paths_expr=self.config.prim_path, name=f"{self.config.name}_view"
        )
        world.reset()

        for _ in range(10):
            world.step(render=False)
            simulation_app.update()

        self.robot_ref.initialize()
        self.robot_view.initialize()

        # self._setup_finger_physics_attr()
        self._setup_physics_gains()
        self._create_fixed_joint(world)
        self._load_robot_descriptors()

        self._set_initial_joint_positions()
        for _ in range(10):
            world.step(render=False)
            simulation_app.update()

        self.setup_gripper("left")
        self.setup_gripper("right")

        self._initialize_wrist_control()

        simulation_app.update()
        self.is_initialized = True
        Logger.info("A2 Robot initialized successfully")

    def _setup_finger_physics_attr(self):

        all_dof_prim_paths = self.robot_view._dof_paths
        stage = omni.usd.get_context().get_stage()

        if all_dof_prim_paths is not None and len(all_dof_prim_paths):
            dof_paths_for_robot = all_dof_prim_paths[0]
            for path in dof_paths_for_robot:
                joint_prim = stage.GetPrimAtPath(path)
                if (
                    joint_prim.IsValid()
                    and joint_prim.GetName() in self.config.finger_joint_names
                ):
                    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
                    if drive:
                        drive.GetStiffnessAttr().Set(self.config.finger_kp)
                        drive.GetDampingAttr().Set(self.config.finger_kd)
                        drive.GetMaxForceAttr().Set(self.config.finger_max_force)

    def _setup_physics_gains(self):

        if not self.robot_view or not self.robot_view.is_valid():
            Logger.error("Robot view not valid for setting gains")
            return

        num_dofs = self.robot_view.num_dof
        kps = np.full(num_dofs, self.config.default_kp)
        kds = np.full(num_dofs, self.config.default_kd)

        # Set of finger joint names
        finger_joint_names = set()
        for arm in ["left", "right"]:
            prefix = "L" if arm == "left" else "R"
            finger_joint_names.update(
                [
                    f"{prefix}_thumb_swing_joint",
                    f"{prefix}_thumb_1_joint",
                    f"{prefix}_thumb_2_joint",
                    f"{prefix}_thumb_3_joint",
                    f"{prefix}_index_1_joint",
                    f"{prefix}_index_2_joint",
                    f"{prefix}_middle_1_joint",
                    f"{prefix}_middle_2_joint",
                    f"{prefix}_ring_1_joint",
                    f"{prefix}_ring_2_joint",
                    f"{prefix}_little_1_joint",
                    f"{prefix}_little_2_joint",
                ]
            )

        # Set specific values for finger joints
        all_dof_names = self.robot_view.dof_names
        for i, name in enumerate(all_dof_names):
            if name in finger_joint_names:
                kps[i] = self.config.finger_kp
                kds[i] = self.config.finger_kd

        self.robot_view.set_gains(kps, kds)
        Logger.info("Physics gains set successfully")

    def _load_robot_descriptors(self):

        for arm, path in self.config.robot_descriptor_paths.items():
            try:
                with open(path, "r") as f:
                    self.robot_descriptors[arm] = yaml.safe_load(f)
                    Logger.info(f"Loaded {arm} arm descriptor from {path}")
            except Exception as e:
                Logger.error(f"Failed to load {arm} arm descriptor: {e}")

    def _set_initial_joint_positions(self):

        if not self.robot_ref or not self.robot_ref.is_valid():
            Logger.error("Robot not ready for setting initial positions")
            return
        num_joints = self.robot_ref.num_dof
        joint_positions = np.zeros(num_joints, dtype=np.float32)

        self._load_robot_descriptors()
        for arm_desc in self.robot_descriptors.values():
            if arm_desc:
                for name, value in zip(
                    arm_desc.get("cspace", []), arm_desc.get("default_q", [])
                ):
                    idx = self.robot_ref.get_dof_index(name)
                    if idx != -1 and idx < num_joints:
                        joint_positions[idx] = value

        # Set joint positions from configuration
        for joint_name, angle in self.config.initial_joint_positions.items():
            idx = self.get_dof_index(joint_name)
            if idx != -1 and idx < num_joints:
                joint_positions[idx] = angle

        self.set_joint_positions(joint_positions)
        Logger.info("Initial joint positions set")

    def setup_gripper(self, arm: str):

        if arm not in self.config.gripper_config:
            Logger.error(f"No gripper config for {arm} arm")
            return

        config = self.config.gripper_config[arm]

        # Get all gripper joint names
        all_joints = config["stage1_joints"] + config["stage2_joints"]
        self.gripper_all_joint_names[arm] = all_joints

        # Get DOF indices
        self.gripper_dof_indices[arm] = {
            "all": self._get_joint_indices(all_joints),
            "stage1": self._get_joint_indices(config["stage1_joints"]),
            "stage2": self._get_joint_indices(config["stage2_joints"]),
        }

        # Initialize gripper angles (open state)
        open_angles = np.zeros(len(all_joints), dtype=np.float32)
        # thumb_swing_joint initial angle is 90 degrees
        if all_joints[0].endswith("thumb_swing_joint"):
            open_angles[0] = np.deg2rad(90.0)

        self.gripper_current_angles[arm] = open_angles
        self.gripper_states[arm] = self.GRIPPER_STATE_OPEN

        # Apply initial gripper state
        if self.gripper_dof_indices[arm]["all"]:
            self.apply_joint_positions(
                open_angles, self.gripper_dof_indices[arm]["all"]
            )

        Logger.info(f"{arm} gripper initialized with {len(all_joints)} joints")

    def _initialize_wrist_control(self):

        for arm in ["left", "right"]:
            self.wrist_dof_indices[arm] = {}
            for joint_type, joint_name in self.config.wrist_joints[arm].items():
                idx = self.get_dof_index(joint_name)
                if idx != -1:
                    self.wrist_dof_indices[arm][joint_type] = idx
                    Logger.info(
                        f"Found {arm} wrist {joint_type} joint: {joint_name} (idx: {idx})"
                    )
                else:
                    Logger.error(
                        f"Could not find {arm} wrist {joint_type} joint: {joint_name}"
                    )

    def _create_fixed_joint(self, world):

        stage = world.stage
        ground_prim_path = "/World/defaultGroundPlane/GroundPlane"
        torso_prim_path = f"{self.config.prim_path}/raise_a2_t2d0_flagship/base_link"

        ground_prim = stage.GetPrimAtPath(ground_prim_path)
        torso_prim = stage.GetPrimAtPath(torso_prim_path)

        if ground_prim.IsValid() and torso_prim.IsValid():
            joint_path = Sdf.Path("/World/ground_torso_fixed_joint_A2")
            joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            joint.CreateBody0Rel().SetTargets([ground_prim.GetPath()])
            joint.CreateBody1Rel().SetTargets([torso_prim.GetPath()])
            Logger.info("Created fixed joint between ground and robot")
        else:
            Logger.warn("Could not create fixed joint")

    def _get_joint_indices(self, joint_names: List[str]) -> List[int]:

        indices = []
        for name in joint_names:
            idx = self.get_dof_index(name)
            if idx != -1:
                indices.append(idx)
            else:
                Logger.warn(f"Joint {name} not found")
        return indices

    def update_gripper(self, arm: str, action: str, dt: float):

        if not self.robot_ref or not self.robot_ref.is_valid():
            return

        # Handle action
        if action == "toggle":
            if self.gripper_states[arm] == self.GRIPPER_STATE_OPEN:
                self.gripper_states[arm] = self.GRIPPER_STATE_CLOSING_S1
            elif self.gripper_states[arm] == self.GRIPPER_STATE_CLOSED:
                self.gripper_states[arm] = self.GRIPPER_STATE_OPENING
        elif action == "open":
            self.gripper_states[arm] = self.GRIPPER_STATE_OPENING
        elif action == "close":
            self.gripper_states[arm] = self.GRIPPER_STATE_CLOSING_S1

        # Update gripper animation
        self._update_gripper_animation(arm, dt)

    def _update_gripper_animation(self, arm: str, dt: float):

        state = self.gripper_states[arm]

        if state == self.GRIPPER_STATE_OPEN or state == self.GRIPPER_STATE_CLOSED:
            return

        config = self.config.gripper_config[arm]
        current_angles = self.gripper_current_angles[arm].copy()
        all_converged = True

        # Determine the target for the current stage
        if state == self.GRIPPER_STATE_CLOSING_S1:
            active_indices = self.gripper_dof_indices[arm]["stage1"]
            target_angles = np.array(config["stage1_closed_angles"])
            joint_names = config["stage1_joints"]
        elif state == self.GRIPPER_STATE_CLOSING_S2:
            active_indices = self.gripper_dof_indices[arm]["stage2"]
            target_angles = np.array(config["stage2_closed_angles"])
            joint_names = config["stage2_joints"]
        elif state == self.GRIPPER_STATE_OPENING:
            active_indices = self.gripper_dof_indices[arm]["all"]
            # Open angle: 90 degrees for thumb_swing, 0 for others
            target_angles = np.zeros(len(self.gripper_all_joint_names[arm]))
            if self.gripper_all_joint_names[arm][0].endswith("thumb_swing_joint"):
                target_angles[0] = np.deg2rad(90.0)
            joint_names = self.gripper_all_joint_names[arm]
        else:
            return

        # Update angles
        angles_to_apply = []
        indices_to_apply = []

        for i, (joint_name, dof_idx) in enumerate(zip(joint_names, active_indices)):
            # Find the index in the full list
            full_idx = self.gripper_all_joint_names[arm].index(joint_name)
            current = current_angles[full_idx]
            target = (
                target_angles[i]
                if state != self.GRIPPER_STATE_OPENING
                else target_angles[full_idx]
            )

            error = target - current
            if abs(error) < self.config.gripper_convergence_threshold:
                current_angles[full_idx] = target
            else:
                all_converged = False
                change = np.clip(
                    error,
                    -self.config.gripper_angular_speed * dt,
                    self.config.gripper_angular_speed * dt,
                )
                current_angles[full_idx] = current + change

            angles_to_apply.append(current_angles[full_idx])
            indices_to_apply.append(dof_idx)

        # Apply joint angles
        if angles_to_apply:
            self.apply_joint_positions(np.array(angles_to_apply), indices_to_apply)

        # Update state
        self.gripper_current_angles[arm] = current_angles

        if all_converged:
            if state == self.GRIPPER_STATE_CLOSING_S1:
                self.gripper_states[arm] = self.GRIPPER_STATE_CLOSING_S2
            elif state == self.GRIPPER_STATE_CLOSING_S2:
                self.gripper_states[arm] = self.GRIPPER_STATE_CLOSED
            elif state == self.GRIPPER_STATE_OPENING:
                self.gripper_states[arm] = self.GRIPPER_STATE_OPEN

    def update_wrist_control(self, arm: str, control_type: str, value: Any, dt: float):

        if control_type == "roll":
            self._update_wrist_roll(arm, value, dt)
        elif control_type == "joystick":
            self._update_wrist_joystick(arm, value)

    def _update_wrist_roll(self, arm: str, direction: str, dt: float):

        if "roll" not in self.wrist_dof_indices[arm]:
            return

        dof_idx = self.wrist_dof_indices[arm]["roll"]
        current_positions = self.get_joint_positions()

        if current_positions is None or dof_idx >= len(current_positions):
            return

        current_angle = current_positions[dof_idx]
        delta = self.config.wrist_roll_speed * dt

        # Reverse for the left arm
        if arm == "left":
            delta = -delta if direction == "negative" else delta
        else:
            delta = -delta if direction == "positive" else delta

        new_angle = np.clip(
            current_angle + delta,
            self.config.wrist_roll_limits["min"],
            self.config.wrist_roll_limits["max"],
        )

        if abs(new_angle - current_angle) > 1e-5:
            self.apply_joint_positions(np.array([new_angle]), [dof_idx])

    def _update_wrist_joystick(self, arm: str, joystick_values: Dict[str, float]):

        indices = []
        positions = []

        if "pitch" in self.wrist_dof_indices[arm] and "pitch" in joystick_values:
            indices.append(self.wrist_dof_indices[arm]["pitch"])
            positions.append(joystick_values["pitch"])

        if "yaw" in self.wrist_dof_indices[arm] and "yaw" in joystick_values:
            indices.append(self.wrist_dof_indices[arm]["yaw"])
            positions.append(joystick_values["yaw"])

        if indices:
            self.apply_joint_positions(np.array(positions), indices)

    def get_end_effector_pose(self, arm: str) -> Tuple[np.ndarray, np.ndarray]:

        ee_link_path = (
            f"{self.config.prim_path}/raise_a2_t2d0_flagship/{arm}_arm_link07"
        )
        try:
            return get_world_pose(ee_link_path)
        except Exception as e:
            Logger.warning(f"Could not get end effector pose for {arm}: {e}")
            return np.zeros(3), np.array([1, 0, 0, 0])

    def set_wrist_roll_state(self, arm: str, direction: str, active: bool):

        if direction == "negative":
            self.wrist_roll_rotating[arm]["neg"] = active
        elif direction == "positive":
            self.wrist_roll_rotating[arm]["pos"] = active

    def get_wrist_roll_state(self, arm: str) -> Dict[str, bool]:

        return self.wrist_roll_rotating[arm]
