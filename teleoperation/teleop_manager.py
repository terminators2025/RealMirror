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
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from omni.isaac.core.objects import FixedCuboid

from utility.logger import Logger

if TYPE_CHECKING:
    from controllers.control_mapping import ControlMapping


class TeleopManager:

    # Lazily resolved reference to HandTrackController class.
    # Avoids a hard top-level import on controllers.hand_track_controller
    # so that teleop_manager can be imported without pulling in scipy etc.
    _HandTrackController: Optional[type] = None

    # Attributes bound in __init__; declared here so the typed methods below see
    # their real types instead of the placeholder values used during construction.
    robot: Any
    control_mapping: "ControlMapping"
    ik_solver: Optional[Any]
    config: Optional[Any]
    slow_homing_manager: Optional[Any]
    controller: Any
    target_cubes: Dict[str, Any]
    teleop_enabled: Dict[str, bool]
    previous_positions: Dict[str, Optional[np.ndarray]]
    previous_joint_positions: Dict[str, Optional[np.ndarray]]
    previous_button_states: Dict[str, Dict[str, Any]]
    last_teleop_toggle_time: Dict[str, float]

    @classmethod
    def _get_hand_track_cls(cls) -> type:
        """Lazily import and return HandTrackController class."""
        if cls._HandTrackController is None:
            from controllers.hand_track_controller import HandTrackController

            cls._HandTrackController = HandTrackController
        return cls._HandTrackController

    def _is_hand_track(self, obj: object) -> bool:
        """Check if *obj* is a HandTrackController without importing it.

        A type-name comparison keeps this module from importing
        controllers.hand_track_controller (and therefore scipy) unless hand
        tracking is actually in use.
        """
        return type(obj).__name__ == "HandTrackController"

    def __init__(
        self,
        robot: Any,
        control_mapping: "ControlMapping",
        ik_solver: Optional[Any] = None,
        config: Optional[Any] = None,
        slow_homing_manager: Optional[Any] = None,
    ) -> None:
        self.robot = robot
        self.control_mapping = control_mapping
        self.ik_solver = ik_solver
        self.config = config
        self.slow_homing_manager = slow_homing_manager
        self.controller = None

        self.teleop_enabled = {"left": False, "right": False}
        self.target_cubes = {"left": None, "right": None}

        self.position_filter_threshold_min = 0.001 if not config else config.target_cube_move_threshold_min
        self.position_filter_threshold_max = 0.10 if not config else config.target_cube_move_threshold_max
        self.joint_jump_threshold_rad = np.deg2rad(30.0) if not config else np.deg2rad(config.joint_jump_threshold_deg)
        self.ik_diff_threshold: float = 1e-3 if not config else config.ik_diff_threshold

        self.previous_positions: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self.previous_joint_positions: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}

        self.previous_button_states: Dict[str, Dict[str, Any]] = {"left": {}, "right": {}}

        self.teleop_toggle_debounce_time: float = 0.5 if not config else config.teleop_toggle_debounce_time
        self.last_teleop_toggle_time: Dict[str, float] = {"left": 0, "right": 0}

        # Cached initial joint positions for wrist orientation mapping
        # (aligned with hand_track g_initial_robot_joint_positions)
        self._initial_joint_positions: Optional[np.ndarray] = None

        # Cached wrist DOF indices (computed once, reused every frame)
        self._wrist_dof_indices_cache: Dict[str, Optional[dict]] = {}

    def initialize_target_cubes(self, world: Any, hide_cubes: bool = False) -> None:

        for arm in ["left", "right"]:
            color = np.array([1, 0, 0]) if arm == "left" else np.array([0, 1, 0])
            cube_name = f"target_cube_{arm}"
            cube_path = f"/World/TargetCube{arm.capitalize()}"

            self.target_cubes[arm] = world.scene.add(
                FixedCuboid(
                    prim_path=cube_path,
                    name=cube_name,
                    position=np.array([0, 0, 0]),
                    scale=np.array([0.05, 0.05, 0.05]),
                    color=color,
                )
            )
            if hide_cubes and self.target_cubes[arm] is not None:
                self.target_cubes[arm].set_visibility(False)

            self._disable_cube_collision(cube_path, world)

        self.align_all_targets_to_effectors(world)

    def _disable_cube_collision(self, cube_path: str, world: Any) -> None:

        stage = world.stage
        prim = stage.GetPrimAtPath(cube_path)
        if prim and prim.IsValid():
            collision_attr = prim.GetAttribute("physics:collisionEnabled")
            if collision_attr:
                collision_attr.Set(False)
                Logger.info(f"Disabled collision for {cube_path}")

    def align_all_targets_to_effectors(self, world: Optional[Any] = None) -> None:

        for arm in ["left", "right"]:
            self.align_target_to_effector(arm)

        if world:
            for _ in range(5):
                world.step(render=False)

        self.control_mapping.reset_offset("left")
        self.control_mapping.reset_offset("right")
        self.previous_positions = {"left": None, "right": None}

        # Cache initial joint positions for wrist orientation mapping
        # (aligned with hand_track g_initial_robot_joint_positions)
        current_positions = self.robot.get_joint_positions()
        if current_positions is not None:
            self._initial_joint_positions = np.copy(current_positions)

        Logger.info("Teleop targets aligned and offsets reset")

    def align_target_to_effector(self, arm: str) -> None:

        if self.target_cubes[arm] and self.robot:
            ee_pos, _ = self.robot.get_end_effector_pose(arm)
            self.target_cubes[arm].set_world_pose(position=ee_pos)
            Logger.info(f"Aligned {arm} target cube to end-effector at: {ee_pos}")

    def toggle_teleop(self, arm: str) -> None:

        current_time = time.time()

        if current_time - self.last_teleop_toggle_time[arm] < self.teleop_toggle_debounce_time:
            return

        self.teleop_enabled[arm] = not self.teleop_enabled[arm]
        self.last_teleop_toggle_time[arm] = current_time

        if self.teleop_enabled[arm]:
            Logger.info(f"{arm.upper()} ARM Teleop: ENABLED")
            self.align_target_to_effector(arm)
        else:
            Logger.info(f"{arm.upper()} ARM Teleop: DISABLED")
            self.control_mapping.reset_offset(arm)
            self.previous_positions[arm] = None
            # Reset hand tracking state for recalibration on re-enable
            # (aligned with hand_track lines 678-680, 692-694)
            if self._is_hand_track(self.controller):
                self.controller.reset_hand_offset(arm)
                self.controller._base_rotation[arm] = None

    def process_controller_data(self, controller_data: Dict, arm: str, world: Optional[Any] = None) -> None:

        if not controller_data:
            return

        controller = self.controller
        if controller is None:
            return

        buttons = controller.get_buttons(controller_data)
        actions = self.control_mapping.map_buttons_to_actions(buttons, arm)

        self._process_button_actions(actions, buttons, arm, world)

        if self.teleop_enabled[arm]:
            position = controller.get_position(controller_data)
            if position:
                self.update_target_position(np.array(position), arm)

            axes = controller.get_axes(controller_data)
            if axes:
                joint_values = self.control_mapping.map_axes_to_joints(axes, arm)
                if joint_values:
                    self.robot.update_wrist_control(arm, "joystick", joint_values, 0)

            # Hand tracking: wrist orientation direct mapping
            if self._is_hand_track(controller):
                wrist_dof_indices = self._get_wrist_dof_indices(arm)
                if wrist_dof_indices and self._initial_joint_positions is not None:
                    wrist_targets = controller.get_wrist_joint_targets(
                        arm, self._initial_joint_positions, wrist_dof_indices
                    )
                    if wrist_targets:
                        self._apply_wrist_targets(wrist_targets, wrist_dof_indices)

    def _process_button_actions(self, actions: Dict, buttons: Dict, arm: str, world: Optional[Any] = None) -> None:
        prev_states = self.previous_button_states[arm]

        if "teleop_toggle" in actions and actions["teleop_toggle"]:
            key = "teleop_toggle"
            if not prev_states.get(key, False):
                self.toggle_teleop(arm)

        if self.teleop_enabled[arm]:
            if "gripper" in actions:
                if self._is_hand_track(self.controller):
                    # Hand tracking: only trigger gripper on pinch edge transitions.
                    # "close" on rising edge (not pinching -> pinching),
                    # "open" on falling edge (pinching -> not pinching).
                    # Animation is driven by the per-frame update_gripper("none", dt)
                    # in the main loop.
                    prev_gripper = prev_states.get("gripper", False)
                    curr_gripper = buttons.get("gripper", False)
                    if curr_gripper and not prev_gripper:
                        self.robot.update_gripper(arm, "close", 0)
                    elif not curr_gripper and prev_gripper:
                        self.robot.update_gripper(arm, "open", 0)
                else:
                    key = "button_x/a"
                    if buttons.get(key, False) and not prev_states.get(key, False):
                        self.robot.update_gripper(arm, actions["gripper"], 0)

            if "wrist_roll" in actions:
                direction = actions["wrist_roll"]
                self.robot.set_wrist_roll_state(arm, direction, True)
            else:
                self.robot.set_wrist_roll_state(arm, "negative", False)
                self.robot.set_wrist_roll_state(arm, "positive", False)
        else:
            # Clear wrist-roll state while teleop is disabled
            self.robot.set_wrist_roll_state(arm, "negative", False)
            self.robot.set_wrist_roll_state(arm, "positive", False)

        if "reset_pose" in actions and actions["reset_pose"]:
            key = "reset_pose"
            if not prev_states.get(key, False):
                self.reset_robot_pose(world)

        if "slow_homing" in actions and actions["slow_homing"]:
            if not prev_states.get("slow_homing", False):
                if self.teleop_enabled["left"] and self.teleop_enabled["right"]:
                    if self.slow_homing_manager:
                        self.slow_homing_manager.trigger()
                        Logger.info("Slow homing triggered")

        self.previous_button_states[arm] = buttons.copy()
        if "teleop_toggle" in actions:
            self.previous_button_states[arm]["teleop_toggle"] = actions["teleop_toggle"]
        if "reset_pose" in actions:
            self.previous_button_states[arm]["reset_pose"] = actions["reset_pose"]
        if "slow_homing" in actions:
            self.previous_button_states[arm]["slow_homing"] = actions["slow_homing"]

    def update_target_position(self, controller_pos: np.ndarray, arm: str):

        if not self.teleop_enabled[arm]:
            return

        controller_pos = np.asarray(controller_pos, dtype=np.float64)
        if controller_pos.shape != (3,) or not np.isfinite(controller_pos).all():
            Logger.warning(f"{arm} controller position rejected (non-finite or bad shape): {controller_pos}")
            return

        if self.control_mapping.position_offset[arm] is None:
            current_target_pos, _ = self.target_cubes[arm].get_world_pose()
            self.control_mapping.set_offset(arm, current_target_pos - controller_pos)
            self.previous_positions[arm] = controller_pos
            Logger.info(f"{arm.upper()} ARM offset calculated: {self.control_mapping.position_offset[arm]}")
            return

        if self.previous_positions[arm] is not None:
            displacement = np.linalg.norm(controller_pos - self.previous_positions[arm])
            if displacement > self.position_filter_threshold_max:
                Logger.warning(
                    f"{arm} PICO jump ignored. Displacement {displacement:.4f} "
                    f"> {self.position_filter_threshold_max:.4f}"
                )
                # ALWAYS update previous_positions even when filtered
                # (aligned with hand_track lines 2300, 2389:
                #  g_previous_hand_position_* = np.copy(hand_pos_np)
                #  is unconditionally executed after the if/else block)
                self.previous_positions[arm] = controller_pos
                return

        target_pos = self.control_mapping.map_position_to_target(controller_pos, arm)
        if not np.isfinite(target_pos).all():
            Logger.warning(f"{arm} target position rejected (non-finite after offset): {target_pos}")
            return

        current_pos, _ = self.target_cubes[arm].get_world_pose()
        if np.linalg.norm(target_pos - current_pos) > self.position_filter_threshold_min:
            self.target_cubes[arm].set_world_pose(position=target_pos)

        self.previous_positions[arm] = controller_pos

    def apply_ik_solutions(self, dt: float) -> None:

        for arm in ["left", "right"]:
            if self.teleop_enabled[arm]:
                self.apply_ik_solution(arm)

        for arm in ["left", "right"]:
            roll_state = self.robot.get_wrist_roll_state(arm)
            if roll_state["neg"]:
                self.robot.update_wrist_control(arm, "roll", "negative", dt)
            elif roll_state["pos"]:
                self.robot.update_wrist_control(arm, "roll", "positive", dt)

    def apply_ik_solution(self, arm: str) -> None:

        if not self.teleop_enabled[arm] or not self.ik_solver or not self.target_cubes[arm]:
            return

        target_pos, _ = self.target_cubes[arm].get_world_pose()
        ik_action, success = self.ik_solver.compute_inverse_kinematics(target_pos.astype(np.float64), None, arm)

        if success and ik_action:
            if not self._check_joint_jump(ik_action, arm):
                # Only apply IK when joint positions changed significantly.
                # Aligned with source project IK_DIFF_THRESHOLD = 1e-3 (hand_track line 2215).
                # When the target cube is stationary the IK result is nearly identical every
                # frame; skipping redundant apply_action() calls removes micro-jitter that
                # originates from the physics engine reacting to identical but not bit-equal
                # position commands. prev_joint_positions is always updated so the comparison
                # stays current (hand_track lines 2501-2503 pattern).
                prev = self.previous_joint_positions[arm]
                if prev is None or np.linalg.norm(ik_action.joint_positions - prev) > self.ik_diff_threshold:
                    self.robot.apply_joint_positions(ik_action.joint_positions, ik_action.joint_indices)
                self.previous_joint_positions[arm] = np.copy(ik_action.joint_positions)

    def _check_joint_jump(self, ik_action, arm: str) -> bool:

        current_positions = self.robot.get_joint_positions()
        if current_positions is None:
            return False

        proposed_positions = ik_action.joint_positions
        current_ik_positions = current_positions[ik_action.joint_indices]

        joint_deltas = np.abs(proposed_positions - current_ik_positions)

        if np.any(joint_deltas > self.joint_jump_threshold_rad):
            max_jump = np.rad2deg(np.max(joint_deltas))
            Logger.warning(f"{arm} IK solution ignored: max joint jump {max_jump:.2f} deg")
            return True

        return False

    def reset_robot_pose(self, world: Optional[Any] = None) -> None:

        if self.robot and self.robot.config.initial_joint_positions:
            Logger.info("Resetting robot to default pose...")

            num_joints = self.robot.robot_ref.num_dof
            joint_positions = np.zeros(num_joints, dtype=np.float32)

            for joint_name, angle in self.robot.config.initial_joint_positions.items():
                idx = self.robot.get_dof_index(joint_name)
                if idx != -1 and idx < num_joints:
                    joint_positions[idx] = angle

            self.robot.set_joint_positions(joint_positions)

            if world:
                for _ in range(10):
                    world.step(render=False)

            self.align_all_targets_to_effectors(world)

            Logger.info("Robot pose reset complete")

    def _get_wrist_dof_indices(self, arm: str) -> Optional[dict]:
        """Get wrist joint DOF indices for the specified arm (cached).

        arm_joints config contains joint names; convert to integer DOF indices
        via robot.get_dof_index(). Results are cached after first successful
        lookup to avoid repeated string→index resolution every frame.

        Args:
            arm: 'left' or 'right'.

        Returns:
            Dict with 'roll', 'yaw', 'pitch' keys mapping to integer DOF indices,
            or None if not available.
        """
        if arm in self._wrist_dof_indices_cache:
            return self._wrist_dof_indices_cache[arm]

        if not hasattr(self.robot, "config") or self.robot.config is None:
            return None
        arm_joints = self.robot.config.arm_joints.get(arm, [])
        if len(arm_joints) >= 3:
            roll_name = arm_joints[-3]
            yaw_name = arm_joints[-2]
            pitch_name = arm_joints[-1]
            roll_idx = self.robot.get_dof_index(roll_name)
            yaw_idx = self.robot.get_dof_index(yaw_name)
            pitch_idx = self.robot.get_dof_index(pitch_name)
            if roll_idx == -1 or yaw_idx == -1 or pitch_idx == -1:
                Logger.warning("Wrist joint DOF index not found for arm: %s", arm)
                return None
            result = {
                "roll": roll_idx,
                "yaw": yaw_idx,
                "pitch": pitch_idx,
            }
            self._wrist_dof_indices_cache[arm] = result
            return result
        return None

    def _apply_wrist_targets(self, targets: dict, dof_indices: dict) -> None:
        """Apply wrist joint targets directly to the robot.

        Args:
            targets: Dict with 'roll', 'yaw', 'pitch' target joint angles.
            dof_indices: Dict with 'roll', 'yaw', 'pitch' integer DOF index mappings.
        """
        indices = [dof_indices["roll"], dof_indices["yaw"], dof_indices["pitch"]]
        positions = np.array([targets["roll"], targets["yaw"], targets["pitch"]], dtype=np.float32)
        self.robot.apply_joint_positions(positions, indices)

    def reset_hand_track_calibration(self) -> None:
        """Reset hand tracking orientation calibration (keyboard C).

        Next frame will auto-recalibrate the wrist orientation base rotation.
        """
        if self._is_hand_track(self.controller):
            self.controller.reset_base_rotation()
