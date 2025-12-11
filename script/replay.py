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
from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "physics_prim_path": "/physicsScene",
    }
)

import argparse
import json
import os
import time
from typing import List, Optional

import carb
import carb.input
import h5py
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction

from comm_config.robot_config import RobotConfig
from comm_config.simulation_config import SimulationConfig
from robots.robot_factory import RobotFactory
from simulation.scene_manager import SceneManager
from utility.logger import Logger
import utility.system_utils as system_utils


class ReplayManager:
    """Manages HDF5 data replay with task-driven configuration.

    This class handles:
    - Loading and parsing HDF5 demonstration data
    - Robot and scene initialization using existing modules
    - Action expansion from 26-DOF to 38-DOF for mimic joints
    - Playback control (play/pause/reset/next demo)
    - Initial state restoration for robot joints and tracked objects
    """

    # Playback configuration
    RECORDING_FREQUENCY = 30.0  # Hz - must match recording frequency

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.keyboard_sub = None

        # Load task configuration
        self._load_task_config(args.task)

        # Initialize robot and simulation configurations
        project_root = system_utils.teleop_root_path()
        robot_config_path = os.path.join(
            project_root,
            "comm_config",
            "configs",
            self.task_config["robot"]["robot_cfg"],
        )
        sim_config_path = os.path.join(project_root, "comm_config", "configs", "simulation_config.json")

        self.robot_type = self.task_config["robot"]["robot_type"]
        self.robot_config = RobotConfig.from_json(robot_config_path)
        self.sim_config = SimulationConfig.from_json(sim_config_path, self.task_config)

        # Initialize world and robot
        self.world = World(
            stage_units_in_meters=self.sim_config.stage_units_in_meters,
            physics_dt=self.sim_config.physics_dt,
            rendering_dt=self.sim_config.rendering_dt,
        )
        self.world.scene.add_default_ground_plane()

        # Setup scene using SceneManager
        self.scene_manager = SceneManager(self.world, self.sim_config, task_config=self.task_config)
        self.scene_manager._load_scene_usd()
        self.scene_manager._create_object_manager()

        self.robot = RobotFactory.create_robot(self.robot_type, self.robot_config)
        self.robot.initialize(self.world, simulation_app=simulation_app, mode="replay")

        self.scene_manager._adjust_ground_plane()
        self.scene_manager._initialize_object_manager()

        # HDF5 data and playback state
        self.hdf5_file: Optional[h5py.File] = None
        self.demo_keys: List[str] = []
        self.current_demo_key: Optional[str] = None
        self.current_frame_index: int = 0
        self.total_frames: int = 0
        self.actions_data: Optional[np.ndarray] = None
        self.is_playing: bool = False
        self.replay_start_wall_time: float = 0.0

        # Robot joint indices for 38-DOF playback
        self.replay_joint_indices: Optional[np.ndarray] = None
        self._setup_joint_indices()

    def _load_task_config(self, task_name: str) -> None:
        """Load task configuration from JSON file.

        Args:
            task_name: Name of the task (e.g., "Task2_Cup_to_Cup_Transfer").

        Raises:
            FileNotFoundError: If task configuration file doesn't exist.
        """
        project_root = system_utils.teleop_root_path()
        task_config_path = os.path.join(project_root, "tasks", f"{task_name}.json")

        if not os.path.exists(task_config_path):
            raise FileNotFoundError(f"Task config not found: {task_config_path}")

        with open(task_config_path, "r", encoding="utf-8") as f:
            self.task_config = json.load(f)

        Logger.info(f"Loaded task configuration: {task_name}")

    def _setup_joint_indices(self) -> None:
        """Setup 38-DOF joint indices for replay (including mimic joints).

        This creates the mapping from 38-DOF action space to robot joint indices.
        The 38-DOF includes:
        - 14 arm joints (7 per arm)
        - 24 hand joints (12 per hand, with mimic followers)
        """
        # Define full 38-DOF joint names (matching simulation model)
        full_joint_names = [
            # Left arm (7 joints)
            "idx13_left_arm_joint1",
            "idx14_left_arm_joint2",
            "idx15_left_arm_joint3",
            "idx16_left_arm_joint4",
            "idx17_left_arm_joint5",
            "idx18_left_arm_joint6",
            "idx19_left_arm_joint7",
            # Right arm (7 joints)
            "idx20_right_arm_joint1",
            "idx21_right_arm_joint2",
            "idx22_right_arm_joint3",
            "idx23_right_arm_joint4",
            "idx24_right_arm_joint5",
            "idx25_right_arm_joint6",
            "idx26_right_arm_joint7",
            # Left hand (12 joints with mimic)
            "L_thumb_swing_joint",
            "L_thumb_1_joint",
            "L_thumb_2_joint",
            "L_thumb_3_joint",
            "L_index_1_joint",
            "L_index_2_joint",
            "L_middle_1_joint",
            "L_middle_2_joint",
            "L_ring_1_joint",
            "L_ring_2_joint",
            "L_little_1_joint",
            "L_little_2_joint",
            # Right hand (12 joints with mimic)
            "R_thumb_swing_joint",
            "R_thumb_1_joint",
            "R_thumb_2_joint",
            "R_thumb_3_joint",
            "R_index_1_joint",
            "R_index_2_joint",
            "R_middle_1_joint",
            "R_middle_2_joint",
            "R_ring_1_joint",
            "R_ring_2_joint",
            "R_little_1_joint",
            "R_little_2_joint",
        ]

        try:
            indices = [self.robot.get_dof_index(name) for name in full_joint_names]
            if any(idx == -1 for idx in indices):
                missing = [name for name, idx in zip(full_joint_names, indices) if idx == -1]
                Logger.error(f"Missing joints in robot model: {missing}")
                return

            self.replay_joint_indices = np.array(indices, dtype=np.int32)
            Logger.info(f"Successfully mapped {len(self.replay_joint_indices)} replay joints")
        except Exception as e:
            Logger.error(f"Failed to map joint indices: {e}")

    def _expand_26dof_to_38dof(self, action_26d: np.ndarray) -> np.ndarray:
        """Expand 26-DOF action to 38-DOF with mimic joint rules.

        The expansion maps:
        - 14 arm joints: 1-to-1 mapping
        - 12 hand joints (6 per hand): Each master joint drives 1-3 follower joints

        Left/Right hand structure (each 6 -> 12):
        - thumb_swing: 1 -> 1 (thumb_swing_joint)
        - thumb_1: 1 -> 3 (thumb_1_joint, thumb_2_joint, thumb_3_joint)
        - index_1: 1 -> 2 (index_1_joint, index_2_joint)
        - middle_1: 1 -> 2 (middle_1_joint, middle_2_joint)
        - ring_1: 1 -> 2 (ring_1_joint, ring_2_joint)
        - little_1: 1 -> 2 (little_1_joint, little_2_joint)

        Args:
            action_26d: 26-DOF action array [14 arms + 12 hands].

        Returns:
            38-DOF action array [14 arms + 24 hands].
        """
        if action_26d.shape[0] != 26:
            Logger.error(f"Expected 26-DOF action, got {action_26d.shape[0]}")
            return np.zeros(38, dtype=np.float32)

        action_38d = np.zeros(38, dtype=np.float32)

        # 1. Arms: Direct 1-to-1 mapping (14 joints)
        action_38d[0:14] = action_26d[0:14]

        # 2. Left hand: 6 -> 12 expansion
        action_38d[14] = action_26d[14]  # thumb_swing
        action_38d[15:18] = action_26d[15]  # thumb_1 -> thumb_1,2,3
        action_38d[18:20] = action_26d[16]  # index_1 -> index_1,2
        action_38d[20:22] = action_26d[17]  # middle_1 -> middle_1,2
        action_38d[22:24] = action_26d[18]  # ring_1 -> ring_1,2
        action_38d[24:26] = action_26d[19]  # little_1 -> little_1,2

        # 3. Right hand: 6 -> 12 expansion
        action_38d[26] = action_26d[20]  # thumb_swing
        action_38d[27:30] = action_26d[21]  # thumb_1 -> thumb_1,2,3
        action_38d[30:32] = action_26d[22]  # index_1 -> index_1,2
        action_38d[32:34] = action_26d[23]  # middle_1 -> middle_1,2
        action_38d[34:36] = action_26d[24]  # ring_1 -> ring_1,2
        action_38d[36:38] = action_26d[25]  # little_1 -> little_1,2

        return action_38d

    def load_hdf5_data(self) -> bool:
        """Load HDF5 dataset and extract demo keys.

        Returns:
            True if data loaded successfully, False otherwise.
        """
        hdf5_path = self.args.hdf5_path

        if not os.path.exists(hdf5_path):
            Logger.error(f"HDF5 file not found: {hdf5_path}")
            return False

        try:
            self.hdf5_file = h5py.File(hdf5_path, "r")
            self.demo_keys = sorted(list(self.hdf5_file.get("data", {}).keys()))

            if not self.demo_keys:
                Logger.error(f"No demo data found in {hdf5_path}")
                return False

            Logger.info(f"Loaded HDF5 dataset with {len(self.demo_keys)} demos: {self.demo_keys}")
            return True
        except Exception as e:
            Logger.error(f"Failed to load HDF5 file: {e}")
            return False

    def reset_and_load_demo(self, demo_key: str) -> None:
        """Reset scene and load initial state for specified demo.

        This restores:
        - Robot joint positions and velocities
        - Tracked object poses and velocities
        - Action data for playback

        Args:
            demo_key: Demo identifier (e.g., "demo_0").
        """
        self.is_playing = False

        if not self.hdf5_file or "data" not in self.hdf5_file or demo_key not in self.hdf5_file["data"]:
            Logger.error(f"Cannot load demo: {demo_key}")
            return

        Logger.info(f"Resetting scene to initial state of {demo_key}...")
        data_group = self.hdf5_file["data"][demo_key]
        initial_state = data_group["initial_state"]

        # 1. Restore robot joint state
        self._restore_robot_joint_state(initial_state)

        # 2. Restore tracked object states
        self._restore_tracked_objects_state(initial_state)

        # 3. Load action data
        self._load_action_data(data_group)

        # 4. Step physics to stabilize scene
        for _ in range(5):
            self.world.step(render=False)

        Logger.info(f"Scene reset complete for {demo_key}. Ready to replay.")

    def _restore_robot_joint_state(self, initial_state: h5py.Group) -> None:
        """Restore robot joint positions and velocities from initial state.

        Args:
            initial_state: HDF5 group containing initial state data.
        """
        joint_pos_key = "articulation/robot/joint_position"

        if joint_pos_key not in initial_state or self.replay_joint_indices is None:
            Logger.error("Cannot restore robot state: missing data or joint indices")
            return

        # Load 26-DOF initial positions from HDF5
        initial_pos_26d = initial_state[joint_pos_key][0]

        if initial_pos_26d.shape[0] != 26:
            Logger.error(f"Expected 26-DOF initial position, got {initial_pos_26d.shape[0]}")
            return

        # Expand to 38-DOF with mimic rules
        initial_pos_38d = self._expand_26dof_to_38dof(initial_pos_26d)

        # Apply to robot
        if hasattr(self.robot, "robot_view") and self.robot.robot_view is not None:
            full_joint_positions = self.robot.robot_view.get_joint_positions(clone=True)[0]
            full_joint_positions[self.replay_joint_indices] = initial_pos_38d
            self.robot.set_joint_positions(full_joint_positions)
            # Reset velocities using robot_ref (Isaac Sim Robot object)
            if self.robot.robot_ref and self.robot.robot_ref.is_valid():
                self.robot.robot_ref.set_joint_velocities(np.zeros_like(full_joint_positions))
            Logger.info("Robot joint state restored")
        else:
            Logger.error("Robot view not available for state restoration")

    def _restore_tracked_objects_state(self, initial_state: h5py.Group) -> None:
        """Restore poses and velocities of tracked objects.

        Args:
            initial_state: HDF5 group containing initial state data.
        """
        if not self.scene_manager.object_manager:
            return

        object_manager = self.scene_manager.object_manager

        rigid_object_group_key = "rigid_object"
        if rigid_object_group_key not in initial_state:
            Logger.warning("No rigid_object group found in initial_state")
            return

        rigid_object_group = initial_state[rigid_object_group_key]
        group_keys = list(rigid_object_group.keys())

        if not group_keys:
            Logger.warning("No group keys found in rigid_object group")
            return

        Logger.info(f"Found {len(group_keys)} object groups to restore: {group_keys}")

        # Process each group
        for group_key in group_keys:
            # Set the active group first
            if hasattr(object_manager, "set_active_group"):
                success = object_manager.set_active_group(group_key)
                if not success:
                    Logger.warning(f"Failed to set active group: {group_key}")
                    continue
                Logger.info(f"Set active group: {group_key}")
            
            # Get object names in this group
            group_data = rigid_object_group[group_key]
            object_names = list(group_data.keys())
            
            if not object_names:
                Logger.warning(f"No objects found in group {group_key}")
                continue
            
            Logger.info(f"Restoring {len(object_names)} objects in group '{group_key}': {object_names}")
            
            # Restore each object in the group
            for obj_name in object_names:
                pose_key = f"rigid_object/{group_key}/{obj_name}/root_pose"
                if pose_key not in initial_state:
                    Logger.warning(f"No initial pose data for object: {group_key}/{obj_name}")
                    continue

                # Get the tracked object from current_tracked_objects (populated by set_active_group)
                if obj_name not in object_manager.current_tracked_objects:
                    Logger.warning(f"Object {obj_name} not in current_tracked_objects for group {group_key}")
                    continue

                tracked_obj = object_manager.current_tracked_objects[obj_name]
                if not tracked_obj.handle or not tracked_obj.handle.is_valid():
                    Logger.warning(f"Invalid handle for object: {group_key}/{obj_name}")
                    continue

                obj_pose = np.array(initial_state[pose_key])[0]
                pos = obj_pose[:3]
                quat_xyzw = obj_pose[3:7]

                quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                tracked_obj.handle.set_world_pose(position=pos, orientation=quat_wxyz)
                tracked_obj.handle.set_linear_velocity(np.zeros(3))
                tracked_obj.handle.set_angular_velocity(np.zeros(3))

                Logger.info(f"Restored state for object: {group_key}/{obj_name} at position {pos}")

    def _load_action_data(self, data_group: h5py.Group) -> None:
        """Load action data for current demo.

        Args:
            data_group: HDF5 group containing demo data.
        """
        if "actions" not in data_group:
            Logger.error("No 'actions' dataset found in demo")
            self.actions_data = None
            self.total_frames = 0
            return

        self.actions_data = data_group["actions"][:]
        self.total_frames = self.actions_data.shape[0]
        self.current_frame_index = 0

        if self.actions_data.shape[1] != 26:
            Logger.error(f"Expected 26-DOF actions, got {self.actions_data.shape[1]}")
            self.actions_data = None
            self.total_frames = 0
            return

        Logger.info(f"Loaded {self.total_frames} frames of action data")

    def toggle_playback(self) -> None:
        """Toggle play/pause state."""
        if not self.current_demo_key:
            return

        # If at end, reset before playing
        if not self.is_playing and self.current_frame_index >= self.total_frames - 1:
            self.reset_current_demo()

        self.is_playing = not self.is_playing

        if self.is_playing:
            # Calculate start time based on current frame position
            self.replay_start_wall_time = time.time() - (self.current_frame_index / self.RECORDING_FREQUENCY)
            Logger.info("▶ REPLAY PLAYING")
        else:
            Logger.info("⏸ REPLAY PAUSED")

    def reset_current_demo(self) -> None:
        """Reset current demo to initial state."""
        if self.current_demo_key:
            Logger.info(f"🔄 RESET DEMO: {self.current_demo_key}")
            self.reset_and_load_demo(self.current_demo_key)

    def select_next_demo(self) -> None:
        """Select and load next demo in sequence."""
        if not self.demo_keys:
            return

        try:
            current_idx = self.demo_keys.index(self.current_demo_key)
        except (ValueError, TypeError):
            current_idx = -1

        next_idx = (current_idx + 1) % len(self.demo_keys)
        self.current_demo_key = self.demo_keys[next_idx]

        Logger.info(f"⏭ NEXT DEMO: {self.current_demo_key}")
        self.reset_and_load_demo(self.current_demo_key)

    def keyboard_event_handler(self, event, *args, **kwargs) -> bool:
        """Handle keyboard events for playback control.

        Args:
            event: Keyboard event from carb.input.

        Returns:
            True to continue event handling.
        """
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.S:
                self.toggle_playback()
            elif event.input == carb.input.KeyboardInput.R:
                self.reset_current_demo()
            elif event.input == carb.input.KeyboardInput.N:
                self.select_next_demo()
        return True

    def setup(self) -> bool:
        """Setup replay manager and load initial demo.

        Returns:
            True if setup successful, False otherwise.
        """
        # Load HDF5 data
        if not self.load_hdf5_data():
            return False

        # Subscribe to keyboard events
        app_input = carb.input.acquire_input_interface()
        self.keyboard_sub = app_input.subscribe_to_keyboard_events(None, self.keyboard_event_handler)

        # Load first demo
        if self.demo_keys:
            self.current_demo_key = self.demo_keys[0]
            self.reset_and_load_demo(self.current_demo_key)

        # Print controls
        Logger.info("\n" + "=" * 50)
        Logger.info("REPLAY CONTROLS:")
        Logger.info("  'S' key: Play/Pause replay")
        Logger.info("  'R' key: Reset current demo")
        Logger.info("  'N' key: Load next demo")
        Logger.info("=" * 50 + "\n")

        return True

    def update(self) -> None:
        """Update replay state and apply actions.

        This should be called in the simulation loop.
        """
        if not self.is_playing or self.actions_data is None or self.replay_joint_indices is None:
            return

        # Calculate target frame based on elapsed time
        elapsed_time = time.time() - self.replay_start_wall_time
        target_frame = int(elapsed_time * self.RECORDING_FREQUENCY)

        # Skip frames if we're behind
        if target_frame > self.current_frame_index:
            self.current_frame_index = target_frame

            # Check if replay finished
            if self.current_frame_index >= self.total_frames:
                self.is_playing = False
                self.current_frame_index = self.total_frames - 1
                Logger.info(f"✓ REPLAY FINISHED: {self.current_demo_key}")
                return

            # Apply action for current frame
            action_26d = self.actions_data[self.current_frame_index]
            action_38d = self._expand_26dof_to_38dof(action_26d)

            action = ArticulationAction(joint_positions=action_38d, joint_indices=self.replay_joint_indices)
            self.robot.robot_ref.get_articulation_controller().apply_action(action)

    def run(self) -> None:
        """Main replay loop."""
        if not self.setup():
            Logger.error("Replay setup failed. Exiting.")
            return

        Logger.info("Starting replay simulation...")
        self.world.play()

        try:
            while simulation_app.is_running():
                self.world.step(render=True)
                self.update()
        except KeyboardInterrupt:
            Logger.info("Replay interrupted by user")
        except Exception as e:
            Logger.error(f"Replay error: {e}")
            import traceback

            traceback.print_exc()

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.keyboard_sub and hasattr(self.keyboard_sub, "unsubscribe"):
            self.keyboard_sub.unsubscribe()

        if self.hdf5_file:
            self.hdf5_file.close()
            Logger.info("HDF5 file closed")


def main():
    """Main entry point for replay script."""
    parser = argparse.ArgumentParser(
        description="Replay robot demonstration data from HDF5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay Task 2 data
  python script/replay.py --task Task2_Cup_to_Cup_Transfer --hdf5_path /path/to/data.hdf5
  
  # Replay Task 4 data
  python script/replay.py --task Task4_Can_Stacking --hdf5_path /path/to/can_data.hdf5

Controls:
  S - Play/Pause replay
  R - Reset current demo
  N - Next demo
        """,
    )
    parser.add_argument(
        "--task", type=str, required=True, help="Task name (e.g., Task2_Cup_to_Cup_Transfer, Task4_Can_Stacking)"
    )
    parser.add_argument(
        "--hdf5_path", type=str, required=True, help="Path to HDF5 dataset file containing recorded demonstrations"
    )

    args, _ = parser.parse_known_args()

    replay_manager = None
    try:
        replay_manager = ReplayManager(args)
        replay_manager.run()
    except Exception as e:
        Logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if replay_manager:
            replay_manager.cleanup()

        if simulation_app.is_running():
            simulation_app.close()


if __name__ == "__main__":
    main()
