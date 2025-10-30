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
import h5py
import numpy as np
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import os
from dataclasses import dataclass
from utility.logger import Logger
from omni.isaac.core.utils.xforms import get_world_pose
from scipy.spatial.transform import Rotation as R


# ============================================================================
# Constants
# ============================================================================
QUAT_WXYZ_TO_XYZW_INDICES = [1, 2, 3, 0]  # Conversion indices
RGBA_TO_BGR_INDICES = [2, 1, 0]  # BGR channel order
IMAGE_SCALE_FACTOR = 255  # Scale factor for uint8 conversion

# HDF5 dataset configuration
HDF5_COMPRESSION_METHOD = "gzip"
HDF5_COMPRESSION_LEVEL = 4
HDF5_FLOAT_DTYPE = "float32"
HDF5_IMAGE_DTYPE = "uint8"


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class PoseData:
    """Encapsulates pose data (position + quaternion).
    
    Attributes:
        position: 3D position vector (x, y, z).
        quaternion_wxyz: Quaternion in wxyz format [w, x, y, z].
    """
    position: np.ndarray  # shape: (3,)
    quaternion_wxyz: np.ndarray  # shape: (4,)
    
    @property
    def quaternion_xyzw(self) -> np.ndarray:
        """Convert quaternion from wxyz to xyzw format.
        
        Returns:
            Quaternion in xyzw format [x, y, z, w].
        """
        return self.quaternion_wxyz[QUAT_WXYZ_TO_XYZW_INDICES]
    
    def to_pose_array(self) -> np.ndarray:
        """Convert to concatenated pose array [pos(3), quat_xyzw(4)].
        
        Returns:
            7D pose array.
        """
        return np.concatenate([self.position, self.quaternion_xyzw])


@dataclass
class VelocityData:
    """Encapsulates velocity data (linear + angular).
    
    Attributes:
        linear: 3D linear velocity vector.
        angular: 3D angular velocity vector.
    """
    linear: np.ndarray  # shape: (3,)
    angular: np.ndarray  # shape: (3,)
    
    def to_velocity_array(self) -> np.ndarray:
        """Convert to concatenated velocity array [linear(3), angular(3)].
        
        Returns:
            6D velocity array.
        """
        return np.concatenate([self.linear, self.angular])


# ============================================================================
# Helper Functions
# ============================================================================
def pose_to_mat44(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Converts position and (w,x,y,z) quaternion to a 4x4 transformation matrix.
    
    Args:
        pos: 3D position vector.
        quat_wxyz: Quaternion in wxyz format.
        
    Returns:
        4x4 homogeneous transformation matrix.
    """
    mat = np.eye(4)
    # Use scipy to convert (w,x,y,z) to a rotation matrix, note that scipy uses (x,y,z,w) format
    mat[:3, :3] = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    mat[:3, 3] = pos
    return mat


class DataRecorder:

    def __init__(
        self, 
        robot, 
        output_dir: str, 
        cameras: Optional[Dict] = None, 
        object_manager = None, 
        teleop_manager = None,
        task_name: Optional[str] = None
    ):
        """Initialize DataRecorder with optional task name for file naming.
        
        Args:
            robot: The robot instance.
            output_dir: The directory to save data.
            cameras: A dictionary of cameras {name: Camera}.
            object_manager: ObjectManager instance (for dynamically tracking currently active objects).
            teleop_manager: TeleopManager instance (for getting teleoperation target poses).
            task_name: Optional task name to include in the filename (e.g., "Task1_Kitchen_Cleanup").
        """
        self.robot = robot

        self.is_recording = False
        self.current_demo_index = 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with optional task name
        if task_name:
            filename = f"recorded_data_{task_name}_{timestamp}.hdf5"
        else:
            filename = f"recorded_data_{timestamp}.hdf5"
        
        self.dataset_filepath = os.path.join(output_dir, filename)
        self.step_data_buffer = {}
        self.initial_state_buffer = {}

        self.cameras = cameras if cameras else {}
        self.object_manager = object_manager

        self.teleop_manager = teleop_manager

        self._setup_joint_config()

    def _setup_joint_config(self):

        self.left_arm_joints = self.robot.config.arm_joints["left"]
        self.right_arm_joints = self.robot.config.arm_joints["right"]
        self.left_hand_joints = self.robot.config.hand_joints["left"]
        self.right_hand_joints = self.robot.config.hand_joints["right"]

        # Get joint indices
        self.left_arm_indices = self._get_joint_indices(self.left_arm_joints)
        self.right_arm_indices = self._get_joint_indices(self.right_arm_joints)
        self.left_hand_indices = self._get_joint_indices(self.left_hand_joints)
        self.right_hand_indices = self._get_joint_indices(self.right_hand_joints)

    def _get_joint_indices(self, joint_names: List[str]) -> List[int]:

        indices = []
        for name in joint_names:
            idx = self.robot.get_dof_index(name)
            if idx != -1:
                indices.append(idx)
        return indices

    def setup(self):
        Logger.info(f"[Recorder] Setup complete. Ready to record to '{self.dataset_filepath}'.")
        if self.object_manager:
            tracked_count = len(self.object_manager.recorder_tracked_objects)
            Logger.info(f"[Recorder] ObjectManager integrated, tracking {tracked_count} objects dynamically")

        Logger.info("\n" + "=" * 50)
        Logger.info("RECORDER CONTROLS:")
        Logger.info("  'S' key: Start/Stop recording")
        Logger.info("  'R' key: Reset current demo")
        Logger.info("  'N' key: Save and start next demo")
        Logger.info("=" * 50 + "\n")

    def toggle_recording(self):

        self.is_recording = not self.is_recording

        if self.is_recording:
            self.capture_initial_state()
            Logger.info(f"\n[DEMO {self.current_demo_index}] --> RECORDING STARTED...")
        else:
            Logger.info(f"\n[DEMO {self.current_demo_index}] --> RECORDING PAUSED.")

    def capture_initial_state(self):
        """Capture initial state of robot, objects, and end effectors."""
        self.initial_state_buffer = {}
        
        self._capture_robot_joint_state()
        self._capture_robot_root_state()
        self._capture_objects_initial_state()
        self._capture_end_effectors_initial_state()

    def _capture_robot_joint_state(self):
        """Capture robot joint positions and velocities."""
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()

        if joint_pos is not None:
            all_indices = (
                self.left_arm_indices + self.right_arm_indices + self.left_hand_indices + self.right_hand_indices
            )
            selected_pos = joint_pos[all_indices]
            self.initial_state_buffer["articulation/robot/joint_position"] = selected_pos.reshape(1, -1)

        if joint_vel is not None:
            all_indices = (
                self.left_arm_indices + self.right_arm_indices + self.left_hand_indices + self.right_hand_indices
            )
            selected_vel = joint_vel[all_indices]
            self.initial_state_buffer["articulation/robot/joint_velocity"] = selected_vel.reshape(1, -1)

    def _capture_robot_root_state(self):
        """Capture robot root pose and velocity."""
        root_pos, root_quat_wxyz = self.robot.get_world_pose()
        root_lin_vel = self.robot.get_linear_velocity()
        root_ang_vel = self.robot.get_angular_velocity()

        if root_pos is not None and root_quat_wxyz is not None:
            root_quat_xyzw = self._convert_quat_wxyz_to_xyzw(root_quat_wxyz)
            self.initial_state_buffer["articulation/robot/root_pose"] = np.concatenate(
                [root_pos, root_quat_xyzw]
            ).reshape(1, -1)

        if root_lin_vel is not None and root_ang_vel is not None:
            self.initial_state_buffer["articulation/robot/root_velocity"] = np.concatenate(
                [root_lin_vel, root_ang_vel]
            ).reshape(1, -1)

    def _capture_objects_initial_state(self):
        """Capture initial pose and velocity of tracked objects."""
        if not self.object_manager or not self.object_manager.current_tracked_objects:
            return

        for obj_name, tracked_obj in self.object_manager.current_tracked_objects.items():
            try:
                if not tracked_obj.handle or not tracked_obj.handle.is_valid():
                    continue

                obj_pos, obj_quat_wxyz = tracked_obj.handle.get_world_pose()
                obj_lin_vel = tracked_obj.handle.get_linear_velocity()
                obj_ang_vel = tracked_obj.handle.get_angular_velocity()

                if obj_pos is not None and obj_quat_wxyz is not None:
                    obj_quat_xyzw = self._convert_quat_wxyz_to_xyzw(obj_quat_wxyz)
                    self.initial_state_buffer[f"rigid_object/{obj_name}/root_pose"] = np.concatenate(
                        [obj_pos, obj_quat_xyzw]
                    ).reshape(1, -1)

                if obj_lin_vel is not None and obj_ang_vel is not None:
                    self.initial_state_buffer[f"rigid_object/{obj_name}/root_velocity"] = np.concatenate(
                        [obj_lin_vel, obj_ang_vel]
                    ).reshape(1, -1)
            except Exception as e:
                Logger.error(f"[Recorder] Failed to get initial state for {obj_name}: {e}")

    def _capture_end_effectors_initial_state(self):
        """Capture end effector poses using IK solver."""
        if not hasattr(self.robot, "ik_solver"):
            return

        try:
            self._capture_single_end_effector_state("left")
            self._capture_single_end_effector_state("right")
        except Exception as e:
            Logger.error(f"[Recorder] Failed to get end effector poses: {e}")

    def _capture_single_end_effector_state(self, arm: str):
        """Capture end effector state for a single arm.
        
        Args:
            arm: Either "left" or "right"
        """
        ee_pos, ee_rot = self.robot.ik_solver[arm].compute_end_effector_pose()
        if ee_pos is not None and ee_rot is not None:
            ee_quat = R.from_matrix(ee_rot).as_quat()  # xyzw
            self.initial_state_buffer[f"eef/{arm}_eef_pose"] = np.concatenate([ee_pos, ee_quat]).reshape(1, -1)

    def _convert_quat_wxyz_to_xyzw(self, quat_wxyz: np.ndarray) -> np.ndarray:
        """Convert quaternion from wxyz format to xyzw format.
        
        Args:
            quat_wxyz: Quaternion in wxyz format [w, x, y, z]
            
        Returns:
            Quaternion in xyzw format [x, y, z, w]
        """
        return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

    def collect_step_data(self) -> None:
        """Collect all step data for the current timestep.
        
        This is the main coordinator function that collects:
        - Robot actions and joint states
        - Robot root states (pose and velocity)
        - Rigid object states
        - End effector states
        - Camera images
        - Robot links states
        - Hand joint states
        - Target cube poses
        
        Raises:
            RuntimeError: If robot is invalid or joint data cannot be retrieved.
        """
        # Early return: validate robot
        if not self.robot or not self.robot.robot_ref.is_valid():
            return
        
        # Get joint data (required for most operations)
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        
        if joint_pos is None or joint_vel is None:
            return
        
        # Collect all data types
        action_data, state_pos = self._collect_robot_actions(joint_pos)
        self._collect_robot_joint_states(joint_pos, joint_vel, action_data, state_pos)
        self._collect_robot_root_states()
        self._collect_rigid_objects_states()
        self._collect_end_effector_states()
        self._collect_camera_images()
        self._collect_robot_links_states()
        self._collect_hand_joint_states()
        self._collect_target_cube_poses()
    
    def _collect_robot_actions(self, joint_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Collect 26-dimensional hybrid robot actions (arms + hands).
        
        Args:
            joint_pos: Full robot joint positions array.
            
        Returns:
            Tuple of (action_data, state_pos):
                - action_data: 26D action array [left_arm(7), right_arm(7), left_hand(6), right_hand(6)]
                - state_pos: Selected joint positions for state recording
        """
        # Collect arm actions from joint positions
        left_arm_action = joint_pos[self.left_arm_indices]
        right_arm_action = joint_pos[self.right_arm_indices]
        
        # Collect hand actions from gripper angles
        left_hand_action = self._extract_hand_action("left")
        right_hand_action = self._extract_hand_action("right")
        
        # Merge into complete action
        action_data = np.concatenate([
            left_arm_action,
            right_arm_action,
            left_hand_action,
            right_hand_action
        ])
        
        self._add_to_buffer("actions", action_data)
        
        # Prepare state positions for return
        all_indices = (
            self.left_arm_indices + 
            self.right_arm_indices + 
            self.left_hand_indices + 
            self.right_hand_indices
        )
        state_pos = joint_pos[all_indices]
        
        return action_data, state_pos
    
    def _extract_hand_action(self, arm: str) -> np.ndarray:
        """Extract hand action for a specific arm from current gripper angles.
        
        Args:
            arm: Arm identifier ("left" or "right").
            
        Returns:
            Hand action array for the specified arm.
        """
        hand_joints = self.left_hand_joints if arm == "left" else self.right_hand_joints
        hand_indices_len = len(self.left_hand_indices if arm == "left" else self.right_hand_indices)
        hand_action = np.zeros(hand_indices_len)
        
        if self.robot.gripper_current_angles[arm] is not None:
            for i, joint_name in enumerate(hand_joints):
                if joint_name in self.robot.gripper_all_joint_names[arm]:
                    idx = self.robot.gripper_all_joint_names[arm].index(joint_name)
                    hand_action[i] = self.robot.gripper_current_angles[arm][idx]
        
        return hand_action
    
    def _collect_robot_joint_states(
        self, 
        joint_pos: np.ndarray, 
        joint_vel: np.ndarray,
        action_data: np.ndarray,
        state_pos: np.ndarray
    ) -> None:
        """Collect robot joint positions and velocities for states and observations.
        
        Args:
            joint_pos: Full joint positions array.
            joint_vel: Full joint velocities array.
            action_data: Precomputed action data.
            state_pos: Precomputed state positions.
        """
        all_indices = (
            self.left_arm_indices + 
            self.right_arm_indices + 
            self.left_hand_indices + 
            self.right_hand_indices
        )
        state_vel = joint_vel[all_indices]
        
        self._add_to_buffer("states/articulation/robot/joint_position", state_pos)
        self._add_to_buffer("states/articulation/robot/joint_velocity", state_vel)
        self._add_to_buffer("obs/actions", action_data)
        self._add_to_buffer("obs/robot_joint_pos", state_pos)
    
    def _collect_robot_root_states(self) -> None:
        """Collect robot root pose and velocity for states and observations."""
        root_pos, root_quat_wxyz = self.robot.get_world_pose()
        root_lin_vel = self.robot.get_linear_velocity()
        root_ang_vel = self.robot.get_angular_velocity()
        
        # Early return if pose data is invalid
        if root_pos is None or root_quat_wxyz is None:
            return
        
        # Convert quaternion format
        root_quat_xyzw = self._convert_quat_wxyz_to_xyzw(root_quat_wxyz)
        
        # Save root pose: [pos(3), quat_xyzw(4)] = 7D
        root_pose = np.concatenate([root_pos, root_quat_xyzw])
        self._add_to_buffer("states/articulation/robot/root_pose", root_pose)
        
        # Save root velocity: [lin_vel(3), ang_vel(3)] = 6D
        if root_lin_vel is not None and root_ang_vel is not None:
            root_velocity = np.concatenate([root_lin_vel, root_ang_vel])
            self._add_to_buffer("states/articulation/robot/root_velocity", root_velocity)
        
        # Save observation data (for compatibility with original scripts)
        self._add_to_buffer("obs/robot_root_pos", root_pos)
        self._add_to_buffer("obs/robot_root_rot", root_quat_xyzw)
    
    def _collect_rigid_objects_states(self) -> None:
        """Collect pose and velocity states for all tracked rigid objects."""
        if not self.object_manager or not self.object_manager.current_tracked_objects:
            return

        for obj_name, tracked_obj in self.object_manager.current_tracked_objects.items():
            try:
                self._collect_single_object_state(obj_name, tracked_obj)
            except Exception as e:
                Logger.error(f"[Recorder] Failed to get state for {obj_name}: {e}")
    
    def _collect_single_object_state(self, obj_name: str, tracked_obj) -> None:
        """Collect state data for a single rigid object.
        
        Args:
            obj_name: Name of the object.
            tracked_obj: Tracked object instance with handle.
            
        Raises:
            ValueError: If object handle is invalid.
        """
        if not tracked_obj.handle or not tracked_obj.handle.is_valid():
            return
        
        obj_pos, obj_quat_wxyz = tracked_obj.handle.get_world_pose()
        obj_lin_vel = tracked_obj.handle.get_linear_velocity()
        obj_ang_vel = tracked_obj.handle.get_angular_velocity()
        
        if obj_pos is None or obj_quat_wxyz is None:
            return
        
        obj_quat_xyzw = self._convert_quat_wxyz_to_xyzw(obj_quat_wxyz)
        
        # Save pose: [pos(3), quat_xyzw(4)] = 7D
        obj_pose = np.concatenate([obj_pos, obj_quat_xyzw])
        self._add_to_buffer(f"states/rigid_object/{obj_name}/root_pose", obj_pose)
        
        # Save velocity: [lin_vel(3), ang_vel(3)] = 6D
        if obj_lin_vel is not None and obj_ang_vel is not None:
            obj_velocity = np.concatenate([obj_lin_vel, obj_ang_vel])
            self._add_to_buffer(f"states/rigid_object/{obj_name}/root_velocity", obj_velocity)
            
            # Save observation data (fully consistent with the original script)
            self._add_to_buffer(f"obs/{obj_name}_pos", obj_pos)
            self._add_to_buffer(f"obs/{obj_name}_rot", obj_quat_xyzw)
            
            # Complete object state: pose + velocity = 13D
            obj_full_state = np.concatenate([obj_pos, obj_quat_xyzw, obj_lin_vel, obj_ang_vel])
            self._add_to_buffer(f"obs/{obj_name}", obj_full_state)
            
            # Save 4x4 transformation matrix
            self._add_to_buffer(
                f"obs/datagen_info/object_pose/{obj_name}",
                pose_to_mat44(obj_pos, obj_quat_wxyz)
            )
    
    def _collect_end_effector_states(self) -> None:
        """Collect end effector poses from robot links and IK solver."""
        try:
            robot_prim_path = self.robot.config.prim_path
            
            # Collect observation data from prim paths (ground truth)
            self._collect_single_eef_obs("right", robot_prim_path)
            self._collect_single_eef_obs("left", robot_prim_path)
            
            # Collect state data from IK solver (for consistency)
            self._collect_eef_states_from_ik()
            
        except Exception as e:
            Logger.error(f"[Recorder] Failed to get end effector poses: {e}")
    
    def _collect_single_eef_obs(self, arm: str, robot_prim_path: str) -> None:
        """Collect end effector observation data for a single arm.
        
        Args:
            arm: Arm identifier ("left" or "right").
            robot_prim_path: Base prim path of the robot (unused, kept for compatibility).
        """
        ee_link_path = self.robot.get_end_effector_link_path(arm)
        eef_pos, eef_quat_wxyz = get_world_pose(ee_link_path)
        eef_quat_xyzw = self._convert_quat_wxyz_to_xyzw(eef_quat_wxyz)
        
        self._add_to_buffer(f"obs/{arm}_eef_pos", eef_pos)
        self._add_to_buffer(f"obs/{arm}_eef_quat", eef_quat_xyzw)
        self._add_to_buffer(
            f"obs/datagen_info/eef_pose/{arm}",
            pose_to_mat44(eef_pos, eef_quat_wxyz)
        )
    
    def _collect_eef_states_from_ik(self) -> None:
        """Collect end effector states from IK solver for state recording."""
        if not hasattr(self.robot, "ik_solver"):
            return
        
        for arm in ["left", "right"]:
            ee_pos, ee_rot = self.robot.ik_solver[arm].compute_end_effector_pose()
            if ee_pos is not None and ee_rot is not None:
                ee_quat = R.from_matrix(ee_rot).as_quat()  # xyzw format
                ee_pose = np.concatenate([ee_pos, ee_quat])
                self._add_to_buffer(f"states/eef/{arm}_eef_pose", ee_pose)
    
    def _collect_camera_images(self) -> None:
        """Collect BGR images from all cameras."""
        for cam_name, camera in self.cameras.items():
            try:
                if not camera or not hasattr(camera, "get_rgba"):
                    continue
                
                rgba = camera.get_rgba()
                
                if rgba is None or rgba.shape[2] != 4:
                    Logger.warning(f"[Recorder] {cam_name} did not return valid RGBA data.")
                    continue
                
                # Extract BGR channels (indices 2, 1, 0) - fully consistent with the original script
                bgr_image = rgba[..., RGBA_TO_BGR_INDICES]
                self._add_to_buffer(f"obs/{cam_name}_bgr", bgr_image)
                
            except Exception as e:
                Logger.error(f"[Recorder] Failed to get image from {cam_name}: {e}")
    
    def _collect_robot_links_states(self) -> None:
        """Collect poses and velocities of all robot links."""
        if not hasattr(self.robot, "robot_view") or self.robot.robot_view is None:
            return
        
        try:
            if not self.robot.robot_view.is_valid() or self.robot.robot_view.count == 0:
                return
            
            poses_tuple = self.robot.robot_view.get_world_poses()
            velocities_array = self.robot.robot_view.get_velocities()
            
            if poses_tuple is None or velocities_array is None:
                return
            
            # poses_tuple[0]: positions (N, 3)
            # poses_tuple[1]: orientations wxyz (N, 4) -> convert to xyzw
            # velocities_array: (N, 6) [linear(3), angular(3)]
            links_state_array = np.concatenate([
                poses_tuple[0],  # positions
                poses_tuple[1][:, QUAT_WXYZ_TO_XYZW_INDICES],  # quaternions wxyz -> xyzw
                velocities_array[:, :3],  # linear velocities
                velocities_array[:, 3:]   # angular velocities
            ], axis=1)
            
            self._add_to_buffer("obs/robot_links_state", links_state_array)
            
        except Exception as e:
            Logger.error(f"[Recorder] Failed to get robot links state: {e}")
    
    def _collect_hand_joint_states(self) -> None:
        """Collect detailed finger joint positions for both hands."""
        if not hasattr(self.robot, "gripper_current_angles"):
            return
        
        hand_joint_positions = []
        
        # Left hand joints
        if self.robot.gripper_current_angles["left"] is not None:
            hand_joint_positions.extend(self.robot.gripper_current_angles["left"])
        
        # Right hand joints
        if self.robot.gripper_current_angles["right"] is not None:
            hand_joint_positions.extend(self.robot.gripper_current_angles["right"])
        
        if hand_joint_positions:
            hand_joint_state = np.array(hand_joint_positions)
            self._add_to_buffer("obs/hand_joint_state", hand_joint_state)
    
    def _collect_target_cube_poses(self) -> None:
        """Collect target cube poses from teleoperation manager."""
        if not self.teleop_manager or not self.teleop_manager.target_cubes:
            return
        
        try:
            for arm in ["left", "right"]:
                if self.teleop_manager.target_cubes[arm] is None:
                    continue
                
                pos, quat_wxyz = self.teleop_manager.target_cubes[arm].get_world_pose()
                if pos is not None and quat_wxyz is not None:
                    self._add_to_buffer(
                        f"obs/datagen_info/target_eef_pose/{arm}",
                        pose_to_mat44(pos, quat_wxyz)
                    )
        except Exception as e:
            Logger.error(f"[Recorder] Failed to get target cube poses: {e}")

    def _add_to_buffer(self, key: str, data: np.ndarray) -> None:
        """Add data to the step data buffer.
        
        Args:
            key: Buffer key for the data.
            data: Data array to store.
        """
        if key not in self.step_data_buffer:
            self.step_data_buffer[key] = []
        self.step_data_buffer[key].append(data)

    def save_and_start_next(self) -> None:
        """Save current demonstration data to HDF5 and prepare for next demo.
        
        This function:
        1. Stops recording if active
        2. Validates data buffer
        3. Saves initial state and step data to HDF5
        4. Increments demo counter
        5. Clears buffers for next demo
        
        Raises:
            IOError: If HDF5 file cannot be written.
        """
        # Stop recording if active
        if self.is_recording:
            self.toggle_recording()
        
        # Validate data buffer
        if not self.step_data_buffer:
            Logger.warning(f"\n[RECORDER] No data to save for demo {self.current_demo_index}.")
            return
        
        Logger.info(f"\n[DEMO {self.current_demo_index}] --> SAVING...")
        
        try:
            self._save_demo_to_hdf5()
            Logger.info(f"[DEMO {self.current_demo_index}] --> SAVED successfully.")
            self.current_demo_index += 1
            self.clear_buffers()
        except Exception as e:
            Logger.error(f"\n[RECORDER] ERROR saving: {e}")
    
    def _save_demo_to_hdf5(self) -> None:
        """Save demonstration data to HDF5 file.
        
        Creates a new demo group and saves:
        - Initial state (articulation, rigid objects, end effectors)
        - Step data (observations, states, actions)
        
        Raises:
            IOError: If file operations fail.
        """
        with h5py.File(self.dataset_filepath, "a") as f:
            demo_group = f.create_group(f"data/demo_{self.current_demo_index}")
            
            # Save initial state
            self._save_initial_state(demo_group)
            
            # Save step data
            self._save_step_data(demo_group)
    
    def _save_initial_state(self, demo_group: h5py.Group) -> None:
        """Save initial state data to HDF5 demo group.
        
        Args:
            demo_group: HDF5 group for the current demonstration.
        """
        initial_group = demo_group.create_group("initial_state")
        for key, data in self.initial_state_buffer.items():
            initial_group.create_dataset(key, data=data, dtype=HDF5_FLOAT_DTYPE)
    
    def _save_step_data(self, demo_group: h5py.Group) -> None:
        """Save step data to HDF5 demo group.
        
        Args:
            demo_group: HDF5 group for the current demonstration.
        """
        for key, data_list in self.step_data_buffer.items():
            if not data_list:
                continue
            
            # Convert list to array with appropriate dtype
            data_array = self._prepare_data_array(data_list)
            
            # Save to appropriate group based on key prefix
            self._save_data_by_key(demo_group, key, data_array)
    
    def _prepare_data_array(self, data_list: List[np.ndarray]) -> np.ndarray:
        """Prepare data list for HDF5 storage.
        
        Args:
            data_list: List of numpy arrays to convert.
            
        Returns:
            Consolidated numpy array with appropriate dtype.
        """
        sample_data = data_list[0]
        
        if sample_data.dtype == np.uint8:
            # Image data
            return np.array(data_list, dtype=HDF5_IMAGE_DTYPE)
        else:
            # Numerical data
            return np.array(data_list, dtype=HDF5_FLOAT_DTYPE)
    
    def _save_data_by_key(self, demo_group: h5py.Group, key: str, data_array: np.ndarray) -> None:
        """Save data array to HDF5 based on key prefix.
        
        Args:
            demo_group: HDF5 group for the current demonstration.
            key: Data key determining storage location.
            data_array: Data to save.
        """
        if key.startswith("obs/"):
            self._save_observation_data(demo_group, key, data_array)
        elif key.startswith("states/"):
            self._save_state_data(demo_group, key, data_array)
        else:
            demo_group.create_dataset(key, data=data_array)
    
    def _save_observation_data(self, demo_group: h5py.Group, key: str, data_array: np.ndarray) -> None:
        """Save observation data to obs/ group.
        
        Args:
            demo_group: HDF5 group for the current demonstration.
            key: Observation key (with "obs/" prefix).
            data_array: Observation data to save.
        """
        group = demo_group.require_group("obs")
        dataset_name = key.replace("obs/", "")
        
        # Handle nested groups (e.g., "datagen_info/object_pose/...")
        if "/" in dataset_name:
            self._save_nested_dataset(group, dataset_name, data_array)
        else:
            self._create_dataset_with_compression(group, dataset_name, data_array)
    
    def _save_state_data(self, demo_group: h5py.Group, key: str, data_array: np.ndarray) -> None:
        """Save state data to states/ group.
        
        Args:
            demo_group: HDF5 group for the current demonstration.
            key: State key (with "states/" prefix).
            data_array: State data to save.
        """
        group = demo_group.require_group("states")
        dataset_name = key.replace("states/", "")
        
        # Handle nested groups (e.g., "articulation/robot/...")
        if "/" in dataset_name:
            self._save_nested_dataset(group, dataset_name, data_array)
        else:
            group.create_dataset(dataset_name, data=data_array)
    
    def _save_nested_dataset(self, parent_group: h5py.Group, dataset_path: str, data_array: np.ndarray) -> None:
        """Save dataset in nested group hierarchy.
        
        Args:
            parent_group: Parent HDF5 group.
            dataset_path: Path with forward slashes (e.g., "datagen_info/object_pose/cup").
            data_array: Data to save.
        """
        parts = dataset_path.split("/")
        
        # Navigate/create nested groups
        group = parent_group
        for part in parts[:-1]:
            group = group.require_group(part)
        
        # Create final dataset
        dataset_name = parts[-1]
        self._create_dataset_with_compression(group, dataset_name, data_array)
    
    def _create_dataset_with_compression(self, group: h5py.Group, name: str, data: np.ndarray) -> None:
        """Create HDF5 dataset with compression for image data.
        
        Args:
            group: HDF5 group to create dataset in.
            name: Dataset name.
            data: Data array to save.
        """
        # Use compression only for image data (uint8, 3D+)
        if data.dtype == np.uint8 and len(data.shape) >= 3:
            group.create_dataset(
                name,
                data=data,
                compression=HDF5_COMPRESSION_METHOD,
                compression_opts=HDF5_COMPRESSION_LEVEL
            )
        else:
            group.create_dataset(name, data=data)

    def clear_buffers(self) -> None:
        """Clear step data and initial state buffers."""
        self.step_data_buffer = {}
        self.initial_state_buffer = {}

    def reset_current_demo(self) -> None:
        """Reset current demonstration without saving.
        
        Stops recording if active and clears all buffers.
        """
        if self.is_recording:
            self.toggle_recording()
        self.clear_buffers()
        Logger.info(f"\n[DEMO {self.current_demo_index}] --> DISCARDED.")
