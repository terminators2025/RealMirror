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
import numpy as np
from utility.logger import Logger
from omni.isaac.core.utils.xforms import get_world_pose
from omni.isaac.core.prims import RigidPrim
from typing import List, Dict, Optional, Tuple
import cv2
import os
import pandas as pd
from pathlib import Path


def object_in_success_zone(prim_path: str, min_bounds: list, max_bounds: list) -> bool:
    """
    Checks if the specified prim is within the given 3D bounding box.
    """
    try:
        position, _ = get_world_pose(prim_path)
        in_x = min_bounds[0] <= position[0] <= max_bounds[0]
        in_y = min_bounds[1] <= position[1] <= max_bounds[1]
        in_z = min_bounds[2] <= position[2] <= max_bounds[2]
        return in_x and in_y and in_z
    except Exception as e:
        Logger.error(f"Termination condition check 'object_in_success_zone' failed: {e}")
        return False


def object_height_below_minimum(prim_path: str, minimum_height: float) -> bool:
    """
    Checks if the height of the specified prim is below a given minimum value.
    """
    try:
        position, _ = get_world_pose(prim_path)
        return position[2] < minimum_height
    except Exception as e:
        Logger.error(f"Termination condition check 'object_height_below_minimum' failed: {e}")
        return False


def log_object_reached_target_details(
    name:str, moving_prim_path: str, target_prim_path: str, xy_threshold: float, z_threshold: float
) -> None:
    """
    A dedicated logging function that calculates and prints position details of a moving object relative to a target.
    It does not return any value.
    """
    try:
        moving_pos, _ = get_world_pose(moving_prim_path)
        target_pos, _ = get_world_pose(target_prim_path)

        dist_xy = np.linalg.norm(moving_pos[:2] - target_pos[:2])
        dist_z = moving_pos[2] - target_pos[2]

        xy_ok = dist_xy < xy_threshold
        if z_threshold < 0:
            z_ok = z_threshold < dist_z < 0
        else:
            z_ok = 0 < dist_z < z_threshold

        Logger.info(f"{name}  [State Details] XY Distance: {dist_xy:.4f}m (Threshold < {xy_threshold}m, Status: {'✓' if xy_ok else '✗'})")
        Logger.info(f"{name}  [State Details] Z Height: {dist_z:.4f}m (Range 0~{z_threshold}m, Status: {'✓' if z_ok else '✗'})")

    except Exception as e:
        Logger.error(f"Failed to log success state details: {e}")


def object_reached_target(
    moving_prim_path: str, target_prim_path: str, xy_threshold: float, z_threshold: float
) -> bool:
    """
    Checks if an object has reached a position above another object.
    This version only performs the calculation and returns a boolean value, it does not print any information.
    """
    try:
        moving_pos, _ = get_world_pose(moving_prim_path)
        target_pos, _ = get_world_pose(target_prim_path)

        # Calculate distance
        dist_xy = np.linalg.norm(moving_pos[:2] - target_pos[:2])
        dist_z = moving_pos[2] - target_pos[2]

        # Check conditions
        xy_ok = dist_xy < xy_threshold
        if z_threshold < 0:
            z_ok = z_threshold < dist_z < 0
        else:
            z_ok = 0 < dist_z < z_threshold

        # Return result only, no printing
        return bool(xy_ok and z_ok)

    except Exception as e:
        Logger.error(f"Termination condition check 'object_reached_target' failed: {e}")
        return False

def object_is_stably_stacked(
    moving_prim: RigidPrim, 
    target_prim: RigidPrim, 
    xy_threshold: float, 
    z_threshold: float,
    velocity_threshold: float = 0.2,
    angular_velocity_threshold: float = 0.15
) -> bool:
    """
    Checks if a moving prim is stably stacked on a target prim.

    This function checks three conditions:
    1. The XY distance between the prims is within the threshold.
    2. The Z distance is within a valid range (above the target but not too high).
    3. The linear and angular velocities of the moving prim are below a threshold, indicating it is stationary.
    """
    if not moving_prim or not target_prim or not moving_prim.is_valid() or not target_prim.is_valid():
        Logger.warning("Invalid prims provided for stability check.")
        return False

    try:
        moving_pos, _ = moving_prim.get_world_pose()
        target_pos, _ = target_prim.get_world_pose()

        # 1. Position check
        dist_xy = np.linalg.norm(moving_pos[:2] - target_pos[:2])
        dist_z = moving_pos[2] - target_pos[2]

        position_ok = (dist_xy < xy_threshold) and ((0 < dist_z < z_threshold) if z_threshold >= 0 else (z_threshold < dist_z < 0))

        if not position_ok:
            return False

        # 2. Velocity check (only if position is correct)
        linear_vel = moving_prim.get_linear_velocity()
        angular_vel = moving_prim.get_angular_velocity()

        # Check if velocity data is valid
        if linear_vel is None or angular_vel is None:
            Logger.warning("Could not retrieve velocity for stability check.")
            return False

        linear_speed = np.linalg.norm(linear_vel)
        angular_speed = np.linalg.norm(angular_vel)

        velocity_ok = (linear_speed < velocity_threshold) and (angular_speed < angular_velocity_threshold)

        return bool(velocity_ok)

    except Exception as e:
        Logger.error(f"Stability check 'object_is_stably_stacked' failed: {e}")
        return False


def log_grasp_state_details(
    robot_articulation_view,
    end_effector_prim_path: str,
    target_object_prim_path: str,
    gripper_joint_indices: list[int],
    distance_threshold: float,
    gripper_closed_threshold: float,
) -> bool:
    """
    Checks and prints detailed status of the robot's grasp on a target object.
    """
    if not robot_articulation_view or not robot_articulation_view.is_valid():
        Logger.warning("Robot articulation view for grasp check is invalid.")
        return False

    Logger.debug("  [Grasp Check Details]")
    try:
        # 1. Check distance
        hand_pos, _ = get_world_pose(end_effector_prim_path)
        object_pos, _ = get_world_pose(target_object_prim_path)
        
        distance = np.linalg.norm(hand_pos - object_pos)
        is_close = distance < distance_threshold
        Logger.debug(f"    - Distance: {distance:.4f}m (Threshold: < {distance_threshold}m, Status: {'✓' if is_close else '✗'})")

        if not is_close:
            Logger.error("    - Result: FAILED (Reason: Too far)")
            return False

        # 2. Check gripper status
        joint_positions = robot_articulation_view.get_joint_positions()
        if joint_positions is None or joint_positions.size == 0:
            Logger.warning("Could not get joint positions for grasp check.")
            return False

        gripper_joints_sum = np.sum(joint_positions[0, gripper_joint_indices])
        is_gripping = gripper_joints_sum > gripper_closed_threshold
        Logger.debug(f"    - Gripper Angle Sum: {gripper_joints_sum:.4f} rad (Threshold: > {gripper_closed_threshold} rad, Status: {'✓' if is_gripping else '✗'})")

        if is_gripping:
            Logger.info("    - Result: SUCCESS")
            return True
        else:
            Logger.error("    - Result: FAILED (Reason: Gripper not closed)")
            return False

    except Exception as e:
        Logger.error(f"Failed to log grasp state details: {e}")
        return False


def check_grasp_state(
    robot_articulation_view,
    end_effector_prim_path: str,
    target_object_prim_path: str,
    gripper_joint_indices: list[int],
    distance_threshold: float,
    gripper_closed_threshold: float,
) -> bool:
    """
    Checks if the robot has successfully grasped the target object.

    This function checks two core conditions:
    1. The end-effector (hand) is close enough to the target object.
    2. The finger joints are in a closed state (the sum of their joint angles is greater than a threshold).

    Args:
        robot_articulation_view: The robot's ArticulationView, used to get joint states.
        end_effector_prim_path (str): Path to the robot's end-effector prim (e.g., "/World/A2/aise_a2_t2d0_flagship/L_middle_1").
        target_object_prim_path (str): Path to the target object's prim.
        gripper_joint_indices (list[int]): List of joint indices for the fingers used to determine grasp status.
        distance_threshold (float): The maximum distance (in meters) between the hand and the object to be considered "close enough". A value of 0.1 can be used as a starting point.
        gripper_closed_threshold (float): The threshold for the sum of finger joint angles (in radians). If the sum exceeds this value, the gripper is considered closed. A value of 1.5 can be used as a starting point.

    Returns:
        bool: True if the grasp conditions are met, otherwise False.
    """
    if not robot_articulation_view or not robot_articulation_view.is_valid():
        Logger.warning("Robot articulation view for grasp check is invalid.")
        return False

    try:
        # 1. Check distance
        hand_pos, _ = get_world_pose(end_effector_prim_path)
        object_pos, _ = get_world_pose(target_object_prim_path)
        
        distance = np.linalg.norm(hand_pos - object_pos)
        is_close = distance < distance_threshold

        if not is_close:
            return False  # If the distance is too great, return False immediately

        # 2. Check gripper status (only if close enough)
        joint_positions = robot_articulation_view.get_joint_positions()
        if joint_positions is None or joint_positions.size == 0:
            Logger.warning("Could not get joint positions for grasp check.")
            return False

        # Calculate the sum of the specified finger joint angles
        gripper_joints_sum = np.sum(joint_positions[0, gripper_joint_indices])
        is_gripping = gripper_joints_sum > gripper_closed_threshold
        
        # Grasp is considered successful only when both conditions are met
        return is_gripping

    except Exception as e:
        Logger.error(f"Grasp state check 'check_grasp_state' failed: {e}")
        return False


# ==================================
# Evaluation Helper Functions
# ==================================

def save_frames_to_video(frames: List[np.ndarray], output_path: str, fps: int = 30) -> bool:
    """
    Saves a list of frames as a video file.
    
    Args:
        frames: List of image frames.
        output_path: Output video path.
        fps: Frames per second.
        
    Returns:
        bool: True if saved successfully, otherwise False.
    """
    if not frames:
        Logger.warning("Frame buffer is empty. Cannot save video.")
        return False
    
    try:
        height, width, _ = frames[0].shape
    except (IndexError, ValueError):
        Logger.warning("Frame buffer is empty or has wrong shape. Cannot save video.")
        return False
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    video_writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    
    if not video_writer.isOpened():
        Logger.error(f"Failed to open video writer for path: {output_path}")
        return False
    
    Logger.info(f"Saving {len(frames)} frames to video: {output_path}")
    
    # Write all frames
    for frame in frames:
        video_writer.write(frame)
    
    video_writer.release()
    Logger.info(f"Video saved successfully to: {output_path}")
    return True


def move_prim_safely(prim: RigidPrim, position: np.ndarray, orientation: np.ndarray) -> None:
    """
    Safely moves a RigidPrim to a specified pose.
    
    Args:
        prim: The RigidPrim object to move.
        position: Target position (x, y, z).
        orientation: Target orientation quaternion (w, x, y, z).
    """
    if not prim or not prim.is_valid():
        return
    
    # Temporarily disable physics
    prim.disable_rigid_body_physics()
    
    # Set the new pose
    prim.set_world_pose(position=position.tolist(), orientation=orientation.tolist())
    
    # Re-enable physics
    prim.enable_rigid_body_physics()
    
    # Reset velocities
    prim.set_linear_velocity(np.zeros(3))
    prim.set_angular_velocity(np.zeros(3))


def create_evaluation_csv(csv_path: Path, model_type: str) -> None:
    """
    Creates the header for the evaluation results CSV file.
    
    Args:
        csv_path: Path to the CSV file.
        model_type: The type of the model.
    """
    with open(csv_path, 'w') as f:
        f.write(f"# MODEL_TYPE: {model_type}\n")
        # Write CSV header
        header = "rollout_idx,x,y,success,rollout_path,task_name,cylinder_prim_path,target_prim_path\n"
        f.write(header)


def load_evaluation_progress(csv_path: Path) -> Tuple[pd.DataFrame, int]:
    """
    Loads evaluation progress from a CSV file.
    
    Args:
        csv_path: Path to the CSV file.
        
    Returns:
        (DataFrame, index of the next rollout to execute).
    """
    if not csv_path.exists():
        return pd.DataFrame(), 0
    
    try:
        df = pd.read_csv(csv_path, comment='#')
        
        # Check for required columns
        required_cols = {'rollout_idx', 'x', 'y', 'success'}
        if not required_cols.issubset(df.columns):
            Logger.warning("CSV file missing required columns. Starting fresh.")
            return pd.DataFrame(), 0
        
        # Find unfinished rollouts
        unfinished = df[df['success'] == -1]
        if not unfinished.empty:
            start_idx = unfinished['rollout_idx'].min()
        else:
            # All are finished, start from the next one
            start_idx = df['rollout_idx'].max() + 1 if not df.empty else 0
        
        return df, start_idx
        
    except Exception as e:
        Logger.error(f"Failed to load evaluation progress: {e}")
        return pd.DataFrame(), 0


def save_evaluation_progress(all_results: List[Dict], csv_path: Path, model_type: str) -> None:
    """
    Saves evaluation progress to a CSV file.
    
    Args:
        all_results: List of all evaluation results.
        csv_path: Path to the CSV file.
        model_type: The type of the model.
    """
    try:
        df = pd.DataFrame(all_results)
        with open(csv_path, 'w') as f:
            f.write(f"# MODEL_TYPE: {model_type}\n")
            df.to_csv(f, index=False, header=True)
        Logger.debug(f"Progress saved to: {csv_path}")
    except Exception as e:
        Logger.error(f"Failed to save progress to CSV: {e}")