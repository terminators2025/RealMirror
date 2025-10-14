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
from typing import Dict, Optional
from utility.logger import Logger


class UnitConverter:
    """
    Converts between radian and gear position representations for robot joints.
    Specifically handles 26-DOF robot configuration with dual arms and hands.
    """
    
    # Joint names for 26-DOF model
    MODEL_JOINT_NAMES_26DOF = [
        "idx13_left_arm_joint1", "idx14_left_arm_joint2", "idx15_left_arm_joint3",
        "idx16_left_arm_joint4", "idx17_left_arm_joint5", "idx18_left_arm_joint6",
        "idx19_left_arm_joint7", "idx20_right_arm_joint1", "idx21_right_arm_joint2",
        "idx22_right_arm_joint3", "idx23_right_arm_joint4", "idx24_right_arm_joint5",
        "idx25_right_arm_joint6", "idx26_right_arm_joint7", "left_thumb_0", "left_thumb_1",
        "left_index", "left_middle", "left_ring", "left_pinky", "right_thumb_0",
        "right_thumb_1", "right_index", "right_middle", "right_ring", "right_pinky"
    ]
    
    # Simulation joint names for state extraction
    STATE_SIM_JOINT_NAMES_26DOF = [
        "idx13_left_arm_joint1", "idx14_left_arm_joint2", "idx15_left_arm_joint3",
        "idx16_left_arm_joint4", "idx17_left_arm_joint5", "idx18_left_arm_joint6",
        "idx19_left_arm_joint7", "idx20_right_arm_joint1", "idx21_right_arm_joint2",
        "idx22_right_arm_joint3", "idx23_right_arm_joint4", "idx24_right_arm_joint5",
        "idx25_right_arm_joint6", "idx26_right_arm_joint7", "L_thumb_swing_joint",
        "L_thumb_1_joint", "L_index_1_joint", "L_middle_1_joint", "L_ring_1_joint",
        "L_little_1_joint", "R_thumb_swing_joint", "R_thumb_1_joint", "R_index_1_joint",
        "R_middle_1_joint", "R_ring_1_joint", "R_little_1_joint"
    ]
    
    def __init__(self, arc_to_gear: bool = False, max_gear: float = 2000.0):
        """
        Initialize the unit converter.
        
        Args:
            arc_to_gear: Whether to convert between radians and gear positions
            max_gear: Maximum gear position value
        """
        self.arc_to_gear = arc_to_gear
        self.max_gear = max_gear
        
        # Maximum angles for hand joints in radians
        self.HAND_JOINT_MAX_ANGLES_RAD = {
            "left_thumb_0":  np.deg2rad(90.0), 
            "left_thumb_1":  np.deg2rad(12.0),
            "left_index":    np.deg2rad(50.0), 
            "left_middle":   np.deg2rad(50.0),
            "left_ring":     np.deg2rad(50.0), 
            "left_pinky":    np.deg2rad(50.0),
            "right_thumb_0": np.deg2rad(90.0), 
            "right_thumb_1": np.deg2rad(12.0),
            "right_index":   np.deg2rad(50.0), 
            "right_middle":  np.deg2rad(50.0),
            "right_ring":    np.deg2rad(50.0), 
            "right_pinky":   np.deg2rad(50.0),
        }
        
        Logger.info(f"UnitConverter initialized with arc_to_gear={arc_to_gear}, max_gear={max_gear}")
    
    def radian_to_gear_vector(self, radian_vector: np.ndarray) -> np.ndarray:
        """
        Convert a 26-DOF radian vector to gear positions for hand joints.
        
        Args:
            radian_vector: 26-dimensional numpy array in radians
            
        Returns:
            26-dimensional numpy array with gear positions for hand joints (indices 14-25)
        """
        if not self.arc_to_gear or radian_vector.shape[0] != 26:
            return radian_vector
            
        output_vector = np.copy(radian_vector)
        
        # Convert hand joints (indices 14-25) from radians to gear positions
        for i in range(14, 26):
            joint_name = self.MODEL_JOINT_NAMES_26DOF[i]
            
            # Special handling for thumb swing joints
            if joint_name in ["left_thumb_0", "right_thumb_0"]:
                output_vector[i] = 1195.0
                continue
                
            max_radian = self.HAND_JOINT_MAX_ANGLES_RAD.get(joint_name)
            if max_radian is not None and max_radian > 0:
                clipped_rad = np.clip(radian_vector[i], 0, max_radian)
                gear_value = (clipped_rad / max_radian) * self.max_gear
                output_vector[i] = round(gear_value)
                
        return output_vector
    
    def gear_to_radian_vector(self, hybrid_vector: np.ndarray) -> np.ndarray:
        """
        Convert a hybrid vector (with gear positions for hand joints) back to radians.
        
        Args:
            hybrid_vector: Array with shape (..., 26) containing gear positions for hand joints
            
        Returns:
            Array with same shape but all values in radians
        """
        if not self.arc_to_gear or hybrid_vector.shape[-1] != 26:
            return hybrid_vector
            
        output_vector = np.copy(hybrid_vector)
        
        # Handle both 1D and 2D arrays
        if hybrid_vector.ndim == 1:
            hybrid_vector = hybrid_vector.reshape(1, -1)
            output_vector = output_vector.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Convert gear positions back to radians for hand joints
        for step_id, step_vector in enumerate(hybrid_vector):
            for i in range(14, 26):
                joint_name = self.MODEL_JOINT_NAMES_26DOF[i]
                
                # Special handling for thumb swing joints
                if joint_name in ["left_thumb_0", "right_thumb_0"]:
                    output_vector[step_id][i] = self.HAND_JOINT_MAX_ANGLES_RAD[joint_name]
                    continue
                    
                max_radian = self.HAND_JOINT_MAX_ANGLES_RAD.get(joint_name)
                if max_radian is not None and max_radian > 0:
                    clipped_gear = round(np.clip(step_vector[i], 0, self.max_gear))
                    output_vector[step_id][i] = (clipped_gear / self.max_gear) * max_radian
        
        if squeeze_output:
            output_vector = output_vector.squeeze(0)
            
        return output_vector
    
    def expand_26d_to_38d_sim_action(self, action_26d_radians: np.ndarray) -> np.ndarray:
        """
        Expand 26-DOF action to 38-DOF simulation action format.
        This handles the mimic joints in the simulation.
        
        Args:
            action_26d_radians: Array with shape (N, 26) in radians
            
        Returns:
            Array with shape (N, 38) for simulation control
        """
        if action_26d_radians.shape[-1] != 26:
            Logger.error(f"Input action dimension is {action_26d_radians.shape[-1]}, expected 26")
            return np.zeros((action_26d_radians.shape[0], 38), dtype=np.float32)
        
        action_38d = np.zeros((action_26d_radians.shape[0], 38), dtype=np.float32)
        
        # Copy arm joints (0-13)
        action_38d[:, 0:14] = action_26d_radians[:, 0:14]
        
        # Expand left hand joints with mimic relationships
        action_38d[:, 14] = action_26d_radians[:, 14]  # L_thumb_swing
        action_38d[:, 15:18] = action_26d_radians[:, 15].reshape(-1, 1)  # L_thumb mimic
        action_38d[:, 18:20] = action_26d_radians[:, 16].reshape(-1, 1)  # L_index mimic
        action_38d[:, 20:22] = action_26d_radians[:, 17].reshape(-1, 1)  # L_middle mimic
        action_38d[:, 22:24] = action_26d_radians[:, 18].reshape(-1, 1)  # L_ring mimic
        action_38d[:, 24:26] = action_26d_radians[:, 19].reshape(-1, 1)  # L_pinky mimic
        
        # Expand right hand joints with mimic relationships
        action_38d[:, 26] = action_26d_radians[:, 20]  # R_thumb_swing
        action_38d[:, 27:30] = action_26d_radians[:, 21].reshape(-1, 1)  # R_thumb mimic
        action_38d[:, 30:32] = action_26d_radians[:, 22].reshape(-1, 1)  # R_index mimic
        action_38d[:, 32:34] = action_26d_radians[:, 23].reshape(-1, 1)  # R_middle mimic
        action_38d[:, 34:36] = action_26d_radians[:, 24].reshape(-1, 1)  # R_ring mimic
        action_38d[:, 36:38] = action_26d_radians[:, 25].reshape(-1, 1)  # R_pinky mimic
        
        return action_38d
