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
import torch
from typing import Optional, Dict, List, Tuple, Any
from PIL import Image
from utility.logger import Logger
from inference.unit_converter import UnitConverter
from inference.model_loader import ModelLoader


class InferenceEngine:
    """
    Main inference engine for robot control.
    Handles observation processing, model inference, and action generation.
    """
    
    def __init__(
        self,
        model_loader: ModelLoader,
        unit_converter: UnitConverter,
        num_actions_in_chunk: int = 5,
        image_resolution: int = 256
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_loader: ModelLoader instance with loaded model
            unit_converter: UnitConverter instance for unit conversions
            num_actions_in_chunk: Number of actions to use from model output
            image_resolution: Resolution for input images
        """
        self.model_loader = model_loader
        self.unit_converter = unit_converter
        self.num_actions_in_chunk = num_actions_in_chunk
        self.image_resolution = image_resolution
        self.device = model_loader.device
        
        # Task name for models that require it (pi0, smolvla)
        self.task_name = None
        
        Logger.info(f"InferenceEngine initialized with {num_actions_in_chunk} actions per chunk")
    
    def set_task_name(self, task_name: str):
        """
        Set the task name for models that require it.
        
        Args:
            task_name: Description of the task
        """
        self.task_name = task_name
        Logger.info(f"Task name set to: {task_name}")
    
    def process_image(self, img_bgr: np.ndarray) -> torch.Tensor:
        """
        Process a BGR image for model input.
        
        Args:
            img_bgr: BGR image as numpy array
            
        Returns:
            Processed image tensor
        """
        # Convert BGR to RGB and normalize to [0, 1]
        img_tensor = torch.from_numpy(img_bgr).to(torch.float32) / 255.0
        # Change from HWC to CHW format
        return img_tensor.permute(2, 0, 1)
    
    def get_observation_state(
        self,
        robot_view,
        state_joint_indices: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Get the current observation state vector from the robot.
        
        Args:
            robot_view: Robot articulation view
            state_joint_indices: Indices of joints to extract for state
            
        Returns:
            26-DOF state vector in hybrid format (gear positions for hands) or None
        """
        if not robot_view or not robot_view.is_valid() or state_joint_indices is None:
            Logger.warning("Robot view or state joint indices not ready")
            return None
        
        try:
            # Get full joint positions
            full_joint_pos = robot_view.get_joint_positions()
            if full_joint_pos is None or full_joint_pos.size == 0:
                Logger.warning("Failed to get valid joint positions from robot view")
                return None
            
            # Extract relevant joints for 26-DOF state
            observation_state_radians = full_joint_pos[0, state_joint_indices].astype(np.float32)
            
            if observation_state_radians.shape[0] != 26:
                Logger.error(f"State vector dimension is {observation_state_radians.shape[0]}, expected 26")
                return None
            
            # Convert to hybrid format (gear positions for hand joints)
            observation_state_hybrid = self.unit_converter.radian_to_gear_vector(observation_state_radians)
            
            return observation_state_hybrid
            
        except Exception as e:
            Logger.error(f"Error getting observation state: {e}")
            return None
    
    def run_inference(
        self,
        state_vector: np.ndarray,
        left_wrist_image: np.ndarray,
        right_wrist_image: np.ndarray,
        head_image: np.ndarray,
        save_images: bool = False
    ) -> Optional[np.ndarray]:
        """
        Run inference with the loaded model.
        
        Args:
            state_vector: 26-DOF state vector in hybrid format
            left_wrist_image: Left wrist camera image (BGR)
            right_wrist_image: Right wrist camera image (BGR)
            head_image: Head camera image (BGR)
            save_images: Whether to save input images for debugging
            
        Returns:
            Expanded 38-DOF action array in radians or None if inference fails
        """
        policy = self.model_loader.get_policy()
        if policy is None:
            Logger.error("No model loaded for inference")
            return None
        
        try:
            # Convert state to tensor
            state = torch.from_numpy(state_vector).to(torch.float32)
            
            # Process images
            if save_images:
                Image.fromarray(left_wrist_image).save(f"./left_wrist_img_{self.model_loader.model_type}.png")
                Image.fromarray(right_wrist_image).save(f"./right_wrist_img_{self.model_loader.model_type}.png")
                Image.fromarray(head_image).save(f"./head_img_{self.model_loader.model_type}.png")
            
            left_wrist_tensor = self.process_image(left_wrist_image)
            right_wrist_tensor = self.process_image(right_wrist_image)
            head_tensor = self.process_image(head_image)
            
            # Move tensors to device and add batch dimension
            state = state.to(self.device, non_blocking=True).unsqueeze(0)
            head_tensor = head_tensor.to(self.device, non_blocking=True).unsqueeze(0)
            left_wrist_tensor = left_wrist_tensor.to(self.device, non_blocking=True).unsqueeze(0)
            right_wrist_tensor = right_wrist_tensor.to(self.device, non_blocking=True).unsqueeze(0)
            
            # Prepare observation dictionary
            observation = {
                "observation.state": state,
                "observation.images.left_wrist_camera": left_wrist_tensor,
                "observation.images.right_wrist_camera": right_wrist_tensor,
                "observation.images.head_camera": head_tensor,
            }
            
            # Add task name for models that require it
            if self.model_loader.model_type in ["pi0", "smolvla"]:
                if self.task_name:
                    observation["task"] = [self.task_name]
                else:
                    Logger.warning(f"Task name not set for {self.model_loader.model_type} model")
            
            # Run inference
            with torch.inference_mode():
                if hasattr(policy, 'select_action_custom'):
                    action_tensor = policy.select_action_custom(observation)
                else:
                    action_tensor = policy.select_action(observation)
            
            # Process output based on dimensions
            if action_tensor.dim() == 3:
                # Take first N actions from chunk
                next_action_tensor = action_tensor[0, :self.num_actions_in_chunk]
            elif action_tensor.dim() == 2:
                next_action_tensor = action_tensor
            else:
                Logger.error(f"Unexpected model output dimension: {action_tensor.shape}")
                return None
            
            # Convert to numpy
            numpy_action_26d_hybrid = next_action_tensor.cpu().numpy()
            
            if numpy_action_26d_hybrid.shape[-1] != 26:
                Logger.error(f"Model output dimension is {numpy_action_26d_hybrid.shape[-1]}, expected 26")
                return None
            
            # Convert gear positions back to radians
            numpy_action_26d_radians = self.unit_converter.gear_to_radian_vector(numpy_action_26d_hybrid)
            
            # Expand to 38-DOF for simulation
            expanded_action_38d_radians = self.unit_converter.expand_26d_to_38d_sim_action(numpy_action_26d_radians)
            
            return expanded_action_38d_radians
            
        except Exception as e:
            Logger.error(f"Error during model inference: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def reset(self):
        """
        Reset the inference engine and model.
        """
        self.model_loader.reset()
        Logger.info("Inference engine reset")


class InferenceConfig:
    """
    Configuration class for inference settings.
    """
    
    def __init__(
        self,
        model_type: str = "act",
        model_path: str = "",
        use_gear_conversion: bool = False,
        max_gear: float = 2000.0,
        num_actions_in_chunk: int = 5,
        image_resolution: int = 256,
        task_name: Optional[str] = None
    ):
        """
        Initialize inference configuration.
        
        Args:
            model_type: Type of model (act, pi0, diffusion, smolvla)
            model_path: Path to pretrained model
            use_gear_conversion: Whether to use gear position conversion
            max_gear: Maximum gear position value
            num_actions_in_chunk: Number of actions to use from model output
            image_resolution: Resolution for input images
            task_name: Task description for models that require it
        """
        self.model_type = model_type
        self.model_path = model_path
        self.use_gear_conversion = use_gear_conversion
        self.max_gear = max_gear
        self.num_actions_in_chunk = num_actions_in_chunk
        self.image_resolution = image_resolution
        self.task_name = task_name
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "InferenceConfig":
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            InferenceConfig instance
        """
        return cls(
            model_type=config_dict.get("model_type", "act"),
            model_path=config_dict.get("model_path", ""),
            use_gear_conversion=config_dict.get("use_gear_conversion", False),
            max_gear=config_dict.get("max_gear", 2000.0),
            num_actions_in_chunk=config_dict.get("num_actions_in_chunk", 5),
            image_resolution=config_dict.get("image_resolution", 256),
            task_name=config_dict.get("task_name", None)
        )
