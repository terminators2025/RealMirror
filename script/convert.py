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

"""
HDF5 to LeRobot Dataset Converter

This script converts teleoperation data recorded in HDF5 format to LeRobot dataset format.
It supports multi-file batch processing, parallel episode processing, and optional 
arc-to-gear conversion for hand joints.

Usage:
    python script/convert.py <output_dir> --data_pairs <data_pairs.json> [--arc2gear]

Example:
    python script/convert.py /path/to/output --data_pairs Task1_data_pair.json --arc2gear
"""

import argparse
import gc
import json
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage


# Configure logging
def setup_logger(name: str = "HDF5ToLeRobot", level: int = logging.INFO) -> logging.Logger:
    """
    Setup and configure logger with colored output.
    
    Args:
        name: Logger name.
        level: Logging level.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter with colors (using ANSI escape codes)
    class ColoredFormatter(logging.Formatter):
        """Custom formatter with colors for different log levels."""
        
        # ANSI color codes
        COLORS = {
            'DEBUG': '\033[36m',      # Cyan
            'INFO': '\033[32m',       # Green
            'WARNING': '\033[33m',    # Yellow
            'ERROR': '\033[31m',      # Red
            'CRITICAL': '\033[35m',   # Magenta
        }
        RESET = '\033[0m'
        
        def format(self, record):
            # Add color to levelname
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
            return super().format(record)
    
    # Set format
    formatter = ColoredFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


# Initialize global logger
logger = setup_logger()


def load_robot_config(config_path: str) -> Dict:
    """
    Load robot configuration from JSON file.
    
    Args:
        config_path: Path to robot configuration JSON file.
        
    Returns:
        Dictionary containing robot configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Process gripper config angles from degrees to radians
    if "gripper_config" in config:
        for arm in config["gripper_config"]:
            gripper = config["gripper_config"][arm]
            if "stage1_closed_angles_deg" in gripper:
                gripper["stage1_closed_angles"] = [
                    np.deg2rad(angle) for angle in gripper["stage1_closed_angles_deg"]
                ]
            if "stage2_closed_angles_deg" in gripper:
                gripper["stage2_closed_angles"] = [
                    np.deg2rad(angle) for angle in gripper["stage2_closed_angles_deg"]
                ]
    
    return config


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Numpy data types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)


class HDF5ToLeRobotConverter:
    """
    Converter for transforming Isaac Sim HDF5 teleoperation data to LeRobot format.

    This converter processes multiple HDF5 files, extracts robot states, actions, and
    camera observations, and saves them in LeRobot's parquet + metadata format.
    """

    # Default camera names matching the data recording format
    CAMERA_NAMES = ["left_wrist_camera_bgr", "right_wrist_camera_bgr", "head_camera_bgr"]

    # Image processing parameters
    IMAGE_RESIZE_TARGET = (256, 256)
    IMAGE_RESIZE_METHOD = PILImage.Resampling.LANCZOS

    def __init__(
        self,
        robot_config: Dict,
        task_names: Optional[List[str]] = None,
        arc_to_gear: bool = False,
        fps: int = 15,
    ):
        """
        Initialize the converter.

        Args:
            robot_config: Robot configuration dictionary containing joint information.
            task_names: List of task names corresponding to HDF5 files.
            arc_to_gear: If True, convert hand joint values from radians to gear positions.
            fps: Frames per second for the dataset.
        """
        self.robot_config = robot_config
        self.arc_to_gear = arc_to_gear
        self.fps = fps

        # Build ordered joint names list
        self._build_joint_names_list()

        # Build hand joint conversion parameters
        self._build_hand_joint_params()

        # Task management: maintain order of first occurrence
        self.all_task_names_from_input = task_names if task_names is not None else []
        unique_tasks_in_order = list(dict.fromkeys(self.all_task_names_from_input))
        self.task_to_idx = {name: i for i, name in enumerate(unique_tasks_in_order)}
        self.tasks = unique_tasks_in_order

        # Initialize dataset features
        self.features = self._initialize_features()
        self.image_columns = self.CAMERA_NAMES
        self.episode_metadata_list = []

        if self.arc_to_gear:
            logger.info("🚀 Arc-to-Gear conversion mode ENABLED. Hand joint gear values will be integers (0-2000).")
        else:
            logger.info("Arc-to-Gear conversion mode DISABLED. Joint values will remain in radians.")

    def _build_joint_names_list(self):
        """Build the ordered list of joint names from robot config."""
        self.joint_names = []

        # Add arm joints (left then right)
        arm_joints = self.robot_config.get("arm_joints", {})
        self.joint_names.extend(arm_joints.get("left", []))
        self.joint_names.extend(arm_joints.get("right", []))

        # Add hand joints (left then right)
        # Map from Isaac sim joint names to recording format
        hand_joints = self.robot_config.get("hand_joints", {})
        left_hand_joints = hand_joints.get("left", [])
        right_hand_joints = hand_joints.get("right", [])

        # Create mapping from config joint names to recording format
        self.hand_joint_mapping = {}
        for joint in left_hand_joints:
            if "thumb_swing" in joint:
                self.hand_joint_mapping[joint] = "left_thumb_0"
            elif "thumb_1" in joint:
                self.hand_joint_mapping[joint] = "left_thumb_1"
            elif "index_1" in joint:
                self.hand_joint_mapping[joint] = "left_index"
            elif "middle_1" in joint:
                self.hand_joint_mapping[joint] = "left_middle"
            elif "ring_1" in joint:
                self.hand_joint_mapping[joint] = "left_ring"
            elif "little_1" in joint:
                self.hand_joint_mapping[joint] = "left_pinky"

        for joint in right_hand_joints:
            if "thumb_swing" in joint:
                self.hand_joint_mapping[joint] = "right_thumb_0"
            elif "thumb_1" in joint:
                self.hand_joint_mapping[joint] = "right_thumb_1"
            elif "index_1" in joint:
                self.hand_joint_mapping[joint] = "right_index"
            elif "middle_1" in joint:
                self.hand_joint_mapping[joint] = "right_middle"
            elif "ring_1" in joint:
                self.hand_joint_mapping[joint] = "right_ring"
            elif "little_1" in joint:
                self.hand_joint_mapping[joint] = "right_pinky"

        # Add hand joint names in recording format
        self.joint_names.extend(
            [
                "left_thumb_0",
                "left_thumb_1",
                "left_index",
                "left_middle",
                "left_ring",
                "left_pinky",
                "right_thumb_0",
                "right_thumb_1",
                "right_index",
                "right_middle",
                "right_ring",
                "right_pinky",
            ]
        )

    def _build_hand_joint_params(self):
        """Build hand joint parameters for arc-to-gear conversion."""
        self.hand_joint_max_angles = {}

        # Extract max angles from gripper config
        gripper_config = self.robot_config.get("gripper_config", {})
        for arm in ["left", "right"]:
            gripper_cfg = gripper_config.get(arm, {})

            # Stage 1: thumb swing joint
            stage1_joints = gripper_cfg.get("stage1_joints", [])
            stage1_angles = gripper_cfg.get("stage1_closed_angles", [])
            for joint, angle in zip(stage1_joints, stage1_angles):
                recording_name = self.hand_joint_mapping.get(joint, None)
                if recording_name:
                    self.hand_joint_max_angles[recording_name] = angle

            # Stage 2: other finger joints
            stage2_joints = gripper_cfg.get("stage2_joints", [])
            stage2_angles = gripper_cfg.get("stage2_closed_angles", [])
            for joint, angle in zip(stage2_joints, stage2_angles):
                recording_name = self.hand_joint_mapping.get(joint, None)
                if recording_name:
                    self.hand_joint_max_angles[recording_name] = angle

    def _radian_to_gear(self, radian_value: float, joint_name: str, max_gear: float = 2000.0) -> float:
        """
        Convert radian value to gear position (0-2000).

        Args:
            radian_value: Joint value in radians.
            joint_name: Name of the joint.
            max_gear: Maximum gear value (default 2000).

        Returns:
            Gear value (integer if arc_to_gear is enabled, otherwise original value).
        """
        # Special cases for thumb swing joints
        if joint_name in ["left_thumb_0", "right_thumb_0"]:
            return 1195

        max_radian = self.hand_joint_max_angles.get(joint_name)
        if max_radian is None or max_radian == 0:
            return radian_value

        radian_value = np.clip(radian_value, 0, max_radian)
        gear_value = (radian_value / max_radian) * max_gear
        return int(round(gear_value))

    def _initialize_features(self) -> Features:
        """Initialize the HuggingFace Dataset features schema."""
        return Features(
            {
                "episode_index": Value("int64"),
                "frame_index": Value("int64"),
                "task_index": Value("int64"),
                "index": Value("int64"),
                "timestamp": Value("float64"),
                "observation.state": Sequence(Value("float32")),
                "left_wrist_camera_bgr": Image(),
                "right_wrist_camera_bgr": Image(),
                "head_camera_bgr": Image(),
                "action": Sequence(Value("float32")),
            }
        )

    def convert_and_save(
        self, h5_paths: List[str], output_dir: str, manager_lock, manager_list
    ):
        """
        Main conversion function: processes all HDF5 files and saves LeRobot dataset.

        Args:
            h5_paths: List of paths to HDF5 files.
            output_dir: Output directory for LeRobot dataset.
            manager_lock: Multiprocessing lock for thread-safe operations.
            manager_list: Multiprocessing list for collecting episode metadata.
        """
        if not h5_paths:
            logger.error("❌ No HDF5 files provided.")
            return

        logger.info(f"🔄 Starting conversion for {len(h5_paths)} HDF5 file(s).")
        output_path = Path(output_dir)
        data_path = output_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        meta_path = output_path / "meta"
        meta_path.mkdir(parents=True, exist_ok=True)

        global_episode_idx_counter = 0
        for i, h5_file in enumerate(h5_paths):
            h5_path_obj = Path(h5_file)

            # Get task name and index
            task_name = self.all_task_names_from_input[i]
            task_index = self.task_to_idx[task_name]

            logger.info(
                f"\nProcessing file {i+1}/{len(h5_paths)}: '{h5_path_obj.name}' "
                f"for task: '{task_name}' (Task Index: {task_index})"
            )

            processed_count = self._process_episodes_frame_by_frame(
                h5_path=h5_path_obj,
                data_dir=data_path,
                task_name=task_name,
                task_index=task_index,
                starting_episode_index=global_episode_idx_counter,
                manager_lock=manager_lock,
                manager_list=manager_list,
            )
            global_episode_idx_counter += processed_count

        self.episode_metadata_list = list(manager_list)
        if not self.episode_metadata_list:
            logger.error("\n❌ No valid episodes were successfully processed across all files.")
            return

        self._generate_and_save_metadata(meta_path)
        logger.info("\n✅ Conversion for all files finished successfully!")

    def _process_episodes_frame_by_frame(
        self,
        h5_path: Path,
        data_dir: Path,
        task_name: str,
        task_index: int,
        starting_episode_index: int,
        manager_lock,
        manager_list,
        max_workers: int = 24,
    ) -> int:
        """
        Process all episodes in an HDF5 file using parallel processing.

        Args:
            h5_path: Path to HDF5 file.
            data_dir: Directory to save episode parquet files.
            task_name: Name of the task.
            task_index: Index of the task.
            starting_episode_index: Starting global episode index.
            manager_lock: Multiprocessing lock.
            manager_list: Multiprocessing list for results.
            max_workers: Maximum number of parallel workers.

        Returns:
            Number of successfully processed episodes.
        """
        processed_episodes_count = 0

        # Get list of demo keys
        with h5py.File(h5_path, "r") as f:
            data_group = f.get("data")
            if not data_group:
                logger.warning(f"No 'data' group found in {h5_path}")
                return 0

            demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]
            demo_keys.sort(key=lambda k: int(k.split("_")[1]))

        # Prepare tasks for parallel processing
        tasks_for_executor = []
        for local_episode_idx, demo_key in enumerate(demo_keys):
            tasks_for_executor.append(
                (
                    demo_key,
                    local_episode_idx,
                    h5_path,
                    data_dir,
                    task_name,
                    task_index,
                    starting_episode_index,
                    manager_lock,
                    manager_list,
                    self.arc_to_gear,
                    self.joint_names,
                    self.hand_joint_max_angles,
                    self.fps,
                    self.features,
                )
            )

        # Execute parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_single_episode, task_args): task_args
                for task_args in tasks_for_executor
            }

            for future in as_completed(futures):
                result = future.result()
                if result:
                    with manager_lock:
                        manager_list.append(result)
                    processed_episodes_count += 1

        return processed_episodes_count

    def _process_single_episode(self, args_tuple):
        """
        Process a single episode (runs in separate process).

        Args:
            args_tuple: Tuple containing all necessary arguments.

        Returns:
            Episode metadata dictionary or None if processing failed.
        """
        (
            demo_key,
            local_episode_idx,
            h5_path,
            data_dir,
            task_name,
            task_index,
            starting_episode_index,
            manager_lock,
            manager_list,
            arc_to_gear_val,
            joint_names_val,
            hand_joint_max_angles_val,
            fps_val,
            features_val,
        ) = args_tuple

        _arc_to_gear = arc_to_gear_val
        _joint_names = joint_names_val
        _hand_joint_max_angles = hand_joint_max_angles_val
        _fps = fps_val
        _features = features_val

        def _radian_to_gear_for_process(radian_value, joint_name, max_gear=2000.0):
            """Local version of radian_to_gear for process isolation."""
            if joint_name in ["left_thumb_0", "right_thumb_0"]:
                return 1195

            max_radian = _hand_joint_max_angles.get(joint_name)
            if max_radian is None or max_radian == 0:
                return radian_value

            radian_value = np.clip(radian_value, 0, max_radian)
            gear_value = (radian_value / max_radian) * max_gear
            return int(round(gear_value))

        episode_idx = starting_episode_index + local_episode_idx
        print(f"   -> Processing episode {local_episode_idx} ('{demo_key}') -> Global Index {episode_idx}...")

        with h5py.File(h5_path, "r") as f:
            episode_group = f[f"data/{demo_key}"]
            current_episode_frames = []

            try:
                actions_dset = episode_group["actions"]
                states_dset = episode_group["states/articulation/robot/joint_position"]
                initial_state_dset = episode_group["initial_state/articulation/robot/joint_position"]
                left_wrist_cam_dset = episode_group["obs/left_wrist_camera_bgr"]
                right_wrist_cam_dset = episode_group["obs/right_wrist_camera_bgr"]
                head_cam_dset = episode_group["obs/head_camera_bgr"]
                num_steps = actions_dset.shape[0]

                if num_steps < 1:
                    print(f"   -> Skipped episode {episode_idx} (no actions recorded).")
                    return None
            except (KeyError, IndexError) as e:
                print(f"   -> Skipped episode {episode_idx} (missing key or data issue: {e}).")
                return None

            # Process each frame
            for t in range(num_steps):
                # Get state: use initial state for t=0, otherwise use previous state
                if t == 0:
                    current_state = initial_state_dset[:].flatten().astype(np.float32)
                else:
                    current_state = states_dset[t - 1].astype(np.float32)

                current_action = actions_dset[t].astype(np.float32)

                # Apply arc-to-gear conversion if enabled
                if _arc_to_gear:
                    converted_state = []
                    converted_action = []
                    for i, joint_name in enumerate(_joint_names):
                        # Check if it's a hand joint (starts with left_ or right_)
                        is_hand_joint = joint_name.startswith(("left_", "right_"))
                        if is_hand_joint:
                            converted_state.append(_radian_to_gear_for_process(current_state[i], joint_name))
                            converted_action.append(_radian_to_gear_for_process(current_action[i], joint_name))
                        else:
                            converted_state.append(current_state[i])
                            converted_action.append(current_action[i])
                    current_state = np.array(converted_state, dtype=np.float32)
                    current_action = np.array(converted_action, dtype=np.float32)

                # Load and resize images
                raw_left_wrist_image = left_wrist_cam_dset[t]
                raw_right_wrist_image = right_wrist_cam_dset[t]
                raw_head_image = head_cam_dset[t]

                pil_left = PILImage.fromarray(raw_left_wrist_image)
                pil_left_resized = pil_left.resize((256, 256), PILImage.Resampling.LANCZOS)
                current_left_wrist_image = np.array(pil_left_resized)

                pil_right = PILImage.fromarray(raw_right_wrist_image)
                pil_right_resized = pil_right.resize((256, 256), PILImage.Resampling.LANCZOS)
                current_right_wrist_image = np.array(pil_right_resized)

                pil_head = PILImage.fromarray(raw_head_image)
                pil_head_resized = pil_head.resize((256, 256), PILImage.Resampling.LANCZOS)
                current_head_image = np.array(pil_head_resized)

                # Create frame data dictionary
                frame_data = {
                    "episode_index": episode_idx,
                    "frame_index": t,
                    "timestamp": float(t / _fps),
                    "index": t,
                    "task_index": task_index,
                    "observation.state": current_state,
                    "action": current_action,
                    "left_wrist_camera_bgr": current_left_wrist_image,
                    "right_wrist_camera_bgr": current_right_wrist_image,
                    "head_camera_bgr": current_head_image,
                }
                current_episode_frames.append(frame_data)

            # Convert to HuggingFace Dataset and save
            if current_episode_frames:
                print(f"   -> Episode {episode_idx}: {len(current_episode_frames)} steps loaded. Converting and saving...")
                episode_dataset = Dataset.from_list(current_episode_frames, features=_features)
                episode_dataset = episode_dataset.rename_columns(
                    {
                        "left_wrist_camera_bgr": "observation.images.left_wrist_camera",
                        "right_wrist_camera_bgr": "observation.images.right_wrist_camera",
                        "head_camera_bgr": "observation.images.head_camera",
                    }
                )
                episode_file = data_dir / f"episode_{episode_idx:06d}.parquet"
                episode_dataset.to_parquet(episode_file)

                # Calculate statistics
                stats = self._calculate_stats_from_raw_list(current_episode_frames)
                if stats:
                    episode_metadata = {
                        "episode_index": episode_idx,
                        "length": len(current_episode_frames),
                        "tasks": [task_name],
                        "stats": stats,
                    }
                    print(f"   -> Episode {episode_idx}: Done.")
                    return episode_metadata

        del current_episode_frames
        gc.collect()
        return None

    def _calculate_stats_from_raw_list(self, episode_frames: List[Dict]) -> Optional[Dict]:
        """
        Calculate statistics (min, max, mean, std) for all features in an episode.

        Args:
            episode_frames: List of frame data dictionaries.

        Returns:
            Dictionary containing statistics for each feature, or None if frames are empty.
        """
        if not episode_frames:
            return None

        all_stats = {}
        column_names = episode_frames[0].keys()
        column_to_feature_map = {
            "left_wrist_camera_bgr": "observation.images.left_wrist_camera",
            "right_wrist_camera_bgr": "observation.images.right_wrist_camera",
            "head_camera_bgr": "observation.images.head_camera",
            "observation.state": "observation.state",
            "action": "action",
            "timestamp": "timestamp",
            "frame_index": "frame_index",
            "episode_index": "episode_index",
            "index": "index",
            "task_index": "task_index",
        }

        for col_name in column_names:
            if col_name not in column_to_feature_map:
                continue

            feature_name = column_to_feature_map[col_name]
            data_list = [frame[col_name] for frame in episode_frames]
            count = len(data_list)

            if col_name in self.image_columns:
                # Image statistics: normalize to [0, 1] before computing
                image_stack = np.stack(data_list).astype(np.float32) / 255.0
                all_stats[feature_name] = {
                    "min": self._reshape_stat(np.min(image_stack, axis=(0, 1, 2))),
                    "max": self._reshape_stat(np.max(image_stack, axis=(0, 1, 2))),
                    "mean": self._reshape_stat(np.mean(image_stack, axis=(0, 1, 2))),
                    "std": self._reshape_stat(np.std(image_stack, axis=(0, 1, 2))),
                    "count": [count],
                }
            else:
                # Scalar or vector statistics
                vector_stack = np.array(data_list, dtype=np.float32)
                if vector_stack.ndim == 1:
                    all_stats[feature_name] = {
                        "min": [vector_stack.min()],
                        "max": [vector_stack.max()],
                        "mean": [vector_stack.mean()],
                        "std": [vector_stack.std()],
                        "count": [count],
                    }
                else:
                    all_stats[feature_name] = {
                        "min": np.min(vector_stack, axis=0),
                        "max": np.max(vector_stack, axis=0),
                        "mean": np.mean(vector_stack, axis=0),
                        "std": np.std(vector_stack, axis=0),
                        "count": [count],
                    }

        return all_stats

    def _reshape_stat(self, stat_array: np.ndarray) -> np.ndarray:
        """Reshape statistics array to (-1, 1, 1) format."""
        return stat_array.reshape(-1, 1, 1)

    def _generate_and_save_metadata(self, meta_path: Path):
        """Generate and save all metadata files."""
        logger.info("\n📝 Generating and saving all metadata files...")
        self._save_info_json(meta_path)
        self._save_episodes_jsonl(meta_path)
        self._save_episodes_stats_jsonl(meta_path)
        self._save_tasks_jsonl(meta_path)

    def _save_json_with_numpy(self, data, path):
        """Save JSON file with Numpy data type support."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=NumpyJSONEncoder, indent=4, ensure_ascii=False)

    def _save_jsonl_with_numpy(self, data_list, path):
        """Save JSONL file with Numpy data type support."""
        with open(path, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(json.dumps(item, cls=NumpyJSONEncoder, ensure_ascii=False) + "\n")

    def _save_info_json(self, meta_path: Path):
        """Save info.json metadata file."""
        total_episodes = len(self.episode_metadata_list)
        total_frames = sum(e["length"] for e in self.episode_metadata_list)

        if not self.episode_metadata_list:
            logger.warning("⚠️ Warning: No episode metadata available to generate info.json.")
            return

        first_ep_stats = self.episode_metadata_list[0]["stats"]
        state_shape = len(first_ep_stats.get("observation.state", {}).get("mean", []))
        action_shape = len(first_ep_stats.get("action", {}).get("mean", []))

        features_dict = {
            "observation.images.left_wrist_camera": {
                "dtype": "image",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.images.right_wrist_camera": {
                "dtype": "image",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.images.head_camera": {
                "dtype": "image",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.state": {"dtype": "float32", "shape": [state_shape], "names": None},
            "action": {"dtype": "float32", "shape": [action_shape], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }

        info = {
            "codebase_version": "v2.1",
            "robot_type": self.robot_config.get("name", "A2"),
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(self.tasks),
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": total_episodes,
            "fps": self.fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/episode_{episode_index:06d}.parquet",
            "video_path": None,
            "features": features_dict,
        }

        self._save_json_with_numpy(info, meta_path / "info.json")
        logger.info("   - Saved info.json")

    def _save_episodes_jsonl(self, meta_path: Path):
        """Save episodes.jsonl metadata file."""
        episodes_to_save = [
            {"episode_index": meta["episode_index"], "tasks": meta["tasks"], "length": meta["length"]}
            for meta in self.episode_metadata_list
        ]
        self._save_jsonl_with_numpy(episodes_to_save, meta_path / "episodes.jsonl")
        logger.info("   - Saved episodes.jsonl")

    def _save_episodes_stats_jsonl(self, meta_path: Path):
        """Save episodes_stats.jsonl metadata file."""
        stats_to_save = [
            {"episode_index": meta["episode_index"], "stats": meta["stats"]}
            for meta in self.episode_metadata_list
        ]
        self._save_jsonl_with_numpy(stats_to_save, meta_path / "episodes_stats.jsonl")
        logger.info("   - Saved episodes_stats.jsonl")

    def _save_tasks_jsonl(self, meta_path: Path):
        """Save tasks.jsonl metadata file."""
        tasks_to_save = [{"task_index": idx, "task": name} for name, idx in self.task_to_idx.items()]
        self._save_jsonl_with_numpy(tasks_to_save, meta_path / "tasks.jsonl")
        logger.info("   - Saved tasks.jsonl")


def main():
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert Isaac Sim HDF5 teleoperation data to LeRobot format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single task with arc-to-gear conversion
  python script/convert.py /path/to/output --data_pairs Task1_data_pair.json --arc2gear

  # Convert multiple tasks without arc-to-gear conversion
  python script/convert.py /path/to/output --data_pairs multi_task_data_pair.json

  # Use custom robot configuration
  python script/convert.py /path/to/output --data_pairs data.json --robot_config custom_robot.json
        """,
    )

    parser.add_argument("output_dir", type=str, help="Path to the output directory for the LeRobot dataset.")

    parser.add_argument(
        "--data_pairs",
        type=str,
        required=True,
        help="Path to JSON file containing list of {h5_path, task_name} pairs.",
    )

    parser.add_argument(
        "--arc2gear",
        action="store_true",
        help="Enable conversion of hand joint values from radians to gear positions (0-2000).",
    )

    parser.add_argument(
        "--robot_config",
        type=str,
        default=None,
        help="Path to robot configuration JSON file (default: comm_config/configs/a2_robot_config.json).",
    )

    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the dataset (default: 15).")

    parser.add_argument(
        "--max_workers", type=int, default=24, help="Maximum number of parallel workers (default: 24)."
    )

    args = parser.parse_args()

    try:
        # Load data pairs
        with open(args.data_pairs, "r", encoding="utf-8") as f:
            data_pairs = json.load(f)

        if not isinstance(data_pairs, list) or not all(
            isinstance(item, dict) and "h5_path" in item and "task_name" in item for item in data_pairs
        ):
            raise ValueError(
                "Invalid format in data_pairs file. Expected JSON list of objects with 'h5_path' and 'task_name' keys."
            )

        h5_paths = [item["h5_path"] for item in data_pairs]
        task_names = [item["task_name"] for item in data_pairs]

    except FileNotFoundError:
        logger.error(f"❌ Error: The data pairs file was not found at '{args.data_pairs}'")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error("❌ Error: Failed to parse the data pairs file. Please ensure it is a valid JSON.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ An error occurred while loading data pairs: {e}")
        sys.exit(1)

    try:
        # Load robot configuration
        if args.robot_config is None:
            # Default robot config path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            robot_config_path = os.path.join(project_root, "comm_config", "configs", "a2_robot_config.json")
        else:
            robot_config_path = args.robot_config

        logger.info(f"Loading robot configuration from: {robot_config_path}")
        robot_config = load_robot_config(robot_config_path)

    except FileNotFoundError:
        logger.error(f"❌ Error: Robot configuration file not found at '{robot_config_path}'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error loading robot configuration: {e}")
        sys.exit(1)

    # Create converter and run conversion
    try:
        with multiprocessing.Manager() as manager:
            manager_lock = manager.Lock()
            manager_list = manager.list()

            converter = HDF5ToLeRobotConverter(
                robot_config=robot_config, task_names=task_names, arc_to_gear=args.arc2gear, fps=args.fps
            )

            converter.convert_and_save(
                h5_paths=h5_paths, output_dir=args.output_dir, manager_lock=manager_lock, manager_list=manager_list
            )

    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
