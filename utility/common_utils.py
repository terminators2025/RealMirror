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
import yaml
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union
import carb
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Usd, UsdGeom, Gf
from utility.logger import Logger



class CommonUtils:
    

    @staticmethod
    def load_yaml_config(file_path: str) -> Optional[Dict]:
        if not os.path.exists(file_path):
            Logger.error(f"YAML file not found: {file_path}")
            return None

        try:
            with open(file_path, "r") as f:
                config = yaml.safe_load(f)
                Logger.info(f"Loaded YAML config from {file_path}")
                return config
        except Exception as e:
            Logger.error(f"Failed to load YAML file {file_path}: {e}")
            return None

    @staticmethod
    def load_json_config(file_path: str) -> Optional[Dict]:
        if not os.path.exists(file_path):
            Logger.error(f"JSON file not found: {file_path}")
            return None

        try:
            with open(file_path, "r") as f:
                config = json.load(f)
                Logger.info(f"Loaded JSON config from {file_path}")
                return config
        except Exception as e:
            Logger.error(f"Failed to load JSON file {file_path}: {e}")
            return None

    @staticmethod
    def save_yaml_config(config: Dict, file_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
                Logger.info(f"Saved YAML config to {file_path}")
                return True
        except Exception as e:
            Logger.error(f"Failed to save YAML file {file_path}: {e}")
            return False

    @staticmethod
    def save_json_config(config: Dict, file_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(config, f, indent=4)
                Logger.info(f"Saved JSON config to {file_path}")
                return True
        except Exception as e:
            Logger.error(f"Failed to save JSON file {file_path}: {e}")
            return False


class TransformUtils:
    

    @staticmethod
    def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
        return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

    @staticmethod
    def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    @staticmethod
    def euler_to_quat(
        euler_angles: np.ndarray, sequence: str = "xyz", degrees: bool = True
    ) -> np.ndarray:
        r = R.from_euler(sequence, euler_angles, degrees=degrees)
        quat_xyzw = r.as_quat()
        return TransformUtils.quat_xyzw_to_wxyz(quat_xyzw)

    @staticmethod
    def quat_to_euler(
        quat_wxyz: np.ndarray, sequence: str = "xyz", degrees: bool = True
    ) -> np.ndarray:
        quat_xyzw = TransformUtils.quat_wxyz_to_xyzw(quat_wxyz)
        r = R.from_quat(quat_xyzw)
        return r.as_euler(sequence, degrees=degrees)

    @staticmethod
    def pose_to_matrix(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        mat = np.eye(4)
        quat_xyzw = TransformUtils.quat_wxyz_to_xyzw(orientation)
        mat[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        mat[:3, 3] = position
        return mat

    @staticmethod
    def matrix_to_pose(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        position = matrix[:3, 3]
        rotation_matrix = matrix[:3, :3]
        r = R.from_matrix(rotation_matrix)
        quat_xyzw = r.as_quat()
        quat_wxyz = TransformUtils.quat_xyzw_to_wxyz(quat_xyzw)
        return position, quat_wxyz

    @staticmethod
    def multiply_transforms(
        pose1: Tuple[np.ndarray, np.ndarray], pose2: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        mat1 = TransformUtils.pose_to_matrix(pose1[0], pose1[1])
        mat2 = TransformUtils.pose_to_matrix(pose2[0], pose2[1])
        result_mat = mat1 @ mat2
        return TransformUtils.matrix_to_pose(result_mat)

    @staticmethod
    def inverse_transform(
        position: np.ndarray, orientation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        mat = TransformUtils.pose_to_matrix(position, orientation)
        inv_mat = np.linalg.inv(mat)
        return TransformUtils.matrix_to_pose(inv_mat)


class MathUtils:
    

    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
 
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return np.zeros_like(vector)
        return vector / norm

    @staticmethod
    def angle_between_vectors(
        v1: np.ndarray, v2: np.ndarray, degrees: bool = False
    ) -> float:

        v1_norm = MathUtils.normalize_vector(v1)
        v2_norm = MathUtils.normalize_vector(v2)
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle) if degrees else angle

    @staticmethod
    def wrap_angle(angle: float, degrees: bool = False) -> float:

        if degrees:
            while angle > 180:
                angle -= 360
            while angle < -180:
                angle += 360
        else:
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += 2 * np.pi
        return angle

    @staticmethod
    def lerp(
        start: Union[float, np.ndarray], end: Union[float, np.ndarray], t: float
    ) -> Union[float, np.ndarray]:
 
        t = np.clip(t, 0.0, 1.0)
        return start + (end - start) * t

    @staticmethod
    def slerp(quat1: np.ndarray, quat2: np.ndarray, t: float) -> np.ndarray:

        q1_xyzw = TransformUtils.quat_wxyz_to_xyzw(quat1)
        q2_xyzw = TransformUtils.quat_wxyz_to_xyzw(quat2)

        r1 = R.from_quat(q1_xyzw)
        r2 = R.from_quat(q2_xyzw)

        slerp_obj = R.from_quat([q1_xyzw, q2_xyzw])
        result = (
            slerp_obj[0]
            if t <= 0
            else (slerp_obj[1] if t >= 1 else R.slerp([0, 1], [r1, r2])(t))
        )

        result_xyzw = result.as_quat()
        return TransformUtils.quat_xyzw_to_wxyz(result_xyzw)

    @staticmethod
    def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:

        if len(data) < window_size:
            return data

        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode="same")


class PrimUtils:
    

    @staticmethod
    def is_prim_valid(prim_path: str) -> bool:
      
        prim = get_prim_at_path(prim_path)
        return prim is not None and prim.IsValid()

    @staticmethod
    def get_prim_children(prim_path: str) -> List[str]:

        prim = get_prim_at_path(prim_path)
        if not prim or not prim.IsValid():
            return []

        children = []
        for child in prim.GetChildren():
            children.append(str(child.GetPath()))
        return children

    @staticmethod
    def find_prims_by_type(root_path: str, prim_type: str) -> List[str]:
        result = []

        def _recursive_search(prim_path: str):
            prim = get_prim_at_path(prim_path)
            if not prim or not prim.IsValid():
                return

            if prim.GetTypeName() == prim_type:
                result.append(prim_path)

            for child in prim.GetChildren():
                _recursive_search(str(child.GetPath()))

        _recursive_search(root_path)
        return result

    @staticmethod
    def get_prim_attribute(prim_path: str, attr_name: str) -> Any:
        prim = get_prim_at_path(prim_path)
        if not prim or not prim.IsValid():
            return None

        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            return attr.Get()
        return None

    @staticmethod
    def set_prim_attribute(prim_path: str, attr_name: str, value: Any) -> bool:
        prim = get_prim_at_path(prim_path)
        if not prim or not prim.IsValid():
            return False

        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            attr.Set(value)
            return True
        return False

    @staticmethod
    def get_prim_world_transform(
        prim_path: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            from omni.isaac.core.utils.xforms import get_world_pose

            pos, quat = get_world_pose(prim_path)
            return pos, quat
        except Exception as e:
            Logger.error(f"Failed to get world transform for {prim_path}: {e}")
            return None


class FilterUtils:
    

    @staticmethod
    def low_pass_filter(
        data: np.ndarray, cutoff_freq: float, sample_rate: float
    ) -> np.ndarray:
        from scipy import signal

        nyquist = sample_rate / 2
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(4, normal_cutoff, btype="low", analog=False)
        return signal.filtfilt(b, a, data)

    @staticmethod
    def median_filter(data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        from scipy import signal

        return signal.medfilt(data, kernel_size=kernel_size)

    @staticmethod
    def kalman_filter_1d(
        measurements: np.ndarray,
        process_variance: float = 1e-5,
        measurement_variance: float = 0.1,
    ) -> np.ndarray:
        n = len(measurements)
        filtered = np.zeros(n)

        x = measurements[0]
        p = 1.0

        for i in range(n):
            x_pred = x
            p_pred = p + process_variance

            k = p_pred / (p_pred + measurement_variance)
            x = x_pred + k * (measurements[i] - x_pred)
            p = (1 - k) * p_pred

            filtered[i] = x

        return filtered


class ValidationUtils:
    

    @staticmethod
    def validate_joint_limits(
        joint_positions: np.ndarray, joint_limits: List[Tuple[float, float]]
    ) -> np.ndarray:
        validated = np.copy(joint_positions)
        for i, (min_val, max_val) in enumerate(joint_limits):
            if i < len(validated):
                validated[i] = np.clip(validated[i], min_val, max_val)
        return validated

    @staticmethod
    def check_collision_free(
        position: np.ndarray, obstacles: List[Tuple[np.ndarray, float]]
    ) -> bool:
        for center, radius in obstacles:
            if np.linalg.norm(position - center) < radius:
                return False
        return True

    @staticmethod
    def is_pose_reachable(
        target_pos: np.ndarray, workspace_center: np.ndarray, workspace_radius: float
    ) -> bool:
        return np.linalg.norm(target_pos - workspace_center) <= workspace_radius


load_yaml = CommonUtils.load_yaml_config
load_json = CommonUtils.load_json_config
save_yaml = CommonUtils.save_yaml_config
save_json = CommonUtils.save_json_config

quat_wxyz_to_xyzw = TransformUtils.quat_wxyz_to_xyzw
quat_xyzw_to_wxyz = TransformUtils.quat_xyzw_to_wxyz
euler_to_quat = TransformUtils.euler_to_quat
quat_to_euler = TransformUtils.quat_to_euler
pose_to_matrix = TransformUtils.pose_to_matrix
matrix_to_pose = TransformUtils.matrix_to_pose

normalize = MathUtils.normalize_vector
lerp = MathUtils.lerp
slerp = MathUtils.slerp

is_prim_valid = PrimUtils.is_prim_valid
get_prim_children = PrimUtils.get_prim_children