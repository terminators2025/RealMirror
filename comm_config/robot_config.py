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

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class RobotConfig:
    name: str
    usd_path: str
    urdf_path: str
    prim_path: str

    robot_descriptor_paths: Dict[str, str] = field(default_factory=dict)

    arm_joints: Dict[str, List[str]] = field(default_factory=dict)
    hand_joints: Dict[str, List[str]] = field(default_factory=dict)
    wrist_joints: Dict[str, Dict[str, str]] = field(default_factory=dict)

    gripper_config: Dict[str, Dict] = field(default_factory=dict)

    finger_joint_names: set[str] = field(default_factory=set)

    default_kp: float = 400.0
    default_kd: float = 40.0
    finger_kp: float = 1.0e10
    finger_kd: float = 6.0e7
    finger_max_force: float = 1.0e10

    initial_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    initial_orientation: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    initial_joint_positions: Dict[str, float] = field(default_factory=dict)

    gripper_angular_speed: float = np.deg2rad(980.0)
    gripper_convergence_threshold: float = np.deg2rad(2.0)
    wrist_roll_speed: float = np.deg2rad(1080.0)
    wrist_roll_limits: Dict[str, float] = field(default_factory=dict)

    camera_configs: Dict[str, Dict] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_path: str):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Configuration file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        instance = cls(
            name=config_dict.get("name", "robot"),
            usd_path=config_dict.get("usd_path", ""),
            urdf_path=config_dict.get("urdf_path", ""),
            prim_path=config_dict.get("prim_path", "/World/Robot"),
        )

        if "robot_descriptor_paths" in config_dict:
            instance.robot_descriptor_paths = config_dict["robot_descriptor_paths"]

        if "arm_joints" in config_dict:
            instance.arm_joints = config_dict["arm_joints"]
        if "hand_joints" in config_dict:
            instance.hand_joints = config_dict["hand_joints"]
        if "wrist_joints" in config_dict:
            instance.wrist_joints = config_dict["wrist_joints"]

        if "finger_joint_names" in config_dict:
            instance.finger_joint_names = set(config_dict["finger_joint_names"])

        if "gripper_config" in config_dict:
            instance.gripper_config = cls._process_gripper_config(
                config_dict["gripper_config"]
            )

        if "physics_params" in config_dict:
            physics = config_dict["physics_params"]
            instance.default_kp = physics.get("default_kp", 400.0)
            instance.default_kd = physics.get("default_kd", 40.0)
            instance.finger_kp = physics.get("finger_kp", 1.0e10)
            instance.finger_kd = physics.get("finger_kd", 6.0e7)
            instance.finger_max_force = physics.get("finger_max_force", 1.0e10)

        if "control_params" in config_dict:
            control = config_dict["control_params"]
            instance.gripper_angular_speed = np.deg2rad(
                control.get("gripper_angular_speed_deg", 980.0)
            )
            instance.gripper_convergence_threshold = np.deg2rad(
                control.get("gripper_convergence_threshold_deg", 2.0)
            )
            instance.wrist_roll_speed = np.deg2rad(
                control.get("wrist_roll_speed_deg", 1080.0)
            )
            if "wrist_roll_limits_deg" in control:
                limits = control["wrist_roll_limits_deg"]
                instance.wrist_roll_limits = {
                    "min": np.deg2rad(limits.get("min", 0.0)),
                    "max": np.deg2rad(limits.get("max", 179.0)),
                }

        if "initial_pose" in config_dict:
            pose = config_dict["initial_pose"]
            instance.initial_position = np.array(pose.get("position", [0.0, 0.0, 0.0]))
            instance.initial_orientation = np.array(
                pose.get("orientation", [1.0, 0.0, 0.0, 0.0])
            )

        if "initial_joint_positions" in config_dict:
            instance.initial_joint_positions = cls._process_initial_joint_positions(
                config_dict["initial_joint_positions"]
            )

        if "camera_configs" in config_dict:
            instance.camera_configs = config_dict["camera_configs"]

        return instance

    @staticmethod
    def _process_gripper_config(gripper_config: Dict) -> Dict:
        processed_config = {}

        for arm, config in gripper_config.items():
            processed_config[arm] = {}

            if "stage1_joints" in config:
                processed_config[arm]["stage1_joints"] = config["stage1_joints"]
            if "stage2_joints" in config:
                processed_config[arm]["stage2_joints"] = config["stage2_joints"]

            if "stage1_closed_angles_deg" in config:
                angles_deg = config["stage1_closed_angles_deg"]
                processed_config[arm]["stage1_closed_angles"] = [
                    np.deg2rad(angle) for angle in angles_deg
                ]
            elif "stage1_closed_angles" in config:
                processed_config[arm]["stage1_closed_angles"] = config[
                    "stage1_closed_angles"
                ]

            if "stage2_closed_angles_deg" in config:
                angles_deg = config["stage2_closed_angles_deg"]
                processed_config[arm]["stage2_closed_angles"] = [
                    np.deg2rad(angle) for angle in angles_deg
                ]
            elif "stage2_closed_angles" in config:
                processed_config[arm]["stage2_closed_angles"] = config[
                    "stage2_closed_angles"
                ]

        return processed_config

    @staticmethod
    def _process_initial_joint_positions(joint_positions: Dict) -> Dict:
        processed_positions = {}

        for joint_name, value in joint_positions.items():
            if joint_name.endswith("_deg"):

                actual_joint_name = joint_name[:-4]
                processed_positions[actual_joint_name] = np.deg2rad(value)
            else:
                processed_positions[joint_name] = value

        return processed_positions

    def _gripper_config_to_dict(self) -> Dict:

        dict_config = {}
        for arm, config in self.gripper_config.items():
            dict_config[arm] = {}
            for key, value in config.items():
                if "closed_angles" in key and not key.endswith("_deg"):

                    dict_config[arm][key + "_deg"] = [np.rad2deg(v) for v in value]
                else:
                    dict_config[arm][key] = value
        return dict_config

    def _joint_positions_to_dict(self) -> Dict:

        dict_positions = {}
        for joint_name, value in self.initial_joint_positions.items():

            if (
                "thumb_swing_joint" in joint_name
                and abs(value - np.deg2rad(90.0)) < 0.001
            ):
                dict_positions[joint_name + "_deg"] = 90.0
            else:
                dict_positions[joint_name] = value
        return dict_positions
