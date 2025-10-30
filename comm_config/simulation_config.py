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
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from utility.logger import Logger

@dataclass
class TaskRelatedObject:
    name: str
    description: str
    cylinder_prim_path: str
    plate_prim_path: str
    basket_prim_path: str
    target_prim_path: str
    
@dataclass
class SimulationConfig:

    # Basic Configuration
    headless: bool = False
    physics_dt: float = 1.0 / 120.0
    rendering_dt: float = 1.0 / 60.0
    stage_units_in_meters: float = 1.0

    # Scene Configuration
    scene_usd_path: Optional[str] = None
    scene_prim_path: str = "/scene"

    # Target Object Configuration
    task_related_objects: Optional[list[TaskRelatedObject]] = None

    # List of sampleable active objects
    sampled_active_object: Optional[list[str]] = None

    # Control Update Frequency
    control_update_interval: float = 0.012  # 12ms

    # IK Configuration
    ik_diff_threshold: float = 1e-3
    joint_jump_threshold_deg: float = 30.0

    # Target Cube Movement Threshold
    target_cube_move_threshold_min: float = 0.001
    target_cube_move_threshold_max: float = 0.10

    # Joystick Deadzone
    joystick_deadzone: float = 0.03

    teleop_toggle_debounce_time: float = 0.5

    def _load_basic_config(self, sim_json_config: Dict[str, Any]):
        """Load basic configuration"""
        if "headless" in sim_json_config:
            self.headless = sim_json_config["headless"]
        if "physics_dt" in sim_json_config:
            self.physics_dt = sim_json_config["physics_dt"]
        if "rendering_dt" in sim_json_config:
            self.rendering_dt = sim_json_config["rendering_dt"]
        if "stage_units_in_meters" in sim_json_config:
            self.stage_units_in_meters = sim_json_config["stage_units_in_meters"]

    def _load_scene_config(self, task_config: Dict[str, Any]):

        if "scene" in task_config:
            scene_config = task_config["scene"][0]
            Logger.info(f"Load scene with {json.dumps(scene_config, indent=4)}")
            if "scene_usd" in scene_config:
                self.scene_usd_path = scene_config["scene_usd"]
            if "prime_path" in scene_config:
                self.scene_prim_path = scene_config["prime_path"]
            if "task_related_objects" in scene_config:
                self.task_related_objects = [
                    TaskRelatedObject(
                        name=obj_data.get("name"),
                        description=obj_data.get("description"),
                        cylinder_prim_path=obj_data.get("cylinder_prim_path"),
                        plate_prim_path=obj_data.get("plate_prim_path"),
                        basket_prim_path=obj_data.get("basket_prim_path"),
                        target_prim_path=obj_data.get("target_prim_path"),
                    )
                    for obj_data in scene_config["task_related_objects"]
                ]
            if "sampled_active_object" in scene_config:
                self.sampled_active_object = scene_config["sampled_active_object"]

    def _load_control_config(self, control_config: Dict[str, Any]):

        if "update_interval" in control_config:
            self.control_update_interval = control_config["update_interval"]
        if "joystick_deadzone" in control_config:
            self.joystick_deadzone = control_config["joystick_deadzone"]
        if "teleop_toggle_debounce_time" in control_config:
            self.teleop_toggle_debounce_time = control_config["teleop_toggle_debounce_time"]

    def _load_ik_config(self, ik_config: Dict[str, Any]):
        """Load IK configuration"""
        if "diff_threshold" in ik_config:
            self.ik_diff_threshold = ik_config["diff_threshold"]
        if "joint_jump_threshold_deg" in ik_config:
            self.joint_jump_threshold_deg = ik_config["joint_jump_threshold_deg"]

    def _load_target_cube_threshold(self, threshold_config: Dict[str, Any]):
        """Load target cube movement threshold"""
        if "min" in threshold_config:
            self.target_cube_move_threshold_min = threshold_config["min"]
        if "max" in threshold_config:
            self.target_cube_move_threshold_max = threshold_config["max"]

    @classmethod
    def from_json(cls, sim_json_path: str, task_config) -> "SimulationConfig":
        json_path_obj = Path(sim_json_path)
        if not json_path_obj.exists():
            raise FileNotFoundError(f'{sim_json_path} not found')

        config = cls()

        try:
            with open(json_path_obj, "r", encoding="utf-8") as f:
                Logger.info(f'{json_path_obj} load successful!')
                sim_json_config = json.load(f)

                config._load_basic_config(sim_json_config)

                config._load_scene_config(task_config)

                if "control" in sim_json_config:
                    config._load_control_config(sim_json_config["control"])

                if "ik" in sim_json_config:
                    config._load_ik_config(sim_json_config["ik"])

                if "target_cube_move_threshold" in sim_json_config:
                    config._load_target_cube_threshold(sim_json_config["target_cube_move_threshold"])

                Logger.info(f"Load from {json_path_obj} successful!")

        except Exception as e:
            Logger.warning(f"Load from {json_path_obj} failure: {e}")
            raise

        return config
