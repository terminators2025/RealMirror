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

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any
from omni.isaac.sensor import Camera
from isaacsim.core.utils.prims import get_prim_at_path
from utility.logger import Logger
from comm_config.robot_config import RobotConfig



@dataclass
class CameraConfig:
    """Configuration and factory class for creating cameras in the simulation."""

    @staticmethod
    def _create_single_camera(
        name: str,
        prim_path: str,
        height: int,
        width: int,
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        focal_length: Optional[float] = None,
        horizontal_aperture: Optional[float] = None,
        clipping_range: Optional[tuple[float, float]] = None,
    ) -> Camera:
        """Initializes and configures a single camera prim."""
        camera = Camera(
            name=name,
            prim_path=prim_path,
            resolution=(width, height),
        )
        camera.initialize()

        if position is not None and orientation is not None:
            camera.set_world_pose(position=position, orientation=orientation)

        _prim = get_prim_at_path(prim_path)
        if focal_length is not None:
            _prim.GetAttribute("focalLength").Set(focal_length)
        if horizontal_aperture is not None:
            _prim.GetAttribute("horizontalAperture").Set(horizontal_aperture)
        if clipping_range is not None:
            _prim.GetAttribute("clippingRange").Set(clipping_range)
        
        Logger.info(f"Successfully created camera: {name} at {prim_path}")
        return camera

    @classmethod
    def create_cameras_from_task_config(
        cls, task_config: Dict[str, Any], robot_config: RobotConfig
    ) -> Dict[str, Camera]:
        """
        Creates all cameras defined in the task configuration.

        Handles both robot-attached cameras and world-space scene cameras.

        Args:
            task_config: The configuration dictionary loaded from a task JSON file.
            robot_config: The configuration object for the robot, used to get the base prim path.

        Returns:
            A dictionary mapping camera names to created Camera objects.
        """
        cameras = {}
        if "scene" not in task_config or not task_config["scene"]:
            raise ValueError(f'No cameras config in task_config')

        # The scene is a list, we process the camera configs from the first scene entry
        scene_config = task_config["scene"][0]
        if "camera_configs" not in scene_config:
            raise ValueError(f'No as config in task_config')

        camera_definitions = scene_config["camera_configs"]
        scene_prim_path = scene_config.get("prime_path", "/scene")
        robot_prim_path = robot_config.prim_path

        for cam_name, cam_props in camera_definitions.items():
            try:
                height = cam_props.get("height", 256)
                width = cam_props.get("width", 256)
                position = cam_props.get("position")
                orientation = cam_props.get("orientation")
                prim_path_suffix = cam_props.get("prim_path_suffix")

                if prim_path_suffix:
                    # It's a robot-attached camera
                    full_prim_path = f"{robot_prim_path}{prim_path_suffix}"
                else:
                    Logger.warning(f"Skipping camera '{cam_name}' due to missing 'prim_path_suffix'.")
                    continue

                clipping_range = cam_props.get("clipping_range")
                if isinstance(clipping_range, list):
                    clipping_range = tuple(clipping_range)

                camera = cls._create_single_camera(
                    name=cam_name,
                    prim_path=full_prim_path,
                    height=height,
                    width=width,
                    position=position,
                    orientation=orientation,
                    focal_length=cam_props.get("focal_length"),
                    horizontal_aperture=cam_props.get("horizontal_aperture"),
                    clipping_range=clipping_range,
                )
                cameras[cam_name] = camera
            except Exception as e:
                Logger.error(f"Failed to create camera '{cam_name}': {e}")

        return cameras