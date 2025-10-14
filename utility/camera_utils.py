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
import carb
from typing import Dict, Optional, Tuple
from omni.isaac.sensor import Camera
from utility.logger import Logger


def get_camera_image(camera: Camera, image_type: str = "rgb") -> Optional[np.ndarray]:
    if not camera or not camera.is_valid():
        return None

    try:
        if image_type == "rgb":
            rgba = camera.get_rgba()
            if rgba is not None and rgba.shape[2] == 4:
                return rgba[..., [2, 1, 0]].astype(np.uint8)
        elif image_type == "depth":
            return camera.get_depth()
        elif image_type == "segmentation":
            return camera.get_image_array_segmentation()
        else:
            Logger.error(f"Unknown image type: {image_type}")
            return None
    except Exception as e:
        Logger.error(f"Failed to get camera image: {e}")
        return None
