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

from typing import Any, List, Optional, Tuple
import numpy as np
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from utility.logger import Logger


class PrimUtils:

    @staticmethod
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
