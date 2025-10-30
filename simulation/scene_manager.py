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
import os
import numpy as np
from typing import Optional
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdGeom, Gf, UsdPhysics
from utility.logger import Logger
from utility.physics_utils import diagnose_scene_physics
from simulation.object_manager import ObjectManager, create_object_manager_from_config


class SceneManager:

    def __init__(self, world, config=None, task_config=None):
        self.world = world
        self.sim_config = config
        self.task_config = task_config
        self.object_manager: Optional[ObjectManager] = None
        self.cameras = {}
        self.enable_physics_diagnostics = True  # Enable physics diagnostics by default
        self.auto_fix_physics = True  # Enable automatic physics fixes by default

    def setup_scene(self):
        self._load_scene_usd()
        self._adjust_ground_plane()
        self._create_object_manager()
        self._initialize_object_manager()
        
        # Run physics diagnostics and fixes after scene is loaded
        if self.enable_physics_diagnostics:
            self._diagnose_and_fix_physics()
    
    def _diagnose_and_fix_physics(self):
        """Run physics diagnostics and optionally apply automatic fixes."""
        scene_path = self.sim_config.scene_prim_path if self.sim_config else "/scene"
        
        try:
            diagnose_scene_physics(
                self.world,
                scene_path=scene_path,
                auto_fix=self.auto_fix_physics
            )
        except Exception as e:
            Logger.warning(f"Physics diagnostics failed: {e}")
            # Don't fail the entire scene setup if diagnostics fail
            pass


    def _create_object_manager(self):
        if not self.task_config:
            Logger.warning("No task config provided, skipping object manager initialization")
            return

        self.object_manager = create_object_manager_from_config(task_config=self.task_config, move_step_length=0.015)

    def _initialize_object_manager(self):
        if self.object_manager:
            stage = self.world.stage
            success = self.object_manager.initialize_all_groups(stage, self.world)

            if success:
                Logger.info(
                    f"ObjectManager initialized with {len(self.object_manager.groups)} groups, "
                    f"active: '{self.object_manager.get_active_group_name()}'"
                )
            else:
                Logger.warning("ObjectManager initialization had some failures")

    def _load_scene_usd(self):
        """Load the scene USD file."""
        if self.sim_config and self.sim_config.scene_usd_path:
            if os.path.exists(self.sim_config.scene_usd_path):
                add_reference_to_stage(
                    usd_path=self.sim_config.scene_usd_path,
                    prim_path=self.sim_config.scene_prim_path,
                )
                Logger.info(f"Loaded scene from {self.sim_config.scene_usd_path}")

    def _adjust_ground_plane(self):
        """Adjust the ground plane position and visibility."""
        bias = -1.20
        ground_path = "/World/defaultGroundPlane/GroundPlane"
        env_path = "/World/defaultGroundPlane/Environment"

        for path in [ground_path, env_path]:
            prim = get_prim_at_path(path)
            if prim and prim.IsValid():
                xform = UsdGeom.Xformable(prim)
                translate_op = None
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        translate_op.Set(Gf.Vec3d(0, 0, bias))
                        break
                if not translate_op:
                    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, bias))

        env_prim = get_prim_at_path(env_path)
        if env_prim and env_prim.IsValid():
            UsdGeom.Imageable(env_prim).GetVisibilityAttr().Set("invisible")

    def set_active_object(self, object_name_or_index):
        """
        Switch the active object group.

        Args:
            object_name_or_index: Group name (str) or index (int)
        """
        if not self.object_manager:
            Logger.warning("ObjectManager not initialized")
            return

        if isinstance(object_name_or_index, int):
            self.object_manager.set_active_group_by_index(object_name_or_index)
        else:
            self.object_manager.set_active_group(object_name_or_index)

    def reset_active_object_pose(self):
        """Reset all objects in the active group to their initial poses."""
        if not self.object_manager:
            Logger.warning("ObjectManager not initialized")
            return

        self.object_manager.reset_active_group_poses()

    def move_active_object(self, direction: str, step_size: float = 0.015):
        """
        Move all objects in the active group.

        Args:
            direction: Movement direction ("up", "down", "left", "right")
            step_size: Movement step size (meters)
        """
        if not self.object_manager:
            Logger.warning("ObjectManager not initialized")
            return

        self.object_manager.move_active_group(direction, step_size)
