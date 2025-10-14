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
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdGeom, Gf, UsdPhysics
from utility.logger import Logger


class SceneManager:

    def __init__(self, world, config=None):
        self.world = world
        self.sim_config = config
        self.all_object_prims = {}
        self.activate_group_name = None
        self.object_groups_name = []
        self.cameras = {}

    def setup_scene(self):
        self._load_scene_usd()
        self._adjust_ground_plane()
        self._initialize_target_objects()

    def _load_scene_usd(self):
        if self.sim_config and self.sim_config.scene_usd_path:
            if os.path.exists(self.sim_config.scene_usd_path):
                add_reference_to_stage(
                    usd_path=self.sim_config.scene_usd_path,
                    prim_path=self.sim_config.scene_prim_path,
                )
                Logger.info(f"Loaded scene from {self.sim_config.scene_usd_path}")

    def _adjust_ground_plane(self):
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
            UsdGeom.Imageable(env_prim).GetVisibilityAttr().Set('invisible')

    def _initialize_target_objects(self):
        if not self.sim_config or not self.sim_config.task_related_objects:
            raise ValueError("No task_related_objects found in config.")
        stage = self.world.stage
        self.world.step(render=False)
        for obj_info in self.sim_config.task_related_objects:
            group_name = obj_info.name
            target_prim_paths = [obj_info.cylinder_prim_path, obj_info.plate_prim_path, obj_info.basket_prim_path, obj_info.target_prim_path]
            for prim_path in target_prim_paths:
                if (
                    prim_path is not None
                    and stage.GetPrimAtPath(prim_path).IsValid()
                    and prim_path not in self.all_object_prims
                ):
                    prim_name = prim_path.replace("/", "_")
                    rigid_prim = self.world.scene.add(
                        RigidPrim(prim_path=prim_path, name=f"{prim_name}")
                    )
                    pos, orn = rigid_prim.get_world_pose()
                    self.all_object_prims[prim_path] = {
                        "prim" : rigid_prim,
                        "initial_poses": (pos.copy(), orn.copy()),
                        "group": group_name
                    }
                    Logger.info(f"Initialized object group '{group_name}' at {prim_path} on [{pos}]")
        self.object_groups_name = [obj_info.name for obj_info in self.sim_config.task_related_objects]
        self.activate_group_name = self.object_groups_name[0]
        Logger.info(f"Active object set to: {self.activate_group_name}")

    def set_active_object(self, object_name_or_index):
        if isinstance(object_name_or_index, int):
            if 0 <= object_name_or_index < len(self.object_groups_name):
                self.activate_group_name = self.object_groups_name[object_name_or_index]
                Logger.info(f"Active object switched to '{self.activate_group_name}' by index {object_name_or_index}.")
            else:
                Logger.warning(f"Invalid object index: {object_name_or_index}")
        elif object_name_or_index in self.object_groups_name:
            self.activate_group_name = object_name_or_index
            Logger.info(f"Active object switched to '{self.activate_group_name}'.")
        else:
            Logger.warning(f"Object '{object_name_or_index}' not found.")

    def reset_active_object_pose(self):
        if not self.activate_group_name or self.activate_group_name not in self.object_groups_name:
            Logger.warning("No active object to reset.")
            return

        activate_group = next((obj for obj in self.sim_config.task_related_objects if obj.name == self.activate_group_name), None)
        if not activate_group:
            raise RuntimeError(f"Active object '{self.activate_group_name}' not in task_related_objects.")
        target_prim_paths = [
            activate_group.cylinder_prim_path,
            activate_group.plate_prim_path,
            activate_group.basket_prim_path,
            activate_group.target_prim_path,
        ]
        for prim_path in target_prim_paths:
            if prim_path is not None and prim_path in self.all_object_prims:
                obj_data = self.all_object_prims[prim_path]
                prim = obj_data["prim"]
                initial_pos, initial_orn = obj_data["initial_poses"]
                prim.set_world_pose(position=initial_pos, orientation=initial_orn)
                prim.set_linear_velocity(np.zeros(3))
                prim.set_angular_velocity(np.zeros(3))
                Logger.info(f"Reset '{prim_path}' of group '{self.activate_group_name}' to its initial pose.")

    def move_active_object(self, direction: str, step_size: float = 0.015):
        if not self.activate_group_name or self.activate_group_name not in self.object_groups_name:
            Logger.warning("No active object to move.")
            Logger.warning("No active object to move.")
            return

        activate_group = next((obj for obj in self.sim_config.task_related_objects if obj.name == self.activate_group_name), None)
        if not activate_group:
            Logger.warning(f"Configuration for active object '{self.activate_group_name}' not found.")
            return
        target_prim_paths = [
            activate_group.cylinder_prim_path,
            activate_group.plate_prim_path,
            activate_group.basket_prim_path,
            activate_group.target_prim_path,
        ]
        for target_prim_path in target_prim_paths:
            if (
                target_prim_path is not None
                and target_prim_path in self.sim_config.sampled_active_object
            ):
                obj_data = self.all_object_prims[target_prim_path]
                prim = obj_data["prim"]
                current_pos, current_orn = prim.get_world_pose()
                move_vector = np.zeros(3)
                if direction == "up":
                    move_vector[0] += step_size
                elif direction == "down":
                    move_vector[0] -= step_size
                elif direction == "left":
                    move_vector[1] += step_size
                elif direction == "right":
                    move_vector[1] -= step_size
                new_pos = current_pos + move_vector
                prim.set_world_pose(position=new_pos, orientation=current_orn)
                Logger.info(f"Moved {target_prim_path} in Group '{self.activate_group_name}'  to {new_pos}.")
