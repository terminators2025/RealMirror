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
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
import numpy as np
from typing import Optional, Dict, List, Tuple
from omni.isaac.core.prims import RigidPrim
from utility.logger import Logger


class TrackedObject:
    def __init__(self, name: str, prim_path: str, is_primary_for_recorder: bool = False, is_movable: bool = False):
        self.name = name
        self.prim_path = prim_path
        self.is_primary_for_recorder = is_primary_for_recorder
        self.is_movable = is_movable
        self.handle: Optional[RigidPrim] = None
        self.initial_pos: Optional[np.ndarray] = None
        self.initial_orn: Optional[np.ndarray] = None
        Logger.debug(f'TrackedObject initialized: name={name}, prim_path={prim_path},  is_movable={is_movable}')
        

    def initialize(self, stage, world, rigid_prim_cache: Dict[str, Tuple[RigidPrim, np.ndarray, np.ndarray]]) -> bool:
        if not stage.GetPrimAtPath(self.prim_path).IsValid():
            Logger.warning(f"Failed to find the target object '{self.name}' Prim on the stage: {self.prim_path}")
            return False

        try:
            if self.prim_path in rigid_prim_cache:
                self.handle, self.initial_pos, self.initial_orn = rigid_prim_cache[self.prim_path]
                Logger.info(
                    f"Reusing existing RigidPrim and cached pose for object '{self.name}' ({self.prim_path}) "
                    f"Initial pose: Pos={self.initial_pos}, Orn={self.initial_orn}"
                )
            else:
                # Create a new RigidPrim and add it to the cache
                self.handle = RigidPrim(prim_path=self.prim_path, name=f"tracked_object_{self.name}")
                self.handle.initialize()

                world.step(render=False)  # Ensure the physical state is updated
                self.initial_pos, self.initial_orn = self.handle.get_world_pose()

                rigid_prim_cache[self.prim_path] = (self.handle, self.initial_pos, self.initial_orn)
                Logger.info(
                    f"Created new RigidPrim for object '{self.name}' ({self.prim_path})"
                    f" Initial pose: Pos={self.initial_pos}, Orn={self.initial_orn}"
                )

            Logger.info(f"Successfully initialized and cached object '{self.name}' ({self.prim_path})")
            return True

        except Exception as e:
            Logger.error(f"Failed to initialize object '{self.name}' ({self.prim_path}): {e}")
            return False

    def sync_pose(self):
        self.initial_pos, self.initial_orn = self.get_pose()

    def reset_pose(self):
        if not self.is_movable:
            Logger.warning(f"Cannot move object '{self.name}': Object is not movable")
            return

        if self.initial_pos is None or self.initial_orn is None:
            Logger.warning(f"Cannot reset pose for '{self.prim_path}': Initial pose not cached")
            return

        if not self.handle or not self.handle.is_valid():
            Logger.warning(f"Cannot reset pose: RigidPrim handle for object '{self.prim_path}' is invalid")
            return

        try:
            self.handle.set_world_pose(position=self.initial_pos, orientation=self.initial_orn)
            self.handle.set_linear_velocity(np.zeros(3))
            self.handle.set_angular_velocity(np.zeros(3))
            Logger.info(f"Successfully reset pose and velocity for '{self.prim_path}'")

        except Exception as e:
            Logger.error(f"Error occurred while resetting pose for object '{self.prim_path}': {e}")

    def move_by_delta(self, axis_index: int, delta: float):
        if self.initial_pos is None:
            Logger.warning(f"Cannot move object '{self.name}': Initial pose not cached")
            return
        if not self.is_movable:
            Logger.warning(f"Cannot move object '{self.name}': Object is not movable")
            return

        self.initial_pos[axis_index] += delta
        self.reset_pose()

    def get_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.handle or not self.handle.is_valid():
            return None

        try:
            return self.handle.get_world_pose()
        except Exception as e:
            Logger.warning(f"Failed to get pose for object '{self.name}': {e}")
            return None

    def get_velocity(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.handle or not self.handle.is_valid():
            return None

        try:
            lin_vel = self.handle.get_linear_velocity()
            ang_vel = self.handle.get_angular_velocity()
            return lin_vel, ang_vel
        except Exception as e:
            Logger.warning(f"Failed to get velocity for object '{self.name}': {e}")
            return None


class ObjectGroup:
    def __init__(self, group_key: str, objects: List[TrackedObject]):
        self.group_key = group_key
        self.objects = objects

    def initialize_all(
        self, stage, world, rigid_prim_cache: Dict[str, Tuple[RigidPrim, np.ndarray, np.ndarray]]
    ) -> bool:
        success = True
        for obj in self.objects:
            if not obj.initialize(stage, world, rigid_prim_cache):
                success = False
        return success

    def reset_all_poses(self):
        """Reset the pose of all objects in the group"""
        for obj in self.objects:
            obj.reset_pose()

    def move_all(self, axis_index: int, delta: float, sampled_active_object):
        movable_count = 0
        for obj in self.objects:
            if obj.is_movable and  obj.prim_path in sampled_active_object:
                obj.move_by_delta(axis_index, delta)
                movable_count += 1

        if movable_count == 0:
            Logger.warning(
                f"No movable objects in the object group '{self.group_key}' "
                f"(none of the objects are in sampled_active_object)"
            )

    def get_primary_object(self) -> Optional[TrackedObject]:
        for obj in self.objects:
            if obj.is_primary_for_recorder:
                return obj
        return self.objects[0] if self.objects else None

    def get_all_objects(self) -> List[TrackedObject]:
        """Get all objects in the group"""
        return self.objects


class ObjectManager:
    def __init__(
        self,
        data_collection_anchor,
        recorder_tracked_objects,
        sampled_active_object,
        move_step_length: float = 0.015,
    ):
        self.groups: Dict[str, ObjectGroup] = {}
        self.active_group_key: Optional[str] = None
        self.move_step_length = move_step_length

        self._rigid_prim_cache: Dict[str, Tuple[RigidPrim, np.ndarray, np.ndarray]] = {}

        self.data_collection_anchor: Dict[str, Dict[str, np.ndarray]] = {}
        if data_collection_anchor is not None:
            for prim_path, pose_data in data_collection_anchor.items():
                self.data_collection_anchor[prim_path] = {
                    "position": np.array(pose_data["position"]),
                    "orientation": np.array(pose_data["orientation"]),
                }
        self.current_tracked_objects: Dict[str, TrackedObject] = {}
        self.recorder_tracked_objects = recorder_tracked_objects
        self.sampled_active_object = sampled_active_object

        Logger.info(
            f"ObjectManager initialized with move step length: {move_step_length}m, "
        )

    def add_group(self, group: ObjectGroup):
        self.groups[group.group_key] = group
        Logger.info(f"Added object group: '{group.group_key}' with {len(group.objects)} objects")

        if self.active_group_key is None:
            self.active_group_key = group.group_key
            Logger.info(f"Set '{group.group_key}' as default active group")

    def clear_rigid_prim_cache(self):
        self._rigid_prim_cache.clear()
        Logger.info("RigidPrim cache cleared")

    def initialize_all_groups(self, stage, world) -> bool:

        Logger.info(f"Initializing {len(self.groups)} object groups...")
        success = True

        for group in self.groups.values():
            if not group.initialize_all(stage, world, self._rigid_prim_cache):
                success = False
                Logger.warning(f"Failed to fully initialize group: {group.group_key}")

        if success:
            Logger.info("All object groups initialized successfully")
        else:
            Logger.warning("Some objects failed to initialize")

        if self.active_group_key:
            self.set_active_group(self.active_group_key)

        return success

    def set_active_group(self, group_key: str) -> bool:
        if group_key not in self.groups:
            Logger.error(f"Attempted to activate a non-existent target object group: '{group_key}'")
            return False

        self.active_group_key = group_key

        self.current_tracked_objects.clear()

        if group_key in self.recorder_tracked_objects:
            for record_key, prim_path in self.recorder_tracked_objects[group_key].items():
                for object in self.groups[group_key].get_all_objects():
                    if object.prim_path == prim_path:
                        self.current_tracked_objects[record_key] = object

        primary_obj = self.groups[group_key].get_primary_object()
        if primary_obj and primary_obj.handle and primary_obj.handle.is_valid():
            Logger.info(
                f"[Dynamic Mode] Switched active target object group to: '{group_key}'"
            )
            self._move_inactive_cylinders_outside()
            self._move_active_to_anchor()
            return True
        else:
            Logger.warning(f"No valid primary object found for data recorder in object group '{group_key}'")
            return False

    def set_active_group_by_index(self, index: int) -> bool:
        group_keys = list(self.groups.keys())
        if 0 <= index < len(group_keys):
            return self.set_active_group(group_keys[index])
        else:
            Logger.warning(f"Invalid group index: {index} (total groups: {len(group_keys)})")
            return False

    def get_active_group(self) -> Optional[ObjectGroup]:
        if self.active_group_key:
            return self.groups.get(self.active_group_key)
        return None

    def get_active_group_name(self) -> Optional[str]:
        return self.active_group_key

    def get_group_names(self) -> List[str]:
        return list(self.groups.keys())

    def reset_active_group_poses(self):
        active_group = self.get_active_group()
        if active_group:
            Logger.info(f"Resetting poses for object group '{self.active_group_key}'")
            active_group.reset_all_poses()

        else:
            Logger.warning("No active object group to reset")

    def _move_active_to_anchor(self):
        active_group = self.get_active_group()
        if active_group:
            for obj in active_group.get_all_objects():
                if obj.name in ["cylinder", "plate", "basket", "target"]:
                    if obj.prim_path in self.data_collection_anchor:
                        anchor_data = self.data_collection_anchor[obj.prim_path]
                        self._move_prim_safely(
                            obj,
                            anchor_data["position"],
                            anchor_data["orientation"],
                        )
                        obj.sync_pose()
                        Logger.info(
                            f"Moved the cylinder of active group '{self.active_group_key}' "
                            f"({obj.prim_path}) to its data collection anchor position"
                        )
                    else:
                        Logger.warning(
                            f"No data collection anchor found for cylinder '{obj.prim_path}' "
                            f"in group '{self.active_group_key}'"
                        )

    def _move_inactive_cylinders_outside(self):
        """Move the cylinders of all groups except the active group outside the scene"""
        outside_position = np.array([999.0, 999.0, 999.0])
        outside_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        for group_key, group in self.groups.items():
            # Skip the currently active group
            if group_key == self.active_group_key:
                continue

            # Find the cylinder object in the group
            for obj in group.get_all_objects():
                if obj.name in ["cylinder", "plate", "basket", "target"]:
                    self._move_prim_safely(obj, outside_position, outside_orientation)
                    Logger.info(f"Moved the cylinder or plate of inactive group '{obj.prim_path}' outside the scene")

    def _move_prim_safely(self, tracked_obj: TrackedObject, position: np.ndarray, orientation: np.ndarray):
        if not tracked_obj.is_movable:
            Logger.warning(f"Cannot move object '{tracked_obj.name}': Object is not movable")
            return
        
        if not tracked_obj.handle or not tracked_obj.handle.is_valid():
            Logger.warning(f"Cannot move object '{tracked_obj.prim_path}': RigidPrim handle is invalid")
            return

        try:
            tracked_obj.handle.set_world_pose(position=position, orientation=orientation)
            tracked_obj.handle.set_linear_velocity(np.zeros(3))
            tracked_obj.handle.set_angular_velocity(np.zeros(3))
            Logger.debug(f"Successfully moved object '{tracked_obj.prim_path}' to position {position}")
        except Exception as e:
            Logger.error(f"Error occurred while moving object '{tracked_obj.prim_path}': {e}")

    def move_active_group(self, direction: str, step_length: Optional[float] = None):
        """
        Move all objects in the active group

        Args:
            direction: Move direction ("up", "down", "left", "right")
            step_length: Move step length (meters), use default value if None
        """
        active_group = self.get_active_group()
        if not active_group:
            Logger.warning("No active object group to move")
            return

        effective_step = step_length if step_length is not None else self.move_step_length

        # Determine the axis and direction of movement
        axis_map = {
            "up": (0, 1),  # Positive X direction
            "down": (0, -1),  # Negative X direction
            "left": (1, 1),  # Positive Y direction
            "right": (1, -1),  # Negative Y direction
        }

        if direction not in axis_map:
            Logger.warning(f"Invalid move direction: {direction}")
            return

        axis_index, direction_sign = axis_map[direction]
        delta = effective_step * direction_sign

        Logger.info(
            f"Moving object group '{self.active_group_key}' " f"{'X' if axis_index == 0 else 'Y'} axis {delta:+.3f}m"
        )

        active_group.move_all(axis_index, delta, self.sampled_active_object)

    def get_tracked_objects_for_recorder(self) -> Dict[str, str]:
        """
        Get the object dictionary for the data recorder

        Returns:
            Dict[Object name, Prim path]
        """
        tracked_objects = {}

        for group in self.groups.values():
            for obj in group.get_all_objects():
                # Generate a unique name for each object
                obj_name = f"{group.group_key}_{obj.name}"
                tracked_objects[obj_name] = obj.prim_path

        return tracked_objects

    def get_active_group_objects_for_recorder(self) -> Dict[str, str]:
        """
        Get the objects of the current active group (for data recorder)

        Returns:
            Dict[Object name, Prim path]
        """
        active_group = self.get_active_group()
        if not active_group:
            return {}

        tracked_objects = {}
        for obj in active_group.get_all_objects():
            # Generate a name for the objects in the current active group
            obj_name = f"{active_group.group_key}_{obj.name}"
            tracked_objects[obj_name] = obj.prim_path

        return tracked_objects

    def get_all_rigid_prims(self) -> Dict[str, RigidPrim]:
        """
        Get all initialized RigidPrim handles

        Returns:
            Dict[Object name, RigidPrim]
        """
        rigid_prims = {}

        for group in self.groups.values():
            for obj in group.get_all_objects():
                if obj.handle and obj.handle.is_valid():
                    obj_name = f"{group.group_key}_{obj.name}"
                    rigid_prims[obj_name] = obj.handle

        return rigid_prims


def create_object_manager_from_config(task_config: Dict, move_step_length: float = 0.015) -> ObjectManager:
    scene_config = task_config.get("scene", [{}])[0] if "scene" in task_config else {}
    teleoper_cfg = scene_config.get("teleoper_cfg", None)
    assert teleoper_cfg is not None, "teleoper_cfg config missing"

    manager = ObjectManager(
        data_collection_anchor=(
            teleoper_cfg["data_collection_anchor"] if "data_collection_anchor" in teleoper_cfg else None
        ),
        recorder_tracked_objects=teleoper_cfg.get("recorder_tracked_objects_config", None),
        sampled_active_object=scene_config.get("sampled_active_object", None),
        move_step_length=move_step_length,
    )

    if "scene" not in task_config or not task_config["scene"]:
        Logger.warning("No scene information in task configuration")
        return manager

    scene_config = task_config["scene"][0]
    Logger.debug(f"Scene configuration: {scene_config.keys()}")

    if "task_related_objects" not in scene_config:
        Logger.warning("No task_related_objects in scene configuration")
        return manager

    task_objects = scene_config["task_related_objects"]
    Logger.debug(f"Found {len(task_objects)} task-related objects")

    movable_objects = teleoper_cfg.get("movable_object_prims", [])
    Logger.debug(f"Movable objects list (movable_object_prims): {movable_objects}")

    for obj_info in task_objects:
        group_name = obj_info.get("name", "unknown")
        Logger.debug(f"Processing object group '{group_name}': {obj_info.keys()}")
        objects_in_group = []

        primary_added = False
        if "cylinder_prim_path" in obj_info and obj_info["cylinder_prim_path"]:
            cylinder_path = obj_info["cylinder_prim_path"]
            objects_in_group.append(
                TrackedObject(
                    name="cylinder",
                    prim_path=cylinder_path,
                    is_primary_for_recorder=True,
                    is_movable=(cylinder_path in movable_objects),
                )
            )
            primary_added = True

        # Priority 2: plate
        if "plate_prim_path" in obj_info and obj_info["plate_prim_path"]:
            plate_path = obj_info["plate_prim_path"]
            objects_in_group.append(
                TrackedObject(
                    name="plate",
                    prim_path=plate_path,
                    is_primary_for_recorder=not primary_added,
                    is_movable=(plate_path in movable_objects),
                )
            )
            if not primary_added:
                primary_added = True

        # Priority 3: basket
        if "basket_prim_path" in obj_info and obj_info["basket_prim_path"]:
            basket_path = obj_info["basket_prim_path"]
            objects_in_group.append(
                TrackedObject(
                    name="basket",
                    prim_path=basket_path,
                    is_primary_for_recorder=not primary_added,
                    is_movable=(basket_path in movable_objects),
                )
            )
            if not primary_added:
                primary_added = True

        if "target_prim_path" in obj_info and obj_info["target_prim_path"]:
            target_path = obj_info["target_prim_path"]
            objects_in_group.append(
                TrackedObject(
                    name="target",
                    prim_path=target_path,
                    is_primary_for_recorder=False,
                    is_movable=(target_path in movable_objects),
                )
            )

        if objects_in_group:
            group = ObjectGroup(group_key=group_name, objects=objects_in_group)
            manager.add_group(group)

            Logger.info(
                f"Created object group '{group_name}' containing {len(objects_in_group)} objects, "
            )
        else:
            Logger.warning(f"Object group '{group_name}' has no valid prim paths, skipping")

    return manager
