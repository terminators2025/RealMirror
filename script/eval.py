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

import argparse

parser = argparse.ArgumentParser(description="Run evaluation for trained models in Isaac Sim")
parser.add_argument("--task", type=str, default="Task1_Kitchen_Cleanup", help="Task configuration name")
parser.add_argument(
    "--model-type",
    type=str,
    default="act",
    choices=["act", "diffusion", "smolvla"],
    help="Type of model to use",
)
parser.add_argument("--model-path", type=str, help="Path to pretrained model")
parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI")
parser.add_argument(
    "--arc2gear",
    action="store_true",
    help="Use gear position conversion for hand joints",
)
parser.add_argument("--num-rollouts", type=int, default=400, help="Number of evaluation rollouts")
parser.add_argument("--max-horizon", type=int, default=400, help="Maximum steps per rollout")
parser.add_argument(
    "--use-stability-check",
    action="store_true",
    help="Enable stability check for success condition",
)
parser.add_argument("--resume-dir", type=str, default=None, help="Path to resume evaluation from")
parser.add_argument("--output-dir", type=str, default="runs/eval", help="Path to save evaluation output")
parser.add_argument(
    "--area-file",
    type=str,
    default="data/eval/data_area/data_area_task1.txt",
    help="Path to area definition file",
)
parser.add_argument("--xy-threshold", type=float, default=0.07, help="XY distance threshold for success")
parser.add_argument("--z-threshold", type=float, default=0.2, help="Z distance threshold for success")
parser.add_argument(
    "--stability-frames",
    type=int,
    default=30,
    help="Number of frames for stability check",
)

parser.add_argument(
    "--num-actions",
    type=int,
    default=5,
    help="Number of actions to use from model output chunk",
)
parser.add_argument(
    "--velocity-threshold",
    type=float,
    default=0.4,
    help="Velocity threshold for stability check in object_is_stably_stacked",
)
parser.add_argument(
    "--angular-velocity-threshold",
    type=float,
    default=2.0,
    help="Angular velocity threshold for stability check in object_is_stably_stacked",
)
parser.add_argument(
    "--resolution",
    type=str,
    default="1920x1080",
    help="UI resolution (e.g., 1920x1080, 1280x720)",
)

args = parser.parse_args()

from typing import Optional
from isaacsim.simulation_app import SimulationApp

width, height = map(int, args.resolution.split("x"))

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "physics_prim_path": "/physicsScene",
        "width": width,
        "height": height,
        "window_width": width,
        "window_height": height,
    }
)

from comm_config.eval_config import EvalConfig
import json
import os
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict
import pandas as pd
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.prims import RigidPrim

from comm_config.robot_config import RobotConfig
from comm_config.simulation_config import SimulationConfig
from comm_config.camera_config import CameraConfig
from robots.robot_factory import RobotFactory
from simulation.scene_manager import SceneManager
from inference import UnitConverter, ModelLoader, InferenceEngine, InferenceConfig
from utility.logger import Logger
from utility.camera_utils import get_camera_image
from utility.plotting_utils import (
    load_area_points_from_txt,
    generate_uniform_grid_points,
    generate_results_plot,
)

from evaluate.inference_evaluator import (
    object_reached_target,
    object_is_stably_stacked,
    log_object_reached_target_details,
    save_frames_to_video,
    move_prim_safely,
    load_evaluation_progress,
    save_evaluation_progress,
)
import utility.system_utils as system_utils


class EvaluationRunner:

    def __init__(self, args):
        self.args = args
        self.world = None
        self.robot = None
        self.robot_view = None
        self.scene_manager = None
        self.cameras = {}
        self.inference_engine = None
        self.unit_converter = None

        self.evaluation_points = []
        self.evaluation_results = []
        self.all_results_data = []
        self.output_dir = Path()
        self.frame_buffer = []
        self.is_recording = False
        self.stability_counter = 0

        self.task_configurations = []
        self.current_task_config = {}
        self.all_task_prims = {}
        self.pristine_initial_poses = {}
        self.eval_config: Optional[EvalConfig] = None
        self.fixed_objects_info = None

        self._initialize()

    def _initialize(self):
        self._load_configurations()

        self.world = World(
            stage_units_in_meters=self.sim_config.stage_units_in_meters,
            physics_dt=self.sim_config.physics_dt,
            rendering_dt=self.sim_config.rendering_dt,
        )
        self.world.scene.add_default_ground_plane()

        self.scene_manager = SceneManager(self.world, self.sim_config)
        self.scene_manager._load_scene_usd()

        self.cameras = CameraConfig.create_cameras_from_task_config(self.task_config, self.robot_config)

        self._setup_task_configurations()
        self._initialize_task_objects()
        self.robot = RobotFactory.create_robot(self.robot_type, self.robot_config)
        self.robot.initialize(self.world, simulation_app=simulation_app, mode="eval")

        self._setup_joint_indices()
        self.scene_manager._adjust_ground_plane()

        for prim in self.all_task_prims.values():
            prim.initialize()
        self.robot_view = self.robot.robot_view
        self._setup_inference()
        Logger.info("EvaluationRunner initialized successfully")

    def _load_configurations(self):
        project_root = system_utils.teleop_root_path()
        task_config_path = os.path.join(project_root, "tasks", f"{self.args.task}.json")

        if not os.path.exists(task_config_path):
            raise FileNotFoundError(f"Task config not found: {task_config_path}")

        with open(task_config_path) as f:
            self.task_config = json.load(f)

        self.robot_type = self.task_config["robot"]["robot_type"]
        robot_config_path = os.path.join(
            project_root,
            "comm_config",
            "configs",
            self.task_config["robot"]["robot_cfg"],
        )
        self.robot_config = RobotConfig.from_json(robot_config_path)
        self._try_load_scene_physics_params()
        self._load_fixed_objects_info()

        sim_config_path = os.path.join(project_root, "comm_config", "configs", "simulation_config.json")
        self.sim_config = SimulationConfig.from_json(sim_config_path, self.task_config)

        if "scene" in self.task_config and self.task_config["scene"] and "eval_cfg" in self.task_config["scene"][0]:
            eval_cfg = self.task_config["scene"][0]["eval_cfg"]
            model_config = eval_cfg["model"]
            model_root_dir = model_config["model_root_dir"]
            Logger.info(f"Model root dir: {model_root_dir}")
            models_path = {}
            for model_name, model_path in model_config["checkpoints_dir"].items():
                models_path[model_name] = os.path.join(model_root_dir, self.args.task, model_path)
                Logger.info(f"{model_name} path: {models_path[model_name]}")

            xy_threshold = eval_cfg["success_criteria"].get("xy_threshold", self.args.xy_threshold)
            z_threshold = eval_cfg["success_criteria"].get("z_threshold", self.args.z_threshold)
            Logger.info(f"Using xy_threshold: {xy_threshold}, z_threshold: {z_threshold}")

            self.eval_config = EvalConfig(
                models_path=models_path,
                xy_threshold=xy_threshold,
                z_threshold=z_threshold,
            )
        Logger.info(f"Loaded configurations for task: {self.args.task}")

    def _try_load_scene_physics_params(self):
        if "physics_params" in self.task_config["robot"]:
            physics = self.task_config["robot"]["physics_params"]
            self.robot_config.default_kp = physics.get("default_kp", 400.0)
            self.robot_config.default_kd = physics.get("default_kd", 40.0)
            self.robot_config.finger_kp = physics.get("finger_kp", 1.0e10)
            self.robot_config.finger_kd = physics.get("finger_kd", 6.0e7)
            self.robot_config.finger_max_force = physics.get("finger_max_force", 1.0e10)

    def _load_fixed_objects_info(self):
        if "fix_objects_info" in self.task_config["scene"][0]:
            self.fixed_objects_info = self.task_config["scene"][0]["fix_objects_info"]

    def _setup_joint_indices(self):
        all_dof_names = self.robot.get_dof_names()
        if not all_dof_names:
            raise RuntimeError("Failed to get robot DOF names")

        dof_map = {name: i for i, name in enumerate(all_dof_names)}

        state_joint_names = UnitConverter.STATE_SIM_JOINT_NAMES_26DOF
        try:
            state_indices = [dof_map[name] for name in state_joint_names]
            self.state_joint_indices = np.array(state_indices, dtype=np.int32)
            Logger.info(f"Mapped {len(self.state_joint_indices)} state joint indices")
        except KeyError as e:
            Logger.error(f"Failed to map state joint: {e}")
            raise

        left_arm_joints = self.robot_config.arm_joints.get("left", [])
        right_arm_joints = self.robot_config.arm_joints.get("right", [])

        extra_left = [
            "idx17_left_arm_joint5",
            "idx18_left_arm_joint6",
            "idx19_left_arm_joint7",
        ]
        for joint in extra_left:
            if joint not in left_arm_joints:
                left_arm_joints.append(joint)

        extra_right = [
            "idx24_right_arm_joint5",
            "idx25_right_arm_joint6",
            "idx26_right_arm_joint7",
        ]
        for joint in extra_right:
            if joint not in right_arm_joints:
                right_arm_joints.append(joint)

        left_hand_mimic = [
            "L_thumb_swing_joint",
            "L_thumb_1_joint",
            "L_thumb_2_joint",
            "L_thumb_3_joint",
            "L_index_1_joint",
            "L_index_2_joint",
            "L_middle_1_joint",
            "L_middle_2_joint",
            "L_ring_1_joint",
            "L_ring_2_joint",
            "L_little_1_joint",
            "L_little_2_joint",
        ]
        right_hand_mimic = [
            "R_thumb_swing_joint",
            "R_thumb_1_joint",
            "R_thumb_2_joint",
            "R_thumb_3_joint",
            "R_index_1_joint",
            "R_index_2_joint",
            "R_middle_1_joint",
            "R_middle_2_joint",
            "R_ring_1_joint",
            "R_ring_2_joint",
            "R_little_1_joint",
            "R_little_2_joint",
        ]

        action_joint_names = left_arm_joints + right_arm_joints + left_hand_mimic + right_hand_mimic

        try:
            action_indices = [dof_map[name] for name in action_joint_names]
            self.action_joint_indices = np.array(action_indices, dtype=np.int32)
            Logger.info(f"Mapped {len(self.action_joint_indices)} action joint indices")
        except KeyError as e:
            Logger.error(f"Failed to map action joint: {e}")
            raise

    def _setup_inference(self):
        model_path = self.args.model_path
        if self.eval_config and self.eval_config.models_path:
            models_path_lower = {k.lower(): v for k, v in self.eval_config.models_path.items()}
            model_path = models_path_lower.get(self.args.model_type.lower(), self.args.model_path)
        Logger.info(f"Model[{self.args.model_type.lower()}] path: {model_path}")
        inference_config = InferenceConfig(
            model_type=self.args.model_type,
            model_path=model_path,
            use_gear_conversion=self.args.arc2gear,
            num_actions_in_chunk=self.args.num_actions,
            task_name=None,
        )

        self.unit_converter = UnitConverter(
            arc_to_gear=inference_config.use_gear_conversion,
            max_gear=inference_config.max_gear,
        )

        model_loader = ModelLoader(
            model_type=inference_config.model_type,
            model_path=inference_config.model_path,
        )

        if not model_loader.load_model():
            raise RuntimeError("Failed to load inference model")

        self.inference_engine = InferenceEngine(
            model_loader=model_loader,
            unit_converter=self.unit_converter,
            num_actions_in_chunk=inference_config.num_actions_in_chunk,
            image_resolution=inference_config.image_resolution,
        )

        Logger.info("Inference components initialized")

    def _setup_task_configurations(self):
        if not self.scene_manager:
            Logger.error("SceneManager not initialized")
            raise RuntimeError(f"SceneManager not initialized")

        if self.sim_config and self.sim_config.task_related_objects:
            for obj_info in self.sim_config.task_related_objects:
                task_config = {
                    "name": obj_info.name,
                    "task_name": (
                        obj_info.description if hasattr(obj_info, "description") else f"Task for {obj_info.name}"
                    ),
                    "cylinder_prim_path": obj_info.cylinder_prim_path,
                    "target_prim_path": obj_info.target_prim_path,
                }

                if hasattr(obj_info, "plate_prim_path"):
                    task_config["plate_prim_path"] = obj_info.plate_prim_path
                if hasattr(obj_info, "basket_prim_path"):
                    task_config["basket_prim_path"] = obj_info.basket_prim_path

                self.task_configurations.append(task_config)
        else:
            raise RuntimeError("No task_related_objects found in sim_config")
        Logger.info(f"Loaded {len(self.task_configurations)} task configurations.")

    def _initialize_task_objects(self):
        stage = omni.usd.get_context().get_stage()
        for config in self.task_configurations:
            for prim_path_key in [
                "cylinder_prim_path",
                "plate_prim_path",
                "basket_prim_path",
                "target_prim_path",
            ]:
                prim_path = config.get(prim_path_key)
                if prim_path and prim_path not in self.all_task_prims:
                    Logger.info(f"[1]Checking additional object path: {prim_path_key} -> {prim_path}")
                    if stage.GetPrimAtPath(prim_path):
                        Logger.info(f"[2]Checking additional object path: {prim_path_key} -> {prim_path}")
                        prim_name = prim_path.replace("/", "_")
                        prim = self.world.scene.add(RigidPrim(prim_path=prim_path, name=prim_name))
                        self.all_task_prims[prim_path] = prim

                        if prim and prim.is_valid():
                            pos, orn = prim.get_world_pose()
                            self.pristine_initial_poses[prim_path] = (
                                np.copy(pos),
                                np.copy(orn),
                            )
                            Logger.info(f"initialized {prim_path_key}: {prim_path} on [{pos[0]}, {pos[1]}]")

    def reset_environment(self):

        self.inference_engine.reset()
        self.stability_counter = 0

        if self.robot and self.robot.robot_ref:
            initial_positions = self.task_config["robot"].get("arm_joint_angles_rad", {})
            hand_joints_to_configure = self.task_config["robot"].get("hand_joints_to_configure", [])
            current_positions = self.robot.get_joint_positions()

            if current_positions is not None:
                for joint_name, angle in initial_positions.items():
                    idx = self.robot.get_dof_index(joint_name)
                    if idx != -1:
                        current_positions[idx] = angle
                hand_joints = ["L_thumb_swing_joint", "R_thumb_swing_joint"]
                for joint_name in hand_joints_to_configure:
                    idx = self.robot.get_dof_index(joint_name)
                    if idx != -1:
                        if joint_name in hand_joints:
                            current_positions[idx] = np.deg2rad(90.0)
                        else:
                            current_positions[idx] = 0.0

                self.robot.set_joint_positions(current_positions)
                Logger.info(f"Robot reset to initial joint positions. {current_positions}")

        for path, prim in self.all_task_prims.items():
            initial_pose = self.pristine_initial_poses.get(path)
            if initial_pose:
                Logger.info(f"  [Reset] Moving {path} to initial position")
                move_prim_safely(prim, initial_pose[0], initial_pose[1])

    def setup_rollout_environment(self, rollout_idx: int):

        self.reset_environment()

        task_idx = rollout_idx % len(self.task_configurations)
        self.current_task_config = self.task_configurations[task_idx]

        Logger.info(f"[Task] Setting up task: {json.dumps(self.current_task_config, indent=4)}")

        if "task_name" in self.current_task_config:
            self.inference_engine.set_task_name(self.current_task_config["task_name"])

        for path, prim in self.all_task_prims.items():
            Logger.info(f"Prim {path} moved safely Area.")
            move_prim_safely(prim, np.array([999.0, 999.0, 999.0]), np.array([1.0, 0.0, 0.0, 0.0]))

        if rollout_idx < len(self.evaluation_points):
            point = self.evaluation_points[rollout_idx]

            prime_key_path = [
                "cylinder_prim_path",
                "target_prim_path",
                "plate_prim_path",
                "basket_prim_path",
            ]
            current_active_objs_prime = [
                self.current_task_config.get(x) for x in prime_key_path if self.current_task_config.get(x)
            ]
            Logger.info(f"current_active_objs_prime: {current_active_objs_prime}")
            for current_active_obj_prime in current_active_objs_prime:
                if current_active_obj_prime and current_active_obj_prime in self.all_task_prims:
                    active_prim = self.all_task_prims[current_active_obj_prime]
                    pristine_pose = self.pristine_initial_poses.get(current_active_obj_prime)
                    if self.fixed_objects_info:
                        for fix_obj in self.fixed_objects_info:
                            if fix_obj.get("prim_path") == current_active_obj_prime:
                                move_prim_safely(
                                    active_prim,
                                    np.array(fix_obj.get("position")),
                                    np.array(fix_obj.get("orientation")),
                                )
                                Logger.info(f"- Placed object[{current_active_obj_prime}] at fixed position")
                    elif pristine_pose:
                        move_prim_safely(active_prim, pristine_pose[0], pristine_pose[1])
                        Logger.info(f"- Placed object[{current_active_obj_prime}] at initial position")

            for active_object_prim_path in self.sim_config.sampled_active_object:
                if active_object_prim_path in current_active_objs_prime:
                    active_prim = self.all_task_prims[active_object_prim_path]
                    pristine_pose = self.pristine_initial_poses.get(active_object_prim_path)
                    if pristine_pose:
                        new_pos = np.copy(pristine_pose[0])
                        new_pos[0] = point[0]
                        new_pos[1] = point[1]
                        move_prim_safely(active_prim, new_pos, pristine_pose[1])
                        Logger.info(
                            f"- Placed active object[{active_object_prim_path}] at [{point[0]:.2f}, {point[1]:.2f}]"
                        )

    def run_inference_step(self):

        state = self.inference_engine.get_observation_state(self.robot_view, self.state_joint_indices)
        if state is None:
            return

        head_image = get_camera_image(self.cameras.get("head_camera"), "rgb")
        left_wrist_image = get_camera_image(self.cameras.get("left_wrist_camera"), "rgb")
        right_wrist_image = get_camera_image(self.cameras.get("right_wrist_camera"), "rgb")

        if head_image is None or left_wrist_image is None or right_wrist_image is None:
            return

        if self.is_recording and head_image is not None:
            self.frame_buffer.append(head_image)

        actions = self.inference_engine.run_inference(
            state, left_wrist_image, right_wrist_image, head_image, save_images=False
        )

        if actions is None:
            return

        for action_step in actions:
            action = ArticulationAction(
                joint_positions=action_step.astype(np.float32),
                joint_indices=self.action_joint_indices,
            )
            self.robot.robot_ref.get_articulation_controller().apply_action(action)

    def check_task_success(self, cur_task) -> bool:
        cylinder_path = cur_task.get("cylinder_prim_path")
        target_path = cur_task.get("target_prim_path")

        if not cylinder_path or not target_path:
            return False

        if self.args.use_stability_check:

            moving_prim = self.all_task_prims.get(cylinder_path)
            target_prim = self.all_task_prims.get(target_path)

            if not moving_prim or not target_prim:
                return False

            is_stable = object_is_stably_stacked(
                moving_prim,
                target_prim,
                self.eval_config.xy_threshold,
                self.eval_config.z_threshold,
                velocity_threshold=self.args.velocity_threshold,
                angular_velocity_threshold=self.args.angular_velocity_threshold,
            )

            if is_stable:
                self.stability_counter += 1
                if self.stability_counter % 10 == 0:
                    Logger.debug(
                        f"  [Stability Check] Stable frame detected Count: {self.stability_counter}/{self.args.stability_frames}"
                    )
            else:
                if self.stability_counter > 0:
                    Logger.debug("  [Stability Check] Unstable frame detected Reset counter")
                self.stability_counter = 0

            return self.stability_counter >= self.args.stability_frames
        else:

            return object_reached_target(
                cylinder_path,
                target_path,
                self.eval_config.xy_threshold,
                self.eval_config.z_threshold,
            )

    def run_single_rollout(self, rollout_idx: int) -> Dict:
        rollout_start_time = time.time()
        Logger.info(f"--- [Rollout {rollout_idx + 1}/{len(self.evaluation_points)}] ---")

        self.setup_rollout_environment(rollout_idx)

        for _ in range(30):
            self.world.step(render=True)

        self.frame_buffer = []
        self.is_recording = True

        is_successful = False
        final_step = self.args.max_horizon

        for step in range(self.args.max_horizon):
            self.world.step(render=True)
            self.run_inference_step()

            if self.check_task_success(self.current_task_config):
                is_successful = True
                final_step = step + 1
                break

        self.is_recording = False

        result = {
            "rollout_idx": rollout_idx,
            "x": self.evaluation_points[rollout_idx][0],
            "y": self.evaluation_points[rollout_idx][1],
            "success": 1 if is_successful else 0,
            "rollout_path": "",
            "task_name": self.current_task_config.get("task_name", ""),
            "cylinder_prim_path": self.current_task_config.get("cylinder_prim_path", ""),
            "target_prim_path": self.current_task_config.get("target_prim_path", ""),
        }

        Logger.info("  [Final State Check]")
        cylinder_path = self.current_task_config.get("cylinder_prim_path")
        target_path = self.current_task_config.get("target_prim_path")
        if cylinder_path and target_path:
            log_object_reached_target_details(
                name=self.current_task_config.get("name", ""),
                moving_prim_path=cylinder_path,
                target_prim_path=target_path,
                xy_threshold=self.eval_config.xy_threshold,
                z_threshold=self.eval_config.z_threshold,
            )

        if is_successful:
            Logger.info(f"--- Result: SUCCESS (at step {final_step}) ---\n")
            if self.output_dir:
                video_path = self.output_dir / f"successful_rollout_{rollout_idx + 1}.mp4"
                save_frames_to_video(self.frame_buffer, str(video_path))
                result["rollout_path"] = str(video_path)

        else:
            Logger.error(f"--- Result: FAILURE (reached max steps) ---")

            if self.output_dir:
                video_path = self.output_dir / f"failed_rollout_{rollout_idx + 1}.mp4"
                save_frames_to_video(self.frame_buffer, str(video_path))
                result["rollout_path"] = str(video_path)

        self.frame_buffer = []

        rollout_duration = time.time() - rollout_start_time
        Logger.info(
            f"Rollout {rollout_idx + 1} completed in {rollout_duration:.2f} seconds ({rollout_duration/60:.2f} minutes)"
        )

        return result

    def run_evaluation(self):
        if "Task3_Assembly_Line_Sorting" in self.args.task:
            self.do_evaluation_task3()
        else:
            self.do_evaluation()

    def do_evaluation(self):
        """Main evaluation function for standard tasks"""
        self._print_evaluation_banner()

        csv_path = self._setup_output_directory()
        start_rollout = self._load_or_initialize_evaluation_data(csv_path)

        self._prepare_world_for_evaluation()

        start_time = time.time()
        self._execute_evaluation_rollouts(start_rollout, csv_path)

        self._log_final_evaluation_results(start_time)
        self.generate_visualizations()

    def _print_evaluation_banner(self):
        """Print evaluation start banner"""
        banner = "=" * 50
        Logger.info(f"\n{banner}\nStarting evaluation: {self.args.num_rollouts} rollouts\n{banner}\n")

    def _setup_output_directory(self):
        """Setup and return output directory and CSV path"""
        if self.args.resume_dir:
            self.output_dir = Path(self.args.resume_dir)
            Logger.info(f"Resuming evaluation from: {self.output_dir}")
        else:
            output_base_dir = Path(self.args.output_dir) if self.args.output_dir else Path("runs/eval")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = output_base_dir / f"{self.args.task}" / f"{self.args.model_type}_{timestamp}"
            Logger.info(f"Starting new evaluation, output to: {self.output_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / "evaluation_data.csv"

    def _load_or_initialize_evaluation_data(self, csv_path):
        """Load existing evaluation data or initialize new data"""
        start_rollout = 0

        if csv_path.exists():
            start_rollout = self._load_existing_evaluation_data(csv_path)

        if start_rollout == 0 and not self.all_results_data:
            self._generate_new_evaluation_data()

        return start_rollout

    def _load_existing_evaluation_data(self, csv_path):
        """Load existing evaluation progress from CSV"""
        df_existing, start_rollout = load_evaluation_progress(csv_path)
        if not df_existing.empty:
            self.all_results_data = df_existing.to_dict("records")
            self.evaluation_points = df_existing[["x", "y"]].to_numpy()
            Logger.info(f"Loaded {len(df_existing)} existing results, starting from rollout {start_rollout + 1}")
        return start_rollout

    def _generate_new_evaluation_data(self):
        """Generate new evaluation points and initialize results data"""
        Logger.info("Generating new evaluation points...")

        base_points = load_area_points_from_txt(self.args.area_file)
        if base_points is None:
            Logger.error("Failed to load area definition file")
            return False

        self.evaluation_points, bbox = generate_uniform_grid_points(base_points, self.args.num_rollouts)
        if self.evaluation_points is None:
            Logger.error("Failed to generate evaluation points")
            return False

        for i, point in enumerate(self.evaluation_points):
            task_idx = i % len(self.task_configurations)
            config = self.task_configurations[task_idx]
            self.all_results_data.append(
                {
                    "rollout_idx": i,
                    "x": point[0],
                    "y": point[1],
                    "success": -1,
                    "rollout_path": "",
                    "task_name": config.get("task_name", ""),
                    "cylinder_prim_path": config.get("cylinder_prim_path", ""),
                    "target_prim_path": config.get("target_prim_path", ""),
                }
            )
        return True

    def _prepare_world_for_evaluation(self):
        """Prepare the simulation world for evaluation"""
        self.world.play()
        Logger.info("Stabilizing scene...")
        for _ in range(50):
            self.world.step(render=True)

    def _execute_evaluation_rollouts(self, start_rollout, csv_path):
        """Execute all evaluation rollouts"""
        success_count = sum(1 for r in self.all_results_data if r.get("success") == 1)

        for i in range(start_rollout, len(self.evaluation_points)):
            result = self.run_single_rollout(i)
            self.all_results_data[i] = result

            if result["success"] == 1:
                success_count += 1

            save_evaluation_progress(self.all_results_data, csv_path, self.args.model_type)

            current_rate = (success_count / (i + 1)) * 100
            Logger.info(f"Current success rate: {current_rate:.2f}% ({success_count}/{i + 1})")

    def _log_final_evaluation_results(self, start_time):
        """Log final evaluation results and statistics"""
        end_time = time.time()
        duration = end_time - start_time

        Logger.info("\n" + "=" * 50)
        Logger.info(f"{self.args.task} Evaluation Complete")
        Logger.info(f"Total time: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        Logger.info(
            f"Final success rate: {(sum(1 for r in self.all_results_data if r.get('success') == 1)/len(self.evaluation_points))*100:.2f}%"
        )
        Logger.info("=" * 50 + "\n")

    def do_evaluation_task3(self):
        """Main evaluation function for Task3 Assembly Line Sorting"""
        banner = "=" * 50
        Logger.info(f"\n{banner}\nStarting evaluation: {self.args.num_rollouts} rollouts\n{banner}\n")

        self.output_dir, csv_path = self._setup_task3_output_directory()
        start_rollout = self._load_or_initialize_task3_data(csv_path)

        self._prepare_task3_world()

        start_time = time.time()
        success_count = sum(1 for r in self.all_results_data if r.get("success") == 1)
        num_rollouts_to_run = len(self.all_results_data)

        for i in range(start_rollout, num_rollouts_to_run):
            self._run_single_task3_rollout(i, num_rollouts_to_run, csv_path)
            if i < len(self.all_results_data) and self.all_results_data[i]["success"] == 1:
                success_count += 1
            current_rate = (success_count / (i + 1)) * 100
            Logger.info(f"Current success rate: {current_rate:.2f}% ({success_count}/{i + 1})")

        self._log_task3_final_results(start_time)

    def _setup_task3_output_directory(self):
        """Setup output directory for Task3 evaluation"""
        if self.args.resume_dir:
            output_dir = Path(self.args.resume_dir)
            Logger.info(f"Resuming evaluation from: {output_dir}")
        else:
            output_base_dir = Path(self.args.output_dir) if self.args.output_dir else Path("runs/eval")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = output_base_dir / f"{self.args.task}" / f"{self.args.model_type}_{timestamp}"
            Logger.info(f"Starting new evaluation, output to: {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "evaluation_data.csv"

        return output_dir, csv_path

    def _load_or_initialize_task3_data(self, csv_path):
        """Load existing evaluation data or initialize new data for Task3"""
        start_rollout = 0
        bbox = None

        if csv_path.exists():
            df_existing, start_rollout = load_evaluation_progress(csv_path)
            if not df_existing.empty:
                self.all_results_data = df_existing.to_dict("records")
                self.evaluation_points = df_existing[["x", "y"]].to_numpy()
                Logger.info(f"Loaded {len(df_existing)} existing results, starting from rollout {start_rollout + 1}")

        if not self.all_results_data:
            for i in range(self.args.num_rollouts):
                record = {"rollout_idx": i, "success": -1, "rollout_path": ""}
                for config in self.task_configurations:
                    record[f'success_{config["name"]}'] = -1
                self.all_results_data.append(record)

        return start_rollout

    def _prepare_task3_world(self):
        """Prepare the world for Task3 evaluation"""
        self.world.play()
        Logger.info("Stabilizing scene...")
        for _ in range(50):
            self.world.step(render=True)

    def _run_single_task3_rollout(self, rollout_idx, num_rollouts_to_run, csv_path):
        """Run a single rollout for Task3 evaluation"""
        rollout_start_time = time.time()
        Logger.info(f"--- [Rollout {rollout_idx + 1}/{num_rollouts_to_run}] ---")

        self._setup_task3_rollout_environment()
        self._ensure_environment_stabilization()
        self.frame_buffer = []
        self.is_recording = True

        final_step = self._execute_task3_simulation_steps(rollout_idx)
        sub_success = self._evaluate_task3_results(rollout_idx)

        self._finalize_task3_rollout_result(rollout_idx, sub_success, final_step, csv_path)
        self.is_recording = False
        self.frame_buffer = []

        rollout_duration = time.time() - rollout_start_time
        Logger.info(
            f"Rollout {rollout_idx + 1} completed in {rollout_duration:.2f} seconds ({rollout_duration/60:.2f} minutes)"
        )

    def _setup_task3_rollout_environment(self):
        """Setup environment for a single Task3 rollout"""
        self.reset_environment()

        # Move all objects out of scene initially
        for object_cfg in self.task_configurations:
            if "cylinder_prim_path" in object_cfg:
                if object_cfg["cylinder_prim_path"] in self.all_task_prims.keys():
                    move_prim_safely(
                        self.all_task_prims[object_cfg["cylinder_prim_path"]],
                        np.array([999.0, 999.0, 999.0]),
                        np.array([1.0, 0.0, 0.0, 0.0]),
                    )
                    Logger.info(f' {object_cfg["cylinder_prim_path"]} Moved Out of the Scene')

            if "plate_prim_path" in object_cfg:
                if object_cfg["plate_prim_path"] in self.all_task_prims.keys():
                    move_prim_safely(
                        self.all_task_prims[object_cfg["plate_prim_path"]],
                        np.array([999.0, 999.0, 999.0]),
                        np.array([1.0, 0.0, 0.0, 0.0]),
                    )
                    Logger.info(f' {object_cfg["plate_prim_path"]} Moved Out of the Scene')

    def _ensure_environment_stabilization(self, steps=30):
        """Stabilize the environment after setup"""
        for _ in range(steps):
            self.world.step(render=True)

    def _execute_task3_simulation_steps(self, rollout_idx):
        """Execute simulation steps with object spawning logic"""
        object_spawn_order = {}
        for config in self.task_configurations:
            if "name" in config:
                object_spawn_order[config["name"]] = config

        spawn_names = list(object_spawn_order.keys())
        final_step = self.args.max_horizon

        for step in range(self.args.max_horizon):
            # Spawn objects at specific steps
            if step == 0 and len(spawn_names) > 0:
                Logger.info(f"Spawning object: {spawn_names[0]} at step {step}")
                self._spawn_task3_object(object_spawn_order[spawn_names[0]])
                self.inference_engine.set_task_name(object_spawn_order[spawn_names[0]].get("task_name", ""))
            elif step == 500 and len(spawn_names) > 1:
                Logger.info(f"Spawning object: {spawn_names[1]} at step {step}")
                self._spawn_task3_object(object_spawn_order[spawn_names[1]])
                self.inference_engine.set_task_name(object_spawn_order[spawn_names[1]].get("task_name", ""))
            elif step == 1000 and len(spawn_names) > 2:
                Logger.info(f"Spawning object: {spawn_names[2]} at step {step}")
                self._spawn_task3_object(object_spawn_order[spawn_names[2]])
                self.inference_engine.set_task_name(object_spawn_order[spawn_names[2]].get("task_name", ""))

            self.world.step(render=True)
            self.run_inference_step()

            # Check for completion after step 1000
            if step > 1000:
                if self._check_all_task3_subtasks_complete():
                    Logger.info("All sub-tasks completed successfully.")
                    final_step = step + 1
                    break

        return final_step

    def _spawn_task3_object(self, obj_config):
        """Spawn a single object for Task3"""
        cylinder_path = obj_config.get("cylinder_prim_path")
        plate_path = obj_config.get("plate_prim_path")

        if cylinder_path and cylinder_path in self.all_task_prims:
            pristine_pose = self.pristine_initial_poses.get(cylinder_path)
            if pristine_pose:
                move_prim_safely(
                    self.all_task_prims[cylinder_path],
                    pristine_pose[0],
                    pristine_pose[1],
                )

        if plate_path and plate_path in self.all_task_prims:
            pristine_pose = self.pristine_initial_poses.get(plate_path)
            if pristine_pose:
                move_prim_safely(self.all_task_prims[plate_path], pristine_pose[0], pristine_pose[1])

    def _check_all_task3_subtasks_complete(self):
        """Check if all Task3 subtasks are completed"""
        for task_object in self.task_configurations:
            if not self.check_task_success(task_object):
                return False
        return True

    def _evaluate_task3_results(self, rollout_idx):
        """Evaluate results for each subtask in Task3"""
        sub_success = {}
        for task_object in self.task_configurations:
            is_object_succeeded = self.check_task_success(task_object)
            sub_success[task_object["name"]] = is_object_succeeded
            log_object_reached_target_details(
                name=task_object.get("name", ""),
                moving_prim_path=task_object["cylinder_prim_path"],
                target_prim_path=task_object["target_prim_path"],
                xy_threshold=self.eval_config.xy_threshold,
                z_threshold=self.eval_config.z_threshold,
            )
            self.all_results_data[rollout_idx][f'success_{task_object["name"]}'] = 1 if is_object_succeeded else 0

        return sub_success

    def _finalize_task3_rollout_result(self, rollout_idx, sub_success, final_step, csv_path):
        """Finalize and save the result of a Task3 rollout"""
        final_success = all(sub_success.values())
        self.all_results_data[rollout_idx]["success"] = 1 if final_success else 0

        if final_success:
            Logger.info(f"--- Result: SUCCESS (at step {final_step}) ---\n")
            self.all_results_data[rollout_idx]["rollout_path"] = ""
        else:
            Logger.info(f"--- Result: FAILURE (reached max steps {self.args.max_horizon}) ---")
            failure_rollout_video_path = self.output_dir / f"failed_rollout_{rollout_idx + 1}.mp4"
            save_frames_to_video(self.frame_buffer, str(failure_rollout_video_path))
            self.all_results_data[rollout_idx]["rollout_path"] = str(failure_rollout_video_path)

        save_evaluation_progress(self.all_results_data, csv_path, self.args.model_type)

    def _log_task3_final_results(self, start_time):
        """Log final results and statistics for Task3 evaluation"""
        end_time = time.time()
        duration = end_time - start_time

        Logger.info("\n" + "=" * 50)
        Logger.info(f"{self.args.task} Evaluation Complete")
        Logger.info(f"Total time: {duration:.2f} seconds ({duration/60:.1f} minutes)")

        if self.args.num_rollouts > 0:
            final_df = pd.DataFrame(self.all_results_data)
            if not final_df.empty:
                completed_rollouts = len(final_df[final_df["success"] != -1])
                if completed_rollouts > 0:
                    total_success_rate = (final_df["success"].sum() / completed_rollouts) * 100
                    Logger.info(
                        f"Final success rate: {total_success_rate:.1f}% ({final_df['success'].sum()}/{completed_rollouts})"
                    )
                    Logger.info("\n--- Sub-task Success Rates ---")
                    for task_object in self.task_configurations:
                        task_name = task_object["name"]
                        sub_rate = (final_df[f"success_{task_name}"].sum() / completed_rollouts) * 100
                        Logger.info(
                            f"  - {task_name.capitalize()} Success Rate: {sub_rate:.1f}% ({final_df[f'success_{task_name}'].sum()}/{completed_rollouts})"
                        )
                    Logger.info("=" * 50 + "\n")

    def generate_visualizations(self):
        if not self.all_results_data:
            Logger.warning("No evaluation data to visualize")
            return

        df = pd.DataFrame(self.all_results_data)
        if df.empty:
            return

        points = df[["x", "y"]].to_numpy()
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        bbox = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

        Logger.info("Generating overall evaluation plot...")
        generate_results_plot(
            points=points,
            results=df["success"].astype(bool).to_numpy(),
            bbox=bbox,
            output_path=str(self.output_dir / "evaluation_results.svg"),
            model_type=self.args.model_type,
        )

        if "cylinder_prim_path" in df.columns:
            Logger.info("Generating per-task evaluation plots...")
            unique_tasks = df["cylinder_prim_path"].unique()

            for task_path in unique_tasks:
                if pd.isna(task_path):
                    continue

                try:
                    task_name = task_path.split("/")[2]
                except IndexError:
                    task_name = task_path.replace("/", "_")

                Logger.info(f"  - Processing task: {task_name}")
                task_df = df[df["cylinder_prim_path"] == task_path]

                if task_df.empty:
                    continue

                task_points = task_df[["x", "y"]].to_numpy()
                task_results = task_df["success"].astype(bool).to_numpy()

                sub_x_min, sub_y_min = task_points.min(axis=0)
                sub_x_max, sub_y_max = task_points.max(axis=0)
                sub_bbox = {
                    "x_min": sub_x_min,
                    "x_max": sub_x_max,
                    "y_min": sub_y_min,
                    "y_max": sub_y_max,
                }

                sub_plot_path = self.output_dir / f"evaluation_results_{task_name}.svg"
                generate_results_plot(
                    points=task_points,
                    results=task_results,
                    bbox=sub_bbox,
                    output_path=str(sub_plot_path),
                    model_type=f"{self.args.model_type} ({task_name})",
                )

        Logger.info(f"All visualizations saved to: {self.output_dir}")

    def shutdown(self):
        Logger.info("Shutting down evaluation runner...")
        self.world.stop()
        simulation_app.close()
        Logger.info("Shutdown complete")


def main():
    try:
        runner = EvaluationRunner(args)
        runner.run_evaluation()
    except KeyboardInterrupt:
        Logger.info("Evaluation interrupted by user")
    except Exception as e:
        Logger.error(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "runner" in locals():
            runner.shutdown()


if __name__ == "__main__":
    main()
