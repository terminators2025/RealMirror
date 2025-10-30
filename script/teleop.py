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
import json
import time
import os

parser = argparse.ArgumentParser(description="Run Teleoperation Simulation")
parser.add_argument(
    "--task",
    type=str,
    default="Task1_Kitchen_Cleanup",
    help="Name of the task configuration file in the 'tasks' directory (e.g., 'Task_1_Kitchen_Cleanup').",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Set data recording output directory.",
)
parser.add_argument(
    "--disable_recorder",
    action="store_true",
    help="Disable data recording.",
)
parser.add_argument(
    "--hide_ik_targets",
    action="store_true",
    help="If set, the visual target cubes for IK teleoperation will be hidden.",
)
parser.add_argument(
    "--pico-json-path",
    type=str,
    default="tools/xr_linker/data/live_pico_data.json",
    help="Path to the pico controller JSON data file.",
)
parser.add_argument(
    "--resolution",
    type=str,
    default="1920x1080",
    help="UI resolution (e.g., 1920x1080, 1280x720)",
)

args = parser.parse_args()

from isaacsim.simulation_app import SimulationApp

width, height = map(int, args.resolution.split("x"))

simulation_app = SimulationApp(
    {
        "headless": False,
        "physics_prim_path": "/physicsScene",
        "width": width,
        "height": height,
        "window_width": width,
        "window_height": height,
    }
)

from omni.isaac.core import World
import carb.input

from comm_config.robot_config import RobotConfig
from comm_config.simulation_config import SimulationConfig
from robots.robot_factory import RobotFactory
from controllers.pico_controller import PicoController
from controllers.control_mapping import A2PicoMapping
from teleoperation.teleop_manager import TeleopManager
from teleoperation.ik_solver import IKSolver
from teleoperation.slow_homing_manager import SlowHomingManager
from data_recording.recorder import DataRecorder
from simulation.scene_manager import SceneManager
from utility.logger import Logger
import utility.system_utils as system_utils
from comm_config.camera_config import CameraConfig
from utility.visualization_window import DataCollectionVisualizer
from typing import Optional


class TeleOp:
    def __init__(self, args):
        self.keyboard_sub = None
        self.recorder: Optional[DataRecorder] = None
        self.visualizer: Optional[DataCollectionVisualizer] = None
        self.args = args

        self._load_task_config(args.task)

        self.world = World(
            stage_units_in_meters=self.sim_config.stage_units_in_meters,
            physics_dt=self.sim_config.physics_dt,
            rendering_dt=self.sim_config.rendering_dt,
        )

        self.world.scene.add_default_ground_plane()

        self.scene_manager = SceneManager(self.world, self.sim_config, task_config=self.task_config)
        self.scene_manager._load_scene_usd()
        self.cameras = CameraConfig.create_cameras_from_task_config(self.task_config, self.robot_config)
        self.scene_manager._create_object_manager()
        self.robot = RobotFactory.create_robot(self.robot_type, self.robot_config)
        self.robot.initialize(self.world, simulation_app=simulation_app, mode="teleop")

        simulation_app.update()
        self.scene_manager._adjust_ground_plane()
        self.scene_manager._initialize_object_manager()
        self.scene_manager._diagnose_and_fix_physics()
        control_mapping = A2PicoMapping(self.sim_config)
        self.pico_controller = PicoController(args.pico_json_path)
        ik_solver = IKSolver(self.robot, self.robot_config)
        ik_solver.initialize()

        self.slow_homing_manager = SlowHomingManager(self.robot, teleop_manager=None, duration=0.2)

        self.slow_homing_manager.setup_controlled_joints(
            left_arm_joints=self.robot_config.arm_joints["left"],
            right_arm_joints=self.robot_config.arm_joints["right"],
            left_hand_joints=self.robot_config.hand_joints["left"],
            right_hand_joints=self.robot_config.hand_joints["right"],
        )

        self.teleop_manager = TeleopManager(
            self.robot, control_mapping, ik_solver, self.sim_config, self.slow_homing_manager
        )
        self.slow_homing_manager.teleop_manager = self.teleop_manager
        self.teleop_manager.initialize_target_cubes(self.world, args.hide_ik_targets)

        if not args.disable_recorder:
            if args.output_dir is None:
                raise ValueError("No output directory specified.")

            if not self.scene_manager.object_manager:
                raise RuntimeError("ObjectManager not initialized. Cannot start data recording.")

            self.recorder = DataRecorder(
                robot=self.robot,
                output_dir=args.output_dir,
                cameras=self.cameras,
                object_manager=self.scene_manager.object_manager,
                teleop_manager=self.teleop_manager,
                task_name=args.task,
            )
            self.recorder.setup()
            self.visualizer = DataCollectionVisualizer()
        self._setup_keyboard_callbacks()

    def _try_load_task_robot_gripper_config(self):
        if "gripper_config" in self.task_config["robot"]:
            self.robot_config.gripper_config = RobotConfig._process_gripper_config(self.task_config["robot"]["gripper_config"])

    def _load_task_config(self, task_name):
        """Loads the specified task configuration file."""
        task_config_path = os.path.join(system_utils.teleop_root_path(), "tasks", f"{task_name}.json")

        if not os.path.exists(task_config_path):
            raise FileNotFoundError(f"Task config file not found: {task_config_path}")

        Logger.info(f"Loading task config file: {task_config_path} successful!")

        with open(task_config_path) as f:
            self.task_config = json.load(f)
            Logger.info(f"Task Config:\n{self.task_config}")

        project_root = system_utils.teleop_root_path()
        robot_config_path = os.path.join(
            project_root,
            "comm_config",
            "configs",
            self.task_config["robot"]["robot_cfg"],
        )
        sim_config_path = os.path.join(project_root, "comm_config", "configs", "simulation_config.json")
        self.robot_type = self.task_config["robot"]["robot_type"]
        self.robot_config = RobotConfig.from_json(robot_config_path)
        self._try_load_task_robot_gripper_config()
        self.sim_config = SimulationConfig.from_json(sim_config_path, self.task_config)

    def _setup_keyboard_callbacks(self):
        """Sets up keyboard event listeners for controlling the simulation."""

        self._define_key_mappings()

        def keyboard_event_cb(event, *args, **kwargs):
            return self._handle_keyboard_event(event)

        app_input = carb.input.acquire_input_interface()
        self.keyboard_sub = app_input.subscribe_to_keyboard_events(None, keyboard_event_cb)  # type: ignore

    def _define_key_mappings(self):
        """Define keyboard mappings for different actions."""
        # Object movement mappings with visualization update
        self.movement_key_map = {
            carb.input.KeyboardInput.UP: lambda: self._move_and_update_viz("up", 0, 1),
            carb.input.KeyboardInput.DOWN: lambda: self._move_and_update_viz("down", 0, -1),
            carb.input.KeyboardInput.LEFT: lambda: self._move_and_update_viz("left", -1, 0),
            carb.input.KeyboardInput.RIGHT: lambda: self._move_and_update_viz("right", 1, 0),
        }

        # Object selection mappings
        self.object_selection_key_map = {
            carb.input.KeyboardInput.NUMPAD_0: lambda: self._reset_active_object_with_viz(),
            carb.input.KeyboardInput.NUMPAD_1: lambda: self._set_active_object_with_viz(0),
            carb.input.KeyboardInput.NUMPAD_2: lambda: self._set_active_object_with_viz(1),
            carb.input.KeyboardInput.NUMPAD_3: lambda: self._set_active_object_with_viz(2),
        }

    def _handle_keyboard_event(self, event):
        """Handle keyboard events based on defined mappings."""
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        if self.recorder:
            if self._handle_recorder_keys(event.input):
                return True

        if event.input in self.movement_key_map:
            self.movement_key_map[event.input]()
            return True

        if event.input in self.object_selection_key_map:
            self.object_selection_key_map[event.input]()
            return True

        return True

    def _handle_recorder_keys(self, key_input):
        """Handle recorder-specific keyboard inputs."""
        if not self.recorder:
            return False

        recorder_actions = {
            carb.input.KeyboardInput.S: self._toggle_recording_with_viz,
            carb.input.KeyboardInput.R: self.recorder.reset_current_demo,
            carb.input.KeyboardInput.N: self.recorder.save_and_start_next,
        }

        if key_input in recorder_actions:
            recorder_actions[key_input]()
            return True
        return False

    def _move_and_update_viz(self, direction: str, dx: int, dy: int):
        """Move object and update visualization position."""
        self.scene_manager.move_active_object(direction)
        if self.visualizer:
            self.visualizer.update_position(dx, dy)

    def _set_active_object_with_viz(self, index: int):
        """Set active object and reset visualization position to origin."""
        self.scene_manager.set_active_object(index)
        if self.visualizer:
            self.visualizer.reset_current_position()

    def _reset_active_object_with_viz(self):
        """Reset active object pose and reset visualization position to origin."""
        self.scene_manager.reset_active_object_pose()
        if self.visualizer:
            self.visualizer.reset_current_position()

    def _toggle_recording_with_viz(self):
        """Toggle recording and mark collection point in visualization."""
        if self.recorder:
            was_recording = self.recorder.is_recording
            self.recorder.toggle_recording()
            if not was_recording and self.recorder.is_recording and self.visualizer:
                self.visualizer.mark_collection_point()

    def run(self):
        """Starts and runs the main simulation loop."""
        Logger.info("Starting teleoperation simulation...")
        self.world.play()

        last_control_update = time.time()

        while simulation_app.is_running():
            self.world.step(render=True)

            current_time = time.time()
            dt = self.world.get_physics_dt()

            if current_time - last_control_update >= self.sim_config.control_update_interval:
                last_control_update = current_time
                pico_data = self.pico_controller.read_data()
                if pico_data:
                    for arm in ["left", "right"]:
                        if pico_data.get(arm):
                            self.teleop_manager.process_controller_data(pico_data[arm], arm, self.world)

            self.slow_homing_manager.update(dt)

            for arm in ["left", "right"]:
                self.robot.update_gripper(arm, "none", dt)

            if not self.slow_homing_manager.is_homing():
                self.teleop_manager.apply_ik_solutions(dt)

            if self.recorder and self.recorder.is_recording:
                self.recorder.collect_step_data()

            simulation_app.update()

    def shutdown(self):
        """Cleans up resources and closes the simulation."""
        # 清理可视化窗口
        if self.visualizer:
            self.visualizer.destroy()

        if self.keyboard_sub:
            try:
                # The keyboard subscription handle has an unsubscribe method
                self.keyboard_sub.unsubscribe()  # type: ignore
            except (AttributeError, TypeError):
                # If unsubscribe is not available, the subscription will be cleaned up automatically
                pass
        self.world.stop()
        simulation_app.close()
        Logger.info("Simulation closed")


def main():

    teleop = None
    try:
        teleop = TeleOp(args)
        teleop.run()
    except KeyboardInterrupt:
        Logger.info("Simulation interrupted by user.")
    except Exception as e:
        Logger.error(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if teleop:
            teleop.shutdown()


if __name__ == "__main__":
    main()
