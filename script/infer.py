import argparse
import json
import os
import numpy as np

parser = argparse.ArgumentParser(
    description="Run robot inference with pretrained models"
)
parser.add_argument(
    "--task", type=str, default="Task1_Kitchen_Cleanup", help="Task configuration name"
)
parser.add_argument(
    "--model-type",
    type=str,
    default="act",
    choices=["act", "diffusion", "smolvla"],
    help="Type of model to use",
)
parser.add_argument(
    "--model-path", type=str, required=True, help="Path to pretrained model"
)
parser.add_argument(
    "--headless", action="store_true", help="Run in headless mode without GUI"
)
parser.add_argument(
    "--arc2gear",
    action="store_true",
    help="Use gear position conversion for hand joints",
)
parser.add_argument(
    "--task-name",
    type=str,
    default=None,
    help="Task description for models that require it",
)
parser.add_argument(
    "--num-actions",
    type=int,
    default=5,
    help="Number of actions to use from model output chunk",
)
parser.add_argument(
    "--max-steps", type=int, default=1000, help="Maximum number of simulation steps"
)

parser.add_argument(
    "--resolution",
    type=str,
    default="1920x1080",
    help="UI resolution (e.g., 1920x1080, 1280x720)",
)

args = parser.parse_args()

width, height = map(int, args.resolution.split("x"))


from isaacsim.simulation_app import SimulationApp

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

from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
import carb.input

from comm_config.robot_config import RobotConfig
from comm_config.simulation_config import SimulationConfig
from comm_config.camera_config import CameraConfig
from robots.robot_factory import RobotFactory
from simulation.scene_manager import SceneManager
from inference import UnitConverter, ModelLoader, InferenceEngine, InferenceConfig
from utility.logger import Logger
from utility.camera_utils import get_camera_image
import utility.system_utils as system_utils


class InferenceRunner:
    """
    Main class for running inference with pretrained models.
    """

    def __init__(self, args):
        """
        Initialize the inference runner.

        Args:
            args: Command line arguments
        """
        self.args = args
        self.is_inferencing = False
        self.keyboard_sub = None

        # Load configurations
        self._load_configurations()

        # Initialize world
        self.world = World(
            stage_units_in_meters=self.sim_config.stage_units_in_meters,
            physics_dt=self.sim_config.physics_dt,
            rendering_dt=self.sim_config.rendering_dt,
        )
        self.world.scene.add_default_ground_plane()

        # Initialize robot
        self.robot = RobotFactory.create_robot(self.robot_type, self.robot_config)
        self.robot.initialize(self.world, simulation_app)

        # Get robot view and joint indices
        self.robot_view = self.robot.robot_view
        self._setup_joint_indices()

        # Initialize scene
        self.scene_manager = SceneManager(self.world, self.sim_config)
        self.scene_manager.setup_scene()

        # Initialize cameras
        self.cameras = CameraConfig.create_cameras_from_task_config(
            self.task_config, self.robot_config
        )

        # Initialize inference components
        self._setup_inference()

        # Setup keyboard controls
        self._setup_keyboard_controls()

        Logger.info("InferenceRunner initialized successfully")

    def _load_configurations(self):
        """Load task and robot configurations."""
        # Load task config
        project_root = system_utils.teleop_root_path()
        task_config_path = os.path.join(project_root, "tasks", f"{self.args.task}.json")

        if not os.path.exists(task_config_path):
            raise FileNotFoundError(f"Task config not found: {task_config_path}")

        with open(task_config_path) as f:
            self.task_config = json.load(f)

        # Load robot config
        self.robot_type = self.task_config["robot"]["robot_type"]
        robot_config_path = os.path.join(
            project_root,
            "comm_config",
            "configs",
            self.task_config["robot"]["robot_cfg"],
        )
        self.robot_config = RobotConfig.from_json(robot_config_path)

        # Load simulation config
        sim_config_path = os.path.join(
            project_root, "comm_config", "configs", "simulation_config.json"
        )
        self.sim_config = SimulationConfig.from_json(sim_config_path, self.task_config)

        Logger.info(f"Loaded configurations for task: {self.args.task}")

    def _setup_joint_indices(self):
        """Setup joint indices for state extraction and action application."""
        # Get all joint names from robot
        all_dof_names = self.robot.get_dof_names()
        if not all_dof_names:
            raise RuntimeError("Failed to get robot DOF names")

        # Create mapping
        dof_map = {name: i for i, name in enumerate(all_dof_names)}

        # Setup state joint indices (26-DOF)
        state_joint_names = UnitConverter.STATE_SIM_JOINT_NAMES_26DOF
        try:
            state_indices = [dof_map[name] for name in state_joint_names]
            self.state_joint_indices = np.array(state_indices, dtype=np.int32)
            Logger.info(f"Mapped {len(self.state_joint_indices)} state joint indices")
        except KeyError as e:
            Logger.error(f"Failed to map state joint: {e}")
            raise

        # Setup action joint indices (38-DOF for simulation)
        # This includes mimic joints for hands
        left_arm_joints = self.robot_config.arm_joints.get("left", [])
        right_arm_joints = self.robot_config.arm_joints.get("right", [])

        # Add extra joints if not present
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

        # Hand mimic joints
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

        # Combine all action joints
        action_joint_names = (
            left_arm_joints + right_arm_joints + left_hand_mimic + right_hand_mimic
        )

        try:
            action_indices = [dof_map[name] for name in action_joint_names]
            self.action_joint_indices = np.array(action_indices, dtype=np.int32)
            Logger.info(f"Mapped {len(self.action_joint_indices)} action joint indices")
        except KeyError as e:
            Logger.error(f"Failed to map action joint: {e}")
            raise

    def _setup_inference(self):
        """Setup inference components."""
        # Create inference configuration
        self.inference_config = InferenceConfig(
            model_type=self.args.model_type,
            model_path=self.args.model_path,
            use_gear_conversion=self.args.arc2gear,
            num_actions_in_chunk=self.args.num_actions,
            task_name=self.args.task_name,
        )

        # Initialize components
        self.unit_converter = UnitConverter(
            arc_to_gear=self.inference_config.use_gear_conversion,
            max_gear=self.inference_config.max_gear,
        )

        self.model_loader = ModelLoader(
            model_type=self.inference_config.model_type,
            model_path=self.inference_config.model_path,
        )

        # Load model
        if not self.model_loader.load_model():
            raise RuntimeError("Failed to load inference model")

        self.inference_engine = InferenceEngine(
            model_loader=self.model_loader,
            unit_converter=self.unit_converter,
            num_actions_in_chunk=self.inference_config.num_actions_in_chunk,
            image_resolution=self.inference_config.image_resolution,
        )

        # Set task name if provided
        if self.inference_config.task_name:
            self.inference_engine.set_task_name(self.inference_config.task_name)

        Logger.info("Inference components initialized")

    def _setup_keyboard_controls(self):
        """Setup keyboard event handlers."""

        def keyboard_event_cb(event, *args, **kwargs):
            if event.type != carb.input.KeyboardEventType.KEY_PRESS:
                return True

            if event.input == carb.input.KeyboardInput.S:
                # Toggle inference
                self.is_inferencing = not self.is_inferencing
                Logger.info(
                    f"Inference {'STARTED' if self.is_inferencing else 'STOPPED'}"
                )

                if self.is_inferencing:
                    # Log initial state
                    state = self.inference_engine.get_observation_state(
                        self.robot_view, self.state_joint_indices
                    )
                    if state is not None:
                        Logger.info(f"Initial state (26-DOF): {np.round(state, 2)}")

            elif event.input == carb.input.KeyboardInput.R:
                # Reset environment
                self.reset_environment()
                Logger.info("Environment reset")

            elif event.input == carb.input.KeyboardInput.Q:
                # Quit
                Logger.info("Quit requested")
                self.shutdown()

            # Object movement controls
            elif event.input == carb.input.KeyboardInput.UP:
                self.scene_manager.move_active_object("up")
            elif event.input == carb.input.KeyboardInput.DOWN:
                self.scene_manager.move_active_object("down")
            elif event.input == carb.input.KeyboardInput.LEFT:
                self.scene_manager.move_active_object("left")
            elif event.input == carb.input.KeyboardInput.RIGHT:
                self.scene_manager.move_active_object("right")

            # Object selection
            elif event.input == carb.input.KeyboardInput.NUMPAD_0:
                self.scene_manager.reset_active_object_pose()
            elif event.input == carb.input.KeyboardInput.NUMPAD_1:
                self.scene_manager.set_active_object(0)
            elif event.input == carb.input.KeyboardInput.NUMPAD_2:
                self.scene_manager.set_active_object(1)
            elif event.input == carb.input.KeyboardInput.NUMPAD_3:
                self.scene_manager.set_active_object(2)

            return True

        # Subscribe to keyboard events
        app_input = carb.input.acquire_input_interface()
        self.keyboard_sub = app_input.subscribe_to_keyboard_events(
            None, keyboard_event_cb
        )

        Logger.info("Keyboard controls setup complete")
        Logger.info("Controls:")
        Logger.info("  S - Start/Stop inference")
        Logger.info("  R - Reset environment")
        Logger.info("  Q - Quit")
        Logger.info("  Arrow Keys - Move active object")
        Logger.info("  Numpad 0 - Reset object pose")
        Logger.info("  Numpad 1-3 - Select object")

    def reset_environment(self):
        """Reset the environment to initial state."""
        # Reset inference engine
        self.inference_engine.reset()
        self.is_inferencing = False

        if self.robot and self.robot.robot_ref:
            initial_positions = self.task_config["robot"].get(
                "arm_joint_angles_rad", {}
            )
            hand_joints_to_configure = self.task_config["robot"].get(
                "hand_joints_to_configure", []
            )
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
                Logger.info(
                    f"Robot reset to initial joint positions. {current_positions}"
                )

        # Reset scene objects
        self.scene_manager.reset_active_object_pose()

        Logger.info("Environment reset complete")

    def run_inference_step(self):
        """Run a single inference step."""
        # Get observation state
        state = self.inference_engine.get_observation_state(
            self.robot_view, self.state_joint_indices
        )
        if state is None:
            Logger.warning("Failed to get observation state")
            return

        # Get camera images
        head_image = get_camera_image(self.cameras.get("head_camera"), "rgb")
        left_wrist_image = get_camera_image(
            self.cameras.get("left_wrist_camera"), "rgb"
        )
        right_wrist_image = get_camera_image(
            self.cameras.get("right_wrist_camera"), "rgb"
        )

        if head_image is None or left_wrist_image is None or right_wrist_image is None:
            Logger.warning("Failed to get camera images")
            return

        # Run inference
        actions = self.inference_engine.run_inference(
            state,
            left_wrist_image,
            right_wrist_image,
            head_image,
            save_images=False,  # Set to True for debugging
        )

        if actions is None:
            Logger.warning("Failed to get actions from inference")
            return

        # Apply actions
        if len(self.action_joint_indices) != actions.shape[-1]:
            Logger.error(
                f"Action dimension mismatch: {len(self.action_joint_indices)} vs {actions.shape[-1]}"
            )
            return

        # Apply each action in the chunk
        for action_step in actions:
            action = ArticulationAction(
                joint_positions=action_step.astype(np.float32),
                joint_indices=self.action_joint_indices,
            )
            self.robot.robot_ref.get_articulation_controller().apply_action(action)

    def run(self):
        """Main run loop."""
        Logger.info("Starting inference runner...")

        # Start simulation
        self.world.play()

        # Stabilize scene
        Logger.info("Stabilizing scene...")
        for _ in range(50):
            self.world.step(render=True)

        # Reset environment
        self.reset_environment()

        Logger.info("Ready for inference. Press 'S' to start.")

        # Main loop
        step_count = 0
        while simulation_app.is_running() and step_count < self.args.max_steps:
            self.world.step(render=True)

            if self.world.is_playing() and self.is_inferencing:
                self.run_inference_step()
                step_count += 1

                if step_count % 100 == 0:
                    Logger.info(f"Inference step: {step_count}/{self.args.max_steps}")

            simulation_app.update()

        if step_count >= self.args.max_steps:
            Logger.info(f"Reached maximum steps: {self.args.max_steps}")

    def shutdown(self):
        """Clean shutdown."""
        Logger.info("Shutting down...")

        if self.keyboard_sub:
            try:
                self.keyboard_sub.unsubscribe()
            except:
                pass

        self.world.stop()
        simulation_app.close()
        Logger.info("Shutdown complete")


def main():
    """Main entry point."""
    try:
        runner = InferenceRunner(args)
        runner.run()
    except KeyboardInterrupt:
        Logger.info("Interrupted by user")
    except Exception as e:
        Logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "runner" in locals():
            runner.shutdown()


if __name__ == "__main__":
    main()
