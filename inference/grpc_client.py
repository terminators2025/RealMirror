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

import grpc
import numpy as np
import uuid
from typing import Optional
from utility.logger import Logger

try:
    from inference.proto import inference_pb2, inference_pb2_grpc
except ImportError:
    Logger.error("Failed to import gRPC proto files. Please run 'cd inference/proto && bash gen.sh'")
    raise


class GrpcInferenceClient:

    def __init__(self, server_address: str, timeout: float = 300.0):
        self.server_address = server_address
        self.timeout = timeout
        self.channel = None
        self.stub = None
        self.episode_id = str(uuid.uuid4())

        Logger.info(f"GrpcInferenceClient initialized for server: {server_address}")

    def connect(self) -> bool:
        try:
            self.channel = grpc.insecure_channel(self.server_address)
            self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)
            Logger.info(f"Connected to gRPC inference server at {self.server_address}")
            return True
        except Exception as e:
            Logger.error(f"Failed to connect to gRPC server: {e}")
            return False

    def disconnect(self):
        if self.channel:
            self.channel.close()
            Logger.info("Disconnected from gRPC inference server")

    def reset_episode(self):
        self.episode_id = str(uuid.uuid4())
        Logger.info(f"Reset episode ID: {self.episode_id}")

    def predict(
        self,
        state: np.ndarray,
        left_wrist_image: np.ndarray,
        right_wrist_image: np.ndarray,
        head_image: np.ndarray,
        task: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Send inference request to the gRPC server.

        Args:
            state: State vector (26-DOF) in hybrid format
            left_wrist_image: Left wrist camera image (H, W, C) in [0, 1] float range
            right_wrist_image: Right wrist camera image (H, W, C) in [0, 1] float range
            head_image: Head camera image (H, W, C) in [0, 1] float range
            task: Optional task description string

        Returns:
            Action array (26-DOF) in hybrid format or None if inference fails
        """
        if self.stub is None:
            Logger.error("gRPC client not connected. Call connect() first.")
            return None

        try:
            head_image_data = self._prepare_image_data(head_image)
            left_wrist_image_data = self._prepare_image_data(left_wrist_image)
            right_wrist_image_data = self._prepare_image_data(right_wrist_image)

            request = inference_pb2.InferenceRequest(
                episode_id=self.episode_id,
                state=state.flatten().tolist(),
                head_camera=head_image_data,
                left_wrist_camera=left_wrist_image_data,
                right_wrist_camera=right_wrist_image_data,
                task=task if task else "",
            )

            response = self.stub.Predict(request, timeout=self.timeout)

            action = np.array(response.prediction, dtype=np.float32)

            return action

        except grpc.RpcError as e:
            Logger.error(f"gRPC error during inference: {e.code()}: {e.details()}")
            return None
        except Exception as e:
            Logger.error(f"Error during gRPC inference: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _prepare_image_data(self, image: np.ndarray) -> inference_pb2.ImageData:
        height, width, channels = image.shape

        image_flat = image.flatten().tolist()

        return inference_pb2.ImageData(data=image_flat, height=height, width=width, channels=channels)


class RemoteInferenceEngine:

    def __init__(self, server_address: str, unit_converter, num_actions_in_chunk: int = 5, timeout: float = 300.0):
        self.grpc_client = GrpcInferenceClient(server_address, timeout)
        self.unit_converter = unit_converter
        self.num_actions_in_chunk = num_actions_in_chunk
        self.task_name = None

        Logger.info("RemoteInferenceEngine initialized")

    def connect(self) -> bool:
        return self.grpc_client.connect()

    def disconnect(self):
        self.grpc_client.disconnect()

    def set_task_name(self, task_name: str):
        self.task_name = task_name
        Logger.info(f"Task name set to: {task_name}")

    def get_observation_state(self, robot_view, state_joint_indices: np.ndarray) -> Optional[np.ndarray]:
        if not robot_view or not robot_view.is_valid() or state_joint_indices is None:
            Logger.warning("Robot view or state joint indices not ready")
            return None

        try:
            full_joint_pos = robot_view.get_joint_positions()
            if full_joint_pos is None or full_joint_pos.size == 0:
                Logger.warning("Failed to get valid joint positions from robot view")
                return None

            observation_state_radians = full_joint_pos[0, state_joint_indices].astype(np.float32)

            if observation_state_radians.shape[0] != 26:
                Logger.error(f"State vector dimension is {observation_state_radians.shape[0]}, expected 26")
                return None

            observation_state_hybrid = self.unit_converter.radian_to_gear_vector(observation_state_radians)

            return observation_state_hybrid

        except Exception as e:
            Logger.error(f"Error getting observation state: {e}")
            return None

    def process_image(self, img_bgr: np.ndarray) -> np.ndarray:
        return img_bgr.astype(np.float32) / 255.0

    def run_inference(
        self,
        state_vector: np.ndarray,
        left_wrist_image: np.ndarray,
        right_wrist_image: np.ndarray,
        head_image: np.ndarray,
        save_images: bool = False,
    ) -> Optional[np.ndarray]:
        try:
            if save_images:
                from PIL import Image

                Image.fromarray(left_wrist_image).save("./left_wrist_img_grpc.png")
                Image.fromarray(right_wrist_image).save("./right_wrist_img_grpc.png")
                Image.fromarray(head_image).save("./head_img_grpc.png")

            left_wrist_float = self.process_image(left_wrist_image)
            right_wrist_float = self.process_image(right_wrist_image)
            head_float = self.process_image(head_image)

            action_26d_hybrid = self.grpc_client.predict(
                state_vector, left_wrist_float, right_wrist_float, head_float, task=self.task_name
            )

            if action_26d_hybrid is None:
                Logger.error("Failed to get action from gRPC server")
                return None

            if action_26d_hybrid.ndim == 1:
                if action_26d_hybrid.shape[0] % 26 != 0:
                    Logger.error(
                        f"Action array size {action_26d_hybrid.shape[0]} is not a multiple of 26. "
                        f"Expected format: (chunk_size * 26)"
                    )
                    return None

                chunk_size = action_26d_hybrid.shape[0] // 26
                action_26d_hybrid = action_26d_hybrid.reshape(chunk_size, 26)

            if action_26d_hybrid.shape[0] > self.num_actions_in_chunk:
                action_26d_hybrid = action_26d_hybrid[: self.num_actions_in_chunk]
                Logger.debug(f"Taking first {self.num_actions_in_chunk} actions from chunk")

            numpy_action_26d_radians = self.unit_converter.gear_to_radian_vector(action_26d_hybrid)

            expanded_action_38d_radians = self.unit_converter.expand_26d_to_38d_sim_action(numpy_action_26d_radians)

            return expanded_action_38d_radians

        except Exception as e:
            Logger.error(f"Error during remote inference: {e}")
            import traceback

            traceback.print_exc()
            return None

    def reset(self):
        self.grpc_client.reset_episode()
        Logger.info("Remote inference engine reset")
