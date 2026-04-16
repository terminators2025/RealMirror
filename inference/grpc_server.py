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

import torch
import grpc
import threading
import uuid
import numpy as np
import time
import queue
from collections import OrderedDict
from typing import Optional, Callable, Dict
from utility.logger import Logger

try:
    from inference.proto import inference_pb2, inference_pb2_grpc
except ImportError:
    Logger.error("Failed to import gRPC proto files. Please run 'cd inference/proto && bash gen.sh'")
    raise

from inference.model_loader import ModelLoader
from inference.temporal_ensembler import TemporalEnsembler
from inference.metrics_collector import MetricsCollector


class InferenceRequestItem:

    def __init__(self, request_id, episode_id, observation_data, callback, request_received_time):
        self.request_id = request_id
        self.episode_id = episode_id
        self.observation_data = observation_data
        self.callback = callback
        self.request_received_time = request_received_time


class InferenceResultItem:

    def __init__(self, request_id, episode_id, output_data, error=None):
        self.request_id = request_id
        self.episode_id = episode_id
        self.output_data = output_data
        self.error = error


class BatchingInferenceEngine:

    def __init__(
        self,
        model_loader: ModelLoader,
        temporal_ensemble_coeff: float = 0.01,
        n_action_steps: int = 30,
        max_batch_size: int = 8,
        batch_timeout: float = 0.05,
        max_ensemblers: int = 100,
        metrics_interval: float = 1.0,
    ):
        self.model_loader = model_loader
        self.model = model_loader.get_policy()
        self.device = model_loader.device
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout

        self.request_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._batching_thread: Optional[threading.Thread] = None

        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        self.n_action_steps = n_action_steps
        self.ensemblers: OrderedDict[str, tuple[TemporalEnsembler, float]] = OrderedDict()
        self.max_ensemblers = max_ensemblers
        self.ensembler_lock = threading.Lock()

        self.metrics = MetricsCollector(report_interval=metrics_interval)

        Logger.info("BatchingInferenceEngine initialized")

    def start(self):
        if self.model:
            self.model.eval()
        self.metrics.start()
        self._batching_thread = threading.Thread(target=self._batching_loop, daemon=True)
        self._batching_thread.start()
        Logger.info("BatchingInferenceEngine started")

    def stop(self):
        self._stop_event.set()
        self.request_queue.put(None)
        if self._batching_thread:
            self._batching_thread.join()
        self.metrics.stop()
        Logger.info("BatchingInferenceEngine stopped")

    def infer(self, episode_id: str, observation_data: Dict, callback: Callable, request_received_time: float) -> str:
        request_id = str(uuid.uuid4())
        request = InferenceRequestItem(request_id, episode_id, observation_data, callback, request_received_time)
        self.request_queue.put(request)
        return request_id

    def _batching_loop(self):
        Logger.info("Batching loop started")
        current_batch_requests = []
        batch_start_time = time.time()

        while not self._stop_event.is_set():
            try:
                timeout = max(0.001, self.batch_timeout - (time.time() - batch_start_time))
                request = self.request_queue.get(timeout=timeout)

                if request is None:
                    continue

                current_batch_requests.append(request)

            except queue.Empty:
                pass

            if len(current_batch_requests) >= self.max_batch_size or (
                time.time() - batch_start_time >= self.batch_timeout and current_batch_requests
            ):

                self._process_batch(current_batch_requests)
                current_batch_requests = []
                batch_start_time = time.time()

        self._flush_queue(current_batch_requests)
        Logger.info("Batching loop stopped")

    def _flush_queue(self, current_batch_requests):
        if current_batch_requests:
            self._process_batch(current_batch_requests)
            current_batch_requests = []

        while not self.request_queue.empty():
            try:
                request = self.request_queue.get_nowait()
                if request is None:
                    continue
                current_batch_requests.append(request)
                if len(current_batch_requests) >= self.max_batch_size:
                    self._process_batch(current_batch_requests)
                    current_batch_requests = []
            except queue.Empty:
                break

        if current_batch_requests:
            self._process_batch(current_batch_requests)

    def _process_batch(self, batch_requests):
        if not batch_requests:
            return

        request_ids = [req.request_id for req in batch_requests]

        self.metrics.record_batch(len(batch_requests))

        current_time = time.time()
        for req in batch_requests:
            queue_latency = (current_time - req.request_received_time) * 1000
            self.metrics.record_queue_latency(queue_latency)

        batched_observation = {}
        try:
            if (
                "observation.state" in batch_requests[0].observation_data
                and batch_requests[0].observation_data["observation.state"] is not None
            ):
                batched_observation["observation.state"] = torch.stack(
                    [req.observation_data["observation.state"] for req in batch_requests]
                ).to(self.device)

            image_keys = [k for k in batch_requests[0].observation_data if "observation.images." in k]
            for key in image_keys:
                if batch_requests[0].observation_data[key] is not None:
                    batched_observation[key] = torch.stack([req.observation_data[key] for req in batch_requests]).to(
                        self.device
                    )

            if "task" in batch_requests[0].observation_data:
                batched_observation["task"] = [req.observation_data["task"] for req in batch_requests]

        except Exception as e:
            Logger.error(f"Error during batch preparation for requests {request_ids}: {e}", exc_info=True)
            for req in batch_requests:
                req.callback(InferenceResultItem(req.request_id, req.episode_id, None, error=str(e)))
            return

        try:
            with torch.no_grad():
                start_time = time.time()
                if hasattr(self.model, "select_action_custom"):
                    raw_outputs = self.model.select_action_custom(batched_observation)
                else:
                    raw_outputs = self.model.select_action(batched_observation)
                end_time = time.time()

                inference_time_ms = (end_time - start_time) * 1000
                self.metrics.record_inference_time(inference_time_ms)

                if isinstance(raw_outputs, (list, tuple)):
                    raw_outputs = torch.stack(raw_outputs)

                processed_outputs = []
                for i, request_item in enumerate(batch_requests):
                    episode_id = request_item.episode_id

                    current_raw_actions = self._extract_action_from_batch(raw_outputs, i)

                    ensembler_instance = self._get_ensembler(episode_id)

                    ensembled_action = ensembler_instance.update(current_raw_actions)

                    output_data_flat = ensembled_action.cpu().numpy().flatten().tolist()
                    processed_outputs.append(output_data_flat)

                for i, request in enumerate(batch_requests):
                    result = InferenceResultItem(request.request_id, request.episode_id, processed_outputs[i])
                    request.callback(result)

        except Exception as e:
            Logger.error(f"Error during model inference for requests {request_ids}: {e}", exc_info=True)
            for req in batch_requests:
                req.callback(InferenceResultItem(req.request_id, req.episode_id, None, error=str(e)))

    def _extract_action_from_batch(self, raw_outputs: torch.Tensor, index: int) -> torch.Tensor:
        if raw_outputs.ndim == 2:
            current_raw_actions = raw_outputs[index].unsqueeze(0).unsqueeze(1)
        elif raw_outputs.ndim == 3:
            current_raw_actions = raw_outputs[index].unsqueeze(0)
        else:
            Logger.error(f"Unexpected output dimension from model: {raw_outputs.ndim}. Expected 2 or 3.")
            action_dim = getattr(self.model, "action_dim", 26)
            chunk_size = self.n_action_steps if self.n_action_steps > 1 else 1
            current_raw_actions = torch.zeros(1, chunk_size, action_dim, device=self.device)

        return current_raw_actions

    def _get_ensembler(self, episode_id: str) -> TemporalEnsembler:
        with self.ensembler_lock:
            if episode_id not in self.ensemblers:
                Logger.info(f"Creating new TemporalEnsembler for episode_id: {episode_id}")
                self.ensemblers[episode_id] = (
                    TemporalEnsembler(self.temporal_ensemble_coeff, self.n_action_steps),
                    time.time(),
                )

                if len(self.ensemblers) > self.max_ensemblers:
                    oldest_episode_id, _ = self.ensemblers.popitem(last=False)
                    Logger.info(
                        f"Removed oldest TemporalEnsembler for episode_id: {oldest_episode_id} "
                        f"(Cache size: {len(self.ensemblers)})"
                    )
            else:
                ensembler, _ = self.ensemblers.pop(episode_id)
                self.ensemblers[episode_id] = (ensembler, time.time())

            return self.ensemblers[episode_id][0]


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):

    def __init__(self, inference_engine: BatchingInferenceEngine):
        self.inference_engine = inference_engine
        self.results_cache: Dict[str, InferenceResultItem] = {}
        self.results_locks: Dict[str, threading.Event] = {}
        self.cache_lock = threading.Lock()
        Logger.info("InferenceServicer initialized")

    def Predict(self, request, context):
        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            episode_id = request.episode_id
            if not episode_id:
                episode_id = str(uuid.uuid4())
                Logger.warning(f"Client did not provide episode_id, generating: {episode_id}")

            observation = self._parse_observation(request)
            if observation is None:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Failed to parse observation data")
                return

            event = threading.Event()

            with self.cache_lock:
                self.results_locks[request_id] = event

            def callback(result: InferenceResultItem):
                with self.cache_lock:
                    self.results_cache[request_id] = result
                    if request_id in self.results_locks:
                        self.results_locks[request_id].set()

            self.inference_engine.infer(episode_id, observation, callback, start_time)

            if not event.wait(timeout=300):
                with self.cache_lock:
                    self.results_locks.pop(request_id, None)
                    self.results_cache.pop(request_id, None)
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, f"Inference timed out for {request_id}")
                return

            with self.cache_lock:
                result = self.results_cache.pop(request_id, None)
                self.results_locks.pop(request_id, None)

            if result and result.output_data is not None:
                end_time = time.time()
                latency = (end_time - start_time) * 1000
                self.inference_engine.metrics.record_e2e_latency(latency)
                return inference_pb2.InferenceResponse(
                    request_id=result.request_id, prediction=result.output_data, episode_id=result.episode_id
                )
            elif result and result.error:
                context.abort(grpc.StatusCode.INTERNAL, f"Inference failed: {result.error}")
            else:
                context.abort(grpc.StatusCode.INTERNAL, f"Failed to get inference result for {request_id}")

        except Exception as e:
            Logger.error(f"Error during gRPC Predict call: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"Internal server error: {e}")

    def _parse_observation(self, request):
        try:
            if not request.state:
                state_tensor = None
                Logger.debug("request.state is empty")
            else:
                state_tensor = torch.tensor(request.state, dtype=torch.float32)

            head_image_tensor = self._process_grpc_image(request.head_camera)
            left_wrist_image_tensor = self._process_grpc_image(request.left_wrist_camera)
            right_wrist_image_tensor = self._process_grpc_image(request.right_wrist_camera)

            observation = {
                "observation.state": state_tensor,
                "observation.images.head_camera": head_image_tensor,
                "observation.images.left_wrist_camera": left_wrist_image_tensor,
                "observation.images.right_wrist_camera": right_wrist_image_tensor,
            }

            if request.task:
                observation["task"] = request.task

            return observation

        except Exception as e:
            Logger.error(f"Failed to parse observation: {e}", exc_info=True)
            return None

    def _process_grpc_image(self, img_data_proto) -> Optional[torch.Tensor]:
        if not img_data_proto.data:
            return None

        if not (img_data_proto.height > 0 and img_data_proto.width > 0 and img_data_proto.channels > 0):
            Logger.error(
                f"Invalid image dimensions: h={img_data_proto.height}, "
                f"w={img_data_proto.width}, c={img_data_proto.channels}"
            )
            return None

        img_array = np.array(img_data_proto.data, dtype=np.float32).reshape(
            img_data_proto.height, img_data_proto.width, img_data_proto.channels
        )

        return torch.tensor(img_array).permute(2, 0, 1)
