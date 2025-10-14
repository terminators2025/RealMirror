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
import torch
from typing import Optional
from utility.logger import Logger


class ModelLoader:
    """
    Handles loading and management of different types of inference models.
    """

    # Supported model types
    SUPPORTED_MODELS = ["act", "pi0", "diffusion", "smolvla"]

    def __init__(self, model_type: str, model_path: str, device: Optional[str] = None):
        """
        Initialize the model loader.

        Args:
            model_type: Type of model to load (act, pi0, diffusion, smolvla)
            model_path: Path to the pretrained model
            device: Device to load model on (cuda/cpu). If None, auto-detect.
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. Supported types: {self.SUPPORTED_MODELS}"
            )

        self.model_type = model_type
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = None

        Logger.info(
            f"ModelLoader initialized for {model_type} model on device: {self.device}"
        )

    def _validate_model_path(self) -> bool:
        """
        Validate that the model path exists.

        Returns:
            True if path exists, False otherwise
        """
        if not os.path.exists(self.model_path):
            Logger.error(f"Model path does not exist: {self.model_path}")
            return False
        return True

    def _get_policy_class(self):
        """
        Get the appropriate policy class based on model type.

        Returns:
            Policy class for the specified model type

        Raises:
            ImportError: If the policy class cannot be imported
        """
        policy_map = {
            "act": ("lerobot.common.policies.act.modeling_act", "ACTPolicy"),
            "pi0": ("lerobot.common.policies.pi0.modeling_pi0", "PI0Policy"),
            "diffusion": (
                "lerobot.common.policies.diffusion.modeling_diffusion",
                "DiffusionPolicy",
            ),
            "smolvla": (
                "lerobot.common.policies.smolvla.modeling_smolvla",
                "SmolVLAPolicy",
            ),
        }

        if self.model_type not in policy_map:
            raise ValueError(f"Model type {self.model_type} not implemented")

        module_path, class_name = policy_map[self.model_type]
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    def _load_and_prepare_policy(self, policy_class):
        """
        Load policy from pretrained weights and prepare for inference.

        Args:
            policy_class: The policy class to instantiate

        Returns:
            Loaded and prepared policy model
        """
        Logger.info(
            f"Loading {self.model_type.upper()} model from: {self.model_path}"
        )

        policy = policy_class.from_pretrained(self.model_path)
        policy.to(self.device)
        policy.eval()

        return policy

    def _compile_model(self):
        """
        Apply torch.compile optimization to the model for better performance.
        Compilation is applied after .to() and .eval() to ensure the compiler
        knows the target device and can optimize for inference mode.
        """
        compile_configs = {
            "act": {
                "target": lambda: self.policy.model,
                "setter": lambda compiled: setattr(self.policy, "model", compiled),
                "backend": "aot_eager",
                "kwargs": {"dynamic": True, "fullgraph": False},
                "name": "ACT model",
            },
            "diffusion": {
                "target": lambda: self.policy.diffusion.unet,
                "setter": lambda compiled: setattr(
                    self.policy.diffusion, "unet", compiled
                ),
                "backend": "inductor",
                "kwargs": {"dynamic": True},
                "name": "Diffusion UNet",
            },
            "smolvla": {
                "target": lambda: self.policy.model.vlm_with_expert,
                "setter": lambda compiled: setattr(
                    self.policy.model, "vlm_with_expert", compiled
                ),
                "backend": "inductor",
                "kwargs": {"dynamic": True},
                "name": "SmolVLA VLM",
            },
        }

        if self.model_type not in compile_configs:
            Logger.info(f"No compilation config for {self.model_type}, skipping")
            return

        try:
            config = compile_configs[self.model_type]
            target_model = config["target"]()
            compiled_model = torch.compile(
                target_model, backend=config["backend"], **config["kwargs"]
            )
            config["setter"](compiled_model)
            Logger.info(
                f"{config['name']} compiled with {config['backend']} backend"
            )
        except Exception as e:
            Logger.warning(
                f"Failed to compile {self.model_type} model, continuing without optimization: {e}"
            )

    def load_model(self) -> bool:
        """
        Load the pretrained model based on the model type.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Step 1: Validate model path
            if not self._validate_model_path():
                return False

            # Step 2: Get the appropriate policy class
            policy_class = self._get_policy_class()

            # Step 3: Load and prepare the policy
            self.policy = self._load_and_prepare_policy(policy_class)

            # Step 4: Apply compilation optimization
            self._compile_model()

            Logger.info(f"Model {self.model_type.upper()} loaded successfully")
            return True

        except Exception as e:
            Logger.error(f"Failed to load model: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_policy(self):
        """
        Get the loaded policy model.

        Returns:
            The loaded policy model or None if not loaded
        """
        return self.policy

    def reset(self):
        """
        Reset the policy model if it has a reset method.
        """
        if self.policy and hasattr(self.policy, "reset"):
            self.policy.reset()
            Logger.info("Policy model reset")
