# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.


import torch
from typing import Optional
from utility.logger import Logger


class TemporalEnsembler:

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self._ensembled_actions: Optional[torch.Tensor] = None
        self._ensembled_actions_count: Optional[torch.Tensor] = None

    def reset(self):
        self._ensembled_actions = None
        self._ensembled_actions_count = None

    @property
    def ensembled_actions(self) -> torch.Tensor:
        """Get ensembled actions, raises error if not initialized."""
        if self._ensembled_actions is None:
            raise ValueError("Ensembled actions are not initialized. Call update() first.")
        return self._ensembled_actions

    @ensembled_actions.setter
    def ensembled_actions(self, value: Optional[torch.Tensor]):
        """Set ensembled actions."""
        self._ensembled_actions = value

    @property
    def ensembled_actions_count(self) -> torch.Tensor:
        """Get ensembled actions count, raises error if not initialized."""
        if self._ensembled_actions_count is None:
            raise ValueError("Ensembled actions count is not initialized. Call update() first.")
        return self._ensembled_actions_count

    @ensembled_actions_count.setter
    def ensembled_actions_count(self, value: Optional[torch.Tensor]):
        """Set ensembled actions count."""
        self._ensembled_actions_count = value

    def _is_initialized(self) -> bool:
        """Check if ensembler is initialized."""
        return self._ensembled_actions is not None and self._ensembled_actions_count is not None

    def _prepare_device(self, actions: torch.Tensor):
        """Move ensemble weights to the same device as actions."""
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)

    def _normalize_actions_shape(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions to shape (batch_size, chunk_size, action_dim)."""
        if actions.ndim == 2 and actions.shape[1] > 0:
            actions = actions.unsqueeze(1)
        elif actions.ndim == 1:
            actions = actions.unsqueeze(0).unsqueeze(1)

        if actions.shape[1] < self.chunk_size:
            padding_needed = self.chunk_size - actions.shape[1]
            actions = torch.cat(
                [actions, torch.zeros(actions.shape[0], padding_needed, actions.shape[2], device=actions.device)], dim=1
            )

        return actions

    def _initialize_ensembled_actions(self, actions: torch.Tensor):
        """Initialize ensembled actions on first call."""
        self._ensembled_actions = actions.clone()
        self._ensembled_actions_count = torch.ones(
            (actions.shape[0], self.chunk_size, 1), dtype=torch.long, device=self._ensembled_actions.device
        )

    def _handle_batch_size_mismatch(self, actions: torch.Tensor) -> torch.Tensor:
        """Handle batch size mismatch by resetting and returning first action."""
        current_batch_size = actions.shape[0]
        Logger.warning(
            f"Batch size mismatch for TemporalEnsembler update: "
            f"previous {self.ensembled_actions.shape[0]}, current {current_batch_size}. "
            f"Resetting ensembler."
        )
        self.reset()
        self._initialize_ensembled_actions(actions)
        return self._consume_first_action(actions)

    def _update_existing_actions(self, actions: torch.Tensor):
        """Update existing ensembled actions with new actions."""
        current_ensembled_len = self.ensembled_actions_count.shape[1]
        target_len_for_update = self.chunk_size - 1

        if current_ensembled_len < target_len_for_update:
            num_to_update = min(current_ensembled_len, actions[:, :-1].shape[1])
        else:
            num_to_update = actions[:, :-1].shape[1]

        if num_to_update > 0:
            self._apply_temporal_ensemble(actions, num_to_update)
            self._append_new_action(actions, num_to_update)

    def _apply_temporal_ensemble(self, actions: torch.Tensor, num_to_update: int):
        """Apply temporal ensemble weighting to existing actions."""
        self.ensembled_actions[:, :num_to_update] *= self.ensemble_weights_cumsum[
            self.ensembled_actions_count[:, :num_to_update] - 1
        ]
        self.ensembled_actions[:, :num_to_update] += (
            actions[:, :num_to_update] * self.ensemble_weights[self.ensembled_actions_count[:, :num_to_update]]
        )
        self.ensembled_actions[:, :num_to_update] /= self.ensemble_weights_cumsum[
            self.ensembled_actions_count[:, :num_to_update]
        ]
        self.ensembled_actions_count[:, :num_to_update] = torch.clamp(
            self.ensembled_actions_count[:, :num_to_update] + 1, max=self.chunk_size
        )

    def _append_new_action(self, actions: torch.Tensor, num_to_update: int):
        """Append the last action from new actions to ensembled actions."""
        if actions.shape[1] > 0:
            self.ensembled_actions = torch.cat([self.ensembled_actions[:, :num_to_update], actions[:, -1:]], dim=1)
            new_ensembled_count_item = torch.ones_like(self.ensembled_actions_count[:, -1:])
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count[:, :num_to_update], new_ensembled_count_item], dim=1
            )

    def _consume_first_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Consume and return the first action from ensembled actions."""
        if self.ensembled_actions.shape[1] > 0:
            action = self.ensembled_actions[:, 0]
            self.ensembled_actions = self.ensembled_actions[:, 1:]
            self.ensembled_actions_count = self.ensembled_actions_count[:, 1:]
        else:
            Logger.warning("TemporalEnsembler is empty, cannot consume action. Returning zeros.")
            action = torch.zeros(actions.shape[0], actions.shape[2], device=actions.device)

        return action

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        """Update temporal ensemble with new actions and return the next action."""
        self._prepare_device(actions)
        actions = self._normalize_actions_shape(actions)

        if not self._is_initialized():
            self._initialize_ensembled_actions(actions)
        else:
            current_batch_size = actions.shape[0]
            if self.ensembled_actions.shape[0] != current_batch_size:
                return self._handle_batch_size_mismatch(actions)

            self._update_existing_actions(actions)

        return self._consume_first_action(actions)
