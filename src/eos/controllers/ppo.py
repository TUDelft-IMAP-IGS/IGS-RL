"""PPO action controller.

Wraps a trained model to produce actions for environment interaction.
Handles inference-mode execution, tensor/device management, and
supports the two-phase Micro-Stepping (AEC) architecture.
"""

import torch

from eos.config import PPOControllerConfig
from eos.core.controller import Controller
from eos.core.model import Model


class PPOController(Controller):
    """Decides actions by querying the model in inference mode.

    For the Micro-Stepping architecture, the controller exposes two
    distinct phases:
    1. get_ordering: Determines the sequence in which vessels act.
    2. get_action_and_value: Selects a specific action for a specific vessel.

    Parameters
    ----------
    model : Model
        The actor-critic model to query.
    cfg : PPOControllerConfig
        Controller settings (e.g. whether to act deterministically).
    device : torch.device
        Device tensors should live on.
    """

    model: Model  # narrow from Model | None (base class) — PPO always has a model

    def __init__(self, model: Model, cfg: PPOControllerConfig, device):
        super().__init__(model)
        self.cfg = cfg
        self.device = device

    def get_ordering(
        self, obs, vessel_availability, deterministic=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase 1: Determine the optimal sequence for vessels to act.

        Parameters
        ----------
        obs : array-like
            The macro-observation before any micro-steps begin.
        vessel_availability : array-like
            Boolean array indicating which vessels are currently IDLE.
        deterministic : bool | None
            Override config. When None, falls back to self.cfg.deterministic.

        Returns
        -------
        ordering : Tensor
            The sampled sequence of vessel indices.
        logprob : Tensor
            Log-probabilities of the ordering choices.
        entropy : Tensor
            Entropies of the ordering choices.
        """
        deterministic = (
            deterministic if deterministic is not None else self.cfg.deterministic
        )
        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        avail_tensor = torch.as_tensor(
            vessel_availability, device=self.device, dtype=torch.bool
        )

        with torch.no_grad():
            ordering, logprob, entropy = self.model.get_ordering(
                obs_tensor,
                vessel_availability=avail_tensor,
                deterministic=deterministic,
            )

        return ordering, logprob, entropy

    def get_ordering_and_value(
        self, obs, vessel_availability, deterministic=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase 1 ordering plus the baseline critic value in one pass.

        Equivalent to calling :meth:`get_ordering` and then querying the
        critic on the same macro-observation, but the model computes both
        from a single backbone forward pass (see
        :meth:`eos.models.transformer.TransformerAgent.get_ordering_and_value`).

        Returns
        -------
        ordering, logprob, entropy : as :meth:`get_ordering`.
        value : Tensor ``(B, 1)`` — critic estimate on ``obs``.
        """
        deterministic = (
            deterministic if deterministic is not None else self.cfg.deterministic
        )
        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        avail_tensor = torch.as_tensor(
            vessel_availability, device=self.device, dtype=torch.bool
        )

        with torch.no_grad():
            ordering, logprob, entropy, value = self.model.get_ordering_and_value(
                obs_tensor,
                vessel_availability=avail_tensor,
                deterministic=deterministic,
            )

        return ordering, logprob, entropy, value

    def get_action_and_value(
        self, obs, vessel_indices, mask=None, deterministic=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase 2: Select an action for a specific vessel.

        Parameters
        ----------
        obs : array-like
            The micro-observation at this exact step.
        vessel_indices : array-like
            1D integer array denoting WHICH vessel is acting per environment.
        mask : array-like | None
            Boolean mask for the acting vessel's options.
        deterministic : bool | None
            Override config. When None, falls back to self.cfg.deterministic.

        Returns
        -------
        action : Tensor
            The chosen action indices.
        logprob : Tensor
            Log-probabilities of the actions.
        entropy : Tensor
            Entropies of the action distributions.
        value : Tensor
            Critic value estimates.
        """
        deterministic = (
            deterministic if deterministic is not None else self.cfg.deterministic
        )
        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        v_idx_tensor = torch.as_tensor(
            vessel_indices, device=self.device, dtype=torch.long
        )

        # The mask might already be a tensor from the runner, but ensure it's on the right device
        if mask is not None:
            mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)

        with torch.no_grad():
            action, logprob, entropy, value = self.model.get_action_and_value(
                obs_tensor,
                vessel_indices=v_idx_tensor,
                action_mask=mask,
                deterministic=deterministic,
            )

        return action, logprob, entropy, value
