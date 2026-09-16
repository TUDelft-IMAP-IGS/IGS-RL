"""MLP actor-critic agent for PPO.

Uses a simple two-hidden-layer MLP for both the actor and critic.

Implements the **Intent-Based Micro-Stepping (AEC)** interface defined by
:class:`~eos.core.model.Model`:

* :meth:`get_ordering` — Phase 1: learned vessel permutation.
* :meth:`get_action_and_value` — Phase 2: per-vessel action selection
  with masked logits and critic value estimate.
"""

# type: ignore

import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from eos.config import MLPModelConfig
from eos.core.model import Model
from eos.models.action_heads import sample_ordering, select_masked_action

from .utils import layer_init


class MLPAgent(Model):
    """MLP-based actor-critic for AEC micro-stepping.

    Architecture
    ------------
    * **Critic** — ``obs → hidden → hidden → 1``
    * **Actor**  — ``obs → hidden → hidden → action_dim``
      For MultiDiscrete spaces the output is reshaped to
      ``(B, n_vessels, opts_per_vessel)`` so that each vessel has its
      own logit slice.
    * **Ordering head** — ``obs → hidden → n_vessels`` — produces logits
      for the vessel permutation used by Phase 1 of the AEC loop.

    Parameters
    ----------
    envs :
        Vectorised gymnasium environment (used to inspect observation /
        action spaces).
    cfg : MLPModelConfig
        Model hyper-parameters (hidden size).
    """

    def __init__(self, envs, cfg: MLPModelConfig):
        super().__init__()

        action_space = envs.single_action_space
        self.is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)

        if self.is_multidiscrete:
            nvec = np.asarray(action_space.nvec, dtype=int)
            if not np.all(nvec == nvec[0]):
                raise ValueError(
                    "MultiDiscrete action space requires equal per-dimension "
                    "sizes for masking."
                )
            self.n_vessels = int(nvec.shape[0])
            self.opts_per_vessel = int(nvec[0])
            action_dim = self.n_vessels * self.opts_per_vessel
        else:
            self.n_vessels = None
            self.opts_per_vessel = int(action_space.n)
            action_dim = int(action_space.n)

        obs_dim = math.prod(envs.single_observation_space.shape)

        # -- Critic --
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, cfg.hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(cfg.hidden_size, cfg.hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(cfg.hidden_size, 1), std=1.0),
        )

        # -- Actor --
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, cfg.hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(cfg.hidden_size, cfg.hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(cfg.hidden_size, action_dim), std=0.01),
        )

        # -- Ordering head --
        # Always created so that the interface is uniform.  For plain
        # Discrete spaces (single vessel) this produces a trivial 1-logit
        # output that is effectively a no-op.
        n_ord = self.n_vessels if self.n_vessels is not None else 1
        self.ordering_head = nn.Sequential(
            layer_init(nn.Linear(obs_dim, cfg.hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(cfg.hidden_size, n_ord), std=0.01),
        )

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------

    def forward(self, x):
        raise NotImplementedError(
            "MLPAgent does not use a shared backbone.  Call get_value, "
            "get_ordering, or get_action_and_value directly."
        )

    # ------------------------------------------------------------------
    # Public API — AEC Micro-Stepping Interface
    # ------------------------------------------------------------------

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return the critic's state-value estimate, shape ``(B, 1)``."""
        return self.critic(x)

    def get_logits(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute (optionally masked) actor logits.

        For Discrete spaces the output shape is ``(B, action_dim)``.
        For MultiDiscrete spaces the output is reshaped to
        ``(B, n_vessels, opts_per_vessel)``.
        """
        logits = self.actor(x)

        # Reshape for MultiDiscrete
        if self.is_multidiscrete:
            logits = logits.view(logits.shape[0], self.n_vessels, self.opts_per_vessel)

        # Apply static mask
        if action_mask is not None:
            mask = torch.as_tensor(action_mask, device=logits.device)
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            logits = logits.masked_fill(~mask, -1e9)

        return logits

    def get_ordering(
        self,
        x: torch.Tensor,
        vessel_availability: torch.Tensor,
        ordering: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase 1: Determine the optimal sequence for vessels to act.

        Uses the ordering head to sample a permutation of vessels via
        iterative sampling without replacement.  Busy vessels are included
        in the permutation but their log-probability and entropy
        contributions are zeroed out since they skip their turn.

        Parameters
        ----------
        x : Tensor
            The macro-observation before any micro-steps begin.
            Shape ``(B, obs_dim)``.
        vessel_availability : Tensor
            Bool tensor ``(B, n_vessels)`` indicating which vessels are
            currently IDLE.
        ordering : Tensor | None
            Stored ordering for replay during PPO training.
        deterministic : bool
            If True, uses argmax instead of sampling.

        Returns
        -------
        ordering_t : ``(B, n_vessels)``
            The sampled sequence of vessel indices.
        ordering_lp_by_vessel : ``(B, n_vessels)``
            Log-probabilities of the ordering choices, indexed by vessel ID.
        ordering_ent_by_vessel : ``(B, n_vessels)``
            Entropies of the ordering choices, indexed by vessel ID.
        """
        ordering_logits = self.ordering_head(x)  # (B, n_vessels)

        return sample_ordering(
            ordering_logits,
            vessel_availability,
            ordering=ordering,
            deterministic=deterministic,
        )

    def get_action_and_value(
        self,
        x: torch.Tensor,
        vessel_indices: torch.Tensor,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase 2: Select an action for a specific vessel (micro-step).

        Parameters
        ----------
        x : Tensor
            The micro-observation at this exact step. Shape ``(B, obs_dim)``.
        vessel_indices : Tensor
            1D integer tensor ``(B,)`` denoting WHICH vessel is acting.
        action : Tensor | None
            Stored action for replay during PPO training.
        action_mask : Tensor | None
            Boolean mask ``(B, opts_per_vessel)`` for the acting vessel's
            options.
        deterministic : bool
            If True, uses argmax instead of sampling.

        Returns
        -------
        action : ``(B,)``
            The sampled discrete action index.
        logprob : ``(B,)``
            Log-probability of the chosen action.
        entropy : ``(B,)``
            Entropy of the action distribution.
        value : ``(B, 1)``
            Critic value estimate for the current micro-state.
        """
        batch_size = x.shape[0]

        # Critic value
        value = self.critic(x)

        # Actor logits
        if self.is_multidiscrete:
            # Reshape to (B, n_vessels, opts_per_vessel)
            all_logits = self.actor(x).view(
                batch_size, self.n_vessels, self.opts_per_vessel
            )
            # Extract only the branch belonging to the currently acting vessel
            selected_logits = all_logits[torch.arange(batch_size), vessel_indices, :]
        else:
            # Discrete: single "vessel", logits are already (B, opts_per_vessel)
            selected_logits = self.actor(x)

        action, logprob, entropy = select_masked_action(
            selected_logits,
            action_mask=action_mask,
            action=action,
            deterministic=deterministic,
        )

        return action, logprob, entropy, value
