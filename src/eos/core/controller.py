"""Abstract base class for action controllers.

A controller wraps a :class:`~eos.core.model.Model` and provides the
inference-time interface used by the :class:`~eos.core.runner.Runner`
to select actions during environment interaction.

Under the **Intent-Based Micro-Stepping (AEC)** architecture, the
controller exposes two phases per macro-step:

1. :meth:`get_ordering` — determine the sequence in which idle vessels
   commit their actions.
2. :meth:`get_action_and_value` — for a single vessel, select an action
   and return the critic's value estimate.

The controller is responsible for device management, deterministic /
stochastic selection policy, and converting raw arrays into tensors
before forwarding them to the model.
"""

from abc import ABC, abstractmethod

import torch

from eos.core.model import Model


class Controller(ABC):
    """Decides actions based on a trained :class:`Model`.

    Subclasses implement the two-phase AEC interface that the runner
    calls during rollout collection:

    * :meth:`get_ordering` — Phase 1 (vessel sequencing).
    * :meth:`get_action_and_value` — Phase 2 (per-vessel action selection).

    Parameters
    ----------
    model : Model | None
        The actor-critic model used to produce action distributions
        and value estimates.  May be ``None`` for model-free controllers
        (e.g. random baselines).
    """

    def __init__(self, model: Model | None):
        self.model = model

    @abstractmethod
    def get_ordering(
        self, obs, vessel_availability, deterministic=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase 1: Determine the optimal sequence for vessels to act.

        Parameters
        ----------
        obs : array-like
            The macro-observation before any micro-steps begin.
        vessel_availability : array-like
            Boolean array ``(B, n_vessels)`` indicating which vessels
            are currently IDLE.
        deterministic : bool | None
            Override the controller's default stochastic/deterministic
            policy.

        Returns
        -------
        ordering : Tensor ``(B, n_vessels)``
            The sampled sequence of vessel indices.
        logprob : Tensor ``(B, n_vessels)``
            Per-vessel log-probabilities of the ordering choices.
        entropy : Tensor ``(B, n_vessels)``
            Per-vessel entropies of the ordering distributions.
        """
        ...

    @abstractmethod
    def get_action_and_value(
        self, obs, vessel_indices, mask=None, deterministic=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase 2: Select an action for a specific vessel.

        Parameters
        ----------
        obs : array-like
            The micro-observation at this exact step.
        vessel_indices : array-like
            1D integer array ``(B,)`` denoting which vessel is acting
            per environment in the batch.
        mask : array-like | None
            Boolean mask ``(B, opts_per_vessel)`` for the acting
            vessel's valid options.
        deterministic : bool | None
            Override the controller's default stochastic/deterministic
            policy.

        Returns
        -------
        action : Tensor ``(B,)``
            The chosen action indices.
        logprob : Tensor ``(B,)``
            Log-probabilities of the actions.
        entropy : Tensor ``(B,)``
            Entropies of the action distributions.
        value : Tensor ``(B, 1)``
            Critic value estimates.
        """
        ...
