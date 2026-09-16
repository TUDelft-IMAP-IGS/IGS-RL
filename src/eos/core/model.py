"""Abstract base class for all neural-network models in EOS.

Every model used by the framework (MLP, Transformer, etc.) must subclass
:class:`Model` and implement the required methods.  The interface is
designed around the **Intent-Based Micro-Stepping (AEC)** architecture,
which splits each macro-step into two phases:

1. **Ordering** (:meth:`get_ordering`) — determine the sequence in
   which idle vessels commit actions.
2. **Action selection** (:meth:`get_action_and_value`) — for a single
   vessel, choose an action and return the critic's value estimate.

Concrete models also expose lower-level helpers such as :meth:`get_value`
and :meth:`get_logits` that are used by the learner and controller.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class Model(nn.Module, ABC):
    """Base class for all EOS neural-network models.

    Inherits from both :class:`torch.nn.Module` (for parameter management,
    GPU transfer, serialisation, etc.) and :class:`ABC` (to enforce that
    subclasses implement the required interface).

    Subclasses **must** implement:

    * :meth:`forward` — the backbone forward pass.
    * :meth:`get_value` — critic state-value estimate.
    * :meth:`get_ordering` — Phase 1 of the AEC micro-stepping loop.
    * :meth:`get_action_and_value` — Phase 2 of the AEC micro-stepping loop.
    """

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Run the forward pass of the model.

        Parameters
        ----------
        x : Any
            Model input (typically a batched observation tensor).

        Returns
        -------
        Any
            Model output (architecture-dependent).
        """
        ...

    @abstractmethod
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return the critic's state-value estimate.

        Parameters
        ----------
        x : Tensor
            Batched observation, shape depends on the concrete model.

        Returns
        -------
        Tensor
            Value estimates, shape ``(B, 1)``.
        """
        ...

    @abstractmethod
    def get_ordering(
        self,
        x: torch.Tensor,
        vessel_availability: torch.Tensor,
        ordering: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase 1: Determine the optimal sequence for vessels to act.

        Produces a permutation of vessel indices that defines the order
        in which idle vessels commit their actions during micro-stepping.

        Parameters
        ----------
        x : Tensor
            The macro-observation before any micro-steps begin.
        vessel_availability : Tensor
            Bool tensor ``(B, n_vessels)`` — ``True`` for idle vessels.
        ordering : Tensor | None
            Pre-recorded ordering for training replay.  When ``None``,
            a new ordering is sampled (or chosen greedily).
        deterministic : bool
            If ``True``, use argmax instead of sampling.

        Returns
        -------
        ordering : ``(B, n_vessels)``
            The sampled vessel-index permutation.
        logprob_by_vessel : ``(B, n_vessels)``
            Per-vessel log-probabilities of the ordering choices.
            Busy vessels have log-prob = 0.
        entropy_by_vessel : ``(B, n_vessels)``
            Per-vessel entropies of the ordering distributions.
            Busy vessels have entropy = 0.
        """
        ...

    @abstractmethod
    def get_action_and_value(
        self,
        x: torch.Tensor,
        vessel_indices: torch.Tensor,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase 2: Select an action for a specific vessel (micro-step).

        Given the current micro-observation and the index of the vessel
        that is acting, produce an action, its log-probability, the
        distribution entropy, and the critic value estimate.

        Parameters
        ----------
        x : Tensor
            The micro-observation at this exact step.
        vessel_indices : Tensor
            Integer tensor ``(B,)`` identifying which vessel is acting
            in each environment of the batch.
        action : Tensor | None
            Pre-recorded action for training replay.  When ``None``,
            a new action is sampled (or chosen greedily).
        action_mask : Tensor | None
            Boolean mask ``(B, opts_per_vessel)`` for the acting
            vessel's valid options.
        deterministic : bool
            If ``True``, use argmax instead of sampling.

        Returns
        -------
        action : ``(B,)``
            The chosen action index.
        logprob : ``(B,)``
            Log-probability of the chosen action.
        entropy : ``(B,)``
            Entropy of the action distribution.
        value : ``(B, 1)``
            Critic value estimate for the current micro-state.
        """
        ...

    def get_ordering_and_value(
        self,
        x: torch.Tensor,
        vessel_availability: torch.Tensor,
        ordering: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase 1 ordering plus the baseline critic value.

        Returns the same ordering tuple as :meth:`get_ordering` together
        with the critic's value estimate on the macro-observation ``x``
        (the value used as the macro-step baseline during collection).

        The default implementation simply calls :meth:`get_ordering` and
        :meth:`get_value` separately.  Models with a shared backbone (e.g.
        :class:`~eos.models.transformer.TransformerAgent`) override this to
        compute both from a *single* forward pass, which is purely an
        optimisation: the value is ``critic(forward(x))`` either way, so
        the returned numbers are identical for a deterministic backbone.

        Returns
        -------
        ordering : ``(B, n_vessels)``
        logprob_by_vessel : ``(B, n_vessels)``
        entropy_by_vessel : ``(B, n_vessels)``
        value : ``(B, 1)``
        """
        ordering_t, logprob_by_vessel, entropy_by_vessel = self.get_ordering(
            x, vessel_availability, ordering=ordering, deterministic=deterministic
        )
        value = self.get_value(x)
        return ordering_t, logprob_by_vessel, entropy_by_vessel, value
