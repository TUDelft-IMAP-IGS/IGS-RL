"""Random action controller for baseline evaluation.

Implements the same two-phase AEC micro-stepping interface as
:class:`~eos.controllers.ppo.PPOController`, but selects actions
uniformly at random from the valid (masked) options.  This is useful
as a performance baseline and for smoke-testing the environment.

Phase 1 (:meth:`get_ordering`):
    Produces a random permutation of the idle vessels.

Phase 2 (:meth:`get_action_and_value`):
    Samples uniformly from the masked action space for the acting vessel.
    Returns dummy zero tensors for log-probabilities, entropies, and values
    since no learned model is involved.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from eos.config import RandomControllerConfig
from eos.core.controller import Controller
from eos.core.model import Model


class RandomController(Controller):
    """Baseline controller that samples valid actions uniformly at random.

    Implements the two-phase AEC micro-stepping interface
    (:meth:`get_ordering` and :meth:`get_action_and_value`) so it can
    be used as a drop-in replacement for :class:`PPOController` in the
    runner.

    Parameters
    ----------
    model : Model | None
        Ignored — no neural network is used.  Accepted for interface
        compatibility.
    cfg : RandomControllerConfig
        Controller configuration.
    device : torch.device | str | None
        Device for output tensors (default: CPU).
    action_space : gym.Space | None
        The environment's action space.  Required so the controller
        knows the number of vessels and options per vessel.
    num_envs : int
        Number of parallel environments.  One independent RNG is
        created per env slot to enable deterministic per-episode replay.
    """

    def __init__(
        self,
        model: Model | None,
        cfg: RandomControllerConfig,
        device=None,
        action_space: gym.Space | None = None,
        num_envs: int = 1,
    ):
        super().__init__(model)
        self.cfg = cfg
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )

        if action_space is None:
            raise ValueError(
                "RandomController requires an action_space to be provided."
            )

        self.action_space = action_space
        self.is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)

        if self.is_multidiscrete:
            nvec = np.asarray(action_space.nvec, dtype=int)
            self.n_vessels = int(nvec.shape[0])
            self.opts_per_vessel = int(nvec[0])
        else:
            self.n_vessels = 1
            self.opts_per_vessel = int(action_space.n)

        # Per-env RNGs for deterministic replay of individual episodes.
        self._rngs: list[np.random.Generator] = [
            np.random.Generator(np.random.PCG64()) for _ in range(num_envs)
        ]

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_envs(self, seeds):
        """Reseed per-env RNGs. seeds should be array-like of length num_envs."""
        seeds_arr = np.asarray(seeds, dtype=np.uint64)
        for i, s in enumerate(seeds_arr):
            self._rngs[i] = np.random.Generator(np.random.PCG64(int(s)))

    # ------------------------------------------------------------------
    # Phase 1: Ordering
    # ------------------------------------------------------------------

    def get_ordering(
        self, obs, vessel_availability, deterministic=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce a random permutation of vessel indices.

        Idle vessels are shuffled randomly.  Busy vessels are appended
        after all idle vessels in an arbitrary order.

        Parameters
        ----------
        obs : array-like
            The macro-observation (ignored — actions are random).
        vessel_availability : array-like
            Boolean array ``(B, n_vessels)`` indicating which vessels
            are currently IDLE.
        deterministic : bool | None
            Ignored — ordering is always random.

        Returns
        -------
        ordering : Tensor ``(B, n_vessels)``
            Random vessel permutation.
        logprob : Tensor ``(B, n_vessels)``
            Zeros (no learned distribution).
        entropy : Tensor ``(B, n_vessels)``
            Zeros (no learned distribution).
        """
        avail = np.asarray(vessel_availability, dtype=bool)
        if avail.ndim == 1:
            avail = avail[np.newaxis, :]
        batch_size = avail.shape[0]

        orderings = np.zeros((batch_size, self.n_vessels), dtype=np.int64)

        for i in range(batch_size):
            idle = np.where(avail[i])[0].copy()
            busy = np.where(~avail[i])[0].copy()
            self._rngs[i].shuffle(idle)
            self._rngs[i].shuffle(busy)
            orderings[i] = np.concatenate([idle, busy])

        ordering_t = torch.as_tensor(orderings, device=self.device, dtype=torch.long)
        zeros = torch.zeros(batch_size, self.n_vessels, device=self.device)

        return ordering_t, zeros, zeros

    def get_ordering_and_value(
        self, obs, vessel_availability, deterministic=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random ordering plus a zero baseline value.

        Mirrors :class:`~eos.controllers.ppo.PPOController` so the random
        baseline is a drop-in in the collector.  There is no critic, so the
        value is zeros ``(B, 1)``.
        """
        ordering_t, lp, ent = self.get_ordering(
            obs, vessel_availability, deterministic=deterministic
        )
        value = torch.zeros(ordering_t.shape[0], 1, device=self.device)
        return ordering_t, lp, ent, value

    # ------------------------------------------------------------------
    # Phase 2: Action selection
    # ------------------------------------------------------------------

    def get_action_and_value(
        self, obs, vessel_indices, mask=None, deterministic=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random valid action for the acting vessel.

        Parameters
        ----------
        obs : array-like
            The micro-observation (ignored — actions are random).
        vessel_indices : array-like
            1D integer array ``(B,)`` denoting which vessel is acting
            per environment.  Ignored for action selection (masks fully
            determine the valid set).
        mask : array-like | None
            Boolean mask ``(B, opts_per_vessel)`` for the acting vessel's
            valid options.  When ``None``, all options are considered valid.
        deterministic : bool | None
            Ignored — actions are always sampled uniformly.

        Returns
        -------
        action : Tensor ``(B,)``
            The randomly chosen action index.
        logprob : Tensor ``(B,)``
            Zeros (no learned distribution).
        entropy : Tensor ``(B,)``
            Zeros (no learned distribution).
        value : Tensor ``(B, 1)``
            Zeros (no critic).
        """
        obs_arr = np.asarray(obs)
        batch_size = obs_arr.shape[0] if obs_arr.ndim > 1 else 1

        actions = np.zeros(batch_size, dtype=np.int64)

        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.ndim == 1:
                mask_arr = mask_arr[np.newaxis, :]
        else:
            mask_arr = np.ones((batch_size, self.opts_per_vessel), dtype=bool)

        for i in range(batch_size):
            valid = np.flatnonzero(mask_arr[i])
            if valid.size > 0:
                actions[i] = int(self._rngs[i].choice(valid))
            else:
                # Fallback: sample from the full range (should not happen
                # in practice — masks always have at least NOOP enabled).
                actions[i] = int(self._rngs[i].integers(0, self.opts_per_vessel))

        action_t = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        zeros_1d = torch.zeros(batch_size, device=self.device)
        zeros_value = torch.zeros(batch_size, 1, device=self.device)

        return action_t, zeros_1d, zeros_1d, zeros_value
