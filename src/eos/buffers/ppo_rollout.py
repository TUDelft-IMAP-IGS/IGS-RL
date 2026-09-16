"""PPO rollout buffer for on-policy trajectory storage.

Stores observations, actions, rewards, dones, values, log-probabilities,
and optionally action masks and vessel orderings (for autoregressive
multi-vessel action spaces).  After a rollout is complete, GAE advantages
are computed in-place, and the buffer can be sampled as a flat
:class:`PPOBatch` named-tuple for minibatch training.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from eos.core.replay_buffer import Buffer
from eos.utils.debug import check_nan


class PPOBatch(NamedTuple):
    """Flat batch returned by :meth:`PPORolloutBuffer.sample`.

    All tensors have a leading dimension of ``num_steps * num_envs``.
    ``action_masks``, ``orderings``, ``vessel_availability``, and
    ``per_vessel_obs`` are ``None`` when not applicable.
    """

    obs: torch.Tensor
    logprobs: torch.Tensor
    per_factor_logprobs: torch.Tensor | None
    actions: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    delta_times: torch.Tensor
    action_masks: torch.Tensor | None
    orderings: torch.Tensor | None
    vessel_availability: torch.Tensor | None
    per_vessel_obs: torch.Tensor | None


class PPORolloutBuffer(Buffer):
    """Fixed-size rollout buffer for PPO with optional masking and orderings.

    Parameters
    ----------
    num_steps : int
        Number of environment steps per rollout.
    num_envs : int
        Number of parallel environments.
    obs_shape : tuple
        Shape of a single observation (excluding batch dims).
    actions_shape : tuple
        Shape of a single action (excluding batch dims).
        Empty tuple ``()`` for ``Discrete``; ``(n_vessels,)`` for
        ``MultiDiscrete``.
    device : torch.device
        Device for all internal tensors.
    action_dim : int | None
        Size of a ``Discrete`` action space (used only for documentation /
        future extensions).  ``None`` for ``MultiDiscrete``.
    action_mask_shape : tuple | None
        Shape of a single action mask.  ``None`` disables mask storage
        end-to-end.

        * Discrete: ``(action_dim,)``
        * MultiDiscrete: ``(n_vessels, opts_per_vessel)``
    n_vessels : int | None
        Number of vessels for autoregressive ordering storage.  When
        provided, the buffer allocates an ``orderings`` tensor to record
        the vessel permutation chosen at each step so that PPO can replay
        the same decision path during training.  Should be set whenever
        the action space is ``MultiDiscrete``.
    obs_shape : tuple
        Shape of a single observation (excluding batch dims).  Used to
        allocate ``per_vessel_obs`` storage when ``n_vessels`` is set.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: tuple,
        actions_shape: tuple,
        device: torch.device,
        action_dim: int | None = None,
        action_mask_shape: tuple | None = None,
        n_vessels: int | None = None,
    ):
        self._obs_shape = obs_shape
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.step = 0

        # Core rollout storage
        self.obs = torch.zeros((num_steps, num_envs) + obs_shape, device=device)
        self.actions = torch.zeros((num_steps, num_envs) + actions_shape, device=device)
        # Logprobs are stored as scalars per env per step.  Under the
        # macro-transition architecture the collector produces a single
        # aggregate log-probability that covers the ordering + all
        # per-vessel action choices, so no per-vessel dimension is needed.
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.delta_times = torch.zeros((num_steps, num_envs), device=device)

        # Optional action-mask storage (None when masking is disabled)
        self.masks: torch.Tensor | None = (
            torch.zeros(
                (num_steps, num_envs) + tuple(action_mask_shape),
                dtype=torch.bool,
                device=device,
            )
            if action_mask_shape is not None
            else None
        )

        # Optional vessel-ordering storage for autoregressive action
        # generation.  Stores the permutation chosen at each step so that
        # PPO can replay the same decision path during training.  Allocated
        # when the action space is MultiDiscrete (n_vessels is provided).
        self.orderings: torch.Tensor | None = (
            torch.zeros(
                (num_steps, num_envs, n_vessels),
                dtype=torch.long,
                device=device,
            )
            if n_vessels is not None
            else None
        )

        # Optional vessel-availability storage.  Records the ground-truth
        # idle flags from the environment at collection time so that
        # replay uses the exact same availability instead of deriving it
        # heuristically from the action mask.
        self.vessel_availability: torch.Tensor | None = (
            torch.zeros(
                (num_steps, num_envs, n_vessels),
                dtype=torch.bool,
                device=device,
            )
            if n_vessels is not None
            else None
        )

        # Optional per-vessel observation storage for sequential AEC.
        # Each vessel sees a different observation during collection
        # (reflecting intents registered by earlier vessels in the
        # ordering).  Shape: (num_steps, num_envs, n_vessels, *obs_shape).
        self.per_vessel_obs: torch.Tensor | None = (
            torch.zeros(
                (num_steps, num_envs, n_vessels) + obs_shape,
                device=device,
            )
            if n_vessels is not None
            else None
        )

        # Per-factor log-probs for per-factor PPO clipping.
        # Shape: (num_steps, num_envs, 2 * n_vessels).
        self.per_factor_logprobs: torch.Tensor | None = (
            torch.zeros(
                (num_steps, num_envs, 2 * n_vessels),
                device=device,
            )
            if n_vessels is not None
            else None
        )

        # Pre-allocated during compute_advantages
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Buffer interface
    # ------------------------------------------------------------------

    def add(
        self,
        obs,
        action,
        reward,
        done,
        value,
        logprob,
        delta_time,
        action_mask=None,
        ordering=None,
        vessel_availability=None,
        per_vessel_obs=None,
        per_factor_logprobs=None,
    ) -> None:
        """Store a single transition at the current step index.

        Parameters
        ----------
        obs, action, reward, done, value, logprob, delta_time :
            Core transition data — scalars, arrays, or tensors that will
            be converted to the buffer's device.
        action_mask : array-like | None
            Boolean action mask.  Must be provided iff mask storage is
            enabled (``action_mask_shape`` was set at init).
        ordering : array-like | None
            Vessel ordering permutation.  Must be provided iff ordering
            storage is enabled (``n_vessels`` was set at init).
        per_vessel_obs : array-like | None
            Per-vessel observations from the sequential AEC loop.
            Shape ``(num_envs, n_vessels, *obs_shape)``.

        Raises
        ------
        RuntimeError
            If masks or orderings are provided/missing inconsistently
            with the buffer configuration.
        """
        self.obs[self.step] = torch.as_tensor(obs, device=self.device)
        self.actions[self.step] = torch.as_tensor(action, device=self.device)
        self.logprobs[self.step] = torch.as_tensor(logprob, device=self.device)
        self.rewards[self.step] = torch.as_tensor(reward, device=self.device)
        self.dones[self.step] = torch.as_tensor(done, device=self.device)
        self.values[self.step] = torch.as_tensor(value, device=self.device)
        self.delta_times[self.step] = torch.as_tensor(
            delta_time, device=self.device, dtype=torch.float32
        )

        # ── Debug: catch NaN at storage time ─────────────────────────
        check_nan(self.rewards[self.step], f"buffer.rewards[{self.step}]")
        check_nan(self.values[self.step], f"buffer.values[{self.step}]")
        check_nan(self.delta_times[self.step], f"buffer.delta_times[{self.step}]")

        # ── Action masks ─────────────────────────────────────────────
        self._store_mask(action_mask)

        # ── Vessel orderings ─────────────────────────────────────────
        self._store_ordering(ordering)

        # ── Vessel availability ──────────────────────────────────────
        self._store_vessel_availability(vessel_availability)

        # ── Per-vessel observations ──────────────────────────────────
        self._store_per_vessel_obs(per_vessel_obs)

        # ── Per-factor log-probs ───────────────────────────────────
        if self.per_factor_logprobs is not None and per_factor_logprobs is not None:
            self.per_factor_logprobs[self.step] = torch.as_tensor(
                per_factor_logprobs, device=self.device
            )

        self.step += 1

    def sample(self) -> PPOBatch:
        """Flatten the buffer into a :class:`PPOBatch` for training.

        Must be called after :meth:`compute_advantages`.

        Returns
        -------
        PPOBatch
            Named tuple with all fields flattened to
            ``(num_steps * num_envs, ...)``.
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Call compute_advantages() before sample().")

        b_obs = self.obs.reshape((-1,) + self.obs.shape[2:])
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.actions.shape[2:])
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)
        b_delta_times = self.delta_times.reshape(-1)

        b_masks = (
            self.masks.reshape((-1,) + self.masks.shape[2:])
            if self.masks is not None
            else None
        )
        b_orderings = (
            self.orderings.reshape((-1,) + self.orderings.shape[2:])
            if self.orderings is not None
            else None
        )
        b_vessel_availability = (
            self.vessel_availability.reshape((-1,) + self.vessel_availability.shape[2:])
            if self.vessel_availability is not None
            else None
        )
        b_per_vessel_obs = (
            self.per_vessel_obs.reshape((-1,) + self.per_vessel_obs.shape[2:])
            if self.per_vessel_obs is not None
            else None
        )
        b_per_factor_logprobs = (
            self.per_factor_logprobs.reshape((-1,) + self.per_factor_logprobs.shape[2:])
            if self.per_factor_logprobs is not None
            else None
        )

        return PPOBatch(
            obs=b_obs,
            logprobs=b_logprobs,
            per_factor_logprobs=b_per_factor_logprobs,
            actions=b_actions,
            advantages=b_advantages,
            returns=b_returns,
            values=b_values,
            delta_times=b_delta_times,
            action_masks=b_masks,
            orderings=b_orderings,
            vessel_availability=b_vessel_availability,
            per_vessel_obs=b_per_vessel_obs,
        )

    def reset(self) -> None:
        """Reset the step counter so the buffer can be re-used."""
        self.step = 0

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_advantages(
        self,
        next_value: torch.Tensor,
        next_done,
        beta: float,
        gae_lambda: float,
    ) -> None:
        """Compute Generalised Advantage Estimation (GAE) in-place.

        After this call, ``self.advantages`` and ``self.returns`` are
        populated and :meth:`sample` can be called.

        Parameters
        ----------
        next_value : Tensor
            Value estimate for the observation *after* the last stored step.
        next_done : array-like
            Done flag for the observation after the last stored step.
        gamma : float
            Discount factor.
        gae_lambda : float
            GAE lambda for bias–variance trade-off.
        """
        self.advantages = torch.zeros_like(self.rewards, device=self.device)
        # Ensure next_done is a tensor on the correct device so that
        # downstream arithmetic stays on-device (avoids implicit
        # tensor/numpy mixing).
        next_done = torch.as_tensor(next_done, device=self.device, dtype=torch.float32)
        lastgaelam = 0.0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]

            # Compute step-specific dynamic gamma
            gamma_t = torch.exp(-beta * self.delta_times[t])

            delta = (
                self.rewards[t]
                + gamma_t * nextvalues * nextnonterminal
                - self.values[t]
            )
            self.advantages[t] = lastgaelam = (
                delta + gamma_t * gae_lambda * nextnonterminal * lastgaelam
            )
        self.returns = self.advantages + self.values

        # ── Debug: catch NaN in computed advantages/returns ───────────
        check_nan(self.advantages, "buffer.advantages_post_gae")
        check_nan(self.returns, "buffer.returns_post_gae")

        return

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _store_mask(self, action_mask) -> None:
        """Validate and store an action mask at the current step."""
        # When mask storage is disabled, nothing to do.
        if self.masks is None:
            if action_mask is not None:
                raise RuntimeError(
                    "Received action_mask but buffer mask storage is "
                    "disabled (action_mask_shape was None at init)."
                )
            return

        if action_mask is None:
            raise RuntimeError(
                "Action masking is enabled but action_mask was None. "
                "This should never happen when env.use_action_masks=true."
            )

        mask_tensor = torch.as_tensor(action_mask, device=self.device)
        if mask_tensor.dtype != torch.bool:
            mask_tensor = mask_tensor.bool()

        self.masks[self.step] = mask_tensor

    def _store_ordering(self, ordering) -> None:
        """Validate and store a vessel ordering at the current step."""
        if self.orderings is None:
            if ordering is not None:
                raise RuntimeError(
                    "Received ordering but buffer ordering storage is "
                    "disabled.  This indicates a mismatch between the "
                    "action space (MultiDiscrete) and buffer configuration."
                )
            return

        if ordering is None:
            raise RuntimeError(
                "Ordering storage is enabled but ordering was None. "
                "The model should always return an ordering for "
                "MultiDiscrete action spaces."
            )

        self.orderings[self.step] = torch.as_tensor(ordering, device=self.device).long()

    def _store_vessel_availability(self, vessel_availability) -> None:
        """Validate and store vessel availability at the current step."""
        if self.vessel_availability is None:
            if vessel_availability is not None:
                raise RuntimeError(
                    "Received vessel_availability but buffer storage is "
                    "disabled (n_vessels was None at init)."
                )
            return

        if vessel_availability is None:
            raise RuntimeError(
                "Vessel availability storage is enabled but "
                "vessel_availability was None."
            )

        self.vessel_availability[self.step] = torch.as_tensor(
            vessel_availability, device=self.device
        ).bool()

    def _store_per_vessel_obs(self, per_vessel_obs) -> None:
        """Validate and store per-vessel observations at the current step."""
        if self.per_vessel_obs is None:
            # Storage disabled — silently ignore (per_vessel_obs may be
            # None for Discrete action spaces).
            return

        if per_vessel_obs is None:
            raise RuntimeError(
                "Per-vessel observation storage is enabled but per_vessel_obs was None."
            )

        self.per_vessel_obs[self.step] = torch.as_tensor(
            per_vessel_obs, device=self.device
        )
