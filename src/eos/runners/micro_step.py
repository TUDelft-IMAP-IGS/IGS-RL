"""Shared AEC micro-stepping rollout logic.

This module implements the Intent-Based Micro-Stepping rollout loop that
is shared across PPO training, random baselines, and evaluation scripts.

The core loop for one "macro-step" (decision epoch) is:

1. Extract ``vessel_availability`` and ``action_masks`` from the cached
   info dict (piggybacked on the previous ``step()`` or ``reset()`` IPC
   payload — zero extra round-trips).
2. Call ``controller.get_ordering_and_value()`` to determine the vessel
   sequence and compute the critic baseline on the macro-state in one pass.
3. For each idle vessel in the ordering:
   a. Slice the acting vessel's mask from the cached full masks.
   b. Call ``controller.get_action_and_value()`` for that vessel.
   c. Construct a joint action (all NOOP except this vessel) and call
      ``env.micro_step()`` to register the intent — this also returns
      the updated full mask reflecting the newly queued PENDING activity.
   d. Use the returned mask for subsequent vessels (no extra IPC call).
4. After all idle vessels have committed, call ``step()`` on the
   unwrapped env to advance the DES clock and collect the real reward.
5. This produces one complete transition (macro-obs → reward → next-obs).
   The info dict from ``step()`` is cached for the next macro-step.

The module is designed so that callers (PPORunner, RandomExperiment, eval.py)
can use :class:`MicroStepCollector` without duplicating the intricate
vessel-iteration and env-interaction logic.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from loguru import logger

from eos.core.controller import Controller
from eos.utils.debug import check_nan
from eos.utils.profiling import StepTimer

# ---------------------------------------------------------------------------
# Data structures returned by the collector
# ---------------------------------------------------------------------------


@dataclass
class MacroStepResult:
    """Batch-level result from a single macro-step across all environments.

    All array fields have a leading ``(num_envs,)`` batch dimension.

    Attributes
    ----------
    next_obs : np.ndarray or dict
        Observation after the DES advance, shape ``(B, ...)``.
    rewards : np.ndarray
        Scalar rewards from advancing the DES, shape ``(B,)``.
    terminated : np.ndarray
        Whether each env episode terminated, shape ``(B,)``.
    truncated : np.ndarray
        Whether each env episode was truncated, shape ``(B,)``.
    infos : dict
        Vectorised info dict from ``envs.step()``.
    orderings : np.ndarray
        Vessel orderings used, shape ``(B, n_vessels)``.
    joint_actions : np.ndarray
        Full joint action arrays, shape ``(B, n_vessels)``.
    agg_logprobs : np.ndarray
        Aggregate log-probs (ordering + action) per env, shape ``(B,)``.
    per_factor_logprobs : np.ndarray or None
        Per-factor log-probs for per-factor PPO clipping,
        shape ``(B, 2 * n_vessels)``.  First ``n_vessels`` columns are
        ordering log-probs; last ``n_vessels`` are action log-probs.
        ``None`` for single-vessel (Discrete) spaces.
    agg_entropies : np.ndarray
        Aggregate entropies (ordering + action) per env, shape ``(B,)``.
    first_values : np.ndarray
        Critic value from the first micro-step per env, shape ``(B,)``.
    vessel_availability : np.ndarray
        Boolean idle flags, shape ``(B, n_vessels)``.
    per_vessel_masks : np.ndarray or None
        Per-vessel action masks captured at decision time,
        shape ``(B, n_vessels, opts_per_vessel)``.  Each vessel's slice
        reflects the mask that was active when that vessel actually acted.
        ``None`` when masking is disabled.
    per_vessel_obs : np.ndarray or None
        Per-vessel observations captured at decision time,
        shape ``(B, n_vessels, *obs_shape)``.  Each vessel's slice is the
        observation that was *actually* presented to the model when that
        vessel acted — reflecting intents registered by earlier vessels in
        the AEC ordering.  ``None`` for single-vessel (Discrete) spaces.
    delta_times_hours: np.ndarray
        Time elapsed in the simulator during the macro step.
    """

    next_obs: Any
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: dict
    orderings: np.ndarray
    joint_actions: np.ndarray
    agg_logprobs: np.ndarray
    per_factor_logprobs: np.ndarray | None
    agg_entropies: np.ndarray
    first_values: np.ndarray
    vessel_availability: np.ndarray
    per_vessel_masks: np.ndarray | None
    per_vessel_obs: np.ndarray | None
    delta_times_hours: np.ndarray


# ---------------------------------------------------------------------------
# Helper: env interaction primitives
# ---------------------------------------------------------------------------


def fetch_action_masks(envs: gym.vector.VectorEnv) -> np.ndarray:
    """Query the vectorised environment for action masks.

    .. note::

        Prefer extracting masks from the info dict returned by
        ``step()`` / ``reset()`` (see :func:`extract_masks_from_info`)
        to avoid a dedicated IPC round-trip.  This function is kept as a
        fallback for environments that do not pack masks into info.

    Returns
    -------
    np.ndarray
        Boolean mask, shape ``(num_envs, n_vessels, opts_per_vessel)``.
    """
    masks = envs.call("action_masks")
    return np.stack(masks).astype(bool)


def fetch_vessel_availability(
    envs: gym.vector.VectorEnv, action_mask: np.ndarray | None = None
) -> np.ndarray:
    """Get vessel availability (idle flags) from the vectorised env.

    .. note::

        Prefer extracting availability from the info dict returned by
        ``step()`` / ``reset()`` (see :func:`extract_availability_from_info`)
        to avoid a dedicated IPC round-trip.  This function is kept as a
        fallback for environments that do not pack availability into info.

    Returns
    -------
    np.ndarray
        Boolean array, shape ``(num_envs, n_vessels)``.
    """
    try:
        avail_list = envs.call("vessel_availability")
        return np.stack(avail_list).astype(bool)
    except Exception as e:
        logger.error(f"Failed to get vessel_availability: {e}")
        raise


def extract_masks_from_info(infos: dict, num_envs: int) -> np.ndarray | None:
    """Extract stacked action masks from a vectorised info dict.

    The ``JointDiscreteActionWrapper`` packs ``action_masks`` into the
    info dict returned by ``step()`` and ``reset()``.  Gymnasium's
    vectorised env automatically stacks per-sub-env numpy arrays into a
    batch array of shape ``(num_envs, ...)``.

    Returns ``None`` when the key is absent (env does not support it).
    """
    masks = infos.get("action_masks")
    if masks is None:
        return None
    masks = np.asarray(masks)
    if masks.shape[0] != num_envs:
        return None
    return masks.astype(bool)


def extract_availability_from_info(infos: dict, num_envs: int) -> np.ndarray | None:
    """Extract stacked vessel availability from a vectorised info dict.

    Returns ``None`` when the key is absent (env does not support it).
    """
    avail = infos.get("vessel_availability")
    if avail is None:
        return None
    avail = np.asarray(avail)
    if avail.shape[0] != num_envs:
        return None
    return avail.astype(bool)


# ---------------------------------------------------------------------------
# Core micro-stepping collector
# ---------------------------------------------------------------------------


class MicroStepCollector:
    """Executes the AEC micro-stepping loop over a vectorised environment.

    This is the shared rollout engine used by PPORunner, RandomExperiment,
    and eval.py.  It handles the two-phase controller protocol:

    * Phase 1 — vessel ordering (``controller.get_ordering``).
    * Phase 2 — per-vessel action selection (``controller.get_action_and_value``).

    After all idle vessels have registered their intents via ``env.micro_step()``,
    the DES is advanced with ``env.step()`` and the real reward is
    collected.

    Parameters
    ----------
    envs : gym.vector.VectorEnv
        A vectorised gymnasium environment (``SyncVectorEnv``).
    controller : Controller
        Any controller implementing the two-phase AEC interface.
    use_action_masks : bool
        Whether to query action masks from the environment.
    deterministic : bool
        Whether to use deterministic (greedy) action selection.
    """

    def __init__(
        self,
        envs: gym.vector.VectorEnv,
        controller: Controller,
        use_action_masks: bool = True,
        deterministic: bool = False,
    ) -> None:
        self.envs = envs
        self.controller = controller
        self.use_action_masks = use_action_masks
        self.deterministic = deterministic

        # Infer env dimensions
        action_space = envs.single_action_space
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            nvec = np.asarray(action_space.nvec, dtype=int)
            self.n_vessels = int(nvec.shape[0])
            self.opts_per_vessel = int(nvec[0])
            self.is_multidiscrete = True
        elif isinstance(action_space, gym.spaces.Discrete):
            self.n_vessels = 1
            self.opts_per_vessel = int(action_space.n)
            self.is_multidiscrete = False
        else:
            raise TypeError(
                f"MicroStepCollector requires Discrete or MultiDiscrete "
                f"action space, got {type(action_space)}"
            )

        self.num_envs = envs.num_envs
        self.noop_index = 0

        # Cached info dict from the most recent ``step()`` or ``reset()``.
        # When available, ``vessel_availability`` and ``action_masks`` are
        # extracted from here instead of making separate IPC calls.
        self._cached_infos: dict | None = None

        # Fine-grained timer that accumulates across macro-steps.
        # The caller (PPORunner / eval) can read ``.timer`` for summaries
        # and call ``.timer.reset()`` between rollouts.
        self.timer = StepTimer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def macro_step(
        self,
        obs: np.ndarray,
        deterministic: bool | None = None,
    ) -> MacroStepResult:
        """Execute one full macro-step for all environments in the batch.

        Parameters
        ----------
        obs : np.ndarray
            Current observations, shape ``(num_envs, ...)``.
        deterministic : bool | None
            Override the collector's default deterministic setting.

        Returns
        -------
        MacroStepResult
            Batch-level result containing next_obs, rewards, masks, etc.
        """
        det = deterministic if deterministic is not None else self.deterministic
        t = self.timer  # alias for brevity

        # Freeze the macro-observation — this is the obs at t=0 before
        # any micro-steps.  It is used for ordering (Phase 1).  Each
        # subsequent vessel will see a *updated* obs reflecting intents
        # registered by earlier vessels (sequential AEC).
        macro_obs = _copy_obs(obs)

        # ── 1. Query vessel availability ──────────────────────────────
        # Try the zero-IPC path first: extract from cached info dict
        # piggybacked on the previous step()/reset() payload.
        with t.phase("availability"):
            vessel_availability = None
            if self._cached_infos is not None:
                vessel_availability = extract_availability_from_info(
                    self._cached_infos, self.num_envs
                )
            if vessel_availability is None:
                # Fallback: dedicated IPC round-trip (first step or env
                # doesn't pack availability into info).
                vessel_availability = fetch_vessel_availability(self.envs)

        # ── 2. Phase 1: vessel ordering + baseline value (one pass) ──
        # The ordering head and the critic both read the global token of
        # the same backbone forward on the pristine macro_obs, so the
        # controller computes them together in a single pass.  Querying the
        # critic on macro_obs guarantees a baseline value for ALL envs,
        # even those whose vessels are all currently busy.
        with t.phase("ordering"):
            (
                ordering,
                ordering_logprob_by_vessel,
                ordering_entropy_by_vessel,
                val_t,
            ) = self.controller.get_ordering_and_value(
                macro_obs, vessel_availability, deterministic=det
            )
            first_values = _to_numpy(val_t).ravel()

        # Convert to numpy for indexing
        ordering_np = _to_numpy(ordering)  # (B, n_vessels)
        ordering_lp_by_vessel_np = _to_numpy(
            ordering_logprob_by_vessel
        )  # (B, n_vessels)
        ordering_ent_by_vessel_np = _to_numpy(
            ordering_entropy_by_vessel
        )  # (B, n_vessels)

        # ── 3. Phase 2: Micro-step each idle vessel ──────────────────
        # Track state per env
        joint_actions = np.zeros((self.num_envs, self.n_vessels), dtype=np.int64)

        # Ordering logprobs/entropies are already indexed by vessel ID
        # and zeroed for busy vessels by sample_ordering — sum once.
        agg_logprobs = ordering_lp_by_vessel_np.sum(axis=1)  # (B,)
        agg_entropies = ordering_ent_by_vessel_np.sum(axis=1)  # (B,)

        # Per-factor logprobs for per-factor PPO clipping.
        # Shape: (B, 2 * n_vessels) — [ordering_lps | action_lps].
        per_factor_logprobs = (
            np.zeros((self.num_envs, 2 * self.n_vessels), dtype=np.float64)
            if self.is_multidiscrete
            else None
        )
        if per_factor_logprobs is not None:
            per_factor_logprobs[:, : self.n_vessels] = ordering_lp_by_vessel_np

        # Per-vessel masks captured at decision time for correct PPO replay
        per_vessel_masks: np.ndarray | None = None
        if self.use_action_masks and self.is_multidiscrete:
            per_vessel_masks = np.ones(
                (self.num_envs, self.n_vessels, self.opts_per_vessel), dtype=bool
            )

            # Busy vessels get NOOP-only mask
            busy = ~vessel_availability  # (B, n_vessels)
            per_vessel_masks[busy] = False
            per_vessel_masks[:, :, self.noop_index][busy] = True

        # Per-vessel observations — each vessel sees the state *after*
        # earlier vessels' intents have been registered.  The ordering
        # head sees macro_obs (t=0); vessel 1 also sees macro_obs;
        # vessel 2 sees the obs after vessel 1's intent, etc.
        assert isinstance(macro_obs, np.ndarray), (
            "Sequential AEC requires array observations, got "
            f"{type(macro_obs).__name__}"
        )
        obs_shape = macro_obs.shape[1:]  # per-env obs shape
        per_vessel_obs: np.ndarray | None = None
        if self.is_multidiscrete:
            per_vessel_obs = np.zeros(
                (self.num_envs, self.n_vessels, *obs_shape), dtype=macro_obs.dtype
            )
            # Busy vessels get the macro_obs (they won't be replayed but
            # the slot needs a valid tensor for batched operations).
            for v in range(self.n_vessels):
                per_vessel_obs[:, v] = macro_obs

        # The observation currently visible to the model.  Starts as
        # macro_obs and is updated after each micro-step with the rebuilt
        # observation reflecting newly registered intents.
        current_obs = _copy_obs(macro_obs)

        # Count real decisions for entropy normalisation
        n_decisions = np.zeros(self.num_envs, dtype=np.float64)

        # Fetch the full action mask once before the loop.
        # This reflects the state before any intents are registered.
        # Try the zero-IPC path first (from cached info), then fall back.
        with t.phase("initial_masks"):
            cached_full_masks: np.ndarray | None = None
            if self.use_action_masks and self.is_multidiscrete:
                if self._cached_infos is not None:
                    cached_full_masks = extract_masks_from_info(
                        self._cached_infos, self.num_envs
                    )
                if cached_full_masks is None:
                    cached_full_masks = fetch_action_masks(
                        self.envs
                    )  # (B, n_vessels, opts)

        env_indices = np.arange(self.num_envs)

        # Iterate through ordering positions
        for pos in range(self.n_vessels):
            # Gather which vessel index is at this position for each env
            vessel_indices = ordering_np[:, pos]  # (B,)

            # Check which envs have an idle vessel at this position
            idle_at_pos = vessel_availability[env_indices, vessel_indices]

            if not idle_at_pos.any():
                # No env has an idle vessel at this ordering position — skip
                continue

            # Slice the acting vessel's mask from the cached full masks
            with t.phase("mask_slice"):
                vessel_mask: np.ndarray | None = None
                if cached_full_masks is not None:
                    vessel_mask = cached_full_masks[
                        env_indices, vessel_indices
                    ]  # (B, opts_per_vessel)

                    # Capture the mask at decision time for PPO replay
                    idle_envs = np.where(idle_at_pos)[0]
                    v_idxs = vessel_indices[idle_envs]
                    per_vessel_masks[idle_envs, v_idxs] = vessel_mask[idle_envs]

            # Capture the observation each vessel actually sees.
            if per_vessel_obs is not None:
                idle_envs_obs = np.where(idle_at_pos)[0]
                v_idxs_obs = vessel_indices[idle_envs_obs]
                per_vessel_obs[idle_envs_obs, v_idxs_obs] = current_obs[idle_envs_obs]

            # Call controller phase 2
            with t.phase("inference"):
                v_idx_tensor = torch.as_tensor(vessel_indices, dtype=torch.long)
                action_t, lp_t, ent_t, val_t = self.controller.get_action_and_value(
                    current_obs, v_idx_tensor, mask=vessel_mask, deterministic=det
                )

                action_np = _to_numpy(action_t)  # (B,)
                lp_np = _to_numpy(lp_t)  # (B,)
                ent_np = _to_numpy(ent_t)  # (B,)
                val_np = _to_numpy(val_t).ravel()  # (B,)

            # Record decisions and build the joint action for env.step()
            with t.phase("action_bookkeeping"):
                step_action = np.zeros((self.num_envs, self.n_vessels), dtype=np.int64)

                # Action recording + logprob/entropy accumulation
                idle_envs = np.where(idle_at_pos)[0]
                v_idxs = vessel_indices[idle_envs]

                joint_actions[idle_envs, v_idxs] = action_np[idle_envs]
                step_action[idle_envs, v_idxs] = action_np[idle_envs]

                # Accumulate action logprob/entropy
                agg_logprobs[idle_envs] += lp_np[idle_envs]
                if per_factor_logprobs is not None:
                    per_factor_logprobs[idle_envs, self.n_vessels + v_idxs] = lp_np[
                        idle_envs
                    ]
                agg_entropies[idle_envs] += ent_np[idle_envs]
                n_decisions[idle_envs] += 1.0

            # Register intent and get updated obs + masks in one IPC call.
            # micro_step now returns (obs, mask) from each sub-env,
            # reflecting the newly queued PENDING activity.
            #
            # IMPORTANT: Inactive envs (idle_at_pos == False) return
            # dummy values to avoid rebuilding the reservation system
            # and observation.  We must NOT let those dummies overwrite
            # the real cached state.
            with t.phase("micro_step_ipc"):
                raw_results = self.envs.call("micro_step", step_action, idle_at_pos)

            with t.phase("mask_update"):
                active_envs = np.where(idle_at_pos)[0]
                if len(active_envs) > 0:
                    # Unpack (obs, mask) tuples from each sub-env
                    new_obs_list = [r[0] for r in raw_results]
                    new_masks_list = [r[1] for r in raw_results]

                    new_obs = np.stack(new_obs_list)
                    # Update current_obs for active envs so the next
                    # vessel in the ordering sees the updated state.
                    current_obs[active_envs] = new_obs[active_envs]

                    if cached_full_masks is not None:
                        new_masks = np.stack(new_masks_list).astype(bool)
                        cached_full_masks[active_envs] = new_masks[active_envs]

        # ── 4. Advance the DES ────────────────────────────────────────
        with t.phase("env_step"):
            dummy_action = np.zeros((self.num_envs, self.n_vessels), dtype=np.int64)
            (
                next_obs,
                rewards,
                terminated,
                truncated,
                yield_infos,
            ) = self.envs.step(dummy_action)

        dt_val = yield_infos.get("delta_time_hours")
        if dt_val is None:
            # Safety net: key absent (e.g. all envs auto-reset simultaneously).
            delta_times_hours = np.zeros(self.num_envs, dtype=np.float64)
        else:
            delta_times_hours = np.asarray(dt_val, dtype=np.float64)

        # ── Debug: catch NaN immediately after env.step() ─────────────
        check_nan(rewards, "micro_step.rewards_from_env")
        check_nan(delta_times_hours, "micro_step.delta_times_hours")
        check_nan(first_values, "micro_step.first_values")

        # Cache the info dict so the *next* macro-step can extract
        # vessel_availability and action_masks without extra IPC calls.
        self._cached_infos = yield_infos

        # ── 5. Build batch-level result ───────────────────────────────
        return MacroStepResult(
            next_obs=next_obs,
            rewards=rewards.astype(np.float64),
            terminated=terminated.astype(bool),
            truncated=truncated.astype(bool),
            infos=yield_infos,
            orderings=ordering_np,
            joint_actions=joint_actions,
            agg_logprobs=agg_logprobs,
            per_factor_logprobs=per_factor_logprobs,
            agg_entropies=agg_entropies,
            first_values=first_values,
            vessel_availability=vessel_availability,
            per_vessel_masks=per_vessel_masks,
            per_vessel_obs=per_vessel_obs,
            delta_times_hours=delta_times_hours,
        )

    def collect_rollout(
        self,
        initial_obs: np.ndarray,
        num_steps: int,
        deterministic: bool | None = None,
    ):
        """Collect ``num_steps`` macro-transitions across all envs.

        This is the main entry point for PPO-style fixed-length rollouts.

        Parameters
        ----------
        initial_obs : np.ndarray
            Starting observations, shape ``(num_envs, ...)``.
        num_steps : int
            Number of macro-steps to collect.
        deterministic : bool | None
            Override the collector's default deterministic setting.

        Yields
        ------
        step_index : int
            The macro-step index (0-based).
        result : MacroStepResult
            Batch-level result for this macro-step.
        """
        obs = initial_obs
        for t in range(num_steps):
            result = self.macro_step(obs, deterministic=deterministic)
            obs = result.next_obs
            yield t, result

    def collect_episode(
        self,
        seed: int | None = None,
        deterministic: bool | None = None,
        max_steps: int | None = None,
        inventory_tracker: list | None = None,
    ) -> tuple[list[MacroStepResult], np.ndarray]:
        """Collect a full episode (until all envs are done).

        Useful for evaluation and the random baseline.

        Parameters
        ----------
        seed : int | None
            Seed for environment reset.
        deterministic : bool | None
            Override the collector's default deterministic setting.
        max_steps : int | None
            Safety limit on the number of macro-steps.
        inventory_tracker : list | None
            If provided, inventory snapshots are appended here after each
            macro-step (only for env 0).

        Returns
        -------
        all_results : list[MacroStepResult]
            One result per macro-step.
        final_obs : np.ndarray
            The final observations after the episode ends.
        """
        reset_kwargs: dict = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        obs, reset_infos = self.envs.reset(**reset_kwargs)
        # Cache reset info so the first macro-step can extract
        # masks / availability without separate IPC calls.
        self._cached_infos = reset_infos

        # Optional: capture initial inventory
        if inventory_tracker is not None:
            self._capture_inventory(inventory_tracker)

        all_results: list[MacroStepResult] = []
        done = np.zeros(self.num_envs, dtype=bool)
        step_count = 0

        while not done.all():
            if max_steps is not None and step_count >= max_steps:
                break

            result = self.macro_step(obs, deterministic=deterministic)
            obs = result.next_obs
            all_results.append(result)

            done = np.logical_or(
                done, np.logical_or(result.terminated, result.truncated)
            )
            step_count += 1

            if inventory_tracker is not None:
                self._capture_inventory(inventory_tracker)

        return all_results, obs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _capture_inventory(self, history: list) -> None:
        """Capture inventory snapshot for env 0 (evaluation convenience)."""
        try:
            snapshots = self.envs.call("inventory_snapshot")
            snap = snapshots[0]
            if snap is not None:
                history.append(snap)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _to_numpy(t) -> np.ndarray:
    """Convert a tensor or array-like to a numpy array."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _copy_obs(obs):
    """Deep-copy an observation (handles dict obs spaces)."""
    if isinstance(obs, dict):
        return {k: np.array(v, copy=True) for k, v in obs.items()}
    return np.array(obs, copy=True)


def _split_vectorized_info(infos: dict, num_envs: int) -> list[dict]:
    """Split a vectorized info dict into per-env dicts.

    Gymnasium's ``VectorEnv._add_info`` adds a boolean mask key ``_<key>``
    for every ``<key>`` in the per-env info dict.  We skip those mask
    entries so they don't pollute downstream consumers.
    """
    result = [{} for _ in range(num_envs)]
    for key, value in infos.items():
        # Skip Gymnasium's per-key boolean mask entries (e.g. "_reward_components")
        if key.startswith("_") and key[1:] in infos:
            continue
        if isinstance(value, (np.ndarray, list, tuple)) and len(value) == num_envs:
            for i in range(num_envs):
                result[i][key] = value[i]
        elif isinstance(value, dict):
            # Recursively split nested dicts (e.g. "episode" → {"r": [...], "l": [...]})
            sub_split = _split_vectorized_info(value, num_envs)
            for i in range(num_envs):
                result[i][key] = sub_split[i]
        else:
            # Scalar or non-array — copy to all envs
            for i in range(num_envs):
                result[i][key] = value
    return result


# ---------------------------------------------------------------------------
# High-level episode runner for eval / random baseline
# ---------------------------------------------------------------------------


def run_evaluation_episode(
    envs: gym.vector.VectorEnv,
    controller: Controller,
    *,
    seed: int = 0,
    num_episodes: int = 1,
    deterministic: bool = True,
    use_action_masks: bool = True,
    capture_inventory: bool = True,
    trace_actions: bool = True,
    max_steps_per_episode: int | None = None,
) -> dict:
    """Run evaluation episodes using the micro-stepping loop.

    This is a convenience function for ``eval.py`` and the random baseline
    that wraps :class:`MicroStepCollector` and produces the standard
    evaluation results dict.

    Parameters
    ----------
    envs : gym.vector.VectorEnv
        Vectorised environment (typically with ``num_envs=1`` for eval).
    controller : Controller
        The controller to evaluate.
    seed : int
        Base seed for episode resets.
    num_episodes : int
        Number of episodes to run.
    deterministic : bool
        Whether to use deterministic action selection.
    use_action_masks : bool
        Whether to query action masks from the environment.
    capture_inventory : bool
        Whether to capture inventory snapshots.
    trace_actions : bool
        Whether to collect action trace descriptions.
    max_steps_per_episode : int | None
        Safety limit on macro-steps per episode.

    Returns
    -------
    dict
        Standard evaluation results with keys:
        ``"episodes"``, ``"summary"``, ``"action_trace"``,
        ``"inventory_history"``, ``"env"``.
    """
    import time

    collector = MicroStepCollector(
        envs,
        controller,
        use_action_masks=use_action_masks,
        deterministic=deterministic,
    )

    episodes_data: list[dict] = []
    all_action_trace: list[dict] = []
    best_return = -float("inf")
    best_inventory_history: list[dict] = []

    for ep_idx in range(num_episodes):
        ep_start = time.time()
        inv_history: list[dict] = [] if capture_inventory else []

        reward_breakdown = {
            "total_reward": 0.0,
            "milestone_term_total": 0.0,
            "completion_bonus_total": 0.0,
            "reward_cost_term_total": 0.0,
        }
        reward_cost_components_total: dict[str, float] = defaultdict(float)
        operational_costs: dict[str, float] = defaultdict(float)
        operational_metrics = {
            "elapsed_hours": 0.0,
            "travel_hours_total": 0.0,
            "travel_hours_by_vessel": defaultdict(float),
            "travel_cost_by_vessel": defaultdict(float),
            "storage_unit_hours_total": 0.0,
            "storage_unit_hours_by_site": defaultdict(float),
            "storage_cost_by_site": defaultdict(float),
            "storage_unit_hours_by_site_resource": defaultdict(float),
        }

        reset_kwargs = {"seed": seed + ep_idx}
        obs, reset_infos = envs.reset(**reset_kwargs)
        # Seed the collector's info cache so the first macro-step of
        # each episode can extract masks/availability without extra IPC.
        collector._cached_infos = reset_infos

        if capture_inventory:
            collector._capture_inventory(inv_history)

        done = np.zeros(envs.num_envs, dtype=bool)
        step_count = 0
        ep_reward_accum = np.zeros(envs.num_envs, dtype=np.float64)
        last_result: MacroStepResult | None = None

        while not done.all():
            if (
                max_steps_per_episode is not None
                and step_count >= max_steps_per_episode
            ):
                break

            result = collector.macro_step(obs, deterministic=deterministic)
            obs = result.next_obs
            last_result = result

            per_env_infos = _split_vectorized_info(result.infos, envs.num_envs)

            # Collect action trace for env 0.
            # We call describe_action_request *after* env.step() so
            # vessel_location reflects the post-step state, but the
            # action text (type, destination, resource) is decoded
            # from the joint_actions array and is always correct.
            #
            # Rows are reordered to reflect the AEC acting order so the
            # trace shows which vessel committed first.
            if trace_actions:
                try:
                    raw = envs.call("describe_action_request", result.joint_actions[0])
                    rows = raw[0]
                    if rows:
                        rows = rows if isinstance(rows, list) else [rows]

                        # Build a vessel-name → row lookup
                        row_by_vessel: dict[str, dict] = {r["vessel"]: r for r in rows}

                        # Enrich with elapsed time from post-step info
                        env0_info = per_env_infos[0]
                        elapsed_hours = float(env0_info.get("elapsed_time_hours", 0.0))
                        delta_hours = float(env0_info.get("delta_time_hours", 0.0))

                        # Determine acting order from the AEC ordering.
                        # result.orderings[0] gives vessel indices in the
                        # order they were sequenced; availability tells us
                        # which actually acted (idle vessels only).
                        ordering = result.orderings[0]  # (n_vessels,)
                        avail = result.vessel_availability[0]  # (n_vessels,)

                        # Map vessel index → vessel name via the rows
                        # (rows come back in vessel-index order from
                        # describe_action_request).
                        idx_to_name = {i: r["vessel"] for i, r in enumerate(rows)}

                        # Idle vessels in acting order, then busy vessels
                        acting_order_counter = 0
                        ordered_rows: list[dict] = []
                        for v_idx in ordering:
                            v_idx = int(v_idx)
                            name = idx_to_name.get(v_idx)
                            if name is None or name not in row_by_vessel:
                                continue
                            if avail[v_idx]:
                                acting_order_counter += 1
                                row = {
                                    "episode": ep_idx + 1,
                                    "step": step_count,
                                    "acting_order": acting_order_counter,
                                    "elapsed_hours": round(elapsed_hours, 2),
                                    "delta_hours": round(delta_hours, 2),
                                    **row_by_vessel.pop(name),
                                }
                                ordered_rows.append(row)

                        # Append any remaining busy vessels (not in the
                        # idle acting sequence) at the end with no order.
                        for name, row in row_by_vessel.items():
                            ordered_rows.append(
                                {
                                    "episode": ep_idx + 1,
                                    "step": step_count,
                                    "acting_order": None,
                                    "elapsed_hours": round(elapsed_hours, 2),
                                    "delta_hours": round(delta_hours, 2),
                                    **row,
                                }
                            )

                        all_action_trace.extend(ordered_rows)
                except Exception:
                    pass

            for env_i in range(envs.num_envs):
                if not done[env_i]:
                    ep_reward_accum[env_i] += result.rewards[env_i]

                if env_i == 0:
                    step_info = per_env_infos[env_i]
                    reward_components = step_info.get("reward_components", {})
                    reward_breakdown["total_reward"] += float(
                        reward_components.get("total", result.rewards[env_i])
                    )
                    reward_breakdown["milestone_term_total"] += float(
                        reward_components.get("milestone_term", 0.0)
                    )
                    reward_breakdown["reward_cost_term_total"] += float(
                        reward_components.get("cost_term_reward_applied", 0.0)
                    )
                    for comp_name, comp_val in (
                        reward_components.get("cost_components_reward_applied") or {}
                    ).items():
                        reward_cost_components_total[comp_name] += float(comp_val)

                    # Terminal terms from the rewarder
                    reward_breakdown["completion_bonus_total"] += float(
                        reward_components.get("completion_bonus_term", 0.0)
                    )

                    delta_hours = float(step_info.get("delta_time_hours", 0.0))
                    operational_metrics["elapsed_hours"] += delta_hours

                    cost_breakdown = reward_components.get("cost_breakdown", {})

                    for comp_name in (
                        "elapsed_time_cost",
                        "travel_cost",
                        "storage_cost",
                    ):
                        comp_total = cost_breakdown.get(f"cost/{comp_name}/total", 0.0)
                        operational_costs[comp_name] += float(comp_total)

                    travel_breakdown = cost_breakdown.get(
                        "cost/travel_cost/breakdown", {}
                    )
                    travel_hours_by_vessel = (
                        travel_breakdown.get("per_vessel_hours", {}) or {}
                    )
                    operational_metrics["travel_hours_total"] += float(
                        travel_breakdown.get("total_travel_hours", 0.0)
                    )
                    travel_cost_by_vessel = travel_breakdown.get("per_vessel", {}) or {}
                    for vessel_name, hours in travel_hours_by_vessel.items():
                        operational_metrics["travel_hours_by_vessel"][vessel_name] += (
                            float(hours)
                        )
                    for vessel_name, cost in travel_cost_by_vessel.items():
                        operational_metrics["travel_cost_by_vessel"][vessel_name] += (
                            float(cost)
                        )

                    storage_breakdown = cost_breakdown.get(
                        "cost/storage_cost/breakdown", {}
                    )
                    operational_metrics["storage_unit_hours_total"] += float(
                        storage_breakdown.get("storage_unit_hours_total", 0.0)
                    )
                    for site_name, unit_hours in (
                        storage_breakdown.get("storage_unit_hours_by_site", {}) or {}
                    ).items():
                        operational_metrics["storage_unit_hours_by_site"][
                            site_name
                        ] += float(unit_hours)

                    # Extract per-site storage cost from the per_site
                    # breakdown dict.  Each site entry has a "subtotal"
                    # key holding the weighted cost for that site.
                    per_site = storage_breakdown.get("per_site", {}) or {}
                    for site_name, site_data in per_site.items():
                        if isinstance(site_data, dict):
                            subtotal = site_data.get("subtotal", 0.0)
                            if subtotal:
                                operational_metrics["storage_cost_by_site"][
                                    site_name
                                ] += float(subtotal)

                    for site_name, site_data in per_site.items():
                        if not isinstance(site_data, dict):
                            continue
                        for resource_name, resource_data in site_data.items():
                            if not isinstance(resource_data, dict):
                                continue
                            operational_metrics["storage_unit_hours_by_site_resource"][
                                f"{site_name}/{resource_name}"
                            ] += float(resource_data.get("storage_unit_hours", 0.0))

                if result.terminated[env_i] or result.truncated[env_i]:
                    done[env_i] = True

            step_count += 1

            if capture_inventory:
                collector._capture_inventory(inv_history)

        # Extract episode statistics
        ep_return = float(ep_reward_accum[0])
        ep_length = step_count

        # Check if the env provides episode stats (via RecordEpisodeStatistics wrapper)
        try:
            if last_result is not None:
                per_env_infos = _split_vectorized_info(last_result.infos, envs.num_envs)
                last_info = per_env_infos[0]
                if "episode" in last_info:
                    ep_info = last_info["episode"]
                    r = ep_info.get("r", ep_return)
                    ep_len = ep_info.get("l", ep_length)
                    ep_return = float(
                        r[0] if isinstance(r, (list, tuple, np.ndarray)) else r
                    )
                    ep_length = int(
                        ep_len[0]
                        if isinstance(ep_len, (list, tuple, np.ndarray))
                        else ep_len
                    )
        except Exception:
            pass

        reward_breakdown["total_reward"] = ep_return

        ep_result = {
            "episode": ep_idx + 1,
            "return": ep_return,
            "length": ep_length,
            "wall_time_s": round(time.time() - ep_start, 2),
            "reward_breakdown": {
                **reward_breakdown,
                "reward_cost_components_total": dict(
                    sorted(reward_cost_components_total.items())
                ),
                "net_formula_check": (
                    reward_breakdown["milestone_term_total"]
                    + reward_breakdown["completion_bonus_total"]
                    - reward_breakdown["reward_cost_term_total"]
                ),
            },
            "cost_breakdown": {
                "elapsed_time_cost": float(operational_costs["elapsed_time_cost"]),
                "travel_cost": float(operational_costs["travel_cost"]),
                "storage_cost": float(operational_costs["storage_cost"]),
                "total_cost": float(sum(operational_costs.values())),
            },
            "operational_metrics": {
                "elapsed_hours": float(operational_metrics["elapsed_hours"]),
                "travel_hours_total": float(operational_metrics["travel_hours_total"]),
                "travel_hours_by_vessel": dict(
                    sorted(operational_metrics["travel_hours_by_vessel"].items())
                ),
                "travel_cost_by_vessel": dict(
                    sorted(operational_metrics["travel_cost_by_vessel"].items())
                ),
                "storage_unit_hours_total": float(
                    operational_metrics["storage_unit_hours_total"]
                ),
                "storage_unit_hours_by_site": dict(
                    sorted(operational_metrics["storage_unit_hours_by_site"].items())
                ),
                "storage_cost_by_site": dict(
                    sorted(operational_metrics["storage_cost_by_site"].items())
                ),
                "storage_unit_hours_by_site_resource": dict(
                    sorted(
                        operational_metrics[
                            "storage_unit_hours_by_site_resource"
                        ].items()
                    )
                ),
            },
        }

        # Extract domain-specific metrics from the last result's info
        if last_result is not None:
            per_env_infos = _split_vectorized_info(last_result.infos, envs.num_envs)
            last_info = per_env_infos[0]
            for k in (
                "goals_completed",
                "goals_failed",
                "goals_remaining",
                "elapsed_time_hours",
            ):
                if k in last_info:
                    val = last_info[k]
                    ep_result[k] = float(
                        val[0] if isinstance(val, (list, tuple, np.ndarray)) else val
                    )

            # Surface accumulated PFM cost objective info_keys (e.g. "cost/storage")
            # from the terminal info so a-posteriori PFM can read the exact raw
            # values the live PFMVectorWrapper consumed at termination.
            for k, val in last_info.items():
                if not (isinstance(k, str) and k.startswith("cost/")):
                    continue
                try:
                    ep_result[k] = float(
                        val[0] if isinstance(val, (list, tuple, np.ndarray)) else val
                    )
                except (TypeError, ValueError):
                    continue

        if ep_return > best_return:
            best_return = ep_return
            best_inventory_history = inv_history

        episodes_data.append(ep_result)
        logger.info(
            f"Episode {ep_idx + 1}/{num_episodes}: "
            f"return={ep_return:.2f}  length={ep_length}  "
            f"steps={step_count}"
        )

    # Build summary
    returns = [e["return"] for e in episodes_data]
    lengths = [e["length"] for e in episodes_data]
    summary = {
        "num_episodes": num_episodes,
        "deterministic": deterministic,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
    }

    for key in ("goals_completed", "goals_failed", "goals_remaining"):
        vals = [e[key] for e in episodes_data if key in e]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))

    sim_times = [
        e["elapsed_time_hours"] for e in episodes_data if "elapsed_time_hours" in e
    ]
    if sim_times:
        summary["sim_elapsed_hours_mean"] = float(np.mean(sim_times))
        # Dispersion over episodes. Meaningless on a deterministic environment
        # (all zero) but the only honest readout when activity durations are
        # stochastic, where the mean alone hides the spread it was drawn from.
        summary["sim_elapsed_hours_std"] = float(np.std(sim_times))
        summary["sim_elapsed_hours_min"] = float(np.min(sim_times))
        summary["sim_elapsed_hours_max"] = float(np.max(sim_times))
        summary["sim_elapsed_hours_n"] = int(len(sim_times))

    reward_keys = (
        "total_reward",
        "milestone_term_total",
        "completion_bonus_total",
        "reward_cost_term_total",
    )
    for key in reward_keys:
        vals = [
            e.get("reward_breakdown", {}).get(key)
            for e in episodes_data
            if key in e.get("reward_breakdown", {})
        ]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))

    cost_keys = ("elapsed_time_cost", "travel_cost", "storage_cost", "total_cost")
    for key in cost_keys:
        vals = [
            e.get("cost_breakdown", {}).get(key)
            for e in episodes_data
            if key in e.get("cost_breakdown", {})
        ]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))

    metric_keys = ("elapsed_hours", "travel_hours_total", "storage_unit_hours_total")
    for key in metric_keys:
        vals = [
            e.get("operational_metrics", {}).get(key)
            for e in episodes_data
            if key in e.get("operational_metrics", {})
        ]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))

    # Log the collector's fine-grained macro-step breakdown
    collector.timer.log_summary(f"Eval macro-step breakdown ({num_episodes} episodes)")

    # Try to get the unwrapped env for downstream visualisation (Gantt
    # charts etc.).  This works for SyncVectorEnv; for AsyncVectorEnv
    # the object cannot cross the process boundary so we fall back to None.
    unwrapped_env = None
    try:
        unwrapped_env = envs.call("unwrapped")[0]
    except Exception:
        pass

    return {
        "episodes": episodes_data,
        "summary": summary,
        "action_trace": all_action_trace,
        "inventory_history": best_inventory_history,
        "env": unwrapped_env,
    }
