"""Preference Function Modeling (PFM) for multi-objective reward normalization.

This module provides two classes:

* :class:`PFMStatisticsTracker` — maintains running statistics (Welford +
  Polyak averaging) for each objective.  Designed to live in the **main
  process** so that all parallel environments normalise against identical
  μ and σ.

* :class:`PFMVectorWrapper` — a :class:`gymnasium.vector.VectorWrapper`
  applied *after* ``AsyncVectorEnv`` construction.  On terminal steps it
  replaces the base-env reward with the PFM Z-score reward; on non-terminal
  steps it passes the DPBRS shaping reward through unchanged.

Theory
------
Wolfert (2026) establishes that the **unique** admissible aggregation of
preferences across incommensurate criteria is the **weighted centroid of
Z-scores** (the P* operator).  Z-score normalisation maps raw metrics into
a dimensionless Linear Preference Space (LPS) that satisfies the four PFM
axioms (Preference Preservation, Comparable Criteria, Meaningful
Zero-Reference, Uniqueness).

Because P* is strictly linear, the Expected Utility Policy Gradient (EUPG)
equivalence holds: early scalarisation inside the environment produces
mathematically identical gradients to late scalarisation over vector
returns.  This allows standard scalar PPO bootstrapping with no changes to
the learner, buffer, or GAE computation.

References
----------
- Wolfert, A.R.M. (2026). Unique Preference Aggregation in Design and
  Decision Making. arXiv:2601.19759v1 [math.OC].
- Ng, Harada & Russell (1999). Policy Invariance Under Reward
  Transformations. ICML.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
from loguru import logger

try:
    from gymnasium.vector import VectorEnv, VectorWrapper
except ImportError:  # pragma: no cover
    raise ImportError("PFM requires gymnasium with vector env support.")

from eos.config import PFMConfig, PFMObjectiveConfig
from eos.envs.simple_monopile_transport.preference_functions import (
    LinearPreferenceFunction,
    PreferenceFunction,
)
from eos.utils.debug import check_nan

# ---------------------------------------------------------------------------
# Preference function bounds resolution
# ---------------------------------------------------------------------------


def resolve_preference_bounds(pfm_cfg: PFMConfig, env_cfg: Any) -> None:
    """Resolve None bounds in PFMObjectiveConfig from the environment config.

    For objectives where ``pf_worst`` or ``pf_best`` is None, this function
    computes sensible defaults from the environment configuration:

    - For "minimize" objectives: worst = budget-derived maximum, best = 0.
    - For "maximize" objectives: worst = 0, best = budget-derived maximum.

    The function mutates ``pfm_cfg.objectives`` in place.

    Parameters
    ----------
    pfm_cfg : PFMConfig
        The PFM config block (objectives are mutated in place).
    env_cfg : EnvConfig
        The environment config block (used to derive worst-case budgets).
    """
    from eos.envs.simple_monopile_transport.costs.budgets import (
        compute_budget_by_info_key,
    )

    budgets = compute_budget_by_info_key(
        env_cfg.time_budget,
        env_cfg.reward.costs,
        getattr(env_cfg, "sim", None),
    )

    for obj in pfm_cfg.objectives:
        needs_worst = obj.pf_worst is None
        needs_best = obj.pf_best is None

        if not needs_worst and not needs_best:
            continue

        budget_max = budgets.get(obj.info_key)

        if budget_max is None:
            raise ValueError(
                f"Cannot auto-resolve preference bounds for objective "
                f"'{obj.name}' (info_key='{obj.info_key}'). "
                f"Please set pf_worst and pf_best explicitly in the config."
            )

        if obj.direction == "minimize":
            # Lower is better: worst = max possible, best = 0
            if needs_worst:
                obj.pf_worst = budget_max
            if needs_best:
                obj.pf_best = 0.0
        else:
            # Higher is better: worst = 0, best = max possible
            if needs_worst:
                obj.pf_worst = 0.0
            if needs_best:
                obj.pf_best = budget_max

        logger.info(
            f"PFM bounds resolved for '{obj.name}': "
            f"pf_worst={obj.pf_worst}, pf_best={obj.pf_best}"
        )


# ---------------------------------------------------------------------------
# Statistics tracker (shared across all vectorised sub-environments)
# ---------------------------------------------------------------------------


@dataclass
class _ObjectiveStats:
    """Per-objective running statistics (Welford + Polyak)."""

    name: str
    weight: float
    sigma_min: float
    info_key: str
    preference_fn: PreferenceFunction

    # Welford online accumulators (Active) — tracks preference scores [0, 100]
    active_mean: float = 0.0
    active_m2: float = 0.0  # sum of squared deviations

    # Polyak-smoothed accumulators (Target) — used for Z-score computation
    target_mean: float = 0.0
    target_var: float = 1.0  # initialise to unit variance


class PFMStatisticsTracker:
    """Centralized running statistics for PFM Z-score normalization.

    Maintains **Active** (Welford online) and **Target** (Polyak-smoothed)
    statistics for each objective.  The PFM Z-scores are always computed
    against the Target statistics, ensuring the normalisation reference
    frame shifts microscopically per episode so the PPO value critic can
    track it.

    This class is **not** a gym wrapper — it is a plain Python object that
    lives in the main process and is owned by :class:`PFMVectorWrapper`.

    Parameters
    ----------
    cfg : PFMConfig
        The PFM configuration block.
    """

    def __init__(self, cfg: PFMConfig) -> None:
        self._cfg = cfg
        self._validate_config(cfg)

        self._objectives: List[_ObjectiveStats] = []
        for obj_cfg in cfg.objectives:
            pf = self._build_preference_function(obj_cfg)
            self._objectives.append(
                _ObjectiveStats(
                    name=obj_cfg.name,
                    weight=obj_cfg.weight,
                    sigma_min=obj_cfg.sigma_min,
                    info_key=obj_cfg.info_key,
                    preference_fn=pf,
                )
            )

        self._active_count: int = 0
        self._burn_in_complete: bool = False

        logger.info(
            f"PFMStatisticsTracker initialised: "
            f"{len(self._objectives)} objectives, "
            f"burn_in={cfg.burn_in_trajectories}, "
            f"tau={cfg.polyak_tau}, "
            f"clip={cfg.clip_bound}, "
            f"baseline={cfg.completion_baseline}, "
            f"catastrophic={cfg.catastrophic_penalty}"
        )
        for obj in self._objectives:
            logger.info(
                f"  objective '{obj.name}': weight={obj.weight}, "
                f"sigma_min={obj.sigma_min}, "
                f"info_key='{obj.info_key}', "
                f"pf={obj.preference_fn!r}"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def burn_in_complete(self) -> bool:
        """Whether the burn-in phase has finished."""
        return self._burn_in_complete

    @property
    def active_count(self) -> int:
        """Number of successful completions observed so far."""
        return self._active_count

    @property
    def n_objectives(self) -> int:
        return len(self._objectives)

    @property
    def objective_info_keys(self) -> List[str]:
        """The info-dict keys needed to extract raw metrics."""
        return [obj.info_key for obj in self._objectives]

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def welford_update(self, raw_values: Sequence[float]) -> None:
        """Update Active (Welford) statistics with a new completed trajectory.

        Parameters
        ----------
        raw_values : Sequence[float]
            Raw metric values, one per objective, in config order.
        """
        self._active_count += 1
        n = self._active_count

        for obj, x in zip(self._objectives, raw_values):
            delta = x - obj.active_mean
            obj.active_mean += delta / n
            delta2 = x - obj.active_mean
            obj.active_m2 += delta * delta2

        # Check burn-in transition
        if not self._burn_in_complete:
            if self._active_count >= self._cfg.burn_in_trajectories:
                self._burn_in_complete = True
                self._initialise_target_from_active()
                logger.info(
                    f"PFM burn-in complete after {self._active_count} trajectories. "
                    f"Target stats initialised."
                )
                for obj in self._objectives:
                    sigma = math.sqrt(max(obj.target_var, 0.0))
                    logger.info(
                        f"  '{obj.name}': mu={obj.target_mean:.4f}, "
                        f"sigma={sigma:.4f} "
                        f"(floor={obj.sigma_min})"
                    )

    def apply_preference_functions(self, raw_values: Sequence[float]) -> List[float]:
        """Map raw physical metrics through preference functions to [0, 100].

        This is Step 2 of the ODESYS threefold formulation (Desirability).
        Directionality is encoded in the preference function itself:
        higher preference score always means more desirable.

        Parameters
        ----------
        raw_values : Sequence[float]
            Raw metric values, one per objective, in config order.

        Returns
        -------
        List[float]
            Preference scores in [0, 100], one per objective.
        """
        return [
            obj.preference_fn.evaluate(x)
            for obj, x in zip(self._objectives, raw_values)
        ]

    def compute_z_scores(
        self, pref_scores: Sequence[float], use_active: bool = False
    ) -> List[float]:
        """Compute PFM Z-scores against the chosen reference statistics.

        Operates on preference scores (0-100), not raw physical metrics.
        Since preference functions already encode directionality (higher
        score = more desirable), Z-scores use a positive sign: a
        preference score above the mean yields a positive Z-score.

        Parameters
        ----------
        pref_scores : Sequence[float]
            Preference scores in [0, 100], one per objective.
        use_active : bool
            When ``True``, Z-score against the running **Active** (Welford)
            statistics instead of the **Target** (Polyak) statistics.  Used
            during the burn-in warm-up ramp, where Target is not yet
            initialised.

        Returns
        -------
        List[float]
            Z-scores, one per objective.
        """
        z_scores: List[float] = []
        for obj, x in zip(self._objectives, pref_scores):
            if use_active:
                mean = obj.active_mean
                var = self._active_variance(obj)
            else:
                mean = obj.target_mean
                var = obj.target_var
            sigma = math.sqrt(max(var, 0.0))
            sigma = max(sigma, obj.sigma_min)
            z = (x - mean) / sigma
            z_scores.append(z)
        return z_scores

    def scalarize(self, z_scores: Sequence[float]) -> float:
        """Compute the PFM P* aggregated preference (weighted centroid).

        Parameters
        ----------
        z_scores : Sequence[float]
            Z-scores, one per objective.

        Returns
        -------
        float
            The scalarised P* value.
        """
        return sum(obj.weight * z for obj, z in zip(self._objectives, z_scores))

    def warmup_scale(self) -> float:
        """Burn-in ramp factor (lambda) for the clipped P* contribution.

        Intended to be called only while still inside the burn-in branch
        (Target stats not yet relied upon).  When ``scale_warmup`` is
        enabled, lambda ramps linearly from ``0`` on the first successful
        completion to ``1`` on the final burn-in completion
        (``_active_count == burn_in_trajectories``), phasing the PFM
        preference signal in gradually instead of switching it on
        discontinuously.

        When ``scale_warmup`` is disabled, returns ``0.0`` for the entire
        burn-in phase (flat ``completion_baseline``), exactly recovering
        the original pre-warm-up behaviour.  Post-burn-in the caller pins
        lambda to ``1.0`` directly, so this method is not consulted there.
        """
        if not self._cfg.scale_warmup:
            return 0.0
        n = self._cfg.burn_in_trajectories
        if n <= 1:
            return 1.0
        # _active_count == k, already incremented for the current
        # trajectory by welford_update at this point.
        k = self._active_count
        return min(1.0, max(0.0, (k - 1) / (n - 1)))

    def polyak_update(self) -> None:
        """Soft-update Target statistics from Active statistics.

        Uses Polyak averaging: ``target ← (1 - τ) * target + τ * active``.
        Called once per successful completion *after* the Z-score has been
        computed (so the current trajectory is normalised against the
        previous target, not the just-updated one).
        """
        tau = self._cfg.polyak_tau
        for obj in self._objectives:
            active_var = self._active_variance(obj)
            obj.target_mean = (1.0 - tau) * obj.target_mean + tau * obj.active_mean
            obj.target_var = (1.0 - tau) * obj.target_var + tau * active_var

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def build_diagnostics(
        self,
        raw_values: Sequence[float] | None,
        pref_scores: Sequence[float] | None,
        z_scores: Sequence[float] | None,
        p_star: float | None,
        pfm_reward: float,
        is_success: bool,
        warmup_scale: float | None = None,
    ) -> Dict[str, Any]:
        """Build a flat diagnostics dict for logging / WandB.

        Parameters
        ----------
        raw_values : Sequence[float] | None
            Raw metric values (None on failure).
        pref_scores : Sequence[float] | None
            Preference scores in [0, 100] (None on failure).
        z_scores : Sequence[float] | None
            Z-scores (None during burn-in or failure).
        p_star : float | None
            Scalarised P* value (None during burn-in or failure).
        pfm_reward : float
            The final PFM reward component (before adding DPBRS).
        is_success : bool
            Whether the episode was a successful completion.

        Returns
        -------
        Dict[str, Any]
            Flat dict with keys suitable for ``pfm/...`` logging.
        """
        diag: Dict[str, Any] = {
            "pfm_reward": pfm_reward,
            "p_star": float(p_star) if p_star is not None else float("nan"),
            "burn_in_complete": self._burn_in_complete,
            "trajectories_seen": self._active_count,
            "is_success": is_success,
            # Weighted sum of preference scores (sum_i w_i * p_i), p_i in [0, 100].
            # Unlike p_star (a weighted sum of Z-scores against a drifting
            # population), this is a *stationary* desirability in [0, 100].
            # Failures contribute 0.0 so the batch mean folds completion rate
            # into the score, making it usable as a checkpoint-selection metric.
            "pref_weighted": 0.0,
        }
        if warmup_scale is not None:
            diag["warmup_scale"] = float(warmup_scale)

        for obj in self._objectives:
            prefix = obj.name
            sigma_active = math.sqrt(max(self._active_variance(obj), 0.0))
            sigma_target = math.sqrt(max(obj.target_var, 0.0))

            diag[f"{prefix}/mu_active"] = obj.active_mean
            diag[f"{prefix}/sigma_active"] = sigma_active
            diag[f"{prefix}/mu_target"] = obj.target_mean
            diag[f"{prefix}/sigma_target"] = sigma_target

        if raw_values is not None:
            for obj, x in zip(self._objectives, raw_values):
                diag[f"{obj.name}/raw"] = float(x)

        if pref_scores is not None:
            for obj, x in zip(self._objectives, pref_scores):
                diag[f"{obj.name}/pref"] = float(x)
            diag["pref_weighted"] = float(
                sum(obj.weight * x for obj, x in zip(self._objectives, pref_scores))
            )

        if z_scores is not None:
            for obj, z in zip(self._objectives, z_scores):
                diag[f"{obj.name}/z_score"] = float(z)

        return diag

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialise_target_from_active(self) -> None:
        """Copy Active stats to Target when burn-in completes."""
        for obj in self._objectives:
            obj.target_mean = obj.active_mean
            obj.target_var = self._active_variance(obj)

    def _active_variance(self, obj: _ObjectiveStats) -> float:
        """Compute the current sample variance from Welford accumulators."""
        if self._active_count < 2:
            return 1.0  # not enough data — return unit variance
        return obj.active_m2 / (self._active_count - 1)

    @staticmethod
    def _build_preference_function(obj_cfg: PFMObjectiveConfig) -> PreferenceFunction:
        """Construct a preference function from an objective config.

        Parameters
        ----------
        obj_cfg : PFMObjectiveConfig
            The objective configuration.

        Returns
        -------
        PreferenceFunction
            The constructed preference function.

        Raises
        ------
        ValueError
            If ``pf_worst`` or ``pf_best`` are None (bounds must be resolved
            before constructing the tracker) or if ``pf_type`` is unknown.
        """
        if obj_cfg.pf_worst is None or obj_cfg.pf_best is None:
            raise ValueError(
                f"Preference function bounds for objective '{obj_cfg.name}' are not "
                f"resolved (pf_worst={obj_cfg.pf_worst}, pf_best={obj_cfg.pf_best}). "
                f"Either set them explicitly in the config or ensure the PFM wrapper "
                f"resolves them from the environment config before tracker construction."
            )

        if obj_cfg.pf_type == "linear":
            return LinearPreferenceFunction(
                worst=obj_cfg.pf_worst,
                best=obj_cfg.pf_best,
            )
        else:
            raise ValueError(
                f"Unknown preference function type '{obj_cfg.pf_type}' for "
                f"objective '{obj_cfg.name}'. Supported types: 'linear'."
            )

    @staticmethod
    def _validate_config(cfg: PFMConfig) -> None:
        """Validate PFM configuration at init time."""
        if not cfg.objectives:
            raise ValueError(
                "PFMConfig.objectives must contain at least one objective."
            )

        total_weight = sum(obj.weight for obj in cfg.objectives)
        if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"PFM objective weights must sum to 1.0, got {total_weight:.6f}. "
                f"Weights: {[obj.weight for obj in cfg.objectives]}"
            )

        for obj in cfg.objectives:
            if obj.sigma_min <= 0:
                raise ValueError(
                    f"sigma_min for objective '{obj.name}' must be positive, "
                    f"got {obj.sigma_min}"
                )
            if obj.direction not in ("minimize", "maximize"):
                raise ValueError(
                    f"direction for objective '{obj.name}' must be 'minimize' or "
                    f"'maximize', got '{obj.direction}'"
                )

        min_valid_reward = cfg.completion_baseline - cfg.clip_bound
        if cfg.catastrophic_penalty >= min_valid_reward:
            logger.warning(
                f"catastrophic_penalty ({cfg.catastrophic_penalty}) is not strictly "
                f"less than the minimum valid reward ({min_valid_reward} = "
                f"completion_baseline {cfg.completion_baseline} - clip_bound "
                f"{cfg.clip_bound}). The agent may learn that deadlocking is "
                f"equivalent to a poor completion."
            )

        if cfg.burn_in_trajectories < 2:
            raise ValueError(
                f"burn_in_trajectories must be >= 2 (need at least 2 samples to "
                f"compute variance), got {cfg.burn_in_trajectories}"
            )


# ---------------------------------------------------------------------------
# Vector environment wrapper (main-process, wraps AsyncVectorEnv)
# ---------------------------------------------------------------------------


class PFMVectorWrapper(VectorWrapper):
    """Centralized PFM reward normalization for vectorized environments.

    Wraps the entire ``AsyncVectorEnv`` (or ``SyncVectorEnv``).  Reward
    modification happens exclusively in the main process, guaranteeing
    all parallel environments normalise against identical μ and σ.

    Behaviour per step
    ------------------
    * **Non-terminal envs:** reward passes through unchanged (pure DPBRS
      shaping from the base environment).
    * **Terminal success (during burn-in):** ``reward += completion_baseline``
    * **Terminal success (post-burn-in):**
      ``reward += completion_baseline + clip(P*(z), -clip_bound, clip_bound)``
    * **Terminal failure:** ``reward += catastrophic_penalty``

    The wrapper also maintains per-env episode return accumulators that
    include the PFM reward modifications, exposed as
    ``infos["pfm_episode_return"]`` on terminal steps.

    Parameters
    ----------
    env : VectorEnv
        The vectorised environment to wrap (``AsyncVectorEnv`` or
        ``SyncVectorEnv``).
    pfm_cfg : PFMConfig
        PFM configuration block from the experiment config.
    tracker : PFMStatisticsTracker | None
        Optional pre-built tracker.  If ``None``, a new one is created
        from ``pfm_cfg``.  Passing an existing tracker is useful for
        sharing statistics between training and evaluation envs.
    """

    def __init__(
        self,
        env: VectorEnv,
        pfm_cfg: PFMConfig,
        tracker: PFMStatisticsTracker | None = None,
    ) -> None:
        super().__init__(env)
        self._cfg = pfm_cfg
        self._tracker = tracker or PFMStatisticsTracker(pfm_cfg)

        # Per-env PFM-inclusive episode return tracking.
        # RecordEpisodeStatistics runs inside the subprocess and sees
        # pre-PFM rewards.  We track the PFM-inclusive returns here so
        # the runner can use them for checkpointing.
        self._pfm_episode_returns = np.zeros(self.num_envs, dtype=np.float64)

        logger.info(f"PFMVectorWrapper applied over {self.num_envs} environments.")

    # ------------------------------------------------------------------
    # VectorWrapper overrides
    # ------------------------------------------------------------------

    def step(self, actions):
        """Step all environments, applying PFM reward normalization on terminal steps."""
        obs, rewards, terminated, truncated, infos = self.env.step(actions)

        # Work on a writable copy of rewards (vectorised envs may return
        # read-only arrays).
        rewards = np.array(rewards, dtype=np.float64, copy=True)

        dones = np.logical_or(terminated, truncated)

        # Pre-allocate PFM info arrays in the vectorised info dict.
        # Gymnasium's vectorised info format stores per-key arrays.
        # We initialise NaN for non-terminal envs and fill terminal ones.
        n = self.num_envs
        pfm_reward_arr = np.full(n, float("nan"), dtype=np.float64)
        pfm_p_star_arr = np.full(n, float("nan"), dtype=np.float64)
        pfm_episode_return_arr = np.full(n, float("nan"), dtype=np.float64)

        for i in range(n):
            if not dones[i]:
                # Non-terminal: accumulate raw reward and pass through
                self._pfm_episode_returns[i] += rewards[i]
                continue

            # ── Terminal step for env i ──────────────────────────────
            is_success = self._extract_scalar_bool(infos, "is_success", i)

            pfm_component: float
            p_star: float | None = None
            z_scores: list[float] | None = None
            pref_scores: list[float] | None = None
            raw_values: list[float] | None = None
            warmup_scale: float | None = None

            if not is_success:
                # Failure / deadlock → catastrophic penalty (exempt from clipping)
                pfm_component = self._cfg.catastrophic_penalty
            else:
                # Extract raw metrics for this env
                raw_values = self._extract_raw_metrics(infos, i)

                # Step 2: Map raw metrics through preference functions (Desirability)
                pref_scores = self._tracker.apply_preference_functions(raw_values)

                # Snapshot burn-in status BEFORE updating stats so we pick
                # the right reference frame: during burn-in we Z-score
                # against the running Active stats (Target is not yet
                # initialised); afterwards we Z-score against Target.
                was_post_burnin = self._tracker.burn_in_complete

                # Always update Welford (Active) stats on preference scores.
                # This may also flip burn-in complete (and seed Target from
                # Active) when _active_count reaches burn_in_trajectories.
                self._tracker.welford_update(pref_scores)

                if not was_post_burnin:
                    # ── Burn-in phase ───────────────────────────────────
                    # warmup_scale() returns the ramp lambda in [0, 1] when
                    # scale_warmup is enabled, or 0.0 when disabled (flat
                    # baseline — original behaviour). At lambda == 0 the
                    # PFM signal contributes nothing, so skip the Z-score
                    # work entirely to keep diagnostics None as before.
                    warmup_scale = self._tracker.warmup_scale()
                    if warmup_scale > 0.0:
                        z_scores = self._tracker.compute_z_scores(
                            pref_scores, use_active=True
                        )
                        p_star = self._tracker.scalarize(z_scores)
                        p_star_clipped = max(
                            -self._cfg.clip_bound,
                            min(self._cfg.clip_bound, p_star),
                        )
                        pfm_component = (
                            self._cfg.completion_baseline
                            + warmup_scale * p_star_clipped
                        )
                    else:
                        pfm_component = self._cfg.completion_baseline
                else:
                    # ── Post-burn-in: full PFM against Target stats ─────
                    z_scores = self._tracker.compute_z_scores(pref_scores)
                    p_star = self._tracker.scalarize(z_scores)

                    # Clip P* (catastrophic penalty is exempt)
                    p_star_clipped = max(
                        -self._cfg.clip_bound,
                        min(self._cfg.clip_bound, p_star),
                    )
                    warmup_scale = 1.0
                    pfm_component = self._cfg.completion_baseline + p_star_clipped

                    # Polyak-update Target from Active (after Z-score computation)
                    self._tracker.polyak_update()

            # Add PFM component to the base reward (which contains DPBRS F_t)
            rewards[i] += pfm_component
            pfm_reward_arr[i] = pfm_component
            if p_star is not None:
                pfm_p_star_arr[i] = p_star

            # Finalise PFM-inclusive episode return
            self._pfm_episode_returns[i] += rewards[i]
            pfm_episode_return_arr[i] = self._pfm_episode_returns[i]

            # Reset accumulator for next episode (auto-reset in vec env
            # means the next step() call will be a fresh episode).
            self._pfm_episode_returns[i] = 0.0

            # Inject per-objective diagnostics into info for this env
            diag = self._tracker.build_diagnostics(
                raw_values=raw_values,
                pref_scores=pref_scores,
                z_scores=z_scores,
                p_star=p_star,
                pfm_reward=pfm_component,
                is_success=is_success,
                warmup_scale=warmup_scale,
            )
            self._inject_diag_into_infos(infos, i, diag)

        # ── Debug: detect NaN reward before it poisons downstream ─────
        check_nan(rewards, "pfm.rewards_post_modification")

        # Store batch-level PFM arrays in info
        infos["pfm_reward"] = pfm_reward_arr
        infos["_pfm_reward"] = dones  # Gymnasium mask convention
        infos["pfm_p_star"] = pfm_p_star_arr
        infos["_pfm_p_star"] = dones
        infos["pfm_episode_return"] = pfm_episode_return_arr
        infos["_pfm_episode_return"] = dones

        # Hijack RecordEpisodeStatistics return with PFM-inclusive value.
        # RecordEpisodeStatistics runs in the subprocess and only sees
        # pre-PFM rewards (pure DPBRS).  Everything downstream — the
        # runner, EMA, checkpointing, WandB charts — reads
        # infos["episode"]["r"], so overwriting it here propagates the
        # correct PFM-inclusive return everywhere automatically.
        if "episode" in infos:
            ep_r = infos["episode"].get("r")
            if ep_r is not None:
                # Make a writable copy if the array is read-only
                # (vectorised envs may return immutable arrays).
                if not ep_r.flags.writeable:
                    infos["episode"]["r"] = np.array(ep_r, copy=True)
                    ep_r = infos["episode"]["r"]
                for i in range(n):
                    if dones[i] and not np.isnan(pfm_episode_return_arr[i]):
                        ep_r[i] = pfm_episode_return_arr[i]

        return obs, rewards, terminated, truncated, infos

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Reset all environments.  Episode return accumulators are zeroed."""
        self._pfm_episode_returns[:] = 0.0
        return self.env.reset(seed=seed, options=options)

    # ------------------------------------------------------------------
    # Delegate VectorEnv methods not on the VectorWrapper base class.
    #
    # gymnasium.vector.VectorWrapper inherits from VectorEnv but does NOT
    # proxy call(), call_async(), call_wait(), get_attr(), or set_attr()
    # which are defined only on AsyncVectorEnv / SyncVectorEnv.  The
    # micro-stepping protocol relies on envs.call("micro_step", ...) so
    # we must forward these explicitly.
    # ------------------------------------------------------------------

    def call(self, name: str, *args, **kwargs):  # type: ignore[override]
        """Delegate ``call()`` to the wrapped vectorised environment."""
        return self.env.call(name, *args, **kwargs)  # type: ignore[attr-defined]

    def call_async(self, name: str, *args, **kwargs):
        """Delegate ``call_async()`` to the wrapped vectorised environment."""
        return self.env.call_async(name, *args, **kwargs)  # type: ignore[attr-defined]

    def call_wait(self, *args, **kwargs):
        """Delegate ``call_wait()`` to the wrapped vectorised environment."""
        return self.env.call_wait(*args, **kwargs)  # type: ignore[attr-defined]

    def get_attr(self, name: str):
        """Delegate ``get_attr()`` to the wrapped vectorised environment."""
        return self.env.get_attr(name)  # type: ignore[attr-defined]

    def set_attr(self, name: str, values):
        """Delegate ``set_attr()`` to the wrapped vectorised environment."""
        return self.env.set_attr(name, values)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_raw_metrics(
        self, infos: Dict[str, Any], env_index: int
    ) -> List[float]:
        """Extract raw objective metrics for a single env from vectorised infos.

        Parameters
        ----------
        infos : Dict[str, Any]
            The vectorised info dict from ``env.step()``.
        env_index : int
            Index of the sub-environment.

        Returns
        -------
        List[float]
            Raw metric values, one per objective, in config order.

        Raises
        ------
        KeyError
            If a required info key is missing.
        """
        values: List[float] = []
        for obj in self._tracker._objectives:
            key = obj.info_key
            if key not in infos:
                raise KeyError(
                    f"PFM objective '{obj.name}' expects info key '{key}' "
                    f"but it was not found in the environment info dict. "
                    f"Available keys: {sorted(infos.keys())}"
                )
            val = infos[key]
            if isinstance(val, np.ndarray):
                values.append(float(val[env_index]))
            elif isinstance(val, (list, tuple)):
                values.append(float(val[env_index]))
            else:
                # Scalar broadcast (shouldn't normally happen in vec envs)
                values.append(float(val))
        return values

    @staticmethod
    def _extract_scalar_bool(infos: Dict[str, Any], key: str, env_index: int) -> bool:
        """Extract a boolean value for a single env from vectorised infos."""
        val = infos.get(key)
        if val is None:
            return False
        if isinstance(val, np.ndarray):
            return bool(val[env_index])
        if isinstance(val, (list, tuple)):
            return bool(val[env_index])
        return bool(val)

    def _inject_diag_into_infos(
        self,
        infos: Dict[str, Any],
        env_index: int,
        diag: Dict[str, Any],
    ) -> None:
        """Inject PFM diagnostics into the vectorised info dict.

        Gymnasium's vectorised info format uses ``infos[key]`` as arrays
        with one entry per env, plus ``infos[_key]`` as a boolean mask
        indicating which envs have valid data for that key.

        For nested PFM diagnostics we store them under a ``"pfm"`` sub-dict
        per env.  Since Gymnasium's ``_split_vectorized_info`` handles
        nested dicts recursively, this will be correctly unpacked by the
        runner.
        """
        if "pfm" not in infos:
            # First terminal env this step — initialise the nested dict
            infos["pfm"] = [{} for _ in range(self.num_envs)]
            infos["_pfm"] = np.zeros(self.num_envs, dtype=bool)

        infos["pfm"][env_index] = diag
        infos["_pfm"][env_index] = True
