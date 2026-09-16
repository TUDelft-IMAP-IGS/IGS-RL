from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

from eos.config import RewardConfig

from .costs import AggregatedCostResult, CostAggregator, CostContext
from .milestone_tracker import MilestoneResult


@dataclass(slots=True)
class RewardMetrics:
    """Per-step signals that the gym environment feeds into the rewarder.

    The environment is responsible for computing these before calling
    :meth:`SMTMultiObjectiveRewarder.step`.
    """

    terminated: bool
    truncated: bool
    success: bool

    # --- Cost-context fields (used by modular cost components) ---
    cost_context: CostContext | None = None

    # --- Milestone fields ---
    milestone_result: MilestoneResult | None = None


class SMTMultiObjectiveRewarder:
    """Compute rewards using milestone DPBRS shaping + sparse terminal cost evaluation.

    The reward signal is composed of:

    * **Milestone shaping** – DPBRS potential-based shaping from a
      :class:`~eos.envs.simple_monopile_transport.milestone_tracker.MilestoneTracker`.
      Provides dense per-step exploration guidance.

    * **Terminal cost evaluation** – on ALL terminal states (success and
      failure), the agent receives a cost deduction
      ``-w * sqrt(total_accumulated_cost)`` where
      ``w = completion_bonus / sqrt(C_max)``. This sqrt squashing
      compresses heavy-tailed costs into [0, completion_bonus].
      On success, the agent additionally receives ``completion_bonus``.

    * **PFM mode** – when PFM is active, the rewarder provides only the
      DPBRS shaping. Terminal reward assignment is handled entirely by
      the PFMVectorWrapper.

    The reward range is [0, completion_bonus] for terminal evaluation,
    plus DPBRS deltas during the episode that telescope to zero.

    Cost components are configured independently via Hydra.  See
    :mod:`eos.envs.simple_monopile_transport.costs` for the component
    catalogue.
    """

    def __init__(self, cfg: RewardConfig, worst_case_cost: float = 0.0) -> None:
        self.cfg = cfg

        # When PFM is active, the rewarder acts as a pure DPBRS pass-through.
        # Cost deductions and terminal bonuses are suppressed — the
        # PFMVectorWrapper handles terminal reward assignment.
        self._pfm_active = (
            hasattr(cfg, "pfm") and cfg.pfm is not None and cfg.pfm.enabled
        )

        # Modular cost system
        self._cost_aggregator = CostAggregator(cfg.costs)

        # Worst-case cost (C_max) for the full time budget (computed by the env).
        # Used to compute the sqrt squash weight.
        self._worst_case_cost = worst_case_cost

        # Squashing weight: w = B / sqrt(C_max)
        # Maps cost range [0, C_max] → [0, B] via sqrt compression.
        # Guarantees reward range is [0, completion_bonus].
        if worst_case_cost > 0:
            self._cost_squash_weight = cfg.completion_bonus / math.sqrt(worst_case_cost)
        else:
            self._cost_squash_weight = 1.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset episode-level accumulators (call at the start of each episode)."""
        self._cost_aggregator.reset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def accumulated_costs(self) -> Dict[str, float]:
        """Per-component accumulated costs since the last reset."""
        return self._cost_aggregator.get_accumulated_costs()

    @property
    def total_accumulated_cost(self) -> float:
        """Global-weighted sum of all accumulated component costs."""
        return self._cost_aggregator.get_total_accumulated_cost()

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, metrics: RewardMetrics) -> tuple[float, Dict[str, Any]]:
        """Compute the reward for a single environment step.

        Parameters
        ----------
        metrics : RewardMetrics
            Pre-computed step signals from the environment.

        Returns
        -------
        reward : float
            Scalar reward for this step.
        components : Dict[str, Any]
            Detailed breakdown for logging / WandB dashboards.
        """
        # --- Accumulate costs (always, for logging and terminal eval) ----
        cost_result: AggregatedCostResult | None = None
        cost_term_step = 0.0
        cost_term_accumulated = 0.0

        if metrics.cost_context is not None:
            cost_result = self._cost_aggregator.step(metrics.cost_context)
            cost_term_step = cost_result.total
            cost_term_accumulated = self._cost_aggregator.get_total_accumulated_cost()

        # --- DPBRS milestone shaping (always applied) --------------------
        milestone_term = 0.0
        milestone_result: MilestoneResult | None = metrics.milestone_result
        if milestone_result is not None:
            milestone_term = milestone_result.total_shaped_reward

        # --- Assemble reward ---------------------------------------------
        reward = milestone_term
        completion_bonus_term = 0.0
        cost_term_applied = 0.0
        cost_components_applied: Dict[str, float] = {}

        # --- Terminal reward ---------------------------------------------
        if metrics.terminated or metrics.truncated:
            if not self._pfm_active:
                # Sqrt-squashed cost deduction (applied on ALL terminal states)
                total_cost = self._cost_aggregator.get_total_accumulated_cost()
                cost_term_applied = self._cost_squash_weight * math.sqrt(total_cost)

                # Per-component breakdown scaled to match squashed total
                gw = self._cost_aggregator.global_weight
                for cn, acc in self._cost_aggregator.get_accumulated_costs().items():
                    cost_components_applied[cn] = acc * gw
                if total_cost > 0:
                    squash_ratio = cost_term_applied / total_cost
                    cost_components_applied = {
                        cn: v * squash_ratio
                        for cn, v in cost_components_applied.items()
                    }

                # Completion bonus only on success
                if metrics.success:
                    completion_bonus_term = float(self.cfg.completion_bonus)

                reward += completion_bonus_term - cost_term_applied
            # When PFM is active, terminal reward is handled by PFMVectorWrapper.

        # --- Build component breakdown for logging -----------------------
        components: Dict[str, Any] = {
            "total": float(reward),
            "milestone_term": float(milestone_term),
            "completion_bonus_term": float(completion_bonus_term),
            "cost_term_step": float(cost_term_step),
            "cost_term_reward_applied": float(cost_term_applied),
            "cost_term_accumulated": float(cost_term_accumulated),
            "terminated": bool(metrics.terminated),
            "truncated": bool(metrics.truncated),
            "success": bool(metrics.success),
            "worst_case_cost": float(self._worst_case_cost),
            "cost_squash_weight": float(self._cost_squash_weight),
        }

        # Detailed per-step cost breakdown (if available)
        if cost_result is not None:
            components["cost_breakdown"] = cost_result.breakdown
            components["cost_components_step"] = {
                comp_name: float(result.total)
                for comp_name, result in cost_result.component_results.items()
            }

        # Per-component cost applied to reward
        components["cost_components_reward_applied"] = {
            comp_name: float(v) for comp_name, v in cost_components_applied.items()
        }

        # Milestone per-goal breakdown (if available)
        if milestone_result is not None:
            components["milestone_total_potential"] = float(
                milestone_result.total_potential
            )
            components["milestone_breakdown"] = milestone_result.breakdown

        return float(reward), components
