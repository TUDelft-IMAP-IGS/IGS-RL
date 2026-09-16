"""Supply-chain milestone tracker for potential-based reward shaping.

This module implements a :class:`MilestoneTracker` that assigns each resource
unit in the simulation a *supply-chain stage potential* based on its current
location (source site → transport vessel → staging site → installer vessel →
goal / installation site).

Using Dynamic Potential-Based Reward Shaping (DPBRS) for Semi-Markov Decision
Processes (SMDPs), the tracker calculates a true dynamic potential Φ(s, t) by
multiplying the spatial potential by a time-dependent urgency factor.

The shaping reward is guaranteed to preserve the optimal policy using the
telescoping formula:
    F = (γ_t * Φ(s', t')) - Φ(s, t)
where γ_t is the step-specific discount factor based on elapsed physical time.

At terminal states the dynamic potential is, by default, explicitly zeroed
(Φ(s_T) = 0) so the DPBRS telescoping sum cancels exactly over finite-horizon
episodes, preserving policy invariance. This is configurable via
:attr:`~eos.config.MilestoneConfig.terminal_zeroing`: setting it to
``"failure_only"`` keeps Φ on a *successful* completion (so the final
goal-completing step earns a bounded ``+`` bonus rather than a ``−Φ_prev``
cliff), which deliberately breaks strict invariance to remove the
completion-avoidance pathology. The gym environment decides per step whether to
pass ``is_terminal`` according to that policy; this tracker simply honours it.

See :class:`~eos.config.MilestoneConfig` for tunable knobs (stage weights,
urgency mode / scale).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from loguru import logger

from eos.config import GoalConfig, MilestoneConfig

from .types import SiteRole, VesselRole

# ---------------------------------------------------------------------------
# Per-step context & result dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MilestoneContext:
    """Read-only snapshot consumed by :meth:`MilestoneTracker.step`.

    The gym environment assembles this every step from simulator state.

    Attributes
    ----------
    site_inventories : Dict[str, Dict[str, int]]
        ``site_name → {resource_name: current_level}``.
    vessel_inventories : Dict[str, Dict[str, int]]
        ``vessel_name → {resource_name: current_level}``.
    vessel_sites : Dict[str, str]
        ``vessel_name -> site_name``.
    elapsed_time_hours : float
        Simulation hours elapsed since the episode started.
    gamma_t : float
        The SMDP discount factor for the current step (e.g., e^(-beta * delta_t)).
    """

    site_inventories: Dict[str, Dict[str, int]] = field(default_factory=dict)
    vessel_inventories: Dict[str, Dict[str, int]] = field(default_factory=dict)
    vessel_sites: Dict[str, str] = field(default_factory=dict)
    elapsed_time_hours: float = 0.0
    gamma_t: float = 1.0
    is_terminal: bool = False


@dataclass(slots=True)
class GoalMilestoneInfo:
    """Per-goal breakdown returned inside :class:`MilestoneResult`.

    Attributes
    ----------
    resource_type : str
        Resource type identifier tracked by this goal.
    location : str
        Goal destination (installation site name).
    raw_potential : float
        Unweighted spatial potential based on supply-chain location.
    urgency : float
        Time-dependent urgency multiplier.
    phi : float
        The true dynamic potential Φ(s, t) = raw_potential * urgency.
    shaped_reward : float
        The mathematically rigorous DPBRS shaping reward: (γ_t * Φ_new) - Φ_old.
    """

    resource_type: str
    location: str
    raw_potential: float
    urgency: float
    phi: float
    shaped_reward: float


@dataclass(slots=True)
class MilestoneResult:
    """Output of :meth:`MilestoneTracker.step`.

    Attributes
    ----------
    total_delta : float
        Sum of the mathematically rigorous shaping rewards across all goals.
    total_potential : float
        Sum of the true dynamic potentials Φ(s, t) across all goals.
    per_goal : List[GoalMilestoneInfo]
        Detailed breakdown per goal.
    breakdown : Dict[str, Any]
        Flat dict suitable for logging / WandB dashboards.
    """

    total_shaped_reward: float = 0.0
    total_potential: float = 0.0
    per_goal: List[GoalMilestoneInfo] = field(default_factory=list)
    breakdown: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class MilestoneTracker:
    """Tracks supply-chain progress for every goal and produces potential deltas.

    Parameters
    ----------
    cfg : MilestoneConfig
        Stage weights, urgency mode, and urgency scale.
    goal_configs : List[GoalConfig]
        The goals the agent must fulfil.  Each goal specifies a resource,
        a target location, a required quantity, an optional deadline (hours),
        and a ``reward_per_unit``.
    site_roles : Dict[str, str]
        Mapping from site name to role string.
    vessel_roles : Dict[str, str]
        Mapping from vessel name to role string.
    """

    def __init__(
        self,
        cfg: MilestoneConfig,
        goal_configs: List[GoalConfig],
        site_roles: Dict[str, str],
        vessel_roles: Dict[str, str],
        phi_max: float | None = None,
    ) -> None:
        self._cfg = cfg
        self._goal_configs = list(goal_configs)
        self._site_roles: Dict[str, SiteRole] = {
            name: SiteRole(role) for name, role in site_roles.items()
        }
        self._vessel_roles: Dict[str, VesselRole] = {
            name: VesselRole(role) for name, role in vessel_roles.items()
        }

        # Compute d_max for "inverse" urgency mode
        deadlines = [g.deadline for g in self._goal_configs if g.deadline is not None]
        self._d_max: float = max(deadlines) if deadlines else 1.0

        # Previous dynamic potentials Φ(s, t) (initialized on reset)
        self._prev_phis: List[float] = [0.0] * len(self._goal_configs)

        # PFM potential scaling: when phi_max is set, scale all potentials
        # so that the theoretical maximum total potential equals phi_max.
        # This aligns DPBRS step-reward magnitudes with PFM terminal rewards.
        # The scale factor is STATIC (set once at init) to preserve the
        # DPBRS policy-invariance guarantee.
        self._phi_scale: float = 1.0
        if phi_max is not None:
            max_raw = sum(
                g.quantity * g.reward_per_unit * cfg.stage_weights.at_goal
                for g in self._goal_configs
            )
            if max_raw > 0:
                self._phi_scale = phi_max / max_raw

        logger.debug(
            f"MilestoneTracker initialised with {len(self._goal_configs)} goals, "
            f"urgency_mode={cfg.urgency_mode}, urgency_scale={cfg.urgency_scale}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, ctx: MilestoneContext) -> None:
        """Reset episode-level state (call at the start of each episode)."""
        for i, goal_cfg in enumerate(self._goal_configs):
            raw_potential = self._compute_goal_potential(goal_cfg, ctx)
            urgency = self._compute_urgency(goal_cfg, ctx.elapsed_time_hours)

            # Initial state dynamic potential Φ(s0, t0)
            self._prev_phis[i] = raw_potential * urgency

    def step(self, ctx: MilestoneContext) -> MilestoneResult:
        """Compute the DPBRS shaped reward for every goal.

        Returns
        -------
        MilestoneResult
            The shaped reward (total_delta) and per-goal breakdown.
        """
        per_goal: List[GoalMilestoneInfo] = []
        total_shaped_reward = 0.0
        total_phi = 0.0

        for i, goal_cfg in enumerate(self._goal_configs):
            # 1. Calculate raw spatial potential based on inventory
            raw_potential = self._compute_goal_potential(goal_cfg, ctx)

            # 2. Calculate current time-dependent urgency
            urgency = self._compute_urgency(goal_cfg, ctx.elapsed_time_hours)

            # 3. Calculate true dynamic potential Φ(s', t')
            current_phi = raw_potential * urgency

            # Terminal zeroing: Φ(s_T) = 0 by definition in DPBRS, so the
            # shaping sum telescopes exactly over finite-horizon episodes.
            # Whether this fires on a successful completion is decided by the
            # caller (gym env) via MilestoneConfig.terminal_zeroing; here we
            # simply honour the is_terminal flag it passes.
            if ctx.is_terminal:
                current_phi = 0.0

            # 4. Calculate SMDP DPBRS: F = γ_t * Φ(s', t') - Φ(s, t)
            shaped_reward = (ctx.gamma_t * current_phi) - self._prev_phis[i]

            info = GoalMilestoneInfo(
                resource_type=goal_cfg.resource_type,
                location=goal_cfg.location,
                raw_potential=raw_potential,
                urgency=urgency,
                phi=current_phi,
                shaped_reward=shaped_reward,
            )
            per_goal.append(info)

            total_shaped_reward += shaped_reward
            total_phi += current_phi

            # 6. Update state for the next step
            self._prev_phis[i] = current_phi

        self._initialised = True

        # Build flat logging breakdown
        breakdown = self._build_breakdown(total_shaped_reward, total_phi, per_goal)

        return MilestoneResult(
            total_shaped_reward=total_shaped_reward,
            total_potential=total_phi,
            per_goal=per_goal,
            breakdown=breakdown,
        )

    # ------------------------------------------------------------------
    # Potential computation
    # ------------------------------------------------------------------

    def _compute_goal_potential(
        self,
        goal: GoalConfig,
        ctx: MilestoneContext,
    ) -> float:
        """Compute the unweighted supply-chain potential for a single goal."""
        resource = goal.resource_type
        quantity = goal.quantity
        reward_per_unit = goal.reward_per_unit

        unit_stages: List[float] = []

        # --- Sites ---
        for site_name, inv in ctx.site_inventories.items():
            level = inv.get(resource, 0)
            if level <= 0:
                continue
            stage = self._site_stage(site_name)
            unit_stages.extend([stage] * level)

        # --- Vessels ---
        for vessel_name, inv in ctx.vessel_inventories.items():
            level = inv.get(resource, 0)
            if level <= 0:
                continue
            stage = self._vessel_stage(vessel_name)
            unit_stages.extend([stage] * level)

        # Sort descending so the best-positioned units are counted first
        unit_stages.sort(reverse=True)

        # Sum the top `quantity` stage values
        counted = min(len(unit_stages), quantity)
        potential = sum(unit_stages[:counted]) * reward_per_unit

        potential *= self._phi_scale

        return potential

    def _site_stage(self, site_name: str) -> float:
        """Return the stage-weight for a resource sitting at *site_name*."""
        role = self._site_roles.get(site_name)
        if role is None:
            return 0.0

        sw = self._cfg.stage_weights
        if role == SiteRole.SOURCE:
            return sw.at_source
        elif role == SiteRole.STAGING:
            return sw.at_staging
        elif role == SiteRole.INSTALLATION:
            return sw.at_goal
        return 0.0

    def _vessel_stage(self, vessel_name: str) -> float:
        """Return the stage-weight for a resource on *vessel_name*.

        The potential is determined solely by the vessel's role in the
        supply-chain relay, **not** by its current geographic location.
        This guarantees a strict DAG of potentials:

            on_heavy_lift < at_staging < on_feeder < on_installer
        """
        role = self._vessel_roles.get(vessel_name)
        if role is None:
            return 0.0

        sw = self._cfg.stage_weights
        if role == VesselRole.HEAVY_LIFT:
            return sw.on_heavy_lift
        elif role == VesselRole.FEEDER:
            return sw.on_feeder
        elif role == VesselRole.INSTALLER:
            return sw.on_installer
        return 0.0

    # ------------------------------------------------------------------
    # Urgency computation
    # ------------------------------------------------------------------

    def _compute_urgency(
        self,
        goal: GoalConfig,
        elapsed_hours: float,
    ) -> float:
        """Compute the urgency multiplier for *goal*."""
        if goal.deadline is None:
            return 1.0

        deadline_hours = goal.deadline
        elapsed = elapsed_hours
        remaining = max(deadline_hours - elapsed, 0.0)

        mode = self._cfg.urgency_mode
        scale = self._cfg.urgency_scale

        if mode == "linear":
            if goal.deadline <= 0:
                return 1.0 + scale
            fraction_elapsed = max(0.0, 1.0 - remaining / deadline_hours)
            return 1.0 + scale * fraction_elapsed

        elif mode == "inverse":
            if goal.deadline <= 0:
                return 1.0
            return self._d_max / goal.deadline

        else:
            logger.warning(f"Unknown urgency mode '{mode}', falling back to 1.0")
            return 1.0

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _build_breakdown(
        self,
        total_shaped_reward: float,
        total_phi: float,
        per_goal: List[GoalMilestoneInfo],
    ) -> Dict[str, Any]:
        """Build a flat dict suitable for WandB / TensorBoard logging."""
        bd: Dict[str, Any] = {
            "milestone/total_shaped_reward": total_shaped_reward,
            "milestone/total_phi": total_phi,
        }
        for info in per_goal:
            prefix = f"milestone/goal_{info.resource_type}@{info.location}"
            bd[f"{prefix}/raw_potential"] = info.raw_potential
            bd[f"{prefix}/urgency"] = info.urgency
            bd[f"{prefix}/phi"] = info.phi
            bd[f"{prefix}/shaped_reward"] = info.shaped_reward
        return bd
