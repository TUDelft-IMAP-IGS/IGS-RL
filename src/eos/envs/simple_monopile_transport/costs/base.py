"""Base abstractions for modular cost components.

Every cost component inherits from :class:`CostComponent` and operates on a
:class:`CostContext` snapshot that the environment assembles each step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class CostContext:
    """Read-only snapshot of everything a cost component may need.

    Built by the environment at every step and handed to every active
    :class:`CostComponent`.

    Attributes
    ----------
    delta_time_hours : float
        Wall-clock hours that elapsed during this simulation step.
    vessel_positions : Dict[str, str]
        Mapping from vessel name to the site name it currently occupies.
    vessel_moved : Dict[str, bool]
        For each vessel, whether it changed position during this step.
    vessel_in_transit : Dict[str, bool]
        Maps vessel name to whether it currently has an active transit
        activity (from the ActivityTracker).
    just_completed_transit_durations : Dict[str, float]
        Maps vessel name to the duration (in hours) of a transit
        activity that just completed this step.
    prev_site_inventories : Dict[str, Dict[str, int]]
        Site inventories *before* this step's DES event.  Used by the
        storage cost component so that cost is computed against the
        level that held for the elapsed interval (left-Riemann sum).
        ``site_name -> {resource_name: level}``.
    site_inventories : Dict[str, Dict[str, int]]
        Site inventories *after* this step's DES event.
        ``site_name -> {resource_name: current_level}``.
    vessel_inventories : Dict[str, Dict[str, int]]
        ``vessel_name -> {resource_name: current_level}``.
    """

    delta_time_hours: float = 0.0
    vessel_positions: Dict[str, str] = field(default_factory=dict)
    vessel_moved: Dict[str, bool] = field(default_factory=dict)
    vessel_in_transit: Dict[str, bool] = field(default_factory=dict)
    just_completed_transit_durations: Dict[str, float] = field(default_factory=dict)
    prev_site_inventories: Dict[str, Dict[str, int]] = field(default_factory=dict)
    site_inventories: Dict[str, Dict[str, int]] = field(default_factory=dict)
    vessel_inventories: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass(slots=True)
class CostResult:
    """Output produced by a single :class:`CostComponent` for one step.

    Attributes
    ----------
    total : float
        Scalar cost for this component in this step (≥ 0).
    breakdown : Dict[str, Any]
        Arbitrary key/value pairs that give insight into *how* the total
        was computed (e.g. per-vessel or per-site sub-costs).
    """

    total: float = 0.0
    breakdown: Dict[str, Any] = field(default_factory=dict)


class CostComponent(ABC):
    """Interface that every pluggable cost component must implement.

    Subclasses are expected to:

    1.  Accept their own ``*Config`` dataclass in ``__init__``.
    2.  Implement :meth:`compute` (pure per-step logic).
    3.  Optionally accumulate state across steps and expose it via
        :attr:`accumulated_cost`.  :meth:`reset` clears that state.

    The :class:`~eos.envs.simple_monopile_transport.costs.aggregator.CostAggregator`
    iterates over all registered components each step and sums their costs.
    """

    def __init__(self) -> None:
        self._accumulated_cost: float = 0.0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique, human-readable identifier (used as key in breakdowns)."""
        ...

    @abstractmethod
    def compute(self, ctx: CostContext) -> CostResult:
        """Compute the cost for a single environment step.

        Parameters
        ----------
        ctx : CostContext
            Current step snapshot.

        Returns
        -------
        CostResult
            Non-negative cost and its breakdown.
        """
        ...

    # ------------------------------------------------------------------
    # Accumulation helpers
    # ------------------------------------------------------------------

    @property
    def accumulated_cost(self) -> float:
        """Total cost accumulated since the last :meth:`reset`."""
        return self._accumulated_cost

    def step(self, ctx: CostContext) -> CostResult:
        """Compute **and** accumulate the cost for this step.

        Prefer calling this over :meth:`compute` directly so that
        episode-level totals are tracked automatically.
        """
        result = self.compute(ctx)
        self._accumulated_cost += result.total
        return result

    def reset(self) -> None:
        """Reset internal accumulated state (called at episode boundaries)."""
        self._accumulated_cost = 0.0
