"""Cost aggregator – manages the lifecycle of all active cost components.

The :class:`CostAggregator` is the single entry-point that the reward system
uses to obtain cost information each step.  It instantiates the appropriate
:class:`~.base.CostComponent` instances based on the Hydra config, delegates
:meth:`step` calls, and collects results into a unified breakdown dict.

Adding a new cost component requires:

1.  Creating a new ``CostComponent`` subclass (see ``travel_cost.py`` as a template).
2.  Adding its config dataclass to ``eos.config``.
3.  Registering it in :meth:`CostAggregator._build_components`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger

from eos.config import CostConfig

from .base import CostComponent, CostContext, CostResult
from .elapsed_time_cost import ElapsedTimeCostComponent
from .storage_cost import StorageCostComponent
from .travel_cost import TravelCostComponent


class AggregatedCostResult:
    """Unified result returned by :meth:`CostAggregator.step`.

    Attributes
    ----------
    total : float
        Weighted sum of all component costs for this step.
    component_results : Dict[str, CostResult]
        Per-component results keyed by :attr:`CostComponent.name`.
    breakdown : Dict[str, Any]
        Flat dictionary suitable for logging / WandB, containing per-component
        totals and their detailed breakdowns.
    """

    __slots__ = ("total", "component_results", "breakdown")

    def __init__(
        self,
        total: float,
        component_results: Dict[str, CostResult],
        breakdown: Dict[str, Any],
    ) -> None:
        self.total = total
        self.component_results = component_results
        self.breakdown = breakdown


class CostAggregator:
    """Owns all active :class:`CostComponent` instances and drives them each step.

    Parameters
    ----------
    cfg : CostConfig
        Top-level cost configuration.  The aggregator inspects each
        sub-config's ``enabled`` flag to decide which components to
        instantiate.
    """

    def __init__(self, cfg: CostConfig) -> None:
        self._cfg = cfg
        self._components: List[CostComponent] = self._build_components(cfg)
        self._cost_watermark: float = 0.0
        self._component_watermarks: Dict[str, float] = {
            c.name: 0.0 for c in self._components
        }

        if self._components:
            names = [c.name for c in self._components]
            logger.debug(f"CostAggregator initialised with components: {names}")
        else:
            logger.debug("CostAggregator initialised with no active cost components.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def components(self) -> List[CostComponent]:
        """Return the list of active cost components (read-only view)."""
        return list(self._components)

    @property
    def global_weight(self) -> float:
        """Global cost multiplier applied on top of component weights."""
        return self._cfg.weight

    def step(self, ctx: CostContext) -> AggregatedCostResult:
        """Compute and accumulate costs for every active component.

        Parameters
        ----------
        ctx : CostContext
            Snapshot of the current environment state.

        Returns
        -------
        AggregatedCostResult
            Aggregated cost (with global weight applied) and per-component
            breakdowns for logging / diagnostics.
        """
        component_results: Dict[str, CostResult] = {}
        unweighted_total = 0.0

        for component in self._components:
            result = component.step(ctx)
            component_results[component.name] = result
            unweighted_total += result.total

        weighted_total = self._cfg.weight * unweighted_total

        # Build a flat breakdown dict for easy logging
        breakdown: Dict[str, Any] = {
            "cost_total": weighted_total,
            "cost_total_unweighted": unweighted_total,
            "cost_global_weight": self._cfg.weight,
        }

        for comp_name, result in component_results.items():
            breakdown[f"cost/{comp_name}/total"] = result.total
            breakdown[f"cost/{comp_name}/accumulated"] = (
                self._get_component_accumulated(comp_name)
            )
            breakdown[f"cost/{comp_name}/breakdown"] = result.breakdown

        return AggregatedCostResult(
            total=weighted_total,
            component_results=component_results,
            breakdown=breakdown,
        )

    def reset(self) -> None:
        """Reset all components (call at episode boundaries)."""
        for component in self._components:
            component.reset()
        self._cost_watermark = 0.0
        self._component_watermarks = {c.name: 0.0 for c in self._components}

    def get_accumulated_costs(self) -> Dict[str, float]:
        """Return accumulated cost per component since last reset.

        Returns
        -------
        Dict[str, float]
            Mapping from component name to its accumulated cost.
        """
        return {c.name: c.accumulated_cost for c in self._components}

    def get_total_accumulated_cost(self) -> float:
        """Return the global-weighted sum of all accumulated component costs."""
        return self._cfg.weight * sum(c.accumulated_cost for c in self._components)

    def mark(self) -> None:
        """Set a cost watermark at the current accumulated total.

        Used by the reward system to track cost incurred between progress
        events.  Call this after applying the cost so that the next call
        to :meth:`get_cost_since_mark` returns only the new cost.
        """
        self._cost_watermark = self.get_total_accumulated_cost()
        for c in self._components:
            self._component_watermarks[c.name] = c.accumulated_cost * self._cfg.weight

    def get_cost_since_mark(self) -> float:
        """Return accumulated cost since the last :meth:`mark` (or :meth:`reset`).

        This is the delta between the current total accumulated cost and
        the watermark set by the most recent :meth:`mark` call.
        """
        return self.get_total_accumulated_cost() - self._cost_watermark

    def get_costs_since_mark(self) -> Dict[str, float]:
        """Return per-component accumulated cost since the last :meth:`mark`.

        Each value is the global-weighted delta between the component's
        current accumulated cost and its watermark set by :meth:`mark`.

        Returns
        -------
        Dict[str, float]
            Mapping from component name to its cost since the last mark.
        """
        return {
            c.name: (c.accumulated_cost * self._cfg.weight)
            - self._component_watermarks.get(c.name, 0.0)
            for c in self._components
        }

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_components(self, cfg: CostConfig) -> List[CostComponent]:
        """Instantiate cost components based on configuration flags.

        To register a new component:
        1.  Add its config dataclass to :mod:`eos.config`.
        2.  Add a field to :class:`CostConfig`.
        3.  Add an ``if`` block here that checks ``enabled`` and instantiates
            the component.

        Parameters
        ----------
        cfg : CostConfig
            The full cost configuration tree.

        Returns
        -------
        List[CostComponent]
            Only the components whose ``enabled`` flag is ``True``.
        """
        components: List[CostComponent] = []

        # -- Elapsed time cost --------------------------------------------
        if cfg.elapsed_time.enabled:
            components.append(ElapsedTimeCostComponent(cfg.elapsed_time))
            logger.debug(
                f"Elapsed time cost enabled – rate: {cfg.elapsed_time.rate}/hour"
            )

        # -- Travel cost --------------------------------------------------
        if cfg.travel.enabled:
            components.append(TravelCostComponent(cfg.travel))
            logger.debug(f"Travel cost enabled – rates: {dict(cfg.travel.rates)}")

        # -- Storage cost -------------------------------------------------
        if cfg.storage.enabled:
            components.append(StorageCostComponent(cfg.storage))
            logger.debug(f"Storage cost enabled – rates: {dict(cfg.storage.rates)}")

        # -- [Future components go here] ----------------------------------

        return components

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_component_accumulated(self, name: str) -> float:
        """Look up a component's accumulated cost by name."""
        for c in self._components:
            if c.name == name:
                return c.accumulated_cost
        return 0.0
