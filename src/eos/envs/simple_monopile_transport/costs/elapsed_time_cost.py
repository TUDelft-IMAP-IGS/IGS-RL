"""Elapsed-time cost component – charges a flat hourly rate for every step.

This replaces the old ``use_time`` / ``time_weight`` mechanism with a proper
cost component that plugs into the modular cost system.  It penalises wall-clock
time regardless of what vessels or sites are doing, incentivising the agent to
find solutions that complete as quickly as possible.
"""

from __future__ import annotations

from typing import Any, Dict

from eos.config import ElapsedTimeCostConfig

from .base import CostComponent, CostContext, CostResult


class ElapsedTimeCostComponent(CostComponent):
    """Charge a flat hourly rate for every simulation step.

    Parameters
    ----------
    cfg : ElapsedTimeCostConfig
        Contains ``rate`` (cost per hour of elapsed simulation time) and a
        component-level ``weight`` multiplier.

    Notes
    -----
    The cost incurred each step is simply::

        rate * weight * delta_time_hours

    This is independent of vessel activity or inventory – it is a pure
    time-pressure signal that rewards faster episode completion.
    """

    def __init__(self, cfg: ElapsedTimeCostConfig) -> None:
        super().__init__()
        self._cfg = cfg

    # ------------------------------------------------------------------
    # CostComponent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "elapsed_time_cost"

    def compute(self, ctx: CostContext) -> CostResult:
        """Compute the elapsed-time cost for a single step.

        Returns
        -------
        CostResult
            ``total`` is the flat time cost; ``breakdown`` contains the
            delta hours and the configured rate for transparency.
        """
        delta_hours = max(0.0, ctx.delta_time_hours)
        if delta_hours == 0.0:
            return CostResult(
                total=0.0, breakdown={"delta_hours": 0.0, "rate": self._cfg.rate}
            )

        cost = self._cfg.rate * self._cfg.weight * delta_hours

        breakdown: Dict[str, Any] = {
            "delta_hours": delta_hours,
            "rate": self._cfg.rate,
            "cost": cost,
        }

        return CostResult(total=cost, breakdown=breakdown)
