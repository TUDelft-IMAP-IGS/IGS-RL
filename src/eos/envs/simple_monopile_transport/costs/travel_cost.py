"""Travel cost component – charges a per-vessel hourly rate while travelling.

Each vessel can have its own rate (e.g. Bokalift is more expensive than HTV).

Travel detection uses the :class:`ActivityTracker` information provided via
:class:`CostContext` rather than the old position-change heuristic.

When a move activity **completes** during a step, the component charges the
vessel's configured rate for the **full activity duration** (as set by the
:class:`ActivityBuilder`).  This guarantees that every move is costed for
exactly the duration that was configured, regardless of how many DES events
were interleaved during transit.

No cost is charged on intermediate steps while the vessel is still in
transit.  This avoids double-counting: the full duration is charged once
on completion, which is the ground-truth travel time known from the
activity itself.
"""

from __future__ import annotations

from typing import Any, Dict

from eos.config import TravelCostConfig

from .base import CostComponent, CostContext, CostResult


class TravelCostComponent(CostComponent):
    """Charge a configurable hourly rate for every vessel that travels.

    Parameters
    ----------
    cfg : TravelCostConfig
        Contains ``rates`` – a mapping from vessel name to cost-per-hour –
        and a component-level ``weight`` multiplier.

    Notes
    -----
    *  If a vessel is not listed in ``rates`` it is assumed to have zero
       travel cost (a warning is **not** emitted so that configs can stay
       sparse).
    *  Transit detection relies on ``just_completed_transit_durations``
       from :class:`CostContext`, which is populated from the
       :class:`ActivityTracker`.  The full activity duration is charged
       in a single step when the move completes.
    """

    def __init__(self, cfg: TravelCostConfig) -> None:
        super().__init__()
        self._cfg = cfg
        # Pre-build a fast lookup; missing vessels default to 0.
        self._rates: Dict[str, float] = dict(cfg.rates) if cfg.rates else {}

    # ------------------------------------------------------------------
    # CostComponent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        return "travel_cost"

    def compute(self, ctx: CostContext) -> CostResult:
        """Compute travel cost for a single step.

        Cost is only incurred when a transit activity **completes**.  The
        full activity duration is used::

            rate[vessel] * weight * activity_duration_hours

        Returns
        -------
        CostResult
            ``total`` is the sum over all vessels; ``breakdown`` contains
            per-vessel cost and travel-hour metrics keyed by vessel name.
        """
        delta_hours = max(0.0, ctx.delta_time_hours)

        per_vessel_cost: Dict[str, float] = {}
        per_vessel_hours: Dict[str, float] = {}
        total = 0.0
        total_travel_hours = 0.0

        # Collect all vessel names we know about from any of the context
        # dicts so that every vessel appears in the breakdown.
        all_vessels: set[str] = set()
        all_vessels.update(ctx.vessel_in_transit.keys())
        all_vessels.update(ctx.just_completed_transit_durations.keys())
        all_vessels.update(ctx.vessel_moved.keys())

        for vessel_name in all_vessels:
            completed_duration_hours = ctx.just_completed_transit_durations.get(
                vessel_name, 0.0
            )

            if completed_duration_hours > 0.0:
                # Transit just completed – charge the full move duration.
                travel_hours = completed_duration_hours
            else:
                # Vessel did not complete a transit this step.
                per_vessel_cost[vessel_name] = 0.0
                per_vessel_hours[vessel_name] = 0.0
                continue

            rate = self._rates.get(vessel_name, 0.0)
            cost = rate * self._cfg.weight * travel_hours

            per_vessel_cost[vessel_name] = cost
            per_vessel_hours[vessel_name] = travel_hours
            total += cost
            total_travel_hours += travel_hours

        breakdown: Dict[str, Any] = {
            "per_vessel": per_vessel_cost,
            "per_vessel_hours": per_vessel_hours,
            "total_travel_hours": total_travel_hours,
            "delta_hours": delta_hours,
        }

        return CostResult(total=total, breakdown=breakdown)
