"""Storage cost component – charges per resource unit stored at each site.

For every simulation step, each site is inspected and a cost is incurred
based on how many units of each resource type were stored there during
the elapsed interval.  The inventory level used is the **pre-event**
snapshot (``prev_site_inventories``) so that the cost reflects the level
that actually held for the time that passed — a correct left-Riemann sum
of the continuous storage integral.

The cost rate is configurable per site and per resource type, allowing
fine-grained control (e.g. storing large monopiles at the marshalling yard
is more expensive than at the fabrication yard).
"""

from __future__ import annotations

from typing import Any, Dict

from eos.config import StorageCostConfig

from .base import CostComponent, CostContext, CostResult


class StorageCostComponent(CostComponent):
    """Charge a configurable hourly rate per resource unit stored at a site.

    Parameters
    ----------
    cfg : StorageCostConfig
        Contains ``rates`` – a nested mapping
        ``{site_name: {resource_name: cost_per_unit_per_hour}}`` –
        and a component-level ``weight`` multiplier.

    Notes
    -----
    *  Sites or resources not listed in ``rates`` are assumed to have zero
       storage cost, so configs can remain sparse.
    *  The cost for a single site/resource pair in one step is::

           rate[site][resource] * weight * quantity * delta_time_hours

       where ``quantity`` is the inventory level **before** this step's
       DES event (i.e. the level that held for the elapsed interval).
    """

    def __init__(self, cfg: StorageCostConfig) -> None:
        super().__init__()
        self._cfg = cfg
        # Pre-build a fast nested lookup; missing entries default to 0.
        self._rates: Dict[str, Dict[str, float]] = (
            {site: dict(resources) for site, resources in cfg.rates.items()}
            if cfg.rates
            else {}
        )

    # ------------------------------------------------------------------
    # CostComponent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        return "storage_cost"

    def compute(self, ctx: CostContext) -> CostResult:
        """Compute storage cost for a single step.

        For every ``(site, resource)`` pair that has a configured rate and
        a non-zero inventory level, the cost is::

            rate * weight * level * delta_hours

        The ``level`` is taken from ``prev_site_inventories`` — the
        snapshot *before* this step's DES event — so the cost is charged
        for the level that actually held during the elapsed interval.
        Falls back to ``site_inventories`` when the pre-event snapshot
        is not available (e.g. first step after reset).

        Returns
        -------
        CostResult
            ``total`` is the sum over all site/resource pairs; ``breakdown``
            contains per-site dictionaries keyed by resource name, plus a
            per-site subtotal and storage unit-hour metrics.
        """
        delta_hours = max(0.0, ctx.delta_time_hours)
        if delta_hours == 0.0:
            return CostResult(
                total=0.0,
                breakdown={
                    "per_site": {},
                    "delta_hours": 0.0,
                    "storage_unit_hours_total": 0.0,
                    "storage_unit_hours_by_site": {},
                },
            )

        per_site: Dict[str, Dict[str, Any]] = {}
        storage_unit_hours_by_site: Dict[str, float] = {}
        total = 0.0
        storage_unit_hours_total = 0.0

        # Use the pre-event inventory snapshot so the cost reflects
        # the level that held for the elapsed interval (left-Riemann).
        # Fall back to post-event inventories when prev is not populated.
        inventories = ctx.prev_site_inventories or ctx.site_inventories

        for site_name, resource_levels in inventories.items():
            site_rates = self._rates.get(site_name)
            if site_rates is None:
                # No rates configured for this site – skip entirely.
                continue

            site_breakdown: Dict[str, Any] = {}
            site_total = 0.0
            site_storage_unit_hours = 0.0

            for resource_name, level in resource_levels.items():
                rate = site_rates.get(resource_name, 0.0)
                if rate == 0.0 or level <= 0:
                    continue

                storage_unit_hours = level * delta_hours
                cost = rate * self._cfg.weight * level * delta_hours
                site_breakdown[resource_name] = {
                    "level": level,
                    "rate": rate,
                    "storage_unit_hours": storage_unit_hours,
                    "cost": cost,
                }
                site_total += cost
                site_storage_unit_hours += storage_unit_hours

            if site_total > 0.0:
                site_breakdown["subtotal"] = site_total
                site_breakdown["storage_unit_hours_subtotal"] = site_storage_unit_hours
                per_site[site_name] = site_breakdown
                storage_unit_hours_by_site[site_name] = site_storage_unit_hours
                total += site_total
                storage_unit_hours_total += site_storage_unit_hours

        breakdown: Dict[str, Any] = {
            "per_site": per_site,
            "delta_hours": delta_hours,
            "storage_unit_hours_total": storage_unit_hours_total,
            "storage_unit_hours_by_site": storage_unit_hours_by_site,
        }

        return CostResult(total=total, breakdown=breakdown)
