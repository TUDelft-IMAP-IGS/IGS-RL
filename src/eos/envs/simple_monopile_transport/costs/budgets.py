"""Worst-case cost budget computation from config.

Provides a single source of truth for computing the maximum possible
accumulated cost for each component, given the time budget and cost
configuration. Used by:

- ``gym_env.py`` for observation normalisation and terminal cost evaluation.
- ``pfm.py`` for auto-resolving preference function bounds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def compute_cost_budgets(
    time_budget: float,
    cost_cfg: Any,
    sim_cfg: Any = None,
) -> tuple[float, float, float]:
    """Compute worst-case cost budgets from config alone.

    Returns static reference scales: the maximum possible accumulated
    cost for each component if the episode ran for the full time budget.

    Parameters
    ----------
    time_budget : float
        Maximum episode duration in hours.
    cost_cfg : CostConfig
        The cost configuration block.
    sim_cfg : SimConfig | None
        The simulation config (needed for storage capacity lookups).

    Returns
    -------
    travel_budget : float
        Worst-case travel cost.
    storage_budget : float
        Worst-case storage cost.
    total_worst_case_cost : float
        Global-weighted sum of worst-case costs across all enabled
        components (elapsed time + travel + storage).
    """
    # Elapsed time: rate × weight × max_hours
    elapsed_time_budget = 0.0
    if cost_cfg.elapsed_time.enabled:
        elapsed_time_budget = (
            time_budget * cost_cfg.elapsed_time.rate * cost_cfg.elapsed_time.weight
        )

    # Travel: sum of all vessel rates × max hours
    travel_budget = 0.0
    if cost_cfg.travel.enabled and cost_cfg.travel.rates:
        total_rate = sum(cost_cfg.travel.rates.values())
        travel_budget = time_budget * total_rate * cost_cfg.travel.weight

    # Storage: sum over all (site, resource) pairs of rate × capacity × max hours
    storage_budget = 0.0
    if cost_cfg.storage.enabled and cost_cfg.storage.rates:
        for site_name, resource_rates in cost_cfg.storage.rates.items():
            if not isinstance(resource_rates, Mapping):
                continue
            site_cfg = sim_cfg.sites.get(site_name) if sim_cfg else None
            for resource_name, rate in resource_rates.items():
                capacity = 0
                if site_cfg and resource_name in site_cfg.resource_types:
                    capacity = site_cfg.resource_types[resource_name].capacity
                elif site_cfg:
                    capacity = site_cfg.total_capacity
                storage_budget += (
                    rate * capacity * time_budget * cost_cfg.storage.weight
                )

    # Total worst-case cost (global weight applied)
    total_worst_case_cost = cost_cfg.weight * (
        elapsed_time_budget + travel_budget + storage_budget
    )

    return travel_budget, storage_budget, total_worst_case_cost


def compute_budget_by_info_key(
    time_budget: float,
    cost_cfg: Any,
    sim_cfg: Any = None,
) -> dict[str, float]:
    """Compute worst-case budget values keyed by PFM info_key.

    Returns a mapping from info_key (e.g. ``"elapsed_time_hours"``,
    ``"cost/travel"``) to the worst-case maximum value for that metric.

    Parameters
    ----------
    time_budget : float
        Maximum episode duration in hours.
    cost_cfg : CostConfig
        The cost configuration block.
    sim_cfg : SimConfig | None
        The simulation config (needed for storage capacity lookups).

    Returns
    -------
    dict[str, float]
        Mapping from info_key to worst-case value. Only includes keys
        for which a budget can be computed.
    """
    travel_budget, storage_budget, _ = compute_cost_budgets(
        time_budget, cost_cfg, sim_cfg
    )

    budgets: dict[str, float] = {"elapsed_time_hours": time_budget}

    if cost_cfg.travel.enabled and travel_budget > 0:
        budgets["cost/travel"] = travel_budget

    if cost_cfg.storage.enabled and storage_budget > 0:
        budgets["cost/storage"] = storage_budget

    if cost_cfg.elapsed_time.enabled:
        budgets["cost/elapsed_time"] = (
            time_budget * cost_cfg.elapsed_time.rate * cost_cfg.elapsed_time.weight
        )

    return budgets
