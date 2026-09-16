"""Modular cost components for the SMT environment.

Public API
----------
- :class:`CostContext` – per-step snapshot consumed by all components.
- :class:`CostResult` – single-component output for one step.
- :class:`CostComponent` – abstract base for pluggable cost components.
- :class:`CostAggregator` – owns and drives all active components.
- :class:`AggregatedCostResult` – unified output from the aggregator.
- :class:`ElapsedTimeCostComponent` – flat per-hour elapsed time cost.
- :class:`TravelCostComponent` – per-vessel hourly travel cost.
- :class:`StorageCostComponent` – per-resource per-site hourly storage cost.
"""

from .aggregator import AggregatedCostResult, CostAggregator
from .base import CostComponent, CostContext, CostResult
from .elapsed_time_cost import ElapsedTimeCostComponent
from .storage_cost import StorageCostComponent
from .travel_cost import TravelCostComponent

__all__ = [
    "AggregatedCostResult",
    "CostAggregator",
    "CostComponent",
    "CostContext",
    "CostResult",
    "ElapsedTimeCostComponent",
    "StorageCostComponent",
    "TravelCostComponent",
]
