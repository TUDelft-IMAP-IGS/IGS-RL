"""Metric utilities for aggregating environment info and rollout stats.

Provides helpers shared across experiment types:

* :func:`aggregate_vector_info` — collapse vectorised env info dicts into
  scalar metrics.
* :func:`safe_mean` — mean that gracefully handles empty lists.
* :func:`create_action_trace_table` — convert an action-trace list into a
  WandB ``Table`` for logging.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def aggregate_vector_info(
    infos: Dict[str, Any], prefix: str = "env/"
) -> Dict[str, float]:
    """Aggregate vectorized env info dict into scalar metrics.

    Parameters
    ----------
    infos : Dict[str, Any]
        Info dict returned by vectorized environments (values may be arrays).
    prefix : str
        Prefix to apply to metric names.

    Returns
    -------
    Dict[str, float]
        Aggregated scalar metrics (mean over envs).
    """
    metrics: Dict[str, float] = {}
    for key, value in infos.items():
        if key in (
            "final_info",
            "final_observation",
            "_episode",
            "structured_obs",
            "goal_info",
            "action_masks",
            "vessel_availability",
        ):
            continue
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            continue

        mask_key = f"_{key}"
        mask = infos.get(mask_key)

        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=np.float64)
            if arr.dtype == object or arr.size == 0:
                continue

            if mask is not None:
                mask_arr = np.asarray(mask, dtype=bool)
                if mask_arr.shape == arr.shape:
                    arr = arr[mask_arr]
                elif mask_arr.shape[0] == arr.shape[0]:
                    arr = arr[mask_arr]

            # Filter NaN (e.g. PFM metrics are NaN for non-terminal envs)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                continue

            metrics[f"{prefix}{key}"] = float(np.mean(arr))
            continue

        if isinstance(value, (int, float, np.floating, np.integer, bool)):
            fval = float(value)
            if np.isnan(fval):
                continue
            metrics[f"{prefix}{key}"] = fval

    return metrics


def safe_mean(values: list[float]) -> float | None:
    """Return mean if values are present, else None."""
    if not values:
        return None
    return float(np.mean(values))


def create_action_trace_table(
    action_trace: list[dict[str, str | int]],
):
    """Convert an action-trace list of dicts into a WandB ``Table``.

    Each dict should contain at least ``step``, ``vessel``,
    ``vessel_location``, and ``action`` keys.

    Parameters
    ----------
    action_trace : list[dict]
        One dict per vessel-action row collected during an episode.

    Returns
    -------
    wandb.Table | None
        A WandB table ready for logging, or ``None`` if the trace is empty.
    """
    if not action_trace:
        return None

    import wandb

    columns = [
        "step",
        "acting_order",
        "activity_id",
        "depends_on",
        "vessel",
        "vessel_status",
        "vessel_location",
        "action",
        "elapsed_hours",
        "delta_hours",
    ]
    data = [[row.get(col) for col in columns] for row in action_trace]
    return wandb.Table(data=data, columns=columns)
