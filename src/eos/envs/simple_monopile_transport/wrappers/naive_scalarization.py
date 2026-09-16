"""Naive scalarization baseline — weighted sum of raw objectives without normalization.

This wrapper serves as baseline B3 in the experiment plan. It replaces the
terminal reward with a simple weighted sum of raw objective values read from
the info dict, demonstrating what happens when objectives with incommensurate
scales are combined without PFM's Z-score normalization.

Non-terminal steps pass through DPBRS milestone shaping unchanged (same
convention as PFMVectorWrapper).
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from loguru import logger


@dataclass
class NaiveObjective:
    """Single objective specification for naive scalarization."""

    name: str
    weight: float
    info_key: str
    direction: str = "minimize"  # "minimize" or "maximize"


class NaiveScalarizationWrapper(gym.vector.VectorWrapper):
    """Replace terminal reward with raw weighted sum of objectives.

    Parameters
    ----------
    env : gym.vector.VectorEnv
        The vectorized environment to wrap.
    objectives : list[NaiveObjective]
        Objective specifications (must match info dict keys).
    completion_baseline : float
        Constant added to the scalarized reward on successful completion
        (ensures reward is positive enough for stable learning).
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        objectives: list[NaiveObjective],
        completion_baseline: float = 10.0,
    ):
        super().__init__(env)
        self._objectives = objectives
        self._completion_baseline = completion_baseline
        logger.info(
            f"NaiveScalarizationWrapper active with {len(objectives)} objectives: "
            f"{[(o.name, o.weight) for o in objectives]}"
        )

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        done = np.logical_or(terminated, truncated)

        for i in range(self.num_envs):
            if not done[i]:
                continue

            # Check for success
            is_success = infos.get("is_success", np.zeros(self.num_envs, dtype=bool))
            if isinstance(is_success, (list, tuple)):
                is_success = np.array(is_success)
            success_i = (
                bool(is_success[i])
                if hasattr(is_success, "__getitem__")
                else bool(is_success)
            )

            if not success_i:
                # Failed episode — zero reward (no information)
                rewards[i] = 0.0
                continue

            # Compute naive weighted sum of raw objectives
            scalarized = 0.0
            for obj in self._objectives:
                raw_val = self._extract_info_value(infos, obj.info_key, i)
                if raw_val is None:
                    continue
                # For minimization objectives, negate so that lower = better reward
                sign = -1.0 if obj.direction == "minimize" else 1.0
                scalarized += obj.weight * sign * raw_val

            rewards[i] = self._completion_baseline + scalarized

        return obs, rewards, terminated, truncated, infos

    def _extract_info_value(self, infos: dict, key: str, env_idx: int) -> float | None:
        """Extract a scalar value from vectorized infos for a specific env."""
        if key not in infos:
            return None
        val = infos[key]
        if isinstance(val, np.ndarray):
            return float(val[env_idx])
        elif isinstance(val, (list, tuple)):
            return float(val[env_idx])
        elif isinstance(val, (int, float)):
            return float(val)
        return None
