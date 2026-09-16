from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class FlattenField:
    """Single scalar entry in the flattened observation."""

    name: str
    extractor: Callable[[Mapping[str, object]], float]


@dataclass(frozen=True)
class FlattenObservationConfig:
    """Config for building a flattened observation representation.

    Notes
    -----
    - Field names are used for debug/inspection and as keys for preprocessors.
    - Preprocessors are applied per field right before value is written.
    """

    global_fields: Sequence[str] = (
        "current_time",
        "num_pending_tasks",
        "in_no_install_window",
        "time_to_install_window_flip",
    )
    vessel_prefix_fields: Sequence[str] = ("position", "status")
    site_resource_fields: Sequence[str] = ("capacity", "load")
    vessel_resource_fields: Sequence[str] = ("capacity", "load")
    vessel_intent_fields: Sequence[str] = ("action_type", "destination", "resource_id")
    goal_fields: Sequence[str] = (
        "required",
        "installed",
        "remaining",
        "progress",
        "deadline",
        "is_failed",
        "is_blocked",
        "dep_progress",
    )
    goal_categorical_fields: Sequence[str] = ("target_site", "resource_type")
    include_goal_id: bool = False
    include_site_id: bool = False
    include_vessel_id: bool = False
    site_resource_order: Sequence[str] | None = None
    vessel_resource_order: Sequence[str] | None = None
    dtype: np.dtype = np.float64
    preprocessors: Mapping[str, Callable[[float], float]] = field(default_factory=dict)


class FlattenObservationWrapper(gym.ObservationWrapper):
    """Flatten structured observations into a single vector with a stable order.

    The order matches:
    1) global fields
    2) per-site stock (per resource)
    3) per-vessel state + inventory (per resource)
    """

    def __init__(self, env: gym.Env, config: FlattenObservationConfig | None = None):
        super().__init__(env)
        self._env_unwrapped = env.unwrapped
        self._config = config or FlattenObservationConfig()

        self._num_sites = self._infer_num_items("sites")
        self._num_vessels = self._infer_num_items("vessels")
        self._num_goals = self._infer_num_items("goals")
        self._resource_names = self._infer_resource_names()

        self._site_resource_order = (
            list(self._config.site_resource_order)
            if self._config.site_resource_order is not None
            else list(self._resource_names)
        )
        self._vessel_resource_order = (
            list(self._config.vessel_resource_order)
            if self._config.vessel_resource_order is not None
            else list(self._resource_names)
        )

        self._fields = self._build_fields()
        self.field_names: List[str] = [field.name for field in self._fields]

        dim = len(self._fields)
        low = np.full((dim,), -np.inf, dtype=self._config.dtype)
        high = np.full((dim,), np.inf, dtype=self._config.dtype)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=self._config.dtype
        )

    def observation(self, observation: Mapping[str, object]) -> np.ndarray:
        values = np.empty(len(self._fields), dtype=self._config.dtype)
        for i, field in enumerate(self._fields):
            values[i] = field.extractor(observation)
        return values

    def _build_fields(self) -> List[FlattenField]:
        fields: List[FlattenField] = []

        # Global fields
        for key in self._config.global_fields:
            name = f"global.{key}"
            fields.append(
                FlattenField(
                    name=name,
                    extractor=lambda obs, k=key, n=name: self._apply_preprocess(
                        n, self._to_scalar(obs["global_obs"][k])
                    ),
                )
            )

        # Sites
        for site_idx in range(self._num_sites):
            if self._config.include_site_id:
                name = f"site[{site_idx}].id"
                fields.append(
                    FlattenField(
                        name=name,
                        extractor=lambda obs, idx=site_idx, n=name: (
                            self._apply_preprocess(
                                n, self._to_scalar(obs["sites"][idx]["id"])
                            )
                        ),
                    )
                )

            for resource_name in self._site_resource_order:
                for field_name in self._config.site_resource_fields:
                    name = f"site[{site_idx}].stock.{resource_name}.{field_name}"
                    fields.append(
                        FlattenField(
                            name=name,
                            extractor=lambda obs, idx=site_idx, r=resource_name, f=field_name, n=name: (
                                self._apply_preprocess(
                                    n,
                                    self._to_scalar(obs["sites"][idx]["stock"][r][f]),
                                )
                            ),
                        )
                    )

        # Vessels
        for vessel_idx in range(self._num_vessels):
            if self._config.include_vessel_id:
                name = f"vessel[{vessel_idx}].id"
                fields.append(
                    FlattenField(
                        name=name,
                        extractor=lambda obs, idx=vessel_idx, n=name: (
                            self._apply_preprocess(
                                n, self._to_scalar(obs["vessels"][idx]["id"])
                            )
                        ),
                    )
                )

            for key in self._config.vessel_prefix_fields:
                size = self._get_vessel_field_size(key)
                if size <= 1:
                    name = f"vessel[{vessel_idx}].{key}"
                    fields.append(
                        FlattenField(
                            name=name,
                            extractor=lambda obs, idx=vessel_idx, k=key, n=name: (
                                self._apply_preprocess(
                                    n, self._to_scalar(obs["vessels"][idx][k])
                                )
                            ),
                        )
                    )
                else:
                    for j in range(size):
                        name = f"vessel[{vessel_idx}].{key}[{j}]"
                        fields.append(
                            FlattenField(
                                name=name,
                                extractor=lambda obs, idx=vessel_idx, k=key, j=j, n=name: (
                                    self._apply_preprocess(
                                        n,
                                        self._vector_entry(obs["vessels"][idx][k], j),
                                    )
                                ),
                            )
                        )

            for resource_name in self._vessel_resource_order:
                for field_name in self._config.vessel_resource_fields:
                    name = (
                        f"vessel[{vessel_idx}].inventory.{resource_name}.{field_name}"
                    )
                    fields.append(
                        FlattenField(
                            name=name,
                            extractor=lambda obs, idx=vessel_idx, r=resource_name, f=field_name, n=name: (
                                self._apply_preprocess(
                                    n,
                                    self._to_scalar(
                                        obs["vessels"][idx]["inventory"][r][f]
                                    ),
                                )
                            ),
                        )
                    )

            # Intent fields (action_type, destination, resource_id)
            for intent_key in self._config.vessel_intent_fields:
                name = f"vessel[{vessel_idx}].intent.{intent_key}"
                fields.append(
                    FlattenField(
                        name=name,
                        extractor=lambda obs, idx=vessel_idx, k=intent_key, n=name: (
                            self._apply_preprocess(
                                n,
                                self._to_scalar(
                                    obs["vessels"][idx].get("intent", {}).get(k, -1)
                                ),
                            )
                        ),
                    )
                )

        # Goals
        for goal_idx in range(self._num_goals):
            if self._config.include_goal_id:
                name = f"goal[{goal_idx}].id"
                fields.append(
                    FlattenField(
                        name=name,
                        extractor=lambda obs, idx=goal_idx, n=name: (
                            self._apply_preprocess(
                                n, self._to_scalar(obs["goals"][idx]["id"])
                            )
                        ),
                    )
                )

            for cat_key in self._config.goal_categorical_fields:
                name = f"goal[{goal_idx}].{cat_key}"
                fields.append(
                    FlattenField(
                        name=name,
                        extractor=lambda obs, idx=goal_idx, k=cat_key, n=name: (
                            self._apply_preprocess(
                                n, self._to_scalar(obs["goals"][idx][k])
                            )
                        ),
                    )
                )

            for field_key in self._config.goal_fields:
                name = f"goal[{goal_idx}].{field_key}"
                fields.append(
                    FlattenField(
                        name=name,
                        extractor=lambda obs, idx=goal_idx, k=field_key, n=name: (
                            self._apply_preprocess(
                                n, self._to_scalar(obs["goals"][idx][k])
                            )
                        ),
                    )
                )

        return fields

    def _apply_preprocess(self, name: str, value: float) -> float:
        func = self._config.preprocessors.get(name)
        return func(value) if func is not None else value

    def _get_vessel_field_size(self, key: str) -> int:
        space = self.env.observation_space
        if isinstance(space, spaces.Dict) and "vessels" in space.spaces:
            vessels_space = space.spaces["vessels"]
            if isinstance(vessels_space, spaces.Tuple) and vessels_space.spaces:
                vessel_space = vessels_space.spaces[0]
                if isinstance(vessel_space, spaces.Dict) and key in vessel_space.spaces:
                    field_space = vessel_space.spaces[key]
                    if isinstance(field_space, spaces.Discrete):
                        return 1
                    if isinstance(field_space, spaces.MultiBinary):
                        return int(field_space.n)
                    if isinstance(field_space, spaces.Box):
                        size = int(np.prod(field_space.shape))
                        return size if size > 0 else 1
        return 1

    def _infer_num_items(self, key: str) -> int:
        space = self.env.observation_space
        if isinstance(space, spaces.Dict) and key in space.spaces:
            sub = space.spaces[key]
            if isinstance(sub, spaces.Tuple):
                return len(sub.spaces)
        return 0

    def _infer_resource_names(self) -> List[str]:
        if hasattr(self._env_unwrapped, "_resource_names"):
            return list(self._env_unwrapped._resource_names)

        space = self.env.observation_space
        if isinstance(space, spaces.Dict) and "sites" in space.spaces:
            sites_space = space.spaces["sites"]
            if isinstance(sites_space, spaces.Tuple) and sites_space.spaces:
                site_space = sites_space.spaces[0]
                if (
                    isinstance(site_space, spaces.Dict)
                    and "stock" in site_space.spaces
                    and isinstance(site_space.spaces["stock"], spaces.Dict)
                ):
                    return list(site_space.spaces["stock"].spaces.keys())

        return []

    @staticmethod
    def _to_scalar(value: object) -> float:
        if isinstance(value, np.ndarray):
            return float(value.reshape(-1)[0]) if value.size else 0.0
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        return float(value)

    @staticmethod
    def _vector_entry(value: object, index: int) -> float:
        if isinstance(value, np.ndarray):
            flat = value.reshape(-1)
            if 0 <= index < flat.size:
                return float(flat[index])
            return 0.0
        if isinstance(value, (list, tuple)):
            if 0 <= index < len(value):
                return float(value[index])
            return 0.0
        return 0.0
