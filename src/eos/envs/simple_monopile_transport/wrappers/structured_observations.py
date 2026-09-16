from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# TODO: Consider how to handle IDS of entities
#  - Resources should have something like IDs in the unique case
#  - Vessels should perhaps have something like a type which distingushed between HTVs, bokalifts etc.
#  - For now we use ID but to make the architecture size agnostic and do something like curriculum learning use type as embedding


@dataclass(frozen=True)
class EntityField:
    """Defines how to extract a single feature column for an entity."""

    name: str
    # Extractor takes the specific entity dict (e.g. vessel_dict), not the full obs
    extractor: Callable[[Mapping[str, object]], float]


@dataclass(frozen=True)
class StructuredObservationConfig:
    """Config for building a structured (Entity x Feature) observation."""

    # TODO: Make default be to use all of them
    global_fields: Sequence[str] = (
        "current_time",
        "num_pending_tasks",
        "step",
        "in_no_install_window",
        "time_to_install_window_flip",
        "accumulated_travel_cost",
        "accumulated_storage_cost",
    )
    vessel_prefix_fields: Sequence[str] = ("position", "status")
    site_resource_fields: Sequence[str] = ("capacity", "load", "next_available_in")
    vessel_resource_fields: Sequence[str] = ("capacity", "load")

    # Intent sub-fields extracted from each vessel's ``intent`` dict.
    # Each entry becomes a separate column in the vessel feature matrix.
    vessel_intent_fields: Sequence[str] = ("action_type", "destination", "resource_id")

    site_categorical_fields: Sequence[str] = ("role",)
    vessel_categorical_fields: Sequence[str] = (
        "position",
        "status",
        "intent.action_type",
        "intent.destination",
        "intent.resource_id",
    )

    # Site prefix fields extracted before per-resource columns.
    site_prefix_fields: Sequence[str] = ("role",)

    # Goal observation fields (continuous scalars).
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

    # Goal categorical fields (embedded via nn.Embedding in the model).
    goal_categorical_fields: Sequence[str] = (
        "id",
        "target_site",
        "resource_type",
    )

    include_site_id: bool = True
    include_vessel_id: bool = True
    include_goal_id: bool = True
    include_visit_modes: bool = (
        False  # TODO: This should be turned on if we switch to using business rules
    )

    site_resource_order: Sequence[str] | None = None
    vessel_resource_order: Sequence[str] | None = None

    dtype: np.dtype = np.float32
    preprocessors: Mapping[str, Callable[[float], float]] = field(default_factory=dict)


class StructuredObservationWrapper(gym.ObservationWrapper):
    """
    Transforms observations into a 2D Tensor: (N_Entities, Max_Features).

    Structure:
    - Row 0: Global Entity
    - Rows 1..N: Site Entities
    - Rows N+1..M: Vessel Entities
    """

    def __init__(self, env: gym.Env, config: StructuredObservationConfig | None = None):
        super().__init__(env)
        self._env_unwrapped = env.unwrapped
        self._config = config or StructuredObservationConfig()

        # 1. Infer Counts
        self._num_sites = self._infer_num_items("sites")
        self._num_vessels = self._infer_num_items("vessels")
        self._resource_names = self._infer_resource_names()
        self._num_goals = self._infer_num_items("goals")

        # 2. Setup Order
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

        # 3. Build "Blueprints" for each entity type
        # These are lists of extractors for a SINGLE instance of that entity
        self._global_blueprint = self._build_global_blueprint()
        self._site_blueprint = self._build_site_blueprint()
        self._vessel_blueprint = self._build_vessel_blueprint()
        self._goal_blueprint = self._build_goal_blueprint()

        # 4. Calculate Dimensions
        self.global_dim = len(self._global_blueprint)
        self.site_dim = len(self._site_blueprint)
        self.vessel_dim = len(self._vessel_blueprint)
        self.goal_dim = len(self._goal_blueprint)

        # The width of the matrix is the max features required by any entity
        self.max_features = max(
            self.global_dim, self.site_dim, self.vessel_dim, self.goal_dim
        )

        # The height is Total Entities
        self.n_global = 1
        self.total_rows = (
            self.n_global + self._num_sites + self._num_vessels + self._num_goals
        )

        # 5. Define Space
        low = np.full(
            (self.total_rows, self.max_features), -np.inf, dtype=self._config.dtype
        )
        high = np.full(
            (self.total_rows, self.max_features), np.inf, dtype=self._config.dtype
        )
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=self._config.dtype
        )

    @property
    def schema(self) -> Dict[str, Any]:
        """
        Returns the slicing information needed by the Neural Network.
        Pass this dictionary to your Agent's constructor!
        """
        # Calculate start indices
        global_start = 0
        site_start = global_start + self.n_global
        vessel_start = site_start + self._num_sites
        goal_start = vessel_start + self._num_vessels

        return {
            "global": {
                "slice": slice(global_start, site_start),
                "feats": self.global_dim,
                "names": [f.name for f in self._global_blueprint],
                "categoricals": {},
                "count": 1,
            },
            "sites": {
                "slice": slice(site_start, vessel_start),
                "feats": self.site_dim,
                "names": [f.name for f in self._site_blueprint],
                "categoricals": self._get_categorical_map(
                    self._site_blueprint,
                    self._config.site_categorical_fields,
                    "sites",
                    self._num_sites,
                ),
                "count": self._num_sites,
            },
            "vessels": {
                "slice": slice(vessel_start, goal_start),
                "feats": self.vessel_dim,
                "names": [f.name for f in self._vessel_blueprint],
                "categoricals": self._get_categorical_map(
                    self._vessel_blueprint,
                    self._config.vessel_categorical_fields,
                    "vessels",
                    self._num_vessels,
                ),
                "count": self._num_vessels,
            },
            "goals": {
                "slice": slice(goal_start, self.total_rows),
                "feats": self.goal_dim,
                "names": [f.name for f in self._goal_blueprint],
                "categoricals": self._get_categorical_map(
                    self._goal_blueprint,
                    self._config.goal_categorical_fields,
                    "goals",
                    self._num_goals,
                ),
                "count": self._num_goals,
            },
        }

    def observation(self, observation: Mapping[str, object]) -> np.ndarray:
        # Initialize with zeros (padding)
        matrix = np.zeros(
            (self.total_rows, self.max_features), dtype=self._config.dtype
        )

        # --- 1. Fill Global Row (Row 0) ---
        # Global is usually stored at root or under "global_obs"
        global_data = observation.get("global_obs", observation)
        for i, field_def in enumerate(self._global_blueprint):
            matrix[0, i] = field_def.extractor(global_data)

        # --- 2. Fill Site Rows ---
        # We fetch the list of sites once
        sites_data = observation["sites"]
        row_offset = self.n_global
        for i in range(self._num_sites):
            site_data = sites_data[i]
            for col, field_def in enumerate(self._site_blueprint):
                matrix[row_offset + i, col] = field_def.extractor(site_data)

        # --- 3. Fill Vessel Rows ---
        vessels_data = observation["vessels"]
        row_offset = self.n_global + self._num_sites
        for i in range(self._num_vessels):
            vessel_data = vessels_data[i]
            for col, field_def in enumerate(self._vessel_blueprint):
                matrix[row_offset + i, col] = field_def.extractor(vessel_data)

        # --- 4. Fill Goal Rows ---
        goals_data = observation.get("goals", ())
        row_offset = self.n_global + self._num_sites + self._num_vessels
        for i in range(self._num_goals):
            if i < len(goals_data):
                goal_data = goals_data[i]
                for col, field_def in enumerate(self._goal_blueprint):
                    matrix[row_offset + i, col] = field_def.extractor(goal_data)

        return matrix

    # --- Blueprint Builders (Define columns for ONE entity) ---

    def _build_global_blueprint(self) -> List[EntityField]:
        fields = []
        for key in self._config.global_fields:
            name = f"global.{key}"
            # Capture loop variable 'key' and 'name'
            fields.append(
                EntityField(
                    name=name,
                    extractor=lambda d, k=key, n=name: self._apply_preprocess(
                        n, self._to_scalar(d.get(k, 0))
                    ),
                )
            )
        return fields

    def _build_site_blueprint(self) -> List[EntityField]:
        fields = []
        if self._config.include_site_id:
            fields.append(
                EntityField(
                    name="id", extractor=lambda d: self._to_scalar(d.get("id", 0))
                )
            )

        # Site prefix fields (e.g. role)
        for key in self._config.site_prefix_fields:
            fields.append(
                EntityField(
                    name=key,
                    extractor=lambda d, k=key: self._to_scalar(d.get(k, 0)),
                )
            )

        for resource in self._site_resource_order:
            for feat in self._config.site_resource_fields:
                name = f"stock.{resource}.{feat}"
                fields.append(
                    EntityField(
                        name=name,
                        extractor=lambda d, r=resource, f=feat, n=name: (
                            self._apply_preprocess(n, self._to_scalar(d["stock"][r][f]))
                        ),
                    )
                )
        return fields

    def _build_vessel_blueprint(self) -> List[EntityField]:
        fields = []
        if self._config.include_vessel_id:
            fields.append(
                EntityField(
                    name="id", extractor=lambda d: self._to_scalar(d.get("id", 0))
                )
            )

        for key in self._config.vessel_prefix_fields:
            # Helper to expand vector fields
            size = self._get_vessel_field_size(key)
            if size <= 1:
                fields.append(
                    EntityField(
                        name=key,
                        extractor=lambda d, k=key, n=key: self._apply_preprocess(
                            n, self._to_scalar(d.get(k, 0))
                        ),
                    )
                )
            else:
                for j in range(size):
                    name = f"{key}[{j}]"
                    fields.append(
                        EntityField(
                            name=name,
                            extractor=lambda d, k=key, idx=j, n=name: (
                                self._apply_preprocess(
                                    n, self._vector_entry(d.get(k, 0), idx)
                                )
                            ),
                        )
                    )

        for resource in self._vessel_resource_order:
            for feat in self._config.vessel_resource_fields:
                name = f"inventory.{resource}.{feat}"
                fields.append(
                    EntityField(
                        name=name,
                        extractor=lambda d, r=resource, f=feat, n=name: (
                            self._apply_preprocess(
                                n, self._to_scalar(d["inventory"][r][f])
                            )
                        ),
                    )
                )

            if self._config.include_visit_modes:
                name = f"visit_modes.{resource}"
                fields.append(
                    EntityField(
                        name=name,
                        extractor=lambda d, r=resource, n=name: self._apply_preprocess(
                            n, self._to_scalar(d.get("visit_modes", {}).get(r, 0))
                        ),
                    )
                )

        # --- Intent fields ---
        for intent_key in self._config.vessel_intent_fields:
            name = f"intent.{intent_key}"
            # Shift by +1 so that the sentinel value -1 (no intent)
            # becomes embedding index 0, and valid IDs 0..N-1 become
            # indices 1..N.  This avoids a collision between NONE and
            # the first valid category (e.g. ActionType.IDLE == 0).
            fields.append(
                EntityField(
                    name=name,
                    extractor=lambda d, k=intent_key, n=name: self._apply_preprocess(
                        n,
                        self._to_scalar(d.get("intent", {}).get(k, -1)) + 1,
                    ),
                )
            )

        return fields

    def _build_goal_blueprint(self) -> List[EntityField]:
        """Build column extractors for a single goal entity row."""
        fields: List[EntityField] = []

        if self._config.include_goal_id:
            fields.append(
                EntityField(
                    name="id", extractor=lambda d: self._to_scalar(d.get("id", 0))
                )
            )

        # Categorical prefix fields: target_site, resource_type
        fields.append(
            EntityField(
                name="target_site",
                extractor=lambda d: self._to_scalar(d.get("target_site", 0)),
            )
        )
        fields.append(
            EntityField(
                name="resource_type",
                extractor=lambda d: self._to_scalar(d.get("resource_type", 0)),
            )
        )

        # Continuous goal fields
        for key in self._config.goal_fields:
            name = key
            fields.append(
                EntityField(
                    name=name,
                    extractor=lambda d, k=key, n=name: self._apply_preprocess(
                        n, self._to_scalar(d.get(k, 0))
                    ),
                )
            )

        # blocked_by multi-hot vector (num_goals columns)
        for j in range(max(self._num_goals, 1)):
            name = f"blocked_by.{j}"
            fields.append(
                EntityField(
                    name=name,
                    extractor=lambda d, idx=j: self._vector_entry(
                        d.get("blocked_by", np.zeros(0)), idx
                    ),
                )
            )

        return fields

    # --- Categorical Cardinality Logic ---

    def _get_categorical_map(
        self,
        blueprint: List[EntityField],
        categorical_fields: Sequence[str],
        entity_key: str,
        entity_count: int,
    ) -> Dict[str, int]:
        """
        Builds the map: Field Name -> Vocab Size
        """
        cat_map = {}
        for f in blueprint:
            if (
                f.name == "id"
                or f.name in categorical_fields
                or (
                    self._config.include_visit_modes
                    and f.name.startswith("visit_modes.")
                )
            ):
                vocab_size = self._resolve_cardinality(f.name, entity_key, entity_count)
                cat_map[f.name] = vocab_size
        return cat_map

    def _resolve_cardinality(
        self, field_name: str, entity_key: str, entity_count: int
    ) -> int:
        """
        Central logic to determine embedding size for a field.
        """
        # 1. IDs: Always size of the entity list
        if field_name == "id":
            return entity_count

        # 2. Position: Always correlates to number of Sites
        if field_name == "position" and entity_key == "vessels":
            return self._num_sites

        # 3. Intent categoricals — sentinel -1 is shifted to 0 by the
        #    encoder, so vocab = valid_range + 1 (for the NONE sentinel).
        if field_name == "intent.action_type":
            # IDLE=0, MOVE=1, LOAD=2, UNLOAD=3  →  4 types + 1 NONE
            return 5
        if field_name == "intent.destination":
            # site IDs 0..n_sites-1  +  1 NONE
            return self._num_sites + 1
        if field_name == "intent.resource_id":
            # resource IDs 0..n_resources-1  +  1 NONE
            return len(self._resource_names) + 1

        if field_name.startswith("visit_modes."):
            return 3  # NEUTRAL, LOADING, UNLOADING

        # Goal categoricals
        if field_name == "target_site" and entity_key == "goals":
            return self._num_sites
        if field_name == "resource_type" and entity_key == "goals":
            return max(len(self._resource_names), 1)
        if field_name == "role" and entity_key == "sites":
            return 3  # source, staging, installation

        # 4. Status/Others: Try to infer from Gym Space (e.g. Discrete(3))
        # If inferrence fails, default to a safe number or 2 if binary.
        inferred = self._infer_discrete_size(entity_key, field_name)
        if inferred > 0:
            return inferred

        # 5. Fallback defaults
        if field_name == "status":
            return 3  # (Idle, Busy, Error) - Safe default

        return 10  # Unknown field default

    def _infer_discrete_size(self, entity_key: str, field_name: str) -> int:
        """Peeks into the Gym observation space to find Discrete(n)."""
        try:
            space = self.env.observation_space
            if isinstance(space, spaces.Dict) and entity_key in space.spaces:
                # Unwrap Tuple -> Dict -> Field
                sub = space.spaces[entity_key].spaces[0]
                if field_name in sub.spaces:
                    target_space = sub.spaces[field_name]
                    if isinstance(target_space, spaces.Discrete):
                        return int(target_space.n)
        except Exception:
            pass
        return 0

    # --- Helpers ---

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
        return float(value)  # type: ignore

    @staticmethod
    def _vector_entry(value: object, index: int) -> float:
        if isinstance(value, np.ndarray):
            flat = value.reshape(-1)
            return float(flat[index]) if 0 <= index < flat.size else 0.0
        if isinstance(value, (list, tuple)):
            return float(value[index]) if 0 <= index < len(value) else 0.0
        return 0.0
