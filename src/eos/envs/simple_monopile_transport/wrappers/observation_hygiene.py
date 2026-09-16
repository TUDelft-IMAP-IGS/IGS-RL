from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from eos.envs.simple_monopile_transport.gym_env import SimpleMonopileTransportEnv


class ObservationHygieneWrapper(ABC, gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        time_budget_hours_attr: str = "_time_budget_hours",
        episode_start_time_attr: str = "_episode_start_time_unix",
    ) -> None:
        super().__init__(env)
        self._env_unwrapped: SimpleMonopileTransportEnv = env.unwrapped
        self._time_budget_hours_attr = time_budget_hours_attr
        self._episode_start_time_attr = episode_start_time_attr

        self._num_sites = self._infer_num_items("sites")
        self._num_vessels = self._infer_num_items("vessels")
        self._num_status = self._infer_num_status()

    def _get_time_budget_hours(self) -> float:
        if hasattr(self._env_unwrapped, self._time_budget_hours_attr):
            value = getattr(self._env_unwrapped, self._time_budget_hours_attr)
            return float(value)
        raise AttributeError(
            f"Env missing '{self._time_budget_hours_attr}' for time budget hours"
        )

    def _get_episode_start_time(self) -> float:
        if hasattr(self._env_unwrapped, self._episode_start_time_attr):
            value = getattr(self._env_unwrapped, self._episode_start_time_attr)
            return float(value)
        raise AttributeError(
            f"Env missing '{self._episode_start_time_attr}' for episode start time"
        )

    def _get_travel_cost_budget(self) -> float:
        """Return the worst-case travel cost budget for normalization."""
        return getattr(self._env_unwrapped, "_travel_cost_budget", 0.0)

    def _get_storage_cost_budget(self) -> float:
        """Return the worst-case storage cost budget for normalization."""
        return getattr(self._env_unwrapped, "_storage_cost_budget", 0.0)

    def _cost_state_augmented(self) -> bool:
        """Whether accumulated cost stats are exposed in the global token.

        When the env config sets ``augment_cost_state=False`` the accumulated
        travel/storage cost fields are zeroed, allowing ablation of the
        path-dependent state augmentation while keeping the observation
        shape unchanged.
        """
        cfg = getattr(self._env_unwrapped, "cfg", None)
        return bool(getattr(cfg, "augment_cost_state", True))

    def _maybe_zero_cost_state(self, global_obs: Dict[str, Any]) -> None:
        """Zero the accumulated cost fields when augmentation is disabled."""
        if self._cost_state_augmented():
            return
        global_obs["accumulated_travel_cost"] = np.array([0.0], dtype=np.float64)
        global_obs["accumulated_storage_cost"] = np.array([0.0], dtype=np.float64)

    def _infer_num_items(self, key: str) -> int:
        space = self.env.observation_space
        if isinstance(space, spaces.Dict) and key in space.spaces:
            sub = space.spaces[key]
            if isinstance(sub, spaces.Tuple):
                return len(sub.spaces)
        return 0

    def _infer_num_status(self) -> int:
        space = self.env.observation_space
        if isinstance(space, spaces.Dict) and "vessels" in space.spaces:
            vessels_space = space.spaces["vessels"]
            if isinstance(vessels_space, spaces.Tuple) and vessels_space.spaces:
                vessel_space = vessels_space.spaces[0]
                if (
                    isinstance(vessel_space, spaces.Dict)
                    and "status" in vessel_space.spaces
                ):
                    status_space = vessel_space.spaces["status"]
                    if isinstance(status_space, spaces.Discrete):
                        return int(status_space.n)
        return 0

    @staticmethod
    def _sorted_by_id(items: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        return sorted(
            items,
            key=lambda item: int(MLPObservationHygieneWrapper._to_int(item["id"])),
        )

    @staticmethod
    def _one_hot(index: int, size: int) -> np.ndarray:
        vec = np.zeros((size,), dtype=np.int64)
        if 0 <= index < size:
            vec[index] = 1
        return vec

    @staticmethod
    def _to_scalar(value: Any) -> float:
        if isinstance(value, np.ndarray):
            return float(value.reshape(-1)[0]) if value.size else 0.0
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        return float(value)

    @staticmethod
    def _to_int(value: Any) -> int:
        if isinstance(value, np.ndarray):
            return int(value.reshape(-1)[0]) if value.size else 0
        if isinstance(value, (np.floating, np.integer)):
            return int(value)
        return int(value)


class MLPObservationHygieneWrapper(ObservationHygieneWrapper):
    """Sanitizes raw simulator state for the RL agent.

    Responsibilities
    ----------------
    1. Relativity: Converts absolute Unix timestamps to normalized elapsed time.
    2. Encoding: One-hot encodes categorical enums (position, status).
    3. Pruning: Strips `id` fields after sorting by id (stable ordering).
    """

    def __init__(
        self,
        env: gym.Env,
        time_budget_hours_attr: str = "_time_budget_hours",
        episode_start_time_attr: str = "_episode_start_time_unix",
    ) -> None:
        super().__init__(env, time_budget_hours_attr, episode_start_time_attr)

        self.observation_space = self._build_observation_space()

    def observation(self, observation: Mapping[str, Any]) -> Dict[str, Any]:
        global_obs = dict(observation["global_obs"])
        current_time = self._to_scalar(global_obs["current_time"])
        start_time = self._get_episode_start_time()
        time_budget = self._get_time_budget_hours()

        if time_budget <= 0:
            normalized_time = 0.0
        else:
            elapsed_hours = (current_time - start_time) / 3600.0
            normalized_time = elapsed_hours / time_budget

        # TODO: Should probably be elapsed time already in obs
        global_obs["current_time"] = np.array([normalized_time], dtype=np.float64)

        # Normalize steps
        max_steps = self._env_unwrapped.cfg.max_episode_steps
        current_step = self._to_scalar(global_obs["step"])
        normalized_step = current_step / max_steps

        global_obs["step"] = np.array([normalized_step], dtype=np.float64)

        # Normalize time_to_install_window_flip the same way as current_time
        if "time_to_install_window_flip" in global_obs:
            raw_flip = self._to_scalar(global_obs["time_to_install_window_flip"])
            normalized_flip = raw_flip / time_budget if time_budget > 0 else 0.0
            global_obs["time_to_install_window_flip"] = np.array(
                [normalized_flip], dtype=np.float64
            )

        # Normalize accumulated costs by worst-case budgets
        travel_budget = self._get_travel_cost_budget()
        if travel_budget > 0:
            raw_travel = self._to_scalar(global_obs.get("accumulated_travel_cost", 0.0))
            global_obs["accumulated_travel_cost"] = np.array(
                [raw_travel / travel_budget], dtype=np.float64
            )

        storage_budget = self._get_storage_cost_budget()
        if storage_budget > 0:
            raw_storage = self._to_scalar(
                global_obs.get("accumulated_storage_cost", 0.0)
            )
            global_obs["accumulated_storage_cost"] = np.array(
                [raw_storage / storage_budget], dtype=np.float64
            )

        self._maybe_zero_cost_state(global_obs)

        sites_sorted = self._sorted_by_id(observation["sites"])
        sites = tuple({"stock": site["stock"]} for site in sites_sorted)

        vessels_sorted = self._sorted_by_id(observation["vessels"])
        vessels: List[Dict[str, Any]] = []
        for vessel in vessels_sorted:
            position = self._one_hot(self._to_int(vessel["position"]), self._num_sites)
            status = self._one_hot(self._to_int(vessel["status"]), self._num_status)

            v_dict: Dict[str, Any] = {
                "position": position,
                "status": status,
                "inventory": vessel["inventory"],
            }

            # Pass intent through unchanged so downstream consumers
            # (e.g. FlattenObservationWrapper) can still see it.
            if "intent" in vessel:
                v_dict["intent"] = vessel["intent"]

            vessels.append(v_dict)

        # Goals passthrough with deadline normalization
        goals_raw = observation.get("goals", ())
        goals_out = []
        for goal in goals_raw:
            g = dict(goal)
            if "deadline" in g:
                raw_deadline = self._to_scalar(g["deadline"])
                normalized_deadline = (
                    raw_deadline * 3600.0 / time_budget if time_budget > 0 else 0.0
                )
                g["deadline"] = np.array([normalized_deadline], dtype=np.float64)
            goals_out.append(g)

        return {
            "global_obs": global_obs,
            "sites": tuple(sites),
            "vessels": tuple(vessels),
            "goals": tuple(goals_out),
        }

    def _build_observation_space(self) -> spaces.Dict:
        base = self.env.observation_space
        if not isinstance(base, spaces.Dict):
            raise TypeError("ObservationHygieneWrapper expects Dict observation space")

        global_space = base.spaces["global_obs"]
        if not isinstance(global_space, spaces.Dict):
            raise TypeError("global_obs must be a Dict space")

        normalized_time_space = spaces.Box(
            low=0.0, high=np.inf, shape=(1,), dtype=np.float64
        )
        normalized_flip_space = spaces.Box(
            low=0.0, high=np.inf, shape=(1,), dtype=np.float64
        )
        global_obs_space = spaces.Dict(
            {
                **global_space.spaces,
                "current_time": normalized_time_space,
                "time_to_install_window_flip": normalized_flip_space,
                "accumulated_travel_cost": spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float64
                ),
                "accumulated_storage_cost": spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float64
                ),
            }
        )

        sites_space = base.spaces["sites"]
        if not isinstance(sites_space, spaces.Tuple):
            raise TypeError("sites must be a Tuple space")
        if not sites_space.spaces:
            raise ValueError("sites space is empty")

        site_space = sites_space.spaces[0]
        if not isinstance(site_space, spaces.Dict):
            raise TypeError("site space must be Dict")
        stock_space = site_space.spaces["stock"]
        sites_out = spaces.Tuple(
            [spaces.Dict({"stock": stock_space}) for _ in range(self._num_sites)]
        )

        vessels_space = base.spaces["vessels"]
        if not isinstance(vessels_space, spaces.Tuple):
            raise TypeError("vessels must be a Tuple space")
        if not vessels_space.spaces:
            raise ValueError("vessels space is empty")
        vessel_space = vessels_space.spaces[0]
        if not isinstance(vessel_space, spaces.Dict):
            raise TypeError("vessel space must be Dict")
        inventory_space = vessel_space.spaces["inventory"]

        # Carry intent space through if the upstream env provides it.
        intent_space = None
        if isinstance(vessel_space, spaces.Dict) and "intent" in vessel_space.spaces:
            intent_space = vessel_space.spaces["intent"]

        def _vessel_dict() -> Dict[str, Any]:
            d: Dict[str, Any] = {
                "position": spaces.MultiBinary(self._num_sites),
                "status": spaces.MultiBinary(self._num_status),
                "inventory": inventory_space,
            }
            if intent_space is not None:
                d["intent"] = intent_space
            return d

        vessels_out = spaces.Tuple(
            [spaces.Dict(_vessel_dict()) for _ in range(self._num_vessels)]
        )

        # Goals space passthrough
        result_spaces = {
            "global_obs": global_obs_space,
            "sites": sites_out,
            "vessels": vessels_out,
        }
        if "goals" in base.spaces:
            result_spaces["goals"] = base.spaces["goals"]
        return spaces.Dict(result_spaces)


class TransformerObservationHygieneWrapper(ObservationHygieneWrapper):
    """
    Prepares raw simulator state for a Transformer Agent.

    Improvements over original:
    1. Preserves IDs (for nn.Embedding).
    2. Preserves Integers for Enums (for nn.Embedding).
    3. Maintains Stable Sorting (Critical for permutation invariance).
    4. Normalizes Time (Critical for neural network stability).
    """

    def __init__(
        self,
        env: gym.Env,
        time_budget_hours_attr: str = "_time_budget_hours",
        episode_start_time_attr: str = "_episode_start_time_unix",
    ) -> None:
        super().__init__(env, time_budget_hours_attr, episode_start_time_attr)

        self.observation_space = self._build_observation_space()

    def observation(self, observation: Mapping[str, Any]) -> Dict[str, Any]:
        global_obs = dict(observation["global_obs"])

        # 1. Time Normalization
        current_time = self._to_scalar(global_obs["current_time"])
        start_time = self._get_episode_start_time()
        time_budget = self._get_time_budget_hours()

        if time_budget <= 0:
            normalized_time = 0.0
        else:
            normalized_time = (current_time - start_time) / 3600.0 / time_budget

        global_obs["current_time"] = np.array([normalized_time], dtype=np.float64)

        # Normalize steps
        max_steps = self._env_unwrapped.cfg.max_episode_steps
        current_step = self._to_scalar(global_obs["step"])
        normalized_step = current_step / max_steps

        global_obs["step"] = np.array([normalized_step], dtype=np.float64)

        # Normalize time_to_install_window_flip the same way as current_time
        if "time_to_install_window_flip" in global_obs:
            raw_flip = self._to_scalar(global_obs["time_to_install_window_flip"])
            normalized_flip = raw_flip / time_budget if time_budget > 0 else 0.0
            global_obs["time_to_install_window_flip"] = np.array(
                [normalized_flip], dtype=np.float64
            )

        # Normalize accumulated costs by worst-case budgets
        travel_budget = self._get_travel_cost_budget()
        if travel_budget > 0:
            raw_travel = self._to_scalar(global_obs.get("accumulated_travel_cost", 0.0))
            global_obs["accumulated_travel_cost"] = np.array(
                [raw_travel / travel_budget], dtype=np.float64
            )

        storage_budget = self._get_storage_cost_budget()
        if storage_budget > 0:
            raw_storage = self._to_scalar(
                global_obs.get("accumulated_storage_cost", 0.0)
            )
            global_obs["accumulated_storage_cost"] = np.array(
                [raw_storage / storage_budget], dtype=np.float64
            )

        self._maybe_zero_cost_state(global_obs)

        # 2. Stable Sorting + normalize site fabrication times
        sites_sorted = self._sorted_by_id(observation["sites"])
        sites_out = []
        for site in sites_sorted:
            s = dict(site)
            if "stock" in s:
                stock_out = {}
                for rtype, rdata in s["stock"].items():
                    rd = dict(rdata)
                    if "next_available_in" in rd:
                        raw_nav = self._to_scalar(rd["next_available_in"])
                        rd["next_available_in"] = np.array(
                            [
                                raw_nav * 3600.0 / time_budget
                                if time_budget > 0
                                else 0.0
                            ],
                            dtype=np.float64,
                        )
                    stock_out[rtype] = rd
                s["stock"] = stock_out
            sites_out.append(s)

        vessels_sorted = self._sorted_by_id(observation["vessels"])

        # 3. Normalize goal deadlines by time budget
        goals_raw = observation.get("goals", ())
        goals_out = []
        for goal in goals_raw:
            g = dict(goal)
            if "deadline" in g:
                raw_deadline = self._to_scalar(g["deadline"])
                normalized_deadline = (
                    raw_deadline * 3600.0 / time_budget if time_budget > 0 else 0.0
                )
                g["deadline"] = np.array([normalized_deadline], dtype=np.float64)
            goals_out.append(g)

        return {
            "global_obs": global_obs,
            "sites": tuple(sites_out),
            "vessels": tuple(vessels_sorted),
            "goals": tuple(goals_out),
        }

    def _build_observation_space(self) -> spaces.Dict:
        # Since we are mostly passing data through, we can largely
        # reuse the original space, just updating the Global Time.
        base = self.env.observation_space
        global_space = base.spaces["global_obs"]

        # Update Time to be a normalized scalar [0, 1]
        normalized_time_space = spaces.Box(
            low=0.0, high=np.inf, shape=(1,), dtype=np.float64
        )
        normalized_flip_space = spaces.Box(
            low=0.0, high=np.inf, shape=(1,), dtype=np.float64
        )

        new_global = spaces.Dict(
            {
                **global_space.spaces,
                "current_time": normalized_time_space,
                "time_to_install_window_flip": normalized_flip_space,
                "accumulated_travel_cost": spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float64
                ),
                "accumulated_storage_cost": spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float64
                ),
            }
        )

        return spaces.Dict({**base.spaces, "global_obs": new_global})
