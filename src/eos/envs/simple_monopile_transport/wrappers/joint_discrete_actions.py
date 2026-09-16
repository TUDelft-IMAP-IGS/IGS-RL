from __future__ import annotations

from typing import Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger

from eos.envs.simple_monopile_transport.gym_env import (
    Action,
    ActionParams,
    IdleParams,
    MoveParams,
    ResourceTransferParams,
    SimpleMonopileTransportEnv,
)
from eos.envs.simple_monopile_transport.types import ActionType, PartnerType


class JointDiscreteActionWrapper(gym.Wrapper):
    """
    Joint, per-vessel discrete actions with a NOOP option.

    Action space: MultiDiscrete([opts_per_vessel] * n_vessels)
    Mask shape: (n_vessels, opts_per_vessel)

    The NOOP action (index 0) results in no action registration for that vessel.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._env_unwrapped: SimpleMonopileTransportEnv = env.unwrapped

        # Collect the observation-wrapper chain (inner → outer) so that
        # micro_step can rebuild a fully-transformed observation after
        # registering an intent without going through step()/reset().
        self._obs_wrappers: List[gym.ObservationWrapper] = []
        cur = env
        while cur is not None and not isinstance(cur, SimpleMonopileTransportEnv):
            if isinstance(cur, gym.ObservationWrapper):
                self._obs_wrappers.insert(0, cur)
            cur = getattr(cur, "env", None)

        base_env = self._env_unwrapped
        self.n_vessels = len(base_env._vessel_names)
        self.n_sites = len(base_env._site_names)
        self.n_resources = len(base_env._resource_names)

        # Event-driven IDLE: a single action meaning "wait until the next
        # DES activity completes."  Controlled by allow_idle_actions config.
        self.allow_idle_actions = bool(
            getattr(base_env.cfg, "allow_idle_actions", True)
        )

        # 1. MOVE: Destination (Site ID)
        self.move_options = self.n_sites

        # 2. LOAD/UNLOAD: Partner (Site or Vessel) + Resource ID
        self._num_transfer_partners = self.n_sites + self.n_vessels
        self.transfer_options = self._num_transfer_partners * self.n_resources

        # 3. IDLE: Single await-event action (no duration parameterisation)
        self.idle_options = 1 if self.allow_idle_actions else 0

        # 4. NOOP: Do nothing for this vessel at this decision epoch
        self.noop_index = 0

        # Total options per vessel (NOOP + Move + Load + Unload + Idle)
        self.opts_per_vessel = (
            1
            + self.move_options
            + self.transfer_options
            + self.transfer_options
            + self.idle_options
        )

        self.action_space = spaces.MultiDiscrete(
            np.array([self.opts_per_vessel] * self.n_vessels, dtype=np.int64)
        )

        # Create Name-to-Index Maps for Masking
        self._site_map = {name: i for i, name in enumerate(base_env._site_names)}
        self._res_map = {name: i for i, name in enumerate(base_env._resource_names)}
        self._vessel_map = {name: i for i, name in enumerate(base_env._vessel_names)}

        # Expose ordered names for external micro-stepping sampling
        self.vessel_names = list(base_env._vessel_names)
        self.site_names = list(base_env._site_names)
        self.resource_names = list(base_env._resource_names)
        self.action_index_metadata = self.get_action_index_metadata()

    def get_action_index_ranges(self) -> dict[str, tuple[int, int]]:
        """
        Return per-vessel action index ranges as half-open intervals [start, end).
        """
        move_offset = 1
        load_offset = move_offset + self.move_options
        unload_offset = load_offset + self.transfer_options
        idle_offset = unload_offset + self.transfer_options
        return {
            "noop": (0, 1),
            "move": (move_offset, load_offset),
            "load": (load_offset, unload_offset),
            "unload": (unload_offset, idle_offset),
            "idle": (idle_offset, idle_offset + self.idle_options),
        }

    def get_action_index_metadata(self) -> dict[str, int | dict[str, int]]:
        """
        Return offsets and sizes used to decode per-vessel action indices.
        """
        move_offset = 1
        load_offset = move_offset + self.move_options
        unload_offset = load_offset + self.transfer_options
        idle_offset = unload_offset + self.transfer_options
        return {
            "noop_index": self.noop_index,
            "move_offset": move_offset,
            "load_offset": load_offset,
            "unload_offset": unload_offset,
            "idle_offset": idle_offset,
            "sizes": {
                "move_options": self.move_options,
                "transfer_options": self.transfer_options,
                "idle_options": self.idle_options,
                "opts_per_vessel": self.opts_per_vessel,
                "n_vessels": self.n_vessels,
                "n_sites": self.n_sites,
                "n_resources": self.n_resources,
            },
        }

    def _build_mask_for_vessel(
        self, v_name: str, valid_specs, allow_noop: bool = False
    ) -> np.ndarray:
        """Build a per-vessel boolean mask from valid ActionSpecs."""
        mask = np.zeros(self.opts_per_vessel, dtype=bool)

        # Base index for the action sections (offset by NOOP)
        move_offset = 1
        load_offset = move_offset + self.move_options
        unload_offset = load_offset + self.transfer_options
        idle_offset = unload_offset + self.transfer_options

        for spec in valid_specs:
            if spec.action_type == ActionType.MOVE:
                if spec.destination_name in self._site_map:
                    dest_idx = self._site_map[spec.destination_name]
                    mask[move_offset + dest_idx] = True

            elif spec.action_type == ActionType.LOAD:
                if (
                    spec.partner_name in self._site_map
                    and spec.resource_name in self._res_map
                ):
                    p_idx = self._site_map[spec.partner_name]
                    r_idx = self._res_map[spec.resource_name]
                    local_idx = p_idx * self.n_resources + r_idx
                    mask[load_offset + local_idx] = True
                elif (
                    spec.partner_name in self._vessel_map
                    and spec.resource_name in self._res_map
                ):
                    p_idx = self.n_sites + self._vessel_map[spec.partner_name]
                    r_idx = self._res_map[spec.resource_name]
                    local_idx = p_idx * self.n_resources + r_idx
                    mask[load_offset + local_idx] = True

            elif spec.action_type == ActionType.UNLOAD:
                if (
                    spec.partner_name in self._site_map
                    and spec.resource_name in self._res_map
                ):
                    p_idx = self._site_map[spec.partner_name]
                    r_idx = self._res_map[spec.resource_name]
                    local_idx = p_idx * self.n_resources + r_idx
                    mask[unload_offset + local_idx] = True
                elif (
                    spec.partner_name in self._vessel_map
                    and spec.resource_name in self._res_map
                ):
                    p_idx = self.n_sites + self._vessel_map[spec.partner_name]
                    r_idx = self._res_map[spec.resource_name]
                    local_idx = p_idx * self.n_resources + r_idx
                    mask[unload_offset + local_idx] = True

        if self.idle_options > 0:
            # IDLE is a *strategic* choice: "I could act, but I choose
            # to wait."  It is only offered when the vessel has at least
            # one substantive action (move/load/unload) available.
            # When no real action exists, the vessel NOOPs instead —
            # this avoids registering idle activities for vessels with
            # nothing to do, which would generate spurious completion
            # events and cascade zero-duration DES steps.
            sim = self._env_unwrapped._sim
            has_substantive_action = mask[:idle_offset].any()
            if has_substantive_action and sim.can_idle(v_name):
                mask[idle_offset] = True

        # ── NOOP logic ──
        # NOOP is available in two cases:
        # 1. No substantive action exists (fallback — vessel can't do
        #    anything, so it must NOOP).
        # 2. Any OTHER vessel is currently busy (voluntary pass — the
        #    DES will advance via that vessel's activity completion,
        #    so this vessel can safely defer without deadlock).
        #
        # NOOP is NOT available when the vessel has actions but no other
        # vessel is busy — forcing at least one vessel to act each
        # macro-step so the DES always has an event to process.
        if not mask.any():
            mask[self.noop_index] = True
        else:
            sim = self._env_unwrapped._sim
            any_other_busy = any(v != v_name for v in sim.busy_vessels)
            if any_other_busy:
                mask[self.noop_index] = True

        return mask

    def mask_for_vessel(self, v_name: str, valid_specs) -> np.ndarray:
        """
        Build a per-vessel mask from externally provided ActionSpecs.
        """
        return self._build_mask_for_vessel(v_name, valid_specs)

    def action_mask_for_vessel(
        self,
        vessel_indices_batch: np.ndarray | int,
        active_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the action mask for a single vessel in this sub-env.

        Designed to be called via ``envs.call("action_mask_for_vessel",
        vessel_indices_batch, active_mask)`` on a vectorised env.  Each
        sub-env extracts its own vessel index from the batch array using
        ``_vector_env_index``.

        Parameters
        ----------
        vessel_indices_batch : np.ndarray or int
            Either a scalar vessel index, or a batch array of shape
            ``(num_envs,)`` from which this sub-env extracts its own
            entry via ``_vector_env_index``.
        active_mask : np.ndarray or None
            Optional boolean batch array of shape ``(num_envs,)``.  When
            provided and this env's entry is ``False``, an all-True mask
            is returned (the caller will skip this env anyway).

        Returns
        -------
        np.ndarray
            Boolean mask of shape ``(opts_per_vessel,)``.
        """
        vessel_indices_batch = np.asarray(vessel_indices_batch)

        if vessel_indices_batch.ndim == 0:
            # Scalar call — use directly
            vessel_index = int(vessel_indices_batch)
        else:
            env_idx = self._env_unwrapped._vector_env_index
            vessel_index = int(vessel_indices_batch[env_idx])

            # If an active mask is provided and this env is inactive,
            # return an all-True mask (caller will ignore it).
            if active_mask is not None:
                active_mask = np.asarray(active_mask)
                if not active_mask[env_idx]:
                    return np.ones(self.opts_per_vessel, dtype=bool)

        v_name = self._env_unwrapped._vessel_names[vessel_index]
        sim = self._env_unwrapped._sim

        if sim.is_vessel_busy(v_name):
            mask = np.zeros(self.opts_per_vessel, dtype=bool)
            mask[self.noop_index] = True
            return mask

        valid_specs = sim.get_possible_actions(v_name, reservations=sim.reservations)
        return self._build_mask_for_vessel(v_name, valid_specs, allow_noop=False)

    def vessel_availability(self) -> np.ndarray:
        """Return a boolean array indicating which vessels are idle (not busy).

        Shape: ``(n_vessels,)``  —  ``True`` means the vessel is idle and
        available for ordering / action assignment.
        """
        sim = self._env_unwrapped._sim
        avail = np.ones(self.n_vessels, dtype=bool)
        for v_name in self._env_unwrapped._vessel_names:
            v_idx = self._vessel_map[v_name]
            if sim.is_vessel_busy(v_name):
                avail[v_idx] = False
        return avail

    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean mask of valid actions for the current state.
        Shape: (n_vessels, opts_per_vessel)
        dtype: bool
        """
        mask = np.zeros((self.n_vessels, self.opts_per_vessel), dtype=bool)
        sim = self._env_unwrapped._sim

        # Build the reservation system ONCE and share across all vessels.
        # Previously each get_possible_actions() call rebuilt it from scratch.
        reservations = sim.reservations

        for v_name in self._env_unwrapped._vessel_names:
            v_idx = self._vessel_map[v_name]

            if sim.is_vessel_busy(v_name):
                # Busy vessel: Force NOOP only
                mask[v_idx, self.noop_index] = True
                continue

            valid_specs = sim.get_possible_actions(v_name, reservations=reservations)

            # Idle vessel: Calculate valid real actions.
            # We explicitly pass allow_noop=False because an IDLE agent
            # must log a logistical intent to advance the DES.
            mask[v_idx, :] = self._build_mask_for_vessel(
                v_name, valid_specs, allow_noop=False
            )

        return mask

    # ------------------------------------------------------------------
    # Gymnasium overrides — pack masks & availability into info
    # ------------------------------------------------------------------

    def _augment_info(self, info: dict) -> dict:
        """Add ``action_masks`` and ``vessel_availability`` to *info*.

        By piggybacking on the same IPC payload as the observation and
        reward, the runner no longer needs separate ``envs.call()``
        round-trips to obtain these arrays.
        """
        info["action_masks"] = self.action_masks()
        info["vessel_availability"] = self.vessel_availability()
        return info

    def step(self, action):
        step_act = None
        if action is not None:
            domain_actions = self.action(action)
            # Only pass down non-empty actions to avoid unnecessary micro-steps
            if domain_actions:
                step_act = domain_actions

        obs, reward, terminated, truncated, info = self.env.step(step_act)
        self._augment_info(info)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._augment_info(info)
        return obs, info

    def describe_action_request(self, action) -> List[Dict[str, str]]:
        action_obj = self.action(action)
        return self._env_unwrapped.describe_action_request(action_obj)

    def _rebuild_obs(self, raw_dict: Dict):
        """Pipe a raw observation dict through the observation-wrapper chain.

        This applies the same transforms (hygiene normalisation →
        structured matrix) that ``step()`` and ``reset()`` apply, but
        without advancing the DES or touching episode-statistics
        wrappers.
        """
        obs: object = raw_dict
        for wrapper in self._obs_wrappers:
            obs = wrapper.observation(obs)
        return obs

    def micro_step(self, actions_batch, active_mask=None) -> tuple:
        """Register per-vessel intents and return updated obs + mask.

        Accepts the **full vectorised batch** of actions so that
        ``envs.call("micro_step", actions_batch, active_mask)`` works
        identically for both ``SyncVectorEnv`` and ``AsyncVectorEnv``.
        Each sub-env extracts its own row via ``_vector_env_index``.

        After registering, rebuilds the observation through the full
        wrapper chain and recomputes the action mask so that the next
        vessel in the AEC ordering can see the committed intents.

        Parameters
        ----------
        actions_batch : array-like
            Stacked joint actions, shape ``(num_envs, n_vessels)``.
        active_mask : array-like | None
            Boolean mask, shape ``(num_envs,)``.  If provided, the
            micro-step is skipped when this env's entry is ``False``.
            ``None`` means "always active".

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(obs, mask)`` where *obs* is the updated observation of
            shape ``(*obs_shape)`` (e.g. ``(total_rows, max_feats)``)
            and *mask* is the updated action mask of shape
            ``(n_vessels, opts_per_vessel)``.  Both are always returned
            (even when skipped) so the runner can ``np.stack`` without
            branching.
        """
        env_idx = self._env_unwrapped._vector_env_index
        actions_batch = np.asarray(actions_batch)

        # Dummy observation for inactive / terminated envs — the runner
        # will ignore these entries but we need a consistently shaped
        # array for np.stack.
        def _dummy():
            obs_shape = self.observation_space.shape or ()
            dummy_obs = np.zeros(obs_shape, dtype=np.float32)
            dummy_mask = np.ones((self.n_vessels, self.opts_per_vessel), dtype=bool)
            return dummy_obs, dummy_mask

        if active_mask is not None:
            active_mask = np.asarray(active_mask)
            if not active_mask[env_idx]:
                return _dummy()

        # If the env is terminated/truncated, skip silently.
        if self._env_unwrapped._terminated or self._env_unwrapped._truncated:
            return _dummy()

        my_action = actions_batch[env_idx]
        domain_actions = self.action(my_action)
        raw_dict = self._env_unwrapped.micro_step(domain_actions)

        # Rebuild the observation through the wrapper chain so the next
        # vessel sees the updated state (pending intents, etc.).
        obs = self._rebuild_obs(raw_dict)
        mask = self.action_masks()
        return obs, mask

    def action(self, action) -> list[Action]:
        action_arr = np.asarray(action, dtype=np.int64)
        if action_arr.shape != (self.n_vessels,):
            raise ValueError(
                f"Expected joint action of shape ({self.n_vessels},), got {action_arr.shape}."
            )

        actions: list[Action] = []
        for vessel_id, local_action in enumerate(action_arr.tolist()):
            # CRITICAL: We skip NOOPs completely. They are not translated
            # into Actions, meaning env.step() will not register them in DES.
            if local_action == self.noop_index:
                continue

            action_obj = self._decode_local_action(vessel_id, int(local_action))
            actions.append(action_obj)

        return actions

    def _decode_local_action(self, vessel_id: int, local_action: int) -> Action:
        # Offset by NOOP
        remainder = int(local_action) - 1
        if remainder < 0:
            raise ValueError("NOOP should be handled before decoding.")

        # Check Move
        if remainder < self.move_options:
            dest_id = remainder
            return Action(
                vessel_id=vessel_id,
                action_type=ActionType.MOVE,
                params=ActionParams(move_params=MoveParams(destination=dest_id)),
            )
        remainder -= self.move_options

        # Check Load
        if remainder < self.transfer_options:
            partner_idx = remainder // self.n_resources
            res_id = remainder % self.n_resources
            if partner_idx < self.n_sites:
                partner_type = PartnerType.SITE
                site_id = partner_idx
                vessel_partner_id = -1
            else:
                partner_type = PartnerType.VESSEL
                vessel_partner_id = partner_idx - self.n_sites
                site_id = -1
            return Action(
                vessel_id=vessel_id,
                action_type=ActionType.LOAD,
                params=ActionParams(
                    resource_transfer_params=ResourceTransferParams(
                        partner_type=partner_type,
                        vessel_id=vessel_partner_id,
                        site_id=site_id,
                        resource_id=res_id,
                    )
                ),
            )
        remainder -= self.transfer_options

        # Check Unload
        if remainder < self.transfer_options:
            partner_idx = remainder // self.n_resources
            res_id = remainder % self.n_resources
            if partner_idx < self.n_sites:
                partner_type = PartnerType.SITE
                site_id = partner_idx
                vessel_partner_id = -1
            else:
                partner_type = PartnerType.VESSEL
                vessel_partner_id = partner_idx - self.n_sites
                site_id = -1
            return Action(
                vessel_id=vessel_id,
                action_type=ActionType.UNLOAD,
                params=ActionParams(
                    resource_transfer_params=ResourceTransferParams(
                        partner_type=partner_type,
                        vessel_id=vessel_partner_id,
                        site_id=site_id,
                        resource_id=res_id,
                    )
                ),
            )
        remainder -= self.transfer_options

        # Check Idle (await-event: duration=None signals event-driven mode)
        if self.idle_options > 0 and remainder >= 0 and remainder < self.idle_options:
            return Action(
                vessel_id=vessel_id,
                action_type=ActionType.IDLE,
                params=ActionParams(idle_params=IdleParams(duration=-1.0)),
            )

        raise ValueError(f"Action index {local_action} implies invalid mapping")
