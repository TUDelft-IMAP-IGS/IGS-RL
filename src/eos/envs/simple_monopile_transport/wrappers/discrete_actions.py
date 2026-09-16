from typing import Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from eos.envs.simple_monopile_transport.gym_env import (
    Action,
    ActionParams,
    IdleParams,
    MoveParams,
    ResourceTransferParams,
    SimpleMonopileTransportEnv,
)
from eos.envs.simple_monopile_transport.types import ActionType, PartnerType


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Maps a single discrete integer (flat action) to the complex Action Dict.
    Includes Action Masking logic by querying the underlying simulator.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # We need access to the underlying env to calculate dimensions
        # using .unwrapped ensures we get the base SimpleMonopileTransportEnv
        self._env_unwrapped: SimpleMonopileTransportEnv = env.unwrapped

        # Access dimensions
        # Note: Ensure these properties are publicly accessible in your gym_wrapper
        base_env = self._env_unwrapped
        self.n_vessels = len(base_env._vessel_names)
        self.n_sites = len(base_env._site_names)
        self.n_resources = len(base_env._resource_names)
        self.allow_idle_actions = bool(
            getattr(base_env.cfg, "allow_idle_actions", True)
        )

        # 1. MOVE: Destination (Site ID)
        self.move_options = self.n_sites

        # 2. LOAD/UNLOAD: Partner (Site or Vessel) + Resource ID
        self._num_transfer_partners = self.n_sites + self.n_vessels
        self.transfer_options = self._num_transfer_partners * self.n_resources

        # 3. IDLE: Duration (optional)
        self.idle_durations = [1.0, 5.0, 10.0] if self.allow_idle_actions else []
        self.idle_options = len(self.idle_durations)

        # Total options per vessel
        self.opts_per_vessel = (
            self.move_options  # Move
            + self.transfer_options  # Load
            + self.transfer_options  # Unload
            + self.idle_options  # Idle (optional)
        )

        # Global NOOP: do nothing (advance simulation) regardless of vessel status.
        # This ensures the action mask is never empty.
        # ! Currently not used here because we assume sim always returns at least one valid action
        self.noop_index = 0

        total_actions = 1 + (self.n_vessels * self.opts_per_vessel)
        self.action_space = spaces.Discrete(total_actions)

        # Create Name-to-Index Maps for Masking
        self._site_map = {name: i for i, name in enumerate(base_env._site_names)}
        self._res_map = {name: i for i, name in enumerate(base_env._resource_names)}
        self._vessel_map = {name: i for i, name in enumerate(base_env._vessel_names)}

    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean mask of valid actions for the current state.
        Shape: (action_space.n,)
        dtype: bool
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        sim = self._env_unwrapped._sim

        # Iterate over all vessels to check what they can do
        for v_name in self._env_unwrapped._vessel_names:
            v_idx = self._vessel_map[v_name]

            # If vessel is busy in the simulator, it cannot accept a new command
            # (Assuming sequential control where we don't interrupt)
            if sim.is_vessel_busy(v_name):
                continue

            # 1. Get Domain-Specific Valid Actions from Simulator Rules
            valid_specs = sim.get_possible_actions(v_name)

            # Base index for this vessel in the flat vector (offset by global NOOP)
            base_offset = 1 + (v_idx * self.opts_per_vessel)

            # 2. Map valid ActionSpecs to flat indices
            for spec in valid_specs:
                if spec.action_type == ActionType.MOVE:
                    # Offset: 0 to move_options
                    if spec.destination_name in self._site_map:
                        dest_idx = self._site_map[spec.destination_name]
                        mask[base_offset + dest_idx] = True

                elif spec.action_type == ActionType.LOAD:
                    # Offset: move_options to + transfer_options
                    if (
                        spec.partner_name in self._site_map
                        and spec.resource_name in self._res_map
                    ):
                        p_idx = self._site_map[spec.partner_name]
                        r_idx = self._res_map[spec.resource_name]
                        local_idx = p_idx * self.n_resources + r_idx
                        mask[base_offset + self.move_options + local_idx] = True
                    elif (
                        spec.partner_name in self._vessel_map
                        and spec.resource_name in self._res_map
                    ):
                        p_idx = self.n_sites + self._vessel_map[spec.partner_name]
                        r_idx = self._res_map[spec.resource_name]
                        local_idx = p_idx * self.n_resources + r_idx
                        mask[base_offset + self.move_options + local_idx] = True

                elif spec.action_type == ActionType.UNLOAD:
                    # Offset: move + load to + transfer_options
                    if (
                        spec.partner_name in self._site_map
                        and spec.resource_name in self._res_map
                    ):
                        p_idx = self._site_map[spec.partner_name]
                        r_idx = self._res_map[spec.resource_name]
                        local_idx = p_idx * self.n_resources + r_idx
                        # Add move_options + transfer_options (skipping LOAD section)
                        mask[
                            base_offset
                            + self.move_options
                            + self.transfer_options
                            + local_idx
                        ] = True
                    elif (
                        spec.partner_name in self._vessel_map
                        and spec.resource_name in self._res_map
                    ):
                        p_idx = self.n_sites + self._vessel_map[spec.partner_name]
                        r_idx = self._res_map[spec.resource_name]
                        local_idx = p_idx * self.n_resources + r_idx
                        mask[
                            base_offset
                            + self.move_options
                            + self.transfer_options
                            + local_idx
                        ] = True

            if self.idle_options > 0:
                # Always allow IDLE if vessel is free
                # (Prevents Empty Mask Errors if specific rules aren't met)
                idle_start = (
                    base_offset + self.move_options + (2 * self.transfer_options)
                )
                for i in range(self.idle_options):
                    mask[idle_start + i] = True

        if not bool(np.any(mask)):
            raise RuntimeError(
                "Empty action mask detected (no valid actions). "
                "NOOP should always be valid; this indicates a bug in the wrapper logic."
            )

        return mask

    def action(self, action_idx: int) -> Action | list[Action]:
        # Global NOOP (no actions registered)
        if int(action_idx) == self.noop_index:
            return []

        action_idx = int(action_idx) - 1

        # 1. Decode Vessel
        vessel_id = int(action_idx // self.opts_per_vessel)
        remainder = int(action_idx % self.opts_per_vessel)

        # 2. Decode Action Type and Params

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

        # Check Idle
        if self.idle_options > 0 and remainder >= 0 and remainder < self.idle_options:
            duration = self.idle_durations[remainder]
            return Action(
                vessel_id=vessel_id,
                action_type=ActionType.IDLE,
                params=ActionParams(idle_params=IdleParams(duration=duration)),
            )

        raise ValueError(f"Action index {action_idx} implies invalid mapping")
