from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger

from eos.config import EnvConfig
from eos.envs.simple_monopile_transport.simulator import SimpleMonopileTransportSim
from eos.envs.simple_monopile_transport.types import ActionType, PartnerType, VisitMode

from .activity_builder import ActionSpec
from .costs import CostContext
from .milestone_tracker import MilestoneContext, MilestoneTracker
from .reward import RewardMetrics, SMTMultiObjectiveRewarder

# Sentinel values for "no intent" fields.  These are chosen so that they
# sit just outside the valid ID range for each categorical and are easy
# to recognise in debug output.
INTENT_NONE_ACTION: int = -1  # no queued action type
INTENT_NONE_DESTINATION: int = -1  # no queued destination site
INTENT_NONE_RESOURCE: int = -1  # no queued resource
INTENT_NONE_PARTNER: int = -1  # no queued partner

# Mapping from site role strings to categorical integers for observations.
SITE_ROLE_TO_INT = {
    "source": 0,
    "staging": 1,
    "installation": 2,
}


@dataclass(slots=True)
class IntentState:
    """Describes a vessel's queued (pending) intent registered via micro-step.

    During the AEC micro-stepping phase, each vessel registers an intent
    (an action queued in the DES but not yet executed).  This dataclass
    captures the *type* of that intent plus the key parameters so that
    subsequent vessels in the ordering can observe what earlier vessels
    have committed to.

    When no intent has been registered the sentinel values
    ``INTENT_NONE_*`` are used (all equal to ``-1``).

    Attributes
    ----------
    action_type : int
        The :class:`ActionType` ordinal of the queued action, or
        ``INTENT_NONE_ACTION`` when no intent is queued.
    destination : int
        Site ID of the move destination (for MOVE intents) or the
        partner site ID (for LOAD/UNLOAD intents), or
        ``INTENT_NONE_DESTINATION``.
    resource_id : int
        Resource ID being transferred (for LOAD/UNLOAD intents), or
        ``INTENT_NONE_RESOURCE``.
    """

    action_type: int = INTENT_NONE_ACTION
    destination: int = INTENT_NONE_DESTINATION
    resource_id: int = INTENT_NONE_RESOURCE

    @property
    def has_intent(self) -> bool:
        return self.action_type != INTENT_NONE_ACTION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": np.array([self.action_type], dtype=np.int64),
            "destination": np.array([self.destination], dtype=np.int64),
            "resource_id": np.array([self.resource_id], dtype=np.int64),
        }

    @staticmethod
    def space(num_action_types: int, num_sites: int, num_resources: int):
        """Gym space for a single intent.

        The ranges include the sentinel value (``-1``) as a valid
        lower bound so the space is well-defined even when no intent
        is queued.
        """
        return spaces.Dict(
            {
                "action_type": spaces.Box(
                    low=-1,
                    high=num_action_types - 1,
                    shape=(1,),
                    dtype=np.int64,
                ),
                "destination": spaces.Box(
                    low=-1,
                    high=num_sites - 1,
                    shape=(1,),
                    dtype=np.int64,
                ),
                "resource_id": spaces.Box(
                    low=-1,
                    high=num_resources - 1,
                    shape=(1,),
                    dtype=np.int64,
                ),
            }
        )


@dataclass(slots=True)
class MoveParams:
    destination: int = -1  # site ID

    def __str__(self) -> str:
        return f"→ site_{self.destination}"

    @staticmethod
    def space(num_sites: int):
        return spaces.Discrete(num_sites, start=0, dtype=np.int64)


@dataclass(slots=True)
class ResourceTransferParams:
    partner_type: PartnerType = PartnerType.SITE
    vessel_id: int = -1
    site_id: int = -1
    resource_id: int = -1  # the id of the resource to move

    def __str__(self) -> str:
        pid = (
            self.vessel_id if self.partner_type == PartnerType.VESSEL else self.site_id
        )
        return f"↔ {self.partner_type.name}_{pid}_{self.resource_id}"

    @staticmethod
    def space(num_sites: int, num_vessels: int, num_resources: int):
        return spaces.Dict(
            {
                "partner_type": spaces.Discrete(2, start=0, dtype=np.int64),
                "vessel_id": spaces.Discrete(num_vessels, start=0, dtype=np.int64),
                "site_id": spaces.Discrete(num_sites, start=0, dtype=np.int64),
                "resource_id": spaces.Discrete(num_resources, start=0, dtype=np.int64),
            }
        )


@dataclass(slots=True)
class IdleParams:
    duration: float = -1.0

    def __str__(self) -> str:
        return f"⏸ {self.duration:.1f}s"

    @staticmethod
    def space():
        return spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float64)


@dataclass(slots=True)
class ActionParams:
    """Container for all possible action parameters.

    This structure is flattened to be compatible with Gymnasium's Dict space.
    Only the parameters relevant to the chosen action_type are used.
    """

    move_params: MoveParams = field(default_factory=MoveParams)
    resource_transfer_params: ResourceTransferParams = field(
        default_factory=ResourceTransferParams
    )
    idle_params: IdleParams = field(default_factory=IdleParams)

    @staticmethod
    def space(num_sites: int, num_vessels: int, num_resources: int):
        return spaces.Dict(
            {
                "move_params": MoveParams.space(num_sites=num_sites),
                "resource_transfer_params": ResourceTransferParams.space(
                    num_sites=num_sites,
                    num_vessels=num_vessels,
                    num_resources=num_resources,
                ),
                "idle_params": IdleParams.space(),
            }
        )


@dataclass(slots=True)
class Action:
    vessel_id: int
    action_type: ActionType
    params: ActionParams

    def __str__(self) -> str:
        """Pretty string representation for logging."""
        action_name = self.action_type.pretty_name()

        if self.action_type == ActionType.MOVE:
            params_str = str(self.params.move_params)
        elif self.action_type in (ActionType.LOAD, ActionType.UNLOAD):
            params_str = str(self.params.resource_transfer_params)
        elif self.action_type == ActionType.IDLE:
            params_str = str(self.params.idle_params)
        else:
            params_str = str(self.params)

        return f"vessel_{self.vessel_id}: {action_name} {params_str}".strip()

    def pretty(
        self,
        vessel_names: List[str] | None = None,
        site_names: List[str] | None = None,
        resource_names: List[str] | None = None,
        vessel_to_site: Dict[str, str] | None = None,
    ) -> str:
        """Human-readable string with name mappings.

        Includes the ``vessel@site:`` prefix for use in debug logs and
        notebooks where the context isn't available in separate columns.

        Parameters
        ----------
        vessel_names : List[str] | None
            List mapping vessel_id -> vessel name. If None falls back to 'vessel_<id>'.
        site_names : List[str] | None
            List mapping site_id -> site name. If None falls back to 'site_<id>'.
        resource_names : List[str] | None
            List mapping resource_id -> resource name for transfer actions.
        vessel_to_site : Dict[str, str] | None
            Dict mapping vessel name -> site name

        Returns
        -------
        str
            A human-friendly action description using provided names.
        """
        # Resolve vessel name
        vessel = (
            vessel_names[self.vessel_id]
            if vessel_names and 0 <= self.vessel_id < len(vessel_names)
            else f"vessel_{self.vessel_id}"
        )

        vessel_site = (
            vessel_to_site.get(vessel, "UKNOWN")
            if vessel_to_site
            else f"vessel_site_{vessel}"
        )

        compact = self.compact(
            vessel_names=vessel_names,
            site_names=site_names,
            resource_names=resource_names,
            vessel_to_site=vessel_to_site,
        )
        return f"{vessel}@{vessel_site}: {compact}"

    def compact(
        self,
        vessel_names: List[str] | None = None,
        site_names: List[str] | None = None,
        resource_names: List[str] | None = None,
        vessel_to_site: Dict[str, str] | None = None,
    ) -> str:
        """Compact action description matching the DES activity name style.

        Unlike :meth:`pretty`, this omits the ``vessel@site:`` prefix so
        it can be used in tabular contexts where vessel and location are
        already in separate columns.

        Parameters
        ----------
        vessel_names : List[str] | None
            List mapping vessel_id -> vessel name.
        site_names : List[str] | None
            List mapping site_id -> site name.
        resource_names : List[str] | None
            List mapping resource_id -> resource name.
        vessel_to_site : Dict[str, str] | None
            Dict mapping vessel name -> site name.

        Returns
        -------
        str
            A compact action description (e.g. ``"HTV → MY"``,
            ``"Load HTV ↔ FY (large_mp)"``).
        """
        vessel = (
            vessel_names[self.vessel_id]
            if vessel_names and 0 <= self.vessel_id < len(vessel_names)
            else f"vessel_{self.vessel_id}"
        )

        if self.action_type == ActionType.MOVE:
            dest_id = self.params.move_params.destination
            dest = (
                site_names[dest_id]
                if site_names and 0 <= dest_id < len(site_names)
                else f"site_{dest_id}"
            )
            return f"{vessel} → {dest}"

        if self.action_type in (ActionType.LOAD, ActionType.UNLOAD):
            params = self.params.resource_transfer_params
            # partner_type: PartnerType.SITE or PartnerType.VESSEL
            if params.partner_type == PartnerType.VESSEL:  # vessel partner
                partner_id = params.vessel_id
                partner = (
                    vessel_names[partner_id]
                    if vessel_names and 0 <= partner_id < len(vessel_names)
                    else f"vessel_{partner_id}"
                )

                partner_site = (
                    vessel_to_site.get(partner, "UKNOWN")
                    if vessel_to_site
                    else f"vessel_site_{partner}"
                )

                partner_str = f"{partner}@{partner_site}"
            else:  # site partner (default)
                partner_id = params.site_id
                partner = (
                    site_names[partner_id]
                    if site_names and 0 <= partner_id < len(site_names)
                    else f"site_{partner_id}"
                )

                partner_str = f"{partner}"

            res_id = params.resource_id
            resource = (
                resource_names[res_id]
                if resource_names and 0 <= res_id < len(resource_names)
                else f"res_{res_id}"
            )
            verb = "Load" if self.action_type == ActionType.LOAD else "Unload"
            return f"{verb} {vessel} ↔ {partner_str} ({resource})"

        if self.action_type == ActionType.IDLE:
            return f"Idle {vessel} ⏸ {self.params.idle_params.duration:.0f}s"

        # Fallback to generic representation
        return str(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create Action object from dictionary (e.g. from space.sample()).

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with keys 'vessel_id', 'action_type', 'params'.
            'params' is expected to be a dict with keys 'move_params',
            'resource_transfer_params', 'idle_params'.

        Returns
        -------
        Action
            Reconstructed Action object.
        """
        vessel_id = int(data["vessel_id"])
        action_type = ActionType(int(data["action_type"]))

        params_dict = data["params"]

        # Parse move params
        move_val = params_dict["move_params"]
        move_params = MoveParams(destination=int(move_val))

        # Parse transfer params
        transfer_val = params_dict["resource_transfer_params"]
        resource_transfer_params = ResourceTransferParams(
            partner_type=PartnerType(int(transfer_val["partner_type"])),
            vessel_id=int(transfer_val["vessel_id"]),
            site_id=int(transfer_val["site_id"]),
            resource_id=int(transfer_val["resource_id"]),
        )

        # Parse idle params
        idle_val = params_dict["idle_params"]
        duration = (
            float(idle_val[0])
            if isinstance(idle_val, (list, np.ndarray))
            else float(idle_val)
        )
        idle_params = IdleParams(duration=duration)

        action_params = ActionParams(
            move_params=move_params,
            resource_transfer_params=resource_transfer_params,
            idle_params=idle_params,
        )

        return cls(vessel_id=vessel_id, action_type=action_type, params=action_params)

    def to_action_spec(
        self,
        vessel_names: List[str],
        site_names: List[str],
        resource_names: List[str],
    ) -> ActionSpec:
        """Convert gym Action to domain-neutral ActionSpec.

        Parameters
        ----------
        vessel_names : List[str]
            Ordered list of vessel names (index = vessel_id)
        site_names : List[str]
            Ordered list of site names (index = site_id)
        resource_names : List[str]
            Available resource IDs in the simulation

        Returns
        -------
        ActionSpec
            Domain-neutral action specification for ActivityBuilder
        """

        vessel_name = vessel_names[self.vessel_id]

        if self.action_type == ActionType.MOVE:
            params = self.params.move_params
            destination_name = site_names[params.destination]
            return ActionSpec(
                action_type=ActionType.MOVE,
                vessel_name=vessel_name,
                destination_name=destination_name,
            )

        elif self.action_type in (ActionType.LOAD, ActionType.UNLOAD):
            params = self.params.resource_transfer_params
            partner_name = (
                site_names[params.site_id]
                if params.partner_type == PartnerType.SITE
                else vessel_names[params.vessel_id]
            )

            resource_id = params.resource_id
            resource_name = resource_names[resource_id]
            return ActionSpec(
                action_type=self.action_type,
                vessel_name=vessel_name,
                partner_name=partner_name,
                resource_name=resource_name,
                amount=1,
            )

        elif self.action_type == ActionType.IDLE:
            params = self.params.idle_params
            return ActionSpec(
                action_type=ActionType.IDLE,
                vessel_name=vessel_name,
                duration=params.duration,
            )

        else:
            raise ValueError(f"Unknown action type: {self.action_type}")

    @classmethod
    def from_action_spec(
        cls,
        spec: ActionSpec,
        vessel_name_to_id: Dict[str, int],
        site_name_to_id: Dict[str, int],
        resource_name_to_id: Dict[str, int],
    ) -> "Action":
        """Create Action object from ActionSpec.

        Parameters
        ----------
        spec : ActionSpec
            The action specification.
        vessel_name_to_id : Dict[str, int]
            Mapping from vessel name to ID.
        site_name_to_id : Dict[str, int]
            Mapping from site name to ID.
        resource_name_to_id : Dict[str, int]
            Mapping from resource name to ID.

        Returns
        -------
        Action
            The constructed Action object.
        """
        if spec.vessel_name not in vessel_name_to_id:
            raise ValueError(f"Unknown vessel name: {spec.vessel_name}")

        vessel_id = vessel_name_to_id[spec.vessel_name]

        action_type = spec.action_type

        params = ActionParams()

        if action_type == ActionType.MOVE:
            if not spec.destination_name:
                raise ValueError("Move action requires destination_name")
            if spec.destination_name not in site_name_to_id:
                raise ValueError(f"Unknown destination: {spec.destination_name}")

            dest_id = site_name_to_id[spec.destination_name]
            params.move_params = MoveParams(destination=dest_id)

        elif action_type in (ActionType.LOAD, ActionType.UNLOAD):
            if not spec.partner_name:
                raise ValueError("Load/Unload action requires partner_name")

            if spec.partner_name in site_name_to_id:
                partner_type = PartnerType.SITE
                site_id = site_name_to_id[spec.partner_name]
                vessel_partner_id = -1
            elif spec.partner_name in vessel_name_to_id:
                partner_type = PartnerType.VESSEL
                vessel_partner_id = vessel_name_to_id[spec.partner_name]
                site_id = -1
            else:
                raise ValueError(f"Unknown partner: {spec.partner_name}")

            resource_id = (
                resource_name_to_id.get(spec.resource_name, -1)
                if spec.resource_name
                else -1
            )

            params.resource_transfer_params = ResourceTransferParams(
                partner_type=partner_type,
                vessel_id=vessel_partner_id,
                site_id=site_id,
                resource_id=resource_id,
            )

        elif action_type == ActionType.IDLE:
            duration = spec.duration if spec.duration is not None else 0.0
            params.idle_params = IdleParams(duration=duration)

        return cls(vessel_id=vessel_id, action_type=action_type, params=params)

    @staticmethod
    def space(num_vessels: int, num_sites: int, num_resources: int):
        return spaces.Dict(
            {
                "vessel_id": spaces.Discrete(num_vessels, start=0, dtype=np.int64),
                "action_type": spaces.Discrete(
                    len(ActionType), start=0, dtype=np.int64
                ),
                "params": ActionParams.space(
                    num_vessels=num_vessels,
                    num_sites=num_sites,
                    num_resources=num_resources,
                ),
            }
        )


@dataclass(slots=True)
class ResourceState:
    """Tracks the load and capacity of a specific resource type.

    Used in observations to represent the state of resources on vessels and sites.
    """

    load: int = 0  # current amount
    capacity: int = 0  # max possible amount
    next_available_in: float = (
        0.0  # hours until next unit fabricated (0 for non-source sites)
    )

    def free(self) -> int:
        return self.capacity - self.load

    def is_full(self) -> bool:
        return self.load >= self.capacity

    def is_empty(self) -> bool:
        return self.load == 0

    @staticmethod
    def space():
        return spaces.Dict(
            {
                "load": spaces.Box(
                    low=0, high=np.iinfo(np.int64).max - 1, shape=(1,), dtype=np.int64
                ),
                "capacity": spaces.Box(
                    low=0, high=np.iinfo(np.int64).max - 1, shape=(1,), dtype=np.int64
                ),
                "next_available_in": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float64
                ),
            }
        )


@dataclass(slots=True)
class ResourceBundle:
    """Container for multiple resource types with their states.

    This is generic and adapts to whatever resources are present in the simulation.
    """

    resources: Dict[str, ResourceState] = field(default_factory=dict)

    @staticmethod
    def from_inventory_dict(
        inventory_dict: Dict[str, Dict[str, int]],
    ) -> "ResourceBundle":
        """Create ResourceBundle from simulator inventory format.

        Parameters
        ----------
        inventory_dict : Dict[str, Dict[str, int]]
            Format: {"resource_name": {"load": X, "capacity": Y}, ...}

        Returns
        -------
        ResourceBundle
            Bundle containing ResourceState for each resource type.
        """
        resources = {
            resource_name: ResourceState(
                load=data["load"],
                capacity=data["capacity"],
                next_available_in=data.get("next_available_in", 0.0),
            )
            for resource_name, data in inventory_dict.items()
        }
        return ResourceBundle(resources=resources)

    def to_vector(self) -> List[float]:
        """Stable ordering for encoders: [res1_load, res1_cap, res2_load, res2_cap, ...]."""
        result = []
        for resource_name in sorted(self.resources.keys()):
            result.extend(
                [
                    self.resources[resource_name].load,
                    self.resources[resource_name].capacity,
                ]
            )
        return result

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            resource_name: {
                "load": np.array([state.load], dtype=np.int64),
                "capacity": np.array([state.capacity], dtype=np.int64),
                "next_available_in": np.array(
                    [state.next_available_in], dtype=np.float64
                ),
            }
            for resource_name, state in self.resources.items()
        }

    @staticmethod
    def space(resource_names: List[str]) -> spaces.Dict:
        """Create gym space for ResourceBundle with given resource types.

        Parameters
        ----------
        resource_names : List[str]
            List of resource identifiers (e.g., ['large_mp', 'small_mp']).

        Returns
        -------
        spaces.Dict
            Gym Dict space with one entry per resource type.
        """
        return spaces.Dict(
            {resource_name: ResourceState.space() for resource_name in resource_names}
        )


class VesselStatus(IntEnum):
    IDLE = 0
    BUSY = 1

    @staticmethod
    def from_bool(is_busy: bool):
        return VesselStatus(int(is_busy))

    @staticmethod
    def space():
        return spaces.Discrete(len(VesselStatus), start=0, dtype=np.int64)


@dataclass(slots=True)
class VesselObs:
    id: int
    position: int
    status: VesselStatus
    visit_modes: Dict[str, np.ndarray]
    inventory: ResourceBundle = field(default_factory=dict)
    intent: IntentState = field(default_factory=IntentState)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize explicitly to match Gym space structure."""
        return {
            "id": self.id,
            "position": self.position,
            "status": int(self.status),
            "visit_modes": self.visit_modes,
            "inventory": self.inventory.to_dict(),
            "intent": self.intent.to_dict(),
        }

    @staticmethod
    def space(
        num_vessels: int,
        num_sites: int,
        resource_names: List[str],
        num_action_types: int = len(ActionType),
        num_resources: int | None = None,
    ):
        if num_resources is None:
            num_resources = len(resource_names)
        return spaces.Dict(
            {
                "id": spaces.Discrete(num_vessels, start=0, dtype=np.int64),
                "position": spaces.Discrete(num_sites, start=0, dtype=np.int64),
                "status": VesselStatus.space(),
                "visit_modes": spaces.Dict(
                    {
                        res: spaces.Box(
                            low=0, high=len(VisitMode) - 1, shape=(1,), dtype=np.int64
                        )
                        for res in resource_names
                    }
                ),
                "inventory": ResourceBundle.space(resource_names),
                "intent": IntentState.space(
                    num_action_types=num_action_types,
                    num_sites=num_sites,
                    num_resources=num_resources,
                ),
            }
        )


@dataclass(slots=True)
class SiteObs:
    id: int
    role: int = 0  # categorical: source=0, staging=1, installation=2
    stock: ResourceBundle = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "stock": self.stock.to_dict(),
        }

    @staticmethod
    def space(num_sites: int, resource_names: List[str], num_site_roles: int = 3):
        return spaces.Dict(
            {
                "id": spaces.Discrete(num_sites, start=0, dtype=np.int64),
                "role": spaces.Discrete(num_site_roles, start=0, dtype=np.int64),
                "stock": ResourceBundle.space(resource_names),
            }
        )


@dataclass(slots=True)
class GoalObs:
    """Observation of a single goal's state.

    Attributes
    ----------
    id : int
        Goal index (0-based).
    target_site : int
        Site ID where the goal must be fulfilled (categorical).
    resource_type : int
        Resource type index (categorical).
    required : int
        Total units required by this goal.
    installed : int
        Units currently installed / delivered.
    remaining : int
        Units still needed (required - installed, clamped >= 0).
    progress : float
        Fraction complete (installed / required), in [0, 1].
    deadline : float
        Hours remaining until deadline (0 if no deadline or already passed).
    is_failed : int
        1 if the goal has failed (deadline missed), 0 otherwise.
    is_blocked : int
        1 if the goal is blocked by unmet dependencies, 0 otherwise.
    dep_progress : float
        Minimum progress of prerequisite goals (1.0 if no deps).
    blocked_by : np.ndarray
        Multi-hot vector of length num_goals indicating which goals block this one.
    """

    id: int = 0
    target_site: int = 0
    resource_type: int = 0
    required: int = 0
    installed: int = 0
    remaining: int = 0
    progress: float = 0.0
    deadline: float = 0.0
    is_failed: int = 0
    is_blocked: int = 0
    dep_progress: float = 1.0
    blocked_by: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "target_site": self.target_site,
            "resource_type": self.resource_type,
            "required": np.array([self.required], dtype=np.int64),
            "installed": np.array([self.installed], dtype=np.int64),
            "remaining": np.array([self.remaining], dtype=np.int64),
            "progress": np.array([self.progress], dtype=np.float64),
            "deadline": np.array([self.deadline], dtype=np.float64),
            "is_failed": np.array([self.is_failed], dtype=np.int64),
            "is_blocked": np.array([self.is_blocked], dtype=np.int64),
            "dep_progress": np.array([self.dep_progress], dtype=np.float64),
            "blocked_by": self.blocked_by.copy(),
        }

    @staticmethod
    def space(
        num_goals: int,
        num_sites: int,
        num_resource_types: int,
    ):
        return spaces.Dict(
            {
                "id": spaces.Discrete(max(num_goals, 1), start=0, dtype=np.int64),
                "target_site": spaces.Discrete(num_sites, start=0, dtype=np.int64),
                "resource_type": spaces.Discrete(
                    max(num_resource_types, 1), start=0, dtype=np.int64
                ),
                "required": spaces.Box(
                    low=0, high=np.iinfo(np.int64).max - 1, shape=(1,), dtype=np.int64
                ),
                "installed": spaces.Box(
                    low=0, high=np.iinfo(np.int64).max - 1, shape=(1,), dtype=np.int64
                ),
                "remaining": spaces.Box(
                    low=0, high=np.iinfo(np.int64).max - 1, shape=(1,), dtype=np.int64
                ),
                "progress": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
                "deadline": spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float64
                ),
                "is_failed": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int64),
                "is_blocked": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int64),
                "dep_progress": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float64
                ),
                "blocked_by": spaces.MultiBinary(max(num_goals, 1)),
            }
        )


@dataclass(slots=True)
class GlobalObs:
    current_time: float
    num_pending_tasks: int
    step: int
    in_no_install_window: int  # 0 or 1
    time_to_install_window_flip: float  # hours until window state changes
    accumulated_travel_cost: float = 0.0
    accumulated_storage_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_time": np.array([self.current_time], dtype=np.float64),
            "num_pending_tasks": np.array([self.num_pending_tasks], dtype=np.int64),
            "step": np.array([self.num_pending_tasks], dtype=np.int64),
            "in_no_install_window": np.array(
                [self.in_no_install_window], dtype=np.int64
            ),
            "time_to_install_window_flip": np.array(
                [self.time_to_install_window_flip], dtype=np.float64
            ),
            "accumulated_travel_cost": np.array(
                [self.accumulated_travel_cost], dtype=np.float64
            ),
            "accumulated_storage_cost": np.array(
                [self.accumulated_storage_cost], dtype=np.float64
            ),
        }

    @staticmethod
    def space():
        return spaces.Dict(
            {
                "current_time": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float64
                ),
                "num_pending_tasks": spaces.Box(
                    low=0, high=np.iinfo(np.int64).max - 1, shape=(1,), dtype=np.int64
                ),
                "step": spaces.Box(
                    low=0, high=np.iinfo(np.int64).max - 1, shape=(1,), dtype=np.int64
                ),
                "in_no_install_window": spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.int64
                ),
                "time_to_install_window_flip": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float64
                ),
                "accumulated_travel_cost": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float64
                ),
                "accumulated_storage_cost": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float64
                ),
            }
        )


@dataclass(slots=True)
class Observation:
    vessels: List[VesselObs]  # per-vessel state
    sites: List[SiteObs]  # per-site state
    goals: List[GoalObs]  # per-goal state
    global_obs: GlobalObs  # scenario-level state

    def pretty(
        self,
        vessel_names: List[str],
        site_names: List[str],
        resource_names: List[str],
    ) -> str:
        parts: List[str] = []
        window_flag = "YES" if self.global_obs.in_no_install_window else "NO"
        flip_hrs = self.global_obs.time_to_install_window_flip / 3600.0
        parts.append(
            f"t={float(self.global_obs.current_time):.1f} pending={int(self.global_obs.num_pending_tasks)}"
            f" no_install={window_flag} flip_in={flip_hrs:.1f}h"
        )

        def fmt_bundle(bundle: ResourceBundle) -> str:
            items = []
            for res in resource_names:
                state = bundle.resources.get(res)
                if state is None:
                    continue
                if state.capacity > 0 or state.load > 0:
                    items.append(f"{res}:{state.load}/{state.capacity}")
            return ", ".join(items) if items else "-"

        def fmt_intent(intent: IntentState) -> str:
            if not intent.has_intent:
                return "NONE"
            try:
                atype = ActionType(intent.action_type).pretty_name()
            except ValueError:
                atype = f"?{intent.action_type}"
            dest = (
                site_names[intent.destination]
                if 0 <= intent.destination < len(site_names)
                else str(intent.destination)
            )
            res = (
                resource_names[intent.resource_id]
                if 0 <= intent.resource_id < len(resource_names)
                else str(intent.resource_id)
            )
            return f"{atype} dest={dest} res={res}"

        parts.append("Vessels:")
        for v in self.vessels:
            vname = vessel_names[v.id] if 0 <= v.id < len(vessel_names) else f"v{v.id}"
            sname = (
                site_names[v.position]
                if 0 <= v.position < len(site_names)
                else f"s{v.position}"
            )
            status = "BUSY" if int(v.status) == int(VesselStatus.BUSY) else "IDLE"
            intent_str = fmt_intent(v.intent)
            parts.append(
                f"  {vname} @ {sname} {status} inv[{fmt_bundle(v.inventory)}]"
                f" intent[{intent_str}]"
            )

        parts.append("Sites:")
        for s in self.sites:
            sname = site_names[s.id] if 0 <= s.id < len(site_names) else f"s{s.id}"
            parts.append(f"  {sname} stock[{fmt_bundle(s.stock)}]")

        if self.goals:
            parts.append("Goals:")
            for g in self.goals:
                site_label = (
                    site_names[g.target_site]
                    if 0 <= g.target_site < len(site_names)
                    else f"s{g.target_site}"
                )
                res_label = (
                    resource_names[g.resource_type]
                    if 0 <= g.resource_type < len(resource_names)
                    else f"r{g.resource_type}"
                )
                status = (
                    "FAILED"
                    if g.is_failed
                    else ("BLOCKED" if g.is_blocked else "ACTIVE")
                )
                parts.append(
                    f"  goal[{g.id}] {res_label}@{site_label} "
                    f"{g.installed}/{g.required} ({g.progress:.0%}) "
                    f"deadline={g.deadline:.1f}h {status}"
                )

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vessels": tuple(v.to_dict() for v in self.vessels),
            "sites": tuple(s.to_dict() for s in self.sites),
            "goals": tuple(g.to_dict() for g in self.goals),
            "global_obs": self.global_obs.to_dict(),
        }

    @staticmethod
    def space(
        num_vessels: int,
        num_sites: int,
        resource_names: List[str],
        num_resources: int | None = None,
        num_goals: int = 0,
        num_resource_types: int | None = None,
    ):
        if num_resources is None:
            num_resources = len(resource_names)
        if num_resource_types is None:
            num_resource_types = len(resource_names)
        return spaces.Dict(
            {
                "vessels": spaces.Tuple(
                    [
                        VesselObs.space(
                            num_vessels=num_vessels,
                            num_sites=num_sites,
                            resource_names=resource_names,
                            num_resources=num_resources,
                        )
                        for _ in range(num_vessels)
                    ]
                ),
                "sites": spaces.Tuple(
                    [
                        SiteObs.space(
                            num_sites=num_sites, resource_names=resource_names
                        )
                        for _ in range(num_sites)
                    ]
                ),
                "goals": spaces.Tuple(
                    [
                        GoalObs.space(
                            num_goals=num_goals,
                            num_sites=num_sites,
                            num_resource_types=num_resource_types,
                        )
                        for _ in range(max(num_goals, 1))
                    ]
                ),
                "global_obs": GlobalObs.space(),
            }
        )


class SimpleMonopileTransportEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg: EnvConfig, render_mode: str | None = None) -> None:
        self.cfg = cfg
        self.render_mode = render_mode or cfg.render.mode
        self._sim = SimpleMonopileTransportSim(cfg=cfg.sim)
        self._terminated = False
        self._truncated = False

        self._vessel_names = self._sim.get_vessel_names()
        self._site_names = self._sim.get_site_names()
        self._resource_names = self._sim.resource_names
        self._num_vessels = len(self._vessel_names)
        self._num_sites = len(self._site_names)
        self._num_resources = len(self._resource_names)
        self._num_goals = len(self._sim.config.goals or [])

        # * This should be consistent between resets
        self._vessel_name_to_id = self._sim_object_names_to_id(self._vessel_names)
        self._site_name_to_id = self._sim_object_names_to_id(self._site_names)
        self._resource_names_to_id = self._sim_object_names_to_id(self._resource_names)

        self._time_budget_hours = cfg.time_budget
        self._episode_start_time_unix = self._sim.des_env.now

        self._renderer = None
        self._last_obs_structured: Observation | None = None
        self._last_info: Dict[str, Any] | None = None

        # Compute worst-case cost budgets (used for obs normalization,
        # C_max termination, and terminal cost squashing).
        self._travel_cost_budget, self._storage_cost_budget, self._worst_case_cost = (
            self._compute_cost_budgets()
        )

        self._rewarder = SMTMultiObjectiveRewarder(
            cfg=self.cfg.reward,
            worst_case_cost=self._worst_case_cost,
        )

        # Milestone tracker for DPBRS shaping
        self._milestone_tracker = self._build_milestone_tracker()

        self.metadata["render_fps"] = int(self.cfg.render.fps)

        self.observation_space = Observation.space(
            num_vessels=self._num_vessels,
            num_sites=self._num_sites,
            resource_names=self._resource_names,
            num_resources=self._num_resources,
            num_goals=self._num_goals,
        )

        # Worst-case cost budgets are computed earlier in __init__
        # (self._travel_cost_budget, self._storage_cost_budget, self._worst_case_cost)
        # and are used by the ObservationHygieneWrapper for normalization.

        self.action_space = Action.space(
            num_vessels=self._num_vessels,
            num_sites=self._num_sites,
            num_resources=self._num_resources,
        )

        self._step = 0
        self._micro_step = 0

        # Per-vessel last registered activity short ID (e.g. "A3").
        # Populated by :meth:`micro_step`, consumed by
        # :meth:`describe_action_request`.
        self._last_activity_id_per_vessel: Dict[str, str | None] = {
            name: None for name in self._vessel_names
        }

        # Per-vessel dependency short IDs from the last registration
        # (e.g. ["A1", "F3"]).  Populated alongside the activity ID.
        self._last_dep_ids_per_vessel: Dict[str, List[str]] = {
            name: [] for name in self._vessel_names
        }

        self.gamma_t = 1

    # ------------------------------------------------------------------
    # Milestone tracker construction
    # ------------------------------------------------------------------

    def _compute_cost_budgets(self) -> tuple[float, float, float]:
        """Compute worst-case cost budgets.

        Returns static reference scales: the maximum possible accumulated
        cost for each component if the episode ran for the full time budget.

        Also returns a **total worst-case cost** (global-weighted sum of all
        enabled components) used by the rewarder as the failure penalty when
        ``terminal_cost_evaluation`` is active.

        Returns
        -------
        travel_budget : float
            Worst-case travel cost (for observation normalization).
        storage_budget : float
            Worst-case storage cost (for observation normalization).
        total_worst_case_cost : float
            Global-weighted sum of worst-case costs across all enabled
            components (elapsed time + travel + storage).
        """
        from eos.envs.simple_monopile_transport.costs.budgets import (
            compute_cost_budgets,
        )

        return compute_cost_budgets(
            self._time_budget_hours, self.cfg.reward.costs, self.cfg.sim
        )

    def _build_milestone_tracker(self) -> MilestoneTracker:
        """Create a :class:`MilestoneTracker` from the current sim config."""
        sim_cfg = self.cfg.sim
        if sim_cfg is None:
            raise RuntimeError(
                "Milestone reward mode requires a sim config (env.sim) to be set."
            )
        site_roles = {name: cfg.role for name, cfg in sim_cfg.sites.items()}
        vessel_roles = {name: cfg.role for name, cfg in sim_cfg.vessels.items()}
        goal_configs = sim_cfg.goals or []
        # Compute phi_max: the maximum DPBRS potential is bounded as a
        # fraction of the terminal bonus to prevent the completion-avoidance
        # pathology.  With terminal zeroing enforced (Φ(s_T)=0), if
        # phi_max ≈ B the agent sees a net-negative terminal step and
        # rationally avoids finishing.  By setting phi_max = fraction * B,
        # the "profit" on completion (B - Φ_prev) remains reliably positive.
        fraction = self.cfg.reward.milestone.phi_max_fraction
        pfm_cfg = getattr(self.cfg.reward, "pfm", None)
        if pfm_cfg is not None and pfm_cfg.enabled:
            terminal_bonus = pfm_cfg.completion_baseline + pfm_cfg.phi_optimism
        else:
            terminal_bonus = self.cfg.reward.completion_bonus
        phi_max = fraction * terminal_bonus

        return MilestoneTracker(
            cfg=self.cfg.reward.milestone,
            goal_configs=goal_configs,
            site_roles=site_roles,
            vessel_roles=vessel_roles,
            phi_max=phi_max,
        )

    def _build_milestone_context(
        self,
        site_inventories: Dict[str, Dict[str, int]],
        vessel_inventories: Dict[str, Dict[str, int]],
        vessel_sites: Dict[str, str],
        gamma_t: float,
        is_terminal: bool = False,
    ) -> MilestoneContext:
        """Assemble a :class:`MilestoneContext`."""
        return MilestoneContext(
            site_inventories=site_inventories,
            vessel_inventories=vessel_inventories,
            vessel_sites=vessel_sites,
            elapsed_time_hours=self._sim.elapsed_time / 3600.0,
            gamma_t=gamma_t,
            is_terminal=is_terminal,
        )

    @property
    def _site_of_vessel(self) -> Dict[str, str]:
        return {
            vessel_name: self._sim.get_vessel_site(vessel_name)
            for vessel_name in self._vessel_names
        }

    def _sim_object_names_to_id(self, names: List[str]) -> Dict[str, int]:
        names_to_id = {}
        for i, name in enumerate(names):
            names_to_id[name] = i

        return names_to_id

    def _get_info(self) -> Dict[str, Any]:
        """Returns extra information which complement the observation"""

        goal_summary = self._sim.get_goal_summary()
        goal_info = self._sim.get_all_goals_info()

        info = {
            "env_step": self._step,
            "sim_step": self._sim.sim_step,
            "elapsed_time_hours": self._sim.elapsed_time / 3600.0,
            "delta_time_hours": 0.0,
            "busy_vessels": self._sim.busy_vessels,
            "idle_vessels": self._sim.idle_vessels,
            "goals_total": goal_summary["total"],
            "goals_completed": goal_summary["completed"],
            "goals_failed": goal_summary["failed"],
            "goals_remaining": goal_summary["remaining"],
            "goal_cum_reward": self._sim.get_last_goal_reward(),
            "goal_info": goal_info,
        }

        # Per-component accumulated costs as top-level keys for PFM objectives.
        # Convention: "cost/{short_name}" where short_name strips the "_cost"
        # suffix from the component name (e.g. "storage_cost" → "cost/storage").
        for comp_name, acc in self._rewarder.accumulated_costs.items():
            short = comp_name.removesuffix("_cost")
            info[f"cost/{short}"] = float(acc)
        info["cost/total"] = float(self._rewarder.total_accumulated_cost)

        return info

    def _get_obs(self) -> Dict[str, Any]:
        """Get the observation associated with the current state"""

        observation = self._build_observation()
        return observation.to_dict()

    def _build_observation(self) -> Observation:
        vessel_obs = []
        for vessel_name in self._vessel_names:
            vessel_id = self._vessel_name_to_id[vessel_name]

            site_name = self._sim.get_vessel_site(vessel_name)
            site_id = self._site_name_to_id[site_name]

            vessel_status = VesselStatus.from_bool(
                self._sim.is_vessel_busy(vessel_name)
            )

            # Extract visit modes per resource natively
            vessel = self._sim._vessels_by_name[vessel_name]
            v_visit_modes = getattr(vessel, "visit_modes", {})
            visit_modes = {
                r: np.array(
                    [int(v_visit_modes.get(r, VisitMode.NEUTRAL))], dtype=np.int64
                )
                for r in self._resource_names
            }

            # Get inventory from simulator and convert to ResourceBundle
            inventory_dict = self._sim.get_vessel_inventory(vessel_name)
            inventory = ResourceBundle.from_inventory_dict(inventory_dict)

            # Build intent state from pending activities
            intent = self._build_vessel_intent(vessel_name)

            vessel_obs.append(
                VesselObs(
                    id=vessel_id,
                    position=site_id,
                    status=vessel_status,
                    visit_modes=visit_modes,
                    inventory=inventory,
                    intent=intent,
                )
            )

        site_obs = []
        for site_name in self._site_names:
            site_id = self._site_name_to_id[site_name]

            # Get inventory from simulator (with fabrication timing) and convert to ResourceBundle
            stock_dict = self._sim.get_site_inventory_extended(site_name)
            stock = ResourceBundle.from_inventory_dict(stock_dict)

            # Resolve site role from config
            site_cfg = self.cfg.sim.sites.get(site_name) if self.cfg.sim else None
            site_role = SITE_ROLE_TO_INT.get(
                site_cfg.role if site_cfg else "staging", 1
            )

            site_obs.append(
                SiteObs(
                    id=site_id,
                    role=site_role,
                    stock=stock,
                )
            )

        # Build goal observations
        goal_obs = []
        goal_states = self._sim.get_goal_states()
        for idx, gs in enumerate(goal_states):
            target_site_id = self._site_name_to_id.get(gs["location"], 0)
            resource_type_id = self._resource_names_to_id.get(gs["resource_type"], 0)
            num_goals = self._num_goals

            # Populate blocked_by multi-hot vector from depends_on indices
            blocked_by = np.zeros(max(num_goals, 1), dtype=np.int64)
            for dep_idx in gs.get("depends_on", []):
                if 0 <= dep_idx < num_goals:
                    blocked_by[dep_idx] = 1

            goal_obs.append(
                GoalObs(
                    id=idx,
                    target_site=target_site_id,
                    resource_type=resource_type_id,
                    required=gs["quantity"],
                    installed=gs["installed"],
                    remaining=gs["remaining"],
                    progress=gs["progress"],
                    deadline=gs["deadline_remaining_hours"],
                    is_failed=int(gs["failed"]),
                    is_blocked=int(gs.get("is_blocked", False)),
                    dep_progress=float(gs.get("dep_progress", 1.0)),
                    blocked_by=blocked_by,
                )
            )

        # Install-window awareness
        in_window = self._sim.is_in_no_install_window()
        raw_flip_time = self._sim.time_until_install_window_flip()
        # Cap inf at the remaining time budget so it stays finite for the agent
        # ! This might cause issues if the simulation time exceeds the time budget, but this should not happen
        remaining_budget_hours = max(
            0.0, self._time_budget_hours - self._sim.elapsed_time / 3600.0
        )
        flip_time = min(raw_flip_time / 3600.0, remaining_budget_hours)

        acc_costs = self._rewarder.accumulated_costs  # Dict[str, float]

        global_obs = GlobalObs(
            current_time=self._sim.des_env.now,
            # Invariant: each busy vessel has at most one active task at any moment
            num_pending_tasks=len(self._sim.busy_vessels),
            step=self._sim.sim_step,
            in_no_install_window=int(in_window),
            time_to_install_window_flip=flip_time,
            accumulated_travel_cost=acc_costs.get("travel_cost", 0.0),
            accumulated_storage_cost=acc_costs.get("storage_cost", 0.0),
        )

        observation = Observation(
            vessels=vessel_obs,
            sites=site_obs,
            goals=goal_obs,
            global_obs=global_obs,
        )
        self._last_obs_structured = observation
        return observation

    def get_observation(self) -> Observation:
        """Return a structured Observation object (useful for debugging)."""
        return self._build_observation()

    def get_name_mappings(self) -> Dict[str, List[str]]:
        """Expose ordered name mappings for external action/obs formatting."""
        return {
            "vessels": list(self._vessel_names),
            "sites": list(self._site_names),
            "resources": list(self._resource_names),
        }

    def describe_action_request(
        self, action: Action | List[Action]
    ) -> List[Dict[str, str]]:
        """Return per-vessel action descriptions with current vessel locations."""
        obs = self._last_obs_structured or self._build_observation()
        vessel_locations = {v.id: v.position for v in obs.vessels}

        actions = action if isinstance(action, list) else [action]
        actions_by_vessel: Dict[int, Action] = {a.vessel_id: a for a in actions}

        rows: List[Dict[str, str]] = []
        for vessel_id, vessel_name in enumerate(self._vessel_names):
            site_id = vessel_locations.get(vessel_id, -1)
            site_name = (
                self._site_names[site_id]
                if 0 <= site_id < len(self._site_names)
                else f"site_{site_id}"
            )

            act = actions_by_vessel.get(vessel_id)
            if act is None:
                action_text = "NOOP"
            else:
                action_text = act.compact(
                    vessel_names=self._vessel_names,
                    site_names=self._site_names,
                    resource_names=self._resource_names,
                    vessel_to_site=self._site_of_vessel,
                )

            vessel_status = "Busy" if vessel_name in self._sim.busy_vessels else "Idle"

            # Include the short activity ID and dependency IDs only for
            # vessels that acted this step (i.e. have an action in the
            # joint_actions array).  Busy vessels show NOOP and should
            # not display stale info from a previous registration.
            if act is not None:
                activity_id = self._last_activity_id_per_vessel.get(vessel_name) or ""
                dep_ids = self._last_dep_ids_per_vessel.get(vessel_name) or []
                depends_on = ", ".join(dep_ids) if dep_ids else ""
            else:
                activity_id = ""
                depends_on = ""

            rows.append(
                {
                    "vessel": vessel_name,
                    "vessel_location": site_name,
                    "vessel_status": vessel_status,
                    "action": action_text,
                    "activity_id": activity_id,
                    "depends_on": depends_on,
                }
            )

        return rows

    def inventory_snapshot(self) -> dict | None:
        """Return a snapshot of the current site inventory state.

        This is designed to be called via ``envs.call("inventory_snapshot")``
        so that the runner can capture inventory without direct sub-env
        access.

        Returns
        -------
        dict | None
            A flat dict with ``"time"``, ``"window_closed"``, and one key
            per ``"{site} - {resource}"`` holding the current load level.
            Returns ``None`` if the snapshot cannot be built.
        """
        try:
            record: dict = {
                "time": self._sim.elapsed_time,
                "window_closed": self._sim.is_in_no_install_window(),
            }
            for site in self._sim.get_site_names():
                inv = self._sim.get_site_inventory(site)
                for res, data in inv.items():
                    record[f"{site} - {res}"] = data["load"]
            return record
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Intent helpers
    # ------------------------------------------------------------------

    def _build_vessel_intent(self, vessel_name: str) -> IntentState:
        """Extract the most recent pending intent for *vessel_name*.

        Queries the activity tracker for PENDING activities associated
        with the vessel, picks the most recently registered one (highest
        ``registered_step``), and maps its DES category / attributes
        back to the observation-level :class:`IntentState`.

        Returns a default (all-sentinel) ``IntentState`` when no pending
        activity exists.
        """
        pending = self._sim.get_vessel_pending_activities(vessel_name)
        if not pending:
            return IntentState()

        # Pick the most recently registered pending activity
        latest = max(pending, key=lambda s: s.registered_step)
        activity = latest.activity
        category = getattr(activity, "category", None)

        # Map DES category → ActionType ordinal
        _CATEGORY_TO_ACTION: Dict[str, int] = {
            "transit": int(ActionType.MOVE),
            "loading": int(ActionType.LOAD),
            "unloading": int(ActionType.UNLOAD),
            "idle": int(ActionType.IDLE),
        }
        action_type = (
            _CATEGORY_TO_ACTION.get(category, INTENT_NONE_ACTION)
            if category
            else INTENT_NONE_ACTION
        )

        # Destination site ID (for MOVE: the move target; for LOAD/UNLOAD:
        # the partner site).
        dest_id = INTENT_NONE_DESTINATION
        if action_type == int(ActionType.MOVE):
            dest_site = latest.sites.get("destination")
            if dest_site is not None and dest_site.name in self._site_name_to_id:
                dest_id = self._site_name_to_id[dest_site.name]
        elif action_type in (int(ActionType.LOAD), int(ActionType.UNLOAD)):
            # For load/unload the "origin" or "destination" site is the
            # partner.  Use whichever is *not* the vessel itself.
            for role in ("origin", "destination"):
                partner_site = latest.sites.get(role)
                if (
                    partner_site is not None
                    and partner_site.name in self._site_name_to_id
                ):
                    dest_id = self._site_name_to_id[partner_site.name]
                    break

        # Resource ID (meaningful for LOAD/UNLOAD)
        resource_id = INTENT_NONE_RESOURCE
        res_name = getattr(activity, "id_", None)
        if res_name and res_name in self._resource_names_to_id:
            resource_id = self._resource_names_to_id[res_name]

        return IntentState(
            action_type=action_type,
            destination=dest_id,
            resource_id=resource_id,
        )

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        # Seed self.np_random to use as our randomness generator
        super().reset(seed=seed)

        # Reinitialize the simulator.  Pass the per-episode seeded RNG so
        # that any stochastic activity durations (when enabled in
        # cfg.sim.activities.stochasticity) are reproducible/replayable.
        self._sim = SimpleMonopileTransportSim(self.cfg.sim, rng=self.np_random)
        self._episode_start_time_unix = self._sim.des_env.now
        self._step = 0
        self._micro_step = 0
        self.gamma_t = 1
        self._terminated = False
        self._truncated = False
        self._last_activity_id_per_vessel = {name: None for name in self._vessel_names}
        self._last_dep_ids_per_vessel = {name: [] for name in self._vessel_names}

        structured = self._build_observation()
        observation = structured.to_dict()

        self._rewarder.reset()

        # Re-create milestone tracker so it picks up fresh sim state
        self._milestone_tracker = self._build_milestone_tracker()
        site_inv = self._get_site_inventories(structured)
        vessel_inv = self._get_vessel_inventories(structured)
        vessel_sites = self._site_of_vessel
        ms_ctx = self._build_milestone_context(
            site_inventories=site_inv,
            vessel_inventories=vessel_inv,
            vessel_sites=vessel_sites,
            gamma_t=self.gamma_t,
        )
        self._milestone_tracker.reset(ms_ctx)

        info = self._get_info()
        self._last_info = info

        return observation, info

    def micro_step(self, action: Action | List[Action]) -> Dict[str, Any]:
        """Register action(s) as PENDING intents without advancing physical time.

        This is a **micro-step** — it queues an action in the DES but does
        not advance the simulation clock.  The observation is rebuilt so
        that subsequent vessels in the AEC ordering can see the intents
        registered by earlier vessels.

        Because no physical time passes, the reward is always ``0.0`` and
        ``terminated``/``truncated`` remain unchanged.

        This method is called directly by the runner (bypassing Gymnasium
        wrappers like ``RecordEpisodeStatistics``) so that per-vessel
        intent registrations are invisible to the wrapper stack.

        Returns
        -------
        Dict[str, Any]
            The raw observation dict reflecting the newly registered
            intents.  Observation wrappers (hygiene, structured) are
            applied by the outermost action wrapper before the runner
            sees the result.

        See Also
        --------
        step : Advances the DES clock (the "real" Gymnasium step).
        """

        if self._terminated or self._truncated:
            raise RuntimeError(
                "Cannot call micro_step() on a terminated or truncated "
                "environment. You must call reset() first."
            )

        actions = action if isinstance(action, list) else [action]

        if actions:
            for act in actions:
                logger.debug(
                    f"Micro-Step {self._micro_step} [Registration]: {act.pretty(self._vessel_names, self._site_names, self._resource_names, vessel_to_site=self._site_of_vessel)}"
                )
                action_spec = act.to_action_spec(
                    vessel_names=self._vessel_names,
                    site_names=self._site_names,
                    resource_names=self._resource_names,
                )
                activity = self._sim.register_action_from_spec(action_spec)

                # Stash the short ID and dependency IDs so
                # describe_action_request can include them in the trace.
                vessel_name = self._vessel_names[act.vessel_id]
                builder = self._sim._activity_builder
                short_id = builder.get_short_id(activity.name)
                self._last_activity_id_per_vessel[vessel_name] = short_id
                self._last_dep_ids_per_vessel[vessel_name] = builder.last_dep_ids

        # Increment logical decision step
        self._micro_step += 1

        # Rebuild and return the observation so downstream consumers
        # (the next vessel in the AEC ordering) can see the registered
        # intents in the state representation.
        return self._get_obs()

    def step(
        self,
        action=None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advance the DES clock.

        This is the canonical Gymnasium ``step()`` — it runs the DES event
        loop, computes the real reward (goal progress, costs, milestones),
        and returns the updated observation.

        The ``action`` parameter is accepted for Gymnasium API compatibility
        but is **ignored**.  All intent registration happens via
        :meth:`micro_step` *before* this method is called.

        Gymnasium wrappers (``RecordEpisodeStatistics``,
        ``NormalizeReward``, ``RecordVideo``, etc.) intercept this method,
        so they see the true reward and termination signals.

        See Also
        --------
        micro_step : Registers a vessel intent (no time advance).
        """

        if self._terminated or self._truncated:
            return (
                self._last_obs_structured.to_dict(),
                0.0,
                self._terminated,
                self._truncated,
                self._last_info,
            )

        if action is not None:
            self.micro_step(action)

        prev_structured = self._last_obs_structured or self._build_observation()
        prev_time = float(self._sim.des_env.now)

        # Advance simulation one internal event
        sim_term = self._sim.step()
        self._step += 1

        # Compute physical time passed (convert to hours at boundary)
        curr_time = float(self._sim.des_env.now)
        delta_time_hours = (curr_time - prev_time) / 3600.0

        self.gamma_t = np.exp(-self.cfg.beta * delta_time_hours)

        # We finished the job or failed
        _term_sim = sim_term
        _term_completed = self._sim.is_completed
        _term_failed = self._sim.is_failed
        _term_time = self._sim.elapsed_time / 3600.0 >= self._time_budget_hours
        _term_steps = self._step >= self.cfg.max_episode_steps
        _term_cost = self._rewarder.total_accumulated_cost >= self._worst_case_cost
        if (
            _term_sim
            or _term_completed
            or _term_failed
            or _term_time
            or _term_steps
            or _term_cost
        ):
            self._terminated = True
            logger.debug(
                f"TERMINATED at step {self._step}: "
                f"sim_term={_term_sim}, completed={_term_completed}, "
                f"failed={_term_failed}, time_exceeded={_term_time}, "
                f"max_steps={_term_steps}, cost_exceeded={_term_cost} "
                f"(cost={self._rewarder.total_accumulated_cost:.4f}, "
                f"worst_case={self._worst_case_cost:.4f})"
            )

        # Get observation at the new point in time
        curr_structured = self._build_observation()
        observation = curr_structured.to_dict()

        # Build CostContext for modular cost components
        cost_context = self._build_cost_context(
            prev_structured, curr_structured, delta_time_hours
        )

        # Compute milestone shaping
        site_inv = self._get_site_inventories(curr_structured)
        vessel_inv = self._get_vessel_inventories(curr_structured)
        vessel_sites = self._site_of_vessel
        # Resolve terminal-zeroing policy (see MilestoneConfig.terminal_zeroing).
        # "failure_only" keeps Φ on a successful completion so the final
        # goal-completing step earns a bounded + bonus instead of a −Φ_prev
        # cliff (removes the completion-avoidance / 11-of-12 stall).
        zeroing_mode = getattr(self.cfg.reward.milestone, "terminal_zeroing", "always")
        if zeroing_mode == "never":
            pass_terminal = False
        elif zeroing_mode == "failure_only":
            pass_terminal = self._terminated and not self._sim.is_completed
        else:  # "always" (default / strict DPBRS)
            if zeroing_mode != "always":
                logger.warning(
                    f"Unknown terminal_zeroing={zeroing_mode!r}; using 'always'."
                )
            pass_terminal = self._terminated
        ms_ctx = self._build_milestone_context(
            site_inventories=site_inv,
            vessel_inventories=vessel_inv,
            vessel_sites=vessel_sites,
            gamma_t=self.gamma_t,
            is_terminal=pass_terminal,
        )
        milestone_result = self._milestone_tracker.step(ms_ctx)

        metrics = RewardMetrics(
            terminated=bool(self._terminated),
            truncated=bool(self._truncated),
            success=bool(self._sim.is_completed),
            cost_context=cost_context,
            milestone_result=milestone_result,
        )

        # Calculate real step reward
        reward, reward_components = self._rewarder.step(metrics)

        # Get diagnostic info
        info = self._get_info()
        info.update(
            {
                "delta_time_hours": float(delta_time_hours),
                "reward_components": reward_components,
                "is_success": bool(self._sim.is_completed),
            }
        )
        self._last_info = info

        return observation, reward, self._terminated, self._truncated, info

    def _build_cost_context(
        self,
        prev_obs: "Observation",
        curr_obs: "Observation",
        delta_time_hours: float,
    ) -> CostContext:
        """Assemble a :class:`CostContext` from pre/post-step observations.

        Parameters
        ----------
        prev_obs : Observation
            Structured observation *before* the step.
        curr_obs : Observation
            Structured observation *after* the step.
        delta_time_hours : float
            Simulation hours that elapsed during the step.
        """
        # Vessel positions (name → site name) and movement flags
        vessel_positions: Dict[str, str] = {}
        vessel_moved: Dict[str, bool] = {}
        prev_pos_by_id = {v.id: v.position for v in prev_obs.vessels}

        for v_obs in curr_obs.vessels:
            vessel_name = self._vessel_names[v_obs.id]
            site_name = self._site_names[int(v_obs.position)]
            vessel_positions[vessel_name] = site_name

            prev_pos = prev_pos_by_id.get(v_obs.id, v_obs.position)
            vessel_moved[vessel_name] = int(prev_pos) != int(v_obs.position)

        # --- Activity-based transit detection ----------------------------
        # Use the ActivityTracker to determine which vessels are currently
        # in transit and which just completed a transit this step.  This
        # replaces the old position-change heuristic that systematically
        # undercounted travel because the DES only updates vessel geometry
        # atomically on move-completion.
        tracker = self._sim.activity_tracker
        vessel_in_transit: Dict[str, bool] = {
            name: False for name in self._vessel_names
        }
        just_completed_transit_durations: Dict[str, float] = {}

        # Vessels with an ACTIVE transit activity (still mid-move).
        for act_state in tracker.get_active_activities():
            if getattr(act_state.activity, "category", None) == "transit":
                for vessel in act_state.vessels.values():
                    if vessel.name in vessel_in_transit:
                        vessel_in_transit[vessel.name] = True

        # Vessels whose transit activity just completed this step.
        for act_state in tracker.get_just_completed_activities():
            if getattr(act_state.activity, "category", None) == "transit":
                duration_s = getattr(act_state.activity, "duration", 0.0) or 0.0
                duration_hours = float(duration_s) / 3600.0
                for vessel in act_state.vessels.values():
                    if vessel.name in vessel_in_transit:
                        just_completed_transit_durations[vessel.name] = (
                            just_completed_transit_durations.get(vessel.name, 0.0)
                            + duration_hours
                        )

        # Site inventories – both pre and post this step's DES event.
        # The prev snapshot is used by the storage cost component for
        # correct left-Riemann sum integration.
        prev_site_inventories = self._get_site_inventories(prev_obs)
        site_inventories = self._get_site_inventories(curr_obs)

        # Vessel inventories (vessel_name → {resource_name: level})
        vessel_inventories = self._get_vessel_inventories(curr_obs)

        return CostContext(
            delta_time_hours=delta_time_hours,
            vessel_positions=vessel_positions,
            vessel_moved=vessel_moved,
            vessel_in_transit=vessel_in_transit,
            just_completed_transit_durations=just_completed_transit_durations,
            prev_site_inventories=prev_site_inventories,
            site_inventories=site_inventories,
            vessel_inventories=vessel_inventories,
        )

    def _get_site_inventories(self, obs: Observation) -> Dict[str, Dict[str, int]]:
        site_inventories: Dict[str, Dict[str, int]] = {}
        for s_obs in obs.sites:
            site_name = self._site_names[s_obs.id]
            inv_dict = self._sim.get_site_inventory(site_name)
            site_inventories[site_name] = {
                res: info["load"] for res, info in inv_dict.items()
            }
        return site_inventories

    def _get_vessel_inventories(self, obs: Observation) -> Dict[str, Dict[str, int]]:
        vessel_inventories: Dict[str, Dict[str, int]] = {}
        for v_obs in obs.vessels:
            vessel_name = self._vessel_names[v_obs.id]
            inv_dict = self._sim.get_vessel_inventory(vessel_name)
            vessel_inventories[vessel_name] = {
                res: info["load"] for res, info in inv_dict.items()
            }
        return vessel_inventories

    def render(self):
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from eos.envs.simple_monopile_transport.rendering import SMTDebugRenderer

            site_locations = (
                {name: cfg.location for name, cfg in self.cfg.sim.sites.items()}
                if self.cfg.sim is not None
                else {}
            )
            self._renderer = SMTDebugRenderer(
                cfg=self.cfg.render,
                site_locations=site_locations,
                vessel_names=self._vessel_names,
                site_names=self._site_names,
                resource_names=self._resource_names,
                render_mode=self.render_mode,
            )

        obs = self._last_obs_structured or self._build_observation()
        return self._renderer.draw(obs, self._last_info)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
        return super().close()
