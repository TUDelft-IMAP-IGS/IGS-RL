from dataclasses import dataclass
from typing import Dict, List, Set

import des_package.core as des_core
import des_package.model as des_model
from loguru import logger

from .types import (
    InstallationAsset,
    Site,
    TransportProcessingResource,
    Vessel,
)


class ActivityTracker:
    """Tracks activities and their associated vessels and sites.

    This class maintains state of all activities in the simulation, providing
    efficient lookup of active activities and their associated simulation objects.

    Parameters
    ----------
    remove_completed : bool
        Whether to automatically remove completed activities from tracking.
        Default is False to maintain history.
    """

    @dataclass
    class ActivityState:
        """State information for a tracked activity.

        Parameters
        ----------
        name : str
            Unique name of the activity.
        activity : des_model.GenericActivity
            Reference to the DES activity.
        state : des_core.CurrentActivityState
            Current state of the activity (PENDING, ACTIVE, PROCESSED, etc.).
        vessels : Dict[str, Vessel]
            Vessels associated with this activity (mover, processor, etc.).
        sites : Dict[str, Site]
            Sites associated with this activity (origin, destination, etc.).
        registered_step : int
            Simulation step when activity was registered.
        start_step : int | None
            Simulation step when activity became active.
        completion_step : int | None
            Simulation step when activity was completed.
        """

        name: str
        activity: des_model.GenericActivity
        state: des_core.CurrentActivityState
        vessels: Dict[str, Vessel]
        sites: Dict[str, Site]
        registered_step: int
        start_step: int | None = None
        completion_step: int | None = None

        def __hash__(self):
            return hash(self.name)

        def __eq__(self, other):
            if not isinstance(other, ActivityTracker.ActivityState):
                return NotImplemented
            return self.name == other.name

    def __init__(self, remove_completed: bool = False):
        self._activity_states: Dict[str, ActivityTracker.ActivityState] = {}
        self._remove_completed = remove_completed

        self._just_started: Set[ActivityTracker.ActivityState] = set()
        self._just_completed: Set[ActivityTracker.ActivityState] = set()

    def register_activity(
        self,
        activity: des_model.GenericActivity,
        current_step: int,
    ) -> None:
        """Register a new activity for tracking.

        Parameters
        ----------
        activity : des_model.GenericActivity
            The activity to track.
        current_step : int
            Current simulation step.
        """

        if activity.state != des_core.CurrentActivityState.PENDING:
            logger.error(
                f"Attempting to register activity which is not pending. step={current_step}, activity={activity.name}, activity_state={activity.state.value}"
            )

        vessels: Dict[str, Vessel] = {}
        sites: Dict[str, Site] = {}

        # Extract vessels and sites from activity attributes
        for attr_name in ("mover", "processor"):
            obj = getattr(activity, attr_name, None)
            if obj and isinstance(
                obj, (TransportProcessingResource, InstallationAsset)
            ):
                vessels[attr_name] = obj

        for attr_name in ("origin", "destination"):
            obj = getattr(activity, attr_name, None)
            if obj is None:
                continue
            if isinstance(obj, Site):
                sites[attr_name] = obj
            elif isinstance(obj, (TransportProcessingResource, InstallationAsset)):
                # Vessel-to-vessel transfers: origin/destination can be
                # vessels (e.g. transport unloading directly to an
                # installer).  Track them so dependency resolution and
                # busy-vessel queries include all real participants.
                if obj.name not in {v.name for v in vessels.values()}:
                    vessels[attr_name] = obj

        # Check additional_logs for more vessels/sites
        if hasattr(activity, "additional_logs"):
            for log_obj in activity.additional_logs:
                if isinstance(
                    log_obj, (TransportProcessingResource, InstallationAsset)
                ):
                    if log_obj.name not in vessels:
                        vessels[log_obj.name] = log_obj
                elif isinstance(log_obj, Site):
                    if log_obj.name not in sites:
                        sites[log_obj.name] = log_obj

        self._activity_states[activity.name] = ActivityTracker.ActivityState(
            name=activity.name,
            activity=activity,
            state=activity.state,
            vessels=vessels,
            sites=sites,
            registered_step=current_step,
        )

    def update_activities(self, current_step: int) -> None:
        """Update activity states based on current simulation state.

        Parameters
        ----------
        current_step : int
            Current simulation step.

        Notes
        -----
        This is intended to be called only once every environment step.
        """
        completed_activities = []
        self._just_completed = set()
        self._just_started = set()

        for activity_name, activity_state in self._activity_states.items():
            old_state = activity_state.state
            new_state = activity_state.activity.state

            # Update state
            activity_state.state = new_state

            # Track state transitions
            if old_state != new_state:
                if (
                    new_state == des_core.CurrentActivityState.ACTIVE
                    and activity_state.start_step is None
                ):
                    activity_state.start_step = current_step
                    self._just_started.add(activity_state)
                    logger.debug(
                        f"Activity '{activity_name}' became active at step {current_step}"
                    )

                elif new_state == des_core.CurrentActivityState.PROCESSED:
                    activity_state.completion_step = current_step
                    self._just_completed.add(activity_state)
                    logger.debug(
                        f"Activity '{activity_name}' completed at step {current_step}"
                    )
                    if self._remove_completed:
                        completed_activities.append(activity_name)

        # Remove completed activities if configured
        for activity_name in completed_activities:
            del self._activity_states[activity_name]

    def get_just_completed_activities(self) -> Set["ActivityTracker.ActivityState"]:
        """Get all activities which completed during the last step.

        Returns
        -------
        Set[ActivityState]
            Activities just completed.
        """
        return self._just_completed

    def get_just_started_activities(self) -> Set["ActivityTracker.ActivityState"]:
        """Get all activities which started during the last step.

        Returns
        -------
        Set[ActivityState]
            Activities just started.
        """
        return self._just_started

    def get_just_completed_vessels(self) -> Set[Vessel]:
        """Get all vessels which completed an activity during the last step.

        Returns
        -------
        Set[Vessel]
            Vessels that were part of a completed activity.
        """
        vessels = set()
        for activity_state in self._just_completed:
            vessels.update(activity_state.vessels.values())
        return vessels

    def get_just_started_vessels(self) -> Set[Vessel]:
        """Get all vessels which started an activity during the last step.

        Returns
        -------
        Set[Vessel]
            Vessels that were part of a started activity.
        """
        vessels = set()
        for activity_state in self._just_started:
            vessels.update(activity_state.vessels.values())
        return vessels

    def get_active_activities(self) -> List["ActivityTracker.ActivityState"]:
        """Get all currently active activities.

        Returns
        -------
        List[ActivityState]
            Activities in ACTIVE state.
        """
        return [
            state
            for state in self._activity_states.values()
            if state.state == des_core.CurrentActivityState.ACTIVE
        ]

    def get_pending_activities(self) -> List["ActivityTracker.ActivityState"]:
        """Get all pending activities.

        Returns
        -------
        List[ActivityState]
            Activities in PENDING state.
        """
        return [
            state
            for state in self._activity_states.values()
            if state.state == des_core.CurrentActivityState.PENDING
        ]

    def get_completed_activities(self) -> List["ActivityTracker.ActivityState"]:
        """Get all completed activities.

        Returns
        -------
        List[ActivityState]
            Activities in PROCESSED state.
        """
        return [
            state
            for state in self._activity_states.values()
            if state.state == des_core.CurrentActivityState.PROCESSED
        ]

    def get_vessels_in_active_activities(self) -> Set[Vessel]:
        """Get all vessels involved in active activities.

        Returns
        -------
        Set[Vessel]
            Set of vessels currently in active activities.
        """
        vessels = set()
        for state in self.get_active_activities():
            vessels.update(state.vessels.values())
        return vessels

    def get_sites_in_active_activities(self) -> Set[Site]:
        """Get all sites involved in active activities.

        Returns
        -------
        Set[str]
            Set of sites currently in active activities.
        """
        sites = set()
        for state in self.get_active_activities():
            sites.update(state.sites.values())
        return sites

    def get_busy_vessels(self) -> Set[Vessel]:
        """Get all busy vessels.

        Returns
        -------
        Set[Vessel]
            Set of busy vessels.

        Notes
        ---
        Currently the concept of busy means either in a Pending or Active activity
        """
        vessels = set()
        for state in self.get_active_activities() + self.get_pending_activities():
            vessels.update(state.vessels.values())
        return vessels

    def get_busy_sites(self) -> Set[Site]:
        """Get all busy sites.

        Returns
        -------
        Set[str]
            Set of busy sites.

        Notes
        ---
        Currently the concept of busy means either in a Pending or Active activity
        """
        sites = set()
        for state in self.get_active_activities() + self.get_pending_activities():
            sites.update(state.sites.values())
        return sites

    def get_activity_by_name(self, name: str) -> "ActivityTracker.ActivityState | None":
        """Get activity state by name.

        Parameters
        ----------
        name : str
            Activity name.

        Returns
        -------
        ActivityState | None
            Activity state if found, None otherwise.
        """
        return self._activity_states.get(name)

    def get_activities_for_vessel(
        self, vessel_name: str
    ) -> List["ActivityTracker.ActivityState"]:
        """Get all activities associated with a specific vessel.

        Parameters
        ----------
        vessel_name : str
            Name of the vessel.

        Returns
        -------
        List[ActivityState]
            Activities involving this vessel.
        """
        return [
            state
            for state in self._activity_states.values()
            if any(v.name == vessel_name for v in state.vessels.values())
        ]

    def get_activities_for_site(
        self, site_name: str
    ) -> List["ActivityTracker.ActivityState"]:
        """Get all activities associated with a specific site.

        Parameters
        ----------
        site_name : str
            Name of the site.

        Returns
        -------
        List[ActivityState]
            Activities involving this site.
        """
        return [
            state
            for state in self._activity_states.values()
            if any(s.name == site_name for s in state.sites.values())
        ]

    def get_summary(self) -> Dict[str, int]:
        """Get summary counts of activities by state.

        Returns
        -------
        Dict[str, int]
            Counts of activities in each state.
        """
        return {
            "total": len(self._activity_states),
            "pending": len(self.get_pending_activities()),
            "active": len(self.get_active_activities()),
            "completed": len(self.get_completed_activities()),
        }
