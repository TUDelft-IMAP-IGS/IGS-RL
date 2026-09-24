"""Discrete-event simulator for monopile transport operations.

This module implements a rule-based simulation environment for offshore wind
turbine monopile installation logistics, modeling vessels, sites, and their
interactions through event-driven activities.

Entity roles are driven entirely by configuration:

* **Site roles** (``SiteConfig.role``):
  - ``"source"`` – resources originate here; load-only, no unloading back.
  - ``"installation"`` – resources are permanently installed; only *installer*
    vessels may unload (install); nothing can be loaded from here.
  - ``"staging"`` – intermediate buffer; load and unload both permitted.

* **Vessel roles** (``VesselConfig.role``):
  - ``"heavy_lift"`` – long-haul bulk carrier (source → staging).
  - ``"feeder"`` – short-haul ferry (staging → installation).
  - ``"installer"`` – can install at installation sites; movement governed
    by the per-vessel ``movable`` flag.

  Both ``"heavy_lift"`` and ``"feeder"`` are transport-class roles: they
  can sail, load, unload, and transfer cargo but cannot install.

Adding new sites or vessels with appropriate roles in the YAML configs is
sufficient – no simulator code changes are required.

Fabrication Scheduling (Phase 2)
--------------------------------
When a ``ResourceTypeConfig`` contains a non-empty ``fabrication_schedule``,
the simulator spawns time-gated DES ``ShiftAmountActivity`` instances that
insert resources into source sites on schedule.  To satisfy the DES
requirement that ``ShiftAmountActivity`` has a co-located origin, a hidden
*phantom vessel* is created at each source site that participates in
fabrication.  These phantom vessels are fully encapsulated: they do not
appear in observations, action masks, vessel name lists, or training data.
"""

import datetime
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Set, Tuple
from zoneinfo import ZoneInfo

import des_package.core as des_core
import des_package.model as des_model
import numpy as np
import shapely.geometry.point
from loguru import logger

from eos.config import (
    GoalConfig,
    SimConfig,
)
from eos.envs.simple_monopile_transport.activity_tracker import ActivityTracker

from .activity_builder import ActionSpec, ActivityBuilder, SequentialActionSpec
from .goal_tracker import GoalTracker
from .reservation_system import ReservationSystem
from .types import (
    ActionType,
    InstallationAsset,
    Site,
    SiteRole,
    TransportProcessingResource,
    Vessel,
    VesselRole,
    VisitMode,
)
from .utils import log_sim_objects

# Note on DES Activity Timers:
# Retrieving the exact expected completion time of an activity dynamically
# is constrained by DES event-queue resolution; activities compute active/pending
# durations upon event triggering or schedule milestones.


@dataclass
class ActionRule:
    """Rule for generating activities based on current simulation state.

    Parameters
    ----------
    name : str
        Descriptive name for the rule.
    predicate : Callable
        Function that determines if the rule applies given vessel and site state.
    build : Callable
        Function that generates ActionSpecs when the rule applies.
    """

    name: str
    predicate: Callable[[Vessel, Site, str], bool]
    build: Callable[[Vessel, Site, str], List[ActionSpec]]


@dataclass
class _RuleHelpers:
    """Bundle of shared helper closures passed between rule-building methods.

    These closures capture the current reservation context and vessel/site
    state so that both business-rule and physical-rule builders can use the
    same capacity and spec-construction logic without duplication.
    """

    move_spec: Callable
    load_spec: Callable
    unload_spec: Callable
    is_object_full: Callable
    is_object_empty: Callable
    can_object_add_resource: Callable
    can_object_give_resource: Callable
    are_vessels_at_same_site: Callable


class SimpleMonopileTransportSim:
    """Discrete-event simulation for monopile transport logistics.

    This simulator models the workflow of transporting monopiles from fabrication
    yards (optionally) through marshalling yards to installation sites, using transport vessels
    and installation assets.

    The simulation uses a rule-based activity generation system where vessel
    actions are determined by the current state (location, cargo level, site
    resources) through composable ActionRule objects.

    Parameters
    ----------
    simulation_start : datetime.datetime, optional
        Start time of the simulation, by default datetime.datetime(2025, 5, 23).

    Attributes
    ----------
    des_env : des_core.Environment
        The underlying DES simulation environment.
    registry : Dict[str, Any]
        Central registry containing all simulation objects and activities.
    sim_step : int
        Current simulation step counter.
    vessels : List[str]
        Names of all vessels in the simulation.
    busy_vessels : List[Vessel]
        Vessels currently engaged in unfinished activities.
    activities_per_vessel : Dict[str, List[des_model.GenericActivity]]
        Possible activities for each vessel at initialization.

    Notes
    -----
    The simulation maintains the following invariants:
    - All vessels and sites are registered in `registry["sim_objects"]`
    - Activities are tracked in `registry["activities"]`
    - Container levels are modified only through put/get operations
    """

    def __init__(
        self,
        cfg: SimConfig,
        simulation_start: datetime.datetime = datetime.datetime(
            2025, 5, 23, tzinfo=ZoneInfo("UTC")
        ),
        rng: "np.random.Generator | None" = None,
    ) -> None:
        """Initialize the simulation environment."""
        # Core simulation infrastructure
        self.registry: Dict[str, Any] = {}
        self.simulation_start: datetime.datetime = simulation_start
        self.des_env: des_core.Environment = des_core.create_environment(
            initial_time=simulation_start, environment_type="standard"
        )
        self.des_env.registry = self.registry

        # Load configuration
        self.config: SimConfig = cfg

        # Build domain objects (sites, vessels)
        self._build_simulation_objects()

        # Phantom vessel names must be initialised before _build_vessel_lookup
        # because _get_all_vessels filters them out.
        self._phantom_vessel_names: Set[str] = set()

        self._vessels_by_name: Dict[str, Vessel] = self._build_vessel_lookup()
        self._sites_by_name: Dict[str, Site] = self._build_site_lookup()

        self._initialize_dummy_activities()

        # Initialize ActivityBuilder for creating activities from action specs
        self._activity_builder = ActivityBuilder(
            env=self.des_env,
            registry=self.registry,
            config=self.config.activities,
            no_install_windows=self.config.no_install_windows,
            rng=rng,
        )

        # Initialize goal tracking for tracking progress towards simulatior goals
        self._goal_cfgs: List[GoalConfig] = self.config.goals or []
        self._goal_tracker = GoalTracker(
            self._goal_cfgs,
            simulation_start=self.simulation_start,
            sites_by_name=self._sites_by_name,
            vessels_by_name=self._vessels_by_name,
        )

        # Initialize ActivityTracker to track activities and participating simulation objects
        self._activity_tracker = ActivityTracker(remove_completed=False)

        # Wire the tracker into the builder so it can auto-inject
        # start_event dependency conditions (intent-based queuing)
        self._activity_builder.set_activity_tracker(self._activity_tracker)

        # Wire the reservation system into the builder so it can perform
        # granular resource claiming in Phase B.
        # (Phase A: the reference is stored but not yet used by the builder.)

        # Simulation state
        self.sim_step: int = 0

        # Persistent incremental reservation system – seeded with vessel
        # physical locations at construction time.  Updated incrementally
        # on action registration (micro-step) and reverted on activity
        # completion (step).
        self._reservation_system = self._build_initial_reservations()

        # Now that the RS exists, wire it into the builder.
        self._activity_builder.set_reservation_system(self._reservation_system)

        # Map activity name → ActionSpec so we can revert the spec's
        # projected effects when the activity completes.
        self._spec_by_activity: Dict[str, ActionSpec] = {}

        # --- Fabrication scheduling (Phase 2) ---
        # Spawn time-gated DES activities for resources that appear on a
        # schedule.  Must happen after _activity_tracker and
        # _reservation_system are ready.
        self._spawn_fabrication_activities()

    @property
    def is_failed(self) -> bool:
        return self._goal_tracker.any_goal_failed()

    @property
    def is_completed(self) -> bool:
        return self._goal_tracker.all_goals_completed()

    @property
    def busy_vessels(self) -> List[str]:
        return [vessel.name for vessel in self._activity_tracker.get_busy_vessels()]

    @property
    def idle_vessels(self) -> List[str]:
        return [
            vessel.name
            for vessel in set(self._vessels_by_name.values())
            - self._activity_tracker.get_busy_vessels()
        ]

    @property
    def just_finished_vessels(self) -> List[str]:
        return [
            vessel.name
            for vessel in self._activity_tracker.get_just_completed_vessels()
            - self._activity_tracker.get_busy_vessels()
        ]

    @property
    def activity_tracker(self) -> ActivityTracker:
        """The :class:`ActivityTracker` managing all registered activities."""
        return self._activity_tracker

    @property
    def resource_names(self) -> List[str]:
        """Get the canonical list of resource types in the simulation.

        Returns
        -------
        List[str]
            Resource type names from the global config.
        """
        return list(self.config.resource_types.keys())

    @property
    def elapsed_time(self) -> float:
        """The elapsed time since simulation start in seconds"""
        return self.des_env.now - self.simulation_start.timestamp()

    # =========================================================================
    # Public API Methods
    # =========================================================================

    def step(self) -> bool:
        """Advance the simulation by one discrete event step.

        Returns
        -------
        bool
            True if terminal state reached, False otherwise.
        """
        if self._is_terminal():
            return True

        self.des_env.next_step()

        self.sim_step += 1

        # Update activity tracker with current state
        self._activity_tracker.update_activities(self.sim_step)

        # --- Revert completed activities from the incremental RS ----------
        for completed_state in self._activity_tracker.get_just_completed_activities():
            spec = self._spec_by_activity.pop(completed_state.name, None)
            if spec is not None:
                # Fabrication activities use a synthetic ActionSpec with
                # action_type=LOAD and vessel_name=phantom.  The standard
                # revert_action_spec would try to unreserve a drain on
                # the partner (source site) and a fill on the vessel
                # (phantom).  But we registered only a fill on the source
                # site, so we handle fabrication reverts specially.
                if self._is_phantom_vessel(spec.vessel_name):
                    # Only unreserve the fill we registered on the source.
                    if (
                        spec.partner_name is not None
                        and spec.resource_name is not None
                        and spec.amount is not None
                        and spec.amount > 0
                    ):
                        self._reservation_system._unreserve_fill(
                            spec.partner_name, spec.resource_name, spec.amount
                        )
                        # Also clean up granular delivery tracking.
                        self._reservation_system.complete_delivery(
                            spec.partner_name,
                            spec.resource_name,
                            completed_state.name,
                        )
                        # Ledger: fabrication delivered resource to site
                        self._reservation_system.ledger_apply_transfer(
                            None, spec.partner_name, spec.resource_name, spec.amount
                        )
                else:
                    self._reservation_system.revert_action_spec(
                        spec, activity_name=completed_state.name
                    )

                    # Reset the vessel's projected location to its current
                    # physical site (the DES has already moved it).
                    try:
                        vessel = self._vessels_by_name[spec.vessel_name]
                        physical_site = self._get_vessel_site(vessel).name
                        self._reservation_system.reset_projected_location(
                            spec.vessel_name, physical_site
                        )
                    except (KeyError, ValueError):
                        logger.warning(
                            f"Could not reset projected location for "
                            f"{spec.vessel_name} after activity completion"
                        )

        # --- Also revert activities that just started (PENDING → ACTIVE)
        # so we don't double-count with the DES's own in-progress state.
        # NOTE: we do NOT revert started activities.  A started-but-not-
        # completed activity is still consuming resources / projecting
        # location; the reservation must stay until completion.

        logger.debug(
            f"Step {self.sim_step}: Finished vessels: {self.just_finished_vessels}"
        )

        # Check goals based on container levels using POSIX timestamp
        self._goal_tracker.check_goals(self.des_env.now)

        # Log goal completion status
        completed_this_step = self._goal_tracker.get_completed_this_step()
        if completed_this_step:
            logger.debug(f"Goals completed this step: {len(completed_this_step)}")
            summary = self._goal_tracker.get_goal_summary()
            logger.debug(
                f"Goal progress: {summary['completed']}/{summary['total']} completed, "
                f"{summary['remaining']} remaining"
            )
        self._goal_tracker.clear_completed_this_step()

        return self._is_terminal()

    def set_activities(
        self, activity_per_vessel: Dict[str, des_model.GenericActivity]
    ) -> None:
        """Register new activities for specified vessels.

        Parameters
        ----------
        activity_per_vessel : Dict[str, des_model.GenericActivity]
            Mapping from vessel name to the activity to register.
        """
        for vessel_name, activity in activity_per_vessel.items():
            des_model.register_additional_processes(
                self.des_env,
                activity,
                simulation_object=self._vessels_by_name[vessel_name],
            )

            # Register activity with tracker
            self._activity_tracker.register_activity(activity, self.sim_step)

    def get_num_vessels(self) -> int:
        return len(self._vessels_by_name)

    def get_num_sites(self) -> int:
        return len(self.des_env.registry["sim_objects"]["Site"])

    def get_vessel_names(self) -> List[str]:
        """Get names of all vessels in the simulation.

        Returns
        -------
        List[str]
            List of vessel names.
        """
        all_vessels = self._get_all_vessels()
        return [vessel.name for vessel in all_vessels]

    def get_site_names(self) -> List[str]:
        """Get names of all sites in the simulation.

        Returns
        -------
        List[str]
            List of site names.
        """
        all_sites = self._get_all_sites()
        return [site.name for site in all_sites]

    def get_vessel_site(self, vessel: str) -> str:
        """Determine the current location of a vessel.

        Parameters
        ----------
        vessel : str
            The name of the vessel whose location to determine.

        Returns
        -------
        Site
            The name of the site where the vessel is currently located.

        Raises
        ------
        ValueError
            If the vessel's location cannot be determined from its geometry.
        """
        vessel = self._vessels_by_name[vessel]
        return self._get_vessel_site(vessel).name

    def is_vessel_busy(self, vessel: str) -> bool:
        """Check if a vessel is currently busy with an activity.

        Parameters
        ----------
        vessel : str
            The name of the vessel to query.

        Returns
        -------
        bool
            True if the vessel is currently executing an activity.
        """
        return vessel in self.busy_vessels

    def can_idle(self, vessel_name: str) -> bool:
        """Check whether a vessel can choose the await-event IDLE action.

        The IDLE action is available when at least one non-idle activity
        exists in the DES that is not solely owned by the requesting
        vessel.  This guarantees that the idle activity's OR start_event
        will eventually fire (no deadlock).

        Parameters
        ----------
        vessel_name : str
            Name of the vessel to check.

        Returns
        -------
        bool
            True if the vessel can idle (there is something to wait for).
        """
        unfinished = (
            self._activity_tracker.get_active_activities()
            + self._activity_tracker.get_pending_activities()
        )

        for act_state in unfinished:
            # Skip idle activities to prevent idle-on-idle deadlock
            if getattr(act_state.activity, "category", None) == "idle":
                continue

            # Skip activities owned solely by the requesting vessel
            vessel_names_in_activity = {v.name for v in act_state.vessels.values()}
            if vessel_names_in_activity == {vessel_name}:
                continue

            return True

        return False

    def get_vessel_activity_name(self, vessel: str) -> List[str]:
        """Get the name of the activity a vessel is currently executing.

        Parameters
        ----------
        vessel : str
            The name of the vessel to query.

        Returns
        -------
        List[str]
            The name of the activities the vessel is executing.
        """
        return [
            activity_state.name
            for activity_state in self._activity_tracker.get_activities_for_vessel(
                vessel
            )
        ]

    def get_vessel_pending_activities(
        self, vessel: str
    ) -> List["ActivityTracker.ActivityState"]:
        """Return PENDING activity states associated with *vessel*.

        These are activities that have been registered (queued) via
        :meth:`register_action_from_spec` but have not yet started
        executing in the DES.  During the AEC micro-stepping phase this
        is the set of *intents* that have been committed for the vessel.

        Parameters
        ----------
        vessel : str
            The name of the vessel to query.

        Returns
        -------
        List[ActivityTracker.ActivityState]
            Pending activity states involving this vessel, ordered by
            registration time (earliest first).
        """
        return [
            state
            for state in self._activity_tracker.get_pending_activities()
            if any(v.name == vessel for v in state.vessels.values())
        ]

    def get_possible_activities(self, vessel: str) -> List[des_model.GenericActivity]:
        vessel_obj = self._vessels_by_name[vessel]
        return self._get_possible_activities(vessel_obj)

    def get_goal_summary(self) -> Dict[str, int]:
        """Get summary counts of goals by status.

        Returns
        -------
        Dict[str, int]
            Dictionary containing:
            - total: total number of goals
            - pending: goals not yet started
            - in_progress: goals currently executing
            - completed: goals finished
            - remaining: goals not yet completed
        """
        return self._goal_tracker.get_goal_summary()

    def get_last_goal_reward(self) -> float:
        """Return the reward accrued from goal progress in the last step."""
        return self._goal_tracker.get_last_step_reward()

    def get_last_goal_progress(self) -> List[Dict[str, Any]]:
        """Return per-goal progress deltas from the last step."""
        return self._goal_tracker.get_last_step_progress()

    def get_remaining_goals_info(self) -> List[Dict[str, any]]:
        """Get detailed information about all remaining goals.

        This is useful for constructing observation spaces in RL environments.

        Returns
        -------
        List[Dict[str, any]]
            List of dictionaries containing goal information including:
            - name: goal name
            - status: current status (pending/in_progress/completed)
            - vessel_name: responsible vessel
            - deadline: deadline if set
            - start_time: when started (if applicable)
            - completion_time: when completed (if applicable)
        """
        return self._goal_tracker.get_goals_info()

    def get_all_goals_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about all goals in the simulation.

        Returns
        -------
        List[Dict[str, any]]
            List of dictionaries containing complete goal information.
        """
        return self._goal_tracker.get_goals_info()

    def get_goal_states(self) -> List[Dict[str, Any]]:
        """Get current state of all goals for observation construction.

        Returns
        -------
        List[Dict[str, Any]]
            One entry per goal with keys: ``location``, ``resource_type``,
            ``quantity``, ``installed``, ``remaining``, ``progress``,
            ``deadline_remaining_hours``, ``completed``, ``failed``,
            ``depends_on``, ``is_blocked``, ``dep_progress``.
        """
        elapsed_hours = self.elapsed_time / 3600.0
        results = []
        for i, goal_state in enumerate(self._goal_tracker.goals):
            cfg = goal_state.config
            installed = min(goal_state.last_level, cfg.quantity)
            remaining = max(0, cfg.quantity - installed)
            progress = installed / cfg.quantity if cfg.quantity > 0 else 1.0

            # Compute remaining deadline hours
            if cfg.deadline is not None:
                deadline_remaining = max(0.0, cfg.deadline - elapsed_hours)
            else:
                deadline_remaining = 0.0

            # Dependency state from GoalTracker
            dep_info = self._goal_tracker.get_goal_dependency_info(i)

            results.append(
                {
                    "location": cfg.location,
                    "resource_type": cfg.resource_type,
                    "quantity": cfg.quantity,
                    "installed": installed,
                    "remaining": remaining,
                    "progress": progress,
                    "deadline_remaining_hours": deadline_remaining,
                    "completed": goal_state.completed,
                    "failed": goal_state.failed,
                    "depends_on": dep_info["depends_on"],
                    "is_blocked": dep_info["is_blocked"],
                    "dep_progress": dep_info["dep_progress"],
                }
            )
        return results

    def is_in_no_install_window(self) -> bool:
        """Check if the current time is within a no-install window.

        Returns
        -------
        bool
            True if current time falls within any configured no-install window.
        """
        if not self.config.no_install_windows:
            return False

        elapsed_hours = self.elapsed_time / 3600.0

        for window in self.config.no_install_windows:
            start_hours, end_hours = window
            if start_hours <= elapsed_hours < end_hours:
                return True

        return False

    def time_until_install_window_flip(self) -> float:
        """Seconds until the install-window state next changes.

        If we are currently **inside** a no-install window, this returns the
        number of seconds until that window ends (i.e. installations become
        possible again).

        If we are currently **outside** any no-install window, this returns
        the number of seconds until the next window starts (i.e. installations
        become blocked).

        If no future state change exists (no windows configured, or all
        windows are in the past), returns ``float('inf')``.

        Returns
        -------
        float
            Seconds until the boolean returned by
            :meth:`is_in_no_install_window` would flip, or ``inf``.
        """
        if not self.config.no_install_windows:
            return float("inf")

        elapsed_hours = self.elapsed_time / 3600.0

        # Check if we are currently inside a window
        for start_hours, end_hours in self.config.no_install_windows:
            if start_hours <= elapsed_hours < end_hours:
                # Inside this window – flip happens when it ends
                return (end_hours - elapsed_hours) * 3600.0

        # Not inside any window – find the earliest future window start
        next_start_hours = float("inf")
        for start_hours, _end_hours in self.config.no_install_windows:
            if start_hours > elapsed_hours:
                next_start_hours = min(next_start_hours, start_hours)

        if next_start_hours < float("inf"):
            return (next_start_hours - elapsed_hours) * 3600.0

        return float("inf")

    def get_vessel_inventory(self, vessel_name: str) -> Dict[str, Dict[str, int]]:
        """Get vessel's current inventory.

        Returns entries for *every* resource type in the simulation so that
        the observation shape is uniform across all vessels.  Types the
        vessel doesn't carry get ``{load: 0, capacity: 0}``.
        """
        vessel = self._vessels_by_name[vessel_name]
        result = {}
        for rtype in self.resource_names:
            if rtype in vessel.container.container_list:
                result[rtype] = {
                    "load": vessel.container.get_level(rtype),
                    "capacity": vessel.container.get_capacity(rtype),
                }
            else:
                result[rtype] = {"load": 0, "capacity": 0}
        return result

    def get_site_inventory(self, site_name: str) -> Dict[str, Dict[str, int]]:
        """Get site's current inventory.

        Returns entries for *every* resource type in the simulation so that
        the observation shape is uniform across all sites.  Types the
        site doesn't handle get ``{load: 0, capacity: 0}``.
        """
        sites = self.des_env.registry["sim_objects"]["Site"]
        site = sites[site_name]
        result = {}
        for rtype in self.resource_names:
            if rtype in site.container.container_list:
                result[rtype] = {
                    "load": site.container.get_level(rtype),
                    "capacity": site.container.get_capacity(rtype),
                }
            else:
                result[rtype] = {"load": 0, "capacity": 0}
        return result

    def get_site_inventory_extended(self, site_name: str) -> Dict[str, Dict[str, Any]]:
        """Get site inventory with fabrication timing information.

        Like :meth:`get_site_inventory` but each entry also contains
        ``next_available_in`` — the number of hours until the next unit
        of that resource type is fabricated at this site.  For non-source
        sites or types without a fabrication schedule, this is ``0.0``.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            ``{resource_type: {"load": int, "capacity": int, "next_available_in": float}}``
        """
        base = self.get_site_inventory(site_name)
        elapsed_hours = self.elapsed_time / 3600.0
        is_source = self._is_source_site(site_name)

        for rtype in self.resource_names:
            next_avail = 0.0
            if is_source:
                rtype_cfg = self.config.resource_types.get(rtype)
                if rtype_cfg is not None:
                    schedule = rtype_cfg.fabrication_schedule.get(site_name)
                    if schedule:
                        # Find next future fabrication time
                        for t in schedule:
                            if t > elapsed_hours:
                                next_avail = t - elapsed_hours
                                break
            base[rtype]["next_available_in"] = next_avail

        return base

    def register_action_from_spec(self, spec: ActionSpec) -> des_model.GenericActivity:
        """Register an activity from an ActionSpec.

        Parameters
        ----------
        spec : ActionSpec
            Domain-neutral action specification from activity_builder module.
        """
        # Maintain ping-ponging state tracking immediately upon commitment
        vessel = self._vessels_by_name[spec.vessel_name]
        if not hasattr(vessel, "visit_modes"):
            vessel.visit_modes = {}

        if spec.action_type == ActionType.MOVE:
            vessel.visit_modes.clear()
        elif spec.action_type == ActionType.LOAD and spec.resource_name:
            vessel.visit_modes[spec.resource_name] = VisitMode.LOADING
        elif spec.action_type == ActionType.UNLOAD and spec.resource_name:
            vessel.visit_modes[spec.resource_name] = VisitMode.UNLOADING

        # Use ActivityBuilder to create the activity
        activity = self._activity_builder.build(spec)

        # Register it
        self.set_activities({spec.vessel_name: activity})

        # --- Update the incremental reservation system ---
        # Pass the activity name so the RS can track granular deliveries.
        self._reservation_system.apply_action_spec(spec, activity_name=activity.name)

        # Remember the spec so we can revert its effects on completion.
        self._spec_by_activity[activity.name] = spec

        return activity

    def register_action_from_sequential_spec(
        self, vessel_names: List[str], spec: SequentialActionSpec
    ) -> des_model.GenericActivity:
        # Use ActivityBuilder to create the activity
        activity = self._activity_builder.build(spec)

        # Register it
        for vessel_name in vessel_names:
            self.set_activities({vessel_name: activity})

        return activity

    # =========================================================================
    # State Query Methods
    # =========================================================================

    def _is_terminal(self) -> bool:
        """Check whether simulation has reached the terminal state.

        Returns
        -------
        bool
            True if all goals are completed or any goal has failed.

        Notes
        -----
        Terminal condition: All goals completed OR any goal failed (deadline missed).
        """
        return (
            self._goal_tracker.all_goals_completed()
            or self._goal_tracker.any_goal_failed()
        )

    def _get_vessel_site(self, vessel: Vessel) -> Site:
        """Determine the current location of a vessel.

        Parameters
        ----------
        vessel : Vessel
            The vessel whose location to determine.

        Returns
        -------
        Site
            The site where the vessel is currently located.

        Raises
        ------
        ValueError
            If the vessel's location cannot be determined from its geometry.
        """
        for site in self.des_env.registry["sim_objects"]["Site"].values():
            if hasattr(vessel, "geometry") and site.geometry == vessel.geometry:
                return site
        raise ValueError(f"Vessel {vessel.name} location unknown")

    def _get_all_vessels(self) -> List[Vessel]:
        """Retrieve all vessels from the registry.

        Phantom fabrication vessels are excluded — they are an internal
        implementation detail and must never appear in observations,
        action masks, or training data.

        Returns
        -------
        List[Vessel]
            Combined list of TransportProcessingResource and InstallationAsset objects.
        """
        vessels: List[Vessel] = []
        for v in (
            self.des_env.registry["sim_objects"]
            .get("TransportProcessingResource", {})
            .values()
        ):
            if v.name not in self._phantom_vessel_names:
                vessels.append(v)
        for v in (
            self.des_env.registry["sim_objects"].get("InstallationAsset", {}).values()
        ):
            if v.name not in self._phantom_vessel_names:
                vessels.append(v)
        return vessels

    def _get_all_sites(self) -> List[Site]:
        """Retrieve all sites from the registry.

        Returns
        -------
        List[Site]
            List of all Site objects.
        """
        return self.des_env.registry["sim_objects"]["Site"].values()

    # =========================================================================
    # Rule-Based Activity Generation
    # =========================================================================

    @property
    def reservations(self) -> ReservationSystem:
        """Return the persistent incremental :class:`ReservationSystem`.

        The reservation system is seeded at simulator construction with
        vessel physical locations, updated incrementally on each action
        registration (``register_action_from_spec``), and reverted when
        activities complete (inside ``step``).

        No full rebuild is performed on access – the returned object is
        the live, incrementally maintained instance.
        """
        return self._reservation_system

    def _build_initial_reservations(self) -> ReservationSystem:
        """Create and seed a :class:`ReservationSystem` with vessel locations.

        Called once during ``__init__`` to establish the baseline state.
        """
        current_locations = {
            v_name: self._get_vessel_site(self._vessels_by_name[v_name]).name
            for v_name in self._vessels_by_name
        }
        return ReservationSystem.from_simulation_state(
            self._activity_tracker,
            current_locations=current_locations,
        )

    def _rebuild_reservations_for_parity_check(self) -> ReservationSystem:
        """Build a fresh :class:`ReservationSystem` from scratch (full scan).

        Useful as a debugging parity check against the incremental RS.
        This is **not** called in the hot path.
        """
        current_locations = {
            v_name: self._get_vessel_site(self._vessels_by_name[v_name]).name
            for v_name in self._vessels_by_name
        }
        return ReservationSystem.from_simulation_state(
            self._activity_tracker,
            current_locations=current_locations,
        )

    def get_possible_actions(
        self,
        vessel_name: str,
        reservations: ReservationSystem | None = None,
    ) -> List[ActionSpec]:
        """Generate possible action specifications for a vessel based on current state.

        Uses a rule-based system where ActionRule objects define state predicates
        and action spec builders.  Resource availability and partner-blocking
        checks are delegated to the :class:`ReservationSystem` so that pending
        activities and already-chosen specs are properly accounted for.

        When ``cfg.use_business_rules`` is ``True`` (default), actions are
        constrained by domain-specific business logic (e.g. which vessel may
        visit which site, directional cargo flow).  When ``False``, only
        physical constraints are enforced (capacity, co-location, weather
        windows).

        Parameters
        ----------
        vessel_name : str
            The name of the vessel for which to generate actions.
        reservations : ReservationSystem | None
            Pre-built reservation context.  When ``None`` the simulator's
            persistent reservation system is used (updated incrementally,
            never rebuilt).

        Returns
        -------
        List[ActionSpec]
            All action specifications that are valid given the current vessel and site state.
        """

        # Use the cached reservation system when none explicitly provided
        if reservations is None:
            reservations = self.reservations

        vessel = self._vessels_by_name[vessel_name]
        vessel_site = self._get_vessel_site(vessel)

        # Convenience accessor for site destinations
        sites: Dict[str, Site] = self.des_env.registry["sim_objects"]["Site"]

        # Resources available in the simulation
        resource_names = self.resource_names

        # ---- Shared helper closures ----

        # --- Ensure the shadow ledger is seeded for this batch -----------
        # The ledger is maintained incrementally; we only need to seed it
        # once (at construction / reset).  If capacities are not yet set,
        # seed now from the live DES state.  This makes the first call
        # after construction or reset populate the baseline.
        if not reservations._capacities:
            _object_capacities: Dict[Tuple[str, str], int] = {}
            for sname, scfg in self.config.sites.items():
                for rname, rslot in scfg.resource_types.items():
                    _object_capacities[(sname, rname)] = rslot.capacity
            for vname, vcfg in self.config.vessels.items():
                for rname, rslot in vcfg.resource_types.items():
                    _object_capacities[(vname, rname)] = rslot.capacity

            _initial_levels: Dict[Tuple[str, str], int] = {}
            for site_name, site in self._sites_by_name.items():
                for rname in resource_names:
                    level = site.container.get_level(rname)
                    if level > 0:
                        _initial_levels[(site_name, rname)] = level
            for vessel_name, vessel_obj in self._vessels_by_name.items():
                for rname in resource_names:
                    level = vessel_obj.container.get_level(rname)
                    if level > 0:
                        _initial_levels[(vessel_name, rname)] = level

            reservations.seed_ledger(_initial_levels, _object_capacities)

        def move_spec(dest: Site) -> ActionSpec:
            return ActionSpec(
                action_type=ActionType.MOVE,
                vessel_name=vessel.name,
                destination_name=dest.name,
            )

        def load_spec(origin: Site, amount: int, resource_name: str) -> ActionSpec:
            return ActionSpec(
                action_type=ActionType.LOAD,
                vessel_name=vessel.name,
                partner_name=origin.name,
                resource_name=resource_name,
                amount=amount,
            )

        def unload_spec(
            dest: Site | Vessel, amount: int, resource_name: str
        ) -> ActionSpec:

            return ActionSpec(
                action_type=ActionType.UNLOAD,
                vessel_name=vessel.name,
                partner_name=dest.name,
                resource_name=resource_name,
                amount=amount,
                duration=None,
            )

        def get_total_content(obj: Vessel | Site, resource_names: List[str]) -> int:
            """Get the total amount of resources currently loaded on a simulation object."""
            total = 0
            for resource_name in resource_names:
                total += obj.container.get_level(resource_name)
            return total

        def get_total_capacity(obj: Vessel | Site) -> int:
            """Get the total capacity of resources a simulation object is able to carry."""
            return (
                self.config.sites[obj.name].total_capacity
                if obj.name in self.config.sites
                else self.config.vessels[obj.name].total_capacity
            )

        def get_effective_content(obj: Vessel | Site) -> int:
            """Projected total content after all committed actions resolve.

            Reads from the RS's shadow ledger (which tracks logical levels
            purely from RS events) and adjusts by aggregate drains/fills
            so that the result reflects the logical end-state of all
            registered intents.
            """
            compensated_total = sum(
                reservations._ledger_levels.get((obj.name, r), 0)
                for r in resource_names
            )
            total_drains = sum(
                reservations.get_total_drain(obj.name, r) for r in resource_names
            )
            total_fills = sum(
                reservations.get_total_fill(obj.name, r) for r in resource_names
            )
            return compensated_total - total_drains + total_fills

        def is_object_full(obj: Vessel | Site) -> bool:
            return not any(
                reservations.can_claim_capacity(obj.name, r) for r in resource_names
            )

        def is_object_empty(obj: Vessel | Site) -> bool:
            return not any(
                reservations.can_claim_resource(obj.name, r) for r in resource_names
            )

        def can_object_add_resource(obj: Vessel | Site, resource_name: str) -> bool:
            # Total-capacity guard: even if a per-resource slot is free,
            # the object may have reached its total capacity across all
            # resource types.
            if get_effective_content(obj) >= get_total_capacity(obj):
                return False
            if is_object_full(obj):
                return False
            return reservations.can_claim_capacity(obj.name, resource_name)

        def can_object_give_resource(obj: Vessel | Site, resource_name: str) -> bool:
            return reservations.can_claim_resource(obj.name, resource_name)

        def are_vessels_at_same_site(vessel_1: Vessel, vessel_2: Vessel) -> bool:
            loc_1 = reservations.get_projected_location(vessel_1.name)
            loc_2 = reservations.get_projected_location(vessel_2.name)

            # Fallback to physical geometry if not found (safeguard)
            if loc_1 is None or loc_2 is None:
                logger.error(
                    f"Location of vessels not found in reservation system v1={vessel_1}, v2={vessel_2}"
                )
                raise

            return loc_1 == loc_2

        # ---- Build the appropriate rule set ----
        helpers = _RuleHelpers(
            move_spec=move_spec,
            load_spec=load_spec,
            unload_spec=unload_spec,
            is_object_full=is_object_full,
            is_object_empty=is_object_empty,
            can_object_add_resource=can_object_add_resource,
            can_object_give_resource=can_object_give_resource,
            are_vessels_at_same_site=are_vessels_at_same_site,
        )

        if self.config.use_business_rules:
            rules = self._build_business_rules(
                vessel=vessel,
                sites=sites,
                reservations=reservations,
                helpers=helpers,
            )
        else:
            rules = self._build_physical_rules(
                vessel=vessel,
                vessel_site=vessel_site,
                sites=sites,
                reservations=reservations,
                helpers=helpers,
            )

        # Evaluate all rules and collect matching action specs
        action_specs: List[ActionSpec] = []
        for rule in rules:
            for resource_name in resource_names:
                if rule.predicate(vessel, vessel_site, resource_name):
                    action_specs.extend(rule.build(vessel, vessel_site, resource_name))

        # Remove duplicate specs (based on equality of dataclass)
        unique_specs: List[ActionSpec] = []
        for spec in action_specs:
            if spec not in unique_specs:
                unique_specs.append(spec)

        if self.config.use_business_rules:
            # --- Elegant Ping-Pong Prevention (Soft Filter) ---
            # Filter out actions that reverse the current visit mode,
            # but fallback to allowing them if no other actions are available.
            filtered_specs: List[ActionSpec] = []
            visit_modes = getattr(vessel, "visit_modes", {})

            for spec in unique_specs:
                is_ping_pong = False
                if spec.action_type == ActionType.LOAD and spec.resource_name:
                    if (
                        visit_modes.get(spec.resource_name, VisitMode.NEUTRAL)
                        == VisitMode.UNLOADING
                    ):
                        is_ping_pong = True
                elif spec.action_type == ActionType.UNLOAD and spec.resource_name:
                    if (
                        visit_modes.get(spec.resource_name, VisitMode.NEUTRAL)
                        == VisitMode.LOADING
                    ):
                        is_ping_pong = True

                if not is_ping_pong:
                    filtered_specs.append(spec)

            if not filtered_specs and unique_specs:
                # Deadlock prevented: allow the ping-pong action so the vessel can "undo"
                # its previous action and hopefully unlock new valid movements.
                logger.debug(
                    f"Allowing ping-pong for {vessel.name} to prevent deadlock."
                )
                return unique_specs
            return filtered_specs
        else:
            return unique_specs

    # =====================================================================
    # Business rules – domain-specific operational constraints
    # =====================================================================

    def _build_business_rules(
        self,
        vessel: Vessel,
        sites: Dict[str, Site],
        reservations: ReservationSystem,
        helpers: "_RuleHelpers",
    ) -> List[ActionRule]:
        """Build action rules that encode domain-specific business logic.

        All site/vessel identity checks are driven by the ``role`` fields in
        :class:`SiteConfig` and :class:`VesselConfig`, so adding new sites or
        vessels requires only config changes.
        """

        move_spec = helpers.move_spec
        load_spec = helpers.load_spec
        unload_spec = helpers.unload_spec
        is_object_full = helpers.is_object_full
        is_object_empty = helpers.is_object_empty
        can_object_add_resource = helpers.can_object_add_resource
        can_object_give_resource = helpers.can_object_give_resource
        are_vessels_at_same_site = helpers.are_vessels_at_same_site

        rules: List[ActionRule] = []

        # Pre-compute role-based site groups
        source_sites = self._get_sites_by_role(SiteRole.SOURCE)
        staging_sites = self._get_sites_by_role(SiteRole.STAGING)
        installation_sites = self._get_sites_by_role(SiteRole.INSTALLATION)

        # Pre-compute role-based vessel groups
        installer_vessels = self._get_vessels_by_role(VesselRole.INSTALLER)

        is_transport = self._is_transport_vessel(vessel.name)
        is_installer = self._is_installer_vessel(vessel.name)

        # =====================================================================
        #  TRANSPORT VESSEL rules
        # =====================================================================
        if is_transport:
            # --- Movement from SOURCE sites ---
            for src_name, src_site in source_sites.items():
                for inst_name, inst_site in installation_sites.items():
                    # Rule 4: block moving to installation if vessel is empty
                    # Rule 6: block moving to installation during no-install window
                    rules.append(
                        ActionRule(
                            name=f"transport-move-from-source-{src_name}-to-install-{inst_name}",
                            predicate=lambda v, s, r, sn=src_name, dest=inst_site: (
                                s.name == sn
                                and not is_object_empty(v)  # Rule 4
                                and not self.is_in_no_install_window()  # Rule 6
                            ),
                            build=lambda v, s, r, d=inst_site: [move_spec(d)],
                        )
                    )

                for stg_name, stg_site in staging_sites.items():
                    # Rule 10: prevent moving to an empty marshalling when both vessel and marshalling are empty
                    # (Also implicitly covers Rule 1/9 edge cases where empty vessel wouldn't move to empty staging)
                    rules.append(
                        ActionRule(
                            name=f"transport-move-from-source-{src_name}-to-staging-{stg_name}",
                            predicate=lambda v, s, r, sn=src_name, dest=stg_site: (
                                s.name == sn
                                and not (
                                    is_object_empty(v) and is_object_empty(dest)
                                )  # Rule 10
                            ),
                            build=lambda v, s, r, d=stg_site: [move_spec(d)],
                        )
                    )

            # --- Movement from STAGING sites ---
            for stg_name, stg_site in staging_sites.items():
                for inst_name, inst_site in installation_sites.items():
                    # Rule 4: block moving to installation if vessel is empty
                    # Rule 6: block moving to installation during no-install window
                    rules.append(
                        ActionRule(
                            name=f"transport-move-from-staging-{stg_name}-to-install-{inst_name}",
                            predicate=lambda v, s, r, sn=stg_name, dest=inst_site: (
                                s.name == sn
                                and not is_object_empty(v)  # Rule 4
                                and not self.is_in_no_install_window()  # Rule 6
                            ),
                            build=lambda v, s, r, d=inst_site: [move_spec(d)],
                        )
                    )

                for src_name, src_site in source_sites.items():
                    # Rule 3: block moving to fabrication if vessel is carrying piles
                    # Rule 7: block moving to fabrication if the fabrication site has no piles
                    rules.append(
                        ActionRule(
                            name=f"transport-move-from-staging-{stg_name}-to-source-{src_name}",
                            predicate=lambda v, s, r, sn=stg_name, dest=src_site: (
                                s.name == sn
                                and is_object_empty(v)  # Rule 3
                                and not is_object_empty(dest)  # Rule 7
                            ),
                            build=lambda v, s, r, d=src_site: [move_spec(d)],
                        )
                    )

            # --- Movement from INSTALLATION sites ---
            for inst_name, inst_site in installation_sites.items():
                for src_name, src_site in source_sites.items():
                    # Empty transport at installation can return to source
                    # Rule 3: block moving to fabrication if vessel is carrying piles
                    # Rule 7: block moving to fabrication if the fabrication site has no piles
                    rules.append(
                        ActionRule(
                            name=f"transport-move-from-install-{inst_name}-to-source-{src_name}",
                            predicate=lambda v, s, r, sn=inst_name, dest=src_site: (
                                s.name == sn
                                and is_object_empty(
                                    v
                                )  # Rule 3 (Also satisfies existing logic to return when empty)
                                and not is_object_empty(dest)  # Rule 7
                            ),
                            build=lambda v, s, r, d=src_site: [move_spec(d)],
                        )
                    )

                for stg_name, stg_site in staging_sites.items():
                    # Transport leaves installation if it is empty, or if all co-located installers are full
                    # Rule 10: prevent moving to an empty marshalling when both vessel and marshalling are empty
                    rules.append(
                        ActionRule(
                            name=f"transport-move-from-install-{inst_name}-to-staging-{stg_name}",
                            predicate=lambda v, s, r, sn=inst_name, dest=stg_site: (
                                s.name == sn
                                and (
                                    is_object_empty(v)
                                    or all(
                                        is_object_full(iv)
                                        for iv in installer_vessels.values()
                                        if are_vessels_at_same_site(v, iv)
                                    )
                                )
                                and not (
                                    is_object_empty(v) and is_object_empty(dest)
                                )  # Rule 10
                            ),
                            build=lambda v, s, r, d=stg_site: [move_spec(d)],
                        )
                    )

            # --- Loading: transport loads from source and staging sites ---
            for src_name, src_site in source_sites.items():
                # Rule 1 / 9: At a fabrication site, allow loading if vessel isn't full and source has resources.
                # Rule 8: Implicitly handled natively by `can_object_add_resource` loops
                rules.append(
                    ActionRule(
                        name=f"transport-load-from-source-{src_name}",
                        predicate=lambda v, s, r, sn=src_name: (
                            s.name == sn
                            and can_object_add_resource(v, r)
                            and can_object_give_resource(s, r)
                        ),
                        build=lambda v, s, r: [load_spec(s, amount=1, resource_name=r)],
                    )
                )

            for stg_name, stg_site in staging_sites.items():
                # Rule 5 (Ping-pong) is now handled globally via soft filter
                rules.append(
                    ActionRule(
                        name=f"transport-load-from-staging-{stg_name}",
                        predicate=lambda v, s, r, sn=stg_name: (
                            s.name == sn
                            and can_object_add_resource(v, r)
                            and can_object_give_resource(s, r)
                        ),
                        build=lambda v, s, r: [load_spec(s, amount=1, resource_name=r)],
                    )
                )

            # --- Unloading: transport unloads to staging sites ---
            for stg_name, stg_site in staging_sites.items():
                # Rule 5 (Ping-pong) is now handled globally via soft filter
                rules.append(
                    ActionRule(
                        name=f"transport-unload-at-staging-{stg_name}",
                        predicate=lambda v, s, r, sn=stg_name: (
                            s.name == sn
                            and can_object_give_resource(v, r)
                            and can_object_add_resource(s, r)
                        ),
                        build=lambda v, s, r: [
                            unload_spec(s, amount=1, resource_name=r)
                        ],
                    )
                )

            # --- Transfer: transport transfers to co-located installer vessels ---
            # Rule 11: Transport vessels do not load from installer assets, they only unload (transfer) to them.
            for ins_name, ins_vessel in installer_vessels.items():
                rules.append(
                    ActionRule(
                        name=f"transport-transfer-to-{ins_name}",
                        predicate=lambda v, s, r, iv=ins_vessel: (
                            can_object_add_resource(iv, r)
                            and can_object_give_resource(v, r)
                            and are_vessels_at_same_site(v, iv)
                        ),
                        build=lambda v, s, r, iv=ins_vessel: [
                            unload_spec(iv, amount=1, resource_name=r)
                        ],
                    )
                )

        # =====================================================================
        #  INSTALLER VESSEL rules
        # =====================================================================
        if is_installer:
            # --- Install: unload at installation sites ---
            for inst_name, inst_site in installation_sites.items():
                rules.append(
                    ActionRule(
                        name=f"installer-install-at-{inst_name}",
                        predicate=lambda v, s, r, sn=inst_name: (
                            s.name == sn
                            and can_object_add_resource(s, r)
                            and can_object_give_resource(v, r)
                            and self._goal_tracker.has_unblocked_goals_for(sn, r)
                        ),
                        build=lambda v, s, r: [
                            unload_spec(s, amount=1, resource_name=r)
                        ],
                    )
                )

            # --- Movable installer rules ---
            if self._is_vessel_movable(vessel.name):
                # Move from installation to staging
                for inst_name in installation_sites:
                    for stg_name, stg_site in staging_sites.items():
                        # Rule 10: prevent moving to empty marshalling when both vessel and marshalling are empty
                        rules.append(
                            ActionRule(
                                name=f"installer-move-from-install-{inst_name}-to-staging-{stg_name}",
                                predicate=lambda v, s, r, sn=inst_name, dest=stg_site: (
                                    s.name == sn
                                    and not (
                                        is_object_empty(v) and is_object_empty(dest)
                                    )  # Rule 10
                                ),
                                build=lambda v, s, r, d=stg_site: [move_spec(d)],
                            )
                        )

                # Move from staging to installation
                for stg_name in staging_sites:
                    for inst_name, inst_site in installation_sites.items():
                        # Rule 4: block moving to installation if vessel is empty
                        # Rule 6: block moving to installation during no-install window
                        rules.append(
                            ActionRule(
                                name=f"installer-move-from-staging-{stg_name}-to-install-{inst_name}",
                                predicate=lambda v, s, r, sn=stg_name, dest=inst_site: (
                                    s.name == sn
                                    and not is_object_empty(v)  # Rule 4
                                    and not self.is_in_no_install_window()  # Rule 6
                                ),
                                build=lambda v, s, r, d=inst_site: [move_spec(d)],
                            )
                        )

                # Load from staging
                for stg_name, stg_site in staging_sites.items():
                    # Rule 5 (Ping-pong) is now handled globally via soft filter
                    rules.append(
                        ActionRule(
                            name=f"installer-load-from-staging-{stg_name}",
                            predicate=lambda v, s, r, sn=stg_name: (
                                s.name == sn
                                and can_object_add_resource(v, r)
                                and can_object_give_resource(s, r)
                            ),
                            build=lambda v, s, r: [
                                load_spec(s, amount=1, resource_name=r)
                            ],
                        )
                    )

                # Unload at staging
                for stg_name, stg_site in staging_sites.items():
                    # Rule 5 (Ping-pong) is now handled globally via soft filter
                    rules.append(
                        ActionRule(
                            name=f"installer-unload-at-staging-{stg_name}",
                            predicate=lambda v, s, r, sn=stg_name: (
                                s.name == sn
                                and can_object_add_resource(s, r)
                                and can_object_give_resource(v, r)
                            ),
                            build=lambda v, s, r: [
                                unload_spec(s, amount=1, resource_name=r)
                            ],
                        )
                    )

        return rules

    # =====================================================================
    # Physical rules – only real-world / capacity constraints
    # =====================================================================

    def _build_physical_rules(
        self,
        vessel: Vessel,
        vessel_site: Site,
        sites: Dict[str, Site],
        reservations: ReservationSystem,
        helpers: "_RuleHelpers",
    ) -> List[ActionRule]:
        """Build action rules that enforce only physical constraints.

        No business-logic restrictions are applied.  Role-based physical
        constraints are still enforced:

        * **Movement** – any vessel with ``movable: true`` can move to any
          other site.  Vessels with ``movable: false`` cannot move.
        * **Loading** – vessels can load from any site that is **not** an
          installation site (installed items are permanent).
        * **Unloading to site** – vessels can unload to source and staging
          sites freely.  Unloading at an installation site is restricted to
          *installer* vessels and subject to the no-install weather window.
        * **Unloading to source** – physically blocked: returning resources to
          a source site serves no purpose and is treated as a physical
          constraint (the site produces resources, it does not accept them).
        * **Transfer** – any vessel can transfer to any co-located vessel.

        Parameters
        ----------
        vessel : Vessel
            The vessel for which rules are being built.
        vessel_site : Site
            The site where *vessel* is currently located.
        sites : Dict[str, Site]
            All sites keyed by name.
        reservations : ReservationSystem
            Current reservation context.
        helpers : _RuleHelpers
            Shared helper closures for capacity / spec construction.

        Returns
        -------
        List[ActionRule]
            Physically-constrained rules (no business logic).
        """

        move_spec = helpers.move_spec
        load_spec = helpers.load_spec
        unload_spec = helpers.unload_spec
        can_object_add_resource = helpers.can_object_add_resource
        can_object_give_resource = helpers.can_object_give_resource
        are_vessels_at_same_site = helpers.are_vessels_at_same_site

        rules: List[ActionRule] = []

        # --- Movement: move to any site the vessel is not already at ---
        # Respects per-vessel `movable` flag from config.
        # Feeder restriction: when `restrict_feeders_to_relay` is set,
        # feeder-role vessels may only move between staging and installation
        # sites (not source sites).
        can_move = self._is_vessel_movable(vessel.name)
        is_installer = self._is_installer_vessel(vessel.name)
        is_feeder = self._get_vessel_role(vessel.name) == VesselRole.FEEDER

        other_sites = [s for s in sites.values() if s.name != vessel_site.name]
        if is_feeder and self.config.restrict_feeders_to_relay:
            other_sites = [s for s in other_sites if not self._is_source_site(s.name)]
        if other_sites and can_move:
            rules.append(
                ActionRule(
                    name="physical-move-to-any-other-site",
                    predicate=lambda v, s, r: True,
                    build=lambda v, s, r: [
                        move_spec(dest)
                        for dest in other_sites
                        if not (
                            self._is_installation_site(dest.name)
                            and self.is_in_no_install_window()
                        )
                    ],
                )
            )

        # --- Loading: load any resource from the current site ---
        # Installation sites: installed items are permanent (physical constraint).
        # Source sites: can be loaded from (that is their purpose).
        # Staging sites: can be loaded from freely.
        rules.append(
            ActionRule(
                name="physical-load-from-current-site",
                predicate=lambda v, s, r: (
                    not self._is_installation_site(s.name)
                    and can_object_add_resource(v, r)
                    and can_object_give_resource(s, r)
                ),
                build=lambda v, s, r: [load_spec(s, amount=1, resource_name=r)],
            )
        )

        # --- Unloading to current site ---
        # Source sites: cannot accept resources (they produce, not consume).
        # Installation sites: only installer vessels may unload (install) and
        #   the no-install weather window must not be active.  Additionally,
        #   goal-dependency ordering is enforced here as a physical constraint:
        #   a resource cannot be installed unless an unblocked goal exists for
        #   that resource at the site (e.g., cannot install a TP before its
        #   foundation dependency is satisfied).
        # Staging sites: any vessel may unload freely.
        rules.append(
            ActionRule(
                name="physical-unload-at-current-site",
                predicate=lambda v, s, r: (
                    can_object_give_resource(v, r)
                    and can_object_add_resource(s, r)
                    and not self._is_source_site(s.name)
                    and not (self._is_installation_site(s.name) and not is_installer)
                    and not (
                        self._is_installation_site(s.name)
                        and not self._goal_tracker.has_unblocked_goals_for(s.name, r)
                    )
                ),
                build=lambda v, s, r: [unload_spec(s, amount=1, resource_name=r)],
            )
        )

        # --- Unloading (transfer) to another vessel at the same site ---
        for other_name, other_vessel in self._vessels_by_name.items():
            if other_name == vessel.name:
                continue
            rules.append(
                ActionRule(
                    name=f"physical-transfer-to-{other_name}",
                    predicate=lambda v, s, r, ov=other_vessel: (
                        can_object_give_resource(v, r)
                        and can_object_add_resource(ov, r)
                        and are_vessels_at_same_site(v, ov)
                    ),
                    build=lambda v, s, r, ov=other_vessel: [
                        unload_spec(ov, amount=1, resource_name=r)
                    ],
                )
            )

        return rules

    def _get_possible_activities(
        self, vessel: TransportProcessingResource
    ) -> List[des_model.GenericActivity]:
        """Generate possible activities for a vessel based on current state.

        This method now delegates to get_possible_actions to get specs,
        and then converts them to activities using the ActivityBuilder.

        Parameters
        ----------
        vessel : TransportProcessingResource
            The vessel for which to generate activities.

        Returns
        -------
        List[des_model.GenericActivity]
            All activities that are valid given the current vessel and site state.
        """
        action_specs = self.get_possible_actions(vessel.name)

        activities = []
        for spec in action_specs:
            activity = self._activity_builder.build(spec)
            activities.append(activity)

        # Debug logging
        vessel_site = self._get_vessel_site(vessel)
        logger.debug(
            f"Generated {len(activities)} activities for {vessel.name} at {vessel_site.name}. "
            f"Vessel inventory: {self.get_vessel_inventory(vessel.name)}. "
            f"Activities: {[act.name for act in activities]}"
        )

        return activities

    # =========================================================================
    # Initialization Helper Methods
    # =========================================================================

    def _initialize_dummy_activities(self) -> None:
        """Initialize dummy activity required by DES.

        DES's dynamic mode requires at least one activity in the
        registry before the simulation can proceed. This creates a zero-duration
        BasicActivity and processes it immediately.
        """
        dummy_activity = des_model.BasicActivity(
            env=self.des_env,
            name="Initialization dummy activity",
            registry=self.des_env.registry,
            duration=0,
        )
        des_model.register_processes([dummy_activity])
        self.des_env.next_step()

    def _build_vessel_lookup(self) -> Dict[str, Vessel]:
        """Build a name-to-vessel lookup dictionary.

        Returns
        -------
        Dict[str, Vessel]
            Mapping from vessel name to vessel object.
        """
        all_vessels = self._get_all_vessels()
        return {vessel.name: vessel for vessel in all_vessels}

    def _build_site_lookup(self) -> Dict[str, Site]:
        """Build a name-to-site lookup dictionary.

        Returns
        -------
        Dict[str, Site]
            Mapping from site name to site object.
        """
        all_sites = self._get_all_sites()
        return {site.name: site for site in all_sites}

    # -----------------------------------------------------------------
    # Role-query helpers – driven entirely by config
    # -----------------------------------------------------------------

    def _get_site_role(self, site_name: str) -> SiteRole:
        """Return the :class:`SiteRole` for *site_name*.

        Falls back to ``SiteRole.STAGING`` when the site config does not
        specify a role (backward-compatible default).
        """
        role_str = getattr(self.config.sites[site_name], "role", "staging")
        return SiteRole(role_str)

    def _is_source_site(self, site_name: str) -> bool:
        """True when *site_name* is a resource-origin (load-only) site."""
        return self._get_site_role(site_name) == SiteRole.SOURCE

    def _is_installation_site(self, site_name: str) -> bool:
        """True when *site_name* is a permanent-installation site."""
        return self._get_site_role(site_name) == SiteRole.INSTALLATION

    def _is_staging_site(self, site_name: str) -> bool:
        """True when *site_name* is an intermediate buffer site."""
        return self._get_site_role(site_name) == SiteRole.STAGING

    def _get_vessel_role(self, vessel_name: str) -> VesselRole:
        """Return the :class:`VesselRole` for *vessel_name*.

        Falls back to ``VesselRole.HEAVY_LIFT`` when the vessel config does
        not specify a role (backward-compatible default).
        """
        role_str = getattr(self.config.vessels[vessel_name], "role", "heavy_lift")
        return VesselRole(role_str)

    def _is_transport_vessel(self, vessel_name: str) -> bool:
        """True when *vessel_name* is a transport-class (non-installer) vessel."""
        return self._get_vessel_role(vessel_name).is_transport

    def _is_installer_vessel(self, vessel_name: str) -> bool:
        """True when *vessel_name* is an installation asset."""
        return self._get_vessel_role(vessel_name) == VesselRole.INSTALLER

    def _is_vessel_movable(self, vessel_name: str) -> bool:
        """True when *vessel_name* is allowed to move between sites.

        Uses the per-vessel ``movable`` flag from :class:`VesselConfig`.
        For backward compatibility, if the flag is not explicitly set and
        the vessel is an installer, falls back to the (deprecated)
        global ``is_installation_vessel_movable`` flag on :class:`SimConfig`.
        """
        vcfg = self.config.vessels[vessel_name]
        # If the config explicitly carries a `movable` field, use it.
        if hasattr(vcfg, "movable"):
            return vcfg.movable
        # Fallback: transport vessels are always movable; installer vessels
        # defer to the deprecated global flag.
        if self._is_installer_vessel(vessel_name):
            return self.config.is_installation_vessel_movable
        return True

    def _get_sites_by_role(self, role: SiteRole) -> Dict[str, Site]:
        """Return all sites whose config role matches *role*."""
        return {
            name: self._sites_by_name[name]
            for name in self._sites_by_name
            if self._get_site_role(name) == role
        }

    def _get_vessels_by_role(self, role: VesselRole) -> Dict[str, "Vessel"]:
        """Return all vessels whose config role matches *role*."""
        return {
            name: self._vessels_by_name[name]
            for name in self._vessels_by_name
            if self._get_vessel_role(name) == role
        }

    def _build_simulation_objects(self) -> None:
        """Create and register all simulation objects (sites and vessels).

        This method instantiates objects based on self.config, using the
        type-based resource model (resource_types + initial_levels).

        When a resource type has a non-empty ``fabrication_schedule`` for a
        source site, the initial level is derived from the schedule (count
        of entries where ``time <= 0``) rather than defaulting to capacity.
        This ensures backward compatibility: configs without fabrication
        schedules behave identically to Phase 1.
        """
        # Canonical list of all resource types in the simulation
        all_resource_types = list(self.config.resource_types.keys())

        # Build Sites
        for name, data in self.config.sites.items():
            location = self._parse_location(data.location)
            is_source = data.role == "source"

            initials = []
            for rtype in all_resource_types:
                slot = data.resource_types.get(rtype)
                if slot is None:
                    continue
                capacity = slot.capacity

                # Check if this (resource_type, source_site) pair has a
                # fabrication schedule.
                rtype_cfg = self.config.resource_types.get(rtype)
                fab_schedule = (
                    rtype_cfg.fabrication_schedule.get(name)
                    if rtype_cfg is not None
                    else None
                )
                has_fab_schedule = fab_schedule is not None and len(fab_schedule) > 0

                # Determine initial level:
                # 1. If initial_levels explicitly specifies a value, use it.
                # 2. If the site is a source AND a fabrication schedule
                #    exists for this (resource_type, site) pair, derive the
                #    initial level from the schedule (count of times <= 0).
                # 3. For source sites with NO schedule, default to capacity
                #    (backward compat with Phase 1).
                # 4. Otherwise default to 0.
                if rtype in data.initial_levels:
                    level = data.initial_levels[rtype]
                elif is_source and has_fab_schedule:
                    level = sum(1 for t in fab_schedule if t <= 0)
                elif is_source:
                    level = capacity
                else:
                    level = 0
                initials.append({"id": rtype, "level": level, "capacity": capacity})

            Site(
                env=self.des_env,
                name=name,
                geometry=location,
                initials=initials,
                store_capacity=len(initials),
            )

        # Build Vessels
        sites_registry = self.des_env.registry["sim_objects"]["Site"]

        for name, data in self.config.vessels.items():
            start_location_name = data.start_location
            if start_location_name not in sites_registry:
                raise ValueError(
                    f"Unknown start location '{start_location_name}' for vessel '{name}'"
                )

            start_location = sites_registry[start_location_name].geometry

            initials = []
            for rtype in all_resource_types:
                slot = data.resource_types.get(rtype)
                if slot is None:
                    continue
                capacity = slot.capacity
                # Vessels start empty unless initial_levels is specified
                level = data.initial_levels.get(rtype, 0)
                initials.append({"id": rtype, "level": level, "capacity": capacity})

            vessel_type = data.type

            common_args = {
                "env": self.des_env,
                "name": name,
                "geometry": start_location,
                "initials": initials,
                "store_capacity": len(initials),
            }

            if vessel_type == "TransportProcessingResource":
                speed = data.speed if data.speed is not None else 10.0
                TransportProcessingResource(
                    **common_args,
                    compute_v=lambda x, s=speed: s,
                    loading_rate=data.loading_rate
                    if data.loading_rate is not None
                    else 1.0,
                    unloading_rate=data.unloading_rate
                    if data.unloading_rate is not None
                    else 1.0,
                )
            elif vessel_type == "InstallationAsset":
                InstallationAsset(**common_args)
            else:
                raise ValueError(f"Unknown vessel type: {vessel_type}")

        # Debug logging
        log_sim_objects(self.des_env.registry)

    def _parse_location(self, loc_data: List[float]) -> shapely.geometry.point.Point:
        """Convert list [x, y] to Point(x, y)."""
        return shapely.geometry.point.Point(loc_data[0], loc_data[1])

    # =====================================================================
    # Fabrication Scheduling (Phase 2)
    # =====================================================================

    _PHANTOM_PREFIX: str = "_phantom_fab_"

    def _spawn_fabrication_activities(self) -> None:
        """Create phantom vessels and time-gated DES activities for fabrication.

        For each ``(resource_type, source_site)`` pair that has a non-empty
        ``fabrication_schedule``, this method:

        1. Pre-collects all ``(resource_type, future_count)`` pairs per
           source site so that each phantom vessel is created once with
           **all** the resource slots it needs.
        2. Creates a hidden *phantom vessel* co-located with each source
           site.  The phantom holds enough capacity to act as the
           ``origin`` for ``ShiftAmountActivity`` (DES requires a
           co-located origin with a container).  Each slot is pre-loaded
           with the total future units for that resource type.
        3. Spawns one ``ShiftAmountActivity`` per future fabrication entry
           (time > 0), gated by a ``start_event`` time condition.
        4. Registers each activity with the ``ActivityTracker`` and records
           a ``reserve_fill`` in the ``ReservationSystem`` so that action
           masks can account for incoming supply.
        5. Stores a synthetic ``ActionSpec`` per activity in
           ``_spec_by_activity`` so the normal completion-revert path in
           ``step()`` handles cleanup automatically.

        Phantom vessels are tracked in ``_phantom_vessel_names`` and
        excluded from ``_get_all_vessels`` so they never leak into
        observations, action masks, or training data.
        """
        sites_registry = self.des_env.registry["sim_objects"]["Site"]

        # --- Pass 1: Pre-collect resource slots per source site ----------
        # phantom_specs[source_site_name] -> list of {"id": rtype, "level": N, "capacity": N}
        phantom_specs: Dict[str, List[Dict[str, Any]]] = {}
        # Also collect the (rtype, source_site_name, future_times) tuples for Pass 2
        fabrication_entries: List[tuple] = []

        for rtype, rtype_cfg in self.config.resource_types.items():
            if not rtype_cfg.fabrication_schedule:
                continue

            for (
                source_site_name,
                schedule_times,
            ) in rtype_cfg.fabrication_schedule.items():
                future_times = [t for t in schedule_times if t > 0]
                if not future_times:
                    continue

                if source_site_name not in sites_registry:
                    logger.warning(
                        f"Fabrication schedule references unknown site "
                        f"'{source_site_name}' for resource '{rtype}'. Skipping."
                    )
                    continue

                # Record for phantom creation
                if source_site_name not in phantom_specs:
                    phantom_specs[source_site_name] = []
                phantom_specs[source_site_name].append(
                    {
                        "id": rtype,
                        "level": len(future_times),
                        "capacity": len(future_times),
                    }
                )

                # Record for activity spawning in pass 2
                fabrication_entries.append((rtype, source_site_name, future_times))

        if not fabrication_entries:
            return

        # --- Pass 2: Create phantom vessels (one per source site) --------
        for source_site_name, initials in phantom_specs.items():
            phantom_name = f"{self._PHANTOM_PREFIX}{source_site_name}"
            self._phantom_vessel_names.add(phantom_name)
            source_site = sites_registry[source_site_name]

            TransportProcessingResource(
                env=self.des_env,
                name=phantom_name,
                geometry=source_site.geometry,
                initials=initials,
                store_capacity=len(initials),
                compute_v=lambda x: 1e-9,
                loading_rate=1.0,
                unloading_rate=1.0,
            )

        # --- Pass 3: Spawn DES activities and register reservations ------
        for rtype, source_site_name, future_times in fabrication_entries:
            source_site = sites_registry[source_site_name]
            phantom_name = f"{self._PHANTOM_PREFIX}{source_site_name}"
            phantom_vessel = self._get_phantom_vessel(phantom_name)

            for fab_time_hours in future_times:
                fab_short_id = self._activity_builder.next_fabrication_id()
                activity = self._create_fabrication_activity(
                    phantom_vessel=phantom_vessel,
                    source_site=source_site,
                    resource_name=rtype,
                    fab_time_hours=fab_time_hours,
                    short_id=fab_short_id,
                )

                # Register with DES (dynamic registration)
                des_model.register_additional_processes(
                    self.des_env,
                    activity,
                    simulation_object=phantom_vessel,
                )

                # Register with ActivityTracker so dependency resolution
                # sees these activities.
                self._activity_tracker.register_activity(activity, self.sim_step)

                # Register a fill in the ReservationSystem so action
                # masks can account for incoming supply at the source.
                self._reservation_system.reserve_fill(
                    source_site_name,
                    rtype,
                    1,
                    source=f"fabrication:{rtype}:{source_site_name}:{fab_time_hours}h",
                )

                # Register as a granular pending delivery so that LOADs
                # can later claim this specific fabrication activity.
                self._reservation_system.register_pending_delivery(
                    source_site_name, rtype, activity.name
                )

                # Store a synthetic ActionSpec so the normal completion-
                # revert path in step() removes the fill projection.
                synthetic_spec = ActionSpec(
                    action_type=ActionType.LOAD,
                    vessel_name=phantom_name,
                    partner_name=source_site_name,
                    resource_name=rtype,
                    amount=1,
                )
                self._spec_by_activity[activity.name] = synthetic_spec

        fab_count = self._activity_builder._fabrication_counter
        logger.debug(
            f"Fabrication scheduling: spawned {fab_count} "
            f"time-gated activities across "
            f"{len(self._phantom_vessel_names)} phantom vessel(s)."
        )

    def _get_phantom_vessel(self, phantom_name: str) -> TransportProcessingResource:
        """Look up a phantom vessel by name from the DES registry.

        Parameters
        ----------
        phantom_name : str
            Name of the phantom vessel.

        Returns
        -------
        TransportProcessingResource
            The phantom vessel object.

        Raises
        ------
        KeyError
            If the phantom vessel is not found in the registry.
        """
        return self.des_env.registry["sim_objects"]["TransportProcessingResource"][
            phantom_name
        ]

    def _create_fabrication_activity(
        self,
        phantom_vessel: TransportProcessingResource,
        source_site: Site,
        resource_name: str,
        fab_time_hours: float,
        short_id: str,
    ) -> des_model.ShiftAmountActivity:
        """Build a single time-gated ``ShiftAmountActivity`` for fabrication.

        The activity transfers 1 unit of *resource_name* from the
        *phantom_vessel* to the *source_site* at the scheduled time.

        Parameters
        ----------
        phantom_vessel : TransportProcessingResource
            The hidden vessel co-located with the source site (origin).
        source_site : Site
            The source site that receives the fabricated resource.
        resource_name : str
            Which resource type is being fabricated.
        fab_time_hours : float
            Scheduled fabrication time in hours from simulation start.
        short_id : str
            Global short ID for this activity (e.g. ``"F3"``).

        Returns
        -------
        des_model.ShiftAmountActivity
            The constructed (but not yet registered) activity.
        """
        # Convert hours → absolute datetime for DES time gate
        fab_epoch = self.simulation_start.timestamp() + fab_time_hours * 3600
        fab_dt = datetime.datetime.fromtimestamp(fab_epoch, tz=self.des_env.tzinfo)

        start_event = [{"type": "time", "start_time": fab_dt}]

        activity_name = (
            f"{short_id}: Fabrication {resource_name} "
            f"@ {source_site.name} t={fab_time_hours}h"
        )

        # Register the name → short ID mapping on the builder
        self._activity_builder.register_name(activity_name, short_id)

        return des_model.ShiftAmountActivity(
            env=self.des_env,
            name=activity_name,
            registry=self.registry,
            processor=phantom_vessel,
            origin=phantom_vessel,
            destination=source_site,
            amount=1,
            duration=0,  # instantaneous transfer
            category="fabrication",
            id_=resource_name,
            start_event=start_event,
        )

    def _is_phantom_vessel(self, name: str) -> bool:
        """Return ``True`` when *name* belongs to a phantom fabrication vessel."""
        return name in self._phantom_vessel_names
