"""Activity builder for creating EventSymphony activities from action specifications.

This module provides a centralized builder class for constructing activities,
making it easy to configure and customize activity properties like durations,
rates, and other parameters.

Intent-Based Queuing
--------------------
When an :class:`ActivityTracker` is attached, the builder automatically injects
``start_event`` conditions that gate each new activity on the completion of all
PENDING / ACTIVE activities for every participating entity (vessel **and**
partner).  This allows actions to be *queued* — the DES keeps them PENDING
until all prerequisites are satisfied — eliminating the need for explicit
NOOP / IDLE polling.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

# TYPE_CHECKING avoids circular imports (ReservationSystem imports ActionSpec)
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import boka_eventsymphony.core as es_core
import boka_eventsymphony.model as es_model
import numpy as np
from loguru import logger

from eos.config import ActivityConfig

from .activity_tracker import ActivityTracker
from .types import ActionType, Site, Vessel

if TYPE_CHECKING:
    from .reservation_system import ReservationSystem

# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ActionSpec:
    """Specification for an action to be converted into an activity.

    Parameters
    ----------
    action_type : ActionType
        Type of action: move, load, unload, or idle.
    vessel_name : str
        Name of the vessel performing the action.
    destination_name : str | None
        For move actions: target site name.
    partner_name : str | None
        For load/unload actions: source/destination name (site or vessel).
    resource_name : str | None
        Resource type being transferred (e.g., large_mp).
    amount : int | None
        Amount of resource to transfer.
    duration : float | None
       Duration in hours.
    start_event: Dict[str, Any] | None
        Requirements for action to start.
    """

    action_type: ActionType
    vessel_name: str
    destination_name: str | None = None
    partner_name: str | None = None
    resource_name: str | None = None
    amount: int | None = None
    duration: float | None = None
    start_event: List[Dict[str, Any]] | None = None


@dataclass
class SequentialActionSpec:
    actions: List[ActionSpec]
    name: str | None = None
    start_event: List[Dict[str, Any]] | None = None


class ActivityBuilder:
    """Builder for creating EventSymphony activities from action specifications.

    This class centralizes activity creation logic and provides easy customization
    of activity properties like durations, amounts, and other parameters.

    Parameters
    ----------
    env : es_core.Environment
        The EventSymphony environment.
    registry : dict
        The simulation registry containing all objects.
    config : ActivityConfig, optional
        Configuration for activity parameters. If None, uses defaults.
    activity_tracker : ActivityTracker | None, optional
        When provided, the builder auto-injects ``start_event`` dependency
        conditions so that new activities stay PENDING until every
        participant's prior work is done.  Can also be set later via
        :meth:`set_activity_tracker`.
    no_install_windows : list of [start_hours, end_hours] pairs, optional
        Time windows (in hours from simulation start) during which
        installation activities may **not** begin.  When an installation
        unload is built while the current time falls inside one of these
        windows, a ``{"type": "time", "start_time": <epoch>}`` start
        event is injected so that EventSymphony delays the activity until
        the window closes.  Adjacent / overlapping windows are merged
        automatically.

    Attributes
    ----------
    config : ActivityConfig
        Current configuration for activity parameters.
    """

    def __init__(
        self,
        env: es_core.Environment,
        registry: dict[str, Any],
        config: ActivityConfig | None = None,
        activity_tracker: ActivityTracker | None = None,
        no_install_windows: List[List[float]] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.env = env
        self.registry = registry
        self.config = config or ActivityConfig()

        # RNG used for stochastic duration perturbation.  Only consulted
        # when ``config.stochasticity.enabled`` is True.  When the feature
        # is enabled but no generator is supplied, a default (unseeded)
        # generator is created lazily so the builder never crashes; in the
        # normal env flow the seeded per-episode ``np_random`` is passed in.
        self._rng: np.random.Generator | None = rng

        # Optional tracker for dependency-aware start_events
        self._activity_tracker: ActivityTracker | None = activity_tracker

        # Optional reservation system for granular resource claiming
        self._reservation_system: ReservationSystem | None = None

        # Pre-process and store merged no-install windows
        self._no_install_windows: List[Tuple[float, float]] = (
            self._merge_no_install_windows(no_install_windows or [])
        )

        # Cache for quick lookups
        self._vessels_by_name: dict[str, Vessel] = {}
        self._sites_by_name: dict[str, Site] = {}
        self._refresh_object_cache()

        # Global activity counter — every activity built by this builder
        # gets a monotonically increasing ID (A1, A2, …) regardless of
        # which vessel performs it.  Fabrication activities use the "F"
        # prefix via :meth:`next_fabrication_id` instead.
        self._global_activity_counter: int = 0

        # Mapping from full ES activity name → short ID (e.g. "A3").
        # Populated by :meth:`_next_activity_id` and
        # :meth:`next_fabrication_id` so that downstream consumers
        # (Phase 1 dependency display) can resolve names to short IDs.
        self._name_to_short_id: Dict[str, str] = {}

        # Fabrication counter (separate "F" prefix series)
        self._fabrication_counter: int = 0

        # Dependency short IDs resolved during the last build() call.
        # Populated by :meth:`_build_action_spec` after the activity is
        # constructed, consumed by the simulator / gym_env to surface
        # dependency info in the action trace.
        self._last_dep_ids: List[str] = []

        # The final start_event list produced by the most recent
        # _build_*_activity call.  Set by each builder method so that
        # _build_action_spec can resolve dependency IDs without needing
        # to read attributes back from the opaque ES activity object.
        self._last_start_event: List[Dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Deferred wiring
    # ------------------------------------------------------------------

    def set_activity_tracker(self, tracker: ActivityTracker) -> None:
        """Attach an :class:`ActivityTracker` after construction.

        This is useful when the tracker is created after the builder (both
        live in the simulator, but the tracker depends on the builder's
        output).

        Parameters
        ----------
        tracker : ActivityTracker
            The tracker instance to attach.
        """
        self._activity_tracker = tracker

    def set_reservation_system(self, rs: ReservationSystem) -> None:
        """Attach a :class:`ReservationSystem` after construction.

        Used by :meth:`_build_load_activity` and
        :meth:`_build_unload_activity` to call
        :meth:`~ReservationSystem.claim_resource` and
        :meth:`~ReservationSystem.claim_capacity` for fine-grained
        per-item/per-slot dependencies.

        Parameters
        ----------
        rs : ReservationSystem
            The reservation system instance to attach.
        """
        self._reservation_system = rs

    # ------------------------------------------------------------------
    # No-install window helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_no_install_windows(
        windows: List[List[float]],
    ) -> List[Tuple[float, float]]:
        """Validate and merge overlapping / adjacent no-install windows.

        Windows like ``[0, 150], [150, 250]`` are merged into
        ``[(0, 250)]`` so that a single contiguous blocked period is
        recognised correctly.

        Parameters
        ----------
        windows : list of [start_hours, end_hours]
            Raw window definitions from config.

        Returns
        -------
        list of (start_hours, end_hours)
            Sorted, non-overlapping, merged windows.

        Raises
        ------
        ValueError
            If any window has ``start >= end`` or contains negative values.
        """
        if not windows:
            return []

        # Validate individual windows
        for w in windows:
            if len(w) != 2:
                raise ValueError(
                    f"Each no-install window must be [start, end], got {w}"
                )
            start, end = w
            if start < 0 or end < 0:
                raise ValueError(
                    f"No-install window bounds must be non-negative, got [{start}, {end}]"
                )
            if start >= end:
                raise ValueError(
                    f"No-install window start must be less than end, got [{start}, {end}]"
                )

        # Sort by start time, then merge overlapping / adjacent
        sorted_windows = sorted(windows, key=lambda w: w[0])
        merged: List[Tuple[float, float]] = [
            (sorted_windows[0][0], sorted_windows[0][1])
        ]

        for start, end in sorted_windows[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:  # overlapping or adjacent
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        if len(merged) != len(windows):
            logger.info(
                f"Merged {len(windows)} no-install windows into {len(merged)}: {merged}"
            )

        return merged

    def _is_installation_unload(self, vessel_name: str, destination_name: str) -> bool:
        """Check whether an unload from *vessel* to *destination* is an installation.

        An unload is considered an installation when the vessel / destination
        pair appears in ``config.installations``.

        Parameters
        ----------
        vessel_name : str
            Name of the vessel performing the unload.
        destination_name : str
            Name of the unload destination (site or vessel).

        Returns
        -------
        bool
        """
        if vessel_name not in self.config.installations:
            return False
        return hasattr(self.config.installations[vessel_name], destination_name)

    def _get_no_install_window_end(self) -> float | None:
        """Return the effective end of the current no-install window in hours.

        If the current simulation time falls inside a (merged) no-install
        window the method returns the window's end time **in hours from
        simulation start**.

        Returns ``None`` when the current time is not inside any window.
        """
        if not self._no_install_windows:
            return None

        elapsed_hours = (self.env.now - self.env.start_time) / 3600.0

        for start_hours, end_hours in self._no_install_windows:
            if start_hours <= elapsed_hours < end_hours:
                return end_hours

        return None

    def _build_no_install_window_start_event(self) -> Dict[str, Any] | None:
        """If we are currently inside a no-install window, return a
        ``{"type": "time", "start_time": <datetime>}`` condition that gates
        the activity until the window closes.

        Returns ``None`` when outside every window.
        """
        window_end_hours = self._get_no_install_window_end()
        if window_end_hours is None:
            return None

        window_end_epoch = self.env.start_time + window_end_hours * 3600.0
        window_end_dt = datetime.datetime.fromtimestamp(
            window_end_epoch, tz=self.env.tzinfo
        )

        logger.debug(
            f"Installation gated by no-install window until "
            f"hour {window_end_hours} ({window_end_dt})"
        )

        return {"type": "time", "start_time": window_end_dt}

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _next_activity_id(self) -> str:
        """Return the next global short activity ID (``A1``, ``A2``, …)."""
        self._global_activity_counter += 1
        return f"A{self._global_activity_counter}"

    def next_fabrication_id(self) -> str:
        """Return the next fabrication short ID (``F1``, ``F2``, …).

        Fabrication activities are pre-simulation and use a separate
        counter so they don't interleave with agent-driven activities.
        """
        self._fabrication_counter += 1
        return f"F{self._fabrication_counter}"

    def register_name(self, activity_name: str, short_id: str) -> None:
        """Record the mapping from a full ES activity name to its short ID.

        Parameters
        ----------
        activity_name : str
            The full activity name used in EventSymphony.
        short_id : str
            The short ID (e.g. ``"A3"`` or ``"F1"``).
        """
        self._name_to_short_id[activity_name] = short_id

    def get_short_id(self, activity_name: str) -> str | None:
        """Look up the short ID for an ES activity name.

        Returns ``None`` when the name is unknown.
        """
        return self._name_to_short_id.get(activity_name)

    @property
    def name_to_short_id(self) -> Dict[str, str]:
        """Read-only view of the full-name → short-ID mapping."""
        return dict(self._name_to_short_id)

    @property
    def last_dep_ids(self) -> List[str]:
        """Short IDs of the dependencies resolved during the last :meth:`build`.

        Returns an empty list when the last activity had no predecessors.
        """
        return list(self._last_dep_ids)

    def _extract_dep_ids(self, start_event: List[Dict[str, Any]] | None) -> List[str]:
        """Resolve activity-type start_event conditions to short IDs.

        Scans *start_event* for ``{"type": "activity", "name": ...}``
        entries and maps each activity name to its short ID via
        :attr:`_name_to_short_id`.  Unknown names are included verbatim
        so that no dependency is silently dropped.

        Parameters
        ----------
        start_event : list or None
            The final start_event list as passed to EventSymphony.

        Returns
        -------
        List[str]
            Ordered, de-duplicated short IDs (e.g. ``["A1", "F3"]``).
        """
        if not start_event:
            return []

        seen: set[str] = set()
        ids: List[str] = []
        for cond in start_event:
            if cond.get("type") != "activity":
                continue
            name: str | None = cond.get("name")
            if name is None:
                continue
            short: str = self._name_to_short_id.get(name, name)
            if short not in seen:
                seen.add(short)
                ids.append(short)
        return ids

    def _refresh_object_cache(self) -> None:
        """Refresh cached lookups of vessels and sites from registry."""
        # Vessels
        self._vessels_by_name.clear()
        for vessel_type in ["TransportProcessingResource", "InstallationAsset"]:
            if vessel_type in self.registry.get("sim_objects", {}):
                self._vessels_by_name.update(self.registry["sim_objects"][vessel_type])

        # Sites
        if "Site" in self.registry.get("sim_objects", {}):
            self._sites_by_name = self.registry["sim_objects"]["Site"]

    # ------------------------------------------------------------------
    # Dependency-event helpers  (intent-based queuing)
    # ------------------------------------------------------------------

    def _collect_dependency_events(
        self, *participant_names: str | None
    ) -> List[Dict[str, Any]]:
        """Build ``start_event`` conditions for all unfinished activities
        involving the given participants.

        For each participant (vessel or site name), every PENDING or ACTIVE
        activity that involves it produces one condition of the form::

            {"type": "activity", "name": "<activity-name>", "state": "done"}

        EventSymphony will keep the new activity PENDING until **all**
        listed predecessor activities reach the ``done`` state.

        Parameters
        ----------
        *participant_names : str | None
            Names of vessels and/or sites that participate in the new
            activity.  ``None`` values are silently skipped.

        Returns
        -------
        List[Dict[str, Any]]
            Possibly-empty list of start_event condition dicts.
        """
        if self._activity_tracker is None:
            return []

        # De-duplicate across participants that share an activity
        seen_activity_names: set[str] = set()
        conditions: List[Dict[str, Any]] = []

        unfinished_states = frozenset(
            {es_core.CurrentActivityState.PENDING, es_core.CurrentActivityState.ACTIVE}
        )

        for name in participant_names:
            if name is None:
                continue

            # Query the tracker — a name can belong to a vessel, a site,
            # or (theoretically) both.  We check vessels first, then sites.
            activity_states: List[ActivityTracker.ActivityState] = []
            if name in self._vessels_by_name:
                activity_states.extend(
                    self._activity_tracker.get_activities_for_vessel(name)
                )
            if name in self._sites_by_name:
                for act in self._activity_tracker.get_activities_for_site(name):
                    # For sites, we only depend on resource-altering activities (Load/Unload).
                    # We do NOT want to block on vessels simply moving to the site.
                    if isinstance(act.activity, es_model.MoveActivity):
                        continue
                    activity_states.append(act)

            for act_state in activity_states:
                if act_state.name in seen_activity_names:
                    continue
                if act_state.state in unfinished_states:
                    seen_activity_names.add(act_state.name)
                    conditions.append(
                        {
                            "type": "activity",
                            "name": act_state.name,
                            "state": "done",
                        }
                    )

        if conditions:
            logger.debug(
                f"Dependency events for participants "
                f"{[n for n in participant_names if n]}: "
                f"{[c['name'] for c in conditions]}"
            )

        return conditions

    @staticmethod
    def _merge_start_events(
        user_events: List[Dict[str, Any]] | None,
        dependency_events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]] | None:
        """Merge user-provided and auto-generated start_event conditions.

        Conditions are de-duplicated by the ``"name"`` field for
        activity-type conditions so that the same predecessor never
        appears twice.

        Parameters
        ----------
        user_events : list or None
            Conditions supplied explicitly on the :class:`ActionSpec`.
        dependency_events : list
            Conditions produced by :meth:`_collect_dependency_events`.

        Returns
        -------
        list or None
            Combined list, or ``None`` when both inputs are empty.
        """
        if not user_events and not dependency_events:
            return None

        combined: List[Dict[str, Any]] = []
        seen_activity_names: set[str] = set()

        for cond in user_events or []:
            combined.append(cond)
            # Track activity-type conditions so deps don't duplicate them
            if cond.get("type") == "activity" and "name" in cond:
                seen_activity_names.add(cond["name"])

        for cond in dependency_events:
            cond_name = cond.get("name")
            if cond_name is not None:
                if cond_name in seen_activity_names:
                    continue
                seen_activity_names.add(cond_name)
            combined.append(cond)

        return combined if combined else None

    # ------------------------------------------------------------------
    # Public build entry point
    # ------------------------------------------------------------------

    def build(
        self,
        spec: ActionSpec | SequentialActionSpec,
    ) -> es_model.GenericActivity:
        """Build an activity from an action specification.

        Parameters
        ----------
        spec : ActionSpec | SequentialActionSpec
            Specification of the action to convert.

        Returns
        -------
        es_model.GenericActivity
            The constructed activity ready to be registered.

        Raises
        ------
        ValueError
            If action_type is unknown or required parameters are missing.
        KeyError
            If referenced vessel/site names don't exist in registry.
        """
        if isinstance(spec, SequentialActionSpec):
            specs = spec.actions
            activities = []
            for spec in specs:
                activity = self._build_action_spec(spec)
                activities.append(activity)

            return self._build_sequential_activity(
                activities, start_event=spec.start_event
            )
        else:
            return self._build_action_spec(spec)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _build_action_spec(self, spec: ActionSpec) -> es_model.GenericActivity:
        vessel = self._vessels_by_name[spec.vessel_name]
        short_id = self._next_activity_id()

        if spec.action_type == ActionType.MOVE:
            activity = self._build_move_activity(vessel, spec, short_id)
        elif spec.action_type == ActionType.LOAD:
            activity = self._build_load_activity(vessel, spec, short_id)
        elif spec.action_type == ActionType.UNLOAD:
            activity = self._build_unload_activity(vessel, spec, short_id)
        elif spec.action_type == ActionType.IDLE:
            activity = self._build_idle_activity(vessel, spec, short_id)
        else:
            raise ValueError(f"Unknown action_type: {spec.action_type}")

        self.register_name(activity.name, short_id)

        # Resolve dependency short IDs from the start_event captured by
        # the builder method (avoids reading opaque ES activity attrs).
        self._last_dep_ids = self._extract_dep_ids(self._last_start_event)

        return activity

    # ------------------------------------------------------------------
    # start_event processing
    # ------------------------------------------------------------------

    def _hygienic_start_event(
        self, start_event: List[Dict[str, Any]] | None
    ) -> List[Dict[str, Any]] | None:
        """Sanitise a ``start_event`` list before handing it to EventSymphony.

        * Conditions with a ``"concept"`` key have their string values
          resolved to the corresponding vessel / site objects.
        * All other condition types (e.g. ``{"type": "activity", ...}``)
          are passed through unchanged.

        Returns ``None`` when the input is ``None`` or empty.
        """
        if start_event is None:
            return None

        def resolve_concept(value: Any) -> Any:
            if isinstance(value, str):
                # Try vessel by name
                if value in self._vessels_by_name:
                    return self._vessels_by_name[value]
                # Try site by name
                if value in self._sites_by_name:
                    return self._sites_by_name[value]
                return value

        processed: List[Dict[str, Any]] = []
        for cond in start_event:
            new_cond = dict(cond)
            if "concept" in new_cond:
                new_cond["concept"] = resolve_concept(new_cond["concept"])
            processed.append(new_cond)

        return processed if processed else None

    def _resolve_start_event(
        self,
        spec: ActionSpec,
        *participant_names: str | None,
    ) -> List[Dict[str, Any]] | None:
        """Build the final ``start_event`` list for an activity.

        Combines three sources (in order):

        1. User-provided conditions from ``spec.start_event``.
        2. Auto-generated dependency conditions from the
           :class:`ActivityTracker` for the given *participant_names*.
        3. Concept-resolution via :meth:`_hygienic_start_event`.

        Parameters
        ----------
        spec : ActionSpec
            The originating action specification (carries user events).
        *participant_names : str | None
            Entity names whose unfinished activities must complete first.

        Returns
        -------
        list or None
            Ready-to-use ``start_event`` for an EventSymphony activity
            constructor, or ``None`` when no conditions apply.
        """
        dependency_events = self._collect_dependency_events(*participant_names)
        merged = self._merge_start_events(spec.start_event, dependency_events)
        return self._hygienic_start_event(merged)

    # ------------------------------------------------------------------
    # Activity builders
    # ------------------------------------------------------------------

    def _build_sequential_activity(
        self,
        activities: List[es_model.GenericActivity],
        name: str | None = None,
        start_event: List[Dict[str, Any]] | None = None,
    ) -> es_model.SequentialActivity:
        if name is None:
            # Build a descriptive, compact name from sub-activity names
            sub_names = [a.name for a in activities]
            joined = " -> ".join(sub_names)
            name = f"Sequential[{joined}]"

        start_event = self._hygienic_start_event(start_event)

        return es_model.SequentialActivity(
            env=self.env,
            name=name,
            registry=self.registry,
            sub_processes=activities,
            start_event=start_event,
        )

    def _maybe_perturb_duration(self, duration_hours: float, cv: float) -> float:
        """Apply stochastic noise to a nominal activity duration.

        Returns ``duration_hours`` unchanged when stochasticity is
        disabled or the activity type's coefficient of variation is
        non-positive, so the deterministic path is fully bypassed (no RNG
        draw) whenever the feature is off.

        Parameters
        ----------
        duration_hours : float
            Nominal (deterministic) duration in hours.
        cv : float
            Coefficient of variation for this activity type's multiplier.

        Returns
        -------
        float
            The (possibly) perturbed duration in hours.
        """
        cfg = self.config.stochasticity
        if not cfg.enabled or cv <= 0.0 or duration_hours <= 0.0:
            return duration_hours

        if self._rng is None:
            # Feature enabled but no seeded generator was wired in; fall
            # back to an unseeded one so we degrade gracefully rather than
            # crash.  The normal env flow always supplies np_random.
            logger.warning(
                "Activity duration stochasticity is enabled but no RNG was "
                "provided; falling back to an unseeded default_rng (episodes "
                "will not be reproducible)."
            )
            self._rng = np.random.default_rng()

        multiplier = self._sample_multiplier(cfg.distribution, cv)
        multiplier = float(np.clip(multiplier, cfg.min_multiplier, cfg.max_multiplier))
        return duration_hours * multiplier

    def _sample_multiplier(self, distribution: str, cv: float) -> float:
        """Draw a duration multiplier with mean 1.0 and the given CV.

        Parameters
        ----------
        distribution : str
            One of ``"lognormal"``, ``"triangular"`` or ``"uniform"``.
        cv : float
            Coefficient of variation (std / mean) of the multiplier.

        Returns
        -------
        float
            A sampled multiplier centred on 1.0.
        """
        assert self._rng is not None  # guaranteed by caller
        rng = self._rng
        dist = distribution.lower()

        if dist == "lognormal":
            # Parametrise so E[X] = 1 and Std[X] / E[X] = cv exactly.
            sigma = float(np.sqrt(np.log1p(cv * cv)))
            mu = -0.5 * sigma * sigma
            return float(rng.lognormal(mean=mu, sigma=sigma))

        if dist == "triangular":
            # Symmetric triangular on [1 - w, 1 + w]; its std is w / sqrt(6),
            # so w = cv * sqrt(6) gives the requested coefficient of variation.
            w = cv * float(np.sqrt(6.0))
            left = max(0.0, 1.0 - w)
            return float(rng.triangular(left, 1.0, 1.0 + w))

        if dist == "uniform":
            # Symmetric uniform on [1 - w, 1 + w]; its std is w / sqrt(3),
            # so w = cv * sqrt(3) gives the requested coefficient of variation.
            w = cv * float(np.sqrt(3.0))
            return float(rng.uniform(max(0.0, 1.0 - w), 1.0 + w))

        raise ValueError(
            f"Unknown stochasticity.distribution '{distribution}'. "
            "Expected 'lognormal', 'triangular' or 'uniform'."
        )

    def _build_move_activity(
        self,
        vessel: Vessel,
        spec: ActionSpec,
        short_id: str,
    ) -> es_model.MoveActivity:
        """Build a move activity.

        Participants: the vessel only.  Arriving at a destination site does
        not require the site to be free.
        """
        if not spec.destination_name:
            raise ValueError("Move action requires destination_name")

        destination = self._sites_by_name[spec.destination_name]

        # Determine origin site to check for duration overrides
        origin_name = "Unknown"
        # Try to find which site the vessel is currently at
        # This assumes the vessel is at a site if it's starting a move
        for name, site in self._sites_by_name.items():
            # Check if geometries match (assuming point locations)
            if hasattr(vessel, "geometry") and hasattr(site, "geometry"):
                if vessel.geometry == site.geometry:
                    origin_name = name
                    break

        # Check for override in move_matrix (symmetric check)
        duration_hours = self.config.move_default

        # Check Origin -> Destination
        if (
            origin_name in self.config.move_matrix
            and destination.name in self.config.move_matrix[origin_name]
        ):
            duration_hours = self.config.move_matrix[origin_name][destination.name]
        # Check Destination -> Origin (Symmetry fallback)
        elif (
            destination.name in self.config.move_matrix
            and origin_name in self.config.move_matrix[destination.name]
        ):
            duration_hours = self.config.move_matrix[destination.name][origin_name]

        duration = duration_hours if spec.duration is None else spec.duration
        duration = self._maybe_perturb_duration(
            duration, self.config.stochasticity.move_cv
        )

        # Resolve start_event: vessel must finish its own prior work
        start_event = self._resolve_start_event(spec, spec.vessel_name)
        self._last_start_event = start_event

        return es_model.MoveActivity(
            env=self.env,
            name=f"{short_id}: {vessel.name} → {destination.name}",
            registry=self.registry,
            mover=vessel,
            destination=destination,
            duration=duration * 3600,  # Convert to seconds
            category="transit",
            start_event=start_event,
        )

    def _build_load_activity(
        self,
        vessel: Vessel,
        spec: ActionSpec,
        short_id: str,
    ) -> es_model.ShiftAmountActivity:
        """Build a load activity.

        When a :class:`ReservationSystem` is attached, fine-grained
        per-item claiming is used for **all** partner types:

        * :meth:`~ReservationSystem.claim_resource` claims a specific
          supply unit at the source (physical stock or future delivery).
        * :meth:`~ReservationSystem.claim_capacity` claims a specific
          free slot on the vessel (physical or freed by a future drain).
        * For vessel partners, operational sequencing dependencies are
          also collected so the partner finishes its current work first.

        When no ``ReservationSystem`` is wired, the legacy site-wide lock
        is used as a safe fallback.
        """
        if not spec.partner_name:
            raise ValueError("Load action requires partner_name")

        # Partner can be either a site or another vessel
        is_site_partner = spec.partner_name in self._sites_by_name
        if is_site_partner:
            origin = self._sites_by_name[spec.partner_name]
        else:
            origin = self._vessels_by_name[spec.partner_name]

        resource_name = spec.resource_name or "default"
        amount = spec.amount or 1

        duration = self.config.load_default if spec.duration is None else spec.duration
        duration = self._maybe_perturb_duration(
            duration, self.config.stochasticity.load_cv
        )

        # --- Dependency resolution ---
        if self._reservation_system is not None:
            # Granular path: fine-grained claims for both supply and capacity.
            vessel_deps = self._collect_dependency_events(spec.vessel_name)

            # Claim 1 unit of resource at the source (supply-side)
            supply_claim = self._reservation_system.claim_resource(
                spec.partner_name, resource_name
            )
            if supply_claim.needs_wait():
                vessel_deps.append(
                    {
                        "type": "activity",
                        "name": supply_claim.activity_name,
                        "state": "done",
                    }
                )

            # Claim 1 free capacity slot on the vessel (capacity-side)
            cap_claim = self._reservation_system.claim_capacity(
                spec.vessel_name, resource_name
            )
            if cap_claim.needs_wait():
                vessel_deps.append(
                    {
                        "type": "activity",
                        "name": cap_claim.activity_name,
                        "state": "done",
                    }
                )

            # For vessel partners, also depend on partner being free
            if not is_site_partner:
                partner_deps = self._collect_dependency_events(spec.partner_name)
                vessel_deps.extend(partner_deps)

            merged = self._merge_start_events(spec.start_event, vessel_deps)
            start_event = self._hygienic_start_event(merged)
        else:
            # Fallback: vessel + partner must both be free (legacy)
            start_event = self._resolve_start_event(
                spec, spec.vessel_name, spec.partner_name
            )

        self._last_start_event = start_event

        return es_model.ShiftAmountActivity(
            env=self.env,
            name=f"{short_id}: Load {vessel.name} ↔ {origin.name} ({resource_name})",
            registry=self.registry,
            processor=vessel,
            origin=origin,
            destination=vessel,
            amount=amount,
            duration=duration * 3600,
            category="loading",
            id_=resource_name,
            start_event=start_event,
        )

    def _build_unload_activity(
        self,
        vessel: Vessel,
        spec: ActionSpec,
        short_id: str,
    ) -> es_model.ShiftAmountActivity:
        """Build an unload activity.

        When a :class:`ReservationSystem` is attached, fine-grained
        per-slot claiming is used for **all** partner types:

        * :meth:`~ReservationSystem.claim_resource` claims a specific
          resource unit from the vessel (physical or future delivery).
        * :meth:`~ReservationSystem.claim_capacity` claims a specific
          free slot at the destination (physical or freed by a future drain).
        * For vessel partners, operational sequencing dependencies are
          also collected so the partner finishes its current work first.

        When no ``ReservationSystem`` is wired, the legacy behaviour is
        preserved: both vessel and partner must be free.

        If the unload is an installation (as defined by
        ``config.installations``) and the current time is inside a
        no-install window, a time-based ``start_event`` is injected so
        that EventSymphony delays the activity until the window closes.
        """
        if not spec.partner_name:
            raise ValueError("Unload action requires partner_name")

        # Partner can be either a site or another vessel
        is_site_partner = spec.partner_name in self._sites_by_name
        if is_site_partner:
            destination = self._sites_by_name[spec.partner_name]
        else:
            destination = self._vessels_by_name[spec.partner_name]

        resource_name = spec.resource_name or "default"
        amount = spec.amount or 1

        duration = (
            self.config.unload_default if spec.duration is None else spec.duration
        )

        if spec.duration is None and vessel.name in self.config.installations:
            load_partners = self.config.installations[vessel.name]

            if hasattr(load_partners, destination.name):
                duration = getattr(load_partners, destination.name)

        duration = self._maybe_perturb_duration(
            duration, self.config.stochasticity.unload_cv
        )

        # --- Dependency resolution ---
        if self._reservation_system is not None:
            # Granular path: fine-grained claims for both supply and capacity.
            vessel_deps = self._collect_dependency_events(spec.vessel_name)

            # Claim 1 unit of resource from the vessel (supply-side)
            supply_claim = self._reservation_system.claim_resource(
                spec.vessel_name, resource_name
            )
            if supply_claim.needs_wait():
                vessel_deps.append(
                    {
                        "type": "activity",
                        "name": supply_claim.activity_name,
                        "state": "done",
                    }
                )

            # Claim 1 free capacity slot at the destination (capacity-side)
            cap_claim = self._reservation_system.claim_capacity(
                spec.partner_name, resource_name
            )
            if cap_claim.needs_wait():
                vessel_deps.append(
                    {
                        "type": "activity",
                        "name": cap_claim.activity_name,
                        "state": "done",
                    }
                )

            # For vessel partners, also depend on partner being free
            if not is_site_partner:
                partner_deps = self._collect_dependency_events(spec.partner_name)
                vessel_deps.extend(partner_deps)

            merged = self._merge_start_events(spec.start_event, vessel_deps)
            start_event = self._hygienic_start_event(merged)
        else:
            # Fallback: vessel + partner vessel must both be free
            start_event = self._resolve_start_event(
                spec, spec.vessel_name, spec.partner_name
            )

        # If this is an installation and we are inside a no-install window,
        # inject a time-gate so EventSymphony delays until the window closes.
        if self._is_installation_unload(vessel.name, destination.name):
            window_event = self._build_no_install_window_start_event()
            if window_event is not None:
                if start_event is None:
                    start_event = [window_event]
                else:
                    start_event.append(window_event)

        self._last_start_event = start_event

        return es_model.ShiftAmountActivity(
            env=self.env,
            name=f"{short_id}: Unload {vessel.name} ↔ {destination.name} ({resource_name})",
            registry=self.registry,
            processor=vessel,
            origin=vessel,
            destination=destination,
            amount=amount,
            duration=duration * 3600,
            category="unloading",
            id_=resource_name,
            start_event=start_event,
        )

    def _build_idle_activity(
        self,
        vessel: Vessel,
        spec: ActionSpec,
        short_id: str,
    ) -> es_model.BasicActivity:
        """Build an idle/wait activity.

        Supports two modes:

        1. **Fixed-duration** (``spec.duration`` is set): The vessel idles
           for a fixed number of seconds.  Legacy mode retained for
           backward compatibility.

        2. **Await-event** (``spec.duration is None``): The vessel waits
           until the next non-idle activity in the DES completes.  This is
           implemented as a zero-duration activity whose ``start_event``
           is an OR-gate over all currently PENDING/ACTIVE non-idle
           activities (excluding those owned solely by this vessel, which
           are already handled by the vessel's own dependency chain).

           .. note::

               The OR-gate is a *snapshot* of queued activities at
               construction time.  Activities registered in future
               macro-steps are not included.  This is acceptable because
               the vessel will wake when any original trigger completes
               and can re-evaluate at that point.
               TODO: Consider a boolean condition event for dynamic
               trigger sets if the snapshot approach proves too coarse.

        Participants: the vessel only.
        """
        if spec.duration is not None:
            # ── Fixed-duration mode (legacy) ──
            duration = spec.duration

            start_event = self._resolve_start_event(spec, spec.vessel_name)
            self._last_start_event = start_event

            return es_model.BasicActivity(
                env=self.env,
                name=f"{short_id}: Idle {vessel.name} ⏸ {duration:.0f}s",
                registry=self.registry,
                additional_logs=[vessel],
                duration=duration,
                category="idle",
                start_event=start_event,
            )

        # ── Await-event mode ──
        # Collect names of all PENDING/ACTIVE non-idle activities that
        # are NOT solely owned by this vessel.
        trigger_names = self._get_idle_trigger_names(vessel.name)

        if not trigger_names:
            # Defensive fallback: if no triggers exist (should not happen
            # because the mask guards against this), use a minimal 1-second
            # idle to avoid an activity that can never start.
            logger.warning(
                f"Await-event idle for {vessel.name} found no trigger "
                f"activities — falling back to 1s fixed idle."
            )
            start_event = self._resolve_start_event(spec, spec.vessel_name)
            self._last_start_event = start_event
            return es_model.BasicActivity(
                env=self.env,
                name=f"{short_id}: Idle {vessel.name} ⏸ 1s (fallback)",
                registry=self.registry,
                additional_logs=[vessel],
                duration=1,
                category="idle",
                start_event=start_event,
            )

        # Build the OR wake-trigger condition
        if len(trigger_names) == 1:
            or_condition: Dict[str, Any] = {
                "type": "activity",
                "state": "done",
                "name": trigger_names[0],
            }
        else:
            or_condition = {
                "or": [
                    {"type": "activity", "state": "done", "name": name}
                    for name in trigger_names
                ]
            }

        # Vessel's own dependency events (must finish own prior work first)
        vessel_deps = self._collect_dependency_events(spec.vessel_name)

        # Merge: [vessel_deps..., or_condition] — all conditions are AND'd
        # by EventSymphony when provided as a list.
        all_conditions = list(vessel_deps) + [or_condition]

        # Merge with any user-provided start_event on the spec
        start_event = self._merge_start_events(spec.start_event, all_conditions)
        start_event = self._hygienic_start_event(start_event)
        self._last_start_event = start_event

        return es_model.BasicActivity(
            env=self.env,
            name=f"{short_id}: Idle {vessel.name} ⏸ await-event",
            registry=self.registry,
            additional_logs=[vessel],
            duration=0,
            category="idle",
            start_event=start_event,
        )

    def _get_idle_trigger_names(self, vessel_name: str) -> List[str]:
        """Return names of activities that can wake an await-event idle.

        Includes all PENDING/ACTIVE activities that are:
        - NOT categorised as "idle" (prevents idle-on-idle deadlock)
        - NOT solely owned by the requesting vessel (those are already
          covered by the vessel's own dependency chain)

        Parameters
        ----------
        vessel_name : str
            The vessel requesting idle triggers.

        Returns
        -------
        List[str]
            Activity names suitable for the OR wake-trigger.
        """
        if self._activity_tracker is None:
            return []

        unfinished = (
            self._activity_tracker.get_active_activities()
            + self._activity_tracker.get_pending_activities()
        )

        trigger_names: List[str] = []
        for act_state in unfinished:
            # Skip idle activities to prevent deadlock
            if getattr(act_state.activity, "category", None) == "idle":
                continue

            # Skip activities owned solely by the requesting vessel
            vessel_names_in_activity = {v.name for v in act_state.vessels.values()}
            if vessel_names_in_activity == {vessel_name}:
                continue

            trigger_names.append(act_state.name)

        return trigger_names

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_move_duration(self, hours: float) -> None:
        """Set the duration for move activities.

        Parameters
        ----------
        hours : float
            Duration in hours for move activities.
        """
        self.config.move_default = hours
        logger.debug(f"Move duration set to {hours} hours")

    def set_load_duration(self, hours: float) -> None:
        """Set the duration for load activities.

        Parameters
        ----------
        hours : float
            Duration in hours for load activities.
        """
        self.config.load_default = hours
        logger.debug(f"Load duration set to {hours} hours")

    def set_unload_duration(self, hours: float) -> None:
        """Set the duration for unload activities.

        Parameters
        ----------
        hours : float
            Duration in hours for unload activities.
        """
        self.config.unload_default = hours
        logger.debug(f"Unload duration set to {hours} hours")

    def compute_move_duration_from_distance(
        self,
        vessel: Vessel,
        destination: Site,
        speed_units_per_hour: float = 10.0,
    ) -> float:
        """Compute move duration based on distance (for future use).

        Parameters
        ----------
        vessel : Vessel
            The vessel that will move.
        destination : Site
            The destination site.
        speed_units_per_hour : float
            Speed of the vessel in distance units per hour.

        Returns
        -------
        float
            Duration in hours.

        Notes
        -----
        This method can be used in the future to dynamically compute
        move durations based on vessel position and destination.
        Currently returns the configured default duration.
        """
        # TODO: Implement actual distance calculation
        # distance = vessel.geometry.distance(destination.geometry)
        # return distance / speed_units_per_hour

        # For now, return configured duration
        return self.config.move_default
