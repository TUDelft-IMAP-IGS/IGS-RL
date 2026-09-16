"""Goal tracking for monitoring simulation objectives and completion status.

This module provides classes for tracking goals based on simulation state
(container levels) and deadlines.
"""

import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from loguru import logger

from eos.config import GoalConfig

from .types import Site, Vessel


@dataclass
class GoalState:
    """Runtime state of a goal.

    Parameters
    ----------
    config : GoalConfig
        The goal configuration (location, resource, quantity, deadline hours offset).
    completed : bool
        Whether the goal has been achieved.
    failed : bool
        Whether the goal has failed (e.g., deadline missed).
    completion_time_posix : float | None
        POSIX timestamp at completion.
    deadline_timestamp : float | None
        Absolute POSIX timestamp computed from simulation start + deadline hours*3600.
    """

    config: GoalConfig
    completed: bool = False
    failed: bool = False
    completion_time_posix: float | None = None
    deadline_timestamp: float | None = None
    last_level: int = 0
    last_step_delta: int = 0


class GoalTracker:
    """Tracks all goals and their completion status in the simulation.

    This class maintains the state of all goals, monitors their progress
    by checking container levels, and provides information for observation
    and termination logic.

    Parameters
    ----------
    goal_configs : List[GoalConfig]
        List of goal configurations to track. The `deadline` field in each
        goal is interpreted as an offset in hours from the start of the
        simulation. A goal fails when the elapsed simulation time (in hours)
        exceeds this offset.
    simulation_start : datetime.datetime
        The start of the simulation
    sites_by_name : Dict[str, Site]
        Mapping from site name to site object.
    vessels_by_name : Dict[str, Vessel]
        Mapping from vessel name to vessel object.
    """

    def __init__(
        self,
        goal_configs: List[GoalConfig],
        simulation_start: datetime.datetime,
        sites_by_name: Dict[str, Site],
        vessels_by_name: Dict[str, Vessel],
    ):
        self.simulation_start = simulation_start
        self._sites_by_name = sites_by_name
        self._vessels_by_name = vessels_by_name
        self._completed_this_step: Set[int] = set()
        self._last_step_reward: float = 0.0

        # Precompute deadline absolute timestamps
        start_ts = simulation_start.timestamp()
        self.goals: List[GoalState] = []
        for cfg in goal_configs:
            state = GoalState(config=cfg)
            if cfg.deadline is not None:
                # cfg.deadline is hours offset
                state.deadline_timestamp = start_ts + cfg.deadline * 3600.0
            state.last_level = self._get_current_level(cfg.location, cfg.resource_type)
            self.goals.append(state)

    def _get_location(self, location_name: str) -> Site | Vessel | None:
        return self._sites_by_name.get(location_name) or self._vessels_by_name.get(
            location_name
        )

    def _get_current_level(self, location_name: str, resource: str) -> int:
        location = self._get_location(location_name)
        if not location:
            logger.error(f"Goal location '{location_name}' not found")
            return 0
        try:
            return int(location.container.get_level(resource))
        except Exception as e:
            logger.error(f"Error checking goal level: {e}")
            return 0

    def check_goals(self, current_posix_time: float) -> None:
        """Check all goals against the current simulation state.

        Parameters
        ----------
        current_posix_time : float
            Current POSIX timestamp from the simulation environment.
        """
        self._last_step_reward = 0.0
        for i, goal in enumerate(self.goals):
            if goal.completed or goal.failed:
                continue

            # Deadline check using absolute timestamp
            if (
                goal.deadline_timestamp is not None
                and current_posix_time > goal.deadline_timestamp
            ):
                goal.failed = True
                logger.debug(
                    f"Goal failed: Deadline missed for "
                    f"{goal.config.resource_type}@{goal.config.location}"
                )
                continue

            # Check container level
            try:
                current_level = self._get_current_level(
                    goal.config.location, goal.config.resource_type
                )
                prev_capped = min(goal.last_level, goal.config.quantity)
                curr_capped = min(current_level, goal.config.quantity)
                delta = max(0, curr_capped - prev_capped)
                goal.last_step_delta = delta
                goal.last_level = current_level
                if delta > 0:
                    self._last_step_reward += delta * goal.config.reward_per_unit

                if current_level >= goal.config.quantity:
                    goal.completed = True
                    goal.completion_time_posix = current_posix_time
                    self._completed_this_step.add(i)
                    elapsed_hours = (
                        current_posix_time - self.simulation_start.timestamp()
                    ) / 3600.0
                    logger.debug(
                        f"Goal completed: {goal.config.quantity} "
                        f"{goal.config.resource_type}@{goal.config.location} "
                        f"(t={elapsed_hours:.1f}h)"
                    )
            except Exception as e:
                logger.error(f"Error checking goal: {e}")

    def clear_completed_this_step(self) -> None:
        """Clear the set of goals completed in the current step."""
        self._completed_this_step.clear()

    def get_completed_this_step(self) -> List[GoalState]:
        """Get goals completed in the current step.

        Returns
        -------
        List[GoalState]
            Goals completed this step.
        """
        return [self.goals[i] for i in self._completed_this_step]

    def all_goals_completed(self) -> bool:
        """Check if all goals are completed.

        Returns
        -------
        bool
            True if all goals are completed.
        """
        return all(goal.completed for goal in self.goals)

    def any_goal_failed(self) -> bool:
        """Check if any goal has failed.

        Returns
        -------
        bool
            True if any goal is failed.
        """
        return any(goal.failed for goal in self.goals)

    def get_goal_summary(self) -> Dict[str, int]:
        """Get summary counts of goals by status.

        Returns
        -------
        Dict[str, int]
            Counts of goals in each status category.
        """
        total = len(self.goals)
        completed = sum(1 for g in self.goals if g.completed)
        failed = sum(1 for g in self.goals if g.failed)
        remaining = total - completed - failed

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "remaining": remaining,
        }

    def get_last_step_reward(self) -> float:
        return self._last_step_reward

    def get_last_step_progress(self) -> List[Dict[str, Any]]:
        return [
            {
                "location": goal.config.location,
                "resource_type": goal.config.resource_type,
                "quantity": goal.config.quantity,
                "reward_per_unit": goal.config.reward_per_unit,
                "last_step_delta": goal.last_step_delta,
            }
            for goal in self.goals
        ]

    def is_goal_blocked(self, goal_index: int) -> bool:
        """Check whether a goal is blocked by incomplete prerequisite goals.

        A goal is blocked if any of its prerequisite goals (referenced by
        index in ``depends_on``) are not yet completed.

        Parameters
        ----------
        goal_index : int
            Index of the goal to check.

        Returns
        -------
        bool
            True if any dependency goal is not completed, False otherwise.
            Returns False when the goal has no dependencies.
        """
        depends_on = self.goals[goal_index].config.depends_on
        if not depends_on:
            return False
        return any(not self.goals[dep].completed for dep in depends_on)

    def get_goal_dep_progress(self, goal_index: int) -> float:
        """Return the minimum progress across all prerequisite goals.

        Progress of a single dependency goal is defined as
        ``min(last_level, quantity) / quantity`` when ``quantity > 0``,
        otherwise ``1.0``.

        Parameters
        ----------
        goal_index : int
            Index of the goal whose dependency progress to compute.

        Returns
        -------
        float
            Minimum progress value across all dependencies, or ``1.0`` if
            the goal has no dependencies.
        """
        depends_on = self.goals[goal_index].config.depends_on
        if not depends_on:
            return 1.0
        return min(
            (
                min(self.goals[dep].last_level, self.goals[dep].config.quantity)
                / self.goals[dep].config.quantity
                if self.goals[dep].config.quantity > 0
                else 1.0
            )
            for dep in depends_on
        )

    def has_unblocked_goals_for(self, site_name: str, resource_type: str) -> bool:
        """Check if any unblocked, actionable goal exists for a site and resource.

        A goal is considered actionable when it is not completed, not failed,
        and not blocked by incomplete dependencies.  This is used by the
        installation masking logic to decide whether an UNLOAD/install action
        should be allowed at a given site for a given resource type.

        Parameters
        ----------
        site_name : str
            Name of the site to check.
        resource_type : str
            Resource type to check.

        Returns
        -------
        bool
            True if at least one matching goal is not completed, not failed,
            and not blocked.
        """
        for i, goal in enumerate(self.goals):
            if (
                goal.config.location == site_name
                and goal.config.resource_type == resource_type
                and not goal.completed
                and not goal.failed
                and not self.is_goal_blocked(i)
            ):
                return True
        return False

    def get_goal_dependency_info(self, goal_index: int) -> Dict[str, Any]:
        """Return dependency state information for a goal.

        This is used by the simulator's ``get_goal_states()`` to populate
        the observation with dependency metadata.

        Parameters
        ----------
        goal_index : int
            Index of the goal to query.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing ``is_blocked``, ``dep_progress``, and
            ``depends_on`` for the specified goal.
        """
        return {
            "is_blocked": self.is_goal_blocked(goal_index),
            "dep_progress": self.get_goal_dep_progress(goal_index),
            "depends_on": list(self.goals[goal_index].config.depends_on),
        }

    def get_goals_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about all goals.

        This is useful for constructing observation spaces in RL environments.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing goal information.
        """
        return [
            {
                "location": goal.config.location,
                "resource_type": goal.config.resource_type,
                "quantity": goal.config.quantity,
                "reward_per_unit": goal.config.reward_per_unit,
                "deadline": goal.config.deadline,
                "completed": goal.completed,
                "failed": goal.failed,
                "completion_time_posix": goal.completion_time_posix,
                "deadline_timestamp": goal.deadline_timestamp,
                "last_level": goal.last_level,
                "last_step_delta": goal.last_step_delta,
            }
            for goal in self.goals
        ]
