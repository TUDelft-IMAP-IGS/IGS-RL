"""Reservation system for tracking projected states from the DES.

This module provides a centralized system that reads the current DES state
(including both ACTIVE and PENDING activities) and projects the final
inventories and locations of all entities at the end of their queues.

Because the environment micro-steps (registers actions directly into the
DES as PENDING activities mid-step), this system does not need to track
"chosen actions" or "blocked partners". The DES is the single source of truth.

The reservation system supports two modes of operation:

1. **Full rebuild** – ``from_simulation_state()`` scans all PENDING and
   ACTIVE activities in the DES and builds the projected state from
   scratch.  Safe but expensive; useful as a parity-check baseline.

2. **Incremental** – ``apply_action_spec()`` and ``revert_action_spec()``
   add / remove the effects of a single ``ActionSpec`` without rescanning
   the entire DES.  The simulator calls these at registration time (micro-
   step) and at completion time (step) respectively.

Granular Resource Tracking
--------------------------
In addition to the aggregate ``_drains`` / ``_fills`` counters (which drive
action masking via ``get_total_drain`` / ``get_total_fill``), the system
maintains **item-level** tracking structures that enable fine-grained
dependency resolution:

* ``_pending_deliveries`` – ordered queue of activity names that will
  deliver a resource unit to a ``(site, resource)`` pair.
* ``_claimed_physical`` – count of on-dock physical units already
  earmarked for a specific pending LOAD.
* ``_ledger_levels`` – shadow ledger of container levels, seeded once
  at construction and maintained purely from RS completion events.
  Never reads DES containers directly.

These structures are populated alongside the existing aggregate layer and
will be consumed by the ``ActivityBuilder`` in Phase B to replace site-wide
locks with per-item dependencies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, Dict, List, Tuple

from loguru import logger

from .activity_builder import ActionSpec
from .activity_tracker import ActivityTracker
from .types import ActionType

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ResourceCommitment:
    """A single committed resource transfer on a simulation object."""

    object_name: str
    resource_name: str
    amount: int
    source: str = ""


class ClaimResult:
    """Result of a :meth:`ReservationSystem.claim_resource` call.

    Either the resource was claimed immediately from physical on-dock
    stock, or the caller must wait for a specific delivery activity to
    complete before the resource becomes available.
    """

    class Kind(Enum):
        IMMEDIATE = auto()
        WAIT_FOR = auto()

    def __init__(self, kind: Kind, activity_name: str | None = None) -> None:
        self.kind = kind
        self.activity_name = activity_name  # set only for WAIT_FOR

    def is_immediate(self) -> bool:
        return self.kind == ClaimResult.Kind.IMMEDIATE

    def needs_wait(self) -> bool:
        return self.kind == ClaimResult.Kind.WAIT_FOR

    def __repr__(self) -> str:
        if self.needs_wait():
            return f"ClaimResult(WAIT_FOR, {self.activity_name!r})"
        return "ClaimResult(IMMEDIATE)"


# ---------------------------------------------------------------------------
# ReservationSystem
# ---------------------------------------------------------------------------


class ReservationSystem:
    """Calculates projected locations and resource commitments from the DES queue."""

    def __init__(self) -> None:
        # (object_name, resource_name) -> cumulative amount
        self._drains: Dict[Tuple[str, str], int] = {}
        self._fills: Dict[Tuple[str, str], int] = {}

        # vessel_name -> site_name
        self._projected_locations: Dict[str, str] = {}

        # Detailed ledger for debugging / introspection
        self._drain_ledger: List[ResourceCommitment] = []
        self._fill_ledger: List[ResourceCommitment] = []

        # --- Granular resource tracking (Phase A) -------------------------
        # Ordered queue of DES activity names that will deliver 1 unit of a
        # resource to a (site, resource) pair.  Entries are appended on
        # UNLOAD / fabrication registration and consumed (popped) when a
        # LOAD claims a pending delivery via ``claim_resource()``.
        self._pending_deliveries: Dict[Tuple[str, str], Deque[str]] = {}

        # Count of physical (already on-dock) units at a (site, resource)
        # pair that have been earmarked for a pending LOAD.  Incremented
        # by ``claim_resource()`` (immediate path), decremented by
        # ``complete_load_claim()`` when the LOAD finishes.
        self._claimed_physical: Dict[Tuple[str, str], int] = {}

        # Shadow ledger: maintained purely from RS events, never reads DES containers.
        # Represents the logical level as if all transfers are atomic at completion.
        self._ledger_levels: Dict[Tuple[str, str], int] = {}
        self._ledger_free_slots: Dict[Tuple[str, str], int] = {}
        self._capacities: Dict[Tuple[str, str], int] = {}

        # --- Granular capacity tracking ---------------------------------
        # Ordered queue of DES activity names that will FREE 1 unit of
        # capacity at (object, resource) by draining from it.  Entries are
        # appended when a LOAD or UNLOAD drains the object and consumed
        # (popleft) when a later deposit claims a future free slot via
        # ``claim_capacity()``.
        self._pending_capacity_releases: Dict[Tuple[str, str], Deque[str]] = {}

        # Count of physical free slots at (object, resource) already
        # earmarked by a pending deposit.  Incremented by
        # ``claim_capacity()`` (immediate path), decremented by
        # ``complete_deposit_claim()`` when the deposit finishes.
        self._claimed_free_slots: Dict[Tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    # Mutation – Spatial Projections
    # ------------------------------------------------------------------

    def set_projected_location(self, vessel_name: str, site_name: str) -> None:
        """Override the projected location for a vessel."""
        self._projected_locations[vessel_name] = site_name

    def reset_projected_location(
        self, vessel_name: str, physical_site_name: str
    ) -> None:
        """Reset the projected location for a vessel back to its physical site.

        This is used when a completed activity's projection should be
        replaced by the vessel's actual current physical location.

        Parameters
        ----------
        vessel_name : str
            Name of the vessel whose projected location to reset.
        physical_site_name : str
            The vessel's current physical site name (from geometry lookup).
        """
        self._projected_locations[vessel_name] = physical_site_name

    # ------------------------------------------------------------------
    # Mutation – Resource Commitments
    # ------------------------------------------------------------------

    def reserve_drain(
        self, object_name: str, resource_name: str, amount: int, source: str = ""
    ) -> None:
        """Record a committed outflow (resource leaving *object_name*)."""
        if amount <= 0:
            logger.warning(f"Received negative amount for drain. amount={amount}")
            return
        key = (object_name, resource_name)
        self._drains[key] = self._drains.get(key, 0) + amount
        self._drain_ledger.append(
            ResourceCommitment(object_name, resource_name, amount, source)
        )

    def reserve_fill(
        self, object_name: str, resource_name: str, amount: int, source: str = ""
    ) -> None:
        """Record a committed inflow (resource arriving at *object_name*)."""
        if amount <= 0:
            logger.warning(f"Received negative amount for fill. amount={amount}")
            return
        key = (object_name, resource_name)
        self._fills[key] = self._fills.get(key, 0) + amount
        self._fill_ledger.append(
            ResourceCommitment(object_name, resource_name, amount, source)
        )

    def _unreserve_drain(
        self, object_name: str, resource_name: str, amount: int
    ) -> None:
        """Remove a previously recorded outflow (internal helper for revert)."""
        if amount <= 0:
            return
        key = (object_name, resource_name)
        current = self._drains.get(key, 0)
        new_val = current - amount
        if new_val < 0:
            logger.warning(
                f"Drain underflow while reverting: {object_name}:{resource_name} "
                f"current={current}, reverting={amount}. Clamping to 0."
            )
            new_val = 0
        if new_val == 0:
            self._drains.pop(key, None)
        else:
            self._drains[key] = new_val

    def _unreserve_fill(
        self, object_name: str, resource_name: str, amount: int
    ) -> None:
        """Remove a previously recorded inflow (internal helper for revert)."""
        if amount <= 0:
            return
        key = (object_name, resource_name)
        current = self._fills.get(key, 0)
        new_val = current - amount
        if new_val < 0:
            logger.warning(
                f"Fill underflow while reverting: {object_name}:{resource_name} "
                f"current={current}, reverting={amount}. Clamping to 0."
            )
            new_val = 0
        if new_val == 0:
            self._fills.pop(key, None)
        else:
            self._fills[key] = new_val

    # ------------------------------------------------------------------
    # Granular Resource Tracking – Registration
    # ------------------------------------------------------------------

    def register_pending_delivery(
        self, site_name: str, resource_name: str, activity_name: str
    ) -> None:
        """Record that *activity_name* will deliver 1 unit to *site_name*.

        Called when an UNLOAD (vessel → site) or a fabrication activity is
        registered.  The delivery sits in a FIFO queue until either:

        * A LOAD claims it via :meth:`claim_resource` (popped from queue).
        * The delivery completes unclaimed and converts to physical stock
          via :meth:`complete_delivery` (removed from queue).

        Parameters
        ----------
        site_name : str
            The destination site receiving the resource.
        resource_name : str
            The resource type being delivered.
        activity_name : str
            The DES activity name producing this delivery.
        """
        key = (site_name, resource_name)
        if key not in self._pending_deliveries:
            self._pending_deliveries[key] = deque()
        self._pending_deliveries[key].append(activity_name)
        logger.debug(
            f"Registered pending delivery: {activity_name} → "
            f"{site_name}:{resource_name} "
            f"(queue depth: {len(self._pending_deliveries[key])})"
        )

    def register_pending_capacity_release(
        self, object_name: str, resource_name: str, activity_name: str
    ) -> None:
        """Record that *activity_name* will free 1 capacity slot at *object_name*.

        Called when a LOAD or UNLOAD drains a resource from an object,
        freeing a capacity slot when the activity completes.  The release
        sits in a FIFO queue until either:

        * A deposit claims it via :meth:`claim_capacity` (popped).
        * The drain completes unclaimed and the slot becomes physical
          free capacity via :meth:`complete_capacity_release`.
        """
        key = (object_name, resource_name)
        if key not in self._pending_capacity_releases:
            self._pending_capacity_releases[key] = deque()
        self._pending_capacity_releases[key].append(activity_name)
        logger.debug(
            f"Registered pending capacity release: {activity_name} → "
            f"{object_name}:{resource_name} "
            f"(queue depth: {len(self._pending_capacity_releases[key])})"
        )

    def claim_resource(self, site_name: str, resource_name: str) -> ClaimResult:
        """Attempt to claim 1 unit of *resource_name* at *site_name*.

        Checks unclaimed physical stock first (fast path).  If none is
        available, pops the earliest pending delivery from the FIFO queue
        (slow path — the caller must inject a dependency on that activity).

        This method is intended to be called by the ``ActivityBuilder``
        when constructing a LOAD activity during the micro-step phase.
        Because micro-steps execute sequentially (AEC ordering) with the
        DES clock frozen, there are no race conditions.

        Returns
        -------
        ClaimResult
            ``IMMEDIATE`` if claimed from physical stock.
            ``WAIT_FOR(activity_name)`` if must wait for a delivery.

        Raises
        ------
        RuntimeError
            If there is neither physical stock nor a pending delivery.
            This should never happen — action masking prevents it.
        """
        key = (site_name, resource_name)
        claimed = self._claimed_physical.get(key, 0)
        physical = self._ledger_levels.get(key, 0)
        unclaimed = physical - claimed

        # Fast path: claim from on-dock stock
        if unclaimed > 0:
            self._claimed_physical[key] = claimed + 1
            logger.debug(
                f"Claimed physical unit at {site_name}:{resource_name} "
                f"(claimed: {claimed + 1}/{physical})"
            )
            return ClaimResult(ClaimResult.Kind.IMMEDIATE)

        # Slow path: wait for the earliest pending delivery.
        # We still increment _claimed_physical so that the compensated
        # snapshot (which unconditionally adds back drained amounts for
        # ACTIVE transfers) is properly offset.  Without this, a later
        # agent would see unclaimed = compensated(1) - claimed(0) = 1
        # and double-book a phantom unit.
        queue = self._pending_deliveries.get(key)
        if queue:
            delivery_activity = queue.popleft()
            self._claimed_physical[key] = claimed + 1
            logger.debug(
                f"Claimed pending delivery {delivery_activity} at "
                f"{site_name}:{resource_name} "
                f"(claimed: {claimed + 1}/{physical}, "
                f"remaining in queue: {len(queue)})"
            )
            return ClaimResult(
                ClaimResult.Kind.WAIT_FOR, activity_name=delivery_activity
            )

        raise RuntimeError(
            f"No resource available to claim at {site_name}:{resource_name}. "
            f"physical={physical}, claimed={claimed}, "
            f"pending_deliveries={list(self._pending_deliveries.get(key, []))}. "
            f"This indicates an action masking bug."
        )

    def claim_capacity(self, object_name: str, resource_name: str) -> ClaimResult:
        """Attempt to claim 1 free capacity slot at *object_name*.

        Checks unclaimed physical free slots first (fast path).  If none
        is available, pops the earliest pending capacity release from the
        FIFO queue (slow path — the caller must inject a dependency on
        that drain activity).

        Returns
        -------
        ClaimResult
            ``IMMEDIATE`` if a physical free slot exists.
            ``WAIT_FOR(activity_name)`` if must wait for a drain.

        Raises
        ------
        RuntimeError
            If there is neither a physical free slot nor a pending release.
        """
        key = (object_name, resource_name)
        claimed = self._claimed_free_slots.get(key, 0)
        free = self._ledger_free_slots.get(key, 0)
        unclaimed = free - claimed

        # Fast path: claim a physical free slot
        if unclaimed > 0:
            self._claimed_free_slots[key] = claimed + 1
            logger.debug(
                f"Claimed physical free slot at {object_name}:{resource_name} "
                f"(claimed: {claimed + 1}/{free})"
            )
            return ClaimResult(ClaimResult.Kind.IMMEDIATE)

        # Slow path: wait for the earliest pending capacity release.
        # We still increment _claimed_free_slots so that the compensated
        # snapshot is properly offset — same rationale as claim_resource.
        queue = self._pending_capacity_releases.get(key)
        if queue:
            release_activity = queue.popleft()
            self._claimed_free_slots[key] = claimed + 1
            logger.debug(
                f"Claimed pending capacity release {release_activity} at "
                f"{object_name}:{resource_name} "
                f"(claimed: {claimed + 1}/{free}, "
                f"remaining in queue: {len(queue)})"
            )
            return ClaimResult(
                ClaimResult.Kind.WAIT_FOR, activity_name=release_activity
            )

        raise RuntimeError(
            f"No capacity available to claim at {object_name}:{resource_name}. "
            f"physical_free={free}, claimed_free={claimed}, "
            f"pending_releases={list(self._pending_capacity_releases.get(key, []))}. "
            f"This indicates an action masking bug."
        )

    # ------------------------------------------------------------------
    # Granular Resource Tracking – Completion Cleanup
    # ------------------------------------------------------------------

    def complete_delivery(
        self, site_name: str, resource_name: str, activity_name: str
    ) -> None:
        """Called when an UNLOAD or fabrication activity completes.

        If the delivery was already claimed by a LOAD (i.e. it was popped
        from the queue by :meth:`claim_resource`), this is a no-op on the
        queue — the claiming LOAD's dependency has resolved naturally.

        If the delivery is still in the queue (no LOAD claimed it yet),
        it is removed.  The resource has now physically arrived and becomes
        unclaimed physical stock; the shadow ledger is updated by
        ``ledger_apply_transfer()``.

        Parameters
        ----------
        site_name : str
            The site that received the delivery.
        resource_name : str
            The resource type delivered.
        activity_name : str
            The DES activity name of the completed delivery.
        """
        key = (site_name, resource_name)
        queue = self._pending_deliveries.get(key)
        if queue is not None:
            try:
                queue.remove(activity_name)
                logger.debug(
                    f"Completed unclaimed delivery {activity_name} at "
                    f"{site_name}:{resource_name} — now physical stock "
                    f"(remaining in queue: {len(queue)})"
                )
            except ValueError:
                # Already claimed by a LOAD → no queue cleanup needed.
                logger.debug(
                    f"Completed delivery {activity_name} at "
                    f"{site_name}:{resource_name} — was already claimed"
                )
            # Clean up empty queues
            if not queue:
                del self._pending_deliveries[key]

    def complete_load_claim(self, site_name: str, resource_name: str) -> None:
        """Called when a LOAD completes.  Decrements the claimed physical count.

        When a LOAD finishes, the resource has physically left the site
        (moved to the vessel).  If this LOAD had claimed a physical unit
        (immediate path), the earmark is released.

        Parameters
        ----------
        site_name : str
            The site from which the resource was loaded.
        resource_name : str
            The resource type loaded.
        """
        key = (site_name, resource_name)
        current = self._claimed_physical.get(key, 0)
        if current > 0:
            new_val = current - 1
            if new_val == 0:
                self._claimed_physical.pop(key, None)
            else:
                self._claimed_physical[key] = new_val
            logger.debug(
                f"Released physical claim at {site_name}:{resource_name} "
                f"(remaining claims: {new_val})"
            )
        else:
            # Both IMMEDIATE and WAIT_FOR paths now increment
            # _claimed_physical, so reaching here means the counter
            # was already fully released (e.g. double-complete guard).
            logger.debug(
                f"LOAD completed at {site_name}:{resource_name} — "
                f"no physical claim to release (already fully released)"
            )

    def complete_capacity_release(
        self, object_name: str, resource_name: str, activity_name: str
    ) -> None:
        """Called when a drain activity completes, freeing a capacity slot.

        If already claimed by a deposit (popped from queue by
        :meth:`claim_capacity`), this is a no-op on the queue.
        If still in the queue, it is removed — the slot is now physically free.
        """
        key = (object_name, resource_name)
        queue = self._pending_capacity_releases.get(key)
        if queue is not None:
            try:
                queue.remove(activity_name)
                logger.debug(
                    f"Completed unclaimed capacity release {activity_name} at "
                    f"{object_name}:{resource_name} — now physical free slot "
                    f"(remaining in queue: {len(queue)})"
                )
            except ValueError:
                logger.debug(
                    f"Completed capacity release {activity_name} at "
                    f"{object_name}:{resource_name} — was already claimed"
                )
            if not queue:
                del self._pending_capacity_releases[key]

    def complete_deposit_claim(self, object_name: str, resource_name: str) -> None:
        """Called when a deposit (LOAD/UNLOAD fill) completes.

        Decrements ``_claimed_free_slots`` if the deposit had an immediate
        physical slot claim.
        """
        key = (object_name, resource_name)
        current = self._claimed_free_slots.get(key, 0)
        if current > 0:
            new_val = current - 1
            if new_val == 0:
                self._claimed_free_slots.pop(key, None)
            else:
                self._claimed_free_slots[key] = new_val
            logger.debug(
                f"Released free-slot claim at {object_name}:{resource_name} "
                f"(remaining claims: {new_val})"
            )
        else:
            # Both IMMEDIATE and WAIT_FOR paths now increment
            # _claimed_free_slots, so reaching here means the counter
            # was already fully released (e.g. double-complete guard).
            logger.debug(
                f"Deposit completed at {object_name}:{resource_name} — "
                f"no free-slot claim to release (already fully released)"
            )

    # ------------------------------------------------------------------
    # Shadow Ledger – Seeding & Transfer
    # ------------------------------------------------------------------

    def seed_ledger(
        self,
        initial_levels: Dict[Tuple[str, str], int],
        capacities: Dict[Tuple[str, str], int],
    ) -> None:
        """Seed the shadow ledger with known initial container state.

        Must be called once after construction (or on episode reset) to
        establish the baseline.  After seeding, the ledger is maintained
        purely by ``revert_action_spec()`` — no DES container reads needed.

        Parameters
        ----------
        initial_levels : dict[(object_name, resource_name), int]
            Current container levels for all objects and resources.
        capacities : dict[(object_name, resource_name), int]
            Maximum capacity for all objects and resources.
        """
        self._capacities = dict(capacities)
        self._ledger_levels = dict(initial_levels)
        self._ledger_free_slots = {
            k: cap - initial_levels.get(k, 0) for k, cap in capacities.items()
        }

    def ledger_apply_transfer(
        self,
        origin_name: str | None,
        destination_name: str | None,
        resource_name: str,
        amount: int,
    ) -> None:
        """Update the shadow ledger to reflect a completed physical transfer.

        Called when an activity completes and the DES has physically moved
        the resource.  This keeps the ledger in sync without reading DES
        container state.

        Parameters
        ----------
        origin_name : str | None
            Object that lost the resource (None if fabrication/external).
        destination_name : str | None
            Object that gained the resource (None if consumed/external).
        resource_name : str
            The resource type transferred.
        amount : int
            Number of units transferred.
        """
        if amount <= 0:
            return
        if origin_name is not None:
            key = (origin_name, resource_name)
            self._ledger_levels[key] = self._ledger_levels.get(key, 0) - amount
            self._ledger_free_slots[key] = self._ledger_free_slots.get(key, 0) + amount
        if destination_name is not None:
            key = (destination_name, resource_name)
            self._ledger_levels[key] = self._ledger_levels.get(key, 0) + amount
            self._ledger_free_slots[key] = self._ledger_free_slots.get(key, 0) - amount

    # ------------------------------------------------------------------
    # Incremental Mutation – ActionSpec-based
    # ------------------------------------------------------------------

    def apply_action_spec(
        self, spec: ActionSpec, activity_name: str | None = None
    ) -> None:
        """Apply the projected effects of an ``ActionSpec`` to the reservation state.

        This is called when a new action is **registered** (micro-step).
        It updates projected locations, resource commitments, and granular
        delivery tracking based on what the action *will* do once it
        executes.

        Parameters
        ----------
        spec : ActionSpec
            The action specification being registered.
        activity_name : str | None
            The DES activity name assigned by the ``ActivityBuilder``.
            Required for UNLOAD actions to register pending deliveries.
            May be ``None`` for legacy / test usage.
        """
        source = f"spec:{spec.action_type.value}:{spec.vessel_name}"

        if spec.action_type == ActionType.MOVE:
            # A move changes the vessel's projected location to the destination
            if spec.destination_name is not None:
                self.set_projected_location(spec.vessel_name, spec.destination_name)

        elif spec.action_type == ActionType.LOAD:
            # LOAD: resource flows FROM partner TO vessel
            resource = spec.resource_name
            amount = spec.amount
            if resource is not None and amount is not None and amount > 0:
                partner = spec.partner_name
                if partner is not None:
                    # Partner (site or vessel) loses resource
                    self.reserve_drain(partner, resource, amount, source=source)
                    # --- Granular: the drain on partner frees capacity ---
                    if activity_name is not None:
                        self.register_pending_capacity_release(
                            partner, resource, activity_name
                        )
                # Vessel gains resource
                self.reserve_fill(spec.vessel_name, resource, amount, source=source)
                # --- Granular: the vessel receives a delivery (for future UNLOAD claims) ---
                if activity_name is not None:
                    self.register_pending_delivery(
                        spec.vessel_name, resource, activity_name
                    )

        elif spec.action_type == ActionType.UNLOAD:
            # UNLOAD: resource flows FROM vessel TO partner
            resource = spec.resource_name
            amount = spec.amount
            if resource is not None and amount is not None and amount > 0:
                # Vessel loses resource
                self.reserve_drain(spec.vessel_name, resource, amount, source=source)
                # --- Granular: the drain on vessel frees capacity ---
                if activity_name is not None:
                    self.register_pending_capacity_release(
                        spec.vessel_name, resource, activity_name
                    )
                partner = spec.partner_name
                if partner is not None:
                    # Partner (site or vessel) gains resource
                    self.reserve_fill(partner, resource, amount, source=source)

                    # --- Granular: register as a pending delivery ---
                    if activity_name is not None:
                        self.register_pending_delivery(partner, resource, activity_name)

        # ActionType.IDLE has no projected effects

    def revert_action_spec(
        self, spec: ActionSpec, activity_name: str | None = None
    ) -> None:
        """Revert the projected effects of an ``ActionSpec`` from the reservation state.

        This is called when an activity **completes** (during step).  The
        DES has already applied the real effects to the simulation objects
        so the projections are no longer needed and must be removed.

        After reverting the resource commitments, the caller is responsible
        for resetting the vessel's projected location to its current
        physical site via :meth:`reset_projected_location`.

        Parameters
        ----------
        spec : ActionSpec
            The action specification whose effects should be reverted.
        activity_name : str | None
            The DES activity name, used for granular delivery cleanup.
        """
        if spec.action_type == ActionType.MOVE:
            # Location projection is handled by the caller via
            # reset_projected_location() after reverting, so nothing
            # to do here for resource commitments.
            pass

        elif spec.action_type == ActionType.LOAD:
            resource = spec.resource_name
            amount = spec.amount
            if resource is not None and amount is not None and amount > 0:
                partner = spec.partner_name
                if partner is not None:
                    self._unreserve_drain(partner, resource, amount)
                    # --- Granular: release physical claim (supply-side) ---
                    self.complete_load_claim(partner, resource)
                    # --- Granular: drain on partner freed capacity → clean up ---
                    if activity_name is not None:
                        self.complete_capacity_release(partner, resource, activity_name)
                self._unreserve_fill(spec.vessel_name, resource, amount)
                # --- Granular: vessel gained resource → release deposit claim ---
                self.complete_deposit_claim(spec.vessel_name, resource)
                # --- Granular: clean up vessel delivery tracking ---
                if activity_name is not None:
                    self.complete_delivery(spec.vessel_name, resource, activity_name)
                # --- Ledger: resource physically moved from partner to vessel ---
                self.ledger_apply_transfer(partner, spec.vessel_name, resource, amount)

        elif spec.action_type == ActionType.UNLOAD:
            resource = spec.resource_name
            amount = spec.amount
            if resource is not None and amount is not None and amount > 0:
                self._unreserve_drain(spec.vessel_name, resource, amount)
                # --- Granular: release supply claim on vessel ---
                self.complete_load_claim(spec.vessel_name, resource)
                # --- Granular: drain on vessel freed capacity → clean up ---
                if activity_name is not None:
                    self.complete_capacity_release(
                        spec.vessel_name, resource, activity_name
                    )
                partner = spec.partner_name
                if partner is not None:
                    self._unreserve_fill(partner, resource, amount)
                    # --- Granular: clean up delivery tracking ---
                    if activity_name is not None:
                        self.complete_delivery(partner, resource, activity_name)
                    # --- Granular: partner gained resource → release deposit claim ---
                    self.complete_deposit_claim(partner, resource)
                # --- Ledger: resource physically moved from vessel to partner ---
                self.ledger_apply_transfer(spec.vessel_name, partner, resource, amount)

        # ActionType.IDLE has no projected effects to revert

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_projected_location(self, vessel_name: str) -> str | None:
        """Return the vessel's projected site after all queued activities resolve."""
        return self._projected_locations.get(vessel_name)

    def get_total_drain(self, object_name: str, resource_name: str) -> int:
        return self._drains.get((object_name, resource_name), 0)

    def get_total_fill(self, object_name: str, resource_name: str) -> int:
        return self._fills.get((object_name, resource_name), 0)

    def can_claim_resource(self, object_name: str, resource_name: str) -> bool:
        """Non-mutating check: can 1 unit of supply be claimed?"""
        key = (object_name, resource_name)
        physical = self._ledger_levels.get(key, 0)
        claimed = self._claimed_physical.get(key, 0)
        if physical - claimed > 0:
            return True
        queue = self._pending_deliveries.get(key)
        return bool(queue)

    def can_claim_capacity(self, object_name: str, resource_name: str) -> bool:
        """Non-mutating check: can 1 free capacity slot be claimed?"""
        key = (object_name, resource_name)
        free = self._ledger_free_slots.get(key, 0)
        claimed = self._claimed_free_slots.get(key, 0)
        if free - claimed > 0:
            return True
        queue = self._pending_capacity_releases.get(key)
        return bool(queue)

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    @classmethod
    def from_simulation_state(
        cls,
        activity_tracker: ActivityTracker,
        current_locations: Dict[str, str] | None = None,
    ) -> ReservationSystem:
        """Build the projected state directly from DES activities.

        Parameters
        ----------
        activity_tracker : ActivityTracker
            The simulation's activity tracker.
        current_locations : dict[str, str] | None
            Dictionary mapping vessel names to their current physical site names.
        """
        rs = cls()

        # 1. Seed base locations from the physical environment
        if current_locations:
            for v_name, s_name in current_locations.items():
                rs.set_projected_location(v_name, s_name)

        # 2. Inspect pending + active activities for projections
        for state in (
            activity_tracker.get_pending_activities()
            + activity_tracker.get_active_activities()
        ):
            activity = state.activity

            # --- Project Locations ---
            # If a MoveActivity is in the queue, fast-forward the location
            if hasattr(activity, "destination") and hasattr(activity, "mover"):
                if (
                    getattr(activity, "category", None) == "transit"
                    or type(activity).__name__ == "MoveActivity"
                ):
                    mover_name = activity.mover.name  # type: ignore[attr-defined]
                    dest_name = activity.destination.name  # type: ignore[attr-defined]
                    rs.set_projected_location(mover_name, dest_name)

            # --- Project Resources ---
            resource_name: str | None = None
            amount: int | None = None

            if hasattr(activity, "id_") and hasattr(activity, "amount"):
                resource_name = activity.id_  # type: ignore[attr-defined]
                amount = activity.amount  # type: ignore[attr-defined]

            if resource_name is None or amount is None or amount <= 0:
                continue

            # Origin → drain (site)
            origin = state.sites.get("origin")
            if origin is not None:
                rs.reserve_drain(
                    origin.name,
                    resource_name,
                    amount,
                    source=f"des-activity:{state.name}",
                )
                # --- Granular: drains free capacity at the origin ---
                rs.register_pending_capacity_release(
                    origin.name, resource_name, state.name
                )

            # Destination → fill (site)
            destination = state.sites.get("destination")
            if destination is not None:
                rs.reserve_fill(
                    destination.name,
                    resource_name,
                    amount,
                    source=f"des-activity:{state.name}",
                )

                # --- Granular: register pending deliveries for unloads
                # and fabrication activities ---
                category = getattr(activity, "category", None)
                if category in ("unloading", "fabrication"):
                    rs.register_pending_delivery(
                        destination.name, resource_name, state.name
                    )

            # Handle vessel-as-origin or vessel-as-destination
            _apply_vessel_resource_commitments(rs, state, resource_name, amount)

        return rs

    # ------------------------------------------------------------------
    # Debugging / introspection
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, object]:
        """Return a human-readable summary of all reservations."""
        return {
            "projected_locations": self._projected_locations.copy(),
            "drains": {
                f"{obj}:{res}": amt for (obj, res), amt in sorted(self._drains.items())
            },
            "fills": {
                f"{obj}:{res}": amt for (obj, res), amt in sorted(self._fills.items())
            },
            "pending_deliveries": {
                f"{s}:{r}": list(q)
                for (s, r), q in sorted(self._pending_deliveries.items())
            },
            "claimed_physical": {
                f"{s}:{r}": c
                for (s, r), c in sorted(self._claimed_physical.items())
                if c > 0
            },
            "pending_capacity_releases": {
                f"{s}:{r}": list(q)
                for (s, r), q in sorted(self._pending_capacity_releases.items())
            },
            "claimed_free_slots": {
                f"{s}:{r}": c
                for (s, r), c in sorted(self._claimed_free_slots.items())
                if c > 0
            },
            "ledger_levels": {
                f"{obj}:{res}": lvl
                for (obj, res), lvl in sorted(self._ledger_levels.items())
                if lvl != 0
            },
            "ledger_free_slots": {
                f"{obj}:{res}": fs
                for (obj, res), fs in sorted(self._ledger_free_slots.items())
                if fs != 0
            },
        }

    def matches(self, other: ReservationSystem) -> bool:
        """Check whether this RS is equivalent to *other*.

        Useful as a parity check: build a fresh RS via
        ``from_simulation_state`` and compare against the incrementally
        maintained one.

        Returns
        -------
        bool
            ``True`` when projected locations, drains, fills, and pending
            deliveries are identical.

        Notes
        -----
        ``_claimed_physical`` is **not** compared because it is ephemeral
        micro-step state that resets each batch and cannot be reconstructed
        from the DES alone.
        """
        return (
            self._projected_locations == other._projected_locations
            and self._drains == other._drains
            and self._fills == other._fills
            and self._pending_deliveries == other._pending_deliveries
            and self._pending_capacity_releases == other._pending_capacity_releases
        )

    def diff(self, other: ReservationSystem) -> Dict[str, object]:
        """Return a human-readable diff between this RS and *other*.

        Parameters
        ----------
        other : ReservationSystem
            The reservation system to compare against (typically a fresh
            full-rebuild).

        Returns
        -------
        dict
            Keys ``locations``, ``drains``, ``fills``, ``pending_deliveries``,
            ``pending_capacity_releases`` each containing a dict of mismatched
            entries with ``(self_value, other_value)`` tuples.
        """
        result: Dict[str, object] = {}

        # Locations
        all_vessels = set(self._projected_locations) | set(other._projected_locations)
        loc_diff = {}
        for v in sorted(all_vessels):
            a = self._projected_locations.get(v)
            b = other._projected_locations.get(v)
            if a != b:
                loc_diff[v] = (a, b)
        if loc_diff:
            result["locations"] = loc_diff

        # Drains
        all_drain_keys = set(self._drains) | set(other._drains)
        drain_diff = {}
        for k in sorted(all_drain_keys):
            a = self._drains.get(k, 0)
            b = other._drains.get(k, 0)
            if a != b:
                drain_diff[f"{k[0]}:{k[1]}"] = (a, b)
        if drain_diff:
            result["drains"] = drain_diff

        # Fills
        all_fill_keys = set(self._fills) | set(other._fills)
        fill_diff = {}
        for k in sorted(all_fill_keys):
            a = self._fills.get(k, 0)
            b = other._fills.get(k, 0)
            if a != b:
                fill_diff[f"{k[0]}:{k[1]}"] = (a, b)
        if fill_diff:
            result["fills"] = fill_diff

        # Pending deliveries
        all_pd_keys = set(self._pending_deliveries) | set(other._pending_deliveries)
        pd_diff = {}
        for k in sorted(all_pd_keys):
            a = list(self._pending_deliveries.get(k, deque()))
            b = list(other._pending_deliveries.get(k, deque()))
            if a != b:
                pd_diff[f"{k[0]}:{k[1]}"] = (a, b)
        if pd_diff:
            result["pending_deliveries"] = pd_diff

        # Pending capacity releases
        all_cr_keys = set(self._pending_capacity_releases) | set(
            other._pending_capacity_releases
        )
        cr_diff = {}
        for k in sorted(all_cr_keys):
            a = list(self._pending_capacity_releases.get(k, deque()))
            b = list(other._pending_capacity_releases.get(k, deque()))
            if a != b:
                cr_diff[f"{k[0]}:{k[1]}"] = (a, b)
        if cr_diff:
            result["pending_capacity_releases"] = cr_diff

        return result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _apply_vessel_resource_commitments(
    rs: ReservationSystem,
    state: ActivityTracker.ActivityState,
    resource_name: str,
    amount: int,
) -> None:
    """Handle resource commitments where vessels (not sites) are origin/destination."""
    activity = state.activity

    act_origin = getattr(activity, "origin", None)
    act_destination = getattr(activity, "destination", None)

    # Origin is a vessel (not already captured as a Site)
    if act_origin is not None:
        origin_name = getattr(act_origin, "name", None)
        if origin_name and origin_name not in {s.name for s in state.sites.values()}:
            rs.reserve_drain(
                origin_name,
                resource_name,
                amount,
                source=f"des-activity-vessel-origin:{state.name}",
            )
            # Drain on vessel frees capacity
            rs.register_pending_capacity_release(origin_name, resource_name, state.name)

    # Destination is a vessel (not already captured as a Site)
    if act_destination is not None:
        dest_name = getattr(act_destination, "name", None)
        if dest_name and dest_name not in {s.name for s in state.sites.values()}:
            rs.reserve_fill(
                dest_name,
                resource_name,
                amount,
                source=f"des-activity-vessel-dest:{state.name}",
            )
            # Fill to vessel → register as pending delivery (supply tracking)
            category = getattr(activity, "category", None)
            if category in ("unloading", "loading"):
                rs.register_pending_delivery(dest_name, resource_name, state.name)
