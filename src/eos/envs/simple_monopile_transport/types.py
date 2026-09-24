from enum import Enum, IntEnum
from typing import Union

import des_package.core as des_core

# ---------------------------------------------------------------------------
# Entity role enums – used in configs and simulator rule logic
# ---------------------------------------------------------------------------


class SiteRole(str, Enum):
    """Role of a site in the logistics network.

    Determines which operations (load / unload) are permitted at the site.

    Attributes
    ----------
    SOURCE : str
        Resources originate here. Vessels may **load from** (pick up) but
        never **unload to** (return) a source site.
    INSTALLATION : str
        Resources are permanently installed here. Only **installer** vessels
        may **unload to** (install at) this site; no vessel may **load from**
        it.
    STAGING : str
        Intermediate storage / buffer. Vessels may both load from and unload
        to a staging site freely.
    """

    SOURCE = "source"
    INSTALLATION = "installation"
    STAGING = "staging"


class VesselRole(str, Enum):
    """Role of a vessel in the logistics network.

    Determines which operations a vessel is allowed to perform.

    Transport-class roles
    ~~~~~~~~~~~~~~~~~~~~~
    Both ``HEAVY_LIFT`` and ``FEEDER`` are *transport-class* vessels: they
    can sail between sites, load and unload cargo, and transfer to other
    vessels – but they **cannot** install at installation sites.  The
    distinction exists so that the milestone tracker can assign strictly
    increasing potentials along the supply-chain relay
    (source → heavy-lift → staging → feeder → installer → goal).

    Attributes
    ----------
    HEAVY_LIFT : str
        Long-haul bulk carrier (source → staging leg).
    FEEDER : str
        Short-haul ferry (staging → installation leg).
    INSTALLER : str
        Installation asset.  Whether it can also sail is governed by
        the per-vessel ``movable`` flag in the vessel config.
    """

    HEAVY_LIFT = "heavy_lift"
    FEEDER = "feeder"
    INSTALLER = "installer"

    @property
    def is_transport(self) -> bool:
        """True for any transport-class role (heavy-lift or feeder)."""
        return self in (VesselRole.HEAVY_LIFT, VesselRole.FEEDER)


# ---------------------------------------------------------------------------
# Simulation object types (DES mixins)
# ---------------------------------------------------------------------------


class Site(
    des_core.Identifiable,
    des_core.Log,
    des_core.Processor,
    des_core.Locatable,
    des_core.HasMultiContainer,
    des_core.HasResource,
):
    pass


class TransportProcessingResource(
    des_core.MultiContainerDependentMovable,
    des_core.HasResource,
    des_core.Processor,
    des_core.Identifiable,
    des_core.Log,
    des_core.LoadingFunction,
    des_core.UnloadingFunction,
):
    pass


class InstallationAsset(
    des_core.Identifiable,
    des_core.Log,
    des_core.Locatable,
    des_core.HasMultiContainer,
    des_core.HasResource,
    des_core.Processor,
):
    pass


Vessel = Union[TransportProcessingResource, InstallationAsset]


# ---------------------------------------------------------------------------
# Action-space enums
# ---------------------------------------------------------------------------


class PartnerType(IntEnum):
    SITE = 0
    VESSEL = 1


class ActionType(IntEnum):
    IDLE = 0
    MOVE = 1
    LOAD = 2
    UNLOAD = 3

    def pretty_name(self) -> str:
        """Return human-readable action type name."""
        return self.name.title()

    def from_str(type: str) -> "ActionType":
        """Convert string to ActionType enum."""
        return ActionType[type.upper()]


class VisitMode(IntEnum):
    NEUTRAL = 0  # Vessel has had no interaction with site for this resource
    LOADING = 1  # Vessel has loaded this resource
    UNLOADING = 2  # Vessel has unloaded this resource
