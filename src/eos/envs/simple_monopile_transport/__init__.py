"""Simple Monopile Transport Simulation Environment.

This module provides a discrete-event simulation environment for offshore wind
monopile transport operations using the boka-eventsymphony library.

The environment models the transport workflow between fabrication yards,
marshalling yards, and installation sites, with transport vessels (HTV) and
installation assets (Bokalift).
"""

from .activity_builder import ActionSpec, ActivityBuilder, ActivityConfig
from .simulator import SimpleMonopileTransportSim
from .types import InstallationAsset, Site, TransportProcessingResource, Vessel

__all__ = [
    "SimpleMonopileTransportSim",
    "Site",
    "TransportProcessingResource",
    "InstallationAsset",
    "Vessel",
    "ActivityBuilder",
    "ActionSpec",
    "ActivityConfig",
]

from gymnasium.envs.registration import register

register(
    id="SimpleMonopileTransport-v0",
    entry_point="eos.envs.simple_monopile_transport.gym_env:SimpleMonopileTransportEnv",
)
