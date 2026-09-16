import boka_eventsymphony.core as es_core

Site = type(
    "Site",
    (
        es_core.Identifiable,
        es_core.Log,
        es_core.Processor,
        es_core.Locatable,
        es_core.HasMultiContainer,
        es_core.HasResource,
    ),
    {},
)

TransportVessel = type(
    "TransportVessel",
    (
        es_core.MultiContainerDependentMovable,
        es_core.HasResource,
        es_core.Processor,
        es_core.Identifiable,
        es_core.Log,
    ),
    {},
)

InstallationAsset = type(
    "InstallationAsset",
    (
        es_core.Identifiable,
        es_core.Log,
        es_core.Locatable,
        es_core.HasContainer,
        es_core.HasResource,
        es_core.Processor,
    ),
    {},
)
