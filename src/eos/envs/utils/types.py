import des_package.core as des_core

Site = type(
    "Site",
    (
        des_core.Identifiable,
        des_core.Log,
        des_core.Processor,
        des_core.Locatable,
        des_core.HasMultiContainer,
        des_core.HasResource,
    ),
    {},
)

TransportVessel = type(
    "TransportVessel",
    (
        des_core.MultiContainerDependentMovable,
        des_core.HasResource,
        des_core.Processor,
        des_core.Identifiable,
        des_core.Log,
    ),
    {},
)

InstallationAsset = type(
    "InstallationAsset",
    (
        des_core.Identifiable,
        des_core.Log,
        des_core.Locatable,
        des_core.HasContainer,
        des_core.HasResource,
        des_core.Processor,
    ),
    {},
)
