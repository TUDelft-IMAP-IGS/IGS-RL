from dataclasses import dataclass
from typing import Dict

from eos.config import (
    SimConfig,
    SiteConfig,
    VesselConfig,
)


# TODO: Finish this implementation and refactor based on it when you have more time
class InventoryTracker:
    @dataclass
    class InventoryItem:
        name: str
        level: int
        capacity: int

    type Inventory = Dict[str, "InventoryTracker.InventoryItem"]

    def __init__(self, cfg: SimConfig):
        self._inventory_per_object: Dict[str, InventoryTracker.Inventory] = (
            self._build_initial_object_inv(cfg)
        )

    def _build_initial_object_inv(
        self, cfg: SimConfig
    ) -> Dict[str, "InventoryTracker.Inventory"]:
        vessel_cfgs = cfg.vessels
        site_cfgs = cfg.sites

        def object_cfg_to_inventory(
            object_cfg: SiteConfig | VesselConfig,
        ) -> InventoryTracker.Inventory:
            resource_types = object_cfg.resource_types
            initial_levels = getattr(object_cfg, "initial_levels", {})

            inventory = {}
            for resource_name, slot in resource_types.items():
                level = initial_levels.get(resource_name, 0)
                inventory_item = InventoryTracker.InventoryItem(
                    resource_name, level, slot.capacity
                )
                inventory[resource_name] = inventory_item

            return inventory

        return {
            object_name: object_cfg_to_inventory(object_cfg)
            for object_name, object_cfg in list(vessel_cfgs.items())
            + list(site_cfgs.items())
        }
