from typing import Any, Dict

from loguru import logger


def log_sim_objects(registry: Dict[str, Any]) -> None:
    # Debug log: List all sim_objects from registry with their internal fields in a single log call
    lines: list[str] = []
    lines.append("Simulation objects:")
    lines.append("=" * 80)

    # Navigate the nested structure: registry['sim_objects'][object_type][instance_name]
    if "sim_objects" in registry:
        for obj_type_name, instances in registry["sim_objects"].items():
            lines.append("")
            lines.append(f"🏷️  Object Type: {obj_type_name}")
            lines.append("   " + "-" * 60)

            for obj_name, obj in instances.items():
                lines.append("")
                lines.append(f"   📦 {obj_name}: {obj}")
            lines.append("")

    lines.append("=" * 80)
    logger.debug("\n".join(lines))
