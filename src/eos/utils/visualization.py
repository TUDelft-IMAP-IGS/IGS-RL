"""Visualisation utilities for simulation state inspection.

Provides :func:`get_state_overview`, which renders a Rich-formatted HTML
overview of the simulation registry (sites, vessels, activities) suitable
for logging to WandB or displaying in a notebook.
"""

# type: ignore

import io
import re

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table


def get_state_overview(env):
    """Return an HTML-formatted registry overview with separate tables for each entity class.

    Parameters
    ----------
    env :
        The DES simulation environment whose registry will be
        inspected.  Must expose ``check_registry_in_env()``,
        ``activity_states()``, and a ``registry`` dict.

    Returns
    -------
    str
        An HTML string (with inline styles) containing Rich-rendered
        tables summarising activities, simulation objects, their
        locations, container/resource levels, and active activities.
    """

    env.check_registry_in_env()

    console = Console(record=True, file=io.StringIO(), width=100)
    # Create tables that will be grouped together
    tables = []

    # First show activity status overview
    if "activities" in env.registry:
        activities_table = Table(title="Activities Overview")
        activities_table.add_column("Status", style="bold")
        activities_table.add_column("Activities", style="green")

        # Group activities by status
        activity_states = env.activity_states()

        # Iterate through each status and add a row for each
        for status in ["ACTIVE", "PENDING", "PROCESSED"]:
            activities_table.add_row(
                status.capitalize(),
                ", ".join([a.name for a in activity_states[status]])
                if activity_states[status]
                else "None",
            )

        tables.append(activities_table)

    # Process each sim object class separately
    if not env.registry["sim_objects"]:
        # Create a table for when no sim objects exist
        no_objects_table = Table(title="Simulation Objects")
        no_objects_table.add_column("Status", style="bold")
        no_objects_table.add_row("No simulation objects in registry")
        tables.append(no_objects_table)
    else:
        for sim_class_name, sim_class_instances in env.registry["sim_objects"].items():
            # Create a table for this class
            class_table = Table(title=f"{sim_class_name} Objects")

            # Add basic columns that all objects have
            class_table.add_column("Object Name", style="bold green")
            class_table.add_column("ID")

            # Determine columns needed for this specific class
            sample_instance = next(iter(sim_class_instances.values()), None)
            columns_schema = []

            if sample_instance:
                # Define all possible columns with their attributes and styles
                possible_columns = [
                    ("geometry", "Location", "yellow"),
                    ("container", "Container\n(Level/Capacity)", "magenta"),
                    ("resource", "Resource\n(Count/Capacity)", "blue"),
                ]

                # Check which attributes are present and add corresponding columns
                for attr, column_name, style in possible_columns:
                    if hasattr(sample_instance, attr):
                        class_table.add_column(column_name, style=style)
                        columns_schema.append(attr)

                # Add column for active activities
                class_table.add_column("Used By Activities", style="cyan")

            # Add rows for each instance of this class
            for sim_object_name, sim_object_instance in sim_class_instances.items():
                row_data = [sim_object_name, str(sim_object_instance.id)]

                # Process each attribute according to the schema
                for attr in columns_schema:
                    if attr == "geometry":
                        location_str = str(sim_object_instance.geometry)
                        # Extract coordinates for Point objects
                        if "POINT" in location_str:
                            match = re.search(
                                r"POINT \(([0-9.-]+) ([0-9.-]+)\)", location_str
                            )
                            if match:
                                x, y = match.groups()
                                location_str = f"({x}, {y})"
                        row_data.append(location_str)

                    elif attr == "container":
                        level = sim_object_instance.container.get_level()
                        capacity = sim_object_instance.container.get_capacity()
                        row_data.append(f"{level}/{capacity}")

                    elif attr == "resource":
                        count = sim_object_instance.resource.count
                        capacity = sim_object_instance.resource.capacity
                        queue_ids = sim_object_instance.current_queue()
                        queue_info = ""
                        if count == capacity and queue_ids:
                            queue_str = ", ".join(queue_ids)
                            queue_info = f"\n[Queue: {len(queue_ids)}] {queue_str}"
                        elif count == capacity:
                            queue_info = "\n[Queue: Empty]"
                        row_data.append(f"{count}/{capacity}{queue_info}")

                row_data.append(
                    "\n".join(sim_object_instance.active_activities)
                    if sim_object_instance.active_activities
                    else "None"
                )

                class_table.add_row(*row_data)

            tables.append(class_table)

    # Create a group of all tables
    group = Group(*tables)
    console.print(
        Panel(group, title=f"State overview at time {env.now}", border_style="bold")
    )

    return console.export_html(inline_styles=True)
