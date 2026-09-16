from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from eos.config import RenderConfig
from eos.envs.simple_monopile_transport.gym_env import (
    Observation,
    ResourceBundle,
    VesselStatus,
)


@dataclass(slots=True)
class SiteLayout:
    name: str
    pos: Tuple[int, int]


class SMTDebugRenderer:
    def __init__(
        self,
        cfg: RenderConfig,
        site_locations: Dict[str, List[float]],
        vessel_names: List[str],
        site_names: List[str],
        resource_names: List[str],
        render_mode: str,
    ) -> None:
        self.cfg = cfg
        self.site_locations = site_locations
        self.vessel_names = vessel_names
        self.site_names = site_names
        self.resource_names = resource_names
        self.render_mode = render_mode

        if render_mode == "rgb_array":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        try:
            import pygame
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "pygame is required for SMT rendering. Install with `uv add pygame` or `pip install pygame`."
            ) from exc

        self.pygame = pygame
        self.pygame.init()
        self._font = self.pygame.font.SysFont("monospace", int(self.cfg.font_size))

        self._window = None
        self._surface = None
        self._clock = self.pygame.time.Clock()

        self._layouts = self._build_site_layout()

    def close(self) -> None:
        if self._window is not None:
            self.pygame.display.quit()
        self.pygame.quit()

    def _build_site_layout(self) -> List[SiteLayout]:
        # Use provided locations if available, otherwise fall back to a circle layout.
        locations: List[Tuple[float, float]] = []
        for name in self.site_names:
            loc = self.site_locations.get(name)
            if loc and len(loc) >= 2:
                locations.append((float(loc[0]), float(loc[1])))

        if len(locations) != len(self.site_names):
            return self._circle_layout()

        xs = [p[0] for p in locations]
        ys = [p[1] for p in locations]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        def norm(v: float, vmin: float, vmax: float) -> float:
            if abs(vmax - vmin) < 1e-8:
                return 0.5
            return (v - vmin) / (vmax - vmin)

        layouts: List[SiteLayout] = []
        margin = 60
        width = self.cfg.width - self.cfg.panel_width - 2 * margin
        height = self.cfg.height - 2 * margin
        for name, (x, y) in zip(self.site_names, locations):
            px = int(margin + norm(x, x_min, x_max) * width)
            py = int(margin + norm(y, y_min, y_max) * height)
            layouts.append(SiteLayout(name=name, pos=(px, py)))
        return layouts

    def _circle_layout(self) -> List[SiteLayout]:
        layouts: List[SiteLayout] = []
        margin = 60
        center_x = (self.cfg.width - self.cfg.panel_width) // 2
        center_y = self.cfg.height // 2
        radius = min(
            center_x - margin,
            center_y - margin,
        )
        count = max(1, len(self.site_names))
        for i, name in enumerate(self.site_names):
            theta = 2 * math.pi * (i / count)
            px = int(center_x + radius * math.cos(theta))
            py = int(center_y + radius * math.sin(theta))
            layouts.append(SiteLayout(name=name, pos=(px, py)))
        return layouts

    def _bundle_summary(self, bundle: ResourceBundle) -> str:
        parts: List[str] = []
        for res in self.resource_names:
            state = bundle.resources.get(res)
            if state is None:
                continue
            parts.append(f"{res}:{state.load}/{state.capacity}")
        return ", ".join(parts) if parts else "-"

    def _draw_text_lines(
        self, surface, lines: Iterable[str], pos: Tuple[int, int], color=(230, 230, 230)
    ) -> int:
        x, y = pos
        for line in lines:
            text = self._font.render(line, True, color)
            surface.blit(text, (x, y))
            y += int(self.cfg.font_size * 1.3)
        return y

    def _get_site_pos(self, site_id: int) -> Tuple[int, int]:
        if 0 <= site_id < len(self._layouts):
            return self._layouts[site_id].pos
        return (40, 40)

    def draw(self, observation: Observation, info: Dict | None) -> np.ndarray | None:
        if self.render_mode == "human":
            if self._window is None:
                self._window = self.pygame.display.set_mode(
                    (self.cfg.width, self.cfg.height)
                )
                self.pygame.display.set_caption("SMT Debug View")
            surface = self._window
        else:
            if self._surface is None:
                self._surface = self.pygame.Surface((self.cfg.width, self.cfg.height))
            surface = self._surface

        # Background gradient
        top = np.array([14, 18, 28], dtype=np.int16)
        bottom = np.array([24, 28, 40], dtype=np.int16)
        for y in range(self.cfg.height):
            t = y / max(1, self.cfg.height - 1)
            color = (top * (1 - t) + bottom * t).astype(int)
            self.pygame.draw.line(surface, color.tolist(), (0, y), (self.cfg.width, y))

        # Draw map panel
        panel_x = self.cfg.width - self.cfg.panel_width
        self.pygame.draw.line(
            surface, (55, 60, 78), (panel_x, 0), (panel_x, self.cfg.height), 2
        )
        self.pygame.draw.rect(
            surface, (28, 28, 36), (panel_x, 0, self.cfg.panel_width, self.cfg.height)
        )

        # Draw sites
        for site in self._layouts:
            self.pygame.draw.circle(surface, (50, 70, 110), site.pos, 26)
            self.pygame.draw.circle(surface, (90, 150, 255), site.pos, 18)
            label = self._font.render(site.name, True, (255, 255, 255))
            surface.blit(label, (site.pos[0] + 22, site.pos[1] - 8))

        # Draw vessels at their current sites
        for vessel in observation.vessels:
            site_pos = self._get_site_pos(vessel.position)
            angle = (vessel.id / max(1, len(observation.vessels))) * 2 * math.pi
            offset = (int(22 * math.cos(angle)), int(22 * math.sin(angle)))
            pos = (site_pos[0] + offset[0], site_pos[1] + offset[1])
            color = (
                (80, 220, 120) if vessel.status == VesselStatus.IDLE else (240, 170, 60)
            )
            self.pygame.draw.circle(surface, (15, 15, 20), pos, 12)
            self.pygame.draw.circle(surface, color, pos, 10)
            label = self._font.render(
                self.vessel_names[vessel.id], True, (240, 240, 240)
            )
            surface.blit(label, (pos[0] + 12, pos[1] - 8))

        # Dashboard panel
        dash_x = panel_x + 16
        dash_y = 16
        lines: List[str] = []
        lines.append("SIMULATOR")
        lines.append(f"t={observation.global_obs.current_time:.2f}")
        lines.append(f"pending={int(observation.global_obs.num_pending_tasks)}")
        if info:
            lines.append(
                f"goals: {info.get('goals_completed', 0)}/{info.get('goals_total', 0)}"
            )
            lines.append(f"failed: {info.get('goals_failed', 0)}")
            lines.append(f"remaining: {info.get('goals_remaining', 0)}")
            lines.append(
                f"busy={info.get('busy_vessels', 0)} idle={info.get('idle_vessels', 0)}"
            )
            lines.append(f"goal_reward={info.get('goal_reward', 0):.2f}")
        dash_y = self._draw_text_lines(surface, lines, (dash_x, dash_y))
        self.pygame.draw.line(
            surface,
            (60, 65, 82),
            (dash_x, dash_y + 4),
            (panel_x + self.cfg.panel_width - 12, dash_y + 4),
            1,
        )
        dash_y += int(self.cfg.font_size)

        # Vessels
        dash_y = self._draw_text_lines(surface, ["VESSELS"], (dash_x, dash_y))
        for vessel in observation.vessels:
            name = self.vessel_names[vessel.id]
            site_name = self.site_names[vessel.position]
            status = "IDLE" if vessel.status == VesselStatus.IDLE else "BUSY"
            inv = (
                self._bundle_summary(vessel.inventory)
                if self.cfg.show_inventory
                else "-"
            )
            dash_y = self._draw_text_lines(
                surface,
                [f"{name} @ {site_name} [{status}]", f"  inv: {inv}"],
                (dash_x, dash_y),
            )

        dash_y += int(self.cfg.font_size)

        # Sites
        dash_y = self._draw_text_lines(surface, ["SITES"], (dash_x, dash_y))
        for site in observation.sites:
            name = self.site_names[site.id]
            inv = self._bundle_summary(site.stock) if self.cfg.show_inventory else "-"
            dash_y = self._draw_text_lines(
                surface, [f"{name}", f"  stock: {inv}"], (dash_x, dash_y)
            )

        dash_y += int(self.cfg.font_size)

        # Goals
        if hasattr(observation, "goals") and observation.goals:
            dash_y = self._draw_text_lines(surface, ["GOALS"], (dash_x, dash_y))
            for goal in observation.goals:
                site_label = (
                    self.site_names[goal.target_site]
                    if 0 <= goal.target_site < len(self.site_names)
                    else f"s{goal.target_site}"
                )
                res_label = (
                    self.resource_names[goal.resource_type]
                    if 0 <= goal.resource_type < len(self.resource_names)
                    else f"r{goal.resource_type}"
                )
                status = (
                    "FAIL" if goal.is_failed else ("BLK" if goal.is_blocked else "OK")
                )
                dash_y = self._draw_text_lines(
                    surface,
                    [
                        f"g{goal.id}: {res_label}@{site_label} {goal.installed}/{goal.required} [{status}]"
                    ],
                    (dash_x, dash_y),
                )

        if self.render_mode == "human":
            self.pygame.display.flip()
            self.pygame.event.pump()
            self._clock.tick(self.cfg.fps)
            return None

        array = self.pygame.surfarray.array3d(surface)
        array = np.transpose(array, (1, 0, 2))
        return array
