"""Lightweight profiling utilities for timing named code phases.

Provides :class:`StepTimer`, a minimal context-manager-based timer that
accumulates wall-clock time for named phases and produces summary dicts
suitable for WandB logging.

Usage
-----
::

    timer = StepTimer()

    for step in range(num_steps):
        with timer.phase("masking"):
            masks = fetch_masks()

        with timer.phase("inference"):
            action = model(obs, masks)

        with timer.phase("env_step"):
            obs, reward, done, info = env.step(action)

    # Log to WandB
    wandb.log(timer.summary(prefix="perf/step/"))

    # Print to console
    timer.log_summary("Rollout step timing")

    # Reset for next rollout
    timer.reset()
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from loguru import logger


class StepTimer:
    """Accumulates wall-clock time for named phases.

    Each call to :meth:`phase` (used as a context manager) records the
    elapsed time for that phase.  Multiple calls to the same phase name
    accumulate.  :meth:`summary` returns a dict of timing metrics and
    :meth:`log_summary` prints a formatted table to the console.

    All timing uses :func:`time.perf_counter` for high resolution.
    The overhead per ``phase`` call is ~1 µs (two ``perf_counter`` reads
    plus a dict update), which is negligible compared to any real work.
    """

    __slots__ = ("_totals", "_counts")

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        """Time a named phase.

        Parameters
        ----------
        name : str
            Identifier for this phase (e.g. ``"masking"``, ``"env_step"``).

        Example
        -------
        ::

            with timer.phase("inference"):
                action = model(obs)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._totals[name] = self._totals.get(name, 0.0) + elapsed
            self._counts[name] = self._counts.get(name, 0) + 1

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def total_seconds(self) -> float:
        """Total accumulated time across all phases (seconds)."""
        return sum(self._totals.values())

    @property
    def phase_names(self) -> list[str]:
        """Phase names in insertion order."""
        return list(self._totals.keys())

    def summary(self, prefix: str = "perf/") -> dict[str, float]:
        """Return a flat dict of timing metrics for WandB logging.

        For each phase ``name`` the following keys are produced:

        - ``{prefix}{name}_total_s``  — total seconds in this phase
        - ``{prefix}{name}_mean_ms``  — mean milliseconds per invocation
        - ``{prefix}{name}_pct``      — percentage of total measured time

        Plus a single ``{prefix}total_s`` for the grand total.

        Parameters
        ----------
        prefix : str
            Key prefix (e.g. ``"perf/step/"`` or ``"perf/update/"``).

        Returns
        -------
        dict[str, float]
            Flat dict ready for ``wandb.log()``.
        """
        total = self.total_seconds
        result: dict[str, float] = {}

        for name, t in self._totals.items():
            count = self._counts[name]
            result[f"{prefix}{name}_total_s"] = t
            result[f"{prefix}{name}_mean_ms"] = (t / count) * 1000.0 if count else 0.0
            result[f"{prefix}{name}_pct"] = (t / total) * 100.0 if total > 0.0 else 0.0

        result[f"{prefix}total_s"] = total
        return result

    def log_summary(self, title: str = "Timing summary") -> None:
        """Print a formatted timing table to the console via loguru.

        Example output::

            ┌─ Rollout step timing ─────────────────────────────────┐
            │  masking      0.342 s   2.67 ms/call  34.2%  (128×)  │
            │  inference    0.412 s   3.22 ms/call  41.2%  (128×)  │
            │  env_step     0.246 s   1.92 ms/call  24.6%  (128×)  │
            │  total        1.000 s                                 │
            └───────────────────────────────────────────────────────┘
        """
        if not self._totals:
            logger.info(f"{title}: (no phases recorded)")
            return

        total = self.total_seconds
        max_name_len = max(len(n) for n in self._totals)
        col_w = max(max_name_len, 6)

        lines: list[str] = []
        for name, t in self._totals.items():
            count = self._counts[name]
            mean_ms = (t / count) * 1000.0 if count else 0.0
            pct = (t / total) * 100.0 if total > 0.0 else 0.0
            lines.append(
                f"  {name:<{col_w}s}  {t:>7.3f} s  {mean_ms:>7.2f} ms/call"
                f"  {pct:>5.1f}%  ({count}x)"
            )

        lines.append(f"  {'total':<{col_w}s}  {total:>7.3f} s")

        border_len = max(len(line) for line in lines) + 2
        header = f" {title} "
        top = f"┌─{header:─<{border_len - 2}s}┐"
        bot = f"└{'─' * border_len}┘"

        logger.info(top)
        for line in lines:
            logger.info(f"│{line:<{border_len}s}│")
        logger.info(bot)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated timings."""
        self._totals.clear()
        self._counts.clear()

    def __repr__(self) -> str:
        n = len(self._totals)
        t = self.total_seconds
        return f"StepTimer({n} phases, {t:.3f} s total)"
