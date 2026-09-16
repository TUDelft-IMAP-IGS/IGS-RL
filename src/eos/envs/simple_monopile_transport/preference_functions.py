"""Preference functions for PFM desirability mapping.

A preference function maps a raw physical metric value to a dimensionless
desirability score in the range [0, 100].  This is Step 2 of the ODESYS
threefold formulation (Capability → Desirability → Solvability).

The preference function encodes directionality: for "minimize" objectives,
a lower physical value yields a higher preference score; for "maximize"
objectives, a higher physical value yields a higher preference score.

All preference functions clamp their output to [0, 100] for robustness
when raw values exceed the configured bounds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class PreferenceFunction(ABC):
    """Abstract base class for preference functions.

    A preference function maps a raw physical metric to a desirability
    score in [0, 100].  Subclasses must implement :meth:`evaluate`.
    """

    @abstractmethod
    def evaluate(self, raw_value: float) -> float:
        """Map a raw physical metric to a preference score in [0, 100].

        Parameters
        ----------
        raw_value : float
            The raw physical metric value (e.g., elapsed hours, cost).

        Returns
        -------
        float
            Desirability score clamped to [0, 100].
        """
        ...

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp a value to [0, 100]."""
        return max(0.0, min(100.0, value))


class LinearPreferenceFunction(PreferenceFunction):
    """Linear preference function mapping worst→0 and best→100.

    For a "minimize" objective (lower is better), ``worst`` is the upper
    bound and ``best`` is the lower bound.  For a "maximize" objective,
    the reverse.

    Parameters
    ----------
    worst : float
        Physical value corresponding to preference score 0 (least desirable).
    best : float
        Physical value corresponding to preference score 100 (most desirable).

    Raises
    ------
    ValueError
        If ``worst == best`` (degenerate — no range to interpolate over).
    """

    def __init__(self, worst: float, best: float) -> None:
        if worst == best:
            raise ValueError(
                f"LinearPreferenceFunction requires worst != best, "
                f"got worst={worst}, best={best}"
            )
        self._worst = worst
        self._best = best
        # Pre-compute the linear mapping: score = slope * (raw - worst)
        # When raw == worst → 0, when raw == best → 100
        self._slope = 100.0 / (best - worst)

    @property
    def worst(self) -> float:
        """Physical value corresponding to preference score 0."""
        return self._worst

    @property
    def best(self) -> float:
        """Physical value corresponding to preference score 100."""
        return self._best

    def evaluate(self, raw_value: float) -> float:
        """Linearly interpolate between worst (0) and best (100)."""
        score = self._slope * (raw_value - self._worst)
        return self._clamp(score)

    def __repr__(self) -> str:
        return f"LinearPreferenceFunction(worst={self._worst}, best={self._best})"
