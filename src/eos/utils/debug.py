"""Runtime debug assertions for numerical health.

All checks in this module are **no-ops** unless :data:`DEBUG_NUMERICS` is
``True``.  The flag is set once at experiment startup from ``cfg.debug``
via :func:`configure`.

Usage
-----
At experiment init::

    from eos.utils.debug import configure as configure_debug
    configure_debug(cfg.debug)

At check sites::

    from eos.utils.debug import check_nan
    check_nan(tensor_or_array, "descriptive.label")

When ``DEBUG_NUMERICS`` is ``False`` the function returns immediately
with zero overhead beyond the function-call itself.
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger

# ---------------------------------------------------------------------------
# Module-level flag — toggled once at startup
# ---------------------------------------------------------------------------

DEBUG_NUMERICS: bool = False


def configure(enabled: bool) -> None:
    """Set the module-level debug flag.  Called once from experiment setup."""
    global DEBUG_NUMERICS
    DEBUG_NUMERICS = bool(enabled)
    if DEBUG_NUMERICS:
        logger.info("DEBUG_NUMERICS enabled — runtime NaN/Inf checks are active.")


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------


def check_nan(
    data,
    label: str,
    *,
    raise_on_fail: bool = True,
) -> bool:
    """Check a tensor or array for NaN/Inf values.

    Parameters
    ----------
    data :
        A :class:`torch.Tensor`, :class:`numpy.ndarray`, or scalar.
    label : str
        Human-readable description shown in the error/warning message.
    raise_on_fail : bool
        If ``True`` (default), raises :class:`RuntimeError` on detection.
        If ``False``, logs a warning and returns ``True`` (meaning "bad").

    Returns
    -------
    bool
        ``True`` if NaN/Inf was detected, ``False`` otherwise.
        Always ``False`` when ``DEBUG_NUMERICS`` is disabled.
    """
    if not DEBUG_NUMERICS:
        return False

    has_nan, has_inf, n_nan, n_inf, numel = _inspect(data)

    if not (has_nan or has_inf):
        return False

    msg = f"[NaN check FAILED] {label}: NaN={n_nan}/{numel}, Inf={n_inf}/{numel}"

    if raise_on_fail:
        raise RuntimeError(msg)
    else:
        logger.warning(msg)
        return True


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _inspect(data):
    """Return (has_nan, has_inf, n_nan, n_inf, numel) for tensor or array."""
    if isinstance(data, torch.Tensor):
        nan_mask = torch.isnan(data)
        inf_mask = torch.isinf(data)
        n_nan = int(nan_mask.sum().item())
        n_inf = int(inf_mask.sum().item())
        numel = data.numel()
    elif isinstance(data, np.ndarray):
        nan_mask = np.isnan(data)
        inf_mask = np.isinf(data)
        n_nan = int(nan_mask.sum())
        n_inf = int(inf_mask.sum())
        numel = data.size
    else:
        # Scalar fallback
        import math

        val = float(data)
        n_nan = 1 if math.isnan(val) else 0
        n_inf = 1 if math.isinf(val) else 0
        numel = 1

    has_nan = n_nan > 0
    has_inf = n_inf > 0
    return has_nan, has_inf, n_nan, n_inf, numel
