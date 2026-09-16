"""Shared environment factory for all experiment types.

Provides :func:`make_env`, which builds a single-env constructor (thunk)
suitable for use with ``gym.vector.SyncVectorEnv`` or
``gym.vector.AsyncVectorEnv``.  This consolidates the duplicated
``make_env`` static methods that previously lived in both
:class:`~eos.experiments.ppo.PPOExperiment` and
:class:`~eos.experiments.random.RandomExperiment`.

Reward normalization is applied at the **vector level** (after
vectorization and any PFM wrapper) via :func:`wrap_normalize_reward`.
This ensures all sub-environments share a single set of running return
statistics, which is compatible with PFM cross-env comparisons.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
from gymnasium.wrappers.vector import NormalizeReward as _GymNormalizeRewardVector

from eos.config import EOSConfig


class NormalizeRewardVector(_GymNormalizeRewardVector):
    """gymnasium.wrappers.vector.NormalizeReward + VectorEnv delegation.

    gymnasium's VectorWrapper base class does not proxy ``call()``,
    ``call_async()``, ``call_wait()``, ``get_attr()``, or ``set_attr()``
    which are defined only on AsyncVectorEnv / SyncVectorEnv.  The
    micro-stepping protocol relies on ``envs.call("micro_step", ...)``,
    so we must forward these explicitly.
    """

    def call(self, name, *args, **kwargs):
        return self.env.call(name, *args, **kwargs)

    def call_async(self, name, *args, **kwargs):
        return self.env.call_async(name, *args, **kwargs)

    def call_wait(self, *args, **kwargs):
        return self.env.call_wait(*args, **kwargs)

    def get_attr(self, name):
        return self.env.get_attr(name)

    def set_attr(self, name, values):
        return self.env.set_attr(name, values)


def wrap_normalize_reward(envs, gamma: float = 0.99):
    """Wrap a VectorEnv with shared reward normalization.

    Applied at the vector level (main process) so all environments
    share a single set of running return statistics.  This is strictly
    better than per-env normalization and is compatible with PFM.
    """
    return NormalizeRewardVector(envs, gamma=gamma)


def make_env(
    cfg: EOSConfig,
    idx: int,
    capture_video: bool,
    run_name: str | None,
    render_mode: str | None = None,
) -> Callable[[], gym.Env]:
    """Return a zero-argument callable that creates a configured gym environment.

    Parameters
    ----------
    cfg : EOSConfig
        Top-level experiment config (``cfg.env`` is inspected for env ID,
        wrappers, render settings, etc.).
    idx : int
        Index of this environment within the vector.  Used to decide
        whether to enable video capture / rendering (only env 0).
    capture_video : bool
        When ``True`` **and** ``idx == 0``, the environment is wrapped
        with ``gym.wrappers.RecordVideo``.
    run_name : str | None
        Human-readable run name — used as part of the video output path.
    render_mode : str | None
        Render mode to pass to ``gym.make`` for env 0.  Ignored for
        ``idx > 0`` unless video capture is active.

    Returns
    -------
    Callable[[], gym.Env]
        A thunk that, when called, instantiates and returns the
        fully-wrapped environment.
    """

    def thunk() -> gym.Env:
        # Under the "spawn" start method, AsyncVectorEnv workers are fresh
        # interpreters that never ran the entry-point's process-global setup
        # (logging configuration, debug-numerics flag).  These don't transfer
        # through the pickled env_fn, so loguru falls back to its default
        # DEBUG-to-stderr handler and floods the console.  Re-apply the setup
        # here, but ONLY in worker processes: the main process already has its
        # handlers configured (including the wandb run.log file sink), and
        # loguru's configure() replaces all handlers, so we must not touch it.
        import multiprocessing as _mp

        if _mp.current_process().name != "MainProcess":
            import random as _random
            import sys as _sys

            import numpy as _np
            import torch as _torch
            from loguru import logger as _logger

            from eos.utils.debug import configure as _configure_debug

            _logger.configure(
                handlers=[{"sink": _sys.stderr, "level": str(cfg.log_level).upper()}]
            )
            _configure_debug(cfg.debug)

            # Deterministically seed this worker's *global* RNGs from
            # cfg.seed + idx.  The SMT env's own randomness uses the
            # per-instance self.np_random (seeded via reset(seed=...) over
            # IPC) and is unaffected by this, but some DES library code
            # (e.g. boka_eventsymphony's stochastic delay plugin) draws
            # from the global np.random.  Under "fork" workers inherited
            # the parent's seeded global RNG; under "spawn" they start from
            # OS entropy, so we re-seed here to keep runs reproducible.
            #
            # torch's RNG is not currently exercised in the env subprocess
            # (all torch sampling happens in the main-process controller),
            # so seeding it is purely defensive/consistent.
            _random.seed(cfg.seed + idx)
            _np.random.seed((cfg.seed + idx) % (2**32))
            _torch.manual_seed(cfg.seed + idx)

        env_id = cfg.env.env_id
        is_custom_env = "CartPole" not in env_id

        if is_custom_env:
            # Ensure the custom environment is registered in *this* process.
            # Under the "spawn" start method, AsyncVectorEnv workers are fresh
            # interpreters that do NOT inherit the parent's gymnasium registry,
            # so the import-time register() in
            # eos.envs.simple_monopile_transport.__init__ must run here before
            # gym.make().  The import is cached/idempotent, so this is a no-op
            # in the parent (and under "fork").
            import eos.envs.simple_monopile_transport  # noqa: F401

        # --- Base environment construction ---
        if capture_video and idx == 0:
            env = (
                gym.make(env_id, render_mode="rgb_array", cfg=cfg.env)
                if is_custom_env
                else gym.make(env_id, render_mode="rgb_array")
            )
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        elif is_custom_env:
            if render_mode is not None and idx == 0:
                env = gym.make(env_id, render_mode=render_mode, cfg=cfg.env)
            else:
                env = gym.make(env_id, cfg=cfg.env)
        else:
            env = gym.make(env_id)

        # --- Domain-specific wrappers (configured via YAML) ---
        if is_custom_env and hasattr(cfg.env, "wrappers") and cfg.env.wrappers:
            from hydra.utils import get_class

            for wrapper_path in cfg.env.wrappers:
                wrapper_class = get_class(wrapper_path)
                env = wrapper_class(env)

        # Tag the unwrapped env with its vector index so that wrappers
        # (e.g. dynamic masking) can route per-env calls correctly.
        setattr(env.unwrapped, "_vector_env_index", idx)

        # Standard episode-statistics wrapper (tracks returns & lengths)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk
