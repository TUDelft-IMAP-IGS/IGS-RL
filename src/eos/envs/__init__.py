"""Gymnasium environments and wrappers for EOS.

Sub-packages
------------
* :mod:`eos.envs.simple_monopile_transport` — a discrete-event simulation
  environment for offshore monopile transport logistics, including
  observation and action wrappers for both MLP and Transformer agents.
* :mod:`eos.envs.utils` — shared environment utility types.

The :mod:`eos.envs.factory` module provides the shared :func:`make_env`
constructor used by all experiment types to create vectorised gymnasium
environments with the correct wrappers applied.
"""
