"""EOS — RL agent for logistic environments

Key sub-packages
----------------
* :mod:`eos.core` — abstract base classes (Model, Controller, Learner,
  Runner, Buffer, Experiment).
* :mod:`eos.models` — concrete neural-network architectures (MLP,
  Transformer) implementing the AEC micro-stepping interface.
* :mod:`eos.controllers` — action-selection controllers (PPO).
* :mod:`eos.learners` — training algorithms (PPO).
* :mod:`eos.runners` — environment interaction loops (PPO rollout).
* :mod:`eos.buffers` — experience storage (PPO rollout buffer).
* :mod:`eos.experiments` — experiment orchestrators (PPO training,
  random baseline).
* :mod:`eos.envs` — gymnasium environments and wrappers.
* :mod:`eos.utils` — metrics, visualisation, and other helpers.
"""
