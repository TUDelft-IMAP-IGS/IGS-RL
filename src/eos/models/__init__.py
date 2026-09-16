"""Model implementations for EOS.

Available agents:

* :class:`MLPAgent` — simple MLP actor-critic.
* :class:`TransformerAgent` — Transformer-based actor-critic with
  schema-driven input encoding.

Both agents implement the **Intent-Based Micro-Stepping (AEC)** interface
defined by :class:`~eos.core.model.Model`:

* ``get_ordering`` — Phase 1: learned vessel permutation.
* ``get_action_and_value`` — Phase 2: per-vessel action selection with
  masked logits and critic value estimate.
"""

from .mlp import MLPAgent
from .transformer import TransformerAgent

__all__ = ["MLPAgent", "TransformerAgent"]
