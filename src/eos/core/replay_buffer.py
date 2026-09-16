"""Abstract base class for experience buffers.

A buffer stores transitions collected by the
:class:`~eos.core.runner.Runner` during environment interaction and
serves batches of experience to the :class:`~eos.core.learner.Learner`
for training.  Different algorithms may use different buffer
implementations (e.g. a fixed-size rollout buffer for PPO, a circular
replay buffer for off-policy methods, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any


class Buffer(ABC):
    """Base class for all experience storage backends.

    Subclasses must implement three methods:

    * :meth:`add` — store a single transition.
    * :meth:`sample` — return a batch of experience for training.
    * :meth:`reset` — clear or reset the buffer for the next collection cycle.
    """

    @abstractmethod
    def add(self, **kwargs) -> None:
        """Store a single transition in the buffer.

        Parameters
        ----------
        **kwargs
            Transition data whose keys depend on the concrete buffer
            implementation (e.g. ``obs``, ``action``, ``reward``,
            ``done``, ``value``, ``logprob``, ``action_mask``,
            ``ordering``).
        """
        ...

    @abstractmethod
    def sample(self) -> Any:
        """Return a batch of experience suitable for learner training.

        The exact return type is implementation-specific.  For example,
        :class:`~eos.buffers.ppo_rollout.PPORolloutBuffer` returns a
        :class:`~eos.buffers.ppo_rollout.PPOBatch` named tuple.

        Returns
        -------
        Any
            A batch of experience data.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the buffer so it can be re-used for the next rollout.

        Typically this just resets the internal step counter without
        deallocating storage, so that the pre-allocated tensors can be
        overwritten in the next collection cycle.
        """
        ...
