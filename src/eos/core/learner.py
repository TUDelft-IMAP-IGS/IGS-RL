"""Abstract base class for model update strategies (learners).

A learner encapsulates the training algorithm (e.g. PPO) and is
responsible for computing losses, performing gradient updates, and
returning training metrics.  It receives a batch of experience from the
:class:`~eos.core.replay_buffer.Buffer` and updates the parameters of
the :class:`~eos.core.model.Model` in-place.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from eos.core.model import Model


class Learner(ABC):
    """Updates a :class:`Model` from a batch of collected experience.

    Subclasses implement :meth:`train` to define the specific optimisation
    procedure (loss computation, gradient steps, learning-rate schedules,
    etc.).

    Parameters
    ----------
    model : Model
        The actor-critic model whose parameters will be optimised.
    """

    def __init__(self, model: Model):
        self.model = model

    @abstractmethod
    def train(self, batch: Any) -> Dict:
        """Run one or more gradient-update steps on the given batch.

        Parameters
        ----------
        batch : Any
            A batch of experience produced by the replay buffer's
            ``sample()`` method.  The exact type depends on the buffer
            implementation (e.g. a :class:`~eos.buffers.ppo_rollout.PPOBatch`
            named tuple for PPO).

        Returns
        -------
        Dict
            Training metrics (e.g. losses, KL divergence, gradient norms,
            explained variance, learning rate) that can be logged by the
            experiment orchestrator.
        """
        ...
