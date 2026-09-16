"""Abstract base class for environment runners.

A runner interacts with the (vectorised) environment by querying the
:class:`~eos.core.controller.Controller` for actions at each step and
storing the resulting transitions in a :class:`~eos.core.replay_buffer.Buffer`.
It also collects episode-level and step-level metrics that are returned
to the experiment orchestrator for logging and checkpointing.

Under the **Intent-Based Micro-Stepping (AEC)** architecture, each
"macro-step" collected by the runner comprises:

1. **Ordering** — the controller determines the sequence in which idle
   vessels commit their actions (Phase 1).
2. **Per-vessel micro-steps** — for each idle vessel in order, the
   controller selects an action and registers it via ``env.step()``
   (no time advance, reward = 0).
3. **DES advance** — ``yield_until_idle()`` fast-forwards the
   discrete-event simulation until at least one vessel becomes idle
   again, producing the real reward and next observation.

The runner delegates the micro-stepping mechanics to
:class:`~eos.runners.micro_step.MicroStepCollector` and focuses on
buffer storage, metric aggregation, and episode bookkeeping.
"""

from abc import ABC, abstractmethod

from eos.core.controller import Controller
from eos.core.replay_buffer import Buffer


class Runner(ABC):
    """Collects environment transitions by stepping with a controller.

    Subclasses implement :meth:`run` to define the specific rollout
    procedure (e.g. fixed-length rollouts for PPO, episode-based
    collection, etc.).  All runners share the same two-phase AEC
    controller interface, but differ in how they store transitions
    and compute training targets.

    Parameters
    ----------
    envs :
        A (vectorised) gymnasium environment instance.
    controller : Controller
        The controller used to select actions at each step.  Must
        implement the two-phase AEC interface (:meth:`get_ordering`
        and :meth:`get_action_and_value`).
    """

    def __init__(self, envs, controller: Controller):
        self.envs = envs
        self.controller = controller

    @abstractmethod
    def run(self, buffer: Buffer, steps: int):
        """Collect ``steps`` macro-transitions and store them in ``buffer``.

        Each macro-transition encompasses a full ordering → micro-steps →
        DES-advance cycle.  The critic value recorded for each transition
        is taken from the *first* micro-step (the moment before any vessel
        has committed an intent).

        Parameters
        ----------
        buffer : Buffer
            The replay / rollout buffer to populate with transition data.
        steps : int
            Number of macro-steps (decision epochs) to collect.

        Returns
        -------
        dict
            A result dictionary that typically includes:

            - ``"next_obs"`` — the observation after the last collected step.
            - ``"next_done"`` — the done flag after the last collected step.
            - ``"metrics"`` — aggregated rollout metrics (episodic returns,
              lengths, environment-specific info, etc.).
        """
        ...
