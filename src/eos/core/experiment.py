"""Abstract base class for experiment orchestrators.

An experiment ties together all framework components (environment, model,
controller, runner, learner, buffer) into a coherent training or
evaluation loop.  Concrete subclasses implement
:meth:`_run_experiment_loop` to define the specific workflow — for
example, :class:`~eos.experiments.ppo.PPOExperiment` runs a synchronous
PPO training loop, while :class:`~eos.experiments.random.RandomExperiment`
executes a random-action baseline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Experiment(ABC):
    """Base class for all EOS experiments.

    Subclasses must implement :meth:`_run_experiment_loop` which contains
    the core logic (training iterations, evaluation episodes, logging,
    checkpointing, etc.).  The public entry point is :meth:`run`, which
    simply delegates to the abstract method.
    """

    def run(self, trial: Any = None) -> Any:
        """Execute the experiment.

        This is the public entry point called by the Hydra launcher.
        It delegates to the subclass-specific :meth:`_run_experiment_loop`.

        Parameters
        ----------
        trial : optuna.trial.Trial | None
            Optional Optuna trial for intermediate metric reporting and
            pruning.  Stored as ``self._optuna_trial`` for subclasses.

        Returns
        -------
        Any
            The eval summary dict (for PPO) or a scalar (for other experiments).
        """
        self._optuna_trial = trial
        return self._run_experiment_loop()

    @abstractmethod
    def _run_experiment_loop(self) -> Any:
        """Implement the experiment's main loop.

        Subclasses should handle their own setup, teardown, logging,
        and error handling within this method (or in helper methods
        called from here).
        """
        ...
