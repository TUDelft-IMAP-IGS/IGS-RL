"""Core abstractions for EOS.

This package defines the abstract base classes that form the backbone of
the modular architecture.  Every concrete component (model, controller,
learner, runner, buffer, experiment) inherits from one of these ABCs,
ensuring a consistent interface across different algorithm implementations.

Classes
-------
* :class:`~eos.core.model.Model` — neural-network base (extends ``nn.Module``).
* :class:`~eos.core.controller.Controller` — action selection at inference time.
* :class:`~eos.core.learner.Learner` — parameter updates from experience batches.
* :class:`~eos.core.runner.Runner` — environment interaction and transition collection.
* :class:`~eos.core.replay_buffer.Buffer` — experience storage and sampling.
* :class:`~eos.core.experiment.Experiment` — top-level experiment orchestrator.
"""
