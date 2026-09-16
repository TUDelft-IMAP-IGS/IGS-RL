"""PPO experiment orchestrator.

Ties together all PPO components (runner, learner, controller, buffer) into
a training loop with logging and checkpointing.
"""

# type: ignore
import copy
import random
import signal
import time
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import OmegaConf

import wandb
from eos.buffers.ppo_rollout import PPORolloutBuffer
from eos.config import EOSConfig
from eos.controllers.ppo import PPOController
from eos.core.experiment import Experiment
from eos.envs.factory import make_env
from eos.learners.ppo import PPOLearner
from eos.runners.ppo import PPORunner
from eos.utils.debug import configure as configure_debug
from eos.utils.profiling import StepTimer


class PPOExperiment(Experiment):
    """Orchestrates the synchronous PPO training loop.

    Lifecycle
    ---------
    1. **Setup** — create environments, model, controller, runner, learner,
       and rollout buffer.
    2. **Training loop** — for each update iteration:
       a. Collect a rollout (runner → buffer).
       b. Compute GAE advantages.
       c. Sample the buffer and train (learner).
       d. Checkpoint the best model.
       e. Log metrics to WandB.
    3. **Teardown** — evaluate the best checkpoint, log final results,
       and close environments.

    Parameters
    ----------
    cfg : EOSConfig
        Fully resolved experiment configuration.
    """

    def __init__(self, cfg: EOSConfig):
        self.cfg = cfg
        self.cfg.env.beta = self.cfg.learner.beta

        self.run_name = cfg.run_name

        # Flag for graceful termination
        self.stop_requested = False
        # Intercept Ctrl+C (SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

        self.setup()

    def _handle_interrupt(self, signum, frame):
        """Catch Ctrl+C and request a graceful shutdown."""
        if self.stop_requested:
            logger.warning("Forced exit requested! Bypassing graceful shutdown...")
            raise KeyboardInterrupt

        logger.info(
            "Interrupt received! Finishing current update... (Press Ctrl+C again to force quit)"
        )
        self.stop_requested = True

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise all components from the resolved config."""
        cfg = self.cfg

        configure_debug(cfg.debug)

        # Derived hyper-parameters
        cfg.learner.batch_size = int(cfg.num_envs * cfg.num_steps)
        cfg.learner.minibatch_size = int(
            cfg.learner.batch_size // cfg.learner.num_minibatches
        )
        cfg.learner.num_iterations = cfg.total_timesteps // cfg.learner.batch_size

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.cuda else "cpu"
        )

        # Reproducibility
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic

        self._setup_envs()
        self._setup_components()
        self._setup_buffer()

        self.global_step = 0
        self.best_score = None
        self.best_ckpt_path = None
        self.best_update = None
        self._return_ema = None
        self._next_ckpt_eval = self.cfg.checkpoint.eval_after_steps
        self._next_snapshot = int(
            getattr(self.cfg.checkpoint, "save_every_steps", 0) or 0
        )
        self._snapshot_index = 0
        # Immutable reference for entropy annealing (the learner shares the
        # cfg object we anneal into, so we can't re-read ent_coef as the base).
        self._init_ent_coef = self.cfg.learner.ent_coef

        if cfg.checkpoint.enable:
            if cfg.track and wandb.run is not None:
                self.checkpoint_dir = Path(cfg.checkpoint.save_dir) / str(self.run_name)
            else:
                self.checkpoint_dir = Path(cfg.checkpoint.save_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

    def _setup_envs(self) -> None:
        """Create the vectorised training environment."""
        cfg = self.cfg

        if cfg.env.render.mode == "human" and cfg.num_envs != 1:
            raise ValueError(
                "Human rendering requires num_envs=1. "
                "Set num_envs=1 or disable render.mode."
            )

        env_fns = [
            make_env(cfg, i, cfg.capture_video, self.run_name, cfg.env.render.mode)
            for i in range(cfg.num_envs)
        ]

        if cfg.env.render.mode == "human":
            self.envs = gym.vector.SyncVectorEnv(env_fns)
        else:
            # Use the "spawn" start method rather than the platform default
            # ("fork" on Linux).  Workers are forked *after* wandb.init() and
            # CUDA initialisation, both of which leave background threads /
            # locks in a fork-hostile state.  A forked child can inherit a
            # lock held by a wandb/logging thread and deadlock on its first
            # log/IO call, freezing the run on the main process's blocking
            # pipe.recv() with no recovery.  "spawn" gives each worker a clean
            # interpreter, eliminating the inherited-lock deadlock at the
            # cost of slower worker startup.
            self.envs = gym.vector.AsyncVectorEnv(env_fns, context="spawn")

        # --- PFM multi-objective reward normalization (main-process wrapper) ---
        pfm_cfg = getattr(cfg.env.reward, "pfm", None)
        if pfm_cfg is not None and pfm_cfg.enabled:
            from eos.envs.simple_monopile_transport.pfm import (
                PFMVectorWrapper,
                resolve_preference_bounds,
            )

            resolve_preference_bounds(pfm_cfg, cfg.env)
            self.envs = PFMVectorWrapper(self.envs, pfm_cfg)

        # --- Shared vector-level reward normalization (outermost wrapper) ---
        if cfg.normalize_rewards:
            from eos.envs.factory import wrap_normalize_reward

            self.envs = wrap_normalize_reward(self.envs, gamma=cfg.learner.gamma)

    def _setup_components(self) -> None:
        """Instantiate model, controller, runner, and learner."""
        cfg = self.cfg

        ModelClass = hydra.utils.get_class(cfg.model._target_)
        self.model = ModelClass(self.envs, cfg.model).to(self.device)

        self.controller = PPOController(self.model, cfg.controller, self.device)
        self.runner = PPORunner(
            self.envs,
            self.controller,
            seed=cfg.seed,
            use_action_masks=bool(cfg.env.use_action_masks),
        )
        self.learner = PPOLearner(self.model, cfg.learner)

    def _setup_buffer(self) -> None:
        """Create the rollout buffer with shapes matching the action space.

        When the action space is ``MultiDiscrete`` the buffer automatically
        stores vessel orderings (needed by the autoregressive replay path).
        No explicit ``autoregressive_actions`` check is required.
        """
        cfg = self.cfg
        obs_shape = self.envs.single_observation_space.shape
        action_shape = self.envs.single_action_space.shape  # () for Discrete
        action_space = self.envs.single_action_space
        action_dim = action_space.n if hasattr(action_space, "n") else None

        action_mask_shape = None
        n_vessels = None

        if cfg.env.use_action_masks:
            if isinstance(action_space, gym.spaces.MultiDiscrete):
                nvec = np.asarray(action_space.nvec, dtype=int)
                if not np.all(nvec == nvec[0]):
                    raise ValueError(
                        "MultiDiscrete action space requires equal "
                        "per-dimension sizes for masking."
                    )
                action_mask_shape = (int(nvec.shape[0]), int(nvec[0]))
            elif action_dim is not None:
                action_mask_shape = (action_dim,)
            else:
                raise ValueError(
                    "Action masking is enabled, but the action space "
                    "does not expose `n` and is not MultiDiscrete."
                )

        # MultiDiscrete → always autoregressive → need ordering storage
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            n_vessels = int(np.asarray(action_space.nvec).shape[0])

        self.buffer = PPORolloutBuffer(
            cfg.num_steps,
            cfg.num_envs,
            obs_shape,
            action_shape,
            self.device,
            action_dim=action_dim,
            action_mask_shape=action_mask_shape,
            n_vessels=n_vessels,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_checkpoint_metric(run_metrics: dict, metric_key: str):
        """Look up a (possibly nested) metric value from the run metrics dict.

        Supports ``"group/key"`` syntax where *group* is mapped to the
        corresponding sub-dict (``rollout_metrics``, ``env_metrics``, or
        ``rich_metrics``).
        """
        if not metric_key:
            return None

        if "/" in metric_key:
            group, key = metric_key.split("/", 1)
            group_map = {
                "rollout": "rollout_metrics",
                "env": "env_metrics",
                "scenario": "rich_metrics",
            }
            metrics_group = run_metrics.get(group_map.get(group, group), {})
            return metrics_group.get(key)

        if metric_key in run_metrics:
            return run_metrics.get(metric_key)

        for group_name in ("rollout_metrics", "env_metrics", "rich_metrics"):
            metrics_group = run_metrics.get(group_name, {})
            if metric_key in metrics_group:
                return metrics_group.get(metric_key)

        return None

    def _maybe_checkpoint(self, run_metrics: dict, update: int) -> None:
        """Save the model if the tracked metric improved, and run periodic eval.

        Evaluation is decoupled from checkpointing: it fires every
        ``eval_after_steps`` environment steps regardless of whether the
        metric improved.

        Both PFM and non-PFM runs use metric-based best-tracking: ``best.pt``
        is overwritten whenever ``checkpoint.metric`` reaches a new best (per
        ``checkpoint.mode``).  For PFM, point ``checkpoint.metric`` at a
        *stationary* signal such as ``env/pfm/pref_weighted`` (mode ``max``);
        the default ``episodic_return_ema`` is non-stationary under PFM.
        """
        cfg = self.cfg
        if not cfg.checkpoint.enable or not run_metrics:
            return

        # --- Periodic evaluation (independent of checkpoint logic) ---
        self._next_ckpt_eval -= cfg.num_steps * cfg.num_envs
        should_eval = self._next_ckpt_eval <= 0
        if should_eval:
            self._next_ckpt_eval = self.cfg.checkpoint.eval_after_steps

        # --- Checkpoint saving: best-tracking on the configured metric ---
        metric_value = self._extract_checkpoint_metric(
            run_metrics, cfg.checkpoint.metric
        )
        if metric_value is not None:
            if cfg.checkpoint.mode == "min":
                is_better = self.best_score is None or metric_value < self.best_score
            else:
                is_better = self.best_score is None or metric_value > self.best_score

            if is_better:
                self.best_score = float(metric_value)
                self.best_update = int(update)
                self._save_best_checkpoint()

        # --- Periodic snapshot: unconditional, independent of the metric ---
        every = int(getattr(cfg.checkpoint, "save_every_steps", 0) or 0)
        if every > 0:
            self._next_snapshot -= cfg.num_steps * cfg.num_envs
            if self._next_snapshot <= 0:
                self._next_snapshot = every
                self._save_snapshot_checkpoint()

        # Run eval on schedule regardless of metric improvement
        if should_eval:
            self._evaluate_and_log_best()

    def _save_best_checkpoint(self) -> None:
        """Save current model as the best checkpoint."""
        cfg = self.cfg
        ckpt_path = self.checkpoint_dir / "best.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.learner.optimizer.state_dict(),
                "global_step": self.global_step,
                "update": self.best_update,
                "score": self.best_score,
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            ckpt_path,
        )
        self.best_ckpt_path = ckpt_path

        if cfg.track:
            wandb.log(
                {
                    "checkpoint/best_score": self.best_score,
                    "checkpoint/update": self.best_update,
                    "global_step": self.global_step,
                }
            )
            try:
                artifact = wandb.Artifact(
                    name=f"{self.run_name}-best",
                    type="model",
                    metadata={
                        "score": self.best_score,
                        "update": self.best_update,
                        "global_step": self.global_step,
                        "metric": cfg.checkpoint.metric,
                        "mode": cfg.checkpoint.mode,
                    },
                )
                artifact.add_file(str(ckpt_path))
                wandb.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"Failed to log best-checkpoint artifact: {e}")

    def _save_snapshot_checkpoint(self) -> None:
        """Save an unconditional periodic snapshot of the current policy.

        Deliberately kept off the ``best.pt`` / ``<run>-best`` path: downstream
        evaluation treats ``best.pt`` / the ``-best`` artifact version as *the*
        metric-selected best checkpoint, so periodic snapshots are logged to a
        distinct snapshot stream to prevent overwriting or confusing best models.
        """
        cfg = self.cfg
        ckpt_path = self.checkpoint_dir / "snapshot.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.learner.optimizer.state_dict(),
                "global_step": self.global_step,
                "update": self._snapshot_index,
                "score": None,  # snapshots are unconditional, not metric-selected
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            ckpt_path,
        )
        self._snapshot_index += 1

        if cfg.track:
            try:
                artifact = wandb.Artifact(
                    name=f"{self.run_name}-snapshot",
                    type="model",
                    metadata={
                        "global_step": self.global_step,
                        "snapshot_index": self._snapshot_index,
                        "save_every_steps": cfg.checkpoint.save_every_steps,
                    },
                )
                artifact.add_file(str(ckpt_path))
                wandb.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"Failed to log snapshot artifact: {e}")

    def _evaluate_and_log_best(self) -> None:
        """Run a standalone evaluation episode and log domain visualizations."""
        from eos.runners.micro_step import run_evaluation_episode
        from eos.utils.eval_logger import log_eval_to_wandb, print_eval_summary

        cfg = self.cfg
        eval_deterministic = cfg.eval.deterministic
        eval_num_episodes = cfg.eval.num_episodes
        score_str = f"{self.best_score:.2f}" if self.best_score is not None else "N/A"
        mode_str = "deterministic" if eval_deterministic else "stochastic"
        logger.info(
            f"Running {mode_str} evaluation ({eval_num_episodes} ep) "
            f"on latest checkpoint (score: {score_str})"
        )

        eval_envs = gym.vector.SyncVectorEnv(
            [make_env(cfg, 0, False, f"{self.run_name}-eval", None)]
        )

        pfm_cfg = getattr(cfg.env.reward, "pfm", None)
        if pfm_cfg is not None and pfm_cfg.enabled:
            from eos.envs.simple_monopile_transport.pfm import PFMVectorWrapper

            # Reuse the training tracker's current μ/σ, but on a *copy* so the
            # eval episodes' Welford/Polyak updates (and _active_count) do not
            # mutate the live training statistics.
            training_tracker = getattr(self.envs, "_tracker", None)
            eval_tracker = (
                copy.deepcopy(training_tracker)
                if training_tracker is not None
                else None
            )
            eval_envs = PFMVectorWrapper(eval_envs, pfm_cfg, tracker=eval_tracker)

        try:
            eval_results = run_evaluation_episode(
                envs=eval_envs,
                controller=self.controller,
                seed=cfg.seed,
                num_episodes=eval_num_episodes,
                deterministic=eval_deterministic,
                use_action_masks=bool(cfg.env.use_action_masks),
                capture_inventory=True,
                trace_actions=True,
            )

            print_eval_summary(eval_results)
            if cfg.track:
                import wandb

                log_eval_to_wandb(
                    eval_results, wandb.run.dir, global_step=self.global_step
                )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
        finally:
            eval_envs.close()

    def _run_final_eval(self) -> dict:
        """Run a deterministic eval on the best (or final) model.

        If a best checkpoint exists on disk, loads those weights first so
        the evaluation reflects the best policy found during training.
        Otherwise evaluates the current (final) model weights. When
        ``cfg.eval.on_final_weights`` is ``True``, best.pt is never loaded
        and the current (final) weights are evaluated regardless.

        Eval settings (num_episodes, deterministic) come from ``cfg.eval``.

        Returns
        -------
        dict
            The eval summary dict containing all metrics (elapsed time,
            return, costs, goals, etc.).  Empty dict on failure.
        """
        from eos.runners.micro_step import run_evaluation_episode

        cfg = self.cfg

        # Load best checkpoint if available — unless we explicitly want the
        # FINAL (latest) weights (stability sweeps measure the checkpoint you
        # actually end up with, not the best-so-far snapshot).
        if cfg.eval.on_final_weights:
            logger.info(
                "Evaluating FINAL (latest) weights (eval.on_final_weights=True); "
                "best.pt not loaded."
            )
        elif self.best_ckpt_path is not None and self.best_ckpt_path.exists():
            ckpt = torch.load(
                self.best_ckpt_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded best checkpoint for final eval: {self.best_ckpt_path}")
        else:
            logger.info("No best checkpoint found; evaluating final weights.")

        eval_envs = gym.vector.SyncVectorEnv(
            [make_env(cfg, 0, False, f"{self.run_name}-final-eval", None)]
        )

        try:
            eval_results = run_evaluation_episode(
                envs=eval_envs,
                controller=self.controller,
                seed=cfg.seed,
                num_episodes=cfg.eval.num_episodes,
                deterministic=cfg.eval.deterministic,
                use_action_masks=bool(cfg.env.use_action_masks),
                capture_inventory=False,
                trace_actions=False,
            )
            summary = eval_results.get("summary", {})
            logger.info(f"Final eval summary: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Final eval failed: {e}")
            return {}
        finally:
            eval_envs.close()

    def _save_final_checkpoint(self) -> None:
        """Save the model state at the end of training.

        This always runs (even if checkpointing by metric is disabled) so
        that ``scripts/eval.py`` can be pointed at either ``best.pt`` or
        ``final.pt`` for post-training evaluation.
        """
        cfg = self.cfg
        ckpt_dir = (
            self.checkpoint_dir
            if self.checkpoint_dir is not None
            else Path(HydraConfig.get().runtime.output_dir)
        )
        final_path = ckpt_dir / "final.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.learner.optimizer.state_dict(),
                "global_step": self.global_step,
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            final_path,
        )
        logger.info(f"Final checkpoint saved to {final_path}")

        if cfg.track:
            try:
                artifact = wandb.Artifact(
                    name=f"{self.run_name}-final",
                    type="model",
                    metadata={"global_step": self.global_step},
                )
                artifact.add_file(str(final_path))
                wandb.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"Failed to log final-checkpoint artifact: {e}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def _run_experiment_loop(self) -> dict:
        """Execute the PPO training loop with logging and checkpointing.

        At the end of training a ``final.pt`` checkpoint is always saved
        alongside the metric-based ``best.pt``.  Post-training evaluation
        should be run via ``scripts/eval.py`` which loads either checkpoint.
        """
        cfg = self.cfg
        run_name = self.run_name
        start_time = time.time()

        try:
            for update in range(1, cfg.learner.num_iterations + 1):
                if self.stop_requested:
                    logger.warning(
                        "Experiment interrupted by user. Initiating clean teardown."
                    )
                    break

                # ── Learning-rate annealing ───────────────────────────
                if cfg.learner.anneal_lr:
                    frac = 1.0 - (update - 1.0) / cfg.learner.num_iterations
                    lrnow = (
                        cfg.learner.init_lr - cfg.learner.final_lr
                    ) * frac + cfg.learner.final_lr
                    self.learner.optimizer.param_groups[0]["lr"] = lrnow

                # ── Entropy-coefficient annealing ─────────────────────
                if cfg.learner.anneal_ent:
                    frac = 1.0 - (update - 1.0) / cfg.learner.num_iterations
                    self.learner.cfg.ent_coef = (
                        self._init_ent_coef - cfg.learner.ent_coef_final
                    ) * frac + cfg.learner.ent_coef_final

                update_timer = StepTimer()

                # ── 1. Collect rollout ────────────────────────────────
                with update_timer.phase("rollout"):
                    context = self.runner.run(self.buffer, cfg.num_steps)
                self.global_step += cfg.num_steps * cfg.num_envs

                run_metrics = context.get("metrics", {})

                # --- Compute EMA of Returns ---
                if run_metrics.get("episodic_returns") is not None:
                    alpha = 0.7  # Smoothing factor (lower = smoother)
                    curr_return = float(run_metrics["episodic_returns"])
                    if self._return_ema is None:
                        self._return_ema = curr_return
                    else:
                        self._return_ema = (
                            alpha * curr_return + (1 - alpha) * self._return_ema
                        )

                    # Inject back into metrics so the checkpoint system can see it!
                    run_metrics["episodic_return_ema"] = self._return_ema

                self._maybe_checkpoint(run_metrics, update)

                # ── Optuna pruning ────────────────────────────────────
                if self._optuna_trial is not None and cfg.tune.prune_after_steps > 0:
                    tune_value = self._extract_checkpoint_metric(
                        run_metrics, cfg.tune.metric
                    )
                    if (
                        tune_value is not None
                        and not np.isnan(tune_value)
                        and self.global_step >= cfg.tune.prune_after_steps
                    ):
                        import optuna as _optuna

                        self._optuna_trial.report(float(tune_value), self.global_step)
                        if self._optuna_trial.should_prune():
                            logger.info(
                                f"Trial pruned at step {self.global_step} "
                                f"(metric={tune_value:.2f})"
                            )
                            raise _optuna.exceptions.TrialPruned()

                # Log episodic stats immediately
                if run_metrics.get("episodic_returns") is not None and cfg.track:
                    log_episodic = {
                        "charts/episodic_return": run_metrics["episodic_returns"],
                        "global_step": self.global_step,
                    }
                    if run_metrics.get("episodic_lengths") is not None:
                        log_episodic["charts/episodic_length"] = run_metrics[
                            "episodic_lengths"
                        ]
                    if run_metrics.get("episodic_time") is not None:
                        log_episodic["charts/episodic_time"] = run_metrics[
                            "episodic_time"
                        ]
                    if run_metrics.get("episodic_return_ema") is not None:
                        log_episodic["charts/episodic_return_ema"] = run_metrics[
                            "episodic_return_ema"
                        ]

                    # Episode-level success rate (fraction of completed episodes that succeeded)
                    if run_metrics.get("success_rate") is not None:
                        log_episodic["charts/success_rate"] = run_metrics[
                            "success_rate"
                        ]

                    # Domain-level terminal metrics
                    if run_metrics.get("episodic_elapsed_hours") is not None:
                        log_episodic["charts/elapsed_time_hours"] = run_metrics[
                            "episodic_elapsed_hours"
                        ]
                    if run_metrics.get("episodic_storage_cost") is not None:
                        log_episodic["charts/storage_cost"] = run_metrics[
                            "episodic_storage_cost"
                        ]
                    if run_metrics.get("episodic_total_cost") is not None:
                        log_episodic["charts/total_cost"] = run_metrics[
                            "episodic_total_cost"
                        ]
                    if run_metrics.get("episodic_travel_cost") is not None:
                        log_episodic["charts/travel_cost"] = run_metrics[
                            "episodic_travel_cost"
                        ]

                    wandb.log(log_episodic)

                # ── 2. Compute GAE advantages ─────────────────────────
                with update_timer.phase("gae"):
                    with torch.no_grad():
                        next_obs_tensor = torch.as_tensor(
                            context["next_obs"], device=self.device, dtype=torch.float32
                        )
                        next_value = self.learner.model.get_value(
                            next_obs_tensor
                        ).reshape(1, -1)

                    self.buffer.compute_advantages(
                        next_value,
                        context["next_done"],
                        cfg.learner.beta,
                        cfg.learner.gae_lambda,
                    )

                # ── 3. Sample & train ─────────────────────────────────
                with update_timer.phase("train"):
                    batch_data = self.buffer.sample()
                    train_metrics = self.learner.train(batch_data)

                # ── 4. Reset buffer for next iteration ────────────────
                self.buffer.reset()

                # ── 5. Logging ────────────────────────────────────────
                if update % cfg.log_every_updates == 0:
                    sps = int(self.global_step / (time.time() - start_time))
                    logger.info(
                        f"Step: {self.global_step} | SPS: {sps} | Return: {run_metrics.get('episodic_returns', 'NO RETURN')}"
                    )
                    update_timer.log_summary(f"Update {update} timing")

                    if cfg.track:
                        log_data = {
                            "global_step": self.global_step,
                            "charts/SPS": sps,
                            "train/learning_rate": self.learner.optimizer.param_groups[
                                0
                            ]["lr"],
                        }

                        # Learner metrics → "train/"
                        for k, v in train_metrics.items():
                            log_data[f"train/{k}"] = v

                        # Rollout metrics → "rollout/"
                        if "rollout_metrics" in run_metrics:
                            for k, v in run_metrics["rollout_metrics"].items():
                                if v is not None:
                                    log_data[f"rollout/{k}"] = v

                        # Env metrics → "env/"
                        if cfg.log_env_info and "env_metrics" in run_metrics:
                            for k, v in run_metrics["env_metrics"].items():
                                log_data[k] = v

                        # Rich metrics → "scenario/"
                        if "rich_metrics" in run_metrics:
                            for k, v in run_metrics["rich_metrics"].items():
                                log_data[f"scenario/{k}"] = v

                        # Performance timing → "perf/"
                        log_data.update(update_timer.summary(prefix="perf/update/"))
                        step_timing = run_metrics.get("step_timing")
                        if step_timing:
                            log_data.update(step_timing)
                        macro_step_timing = run_metrics.get("macro_step_timing")
                        if macro_step_timing:
                            log_data.update(macro_step_timing)

                        wandb.log(log_data)

        except KeyboardInterrupt:
            # This will now only catch forced exits (the second Ctrl+C)
            logger.warning("Experiment forcefully interrupted by user.")
        except Exception as e:
            logger.error(f"Experiment failed with error: {e}")
            raise
        finally:
            # ── Teardown — always runs ────────────────────────────────
            self._save_final_checkpoint()

            if hasattr(self, "envs"):
                self.envs.close()

            if self.best_ckpt_path is not None:
                score_str = (
                    f"{self.best_score:.3f}" if self.best_score is not None else "N/A"
                )
                logger.info(
                    f"Best checkpoint: {self.best_ckpt_path} "
                    f"(score={score_str}, update={self.best_update})"
                )
                logger.info(
                    "Run evaluation with:  "
                    f"uv run scripts/eval.py --checkpoint {self.best_ckpt_path}"
                )

            # ── Final eval ────────────────────────────────────────────
            # Always run a deterministic eval at the end of training.
            # Returns the eval summary dict to the caller.
            eval_summary = self._run_final_eval()

            logger.success(f"Experiment {run_name} completed.")

            return eval_summary
