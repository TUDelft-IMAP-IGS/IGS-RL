import copy
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

import hydra
import torch
from gymnasium.envs.registration import register
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import optuna
import wandb
from eos.core.experiment import Experiment

# Ensure src is in path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import signal

from eos.config import EOSConfig, register_configs

register_configs()

# =========================================================================
# MULTIRUN GRACEFUL SHUTDOWN FIX
# =========================================================================
if "-m" in sys.argv or "--multirun" in sys.argv:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
# =========================================================================

logging.getLogger("boka_eventsymphony").setLevel(logging.WARNING)


def run_experiment(cfg: EOSConfig, run_name: str, trial=None) -> dict:
    """Core execution logic for a single experiment run.

    Parameters
    ----------
    cfg : EOSConfig
        Fully resolved experiment configuration.
    run_name : str
        Unique name for this run (used for WandB and logging).
    trial : optuna.Trial | None
        Optional Optuna trial for reporting intermediate metrics and pruning.

    Returns
    -------
    dict
        Eval summary dict from the final deterministic evaluation.
        Contains metrics like ``sim_elapsed_hours_mean``, ``return_mean``, etc.
    """
    safe_log_level = str(cfg.log_level).upper()
    safe_dir_level = str(cfg.log_dir_level).upper()

    # 1. Atomically configure the console logger
    logger.configure(handlers=[{"sink": sys.stderr, "level": safe_log_level}])

    # 2. Initialize WandB
    if cfg.track:
        # Check if we are running on a Databricks cluster
        if "DATABRICKS_RUNTIME_VERSION" in os.environ:
            # Try to get the W&B key from the environment first (set in notebook)
            wandb_key = os.environ.get("WANDB_API_KEY")

            if not wandb_key:
                # Try to get dbutils from the IPython notebook context
                try:
                    import IPython  # type: ignore

                    ip_shell = IPython.get_ipython()
                    if ip_shell is not None:
                        dbutils = ip_shell.user_ns.get("dbutils")
                    else:
                        dbutils = None
                except ImportError:
                    dbutils = None

                if dbutils is None:
                    raise RuntimeError(
                        "Cannot access Databricks secrets. Either run from a "
                        "Databricks notebook or set the WANDB_API_KEY environment "
                        "variable before launching the script."
                    )

                # Securely fetch the key from the Databricks vault
                wandb_key = dbutils.secrets.get(scope="wandb", key="api_key")

            # Authenticate the cluster with WandB
            wandb.login(key=wandb_key)

        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            sync_tensorboard=False,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            group=cfg.exp_name,
            tags=[f"seed_{cfg.seed}", cfg.exp_name],
            job_type="train",
            monitor_gym=True,
            save_code=True,
            reinit=True,  # Crucial: Allows multiple WandB runs in the same Python process
        )

        log_file = Path(wandb.run.dir) / "run.log"
        logger.add(log_file, level=safe_dir_level)
        wandb.save(str(log_file), policy="live")

    if OmegaConf.select(cast(DictConfig, cfg), "_target_") is None:
        raise ValueError("Experiment config must define a '_target_' class!")

    ExperimentClass = hydra.utils.get_class(cfg._target_)
    experiment: Experiment = ExperimentClass(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    logger.info(f"Using device: {device}")

    # Run the experiment orchestrator (pass trial for pruning support)
    eval_summary = experiment.run(trial=trial)

    if cfg.track:
        wandb.finish()

    return eval_summary if isinstance(eval_summary, dict) else {}


# =========================================================================
# OPTUNA OBJECTIVES
# =========================================================================


def objective_pfm(trial: optuna.Trial, base_cfg: EOSConfig) -> float:
    """Optuna objective for PFM scenarios.

    Search space covers PPO training dynamics and PFM-specific parameters.
    Model architecture is held fixed.
    """
    cfg = copy.deepcopy(base_cfg)

    # ── PPO Training Dynamics ─────────────────────────────────────────────
    cfg.learner.init_lr = trial.suggest_float("init_lr", 1e-5, 1e-3, log=True)
    cfg.learner.ent_coef = trial.suggest_float("ent_coef", 0.005, 0.15, log=True)
    cfg.learner.clip_coef = trial.suggest_categorical("clip_coef", [0.1, 0.2, 0.3])
    cfg.learner.update_epochs = trial.suggest_categorical("update_epochs", [2, 4, 8])
    cfg.learner.gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    cfg.learner.max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    cfg.learner.vf_coef = trial.suggest_float("vf_coef", 0.25, 1.0)

    # ── PFM Parameters ────────────────────────────────────────────────────
    cfg.env.reward.pfm.polyak_tau = trial.suggest_float(
        "polyak_tau", 1e-4, 0.01, log=True
    )
    cfg.env.reward.pfm.burn_in_trajectories = trial.suggest_categorical(
        "burn_in_trajectories", [50, 100, 200, 500]
    )
    cfg.env.reward.pfm.phi_optimism = trial.suggest_float("phi_optimism", 1.0, 5.0)
    cfg.env.reward.pfm.completion_baseline = trial.suggest_float(
        "completion_baseline", 5.0, 15.0
    )

    # ── Run name ──────────────────────────────────────────────────────────
    now = datetime.now()
    run_name = f"{cfg.exp_name}__t{trial.number}__{now.strftime('%Y%m%dT%H%M%S')}"
    cfg.run_name = run_name

    # ── Execute ───────────────────────────────────────────────────────────
    try:
        eval_summary = run_experiment(cfg, run_name, trial=trial)
        metric_key = base_cfg.tune.metric
        value = eval_summary.get(metric_key)
        if value is None:
            raise optuna.exceptions.TrialPruned()
        return float(value)
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        logger.exception(f"Trial {trial.number} failed: {e}")
        raise optuna.exceptions.TrialPruned()


def objective_time_only(trial: optuna.Trial, base_cfg: EOSConfig) -> float:
    """Optuna objective for time-only scenarios (no PFM).

    Runs multiple seeds per trial (Option C pruning): each completed seed
    is reported as a step to the pruner.  Bad configs are killed early
    after the first seed instead of burning compute on all seeds.

    Search space: PPO dynamics + milestone shaping + SMDP beta.
    Objective: mean elapsed_time_hours across seeds (minimize).
    """
    cfg = copy.deepcopy(base_cfg)
    time_budget = cfg.env.time_budget  # penalty for failed seeds

    # ── PPO Training Dynamics ─────────────────────────────────────────────
    cfg.learner.init_lr = trial.suggest_float("init_lr", 5e-5, 5e-4, log=True)
    cfg.learner.ent_coef = trial.suggest_float("ent_coef", 0.003, 0.03, log=True)
    cfg.learner.clip_coef = trial.suggest_categorical("clip_coef", [0.1, 0.2, 0.3])
    cfg.learner.update_epochs = trial.suggest_categorical("update_epochs", [6, 8, 12])

    # ── SMDP time discounting ─────────────────────────────────────────────
    cfg.learner.beta = trial.suggest_categorical("beta", [1e-5, 1e-4, 1e-3])
    cfg.env.beta = cfg.learner.beta  # keep env and learner in sync

    # ── Milestone shaping ─────────────────────────────────────────────────
    cfg.env.reward.milestone.phi_max_fraction = trial.suggest_categorical(
        "phi_max_fraction", [0.01, 0.1, 0.2]
    )

    # ── Disable WandB for sweep seeds ─────────────────────────────────────
    cfg.track = False

    # ── Multi-seed execution with pruning ────────────────────────
    seeds = list(base_cfg.tune.seeds)
    metric_key = base_cfg.tune.metric
    scores: list[float] = []

    for i, seed in enumerate(seeds):
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed.seed = seed

        now = datetime.now()
        run_name = (
            f"{cfg.exp_name}__t{trial.number}__s{seed}__{now.strftime('%Y%m%dT%H%M%S')}"
        )
        cfg_seed.run_name = run_name

        try:
            eval_summary = run_experiment(cfg_seed, run_name, trial=None)
            # Extract the tune metric from the eval summary
            value = eval_summary.get(metric_key)
            if value is not None and not (isinstance(value, float) and value != value):
                scores.append(float(value))
            else:
                logger.warning(
                    f"Trial {trial.number} seed {seed}: metric '{metric_key}' "
                    f"not found in eval summary. Available: {list(eval_summary.keys())}"
                )
                scores.append(time_budget)
        except Exception as e:
            logger.warning(f"Trial {trial.number} seed {seed} failed: {e}")
            scores.append(time_budget)

        # Report running mean to Optuna after each seed
        running_mean = sum(scores) / len(scores)
        trial.report(running_mean, step=i)

        # Check if we should prune (bad config → kill early)
        if trial.should_prune():
            logger.info(
                f"Trial {trial.number} pruned after seed {seed} "
                f"(running mean={running_mean:.1f})"
            )
            raise optuna.exceptions.TrialPruned()

    final_mean = sum(scores) / len(scores)
    logger.info(
        f"Trial {trial.number} complete: "
        f"scores={[f'{s:.1f}' for s in scores]}, "
        f"mean={final_mean:.1f}"
    )
    return final_mean


# Assumed mean physical duration of one macro-action (hours). Couples the
# reward-normaliser discount (learner.gamma) to the SMDP per-action discount
# gamma_t = exp(-beta * delta_t): with <delta_t> ~ 14h, a sampled beta implies
# gamma ~ exp(-beta * 14). (GAE discounts with gamma_t, not gamma; gamma only
# feeds the running-return normaliser, so the two should agree.)
AVG_ACTION_HOURS = 14.0


def objective_time_only_smdp(trial: optuna.Trial, base_cfg: EOSConfig) -> float:
    """Optuna objective for the SMDP time-only sweep (Scenario B).

    Search space: init_lr (log 1e-4..5e-4), ent_coef (log 5e-3..0.2),
    gae_lambda (0.90..0.98), beta (log 1e-5..1e-3, with gamma derived from it),
    phi_max_fraction {0.1, 0.25, 0.5}. Objective: mean sim_elapsed_hours_mean
    across seeds (minimize); a missing/NaN eval metric falls back to the time
    budget.
    """
    cfg = copy.deepcopy(base_cfg)
    time_budget = cfg.env.time_budget  # fallback if eval returns no metric

    # ── PPO training dynamics ───────────────────────────────────────
    cfg.learner.init_lr = trial.suggest_float("init_lr", 1e-4, 5e-4, log=True)
    cfg.learner.ent_coef = trial.suggest_float("ent_coef", 5e-3, 0.2, log=True)
    cfg.learner.gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.98)

    # ── SMDP time discounting (beta) + coupled gamma ──────────────────
    beta = trial.suggest_float("beta", 1e-5, 1e-3, log=True)
    cfg.learner.beta = beta
    cfg.env.beta = beta  # env (gamma_t) and learner (GAE) must agree
    cfg.learner.gamma = float(math.exp(-beta * AVG_ACTION_HOURS))

    # ── Milestone shaping ─────────────────────────────────────
    cfg.env.reward.milestone.phi_max_fraction = trial.suggest_categorical(
        "phi_max_fraction", [0.1, 0.25, 0.5]
    )

    # ── Disable WandB for sweep seeds ─────────────────────────────
    cfg.track = False

    # ── Multi-seed execution with Option-C pruning ────────────────────
    seeds = list(base_cfg.tune.seeds)
    metric_key = base_cfg.tune.metric
    scores: list[float] = []

    for i, seed in enumerate(seeds):
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed.seed = seed

        now = datetime.now()
        run_name = (
            f"{cfg.exp_name}__t{trial.number}__s{seed}__{now.strftime('%Y%m%dT%H%M%S')}"
        )
        cfg_seed.run_name = run_name

        try:
            eval_summary = run_experiment(cfg_seed, run_name, trial=None)
            value = eval_summary.get(metric_key)
            if value is not None and not (isinstance(value, float) and value != value):
                scores.append(float(value))
            else:
                logger.warning(
                    f"Trial {trial.number} seed {seed}: metric '{metric_key}' "
                    f"not found in eval summary. Available: {list(eval_summary.keys())}"
                )
                scores.append(time_budget)
        except Exception as e:
            logger.warning(f"Trial {trial.number} seed {seed} failed: {e}")
            scores.append(time_budget)

        running_mean = sum(scores) / len(scores)
        trial.report(running_mean, step=i)
        # With a single seed the only report happens after that seed already
        # consumed its full compute budget, so pruning saves nothing. Only
        # allow between-seed pruning when there is more than one seed.
        if len(seeds) > 1 and trial.should_prune():
            logger.info(
                f"Trial {trial.number} pruned after seed {seed} "
                f"(running mean={running_mean:.1f})"
            )
            raise optuna.exceptions.TrialPruned()

    final_mean = sum(scores) / len(scores)
    logger.info(
        f"Trial {trial.number} complete: gamma={cfg.learner.gamma:.5f} "
        f"scores={[f'{s:.1f}' for s in scores]}, mean={final_mean:.1f}"
    )
    return final_mean


# Maps each PFM objective's info_key to the matching key produced by
# run_evaluation_episode()'s deterministic eval summary.
_PFM_INFO_KEY_TO_SUMMARY = {
    "elapsed_time_hours": "sim_elapsed_hours_mean",
    "cost/storage": "storage_cost_mean",
    "cost/travel": "travel_cost_mean",
    "cost/elapsed_time": "elapsed_time_cost_mean",
}


def _pfm_preference_composite(cfg: EOSConfig, summary: dict) -> float | None:
    """Weighted PFM preference composite in [0, 100] from an eval summary.

    Maps each PFM objective's raw physical metric (read from the deterministic
    eval ``summary``) through its preference function, then returns the
    configured weighted sum (e.g. ``0.5*time_pref + 0.5*storage_pref``).

    A heuristic proxy for the true PFM objective: it scores realized terminal
    metrics in dimensionless preference space [0, 100] without the drifting
    Z-score normalization, giving a stationary, run-comparable selection signal.

    Returns ``None`` when the run did not fully succeed (unfinished or failed
    installation) or a required metric is missing; caller scores that as worst.
    """
    from eos.envs.simple_monopile_transport.pfm import (
        PFMStatisticsTracker,
        resolve_preference_bounds,
    )

    pfm_cfg = getattr(cfg.env.reward, "pfm", None)
    if pfm_cfg is None or not pfm_cfg.enabled or not summary:
        return None

    # A "bad" run = installation not completed. Score it as worst.
    if float(summary.get("goals_remaining_mean", 0.0) or 0.0) > 0.0:
        return None
    if float(summary.get("goals_failed_mean", 0.0) or 0.0) > 0.0:
        return None

    raw_values: list[float] = []
    for obj in pfm_cfg.objectives:
        skey = _PFM_INFO_KEY_TO_SUMMARY.get(obj.info_key)
        raw = summary.get(skey) if skey else None
        if raw is None:
            logger.warning(
                f"PFM objective '{obj.name}' (info_key='{obj.info_key}') has no "
                f"eval-summary value; cannot score trial. "
                f"Available summary keys: {sorted(summary.keys())}"
            )
            return None
        raw_values.append(float(raw))

    # Ensure preference-function bounds are populated (no-op if already set).
    resolve_preference_bounds(pfm_cfg, cfg.env)
    tracker = PFMStatisticsTracker(pfm_cfg)
    pref_scores = tracker.apply_preference_functions(raw_values)
    weights = [float(obj.weight) for obj in pfm_cfg.objectives]
    return float(sum(w * p for w, p in zip(weights, pref_scores)))


def objective_pfm_balanced(trial: optuna.Trial, base_cfg: EOSConfig) -> float:
    """Optuna objective for *balanced* multi-objective PFM scenarios.

    Maximizes the weighted PFM preference composite (e.g.
    ``0.5*time_pref + 0.5*storage_pref``) averaged across seeds.

    Runs multiple seeds per trial (Option C pruning): each completed seed is
    reported to the pruner, so unstable / weak configs are killed early. The
    search space covers PPO training dynamics and PFM-specific parameters;
    model architecture and the objective weights are held fixed.
    """
    cfg = copy.deepcopy(base_cfg)

    # ── PPO Training Dynamics (only LR + entropy are swept; the rest are
    #    pinned in the experiment config) ──────────────────────────────────
    cfg.learner.init_lr = trial.suggest_float("init_lr", 1e-5, 1e-3, log=True)
    cfg.learner.ent_coef = trial.suggest_float("ent_coef", 0.005, 0.05, log=True)

    # ── PFM Parameters ───────────────────────────────────────────
    cfg.env.reward.pfm.polyak_tau = trial.suggest_float(
        "polyak_tau", 1e-4, 0.01, log=True
    )
    cfg.env.reward.pfm.phi_optimism = trial.suggest_float("phi_optimism", 0.0, 3.0)
    cfg.env.reward.pfm.completion_baseline = trial.suggest_float(
        "completion_baseline", 3.0, 10.0
    )

    cfg.env.reward.pfm.clip_bound = (
        trial.suggest_categorical("clip_bound_ratio", [0.0, 0.5, 1.0])
        * cfg.env.reward.pfm.completion_baseline
    )

    # ── Disable WandB for sweep seeds ───────────────────────────────
    cfg.track = False

    # ── Multi-seed execution with pruning ──────────────────────────
    seeds = list(base_cfg.tune.seeds)
    scores: list[float] = []

    for i, seed in enumerate(seeds):
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed.seed = seed

        now = datetime.now()
        run_name = (
            f"{cfg.exp_name}__t{trial.number}__s{seed}__{now.strftime('%Y%m%dT%H%M%S')}"
        )
        cfg_seed.run_name = run_name

        try:
            eval_summary = run_experiment(cfg_seed, run_name, trial=None)
            value = _pfm_preference_composite(cfg_seed, eval_summary)
            if value is None:
                logger.warning(
                    f"Trial {trial.number} seed {seed}: no valid PFM preference "
                    f"score (incomplete/failed run); scoring 0."
                )
                value = 0.0
            scores.append(float(value))
        except Exception as e:
            logger.warning(f"Trial {trial.number} seed {seed} failed: {e}")
            scores.append(0.0)

        # Report running mean to Optuna after each seed (higher = better).
        running_mean = sum(scores) / len(scores)
        trial.report(running_mean, step=i)

        if trial.should_prune():
            logger.info(
                f"Trial {trial.number} pruned after seed {seed} "
                f"(running mean={running_mean:.2f})"
            )
            raise optuna.exceptions.TrialPruned()

    final_mean = sum(scores) / len(scores)
    logger.info(
        f"Trial {trial.number} complete: "
        f"scores={[f'{s:.2f}' for s in scores]}, mean={final_mean:.2f}"
    )
    return final_mean


def objective_pfm_stable(trial: optuna.Trial, base_cfg: EOSConfig) -> float:
    """Optuna objective for a *stable* balanced-PFM run (Scenario A, 50/50).

    Goal: find PFM + PPO hyperparameters whose FINAL (latest) checkpoint is
    reliable — i.e. no catastrophic forgetting late in training. Because the
    balanced preference composite is scored on the *final* weights (via
    ``eval.on_final_weights=True``), a run that forgets scores poorly here, so
    a single scalar rewards both high performance and stability at once
    (Option A). No pruning: every trial runs the full budget.

    Search space (6 dims):
      Reference-frame stability (PFM):
        - polyak_tau        log [1e-5, 1e-3]
        - sigma_min shared      [6.0, 12.0]   (applied to every objective)
        - completion_baseline   [3.0, 8.0]
        - clip_bound_ratio  {0.3, 0.5, 0.7}  -> clip_bound = ratio * baseline;
                                                catastrophic_penalty = -baseline
      Policy plasticity (PPO):
        - init_lr           log [5e-5, 3e-4]
        - ent_coef          log [5e-3, 3e-2]

    Fixed: beta = 0 (SMDP off — testing whether we can do without it), gamma,
    target_kl, update_epochs, scale_warmup=True, and the 0.5/0.5 weights.
    """
    cfg = copy.deepcopy(base_cfg)

    # ── PPO plasticity ────────────────────────────────────────────────
    cfg.learner.init_lr = trial.suggest_float("init_lr", 5e-5, 3e-4, log=True)
    cfg.learner.ent_coef = trial.suggest_float("ent_coef", 5e-3, 3e-2, log=True)

    # SMDP discount off: the point of this sweep is to see if stability is
    # achievable without the beta time-discount.
    cfg.learner.beta = 0.0

    # ── PFM reference-frame stability ─────────────────────────────────
    cfg.env.reward.pfm.polyak_tau = trial.suggest_float(
        "polyak_tau", 1e-5, 1e-3, log=True
    )
    baseline = trial.suggest_float("completion_baseline", 3.0, 8.0)
    clip_ratio = trial.suggest_categorical("clip_bound_ratio", [0.3, 0.5, 0.7])
    cfg.env.reward.pfm.completion_baseline = baseline
    cfg.env.reward.pfm.clip_bound = clip_ratio * baseline
    # Keep the completion-avoidance invariant satisfied automatically:
    #   catastrophic_penalty = -baseline  <  B_eff = baseline*(1 - ratio) > 0.
    cfg.env.reward.pfm.catastrophic_penalty = -baseline

    sigma_min = trial.suggest_float("sigma_min", 6.0, 12.0)
    for obj in cfg.env.reward.pfm.objectives:
        obj.sigma_min = sigma_min

    # ── Score the FINAL weights so forgetting is penalised, not masked ─
    cfg.eval.on_final_weights = True

    # ── Disable WandB for sweep seeds ─────────────────────────────────
    cfg.track = False

    # ── Single-seed execution, no pruning ─────────────────────────────
    seeds = list(base_cfg.tune.seeds)
    scores: list[float] = []

    for seed in seeds:
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed.seed = seed

        now = datetime.now()
        run_name = (
            f"{cfg.exp_name}__t{trial.number}__s{seed}__{now.strftime('%Y%m%dT%H%M%S')}"
        )
        cfg_seed.run_name = run_name

        try:
            eval_summary = run_experiment(cfg_seed, run_name, trial=None)
            value = _pfm_preference_composite(cfg_seed, eval_summary)
            if value is None:
                logger.warning(
                    f"Trial {trial.number} seed {seed}: no valid PFM preference "
                    f"score on final weights (incomplete/failed run); scoring 0."
                )
                value = 0.0
            scores.append(float(value))
        except Exception as e:
            logger.warning(f"Trial {trial.number} seed {seed} failed: {e}")
            scores.append(0.0)

    final_mean = sum(scores) / len(scores)
    logger.info(
        f"Trial {trial.number} complete (final-weights composite): "
        f"scores={[f'{s:.2f}' for s in scores]}, mean={final_mean:.2f}"
    )
    return final_mean


# =========================================================================
# MAIN
# =========================================================================


def _pfm_name_suffix(cfg: EOSConfig) -> str:
    """Build a run-name suffix encoding PFM objective names and weights.

    Returns an empty string when PFM is not enabled. Example:
    ``__time0.5-storage_cost0.5``. Reads the resolved config, so it reflects
    any per-run weight overrides (e.g. Pareto sweeps).
    """
    reward = getattr(cfg.env, "reward", None)
    pfm = getattr(reward, "pfm", None) if reward is not None else None
    if pfm is None or not getattr(pfm, "enabled", False):
        return ""
    parts = [f"{obj.name}{float(obj.weight):g}" for obj in pfm.objectives]
    return ("__" + "-".join(parts)) if parts else ""


@hydra.main(version_base=None, config_path="../configs", config_name="eos")
def main(cfg: EOSConfig) -> None:
    # Register the environment once for the whole process
    register(
        id="SimpleMonopileTransport-v0",
        entry_point="eos.envs.simple_monopile_transport.gym_env:SimpleMonopileTransportEnv",
    )

    logger.info(f"Running training with config: {OmegaConf.to_yaml(cfg)}")

    if not cfg.tune.enable:
        # ==========================================
        # STANDARD RUN (No Optuna)
        # ==========================================
        logger.info("Running standard single experiment...")
        now = datetime.now()
        run_name = (
            f"{cfg.exp_name}{_pfm_name_suffix(cfg)}__{cfg.seed}__"
            f"{now.strftime('%Y-%m-%dT%H-%M-%S')}"
        )
        cfg.run_name = run_name

        run_experiment(cfg, run_name)

    else:
        # ==========================================
        # OPTUNA SWEEP
        # ==========================================
        journal_path = cfg.tune.storage
        if not os.path.isabs(journal_path):
            journal_path = str(Path(__file__).parent.parent / journal_path)

        logger.info(f"Starting Optuna sweep: {cfg.tune.study_name}")
        logger.info(f"Journal: {journal_path}")
        logger.info(f"Direction: {cfg.tune.direction}")
        logger.info(f"Trials per worker: {cfg.tune.n_trials}")
        logger.info(f"Prune after: {cfg.tune.prune_after_steps} steps")
        logger.info(f"Metric: {cfg.tune.metric}")

        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(journal_path),
        )

        # Pruner: prune based on per-seed reports (Option C)
        # n_warmup_steps=0 means pruning can happen after the first seed
        # n_startup_trials=5 means the first 5 trials run without pruning
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
        )

        study = optuna.create_study(
            study_name=cfg.tune.study_name,
            storage=storage,
            direction=cfg.tune.direction,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(n_startup_trials=10),
            pruner=pruner,
        )

        # Select objective function
        if cfg.tune.objective == "time_only":
            obj_fn = lambda trial: objective_time_only(trial, cfg)
        elif cfg.tune.objective == "time_only_smdp":
            obj_fn = lambda trial: objective_time_only_smdp(trial, cfg)
        elif cfg.tune.objective == "pfm_balanced":
            obj_fn = lambda trial: objective_pfm_balanced(trial, cfg)
        elif cfg.tune.objective == "pfm_stable":
            obj_fn = lambda trial: objective_pfm_stable(trial, cfg)
        else:
            obj_fn = lambda trial: objective_pfm(trial, cfg)

        study.optimize(obj_fn, n_trials=cfg.tune.n_trials)

        print("\n=========================================================")
        print(f"Best trial for {cfg.tune.study_name}:")
        best_trial = study.best_trial
        print(f"  Value: {best_trial.value}")
        print(f"  Direction: {cfg.tune.direction}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        print("=========================================================")


if __name__ == "__main__":
    main()
