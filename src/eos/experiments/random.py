"""Random-action baseline experiment.

Uses the RandomController to sample actions across multiple parallel
environments via the shared AEC micro-stepping rollout loop.

Each parallel env slot has its own independent RNG in the controller,
seeded with the episode's env seed.  This guarantees that any single
episode is fully deterministic given its seed, enabling exact replay
of the best episode found during the search phase.
"""

from __future__ import annotations

import math
import random
import signal
import time

import gymnasium as gym
import numpy as np
from loguru import logger

from eos.config import EOSConfig, RandomControllerConfig
from eos.controllers.random import RandomController
from eos.core.experiment import Experiment
from eos.envs.factory import make_env
from eos.runners.micro_step import (
    MicroStepCollector,
    _split_vectorized_info,
    run_evaluation_episode,
)
from eos.utils.eval_logger import (
    log_eval_to_wandb,
    print_eval_summary,
)
from eos.utils.profiling import StepTimer


class RandomExperiment(Experiment):
    def __init__(self, cfg: EOSConfig) -> None:
        self.cfg = cfg
        self.cfg.env.beta = self.cfg.learner.beta

        self.stop_requested = False
        signal.signal(signal.SIGINT, self._handle_interrupt)
        self.setup()

    def _handle_interrupt(self, signum, frame):
        if self.stop_requested:
            logger.warning("Forced exit requested! Bypassing graceful shutdown...")
            raise KeyboardInterrupt
        logger.info(
            "Interrupt received! Finishing current batch... (Press Ctrl+C again to force quit)"
        )
        self.stop_requested = True

    def setup(self) -> None:
        cfg = self.cfg
        self.run_name = cfg.run_name
        self.num_envs = max(1, cfg.num_envs)

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        if cfg.env.render.mode == "human":
            raise ValueError("RandomExperiment only supports non-human rendering.")

        env_fns = [
            make_env(cfg, i, cfg.capture_video, self.run_name, cfg.env.render.mode)
            for i in range(self.num_envs)
        ]
        self.envs = gym.vector.SyncVectorEnv(env_fns)

        # --- PFM multi-objective reward normalization (main-process wrapper) ---
        pfm_cfg = getattr(cfg.env.reward, "pfm", None)
        if pfm_cfg is not None and pfm_cfg.enabled:
            from eos.envs.simple_monopile_transport.pfm import PFMVectorWrapper

            self.envs = PFMVectorWrapper(self.envs, pfm_cfg)

        # Safely extract random config or fallback to defaults
        random_cfg = getattr(cfg.controller, "random_schema", None)
        if random_cfg is None:
            random_cfg = RandomControllerConfig()

        self.controller = RandomController(
            model=None,
            cfg=random_cfg,
            action_space=self.envs.single_action_space,
            num_envs=self.num_envs,
        )

        # Create the shared micro-stepping collector
        self._collector = MicroStepCollector(
            envs=self.envs,
            controller=self.controller,
            use_action_masks=bool(cfg.env.use_action_masks),
            deterministic=False,  # Random is always stochastic
        )

        self.global_step = 0
        self._best_return = None
        self._best_length = None
        self._best_seed = None
        self._episode_results = []
        self._returns = []
        self._lengths = []
        self.eval_results = None

    def _run_episode_batch(self, batch_start: int, batch_size: int):
        cfg = self.cfg
        timer = StepTimer()

        if getattr(cfg.random, "deterministic_seed", True):
            base_seed = cfg.seed + batch_start
            reset_kwargs = {"seed": base_seed}
        else:
            base_seed = cfg.seed + batch_start
            reset_kwargs = {"seed": cfg.seed} if batch_start == 0 else {}

        ep_seeds = [base_seed + i for i in range(self.num_envs)]

        # Seed each env-slot's RNG in the controller with the episode seed.
        # This ensures each episode is fully deterministic given its seed,
        # independent of what other parallel envs do.
        self.controller.seed_envs(ep_seeds)

        # Reset the collector's fine-grained timer for this batch
        self._collector.timer.reset()

        with timer.phase("env_reset"):
            obs, reset_infos = self.envs.reset(**reset_kwargs)
            # Seed the collector's info cache so the first macro-step
            # can extract masks/availability without extra IPC calls.
            self._collector._cached_infos = reset_infos

        done = np.zeros(self.num_envs, dtype=bool)
        ep_returns = np.zeros(self.num_envs, dtype=np.float64)
        step_counts = np.zeros(self.num_envs, dtype=np.int64)
        ep_final_infos = [{}] * self.num_envs

        while not done[:batch_size].all():
            with timer.phase("macro_step"):
                result = self._collector.macro_step(obs, deterministic=False)
                obs = result.next_obs

            with timer.phase("bookkeeping"):
                per_env_infos = _split_vectorized_info(result.infos, self.num_envs)

                for i in range(batch_size):
                    if done[i]:
                        continue

                    # Accumulate reward from DES advance
                    ep_returns[i] += result.rewards[i]
                    step_counts[i] += 1

                    # Check for episode completion
                    if result.terminated[i] or result.truncated[i]:
                        done[i] = True
                        info = per_env_infos[i]
                        ep_final_infos[i] = dict(info) if info else {}

                        # Try to get episode stats from RecordEpisodeStatistics
                        if "episode" in info:
                            ep_info = info["episode"]
                            r = ep_info.get("r", None)
                            ep_len = ep_info.get("l", None)
                            if r is not None:
                                ep_returns[i] = float(
                                    r[0]
                                    if isinstance(r, (list, tuple, np.ndarray))
                                    else r
                                )
                            if ep_len is not None:
                                step_counts[i] = int(
                                    ep_len[0]
                                    if isinstance(ep_len, (list, tuple, np.ndarray))
                                    else ep_len
                                )

                # Count active envs for global step tracking
                active = ~done[:batch_size]
                self.global_step += int(active.sum())

        # Log the collector's fine-grained macro-step breakdown
        self._collector.timer.log_summary(
            f"Macro-step breakdown (batch {batch_start}..{batch_start + batch_size})"
        )

        return (
            ep_returns[:batch_size],
            step_counts[:batch_size],
            ep_final_infos[:batch_size],
            ep_seeds[:batch_size],
            timer,
        )

    def _run_experiment_loop(self) -> None:
        cfg = self.cfg
        start_time = time.time()
        num_episodes = cfg.random.num_episodes
        num_batches = math.ceil(num_episodes / self.num_envs)
        total_completed = 0

        if cfg.track:
            import wandb

        try:
            for batch_idx in range(num_batches):
                if self.stop_requested:
                    break

                batch_start = batch_idx * self.num_envs
                batch_size = min(self.num_envs, num_episodes - batch_start)

                (
                    ep_returns,
                    ep_lengths,
                    ep_final_infos,
                    ep_seeds,
                    batch_timer,
                ) = self._run_episode_batch(batch_start, batch_size)

                for i in range(batch_size):
                    ep_return = float(ep_returns[i])
                    ep_length = int(ep_lengths[i])
                    ep_seed = ep_seeds[i]
                    episode_idx = batch_start + i
                    total_completed += 1

                    self._returns.append(ep_return)
                    self._lengths.append(ep_length)

                    ep_result = {
                        "episode": episode_idx + 1,
                        "return": ep_return,
                        "length": ep_length,
                        "seed": ep_seed,
                    }

                    fi = ep_final_infos[i]
                    for key in (
                        "goals_completed",
                        "goals_failed",
                        "goals_remaining",
                        "elapsed_time_hours",
                    ):
                        if key in fi:
                            val = fi[key]
                            ep_result[key] = float(
                                val[0]
                                if isinstance(val, (list, tuple, np.ndarray))
                                else val
                            )

                    self._episode_results.append(ep_result)

                    # --- LIVE WANDB LOGGING ---
                    if cfg.track:
                        sps = int(
                            self.global_step / max(1e-9, time.time() - start_time)
                        )
                        ep_wandb = {
                            "charts/episodic_return": ep_return,
                            "charts/episodic_length": float(ep_length),
                            "charts/best_return": float(self._best_return or ep_return),
                            "charts/SPS": sps,
                            "random/episode": episode_idx + 1,
                            "random/seed": ep_seed,
                            "global_step": self.global_step,
                        }
                        for key in (
                            "goals_completed",
                            "goals_failed",
                            "goals_remaining",
                            "elapsed_time_hours",
                        ):
                            if key in ep_result:
                                ep_wandb[f"random/{key}"] = ep_result[key]
                        wandb.log(ep_wandb)

                    # --- UPDATE BEST EPISODE ---
                    if self._best_return is None or ep_return > self._best_return:
                        self._best_return = ep_return
                        self._best_length = ep_length
                        self._best_seed = ep_seed

                elapsed = time.time() - start_time
                sps = int(self.global_step / max(1e-9, elapsed))
                logger.info(
                    f"Batch {batch_idx + 1}/{num_batches} done | {total_completed}/{num_episodes} eps | {sps} SPS | best={self._best_return or 0:.2f}"
                )

        except KeyboardInterrupt:
            logger.warning("Experiment forcefully interrupted by user.")
        finally:
            self.envs.close()

            # Run the single deterministic eval replay of the best episode
            self._replay_best_episode()

            if cfg.track:
                import wandb

                wandb.finish()

            logger.success(f"Experiment {self.run_name} completed.")

    def _replay_best_episode(self) -> None:
        """Replay the best episode found during search with deterministic seeding.

        Because each env-slot's RNG is seeded with the episode's env seed,
        replaying with the same seed produces the EXACT same action sequence
        and trajectory as during the search phase.
        """
        if self._best_seed is None:
            return

        cfg = self.cfg
        logger.info(
            f"Replaying best episode deterministically "
            f"(search return={self._best_return:.2f}, seed={self._best_seed})"
        )

        # Create a single-env setup for the replay
        eval_envs = gym.vector.SyncVectorEnv(
            [make_env(cfg, 0, False, f"{self.run_name}-eval", None)]
        )

        # --- PFM wrapper for evaluation ---
        pfm_cfg = getattr(cfg.env.reward, "pfm", None)
        if pfm_cfg is not None and pfm_cfg.enabled:
            from eos.envs.simple_monopile_transport.pfm import PFMVectorWrapper

            # Share the training tracker so eval uses the same μ/σ
            training_tracker = getattr(self.envs, "_tracker", None)
            eval_envs = PFMVectorWrapper(eval_envs, pfm_cfg, tracker=training_tracker)

        # Create a fresh single-env controller seeded with the best episode's seed.
        # This guarantees identical actions to the search phase.
        eval_controller = RandomController(
            model=None,
            cfg=getattr(cfg.controller, "random_schema", None)
            or RandomControllerConfig(),
            action_space=eval_envs.single_action_space,
            num_envs=1,
        )
        eval_controller.seed_envs([self._best_seed])

        try:
            eval_results = run_evaluation_episode(
                envs=eval_envs,
                controller=eval_controller,
                seed=self._best_seed,
                num_episodes=1,
                deterministic=False,  # Random controller ignores this; RNG is pre-seeded
                use_action_masks=bool(cfg.env.use_action_masks),
                capture_inventory=True,
                trace_actions=True,
            )

            # The eval episode's return should match the search-phase return
            # for this seed (since both env and controller are identically seeded).
            eval_return = eval_results["episodes"][0]["return"]
            search_return = self._best_return
            delta = abs(eval_return - search_return)
            if delta > 0.01:
                logger.warning(
                    f"Replay return ({eval_return:.4f}) differs from search return "
                    f"({search_return:.4f}) by {delta:.4f}. "
                    f"This indicates a seeding inconsistency."
                )
            else:
                logger.success(
                    f"Replay return ({eval_return:.4f}) matches search return "
                    f"({search_return:.4f}). Deterministic replay verified."
                )

            # Add search-phase aggregate statistics to the summary
            eval_results["summary"]["search_num_episodes"] = len(self._returns)
            eval_results["summary"]["search_return_mean"] = (
                float(np.mean(self._returns)) if self._returns else float("nan")
            )
            eval_results["summary"]["search_return_std"] = (
                float(np.std(self._returns)) if self._returns else float("nan")
            )
            eval_results["summary"]["search_return_min"] = (
                float(np.min(self._returns)) if self._returns else float("nan")
            )
            eval_results["summary"]["search_return_max"] = (
                float(np.max(self._returns)) if self._returns else float("nan")
            )

            # Add domain-specific means from search
            for key in ("goals_completed", "goals_failed", "goals_remaining"):
                vals = [e[key] for e in self._episode_results if key in e]
                if vals:
                    eval_results["summary"][f"search_{key}_mean"] = float(np.mean(vals))

            sim_times = [
                e["elapsed_time_hours"]
                for e in self._episode_results
                if "elapsed_time_hours" in e
            ]
            if sim_times:
                eval_results["summary"]["search_sim_elapsed_hours_mean"] = float(
                    np.mean(sim_times)
                )

            # Store all search episode results for reference
            eval_results["search_episodes"] = self._episode_results

            # Store for external access (e.g. run_experiment.py)
            self.eval_results = eval_results

            print_eval_summary(eval_results)

            if cfg.track:
                import wandb

                if self._episode_results:
                    cols = list(self._episode_results[0].keys())
                    data = [[ep.get(c) for c in cols] for ep in self._episode_results]
                    wandb.log(
                        {"eval/search_episodes": wandb.Table(data=data, columns=cols)}
                    )

                log_eval_to_wandb(
                    eval_results, wandb.run.dir, global_step=self.global_step
                )

        except Exception as e:
            logger.error(f"Evaluation replay failed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            eval_envs.close()

    def _build_summary(self) -> dict:
        summary = {
            "num_episodes": len(self._returns),
            "return_mean": float(np.mean(self._returns))
            if self._returns
            else float("nan"),
            "return_std": float(np.std(self._returns))
            if self._returns
            else float("nan"),
            "return_min": float(np.min(self._returns))
            if self._returns
            else float("nan"),
            "return_max": float(np.max(self._returns))
            if self._returns
            else float("nan"),
            "length_mean": float(np.mean(self._lengths))
            if self._lengths
            else float("nan"),
            "length_std": float(np.std(self._lengths))
            if self._lengths
            else float("nan"),
        }
        for key in ("goals_completed", "goals_failed", "goals_remaining"):
            vals = [e[key] for e in self._episode_results if key in e]
            if vals:
                summary[f"{key}_mean"] = float(np.mean(vals))

        sim_times = [
            e["elapsed_time_hours"]
            for e in self._episode_results
            if "elapsed_time_hours" in e
        ]
        if sim_times:
            summary["sim_elapsed_hours_mean"] = float(np.mean(sim_times))

        return summary
