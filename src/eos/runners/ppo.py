"""PPO rollout runner using AEC micro-stepping.

Collects environment transitions by stepping the vectorised environment
using the two-phase Intent-Based Micro-Stepping protocol:

1. **Ordering** — determine the sequence in which idle vessels act.
2. **Per-vessel action** — for each idle vessel, select an action and
   register the intent via ``env.micro_step()`` (no time advance, reward=0).
3. **DES advance** — call ``env.step()`` to advance the DES clock
   and collect the real reward.

Each iteration of this loop produces one "macro-step" transition that
is stored in the :class:`~eos.buffers.ppo_rollout.PPORolloutBuffer`.
The critic value used for GAE is taken from the *first* micro-step
of each macro-step (the moment before any vessel has committed).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

import wandb
from eos.buffers.ppo_rollout import PPORolloutBuffer
from eos.controllers.ppo import PPOController
from eos.core.runner import Runner
from eos.runners.micro_step import (
    MicroStepCollector,
    _split_vectorized_info,
)
from eos.utils.metrics import aggregate_vector_info, safe_mean
from eos.utils.profiling import StepTimer


def _create_goals_table(goals_info: list[dict]) -> wandb.Table | None:
    """Convert a list of goal-info dicts into a WandB Table for logging."""
    if not goals_info:
        return None

    columns = list(goals_info[0].keys())
    data = [[g.get(col) for col in columns] for g in goals_info]
    return wandb.Table(data=data, columns=columns)


class PPORunner(Runner):
    """Collects rollout data via the AEC micro-stepping protocol.

    Each call to :meth:`run` fills a :class:`PPORolloutBuffer` with
    ``steps`` macro-transitions and returns aggregated metrics.

    A single macro-step comprises:

    * Phase 1 — vessel ordering (one call to ``controller.get_ordering``).
    * Phase 2 — per-vessel action selection (one ``controller.get_action_and_value``
      per idle vessel, each followed by ``env.micro_step()`` to register the intent).
    * DES advance — ``env.step()`` to collect reward & next obs.

    The first micro-step's critic value is recorded as the value estimate
    for the whole macro-step (required for correct PPO advantage computation).

    Parameters
    ----------
    envs :
        Vectorised gymnasium environment (``SyncVectorEnv``).
    controller : PPOController
        The PPO controller implementing the two-phase AEC interface.
    seed : int | None
        Seed for the initial environment reset.
    use_action_masks : bool
        Whether to query ``env.action_masks()`` each step.
    """

    def __init__(
        self,
        envs,
        controller: PPOController,
        seed: int | None = None,
        use_action_masks: bool = False,
    ) -> None:
        super().__init__(envs, controller)
        self._seed = seed
        self._use_action_masks = bool(use_action_masks)

        # Initial reset — capture info so the collector can extract
        # action_masks and vessel_availability without extra IPC calls.
        self.obs, reset_infos = self.envs.reset(seed=seed)
        self._trace_env_index = 0

        # Create the shared micro-stepping collector
        self._collector = MicroStepCollector(
            envs=self.envs,
            controller=self.controller,
            use_action_masks=self._use_action_masks,
            deterministic=False,  # PPO training is always stochastic
        )
        # Seed the collector's info cache from the initial reset so the
        # very first macro-step can use the zero-IPC path.
        self._collector._cached_infos = reset_infos

        # Detect render mode
        self._render_mode = getattr(self.envs, "render_mode", None)
        if self._render_mode is None:
            try:
                if hasattr(self.envs, "get_wrapper_attr"):
                    self._render_mode = self.envs.get_wrapper_attr("render_mode")
                else:
                    self._render_mode = self.envs.get_attr("render_mode")[0]
            except Exception:
                self._render_mode = None

    def run(self, buffer: PPORolloutBuffer, steps: int) -> dict:
        """Collect ``steps`` macro-transitions and store them in ``buffer``.

        Returns
        -------
        dict
            ``next_obs``, ``next_done``, and aggregated ``metrics``.
        """
        episodic_returns: list[float] = []
        episodic_lengths: list[float] = []
        episodic_times: list[float] = []
        episodic_successes: list[float] = []
        episodic_elapsed_hours: list[float] = []
        episodic_storage_cost: list[float] = []
        episodic_total_cost: list[float] = []
        episodic_travel_cost: list[float] = []
        rollout_rewards = np.empty(steps, dtype=np.float64)
        mask_densities: list[float] = []
        scalar_metrics_buffer: dict[str, list] = defaultdict(list)
        latest_goals_snapshot = None
        episodes_completed: int = 0

        done = np.zeros(self.envs.num_envs, dtype=bool)

        timer = StepTimer()

        # Reset the collector's fine-grained timer so it only
        # accumulates for this rollout window.
        self._collector.timer.reset()

        for t in range(steps):
            # ----------------------------------------------------------
            # 1) Execute one macro-step (ordering + micro-steps + DES advance)
            # ----------------------------------------------------------
            with timer.phase("macro_step"):
                result = self._collector.macro_step(self.obs, deterministic=False)

            # ----------------------------------------------------------
            # 2) Process result and store in buffer
            # ----------------------------------------------------------
            with timer.phase("bookkeeping"):
                # All fields are already batch arrays — no re-stacking needed
                rewards = result.rewards
                terminated = result.terminated
                truncated = result.truncated
                done = np.logical_or(terminated, truncated)

                rollout_rewards[t] = float(np.mean(rewards))

                # Attempt to render if needed
                if self._render_mode and self._render_mode != "human":
                    try:
                        self.envs.render()
                    except Exception:
                        pass

                # Use the per-vessel masks captured at decision time
                stored_mask = result.per_vessel_masks
                if stored_mask is not None:
                    mask_densities.append(float(np.mean(stored_mask)))

                # Process env-level info from the DES advance
                per_env_infos = _split_vectorized_info(result.infos, self.envs.num_envs)
                for env_i in range(self.envs.num_envs):
                    info = per_env_infos[env_i]
                    if info:
                        env_metrics = aggregate_vector_info(
                            {
                                k: v
                                for k, v in info.items()
                                if isinstance(v, (int, float, np.integer, np.floating))
                            },
                            prefix="env/",
                        )
                        for k, v in env_metrics.items():
                            scalar_metrics_buffer[k].append(v)

                        if "reward_components" in info and isinstance(
                            info["reward_components"], dict
                        ):
                            for k, v in info["reward_components"].items():
                                if isinstance(v, (int, float, np.integer, np.floating)):
                                    scalar_metrics_buffer[f"reward/{k}"].append(
                                        float(v)
                                    )

                        # PFM multi-objective diagnostics
                        if "pfm" in info and isinstance(info["pfm"], dict):
                            for k, v in info["pfm"].items():
                                if isinstance(v, (int, float, np.integer, np.floating)):
                                    scalar_metrics_buffer[f"pfm/{k}"].append(float(v))

                        if "goals_info" in info and env_i == self._trace_env_index:
                            latest_goals_snapshot = info["goals_info"]

                # Check for episode completions
                for env_i in range(self.envs.num_envs):
                    if terminated[env_i] or truncated[env_i]:
                        info = per_env_infos[env_i]
                        # Try to extract episode stats from RecordEpisodeStatistics
                        if "episode" in info:
                            ep_info = info["episode"]
                            r = ep_info.get("r", 0.0)
                            ep_len = ep_info.get("l", 0)
                            t_ep = ep_info.get("t", 0.0)
                            episodic_returns.append(
                                float(
                                    r[0]
                                    if isinstance(r, (list, tuple, np.ndarray))
                                    else r
                                )
                            )
                            episodic_lengths.append(
                                float(
                                    ep_len[0]
                                    if isinstance(ep_len, (list, tuple, np.ndarray))
                                    else ep_len
                                )
                            )
                            episodic_times.append(
                                float(
                                    t_ep[0]
                                    if isinstance(t_ep, (list, tuple, np.ndarray))
                                    else t_ep
                                )
                            )
                            episodes_completed += 1

                        # Track per-episode success for success rate
                        is_success = info.get("is_success", False)
                        if isinstance(is_success, (np.ndarray, list, tuple)):
                            is_success = bool(is_success[0])
                        episodic_successes.append(float(bool(is_success)))

                        # Extract terminal domain metrics for episode-level logging
                        if "elapsed_time_hours" in info:
                            val = info["elapsed_time_hours"]
                            episodic_elapsed_hours.append(
                                float(
                                    val[0]
                                    if isinstance(val, (list, tuple, np.ndarray))
                                    else val
                                )
                            )
                        if "cost/storage" in info:
                            val = info["cost/storage"]
                            episodic_storage_cost.append(
                                float(
                                    val[0]
                                    if isinstance(val, (list, tuple, np.ndarray))
                                    else val
                                )
                            )
                        if "cost/total" in info:
                            val = info["cost/total"]
                            episodic_total_cost.append(
                                float(
                                    val[0]
                                    if isinstance(val, (list, tuple, np.ndarray))
                                    else val
                                )
                            )
                        if "cost/travel" in info:
                            val = info["cost/travel"]
                            episodic_travel_cost.append(
                                float(
                                    val[0]
                                    if isinstance(val, (list, tuple, np.ndarray))
                                    else val
                                )
                            )

                # ----------------------------------------------------------
                # 3) Store transition in the buffer
                # ----------------------------------------------------------
                buffer.add(
                    obs=self.obs,
                    action=result.joint_actions,
                    reward=rewards,
                    done=done,
                    value=result.first_values,
                    logprob=result.agg_logprobs,
                    delta_time=result.delta_times_hours,
                    action_mask=stored_mask,
                    ordering=result.orderings,
                    vessel_availability=result.vessel_availability,
                    per_vessel_obs=result.per_vessel_obs,
                    per_factor_logprobs=result.per_factor_logprobs,
                )

                self.obs = result.next_obs

        # ----------------------------------------------------------
        # Aggregate metrics over the rollout
        # ----------------------------------------------------------
        final_env_metrics = {
            k: float(np.nanmean(v))
            for k, v in scalar_metrics_buffer.items()
            if v and not all(np.isnan(x) for x in v)
        }

        rich_metrics = {}
        if latest_goals_snapshot:
            rich_metrics["goals_table"] = _create_goals_table(latest_goals_snapshot)

        # Log the collector's fine-grained macro-step breakdown
        self._collector.timer.log_summary(
            f"Macro-step breakdown ({steps} steps x {self.envs.num_envs} envs)"
        )

        return {
            "next_obs": self.obs,
            "next_done": done,
            "metrics": {
                "episodic_returns": safe_mean(episodic_returns),
                "episodic_lengths": safe_mean(episodic_lengths),
                "episodic_time": safe_mean(episodic_times),
                "success_rate": safe_mean(episodic_successes),
                "episodic_elapsed_hours": safe_mean(episodic_elapsed_hours),
                "episodic_storage_cost": safe_mean(episodic_storage_cost),
                "episodic_total_cost": safe_mean(episodic_total_cost),
                "episodic_travel_cost": safe_mean(episodic_travel_cost),
                "rollout_metrics": {
                    "reward_mean": float(rollout_rewards.mean()),
                    "action_mask_density_mean": safe_mean(mask_densities),
                    "episodes_completed": episodes_completed,
                },
                "env_metrics": final_env_metrics,
                "rich_metrics": rich_metrics,
                "step_timing": timer.summary(prefix="perf/step/"),
                "macro_step_timing": self._collector.timer.summary(
                    prefix="perf/macro/"
                ),
            },
        }
