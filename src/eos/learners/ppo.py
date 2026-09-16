"""PPO (Proximal Policy Optimisation) learner.

Implements the clipped surrogate objective with optional value-function
clipping, entropy bonus, advantage normalisation, and early stopping via
a target KL threshold.

The learner receives a :class:`~eos.buffers.ppo_rollout.PPOBatch` (a
``NamedTuple``) from the rollout buffer and performs multiple epochs of
minibatch gradient updates on the model.

For MultiDiscrete (multi-vessel) action spaces the learner uses the
model's :meth:`replay_macro_step` method which efficiently replays
the full two-phase AEC micro-stepping decision (ordering + per-vessel
actions) in a single forward pass through the transformer backbone.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from eos.buffers.ppo_rollout import PPOBatch
from eos.config import PPOLearnerConfig
from eos.core.learner import Learner
from eos.core.model import Model


class PPOLearner(Learner):
    """Updates the actor-critic model using the PPO clipped objective.

    Parameters
    ----------
    model : Model
        The actor-critic model whose parameters will be optimised.
    cfg : PPOLearnerConfig
        Hyper-parameters for the PPO update (learning rate, clip
        coefficient, number of epochs, etc.).
    """

    def __init__(self, model: Model, cfg: PPOLearnerConfig):
        super().__init__(model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.init_lr, eps=1e-5)
        self.cfg = cfg

    def train(self, batch: PPOBatch) -> dict:
        """Run PPO update epochs on the given batch.

        Parameters
        ----------
        batch : PPOBatch
            Named tuple produced by
            :meth:`~eos.buffers.ppo_rollout.PPORolloutBuffer.sample`.

        Returns
        -------
        dict
            Training metrics including losses, KL divergence, clip
            fraction, gradient norm, explained variance, and current
            learning rate.
        """
        b_obs = batch.obs
        b_logprobs = batch.logprobs
        b_actions = batch.actions
        b_advantages = batch.advantages
        b_returns = batch.returns
        b_values = batch.values
        b_delta_times = batch.delta_times
        b_action_masks = batch.action_masks
        b_orderings = batch.orderings
        b_vessel_availability = batch.vessel_availability
        b_per_vessel_obs = batch.per_vessel_obs
        b_per_factor_logprobs = batch.per_factor_logprobs

        # ── NaN diagnostic: catch corrupt batch data early ────────────
        for name, tensor in [
            ("returns", b_returns),
            ("advantages", b_advantages),
            ("values", b_values),
            ("logprobs", b_logprobs),
        ]:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                valid = tensor[~(torch.isnan(tensor) | torch.isinf(tensor))]
                rng = (
                    f"[{valid.min().item():.4f}, {valid.max().item():.4f}]"
                    if valid.numel() > 0
                    else "[ALL NaN/Inf]"
                )
                raise RuntimeError(
                    f"NaN/Inf in batch.{name}: "
                    f"NaN={torch.isnan(tensor).sum().item()}, "
                    f"Inf={torch.isinf(tensor).sum().item()}, "
                    f"range={rng}"
                )

        # Detect whether we are in the two-phase AEC regime.
        # When orderings are present, the model exposes replay_macro_step
        # and logprobs are scalar aggregates (ordering + per-vessel).
        use_macro_replay = b_orderings is not None and hasattr(
            self.model, "replay_macro_step"
        )

        b_inds = np.arange(self.cfg.batch_size)

        # Pre-compute max possible updates for clipfracs storage
        updates_per_epoch = self.cfg.batch_size // self.cfg.minibatch_size
        max_updates = self.cfg.update_epochs * updates_per_epoch
        clipfracs = np.empty(max_updates, dtype=np.float32)

        # Accumulators for per-update metrics
        avg_pg_loss = 0.0
        avg_v_loss = 0.0
        avg_entropy_loss = 0.0
        avg_old_approx_kl = 0.0
        avg_approx_kl = 0.0
        avg_grad_norm = 0.0
        num_updates = 0

        for epoch in range(self.cfg.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, self.cfg.batch_size, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_inds = b_inds[start:end]

                mb_masks = (
                    b_action_masks[mb_inds] if b_action_masks is not None else None
                )

                # ---------------------------------------------------------
                # Replay the model to get fresh log-probs, entropy, and
                # value estimates along the stored decision path.
                # ---------------------------------------------------------
                if use_macro_replay:
                    # Two-phase AEC replay: ordering + per-vessel actions
                    mb_orderings = b_orderings[mb_inds]
                    mb_vessel_avail = (
                        b_vessel_availability[mb_inds]
                        if b_vessel_availability is not None
                        else None
                    )
                    mb_per_vessel_obs = (
                        b_per_vessel_obs[mb_inds]
                        if b_per_vessel_obs is not None
                        else None
                    )
                    newlogprob, entropy, newvalue = self.model.replay_macro_step(
                        obs=b_obs[mb_inds],
                        actions=b_actions.long()[mb_inds],
                        ordering=mb_orderings,
                        action_mask=mb_masks,
                        vessel_availability=mb_vessel_avail,
                        per_vessel_obs=mb_per_vessel_obs,
                    )
                    # newlogprob: (MB, 2K) — per-factor log-probs
                    # entropy:    (MB,) — aggregate scalar
                    # newvalue:   (MB, 1)
                else:
                    # Single-action (Discrete) path — simple replay
                    result = self.model.get_action_and_value(
                        b_obs[mb_inds],
                        b_actions.long()[mb_inds],
                        action_mask=mb_masks,
                    )
                    # Discrete path returns (action, logprob, entropy, value)
                    _, newlogprob, entropy, newvalue = result

                # ── Probability ratio & KL ───────────────────────
                # Per-factor clipping for autoregressive action spaces:
                # clip each sub-decision's ratio independently against the
                # shared macro-advantage, then sum. This prevents trust-
                # region shrinkage where the joint ratio breaches the clip
                # bound even though individual sub-policies barely moved.
                if use_macro_replay and b_per_factor_logprobs is not None:
                    # newlogprob here is per_factor_logprobs: (MB, 2K)
                    pf_logratio = newlogprob - b_per_factor_logprobs[mb_inds]
                    # Hard clamp to prevent float32 overflow in exp().
                    # ±20 allows ratios up to exp(20) ≈ 5e8 — far beyond
                    # the clip bound, so this never affects valid gradients.
                    pf_logratio = pf_logratio.clamp(-20.0, 20.0)

                    # Aggregate logratio for KL diagnostics (and the joint ratio).
                    logratio = pf_logratio.sum(dim=1)  # (MB,)
                    ratio = logratio.clamp(-20.0, 20.0).exp()

                    if self.cfg.per_factor_clip:
                        pf_ratio = pf_logratio.exp()  # (MB, 2K)
                    else:
                        # Joint-ratio ablation: collapse the 2K factors into a
                        # single joint ratio before clipping (standard PPO
                        # surrogate). pf_ratio=None routes the policy loss
                        # through the joint-clipping branch below.
                        pf_ratio = None
                else:
                    logratio = newlogprob - b_logprobs[mb_inds]
                    logratio = logratio.clamp(-20.0, 20.0)
                    ratio = logratio.exp()
                    pf_ratio = None

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs[num_updates] = (
                        ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()
                    )

                # Early stopping on KL divergence.
                # NOTE: target_kl should be scaled with the number of
                # sub-decisions K (e.g. 0.05 for K=6 instead of the
                # single-action default of 0.01) because KL divergence
                # is additive across conditionally independent factors.
                if self.cfg.target_kl is not None and approx_kl > self.cfg.target_kl:
                    break

                # ── Advantage normalisation ──────────────────────
                mb_advantages = b_advantages[mb_inds]

                if self.cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # ── Policy (actor) loss ────────────────────────
                if pf_ratio is not None:
                    # Per-factor clipping: clip each sub-decision's ratio
                    # independently, then sum.  The shared advantage is
                    # broadcast across all K factors.
                    adv = mb_advantages.unsqueeze(1)  # (MB, 1)
                    pf_loss1 = -adv * pf_ratio
                    pf_loss2 = -adv * torch.clamp(
                        pf_ratio,
                        1 - self.cfg.clip_coef,
                        1 + self.cfg.clip_coef,
                    )
                    pg_loss = torch.max(pf_loss1, pf_loss2).sum(dim=1).mean()
                else:
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # ── Value (critic) loss ──────────────────────────────
                newvalue = newvalue.view(-1)
                if self.cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.cfg.clip_coef,
                        self.cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # ── Entropy loss ─────────────────────────────────────
                # For the macro-replay path, entropy is already an
                # aggregate scalar per sample that only counts idle
                # vessels.  For the Discrete path, use the mask-weighted
                # mean as before.
                if use_macro_replay:
                    entropy_mean = entropy.mean()
                else:
                    entropy_mean = self._compute_entropy_mean(entropy, mb_masks)

                # ── Combined loss & optimisation step ────────────────
                loss = (
                    pg_loss
                    - self.cfg.ent_coef * entropy_mean
                    + v_loss * self.cfg.vf_coef
                )

                # ── NaN diagnostic: catch poison before it enters backward ─
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(
                        f"NaN/Inf in combined loss at epoch {epoch}, "
                        f"update {num_updates}: "
                        f"pg_loss={pg_loss.item():.6f}, "
                        f"v_loss={v_loss.item():.6f}, "
                        f"entropy={entropy_mean.item():.6f}, "
                        f"loss={loss.item()}, "
                        f"approx_kl={approx_kl.item():.6f}"
                    )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )

                # ── NaN diagnostic: catch NaN gradients before optimizer step ─
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    # Find which parameter group has NaN grads
                    nan_params = [
                        name
                        for name, p in self.model.named_parameters()
                        if p.grad is not None and torch.isnan(p.grad).any()
                    ]
                    raise RuntimeError(
                        f"NaN/Inf gradient norm at epoch {epoch}, "
                        f"update {num_updates}: "
                        f"grad_norm={grad_norm}, "
                        f"loss={loss.item():.6f}, "
                        f"NaN-grad params ({len(nan_params)}): "
                        f"{nan_params[:10]}"
                    )

                self.optimizer.step()

                # Accumulate metrics
                avg_pg_loss += pg_loss.item()
                avg_v_loss += v_loss.item()
                avg_entropy_loss += entropy_mean.item()
                avg_old_approx_kl += old_approx_kl.item()
                avg_approx_kl += approx_kl.item()
                avg_grad_norm += float(grad_norm)
                num_updates += 1

        # ── Explained variance (critic quality diagnostic) ───────────
        # Computed on-device to avoid a GPU→CPU transfer of the full
        # values/returns tensors.
        with torch.no_grad():
            var_y = torch.var(b_returns)
            explained_var = (
                float("nan")
                if var_y.item() == 0
                else float(1.0 - torch.var(b_returns - b_values) / var_y)
            )

            gammas = torch.exp(-self.cfg.beta * b_delta_times)

            # 1. The mathematical average discount applied
            avg_gamma = gammas.mean().item()

            # 2. The harshest discount applied in this batch (from the longest step)
            min_gamma = gammas.min().item()

            # 3. Human-readable average step duration (in hours)
            avg_step_hours = b_delta_times.mean().item()

        return {
            # Primary objectives
            "policy_loss": avg_pg_loss / num_updates,
            "value_loss": avg_v_loss / num_updates,
            "entropy_loss": avg_entropy_loss / num_updates,
            # Stability diagnostics
            "approx_kl": avg_approx_kl / num_updates,
            "clip_fraction": float(clipfracs[:num_updates].mean()),
            "grad_norm": avg_grad_norm / num_updates,
            "explained_variance": explained_var,
            # Time & Discount diagnostics
            "avg_gamma": avg_gamma,
            "min_gamma": min_gamma,
            "avg_step_hours": avg_step_hours,
            # Hyperparameters
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_entropy_mean(
        entropy: torch.Tensor,
        action_masks: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute the (optionally mask-weighted) mean entropy.

        For masked MultiDiscrete action spaces, vessels whose only valid
        action is the forced NOOP (exactly 1 valid slot) contribute zero
        entropy and should not dilute the denominator.

        Parameters
        ----------
        entropy : Tensor
            Per-sample entropy.  Shape ``(B,)`` for Discrete or
            ``(B, n_vessels)`` for MultiDiscrete.
        action_masks : Tensor | None
            Action masks with shape ``(B, action_dim)`` for Discrete or
            ``(B, n_vessels, opts_per_vessel)`` for MultiDiscrete.
            ``None`` when masking is disabled.

        Returns
        -------
        Tensor
            Scalar mean entropy.
        """
        if action_masks is None:
            return entropy.mean()

        mask_t = torch.as_tensor(action_masks, device=entropy.device)
        if mask_t.dtype != torch.bool:
            mask_t = mask_t.bool()

        # Count valid actions per head: (B,) or (B, n_vessels)
        valid_counts = mask_t.sum(dim=-1)

        # A head is "active" (has a real choice) when more than 1 action
        # is available — when valid_count == 1 it's a forced NOOP.
        active = (valid_counts > 1).to(dtype=entropy.dtype)
        env_joint_entropies = (entropy * active).sum(dim=-1)

        return env_joint_entropies.mean()
