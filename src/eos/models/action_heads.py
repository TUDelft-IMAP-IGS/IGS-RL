"""Shared action-head utilities for AEC micro-stepping models.

This module extracts the common logic used by both :class:`MLPAgent` and
:class:`TransformerAgent` during the two-phase micro-stepping loop:

* :func:`sample_ordering` — Phase 1: iteratively sample a vessel
  permutation without replacement, zero-out busy vessels' contributions.
* :func:`select_masked_action` — Phase 2: apply an action mask, build a
  Categorical distribution, and sample (or replay) a single action.

By keeping these as pure functions that operate on pre-computed logits,
the model classes stay focused on *how* logits are produced (MLP vs.
Transformer backbone) while sharing identical sampling semantics.
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Phase 1 — Vessel ordering
# ---------------------------------------------------------------------------


def sample_ordering(
    ordering_logits: torch.Tensor,
    vessel_availability: torch.Tensor,
    ordering: torch.Tensor | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a vessel permutation via iterative without-replacement selection.

    All vessels (busy *and* idle) appear exactly once in the resulting
    permutation, but busy vessels have their log-probability and entropy
    contributions zeroed out since they will be skipped by the runner.

    Parameters
    ----------
    ordering_logits : Tensor ``(B, n_vessels)``
        Raw (un-masked) logits from the ordering head.
    vessel_availability : Tensor ``(B, n_vessels)``
        Bool tensor — ``True`` for idle vessels, ``False`` for busy.
    ordering : Tensor ``(B, n_vessels)`` | None
        Pre-recorded ordering for training replay.  When ``None``, a
        fresh ordering is sampled (or chosen greedily).
    deterministic : bool
        If ``True``, use argmax instead of sampling at each position.

    Returns
    -------
    ordering_t : ``(B, n_vessels)``
        The sampled vessel-index permutation.
    ordering_lp_by_vessel : ``(B, n_vessels)``
        Per-vessel log-probabilities of the ordering choices (indexed by
        vessel ID, not by selection position).  Busy vessels get 0.
    ordering_ent_by_vessel : ``(B, n_vessels)``
        Per-vessel entropies, same indexing convention as above.
    """
    batch_size, n_vessels = ordering_logits.shape
    device = ordering_logits.device

    # 1. Determine the ordering permutation
    if ordering is None:
        scores = ordering_logits.clone()
        if not deterministic:
            # Gumbel-max trick for sampling without replacement
            # U ~ Uniform(0, 1), G = -log(-log(U))
            u = torch.rand_like(scores).clamp(min=1e-8, max=1.0 - 1e-8)
            gumbel_noise = -torch.log(-torch.log(u))
            scores = scores + gumbel_noise

        # Ensure busy vessels are placed at the end by giving them a huge penalty
        scores = scores.masked_fill(~vessel_availability, -torch.inf)
        ordering_t = scores.argsort(dim=-1, descending=True)
    else:
        ordering_t = ordering.long()

    # 2. Compute probabilities and entropies vectorized over the sequence
    # Mask out busy vessels entirely from the distribution
    masked_logits = ordering_logits.masked_fill(~vessel_availability, -1e9)

    # Gather logits into the order they were picked: (B, n_vessels)
    ordered_logits = masked_logits.gather(1, ordering_t)

    # Create an upper triangular mask to represent "unpicked" vessels at each step
    # mask[t, k] = True if k >= t
    idx = torch.arange(n_vessels, device=device)
    mask = idx.unsqueeze(0) >= idx.unsqueeze(1)  # (n_vessels, n_vessels)

    # Expand to (B, n_vessels, n_vessels). Dim 1 is step (t), dim 2 is candidate (k).
    expanded_logits = ordered_logits.unsqueeze(1).expand(-1, n_vessels, -1)

    # At step t, vessels with k < t have already been picked, mask them out
    step_logits = expanded_logits.masked_fill(~mask, -1e9)  # (B, t, k)

    # 3. Use Categorical to compute log_probs and entropies for each step safely
    dist = Categorical(logits=step_logits)

    # Because step_logits aligns with ordered_logits, the chosen vessel at step t
    # is always at index t in the candidate dimension.
    actions = idx.unsqueeze(0).expand(batch_size, -1)

    log_probs_ordered = dist.log_prob(actions)
    entropies_ordered = dist.entropy()

    # 4. Zero out contributions from padding steps
    # The first `num_idle` steps are valid choices among idle vessels.
    # Subsequent steps are deterministic padding (forcing busy vessels).
    num_idle = vessel_availability.sum(dim=-1, keepdim=True)  # (B, 1)
    is_padding = idx.unsqueeze(0) >= num_idle  # broadcasts to (B, n_vessels)

    log_probs_ordered = torch.where(
        is_padding, torch.zeros_like(log_probs_ordered), log_probs_ordered
    )
    entropies_ordered = torch.where(
        is_padding, torch.zeros_like(entropies_ordered), entropies_ordered
    )

    # 5. Re-index from selection-position order → vessel-ID order
    inv_ordering = ordering_t.argsort(dim=1)
    ordering_lp_by_vessel = log_probs_ordered.gather(1, inv_ordering)
    ordering_ent_by_vessel = entropies_ordered.gather(1, inv_ordering)

    return ordering_t, ordering_lp_by_vessel, ordering_ent_by_vessel


# ---------------------------------------------------------------------------
# Phase 2 — Masked action selection
# ---------------------------------------------------------------------------


def select_masked_action(
    logits: torch.Tensor,
    action_mask: torch.Tensor | None = None,
    action: torch.Tensor | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a mask, build a Categorical, and sample or replay an action.

    This is a pure function that encapsulates the mask → distribution →
    sample/replay pattern shared by every model's Phase 2.

    Parameters
    ----------
    logits : Tensor ``(B, opts_per_vessel)``
        Raw actor logits for the acting vessel (already selected from
        the per-vessel branches by the caller).
    action_mask : Tensor ``(B, opts_per_vessel)`` | None
        Boolean mask of valid actions.  Invalid entries are filled with
        ``-1e9`` before the softmax.  ``None`` means all actions valid.
    action : Tensor ``(B,)`` | None
        Pre-recorded action for training replay.  When ``None``, a
        fresh action is sampled (or chosen greedily).
    deterministic : bool
        If ``True``, use argmax instead of sampling.

    Returns
    -------
    action : ``(B,)``
        The chosen action index.
    logprob : ``(B,)``
        Log-probability of the chosen action under the masked
        distribution.
    entropy : ``(B,)``
        Entropy of the masked action distribution.
    """
    if action_mask is not None:
        mask = torch.as_tensor(action_mask, device=logits.device)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        logits = logits.masked_fill(~mask, -1e9)

    dist = Categorical(logits=logits)

    if action is None:
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
    else:
        action = action.long()

    return action, dist.log_prob(action), dist.entropy()
