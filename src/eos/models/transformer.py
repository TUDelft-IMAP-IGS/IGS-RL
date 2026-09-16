"""Transformer actor-critic agent for PPO.

Uses a schema-driven input encoding that embeds categorical features and
projects continuous features per entity type, then contextualises the full
sequence with a standard Transformer encoder.

Implements the **Intent-Based Micro-Stepping (AEC)** interface defined by
:class:`~eos.core.model.Model`:

* :meth:`get_ordering` — Phase 1: learned vessel permutation.
* :meth:`get_action_and_value` — Phase 2: per-vessel action selection
  with masked logits and critic value estimate.
"""

# type: ignore
import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from eos.config import TransformerModelConfig
from eos.core.model import Model
from eos.models.action_heads import sample_ordering, select_masked_action

from .utils import layer_init

# Architecture Note (Roadmap):
# Pairwise/graph information (e.g. transit duration matrix, spatial distances)
# can be injected explicitly as relative position encodings, attention biases,
# or edge features in a graph-transformer variant if spatial layouts become highly dynamic.


class TransformerAgent(Model):
    """Transformer-based actor-critic for AEC micro-stepping.

    Architecture
    ------------
    * **Input encoding** — per-entity-type linear projections with learned
      categorical embeddings and additive type embeddings.
    * **Transformer encoder** — standard ``nn.TransformerEncoder`` that
      contextualises the full entity sequence.
    * **Critic** — linear head on the global token (row 0) → ``(B, 1)``.
    * **Actor branches** — one linear head per vessel, each producing
      ``opts_per_vessel`` logits.
    * **Ordering head** — linear head that produces logits for the vessel
      permutation used by Phase 1 of the AEC loop.

    Parameters
    ----------
    envs :
        Vectorised gymnasium environment (used to inspect observation /
        action spaces and the structured-observation schema).
    cfg : TransformerModelConfig
        Model hyper-parameters (d_model, n_heads, etc.).
    """

    def __init__(self, envs, cfg: TransformerModelConfig):
        super().__init__()

        action_space = envs.single_action_space
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            nvec = np.asarray(action_space.nvec, dtype=int)
            if not np.all(nvec == nvec[0]):
                raise ValueError(
                    "MultiDiscrete action space requires equal per-dimension "
                    "sizes for masking."
                )
            self.action_space_shape = nvec
            self.is_multidiscrete = True
            self.n_vessels = int(nvec.shape[0])
            self.opts_per_vessel = int(nvec[0])
        else:
            self.action_space_shape = np.array(
                [envs.single_action_space.n], dtype=np.int64
            )
            self.is_multidiscrete = False
            self.n_vessels = None
            self.opts_per_vessel = int(self.action_space_shape[0])

        self.d_model = cfg.d_model

        # --- 1. CONFIGURATION ---
        # Get the structured-observation schema from the wrapper.  The schema
        # describes each entity type's feature count, categorical fields, and
        # the slice of rows it occupies in the padded observation tensor.
        self.schema = envs.get_attr("schema")[0]

        # --- 2. SETUP EMBEDDINGS & ENCODERS ---
        self.embeddings = nn.ModuleDict()
        self.encoders = nn.ModuleDict()
        self.type_embeddings = nn.ParameterDict()

        self.emb_dim = cfg.emb_dim  # Dimension for categorical embeddings
        self.freeze_ordering = getattr(cfg, "freeze_ordering", False)

        for name, config in self.schema.items():
            categorical_map = config.get("categoricals", {})

            # A. Create categorical embeddings
            for cat_field, vocab_size in categorical_map.items():
                key = f"{name}_{cat_field}".replace(".", "_")
                self.embeddings[key] = nn.Embedding(vocab_size, self.emb_dim)

            # B. Calculate input dimension
            # input_width = num_continuous + num_categorical * emb_dim
            n_cats = len(categorical_map)
            n_total = config["feats"]
            n_continuous = n_total - n_cats
            input_dim = n_continuous + (n_cats * self.emb_dim)

            # C. Linear encoder → d_model
            self.encoders[name] = nn.Sequential(
                layer_init(nn.Linear(input_dim, self.d_model)),
                nn.LayerNorm(self.d_model),
                nn.ReLU(),
            )

            # D. Additive type embedding (distinguishes entity types)
            self.type_embeddings[name] = nn.Parameter(torch.randn(1, 1, self.d_model))

        # --- 3. TRANSFORMER ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            batch_first=cfg.batch_first,
            norm_first=True,  # Pre-norm for more stable training
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers
        )

        # --- 4. ACTOR & CRITIC HEADS ---
        self.actor_branches = nn.ModuleList(
            [
                layer_init(
                    nn.Linear(self.d_model, out_features=num_actions),
                    math.sqrt(0.01),
                )
                for num_actions in self.action_space_shape
            ]
        )
        self.critic = layer_init(nn.Linear(self.d_model, 1))

        # --- 5. ORDERING HEAD ---
        # Always created (even for Discrete spaces with a single "vessel")
        # so that the interface is uniform.
        n_ord = self.n_vessels if self.n_vessels is not None else 1
        self.ordering_head = nn.Sequential(
            layer_init(nn.Linear(self.d_model, self.d_model)),
            nn.Tanh(),
            layer_init(nn.Linear(self.d_model, n_ord), std=0.01),
        )

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the structured observation tensor through the Transformer.

        Parameters
        ----------
        x : Tensor
            Padded observation of shape ``(B, total_rows, max_feats)``.

        Returns
        -------
        Tensor
            Contextualised sequence of shape ``(B, total_rows, d_model)``.
        """
        embeddings = []

        for name, config in self.schema.items():
            # 1. Slice raw data for this entity type: (B, N_entities, N_feats)
            raw_data = x[:, config["slice"], : config["feats"]]

            categorical_map = config.get("categoricals", {})

            # 2. Process features: embed categoricals, pass through continuous
            processed_feats = []
            for i, feat_name in enumerate(config["names"]):
                col_data = raw_data[:, :, i : i + 1]  # (B, N, 1)

                if feat_name in categorical_map:
                    vocab_size = categorical_map[feat_name]
                    indices = col_data.long().squeeze(-1)
                    indices = indices.clamp(0, vocab_size - 1)

                    key = f"{name}_{feat_name}".replace(".", "_")
                    emb_vec = self.embeddings[key](indices)  # (B, N, emb_dim)
                    processed_feats.append(emb_vec)
                else:
                    processed_feats.append(col_data)

            # 3. Concatenate all features for this entity type
            combined_input = torch.cat(processed_feats, dim=2)

            # 4. Project to d_model
            encoded = self.encoders[name](combined_input)

            # 5. Add type embedding
            encoded = encoded + self.type_embeddings[name]

            embeddings.append(encoded)

        # Reassemble the full sequence and contextualise
        combined_seq = torch.cat(embeddings, dim=1)
        contextualized = self.transformer(combined_seq)

        return contextualized

    # ------------------------------------------------------------------
    # Public API — AEC Micro-Stepping Interface
    # ------------------------------------------------------------------

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return the critic's state-value estimate, shape ``(B, 1)``."""
        x = self.forward(x)
        return self.critic(x[:, 0, :])

    def get_logits(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute (optionally masked) actor logits from the global token.

        ``x`` should already be the contextualised global-token row
        (shape ``(B, d_model)``).

        For Discrete spaces the output shape is ``(B, action_dim)``.
        For MultiDiscrete spaces the output is stacked to
        ``(B, n_vessels, opts_per_vessel)``.
        """
        logits = [branch(x) for branch in self.actor_branches]
        logits = torch.stack(logits, dim=1).squeeze(dim=1)

        if action_mask is not None:
            mask = torch.as_tensor(action_mask, device=logits.device)
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            logits = logits.masked_fill(~mask, -1e9)

        return logits

    def get_ordering(
        self,
        x: torch.Tensor,
        vessel_availability: torch.Tensor,
        ordering: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase 1: Determine the optimal sequence for vessels to act.

        Uses the ordering head to sample a permutation of vessels via
        iterative sampling without replacement.  Busy vessels are included
        in the permutation but their log-probability and entropy
        contributions are zeroed out since they skip their turn.

        Parameters
        ----------
        x : Tensor
            The macro-observation before any micro-steps begin.
            Shape ``(B, seq_len, max_feats)``.
        vessel_availability : Tensor
            Bool tensor ``(B, n_vessels)`` indicating which vessels are
            currently IDLE.
        ordering : Tensor | None
            Stored ordering for replay during PPO training.
        deterministic : bool
            If True, uses argmax instead of sampling.

        Returns
        -------
        ordering_t : ``(B, n_vessels)``
            The sampled sequence of vessel indices.
        ordering_lp_by_vessel : ``(B, n_vessels)``
            Log-probabilities of the ordering choices, indexed by vessel ID.
        ordering_ent_by_vessel : ``(B, n_vessels)``
            Entropies of the ordering choices, indexed by vessel ID.
        """
        x = self.forward(x)
        global_state = x[:, 0, :]
        ordering_logits = self.ordering_head(global_state)  # (B, n_vessels)
        if self.freeze_ordering:
            ordering_logits = torch.zeros_like(ordering_logits)

        return sample_ordering(
            ordering_logits,
            vessel_availability,
            ordering=ordering,
            deterministic=deterministic,
        )

    def get_ordering_and_value(
        self,
        x: torch.Tensor,
        vessel_availability: torch.Tensor,
        ordering: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-pass ordering + baseline value.

        Both the ordering head and the critic read the global token of the
        *same* ``forward(x)`` output, so this returns exactly what calling
        :meth:`get_ordering` and :meth:`get_value` separately would (the
        backbone is deterministic: dropout is disabled and only LayerNorm
        is used), while running the expensive backbone only once.
        """
        contextualized = self.forward(x)  # (B, N_seq, d_model)
        global_state = contextualized[:, 0, :]  # (B, d_model)

        value = self.critic(global_state)

        ordering_logits = self.ordering_head(global_state)  # (B, n_vessels)
        if self.freeze_ordering:
            ordering_logits = torch.zeros_like(ordering_logits)

        ordering_t, logprob_by_vessel, entropy_by_vessel = sample_ordering(
            ordering_logits,
            vessel_availability,
            ordering=ordering,
            deterministic=deterministic,
        )
        return ordering_t, logprob_by_vessel, entropy_by_vessel, value

    def get_action_and_value(
        self,
        x: torch.Tensor,
        vessel_indices: torch.Tensor,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase 2: Select an action for a specific vessel (micro-step).

        Parameters
        ----------
        x : Tensor
            The micro-observation at this exact step.
            Shape ``(B, seq_len, max_feats)``.
        vessel_indices : Tensor
            1D integer tensor ``(B,)`` denoting WHICH vessel is acting.
        action : Tensor | None
            Stored action for replay during PPO training.
        action_mask : Tensor | None
            Boolean mask ``(B, opts_per_vessel)`` for the acting vessel's
            options.
        deterministic : bool
            If True, uses argmax instead of sampling.

        Returns
        -------
        action : ``(B,)``
            The sampled discrete action index.
        logprob : ``(B,)``
            Log-probability of the chosen action.
        entropy : ``(B,)``
            Entropy of the action distribution.
        value : ``(B, 1)``
            Critic value estimate for the current micro-state.
        """
        x = self.forward(x)  # (B, N_seq, d_model)
        global_state = x[:, 0, :]  # (B, d_model)

        batch_size = x.shape[0]

        # Critic value
        value = self.critic(global_state)

        # Extract the starting row index for vessels from the schema
        vessel_start = self.schema["vessels"]["slice"].start

        # Actor logits: one branch per vessel → (B, n_vessels, opts_per_vessel)
        all_logits = torch.stack(
            [
                self.actor_branches[v](x[:, vessel_start + v, :])
                for v in range(len(self.actor_branches))
            ],
            dim=1,
        )

        # Extract only the branch belonging to the currently acting vessel
        selected_logits = all_logits[torch.arange(batch_size), vessel_indices, :]

        action, logprob, entropy = select_masked_action(
            selected_logits,
            action_mask=action_mask,
            action=action,
            deterministic=deterministic,
        )

        return action, logprob, entropy, value

    # ------------------------------------------------------------------
    # Training replay — full macro-step with sequential per-vessel obs
    # ------------------------------------------------------------------

    def replay_macro_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        ordering: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        vessel_availability: torch.Tensor | None = None,
        per_vessel_obs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay a stored macro-step to recompute logprobs, entropy, value.

        This is the training counterpart of the two-phase collection loop
        in :class:`~eos.runners.micro_step.MicroStepCollector`.

        **Sequential AEC replay** — when ``per_vessel_obs`` is provided
        (the normal path for MultiDiscrete), each vessel's action is
        replayed against the observation that vessel *actually* saw
        during collection.  This means one backbone forward pass per
        vessel (matching collection semantics exactly), rather than a
        single shared pass.  The ordering head still uses the macro-obs
        (``obs``) because the ordering decision is made before any
        intents are registered.

        When ``per_vessel_obs`` is ``None`` (legacy / Discrete fallback),
        all vessels share the macro-obs as before.

        Parameters
        ----------
        obs : Tensor ``(B, seq_len, max_feats)``
            Macro-observation — the state before any micro-steps.  Used
            for the ordering head and the critic value estimate.
        actions : Tensor ``(B, n_vessels)``
            Joint action recorded during collection.  Each entry is the
            discrete action index chosen for that vessel (0 = NOOP for
            busy vessels).
        ordering : Tensor ``(B, n_vessels)``
            Vessel permutation recorded during collection.
        action_mask : Tensor ``(B, n_vessels, opts_per_vessel)`` | None
            Per-vessel action masks recorded at each vessel's decision
            time during collection.
        vessel_availability : Tensor ``(B, n_vessels)`` | None
            Ground-truth idle flags recorded during collection.
        per_vessel_obs : Tensor ``(B, n_vessels, seq_len, max_feats)`` | None
            Per-vessel observations captured at each vessel's decision
            time during collection.  When provided, vessel *v*'s action
            is replayed against ``per_vessel_obs[:, v]`` instead of
            ``obs``.

        Returns
        -------
        per_factor_logprobs : ``(B, 2 * n_vessels)``
            Per-factor log-probs for per-factor PPO clipping.  First
            ``n_vessels`` columns are ordering; last ``n_vessels`` are
            per-vessel action log-probs.  Busy vessels contribute 0.
        aggregate_entropy : ``(B,)``
            Sum of ordering entropies + per-vessel action entropies for
            idle vessels.
        value : ``(B, 1)``
            Critic value estimate from the global token (macro-obs).
        """
        n_vessels = self.n_vessels or 1

        def _safe_range(t: torch.Tensor) -> str:
            """Format tensor range, handling all-NaN/empty cases."""
            valid = t[~(torch.isnan(t) | torch.isinf(t))]
            if valid.numel() == 0:
                return "[ALL NaN/Inf]"
            return f"[{valid.min().item():.4f}, {valid.max().item():.4f}]"

        # ── 1. Backbone on macro-obs (for ordering + critic) ─────────
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            nan_count = torch.isnan(obs).sum().item()
            inf_count = torch.isinf(obs).sum().item()
            raise RuntimeError(
                f"NaN/Inf in replay_macro_step INPUT obs: "
                f"{nan_count} NaN, {inf_count} Inf, "
                f"shape={obs.shape}, range={_safe_range(obs)}"
            )

        x_macro = self.forward(obs)  # (B, N_seq, d_model)

        if torch.isnan(x_macro).any():
            raise RuntimeError(
                f"NaN in transformer backbone output (macro-obs path): "
                f"x_macro shape={x_macro.shape}, "
                f"NaN count={torch.isnan(x_macro).sum().item()}/{x_macro.numel()}, "
                f"input obs range={_safe_range(obs)}"
            )

        global_state_macro = x_macro[:, 0, :]  # (B, d_model)
        batch_size = global_state_macro.shape[0]
        device = global_state_macro.device

        # ── 2. Critic value (from macro-obs, before any intents) ─────
        value = self.critic(global_state_macro)  # (B, 1)

        # ── 3. Resolve vessel availability & action mask ─────────────
        mask_t: torch.Tensor | None = None
        if action_mask is not None:
            mask_t = action_mask.to(device=device)
            if mask_t.dtype != torch.bool:
                mask_t = mask_t.bool()

        if vessel_availability is not None:
            v_avail = vessel_availability.to(device=device).bool()
        else:
            v_avail = torch.ones(batch_size, n_vessels, dtype=torch.bool, device=device)

        # ── 4. Replay ordering (Phase 1 — uses macro-obs) ────────────
        ordering_logits = self.ordering_head(global_state_macro)  # (B, n_vessels)
        if self.freeze_ordering:
            ordering_logits = torch.zeros_like(ordering_logits)

        if torch.isnan(ordering_logits).any():
            # Check if the problem is in global_state or in the head weights
            head_has_nan = any(
                torch.isnan(p).any().item() for p in self.ordering_head.parameters()
            )
            raise RuntimeError(
                f"NaN in ordering_logits: "
                f"shape={ordering_logits.shape}, "
                f"NaN count={torch.isnan(ordering_logits).sum().item()}, "
                f"global_state has NaN={torch.isnan(global_state_macro).any().item()}, "
                f"ordering_head weights have NaN={head_has_nan}"
            )

        _, ordering_lp_by_vessel, ordering_ent_by_vessel = sample_ordering(
            ordering_logits,
            v_avail,
            ordering=ordering.to(device=device).long(),
        )

        # ── 5. Replay per-vessel actions (Phase 2) ───────────────────
        # Each vessel gets its own backbone pass on the observation it
        # actually saw during collection (sequential AEC).
        actions_device = actions.to(device=device).long()  # (B, n_vessels)
        idle_f = v_avail.float()  # (B, n_vessels)

        action_lps = torch.zeros(batch_size, n_vessels, device=device)
        action_ents = torch.zeros(batch_size, n_vessels, device=device)

        if per_vessel_obs is not None:
            # ── Sequential AEC path ──────────────────────────────────
            # Batch all vessel observations together for a single large
            # forward pass: (B * n_vessels, seq_len, max_feats).
            pvo = per_vessel_obs.to(device=device)  # (B, V, S, F)

            if torch.isnan(pvo).any() or torch.isinf(pvo).any():
                nan_count = torch.isnan(pvo).sum().item()
                inf_count = torch.isinf(pvo).sum().item()
                # Find which vessels have NaN
                nan_per_vessel = [
                    torch.isnan(pvo[:, v]).any().item() for v in range(pvo.shape[1])
                ]
                raise RuntimeError(
                    f"NaN/Inf in per_vessel_obs: "
                    f"{nan_count} NaN, {inf_count} Inf, "
                    f"shape={pvo.shape}, "
                    f"per-vessel NaN={nan_per_vessel}, "
                    f"range={_safe_range(pvo)}"
                )

            B, V, S, F = pvo.shape
            pvo_flat = pvo.reshape(B * V, S, F)

            vessel_start = self.schema["vessels"]["slice"].start

            x_all = self.forward(pvo_flat)  # (B*V, N_seq, d_model)

            if torch.isnan(x_all).any():
                raise RuntimeError(
                    f"NaN in transformer backbone output (per-vessel-obs path): "
                    f"x_all shape={x_all.shape}, "
                    f"NaN count={torch.isnan(x_all).sum().item()}/{x_all.numel()}, "
                    f"input pvo range={_safe_range(pvo)}"
                )

            # Reshape full sequence back to (B, V, N_seq, d_model)
            x_per_vessel = x_all.reshape(B, V, x_all.shape[1], -1)

            for v in range(n_vessels):
                # Extract the specific token for vessel 'v' from the observation it saw
                vessel_token = x_per_vessel[:, v, vessel_start + v, :]  # (B, d_model)
                v_logits = self.actor_branches[v](vessel_token)  # (B, opts)

                if mask_t is not None:
                    v_mask = mask_t[:, v, :]
                    v_logits = v_logits.masked_fill(~v_mask, -1e9)

                _, lp, ent = select_masked_action(
                    v_logits,
                    action_mask=None,
                    action=actions_device[:, v],
                )
                action_lps[:, v] = lp * idle_f[:, v]
                action_ents[:, v] = ent * idle_f[:, v]
        else:
            # ── Legacy shared-obs path (Discrete fallback) ───────────
            vessel_start = self.schema["vessels"]["slice"].start
            all_logits = torch.stack(
                [
                    self.actor_branches[v](x_macro[:, vessel_start + v, :])
                    for v in range(n_vessels)
                ],
                dim=1,
            )  # (B, n_vessels, opts_per_vessel)

            for v in range(n_vessels):
                v_logits = all_logits[:, v, :]

                if mask_t is not None:
                    v_mask = mask_t[:, v, :]
                    v_logits = v_logits.masked_fill(~v_mask, -1e9)

                _, lp, ent = select_masked_action(
                    v_logits,
                    action_mask=None,
                    action=actions_device[:, v],
                )
                action_lps[:, v] = lp * idle_f[:, v]
                action_ents[:, v] = ent * idle_f[:, v]

        # ── 6. Aggregate ─────────────────────────────────────────────
        # Per-factor logprobs: (B, 2*n_vessels) — [ordering | action].
        # The learner uses these for per-factor PPO clipping, which
        # prevents trust-region shrinkage in autoregressive action spaces.
        per_factor_logprobs = torch.cat(
            [ordering_lp_by_vessel, action_lps], dim=1
        )  # (B, 2*n_vessels)

        aggregate_entropy = ordering_ent_by_vessel.sum(dim=1) + action_ents.sum(dim=1)

        return per_factor_logprobs, aggregate_entropy, value
