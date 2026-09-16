"""Structured configuration for EOS.

All dataclass configs are registered with Hydra's ConfigStore so that they
can be composed via YAML overrides.  The top-level :class:`EOSConfig` groups
every sub-config needed for an experiment (environment, learner, model, etc.).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# ---------------------------------------------------------------------------
# Environment sub-configs
# ---------------------------------------------------------------------------


@dataclass
class ResourceTypeConfig:
    """Top-level definition of a resource type.

    Attributes
    ----------
    quantity : int
        Total number of units of this resource type in the simulation.
    fabrication_schedule : Dict[str, List[float]]
        Mapping from source site name to a sorted list of fabrication
        times (hours).  Empty until Phase 2.
    deadlines : List[float]
        Per-unit delivery deadlines (hours).  Empty until Phase 3.
    """

    quantity: int = 0
    fabrication_schedule: Dict[str, List[float]] = field(default_factory=dict)
    deadlines: List[float] = field(default_factory=list)


@dataclass
class SiteResourceSlot:
    """Per-type capacity declaration for a site or vessel.

    Attributes
    ----------
    capacity : int
        Maximum number of units of this resource type the entity can hold.
    """

    capacity: int = 0


@dataclass
class SiteConfig:
    """Static definition of a logistics site.

    Attributes
    ----------
    location : List[float]
        Geographic coordinates [x, y] of the site.
    resource_types : Dict[str, SiteResourceSlot]
        Per-resource-type capacity declarations for the site.
    initial_levels : Dict[str, int]
        Starting levels per resource type.
    total_capacity : int
        Maximum total number of resource units the site can hold.
    role : str
        Role of the site in the logistics network.  One of:

        * ``"source"`` – resources originate here; vessels may **load from**
          but never **unload to** a source site.
        * ``"installation"`` – resources are permanently installed here; only
          *installer* vessels may unload (install); nothing can be loaded.
        * ``"staging"`` – intermediate buffer; load and unload are both
          permitted.
    """

    location: List[float]
    resource_types: Dict[str, SiteResourceSlot] = field(default_factory=dict)
    initial_levels: Dict[str, int] = field(default_factory=dict)
    total_capacity: int = 0
    role: str = "staging"


@dataclass
class VesselConfig:
    """Static definition of a transport vessel.

    Attributes
    ----------
    type : str
        EventSymphony class name (``"TransportProcessingResource"`` or
        ``"InstallationAsset"``).
    start_location : str
        Name of the site where the vessel starts.
    resource_types : Dict[str, SiteResourceSlot]
        Per-resource-type capacity declarations for the vessel.
    initial_levels : Dict[str, int]
        Starting levels per resource type.  Vessels default to empty
        (all zeros) unless overridden here (e.g. pre-loaded cargo).
    total_capacity : int
        Maximum total number of resource units the vessel can carry.
    role : str
        Role of the vessel.  One of:

        * ``"heavy_lift"`` – long-haul bulk carrier (source → staging).
        * ``"feeder"`` – short-haul ferry (staging → installation).
        * ``"installer"`` – can install resources at installation sites.
          Whether it can also sail is controlled by *movable*.

        Both ``"heavy_lift"`` and ``"feeder"`` are transport-class roles:
        they can sail, load, unload, and transfer cargo but **cannot**
        install at installation sites.
    movable : bool
        Whether the vessel is allowed to move between sites.  Typically
        ``True`` for transport vessels and ``False`` for stationary
        installers, but can be overridden per-vessel.
    speed : float | None
        Vessel speed (distance-units per hour).
    loading_rate : float | None
        Loading rate parameter forwarded to EventSymphony.
    unloading_rate : float | None
        Unloading rate parameter forwarded to EventSymphony.
    """

    type: str
    start_location: str
    resource_types: Dict[str, SiteResourceSlot] = field(default_factory=dict)
    initial_levels: Dict[str, int] = field(default_factory=dict)
    total_capacity: int = 0
    role: str = "heavy_lift"
    movable: bool = True
    speed: float | None = None
    loading_rate: float | None = None
    unloading_rate: float | None = None


@dataclass
class InstallationTimesConfig:
    installation_site: float = 0.0


@dataclass
class StochasticityConfig:
    """Optional stochastic perturbation of activity durations.

    Disabled by default.  When ``enabled`` is ``False`` the activity
    builder never touches the RNG and every activity duration is
    byte-for-byte identical to the deterministic configuration, so
    setting this flag back to ``False`` (or deleting the block) fully
    restores deterministic behaviour.

    When enabled, each activity's nominal duration is multiplied by a
    random factor centred on ``1.0``.  The spread is controlled per
    activity type by a coefficient of variation (std / mean of the
    multiplier).  A coefficient of ``0.0`` leaves that activity type
    deterministic even while stochasticity is globally enabled.

    The RNG is the environment's per-episode ``np_random`` (seeded via
    ``reset(seed=...)``), so stochastic episodes remain reproducible and
    replayable.

    Attributes
    ----------
    enabled : bool
        Master switch.  ``False`` (default) => fully deterministic.
    distribution : str
        Multiplier distribution: ``"lognormal"`` (default, right-skewed,
        strictly positive), ``"triangular"`` or ``"uniform"`` (both
        symmetric and bounded).  Every choice has mean ``1.0`` and the
        requested coefficient of variation.
    move_cv, load_cv, unload_cv : float
        Coefficient of variation of the duration multiplier for move,
        load and unload activities respectively.  ``0.0`` keeps that
        activity type deterministic.
    min_multiplier, max_multiplier : float
        Hard clamps applied to every sampled multiplier, guarding against
        pathological draws (e.g. negative or extreme durations).
    """

    enabled: bool = False
    distribution: str = "lognormal"
    move_cv: float = 0.0
    load_cv: float = 0.0
    unload_cv: float = 0.0
    min_multiplier: float = 0.1
    max_multiplier: float = 5.0


@dataclass
class ActivityConfig:
    """Default durations for the various activity types and optional overrides."""

    move_default: float = 1.0
    load_default: float = 3.0
    unload_default: float = 3.0
    install_default: float = 6.0
    transfer_default: float = 4.0

    # Matrix of move durations: Origin -> Destination -> Duration
    move_matrix: Dict[str, Any] = field(default_factory=dict)

    # Unloads which should be considered installation activities
    installations: Dict[str, InstallationTimesConfig] = field(default_factory=dict)

    # Optional stochastic perturbation of activity durations (off by default)
    stochasticity: StochasticityConfig = field(default_factory=StochasticityConfig)


@dataclass
class GoalConfig:
    """A single delivery / installation goal the agent must fulfil.

    Attributes
    ----------
    location : str
        Target site name.
    resource_type : str
        Name of the resource type to deliver.
    quantity : int
        Number of units required.
    deadline : float | None
        Optional deadline (hours).
    reward_per_unit : float
        Reward granted per delivered unit.
    depends_on : List[int]
        Indices of goals that must be completed first.  Wired in Phase 4.
    """

    location: str = MISSING
    resource_type: str = MISSING
    quantity: int = MISSING
    deadline: float | None = None
    reward_per_unit: float = 1.0
    depends_on: List[int] = field(default_factory=list)


@dataclass
class SimConfig:
    """Domain-specific simulation parameters (sites, vessels, goals).

    Notes
    -----
    ``is_bokalift_movable`` is **deprecated**.  Use the per-vessel
    ``movable`` flag on :class:`VesselConfig` instead.  When the
    per-vessel flag is explicitly set it takes precedence; when it is
    left at its default the simulator falls back to this global flag
    for backward compatibility with older configs.
    """

    resource_types: Dict[str, ResourceTypeConfig] = field(default_factory=dict)
    sites: Dict[str, SiteConfig] = MISSING
    vessels: Dict[str, VesselConfig] = MISSING
    activities: ActivityConfig = field(default_factory=ActivityConfig)
    goals: List[GoalConfig] | None = field(default_factory=list)

    no_install_windows: List[List[float]] = field(default_factory=list)
    use_business_rules: bool = True
    restrict_feeders_to_relay: bool = False


# ---------------------------------------------------------------------------
# Modular cost component configs
# ---------------------------------------------------------------------------


@dataclass
class TravelCostConfig:
    """Per-vessel hourly travel cost.

    ``rates`` maps vessel names to their cost-per-hour while in transit.
    Vessels not listed are assumed to have zero travel cost.
    """

    enabled: bool = False
    weight: float = 1.0
    rates: Dict[str, float] = field(default_factory=dict)


@dataclass
class ElapsedTimeCostConfig:
    """Flat per-hour cost for total elapsed simulation time.

    Penalises wall-clock time regardless of vessel activity, incentivising
    the agent to find solutions that complete as quickly as possible.
    This replaces the old ``use_time`` / ``time_weight`` mechanism.
    """

    enabled: bool = False
    weight: float = 1.0
    rate: float = 1.0  # cost per hour of elapsed simulation time


@dataclass
class StorageCostConfig:
    """Per-resource per-site hourly storage cost.

    ``rates`` is a nested mapping:
    ``{site_name: {resource_name: cost_per_unit_per_hour}}``.
    Sites or resources not listed are assumed to have zero storage cost.
    """

    enabled: bool = False
    weight: float = 1.0
    rates: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostConfig:
    """Container for all modular cost components.

    ``weight`` is a global multiplier applied on top of each component's
    own weight.  Set it to ``0`` to disable all cost penalties at once.

    To add a new cost component:
    1.  Create a ``*CostConfig`` dataclass above.
    2.  Add a field here.
    3.  Register the component in
        :meth:`~eos.envs.simple_monopile_transport.costs.aggregator.CostAggregator._build_components`.
    """

    weight: float = 1.0
    elapsed_time: ElapsedTimeCostConfig = field(default_factory=ElapsedTimeCostConfig)
    travel: TravelCostConfig = field(default_factory=TravelCostConfig)
    storage: StorageCostConfig = field(default_factory=StorageCostConfig)


# ---------------------------------------------------------------------------
# Milestone (supply-chain progress) config
# ---------------------------------------------------------------------------


@dataclass
class MilestoneStageWeights:
    """Potential values for each supply-chain stage.

    Each value represents the *cumulative* fraction of a goal's
    ``reward_per_unit`` that has been "earned" when a resource unit reaches
    that stage.  The reward given at a transition is the *delta* between
    the new and old stage potentials.

    The stages form a monotonically increasing sequence along the
    logistics relay pipeline::

        at_source → on_heavy_lift → at_staging → on_feeder → on_installer → at_goal
    """

    at_source: float = 0.0
    on_heavy_lift: float = 0.2
    at_staging: float = 0.4
    on_feeder: float = 0.6
    on_installer: float = 0.8
    at_goal: float = 1.0


@dataclass
class MilestoneConfig:
    """Configuration for milestone-based (potential-based) reward shaping.

    When the reward mode is ``"milestone"``, the
    :class:`~eos.envs.simple_monopile_transport.milestone_tracker.MilestoneTracker`
    computes a per-step *potential delta* for every goal based on where
    resources sit in the supply chain.  The delta is scaled by a
    per-goal **urgency** factor derived from deadline proximity.

    Urgency modes
    ~~~~~~~~~~~~~
    * ``"linear"`` – ``urgency = 1 + scale × max(0, 1 − remaining/deadline)``
      Grows from 1.0 at episode start to ``1 + scale`` at the deadline.
    * ``"inverse"`` – ``urgency = d_max / d_g``
      Static ratio: tighter-deadline goals always get proportionally
      higher weight regardless of elapsed time.

    Attributes
    ----------
    stage_weights : MilestoneStageWeights
        Potential values per supply-chain stage.
    urgency_scale : float
        Amplification factor for deadline proximity (``linear`` mode).
    urgency_mode : str
        How deadline proximity is converted to a scalar multiplier.
    phi_max_fraction : float
        Bounds the maximum DPBRS potential as a fraction of the terminal
        bonus to prevent the completion-avoidance pathology.
    terminal_zeroing : str
        When the dynamic potential Φ is forced to 0 at a terminal step.

        * ``"always"`` (default) – zero on *any* termination (success,
          failure, or time-limit). Strict DPBRS: the shaping sum telescopes
          exactly, so policy invariance is preserved.
        * ``"failure_only"`` – zero on failure / timeout, but keep Φ on a
          successful completion. The final goal-completing step then earns a
          bounded ``+`` shaping bonus (γ·Φ − Φ_prev) instead of the
          ``−Φ_prev`` cliff, deliberately breaking strict invariance to remove
          the completion-avoidance pathology (the 11/12 stall).
        * ``"never"`` – never zero. Strongest completion pull but also banks a
          phantom Φ at timeouts; generally not recommended.
    """

    stage_weights: MilestoneStageWeights = field(default_factory=MilestoneStageWeights)
    urgency_scale: float = 1.0
    urgency_mode: str = "linear"
    phi_max_fraction: float = 0.1
    terminal_zeroing: str = "always"  # "always" | "failure_only" | "never"


# ---------------------------------------------------------------------------
# PFM (Preference Function Modeling) config
# ---------------------------------------------------------------------------


@dataclass
class PFMObjectiveConfig:
    """Configuration for a single objective in the PFM multi-objective reward.

    Each objective maps to a raw physical metric exposed in the environment's
    ``info`` dict at terminal steps.  The ``weight`` fields across all
    objectives must sum to 1.0 (enforced at wrapper init).

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. ``"time"``, ``"cost"``).
    weight : float
        *A priori* stakeholder preference weight.  All weights must sum to 1.
    sigma_min : float
        Minimum standard deviation floor.  Prevents a highly-optimised
        objective from becoming hyper-sensitive and overriding other
        objectives' weights.  Set to the irreducible noise floor of the
        metric (e.g. weather-induced jitter).
    direction : str
        ``"minimize"`` (lower is better) or ``"maximize"`` (higher is better).
        Controls the sign of the Z-score so that improvement always maps to
        a higher P* value.
    info_key : str
        Top-level key in the environment ``info`` dict where the raw metric
        is exposed (e.g. ``"elapsed_time_hours"``).
    """

    name: str = MISSING
    weight: float = MISSING
    sigma_min: float = MISSING
    direction: str = "minimize"
    info_key: str = MISSING
    pf_type: str = "linear"
    pf_worst: float | None = None
    pf_best: float | None = None


@dataclass
class PFMConfig:
    """Preference Function Modeling reward normalization.

    When ``enabled``, a :class:`PFMVectorWrapper` is applied *after*
    vectorisation.  It transforms raw terminal ``[Time, Cost, …]`` vectors
    into a dimensionless Linear Preference Space (LPS) via Z-score
    normalization, then linearly scalarises them using the stakeholder
    ``objectives`` weights (the unique admissible aggregation operator
    under PFM axioms — Wolfert, 2026).

    The wrapper maintains running statistics (Welford + Polyak averaging)
    centrally in the main process, guaranteeing all parallel environments
    normalise against identical μ and σ.

    Attributes
    ----------
    enabled : bool
        Master toggle.  When ``False``, the entire PFM pipeline is skipped
        and the reward system behaves identically to the pre-PFM baseline.
    objectives : List[PFMObjectiveConfig]
        Ordered list of objectives.  Weights must sum to 1.
    burn_in_trajectories : int
        Number of *successful* completions observed before PFM Z-scores
        activate.  During burn-in the agent receives a flat
        ``completion_baseline`` for every successful completion.
    completion_baseline : float
        Constant reward added on every successful terminal step (both
        during and after burn-in).  Provides a permanent reward gap
        between completing and failing.
    catastrophic_penalty : float
        Reward for deadlock / failure terminal steps.  Should satisfy
        ``catastrophic_penalty < completion_baseline - clip_bound`` to
        guarantee the agent always prefers completing over aborting.
        Exempt from reward clipping.
    polyak_tau : float
        Soft-update coefficient for the Target statistics tracker
        (τ ≈ 0.001).  Lower values produce a more stable LPS reference
        frame at the cost of slower adaptation.
    clip_bound : float
        Symmetric bound for clipping the PFM P* score.  The final
        terminal reward is clamped to ``[-clip_bound, +clip_bound]``
        before ``completion_baseline`` is added.  The
        ``catastrophic_penalty`` is exempt from this clipping.
    phi_optimism : float
        Number of standard deviations above the baseline to assume the
        converged policy will achieve.  The effective ``phi_max`` used
        for DPBRS scaling is computed as::

            phi_max = completion_baseline + phi_optimism

        This aligns the DPBRS magnitude with the expected terminal
        reward of an "optimistic" converged solution, avoiding extreme
        reward-scale mismatches between the dense shaping signal and
        the sparse PFM terminal.
    scale_warmup : bool
        When ``True``, the PFM preference signal is phased in *during*
        burn-in: the clipped P* contribution is multiplied by a ramp
        ``lambda`` that climbs linearly from ``0`` on the first successful
        completion to ``1`` on the final burn-in completion
        (``_active_count == burn_in_trajectories``).  During burn-in the
        Z-score is taken against the running Active (Welford) statistics,
        since the Target frame is not yet initialised.  This keeps a weak
        but growing preference gradient alive throughout burn-in so the
        policy never collapses into a preference-blind "complete-at-any-
        cost" basin, and makes the handoff to full PFM continuous —
        avoiding the catastrophic forgetting seen at a hard turn-on.
        When ``False`` (default), ``lambda = 0`` for the entire burn-in
        (flat ``completion_baseline``) and ``lambda = 1`` afterwards,
        exactly recovering the original behaviour.
    """

    enabled: bool = False
    objectives: List[PFMObjectiveConfig] = field(default_factory=list)

    # Burn-in
    burn_in_trajectories: int = 500

    # Terminal reward structure
    completion_baseline: float = 10.0
    catastrophic_penalty: float = 0.0

    # Normalization
    polyak_tau: float = 0.001
    clip_bound: float = 5.0

    # Handoff smoothing: ramp the PFM signal in over burn-in (Z-scored
    # against running Active stats) so it reaches full strength exactly
    # when Target is seeded, avoiding a discontinuous turn-on.
    scale_warmup: bool = False

    # PBRS alignment
    phi_optimism: float = 3.0


# ---------------------------------------------------------------------------
# Reward config
# ---------------------------------------------------------------------------


@dataclass
class RewardConfig:
    """Reward configuration for milestone DPBRS shaping + sparse terminal cost evaluation.

    The reward system has a single architecture:

    * **Per-step**: DPBRS milestone shaping provides dense exploration
      guidance as resources advance through the supply chain.
    * **Terminal success**: ``completion_bonus - w * sqrt(total_cost)``
      where ``w = completion_bonus / sqrt(C_max)``.
    * **Terminal failure**: ``0`` (no reward).

    When PFM is active, the rewarder provides only DPBRS shaping and
    terminal reward is handled by the PFMVectorWrapper.

    Cost components are configured via the nested ``costs`` sub-config.
    """

    # Terminal bonus for successful completion.
    # Also sets the reward range: [0, completion_bonus].
    completion_bonus: float = 50.0

    # Modular cost components (elapsed time, travel, storage, …)
    costs: CostConfig = field(default_factory=CostConfig)

    # Milestone (supply-chain progress) DPBRS shaping
    milestone: MilestoneConfig = field(default_factory=MilestoneConfig)

    # Preference Function Modeling (multi-objective normalization)
    pfm: PFMConfig = field(default_factory=PFMConfig)


@dataclass
class RenderConfig:
    """Visualisation / video-capture settings."""

    mode: str | None = None
    width: int = 1280
    height: int = 720
    panel_width: int = 420
    font_size: int = 16
    fps: int = 30
    show_inventory: bool = True


@dataclass
class EnvConfig:
    """Everything the environment needs: gym ID, wrappers, masking, reward, etc."""

    env_id: str = MISSING

    # Detailed simulation config (optional, only for sim envs)
    sim: SimConfig | None = None

    # List of wrapper class paths to apply in order
    wrappers: List[str] = field(default_factory=list)

    # Whether the environment exposes action masks via an ``action_masks()`` method.
    # When False, masking is completely disabled end-to-end (runner / buffer / model).
    use_action_masks: bool = False

    # Allow IDLE actions in discrete wrappers.  When True, vessels can
    # choose an event-driven "await-event" idle that completes when the
    # next non-idle DES activity finishes.  This prevents aimless sailing
    # when no productive action is available.
    allow_idle_actions: bool = True

    # Ablation toggle (Ch.5 ESR -> POMDP claim): when False, the accumulated
    # travel/storage cost statistics in the global observation token are zeroed,
    # removing the path-dependent state augmentation while keeping the obs shape
    # (and network capacity) identical.
    augment_cost_state: bool = True

    reward: RewardConfig = field(default_factory=RewardConfig)
    time_budget: float = 240.0  # 10 days in hours
    render: RenderConfig = field(default_factory=RenderConfig)
    max_episode_steps: int = 256

    # The beta used by the learner to compute gamma_t (gamma_t = torch.exp(-beta * self.delta_times[t]))
    beta: float | None = None


# ---------------------------------------------------------------------------
# Learner configs
# ---------------------------------------------------------------------------


@dataclass
class PPOLearnerConfig:
    """Hyper-parameters for the PPO update rule."""

    init_lr: float = 2.75e-4
    final_lr: float = 1.0e-5
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    clip_vloss: bool = True
    # Per-factor PPO clipping for autoregressive (MultiDiscrete) action spaces.
    # When True, each of the 2K sub-decision ratios is clipped independently
    # against the shared macro-advantage. When False (ablation), the factors are
    # collapsed into a single joint ratio before clipping (standard PPO surrogate),
    # which is prone to "gradient blackout" as K grows.
    per_factor_clip: bool = True
    ent_coef: float = 0.01
    # Optional linear annealing of the entropy coefficient from ``ent_coef``
    # down to ``ent_coef_final`` over training (same schedule as ``anneal_lr``).
    # Lets the policy stay exploratory early but sharpen late so deterministic
    # (argmax) rollouts commit. Disabled by default (constant ``ent_coef``).
    anneal_ent: bool = False
    ent_coef_final: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    gamma: float = 0.99
    beta: float = 0.001
    anneal_lr: bool = True
    gae_lambda: float = 0.95
    # Early-stopping threshold on approximate KL divergence.
    # For autoregressive / factored action spaces with K sub-decisions,
    # KL is additive across factors, so this should be ~K× larger than
    # the single-action default of 0.01 (e.g. 0.05–0.08 for K=6).
    target_kl: float | None = None

    # Derived at runtime by the experiment
    num_iterations: int | None = None
    batch_size: int | None = None
    minibatch_size: int | None = None


# ---------------------------------------------------------------------------
# Controller configs
# ---------------------------------------------------------------------------


@dataclass
class ControllerConfig:
    """Controller settings (e.g. whether to act greedily)."""

    _target_: str = "..."


@dataclass
class PPOControllerConfig(ControllerConfig):
    _target_: str = "eos.controllers.PPOController"
    deterministic: bool = False


@dataclass
class RandomControllerConfig(ControllerConfig):
    _target_: str = "eos.controllers.RandomController"


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Base model config — concrete models override ``_target_``."""

    _target_: str = "..."


@dataclass
class MLPModelConfig(ModelConfig):
    """Config for the MLP actor-critic agent.

    Implements the AEC micro-stepping interface with a learned vessel
    ordering head and per-vessel actor logits.
    """

    _target_: str = "eos.models.MLPAgent"
    hidden_size: int = 64


@dataclass
class TransformerModelConfig(ModelConfig):
    """Config for the Transformer actor-critic agent.

    Uses a schema-driven input encoding that embeds categorical features
    and projects continuous features per entity type, then contextualises
    the sequence with a Transformer encoder.

    Implements the AEC micro-stepping interface with a learned vessel
    ordering head and per-vessel actor branches.
    """

    _target_: str = "eos.models.TransformerAgent"
    d_model: int = 128
    n_heads: int = 4
    dim_feedforward: int = 512
    batch_first: bool = True
    num_layers: int = 2
    emb_dim: int = 8
    freeze_ordering: bool = False


# ---------------------------------------------------------------------------
# Experiment-level configs
# ---------------------------------------------------------------------------


@dataclass
class CheckpointConfig:
    """When and how to save the best model checkpoint."""

    enable: bool = True
    eval_after_steps: int = 200000
    metric: str = "episodic_returns"
    mode: str = "max"
    save_dir: str = MISSING
    # Save an extra snapshot every N environment steps, independent of whether
    # the metric improved. 0 disables it (the historical behaviour).
    #
    # Best-so-far saving alone yields a monotone chain in `metric`, which for a
    # weighted-objective metric converges on that weighting's corner of the
    # trade-off — leaving the archive sparse exactly where intermediate
    # solutions would be. Snapshots sample the trajectory unconditionally, so
    # an a-posteriori multi-objective selection has genuine candidates to
    # choose between. Logged to a separate `<run>-snapshot` artifact stream so
    # `<run>-best` keeps meaning "the best checkpoint".
    save_every_steps: int = 0


@dataclass
class EvalConfig:
    """Post-training evaluation settings."""

    enable: bool = True
    num_episodes: int = 1
    deterministic: bool = True
    capture_video: bool = True
    render_mode: str | None = None
    # When True, the end-of-training deterministic eval scores the FINAL
    # (latest) weights instead of loading best.pt. Stability sweeps use this
    # to measure the reliability of the checkpoint you actually end up with.
    on_final_weights: bool = False


@dataclass
class TuneConfig:
    """Optuna hyperparameter sweep configuration."""

    enable: bool = False
    study_name: str = "smt_ppo_tuning"
    n_trials: int = 75
    storage: str = "optuna_journal.log"
    direction: str = "minimize"  # "minimize" or "maximize"
    prune_after_steps: int = (
        0  # Report & allow pruning after this many steps (0 = no pruning)
    )
    metric: str = "env/pfm/time/raw"  # Metric to optimize (group/key format)
    seeds: List[int] = field(default_factory=lambda: [1])  # Seeds for multi-seed trials
    objective: str = "pfm"  # Which objective function to use: "pfm" or "time_only"


@dataclass
class RandomExperimentConfig:
    """Settings specific to the random-action baseline experiment."""

    num_episodes: int = 10
    deterministic_seed: bool = True


@dataclass
class EOSConfig:
    """Top-level configuration for an EOS experiment run.

    Groups every sub-config (env, learner, model, …) and global settings
    such as seed, logging, and WandB integration.
    """

    # Global settings
    exp_name: str = "experiment"
    run_name: str | None = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = False
    track: bool = False
    wandb_project_name: str = "eos"
    wandb_entity: str = "dimitrisxynogalas-tu-delft"
    capture_video: bool = False
    total_timesteps: int = 500000
    num_envs: int = 4
    num_steps: int = 128
    normalize_rewards: bool = True
    debug: bool = False
    log_level: str = "DEBUG"
    log_dir_level: str = "DEBUG"
    log_every_updates: int = 1
    log_env_info: bool = True
    log_rollout_stats: bool = True

    # Modular components
    env: EnvConfig = MISSING
    learner: PPOLearnerConfig = MISSING
    controller: ControllerConfig = MISSING
    model: ModelConfig = MISSING
    _target_: str = "eos.core.experiment.Experiment"

    checkpoint: CheckpointConfig = MISSING
    eval: EvalConfig = MISSING
    tune: TuneConfig = field(default_factory=TuneConfig)

    # Random-baseline experiment settings (used by RandomExperiment)
    random: RandomExperimentConfig = field(default_factory=RandomExperimentConfig)


# ---------------------------------------------------------------------------
# Hydra registration
# ---------------------------------------------------------------------------


def register_configs() -> None:
    """Register all structured configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="eos_schema", node=EOSConfig)
    cs.store(group="model", name="mlp_schema", node=MLPModelConfig)
    cs.store(group="model", name="transformer_schema", node=TransformerModelConfig)
    cs.store(group="controller", name="ppo_controller_schema", node=PPOControllerConfig)
    cs.store(
        group="controller", name="random_controller_schema", node=RandomControllerConfig
    )
