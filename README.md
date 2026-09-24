# EOS: Reinforcement Learning for Logistics Discrete-Event Simulation

**EOS** is a Reinforcement Learning (RL) framework engineered to optimize complex marine logistics in Discrete-Event Simulation (DES) environments. The primary environment is `SimpleMonopileTransport-v0` (SMT), which models the transport, staging, and installation of offshore wind turbine monopiles using Heavy Transport Vessels (HTVs), Installation Vessels (e.g. *installation_vessel*), and Feeder Barges between Fabrication Yards, Marshalling Ports, and Offshore Installation Sites.

The simulation core is powered by the [`des_package`](https://pkgs.internal.example.com/org/simulation/_packaging/internal-feed/pypi/simple/) DES engine. The primary RL algorithm is Proximal Policy Optimization (PPO), utilizing an entity-centric Transformer architecture capable of handling variable fleets, site inventories, and multi-vessel simultaneous decision-making.

---

## Table of Contents
1. [Core Architectural Paradigms](#core-architectural-paradigms)
   - [AEC Intent-Based Micro-Stepping](#aec-intent-based-micro-stepping)
   - [Reservation System & Shadow Ledger](#reservation-system--shadow-ledger)
   - [Reward Architecture (DPBRS + PFM)](#reward-architecture-dpbrs--pfm)
   - [Structured Observations & Transformer Agent](#structured-observations--transformer-agent)
2. [Repository Structure](#repository-structure)
3. [Configuration System](#configuration-system)
4. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Training Models](#training-models)
   - [Evaluating Models](#evaluating-models)
   - [Inspecting Configurations](#inspecting-configurations)
5. [Extending EOS](#extending-eos)

---

## Core Architectural Paradigms

### AEC Intent-Based Micro-Stepping
In standard discrete-event simulations, multiple agents (vessels) often become idle and require decisions at the exact same simulation timestamp t. However, their choices are coupled: if Vessel A reserves the last monopile at a port, Vessel B can no longer choose to load it.

To solve this without time advancement artifacts, EOS implements a two-phase **Agent-Environment-Cycle (AEC) micro-step loop**:

```text
[Simulation Event at time t] (Multiple vessels need orders)
       │
       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Phase 1: Ordering Head (TransformerAgent.get_ordering)      │
 │ Predict priority sequence for all active/idle vessels       │
 └─────────────────────────────┬───────────────────────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               │
 ┌───────────────────────────────────────────────┐     │
 │ Phase 2: Sequential Micro-Steps               │     │
 │ For each vessel in ordered sequence:          │     │
 │   1. Compute valid action mask                │     │ For all
 │   2. Select action via get_action_and_value() │     │ idle
 │   3. Commit intent via env.micro_step(action) │     │ vessels
 │      (Locks resources; DES clock does NOT advance)  │
 └───────────────────────┬───────────────────────┘     │
                         │ ◄───────────────────────────┘
                         ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ DES Advance (env.step())                                    │
 │ Fast-forward simulation clock to the next decision epoch    │
 └─────────────────────────────────────────────────────────────┘
```

1. **Phase 1: Ordering**: The agent processes the global state and outputs an autoregressive priority permutation of the available vessels.
2. **Phase 2: Micro-steps**: In that order, each vessel selects an action from dynamically masked options. Committing an action updates the **Reservation System**, adjusting masks for subsequent vessels within the same macro-step *without* advancing the DES clock.
3. **DES Step**: Once all active vessels have committed actions, the DES simulation runs until the next decision event occurs.

### Reservation System & Shadow Ledger
Because intents are committed sequentially before the physics engine processes them, the environment maintains a **Reservation System** (`reservation_system.py`) and a shadow ledger:
- **Supply claiming**: When a vessel commits to loading at Site S, the available stock at S is immediately decremented in the ledger, preventing other vessels from over-allocating supply.
- **Capacity claiming**: When a vessel commits to delivering to Site D, destination storage capacity is reserved to avoid deadlock or overflow.
- **Dependency resolution**: When activities require handoffs (e.g. barge feeding an installer), start events are queued conditionally on resource availability.

### Reward Architecture (DPBRS + PFM)
EOS supports both scalar cost minimization and multi-objective optimization:

1. **Dense DPBRS Shaping (Step-Level)**:
   The `MilestoneTracker` computes Dynamic Potential-Based Reward Shaping:
   ```text
   F(s, t, s', t') = γ_t · Φ(s', t') - Φ(s, t)
   ```
   where `γ_t = exp(-β · Δt)` applies continuous-time Semi-Markov Decision Process (SMDP) discounting based on elapsed simulation hours `Δt`. Monopiles gain potential as they progress through the logistics chain:
   ```text
   Source  ──►  Heavy Lift  ──►  Staging  ──►  Feeder  ──►  Installer  ──►  Goal
   ```
   - **Terminal Zeroing**: `Φ(s_T) = 0` is strictly enforced at terminal states, guaranteeing policy invariance.
   - **Bounded Φ_max**: Potential is capped as `Φ_max = phi_max_fraction * terminal_bonus` to prevent completion avoidance.

2. **Terminal Evaluation / PFM (Episode-Level)**:
   - **Scalar Cost Mode**: Terminal reward evaluates mission completion against accumulated operational costs (vessel charter/fuel day-rates, storage yard holding costs).
   - **PFM (Preference Function Modeling)**: Uses online Welford statistics and Polyak target updates to normalize incommensurate objectives (e.g. time vs. storage cost) into dimensionless Z-scores, aggregated via a weighted centroid operator (P*).

### Structured Observations & Transformer Agent
The environment produces structured 2D observations `(N_entities, Max_features)` via `StructuredObservationWrapper`:
- **Row 0**: Global state (normalized simulation time, weather limits, active tasks, accumulated costs).
- **Rows 1 .. S**: Site entities (inventory levels, storage limits, coordinates).
- **Rows S+1 .. V**: Vessel entities (location, speed, payload inventory, current task, time remaining).
- **Rows V+1 .. G**: Goal entities (target installation quantities, remaining quotas, deadlines).

The `TransformerAgent` uses entity-type embeddings followed by Multi-Head Self-Attention across all entities, providing permutation invariance and natural generalization across different fleet sizes and port configurations.

---

## Repository Structure

```text
eos/
├── configs/                       # Hydra configuration tree
│   ├── eos.yaml                   # Top-level default configuration
│   ├── scenario/                  # Scenario definitions (sites, fleet, monopiles)
│   ├── method/                    # Agent architecture (Transformer vs. MLP, joint/discrete)
│   ├── reward/                    # Reward shaping and cost objectives
│   ├── experiment/                # One-click experiment recipes
│   ├── env/                       # Low-level DES environment configs
│   ├── model/                     # Neural network hyperparameter configs
│   ├── learner/                   # PPO hyperparameter configs
│   └── controller/                # Action execution controller configs
│
├── src/eos/                       # Core Python package
│   ├── core/                      # Abstract Base Classes (interfaces)
│   │   ├── model.py               # Model interface (get_ordering, get_action_and_value)
│   │   ├── controller.py          # Controller interface
│   │   ├── runner.py              # Environment interaction runner interface
│   │   ├── learner.py             # RL learning algorithm interface
│   │   ├── buffer.py              # Experience buffer interface
│   │   └── experiment.py          # Experiment orchestrator interface
│   │
│   ├── models/                    # Neural network implementations
│   │   ├── transformer.py         # Entity Transformer with ordering and critic heads
│   │   ├── mlp.py                 # Multi-Layer Perceptron baseline
│   │   ├── action_heads.py        # Masked categorical action distribution heads
│   │   └── autoregressive.py      # Autoregressive sequence heads
│   │
│   ├── controllers/               # Policy execution wrappers
│   │   ├── ppo.py                 # PPO inference controller
│   │   └── random.py              # Uniform random action controller (heuristic baseline)
│   │
│   ├── runners/                   # Rollout collection loops
│   │   ├── micro_step.py          # MicroStepCollector (AEC two-phase loop & eval runner)
│   │   └── ppo.py                 # Vectorized PPO rollout runner
│   │
│   ├── learners/                  # Optimization algorithms
│   │   └── ppo.py                 # PPO with Generalized Advantage Estimation (GAE) & SMDP discounting
│   │
│   ├── buffers/                   # Replay & rollout storage
│   │   └── ppo_rollout.py         # Rollout buffer tracking rewards, values, logprobs, masks
│   │
│   ├── experiments/               # High-level training workflows
│   │   ├── ppo.py                 # Synchronous vectorized PPO training pipeline
│   │   └── random.py              # Random policy baseline evaluator
│   │
│   ├── envs/                      # Simulation environments
│   │   ├── factory.py             # make_env() constructor applying wrappers
│   │   └── simple_monopile_transport/
│   │       ├── gym_env.py         # Gymnasium Environment implementation
│   │       ├── simulator.py       # DES engine wrapper and action generator
│   │       ├── reservation_system.py # Resource reservation & shadow ledger
│   │       ├── activity_builder.py# Discrete-event action construction & stochasticity
│   │       ├── milestone_tracker.py# DPBRS potential calculation
│   │       ├── pfm.py             # Preference Function Modeling normalizer & wrapper
│   │       ├── reward.py          # Multi-objective reward aggregator
│   │       ├── costs/             # Modular cost tracking (time, travel, storage)
│   │       └── wrappers/          # Observation and action transformation wrappers
│   │
│   ├── utils/                     # Metrics, logging, and evaluation utilities
│   │   ├── eval_logger.py         # Evaluation summaries and WandB tables
│   │   ├── metrics.py             # Metric calculations
│   │   ├── visualization.py       # Trajectory plotting and rendering
│   │   └── profiling.py           # Execution timer profiling
│   │
│   └── config.py                  # Strongly-typed dataclass schemas for Hydra
│
├── scripts/                       # Executable entrypoints
│   ├── train.py                   # Main training script (standard runs & Optuna sweeps)
│   ├── eval.py                    # Standalone checkpoint evaluation and video rendering
│   └── show_config.py             # Configuration composition and inspection utility
│
├── pyproject.toml                 # Package definition and uv dependency specification
└── README.md                      # This documentation
```

---

## Configuration System

EOS uses [Hydra](https://hydra.cc/) with type-safe OmegaConf dataclasses defined in `src/eos/config.py`. Configurations are composed along orthogonal axes:

| Axis | Path | Description | Examples |
|------|------|-------------|----------|
| **Scenario** | `configs/scenario/` | Fleet composition, site locations, monopile count | `smt_basic`, `smt_barge`, `smt_is`, `smt_movable_operator` |
| **Method** | `configs/method/` | Model architecture, action space, policy | `ppo_transformer_joint`, `ppo_transformer_discrete`, `ppo_mlp`, `random` |
| **Reward** | `configs/reward/` | Milestone weights, cost components, PFM | `time_only`, `time_and_storage`, `time_and_travel`, `pfm_time_and_storage` |
| **Experiment** | `configs/experiment/` | Pre-packaged compositions of scenario + method + reward | `smt_basic__ppo_transformer__time_and_storage` |

You can compose any configuration directly on the command line:
```bash
python scripts/train.py scenario=smt_barge method=ppo_transformer_joint reward=time_and_storage
```

---

## Getting Started

### Installation

EOS uses [`uv`](https://docs.astral.sh/uv/) for high-performance Python package management.

1. **Prerequisites**:
   - Python 3.12 (`>=3.12, <3.13`)
   - `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
   - Access to the internal package feed (`des_package`, `helper_tools`)

2. **Sync the virtual environment**:
   ```bash
   uv sync
   ```

### Training Models

To launch a training run:

```bash
# Basic training run with PPO + Transformer on SMT-Basic
uv run python scripts/train.py scenario=smt_basic method=ppo_transformer_joint reward=time_only

# Feeder barge scenario with multi-objective PFM reward
uv run python scripts/train.py scenario=smt_barge method=ppo_transformer_joint reward=pfm_time_and_storage

# Override specific hyperparameters via CLI
uv run python scripts/train.py \
    scenario=smt_basic \
    total_timesteps=2000000 \
    num_envs=16 \
    learner.init_lr=2e-4 \
    seed=101
```

Checkpoints and configs are automatically saved under `outputs/checkpoints/` and logged to [Weights & Biases](https://wandb.ai/) if `track=true`.

### Evaluating Models

To evaluate a trained checkpoint deterministically and output physical KPIs (makespan, costs, Gantt schedule):

```bash
uv run python scripts/eval.py \
    --checkpoint outputs/checkpoints/best.pt \
    --config outputs/checkpoints/config.yaml \
    --num-episodes 5
```

To evaluate a random-action baseline:
```bash
uv run python scripts/eval.py --agent random --num-episodes 10
```

### Inspecting Configurations

To verify how Hydra composes configuration groups without launching training:

```bash
uv run python scripts/show_config.py scenario=smt_barge reward=pfm_time_and_storage
```

---

## Extending EOS

### Adding a New Scenario
1. Define the sites, vessels, and monopile quantities under `configs/env/sim/`.
2. Create a scenario definition in `configs/scenario/<new_scenario>.yaml`:
   ```yaml
   # @package _global_
   defaults:
     - /env/sim/objects: my_fleet
     - /env/sim/inventory: my_inventory
     - /env/sim/goals: my_goals
     - /env/sim/activities: default
     - /env/sim/install_windows: none

   exp_name: "smt_my_scenario"
   ```

### Adding a Custom Cost Component
1. Subclass `CostComponent` in `src/eos/envs/simple_monopile_transport/costs/base.py`.
2. Implement `calculate_step_cost(self, sim, active_activities, delta_hours) -> float`.
3. Register the cost component in `src/eos/envs/simple_monopile_transport/costs/aggregator.py`.
4. Add the configuration schema to `CostConfig` in `src/eos/config.py`.

### Implementing a New Neural Architecture
1. Subclass `Model` in `src/eos/core/model.py`.
2. Implement:
   - `forward(obs)`: Contextual representation of the state.
   - `get_ordering(obs, vessel_availability)`: Phase 1 priority permutation.
   - `get_action_and_value(obs, vessel_indices, mask)`: Phase 2 masked action choice and state value.
3. Add the corresponding Hydra config under `configs/model/` and instantiate it in `configs/method/`.
