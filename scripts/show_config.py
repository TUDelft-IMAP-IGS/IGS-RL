#!/usr/bin/env python
"""Print the resolved Hydra config for a given experiment.

Usage:
    # Show full config for T1_1
    python scripts/show_config.py experiment=T1_1__full_system

    # Show full config with CLI overrides (e.g. Pareto sweep weights)
    python scripts/show_config.py experiment=T2_1__pfm \
        env.reward.pfm.objectives.0.weight=0.9 env.reward.pfm.objectives.1.weight=0.1

    # Show only a specific section
    python scripts/show_config.py experiment=T1_1__full_system --section env.reward

    # Show only key experiment settings (compact view)
    python scripts/show_config.py experiment=T2_1__pfm --summary
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eos.config import register_configs  # noqa: E402

CONFIGS_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def resolve_config(overrides: list[str]):
    """Compose the full config using Hydra's Compose API."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="eos", overrides=overrides)
    return cfg


def print_summary(cfg):
    """Print a compact summary of the key experiment settings."""
    print("=" * 60)
    print(f"  Experiment: {cfg.exp_name}")
    print("=" * 60)

    # Scenario / Environment
    print(f"\n{'─' * 60}")
    print("  SCENARIO")
    print(f"{'─' * 60}")
    print(f"  env_id:             {cfg.env.env_id}")
    sim = cfg.env.sim
    print(f"  use_business_rules: {sim.use_business_rules}")

    # Sites & vessels
    if hasattr(sim, "sites"):
        sites = list(sim.sites.keys()) if sim.sites else []
        print(f"  sites:              {sites}")
    if hasattr(sim, "vessels"):
        vessels = list(sim.vessels.keys()) if sim.vessels else []
        print(f"  vessels:            {vessels}")

    # Inventory
    if hasattr(sim, "resource_types"):
        for rt_name, rt in sim.resource_types.items():
            print(f"  {rt_name}:         qty={rt.quantity}")
            if hasattr(rt, "fabrication_schedule") and rt.fabrication_schedule:
                for site, sched in rt.fabrication_schedule.items():
                    sched_list = list(sched) if sched else []
                    print(f"    fab_schedule ({site}): {sched_list}")

    # Activities
    if hasattr(sim, "activities"):
        acts = sim.activities
        if hasattr(acts, "move_matrix") and acts.move_matrix:
            print(f"  move_matrix:")
            for src, dests in acts.move_matrix.items():
                for dst, t in dests.items():
                    print(f"    {src} → {dst}: {t}h")
        if hasattr(acts, "installations") and acts.installations:
            for vessel, sites in acts.installations.items():
                for site, t in sites.items():
                    print(f"  install ({vessel} @ {site}): {t}h")
        print(f"  load_default:       {acts.load_default}h")
        print(f"  unload_default:     {acts.unload_default}h")

    # Reward
    print(f"\n{'─' * 60}")
    print("  REWARD")
    print(f"{'─' * 60}")
    reward = cfg.env.reward
    print(f"  completion_bonus:   {reward.completion_bonus}")

    # PFM
    pfm = getattr(reward, "pfm", None)
    if pfm and getattr(pfm, "enabled", False):
        print(f"  PFM:                enabled")
        print(f"    polyak_tau:       {pfm.polyak_tau}")
        print(f"    burn_in:         {pfm.burn_in_trajectories}")
        print(f"    clip_bound:      {pfm.clip_bound}")
        print(f"    phi_optimism:    {pfm.phi_optimism}")
        for obj in pfm.objectives:
            print(
                f"    objective '{obj.name}': weight={obj.weight}, "
                f"sigma_min={obj.sigma_min}, dir={obj.direction}, "
                f"key={obj.info_key}"
            )
    else:
        print(f"  PFM:                disabled")

    # Naive scalarization
    naive = getattr(reward, "naive_scalarization", None)
    if naive and getattr(naive, "enabled", False):
        print(f"  Naive scalar.:      enabled")
        for obj in naive.objectives:
            print(
                f"    objective '{obj.name}': weight={obj.weight}, "
                f"dir={obj.direction}, key={obj.info_key}"
            )

    # Milestone
    ms = reward.milestone
    weights = {k: v for k, v in ms.stage_weights.items()}
    all_zero = all(v == 0.0 for v in weights.values())
    if all_zero:
        print(f"  DPBRS milestones:   DISABLED (all weights = 0)")
    else:
        print(f"  DPBRS milestones:   enabled")
        for stage, w in weights.items():
            if w > 0:
                print(f"    {stage}: {w}")

    # Costs
    costs = reward.costs
    print(f"  costs.weight:       {costs.weight}")
    if hasattr(costs, "elapsed_time") and getattr(costs.elapsed_time, "enabled", False):
        print(f"    elapsed_time:    rate={costs.elapsed_time.rate}")
    if hasattr(costs, "travel") and getattr(costs.travel, "enabled", False):
        rates = dict(costs.travel.rates) if costs.travel.rates else {}
        print(f"    travel:          {rates}")
    if hasattr(costs, "storage") and getattr(costs.storage, "enabled", False):
        rates = dict(costs.storage.rates) if costs.storage.rates else {}
        print(f"    storage:         {rates}")

    # Method / Model
    print(f"\n{'─' * 60}")
    print("  METHOD")
    print(f"{'─' * 60}")
    print(f"  _target_:           {cfg._target_}")
    if hasattr(cfg, "model"):
        model = cfg.model
        print(f"  model._target_:     {model._target_}")
        if hasattr(model, "d_model"):
            print(
                f"    d_model={model.d_model}, n_heads={model.n_heads}, "
                f"layers={model.num_layers}"
            )
        if hasattr(model, "freeze_ordering"):
            print(f"    freeze_ordering:  {model.freeze_ordering}")
        if hasattr(model, "hidden_size"):
            print(f"    hidden_size:      {model.hidden_size}")

    # Training
    print(f"\n{'─' * 60}")
    print("  TRAINING")
    print(f"{'─' * 60}")
    print(f"  total_timesteps:    {cfg.total_timesteps:,}")
    print(f"  num_envs:           {cfg.num_envs}")
    print(f"  num_steps:          {cfg.num_steps}")
    print(f"  seed:               {cfg.seed}")
    if hasattr(cfg, "learner"):
        lr = cfg.learner
        print(f"  init_lr:            {lr.init_lr}")
        print(f"  ent_coef:           {lr.ent_coef}")
        print(f"  gamma:              {cfg.learner.gamma}")

    print(f"\n{'=' * 60}")


def main():
    args = sys.argv[1:]

    # Separate our flags from Hydra overrides
    section = None
    summary = False
    hydra_overrides = []

    i = 0
    while i < len(args):
        if args[i] == "--section" and i + 1 < len(args):
            section = args[i + 1]
            i += 2
        elif args[i] == "--summary":
            summary = True
            i += 1
        elif args[i] == "--help" or args[i] == "-h":
            print(__doc__)
            sys.exit(0)
        else:
            hydra_overrides.append(args[i])
            i += 1

    if not hydra_overrides:
        print(
            "Usage: python scripts/show_config.py experiment=<name> [overrides...] [--summary] [--section <path>]"
        )
        sys.exit(1)

    cfg = resolve_config(hydra_overrides)

    if summary:
        print_summary(cfg)
    elif section:
        # Navigate to the requested section
        parts = section.split(".")
        node = cfg
        for part in parts:
            if hasattr(node, part):
                node = getattr(node, part)
            else:
                print(f"ERROR: section '{section}' not found (failed at '{part}')")
                sys.exit(1)
        print(OmegaConf.to_yaml(node))
    else:
        # Full config
        print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
