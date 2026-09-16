from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch
from gymnasium.envs.registration import register
from loguru import logger
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eos.config import register_configs
from eos.controllers.random import RandomController
from eos.envs.factory import make_env
from eos.runners.micro_step import run_evaluation_episode
from eos.utils.eval_logger import (
    log_eval_to_wandb,
    print_eval_summary,
)

register_configs()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained EOS agent or a random baseline."
    )
    p.add_argument(
        "--agent",
        type=str,
        default="ppo",
        choices=["ppo", "random"],
        help="Which agent to evaluate.",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None, help="Path to the .pt checkpoint file."
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml.",
    )
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--stochastic", action="store_true", default=False)
    p.add_argument("--capture-video", action="store_true", default=False)
    p.add_argument(
        "--render-mode", type=str, default="rgb_array", choices=["rgb_array", "human"]
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--track", action="store_true", default=False)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO", choices=["INFO", "DEBUG"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--run-id", type=str, default=None, help="WandB run ID to resume.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    if not args.checkpoint and not args.config:
        logger.error("Must provide either --checkpoint or --config")
        sys.exit(1)

    # ── 1. Configuration Loading ───────────────────────────────────────────
    checkpoint, ckpt_global_step, path_run_id = {}, None, None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        logger.info(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        cfg = OmegaConf.create(checkpoint["config"])
        ckpt_global_step = checkpoint.get("global_step")

        # Extract WandB run_id from path if present
        for part in ckpt_path.parts:
            if part.startswith("run-") and "-" in part:
                path_run_id = part.split("-")[-1]
                break
    else:
        logger.info(f"Loading raw config: {args.config}")
        cfg = OmegaConf.load(args.config)

    logger.info(f"Checkpoint config: {OmegaConf.to_yaml(cfg)}")

    # Print checkpoint metadata
    if "score" in checkpoint:
        logger.info(f"Checkpoint best score: {checkpoint['score']:.3f}")
    if "update" in checkpoint:
        logger.info(f"Checkpoint saved at update: {checkpoint['update']}")
    if "global_step" in checkpoint:
        logger.info(f"Checkpoint global step: {checkpoint['global_step']}")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    )
    logger.info(f"Using device: {device}")

    register(
        id="SimpleMonopileTransport-v0",
        entry_point="eos.envs.simple_monopile_transport.gym_env:SimpleMonopileTransportEnv",
    )

    # ── 2. Controller Instantiation ─────────────────────────────────────────
    seed = args.seed if args.seed is not None else cfg.seed
    tmp_env = gym.vector.SyncVectorEnv([make_env(cfg, 0, False, "eval_tmp", None)])

    import hydra as _hydra

    if args.agent == "ppo":
        # Load the Model
        ModelClass = _hydra.utils.get_class(cfg.model._target_)
        model = ModelClass(tmp_env, cfg.model).to(device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {cfg.model._target_} ({param_count:,} parameters)")

        logger.info(f"Action space: {tmp_env.single_action_space}")

        # Load the controller
        ControllerClass = _hydra.utils.get_class(cfg.controller._target_)
        controller = ControllerClass(model, cfg.controller, device)

    elif args.agent == "random":
        from eos.config import RandomControllerConfig

        # Instantiate the Random Controller (no model needed)
        random_cfg = RandomControllerConfig()
        controller = RandomController(
            model=None, cfg=random_cfg, action_space=tmp_env.single_action_space
        )
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")

    tmp_env.close()

    # ── 3. Build evaluation environment ─────────────────────────────────────
    run_name = cfg.run_name or "eval"
    eval_env = gym.vector.SyncVectorEnv(
        [
            make_env(
                cfg,
                0,
                args.capture_video,
                f"{run_name}_eval",
                args.render_mode,
            )
        ]
    )

    # --- PFM multi-objective reward normalization ---
    pfm_cfg = getattr(cfg.env.reward, "pfm", None)
    if pfm_cfg is not None and pfm_cfg.enabled:
        from eos.envs.simple_monopile_transport.pfm import PFMVectorWrapper

        eval_env = PFMVectorWrapper(eval_env, pfm_cfg)

    # ── 4. Run evaluation via shared micro-stepping loop ────────────────────
    deterministic = not args.stochastic
    use_action_masks = bool(cfg.env.use_action_masks)

    eval_results = run_evaluation_episode(
        eval_env,
        controller,
        seed=seed,
        num_episodes=args.num_episodes,
        deterministic=deterministic,
        use_action_masks=use_action_masks,
        capture_inventory=True,
        trace_actions=True,
    )

    eval_env.close()

    # ── 5. Output and tracking ──────────────────────────────────────────────
    print_eval_summary(eval_results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            "summary": eval_results["summary"],
            "episodes": eval_results["episodes"],
            "inventory_history": eval_results.get("inventory_history"),
        }
        with open(output_path, "w") as f:
            json.dump(serialisable, f, indent=2, default=str)
        logger.info(f"Results written to {output_path}")

    if args.track:
        import wandb

        target_run_id = (
            args.run_id
            or path_run_id
            or checkpoint.get("run_id")
            or checkpoint.get("wandb_run_id")
        )

        wandb_config = dict(OmegaConf.to_container(cfg, resolve=True))
        wandb_config.update(
            {
                "checkpoint": args.checkpoint,
                "agent": args.agent,
                "num_episodes": args.num_episodes,
                "deterministic": deterministic,
                "seed": seed,
            }
        )

        if target_run_id:
            logger.info(f"Resuming WandB run: {target_run_id}")
            if "DATABRICKS_RUNTIME_VERSION" in os.environ:
                from pyspark.dbutils import DBUtils  # type: ignore
                from pyspark.sql import SparkSession  # type: ignore

                spark = SparkSession.builder.getOrCreate()
                dbutils = DBUtils(spark)

                # Securely fetch the key from the Databricks vault
                wandb_key = dbutils.secrets.get(scope="wandb", key="api_key")

                # Authenticate the cluster with WandB
                wandb.login(key=wandb_key)

            wandb.init(
                project=cfg.wandb_project_name,
                entity=cfg.wandb_entity,
                id=target_run_id,
                resume="must",
            )
            if wandb.run:
                wandb.run.tags = tuple(set(wandb.run.tags or []) | {"eval"})
                wandb.config.update(wandb_config, allow_val_change=True)
        else:
            if "DATABRICKS_RUNTIME_VERSION" in os.environ:
                from pyspark.dbutils import DBUtils  # type: ignore
                from pyspark.sql import SparkSession  # type: ignore

                spark = SparkSession.builder.getOrCreate()
                dbutils = DBUtils(spark)

                # Securely fetch the key from the Databricks vault
                wandb_key = dbutils.secrets.get(scope="wandb", key="api_key")

                # Authenticate the cluster with WandB
                wandb.login(key=wandb_key)

            wandb.init(
                project=cfg.wandb_project_name,
                entity=cfg.wandb_entity,
                config=wandb_config,
                name=f"eval_{args.agent}_{seed}",
                job_type="eval",
                tags=["eval"],
            )

        log_eval_to_wandb(eval_results, wandb.run.dir, global_step=ckpt_global_step)
        wandb.finish()

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
