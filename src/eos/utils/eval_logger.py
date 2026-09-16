from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# Ensure this import works in your environment
from eos.utils.metrics import create_action_trace_table


def capture_inventory_snapshot(sim, history: list[dict]) -> None:
    """Takes a snapshot of the simulator's inventory state."""
    try:
        record = {
            "time": sim.elapsed_time,
            "window_closed": sim.is_in_no_install_window(),
        }
        for site in sim.get_site_names():
            inv = sim.get_site_inventory(site)
            for res, data in inv.items():
                record[f"{site} - {res}"] = data["load"]
        history.append(record)
    except Exception:
        pass


def create_custom_gantt_chart(df: pd.DataFrame, simulation_start=None):
    """Builds a custom Plotly Gantt chart from an EventSymphony overview dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Overview dataframe from ``get_overview_dataframe``.
    simulation_start : datetime.datetime, optional
        The simulation start time. When provided, the x-axis shows elapsed
        hours since sim start instead of absolute dates.
    """
    import plotly.graph_objects as go

    # Ensure timestamps are actual datetime objects (strip tz for consistency)
    df["START"] = pd.to_datetime(df["START"], utc=True).dt.tz_localize(None)
    df["STOP"] = pd.to_datetime(df["STOP"], utc=True).dt.tz_localize(None)

    # Convert to elapsed hours if simulation_start is provided
    if simulation_start is not None:
        sim_start_ts = pd.Timestamp(simulation_start).tz_localize(None)
        df["start_hours"] = (df["START"] - sim_start_ts).dt.total_seconds() / 3600.0
        df["stop_hours"] = (df["STOP"] - sim_start_ts).dt.total_seconds() / 3600.0
    else:
        # Fallback: use earliest START as reference
        sim_start_ts = df["START"].min()
        df["start_hours"] = (df["START"] - sim_start_ts).dt.total_seconds() / 3600.0
        df["stop_hours"] = (df["STOP"] - sim_start_ts).dt.total_seconds() / 3600.0

    # Create string formats for the hover tooltip
    df["start_str"] = df["start_hours"].apply(lambda h: f"{h:.1f}h")
    df["stop_str"] = df["stop_hours"].apply(lambda h: f"{h:.1f}h")

    # Helper to safely format inventory dicts (e.g. {'large_mp': 2} -> "large_mp: 2")
    def _fmt_inv(inv):
        if isinstance(inv, dict):
            return ", ".join(f"{k}: {v}" for k, v in inv.items())
        return str(inv)

    # Build the rich hover context column
    df["HoverContext"] = df.apply(
        lambda row: (
            f"<b>Activity:</b> {row.get('activities', 'N/A')}<br>"
            f"<b>Cargo Start:</b> {_fmt_inv(row.get('container_level_START', {}))}<br>"
            f"<b>Cargo End:</b> {_fmt_inv(row.get('container_level_STOP', {}))}"
        ),
        axis=1,
    )

    # Standardize category colors
    color_map = {
        "transit": "#636EFA",  # Blue
        "loading": "#00CC96",  # Green
        "unloading": "#EF553B",  # Red
        "installing": "#FFA15A",  # Orange
        "idle": "#B6E880",  # Light green
    }

    # Build Gantt chart using horizontal bars with elapsed hours on x-axis
    fig = go.Figure()

    # Get unique assets and categories
    assets = df["asset"].unique().tolist()
    categories_seen = set()

    for _, row in df.iterrows():
        cat = row.get("category", "unknown")
        color = color_map.get(cat, "#999999")
        show_legend = cat not in categories_seen
        categories_seen.add(cat)

        duration = row["stop_hours"] - row["start_hours"]
        fig.add_trace(
            go.Bar(
                x=[duration],
                y=[row["asset"]],
                base=[row["start_hours"]],
                orientation="h",
                marker_color=color,
                name=cat,
                legendgroup=cat,
                showlegend=show_legend,
                customdata=[
                    [
                        row["HoverContext"],
                        row["start_str"],
                        row["stop_str"],
                    ]
                ],
                hovertemplate=(
                    "<b>Asset:</b> %{y}<br>"
                    "<b>Start:</b> %{customdata[1]}<br>"
                    "<b>Stop:</b> %{customdata[2]}<br><br>"
                    "%{customdata[0]}"
                    "<extra></extra>"
                ),
            )
        )

    # Make the layout clean and modern
    fig.update_layout(
        title="Vessel Logistics Timeline",
        xaxis_title="Project time (Hours)",
        yaxis_title="Vessel",
        template="plotly_white",
        legend_title="Action Category",
        barmode="overlay",
        yaxis=dict(
            autorange="reversed",
            categoryorder="array",
            categoryarray=assets,
        ),
    )

    return fig


def _fmt_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:>10.3f}"
    return f"{value!s:>10s}"


def _extract_best_episode(eval_results: dict) -> dict | None:
    episodes = eval_results.get("episodes") or []
    if not episodes:
        return None
    try:
        return max(episodes, key=lambda ep: float(ep.get("return", -float("inf"))))
    except Exception:
        return episodes[0]


def _flatten_numeric_dict(
    data: dict[str, Any] | None,
    prefix: str = "",
) -> dict[str, float]:
    flat: dict[str, float] = {}
    if not data:
        return flat

    for key, value in data.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_numeric_dict(value, full_key))
        elif isinstance(value, (int, float, np.integer, np.floating, bool)):
            flat[full_key] = float(value)
    return flat


def _summarise_breakdown_numeric_fields(
    episodes: list[dict],
    field_name: str,
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for ep in episodes:
        breakdown = ep.get(field_name)
        if not isinstance(breakdown, dict):
            continue
        for key, value in _flatten_numeric_dict(breakdown).items():
            totals[key] += float(value)
    return dict(sorted(totals.items()))


# Human-friendly labels for known cost component names.
_COST_COMPONENT_LABELS: dict[str, str] = {
    "elapsed_time_cost": "Elapsed time cost",
    "travel_cost": "Travel cost",
    "storage_cost": "Storage cost",
}


def _create_reward_breakdown_figure(episode: dict):
    import plotly.graph_objects as go

    reward_breakdown = episode.get("reward_breakdown") or {}

    milestone_shaping = float(reward_breakdown.get("milestone_term_total", 0.0))

    labels = [
        "Milestone shaping",
        "Completion bonus",
    ]
    values = [
        milestone_shaping,
        float(reward_breakdown.get("completion_bonus_total", 0.0)),
    ]
    measures = [
        "relative",
        "relative",
    ]

    # Per-component cost bars (e.g. elapsed time, travel, storage).
    # Falls back to a single "Cost applied" bar when the per-component
    # breakdown is unavailable.
    cost_components = reward_breakdown.get("reward_cost_components_total") or {}
    if cost_components:
        for comp_name, comp_val in sorted(cost_components.items()):
            label = _COST_COMPONENT_LABELS.get(
                comp_name, comp_name.replace("_", " ").title()
            )
            labels.append(label)
            values.append(-float(comp_val))
            measures.append("relative")
    else:
        labels.append("Cost applied")
        values.append(-float(reward_breakdown.get("reward_cost_term_total", 0.0)))
        measures.append("relative")

    # Final total bar
    labels.append("Total reward")
    values.append(
        float(reward_breakdown.get("total_reward", episode.get("return", 0.0)))
    )
    measures.append("total")

    fig = go.Figure(
        go.Waterfall(
            name="Reward breakdown",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )
    fig.update_layout(
        title="Evaluation Reward Breakdown",
        yaxis_title="Reward contribution",
        template="plotly_white",
        showlegend=False,
    )
    return fig


def _create_cost_breakdown_figure(episode: dict):
    import plotly.graph_objects as go

    cost_breakdown = episode.get("cost_breakdown") or {}
    labels = []
    values = []

    for key in ("elapsed_time_cost", "storage_cost", "travel_cost", "other_cost"):
        value = float(cost_breakdown.get(key, 0.0))
        if value > 0.0:
            labels.append(key.replace("_", " "))
            values.append(value)

    if not labels:
        labels = ["no_cost_recorded"]
        values = [1.0]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.35, sort=False)])
    fig.update_layout(
        title="Evaluation Cost Breakdown",
        template="plotly_white",
    )
    return fig


def _create_operational_metrics_figure(episode: dict):
    import plotly.graph_objects as go

    operational_metrics = episode.get("operational_metrics") or {}

    # Prefer cost (hours × rate) when available; fall back to raw hours.
    travel_cost_by_vessel = operational_metrics.get("travel_cost_by_vessel") or {}
    travel_hours_by_vessel = operational_metrics.get("travel_hours_by_vessel") or {}
    storage_cost_by_site = operational_metrics.get("storage_cost_by_site") or {}
    storage_hours_by_site = operational_metrics.get("storage_unit_hours_by_site") or {}

    fig = go.Figure()

    if travel_cost_by_vessel:
        fig.add_trace(
            go.Bar(
                x=list(travel_cost_by_vessel.keys()),
                y=[float(v) for v in travel_cost_by_vessel.values()],
                name="Travel cost by vessel",
            )
        )
    elif travel_hours_by_vessel:
        fig.add_trace(
            go.Bar(
                x=list(travel_hours_by_vessel.keys()),
                y=[float(v) for v in travel_hours_by_vessel.values()],
                name="Travel hours by vessel (no rates)",
            )
        )

    if storage_cost_by_site:
        fig.add_trace(
            go.Bar(
                x=list(storage_cost_by_site.keys()),
                y=[float(v) for v in storage_cost_by_site.values()],
                name="Storage cost by site",
            )
        )
    elif storage_hours_by_site:
        fig.add_trace(
            go.Bar(
                x=list(storage_hours_by_site.keys()),
                y=[float(v) for v in storage_hours_by_site.values()],
                name="Storage unit-hours by site (no rates)",
            )
        )

    if not fig.data:
        fig.add_trace(go.Bar(x=["no_metrics_recorded"], y=[0.0], name="No metrics"))

    fig.update_layout(
        title="Evaluation Operational Metrics",
        xaxis_title="Entity",
        yaxis_title="Cost (rate × hours)",
        template="plotly_white",
        barmode="group",
    )
    return fig


def print_eval_summary(eval_results: dict) -> None:
    """Print a formatted summary table to the console."""
    summary = eval_results.get("summary", {})
    episodes = eval_results.get("episodes") or []
    best_episode = _extract_best_episode(eval_results)

    logger.info("=" * 60)
    logger.info("  ACTION TRACE (Best / First Episode)")
    logger.info("=" * 60)
    action_trace = eval_results.get("action_trace") or []
    trace_table = create_action_trace_table(action_trace)
    if trace_table:
        trace_df = trace_table.get_dataframe()
        logger.info(f"\n{trace_df.to_string(index=False)}")
    logger.info("=" * 60)

    logger.info("=" * 60)
    logger.info("  EVALUATION SUMMARY")
    logger.info("=" * 60)
    for k, v in summary.items():
        logger.info(f"  {k:<30s} {_fmt_metric_value(v)}")
    logger.info("=" * 60)

    if best_episode:
        logger.info("=" * 60)
        logger.info("  BEST EPISODE REWARD BREAKDOWN")
        logger.info("=" * 60)
        for k, v in (best_episode.get("reward_breakdown") or {}).items():
            logger.info(f"  {k:<30s} {_fmt_metric_value(v)}")
        logger.info("=" * 60)

        logger.info("=" * 60)
        logger.info("  BEST EPISODE COST BREAKDOWN")
        logger.info("=" * 60)
        for k, v in (best_episode.get("cost_breakdown") or {}).items():
            if isinstance(v, dict):
                continue
            logger.info(f"  {k:<30s} {_fmt_metric_value(v)}")
        logger.info("=" * 60)

        logger.info("=" * 60)
        logger.info("  BEST EPISODE OPERATIONAL METRICS")
        logger.info("=" * 60)
        for k, v in (best_episode.get("operational_metrics") or {}).items():
            if isinstance(v, dict):
                continue
            logger.info(f"  {k:<30s} {_fmt_metric_value(v)}")
        logger.info("=" * 60)

    if episodes:
        reward_totals = _summarise_breakdown_numeric_fields(
            episodes, "reward_breakdown"
        )
        cost_totals = _summarise_breakdown_numeric_fields(episodes, "cost_breakdown")
        op_totals = _summarise_breakdown_numeric_fields(episodes, "operational_metrics")

        if reward_totals:
            logger.info("=" * 60)
            logger.info("  AGGREGATED REWARD BREAKDOWN (ALL EPISODES)")
            logger.info("=" * 60)
            for k, v in reward_totals.items():
                logger.info(f"  {k:<30s} {_fmt_metric_value(v)}")
            logger.info("=" * 60)

        if cost_totals:
            logger.info("=" * 60)
            logger.info("  AGGREGATED COST BREAKDOWN (ALL EPISODES)")
            logger.info("=" * 60)
            for k, v in cost_totals.items():
                logger.info(f"  {k:<30s} {_fmt_metric_value(v)}")
            logger.info("=" * 60)

        if op_totals:
            logger.info("=" * 60)
            logger.info("  AGGREGATED OPERATIONAL METRICS (ALL EPISODES)")
            logger.info("=" * 60)
            for k, v in op_totals.items():
                logger.info(f"  {k:<30s} {_fmt_metric_value(v)}")
            logger.info("=" * 60)


def log_eval_to_wandb(
    eval_results: dict, run_dir: str, global_step: int | None = None
) -> None:
    """Log evaluation results, action traces, and domain visualisations to WandB.

    All metrics and plots are collected into a single ``wandb.log()`` call
    so that they share the same WandB step and appear together in the
    dashboard without duplicates.
    """
    import plotly.graph_objects as go

    import wandb

    summary = eval_results.get("summary", {})
    episodes = eval_results.get("episodes") or []
    best_episode = _extract_best_episode(eval_results)

    # Only log core & domain metrics to wandb; reward/cost/operational
    # detail breakdowns are printed to the terminal but would clutter
    # the wandb dashboard.
    _WANDB_SUMMARY_KEYS = {
        "num_episodes",
        "deterministic",
        "return_mean",
        "return_std",
        "return_min",
        "return_max",
        "length_mean",
        "length_std",
        "goals_completed_mean",
        "goals_failed_mean",
        "goals_remaining_mean",
        "sim_elapsed_hours_mean",
    }

    # Collect everything into a single dict for one wandb.log() call.
    wandb_payload: dict[str, Any] = {
        f"eval/{k}": v for k, v in summary.items() if k in _WANDB_SUMMARY_KEYS
    }

    if global_step is not None:
        wandb_payload["global_step"] = global_step

    # Action trace table & CSV
    action_trace = eval_results.get("action_trace") or []
    trace_table = create_action_trace_table(action_trace)
    if trace_table is not None:
        wandb_payload["eval/action_trace"] = trace_table
        try:
            trace_df = trace_table.get_dataframe()
            trace_path = Path(run_dir) / "eval" / "action_trace.csv"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_df.to_csv(trace_path, index=False)
            wandb.save(str(trace_path), policy="end")
        except Exception:
            pass

    if best_episode:
        try:
            reward_fig = _create_reward_breakdown_figure(best_episode)
            wandb_payload["eval/reward_breakdown_plot"] = wandb.Html(
                reward_fig.to_html(full_html=False, include_plotlyjs="cdn")
            )
        except Exception as e:
            logger.warning(f"Could not generate reward breakdown plot: {e}")

        try:
            cost_fig = _create_cost_breakdown_figure(best_episode)
            wandb_payload["eval/cost_breakdown_plot"] = wandb.Html(
                cost_fig.to_html(full_html=False, include_plotlyjs="cdn")
            )
        except Exception as e:
            logger.warning(f"Could not generate cost breakdown plot: {e}")

        try:
            operational_fig = _create_operational_metrics_figure(best_episode)
            wandb_payload["eval/operational_metrics_plot"] = wandb.Html(
                operational_fig.to_html(full_html=False, include_plotlyjs="cdn")
            )
        except Exception as e:
            logger.warning(f"Could not generate operational metrics plot: {e}")

        try:
            breakdown_path = Path(run_dir) / "eval" / "best_episode_breakdown.json"
            breakdown_path.parent.mkdir(parents=True, exist_ok=True)
            with open(breakdown_path, "w") as f:
                json.dump(best_episode, f, indent=2, default=str)
            wandb.save(str(breakdown_path), policy="end")
        except Exception:
            pass

    # Visualizations
    unwrapped_env = eval_results.get("env")
    if unwrapped_env is not None:
        # 1. Inventory Plotly Chart
        inventory_history = eval_results.get("inventory_history")
        if inventory_history:
            try:
                fig = go.Figure()

                # Deduplicate snapshots at the same time — when multiple
                # macro-steps fire at the same sim time (e.g. fabrication
                # then immediate load), keep only the last snapshot per
                # time point so the plot shows the settled state.
                deduped_history: list[dict] = []
                for rec in inventory_history:
                    if deduped_history and deduped_history[-1]["time"] == rec["time"]:
                        deduped_history[-1] = rec
                    else:
                        deduped_history.append(rec)

                times = [r["time"] / 3600.0 for r in deduped_history]

                keys = [
                    k
                    for k in deduped_history[0].keys()
                    if k not in ("time", "window_closed")
                ]
                for key in keys:
                    values = [r[key] for r in deduped_history]
                    if any(v > 0 for v in values):
                        fig.add_trace(
                            go.Scatter(
                                x=times,
                                y=values,
                                mode="lines",
                                name=key,
                                line_shape="hv",
                            )
                        )

                in_window = False
                start_t = 0
                for i, r in enumerate(deduped_history):
                    if r["window_closed"] and not in_window:
                        in_window = True
                        start_t = times[i]
                    elif not r["window_closed"] and in_window:
                        in_window = False
                        fig.add_vrect(
                            x0=start_t,
                            x1=times[i],
                            fillcolor="red",
                            opacity=0.15,
                            layer="below",
                            line_width=0,
                        )
                if in_window:
                    fig.add_vrect(
                        x0=start_t,
                        x1=times[-1],
                        fillcolor="red",
                        opacity=0.15,
                        layer="below",
                        line_width=0,
                    )

                fig.update_layout(
                    title="Site Inventory Levels Over Time",
                    xaxis_title="Project time (Hours)",
                    yaxis_title="Asset Count",
                    template="plotly_white",
                )

                wandb_payload["eval/inventory_plot"] = wandb.Html(
                    fig.to_html(full_html=False, include_plotlyjs="cdn")
                )
            except Exception as e:
                logger.warning(f"Could not generate inventory plot: {e}")

        # 2. Gantt Chart & State
        try:
            from boka_eventsymphony.plot import get_overview_dataframe

            from eos.utils.visualization import get_state_overview

            sim = unwrapped_env._sim
            state_overview = get_state_overview(sim.es_env)
            wandb_payload["eval/simulation_state"] = wandb.Html(state_overview)

            df = get_overview_dataframe(
                sim._get_all_vessels(), sim.es_env.registry.get("activities", [])
            )

            gantt_plot = create_custom_gantt_chart(
                df, simulation_start=sim.simulation_start
            )
            wandb_payload["eval/gantt_plot"] = wandb.Html(
                gantt_plot.to_html(full_html=False, include_plotlyjs="cdn")
            )
        except Exception as e:
            logger.warning(f"Could not generate domain visualisations: {e}")

    # Single wandb.log call — all eval metrics share the same step.
    if wandb_payload:
        wandb.log(wandb_payload)
