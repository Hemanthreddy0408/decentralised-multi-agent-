"""visualization.py - The 10 required plots (PHASE 13). Reads results only; no simulation logic."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import metrics

COLORS = {"baseline": "#7f7f7f", "selfish": "#d95f02", "cooperative": "#1b9e77"}


def _save(fig, plots_dir, name):
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(Path(plots_dir) / name, dpi=130)
    plt.close(fig)


def _bar(results, value_fn, title, ylabel, fname, plots_dir, fmt="{:.1f}"):
    names = list(results)
    vals = [value_fn(results[n]) for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([n.capitalize() for n in names], vals, color=[COLORS[n] for n in names])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom")
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("Configuration")
    _save(fig, plots_dir, fname)


def make_all_plots(results: dict, plots_dir: str | Path = "plots", selected_agents=(0, 1, 2, 3)) -> list:
    n_days = results[next(iter(results))].n_steps // 24
    P = plots_dir
    _bar(results, metrics.total_cost, "Total neighbourhood electricity cost", "Cost (INR)", "01_total_cost.png", P)
    _bar(results, metrics.peak_grid_load, "Peak aggregate grid load", "Peak load (kW)", "02_peak_load.png", P)
    _bar(results, lambda r: 100 * metrics.grid_dependency(r), "Grid dependency (grid import / demand)",
         "Grid dependency (%)", "03_grid_dependency.png", P)

    # 4 local energy traded over time
    fig, ax = plt.subplots(figsize=(10, 4))
    for sc in ("selfish", "cooperative"):
        if sc in results:
            ax.plot(results[sc].p2p_bought.sum(axis=1), label=sc.capitalize(), color=COLORS[sc], lw=1.2)
    ax.set_title("Local (peer-to-peer) energy traded per hour"); ax.set_xlabel("Simulation hour")
    ax.set_ylabel("Energy traded (kWh/h)"); ax.legend()
    _save(fig, P, "04_local_trading_over_time.png")

    # 5 hourly grid demand
    fig, ax = plt.subplots(figsize=(10, 4))
    for sc, r in results.items():
        ax.plot(metrics.aggregate_grid_import(r), label=sc.capitalize(), color=COLORS[sc], lw=1.1)
    ax.set_title("Hourly aggregate demand on the main grid"); ax.set_xlabel("Simulation hour")
    ax.set_ylabel("Grid import (kW)"); ax.legend()
    _save(fig, P, "05_hourly_grid_demand.png")

    # 6 solar vs demand (identical in all scenarios -> use any)
    r0 = next(iter(results.values()))
    steps = min(r0.n_steps, 72)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r0.solar.sum(axis=1)[:steps], label="Solar generation", color="#e6ab02")
    ax.plot(r0.demand.sum(axis=1)[:steps], label="Demand", color="#386cb0")
    ax.set_title("Neighbourhood solar generation vs demand (first 3 days)"); ax.set_xlabel("Simulation hour")
    ax.set_ylabel("Energy (kWh/h)"); ax.legend()
    _save(fig, P, "06_solar_vs_demand.png")

    # 7 battery SOC of selected households (cooperative run if available)
    rs = results.get("cooperative", r0)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in selected_agents:
        if i < rs.n_agents:
            ax.plot(100 * rs.soc[:48, i] / max(rs.soc[:, i].max(), 1e-9), label=f"Household {i}")
    ax.set_title(f"Battery state of charge, {rs.scenario} (first 2 days, % of own max observed)")
    ax.set_xlabel("Simulation hour"); ax.set_ylabel("SOC (%)"); ax.legend(ncol=2)
    _save(fig, P, "07_battery_soc.png")

    # 8 selfish vs cooperative per-agent cost
    if "selfish" in results and "cooperative" in results:
        idx = np.arange(rs.n_agents); w = 0.4
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(idx - w / 2, metrics.per_agent_cost(results["selfish"]), w, label="Selfish", color=COLORS["selfish"])
        ax.bar(idx + w / 2, metrics.per_agent_cost(results["cooperative"]), w, label="Cooperative", color=COLORS["cooperative"])
        ax.set_title("Per-household electricity cost: selfish vs cooperative")
        ax.set_xlabel("Household id"); ax.set_ylabel("Cost over simulation (INR)"); ax.set_xticks(idx); ax.legend()
        _save(fig, P, "08_selfish_vs_cooperative_cost.png")

    _bar(results, metrics.fairness, "Fairness of benefits (Jain's index, 1 = perfectly fair)",
         "Jain's fairness index", "09_fairness.png", P, fmt="{:.3f}")

    # 10 energy flow in a representative hour (day 2, 19:00) for the cooperative run
    t = min(24 + 19, rs.n_steps - 1)
    ids = np.arange(rs.n_agents)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    parts = [("Solar", rs.solar[t], "#e6ab02"), ("Battery discharge", rs.discharge_out[t], "#7570b3"),
             ("P2P bought", rs.p2p_bought[t], "#1b9e77"), ("Grid import", rs.grid_import[t], "#7f7f7f")]
    bottom = np.zeros(rs.n_agents)
    for label, vals, color in parts:
        ax.bar(ids, vals, bottom=bottom, label=label, color=color); bottom += vals
    uses = [("Battery charge", rs.charge_in[t], "#a6761d"), ("P2P sold", rs.p2p_sold[t], "#66c2a5"),
            ("Grid export", rs.grid_export[t], "#bdbdbd")]
    neg = np.zeros(rs.n_agents)
    for label, vals, color in uses:
        ax.bar(ids, -vals, bottom=-neg, label=label, color=color); neg += vals
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"Energy flows per household, {rs.scenario}, day 2 hour {t % 24}:00 "
                 "(up = energy sources, down = extra outflows)")
    ax.set_xlabel("Household id"); ax.set_ylabel("Energy (kWh)"); ax.set_xticks(ids); ax.legend(ncol=4, fontsize=8)
    _save(fig, P, "10_energy_flow_representative_hour.png")
    return sorted(str(p) for p in Path(P).glob("*.png"))
